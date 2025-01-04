import os
import logging
import glob
import torch
import shutil
import datetime
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed, gather
from accelerate.logging import get_logger

from pointcept.datasets.utils import point_collate_fn, collate_fn
from pointcept.models import build_model
from pointcept.datasets import build_dataset
from pointcept.utils.optimizer import build_optimizer
from pointcept.utils.scheduler import build_scheduler
from pointcept.utils.config import Config, DictAction

from torch import Tensor
from torch.utils.data import DataLoader, default_collate
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from pathlib import Path
from safetensors.torch import load_file
from itertools import chain
from typing import Sequence, Mapping
from collections import defaultdict
import pickle as pkl

from models import *
from datasets import *
from utils.metrics import build_metric

def mf_collate_fn(batch, key=''):
    if not isinstance(batch, Sequence):
        raise TypeError(f"{batch.dtype} is not supported.")

    if isinstance(batch[0], torch.Tensor):
        if 'keypoint' in key or 'pose' in key:
            return torch.stack(list(batch))
        return torch.cat(list(batch))
    elif isinstance(batch[0], str):
        # str is also a kind of Sequence, judgement should before Sequence
        return list(batch)
    elif isinstance(batch[0], Sequence):
        for i, data in enumerate(batch):
            for f in data:
                f['frame_group'] = torch.tensor([i])
        return mf_collate_fn(list(chain(*batch)))
    elif isinstance(batch[0], Mapping):
        batch = {key: mf_collate_fn([d[key] for d in batch], key) for key in batch[0]}
        for key in batch.keys():
            if "offset" in key:
                batch[key] = torch.cumsum(batch[key], dim=0)
        return batch
    else:
        return default_collate(batch)

def prepare_env(cfg: Config, mode='train'):
    set_seed(cfg.seed)
    accelerator = Accelerator(
        log_with='wandb' if 'NO_LOG' not in os.environ else None,
        project_config=ProjectConfiguration(
            project_dir=cfg.work_dir,
            logging_dir=cfg.work_dir,
            automatic_checkpoint_naming=True,
            total_limit=5,
        ),
    )

    if accelerator.is_main_process and mode == 'train':
        if os.path.exists(accelerator.project_dir):
            for ckptdir in glob.glob(f'{accelerator.project_dir}/checkpoints/checkpoint_*'):
                shutil.rmtree(ckptdir)
    
    if accelerator.is_main_process:
        if not os.path.exists(cfg.work_dir):
            os.makedirs(cfg.work_dir, exist_ok=True)

    accelerator.wait_for_everyone()

    device = accelerator.device
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'{cfg.work_dir}/'
                                f'{timestamp}_{mode}_{accelerator.process_index}.log', 
                                mode='w'),
            logging.StreamHandler()
        ]
    )
    
    logger = get_logger(__name__)
    logger.info(cfg)
    return accelerator, device, logger

def main(cfg: Config):
    accelerator, device, logger = prepare_env(cfg)

    train_dataloader = DataLoader(
        build_dataset(cfg.data.train_dataset),
        batch_size=cfg.data.train_batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=True,
        collate_fn=mf_collate_fn,
        pin_memory=True
    )
    
    val_dataset = build_dataset(cfg.data.val_dataset)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.data.val_batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=False,
        collate_fn=mf_collate_fn,
        pin_memory=True
    )
    
    total_steps = (len(train_dataloader) * cfg.num_epochs)
    
    model = build_model(cfg.model).to(device)
    optimizer = build_optimizer(cfg.optimizer, model)
    
    cfg.scheduler.total_steps = total_steps
    step_on_loss = cfg.scheduler.pop('step_on_loss', None)
    step_by_epoch = cfg.scheduler.pop('step_by_epoch', False)
    lr_scheduler = build_scheduler(cfg.scheduler, optimizer)

    (
        train_dataloader, val_dataloader,
        model, optimizer, lr_scheduler
    ) = accelerator.prepare(
        train_dataloader, val_dataloader,
        model, optimizer, lr_scheduler
    )

    total_steps = (len(train_dataloader) * cfg.num_epochs)

    if accelerator.is_main_process and 'NO_LOG' not in os.environ:
        accelerator.init_trackers('pc_pose', 
            init_kwargs=dict(
                wandb=dict(
                    name=cfg.exp_name
                )
        ))

    progress_bar = tqdm(
        range(total_steps),
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    global_step = 0
    for n_epoch in range(cfg.num_epochs):
        model.train()
        for i, batch in enumerate(train_dataloader):
            for k in batch:
                if isinstance(batch[k], Tensor):
                    batch[k] = batch[k].to(device)
            loss = model(batch, return_loss=True)
            loss_all = sum(loss.values()) if isinstance(loss, dict) else loss
            accelerator.backward(loss_all)
            optimizer.step()
            if not step_by_epoch:
                if step_on_loss is not None:
                    lr_scheduler.step(loss_all)
                else:
                    lr_scheduler.step()
            optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                loss_dict = {}
                if isinstance(loss, Tensor):
                    loss_dict = {"train/loss": loss}
                else:
                    loss_dict = {f"train/{k}_loss": v.detach().item() for k, v in loss.items()}
                
                log = {
                    **loss_dict,
                    "train/lr": lr_scheduler.get_last_lr()[0],
                    "train/step": global_step,
                }
                progress_bar.set_description(' '.join(f"{k}: {v:.4e}" for k, v in log.items()))
                if i % cfg.log_interval == 0:
                    accelerator.log(log, step=global_step)
        
        if step_by_epoch:
            if step_on_loss is not None:
                lr_scheduler.step(loss_all)
            else:
                lr_scheduler.step()
        
        if (n_epoch + 1) % cfg.val_interval == 0:
            model.eval()
            metrics = [
                build_metric(m) for m in cfg.metrics
            ]
            with torch.no_grad():
                for i, batch in enumerate(val_dataloader):
                    for k in batch:
                        if isinstance(batch[k], Tensor):
                            batch[k] = batch[k].to(device)
                    pred = model(batch, return_loss=False)
                    
                    if not isinstance(pred, dict):
                        pred = dict(pred_keypoints_3d=pred)
                        
                    collected = dict()
                    for k in pred:
                        gk = k.replace('pred_', '')
                        pred_k, gt_k = accelerator.gather_for_metrics((pred[k], batch[gk]))
                        collected[k] = pred_k.cpu()
                        collected[gk] = gt_k.cpu()
                    
                    for metric in metrics:
                        metric.update(collected)

            log = {f"val/{k}":v for k, v in chain(*[metric.results().items() for metric in metrics])}
            log['val/epoch'] = n_epoch + 1
            accelerator.log(log, step=global_step)
            logger.info(log)
            
        if (n_epoch + 1) % cfg.save_interval == 0:
            accelerator.save_state()

def test(cfg: Config, ckpt: Path):        
    accelerator, device, logger = prepare_env(cfg, mode='test')

    val_dataset = build_dataset(cfg.data.val_dataset)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.data.val_batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=False,
        collate_fn=mf_collate_fn,
        pin_memory=True
    )
    model = build_model(cfg.model).to(device)
    val_dataloader, model = accelerator.prepare(val_dataloader, model)
    model.eval()

    logger.info(f"Loading checkpoint: {ckpt}")
    if ckpt.exists() and ckpt.suffix == '.safetensors':
        ckpt = load_file(ckpt)
        missing, unexpected = model.module.load_state_dict(ckpt, strict=False)
        logger.info(f"Missing keys: {missing} \n Unexpected keys: {unexpected}")
    elif (ckpt / 'model.safetensors').exists():
        accelerator.load_state(ckpt)
    elif (ckpt / 'checkpoints').exists():
        ckpt = sorted((ckpt / 'checkpoints').glob('checkpoint_*'))[-1]
        accelerator.load_state(ckpt)
    else:
        logger.error(f"Using NO checkpoints!")
    
    
    preds = defaultdict(list)
    metrics = [
        build_metric(m) for m in cfg.metrics
    ]
    with torch.no_grad():
        for i, batch in enumerate(val_dataloader):
            for k in batch:
                if isinstance(batch[k], Tensor):
                    batch[k] = batch[k].to(device)
            pred = model(batch, return_loss=False)
            
            if not isinstance(pred, dict):
                pred = dict(pred_keypoints_3d=pred)
                
            collected = dict()
            for k in pred:
                gk = k.replace('pred_', '')
                pred_k, gt_k = accelerator.gather_for_metrics((pred[k], batch[gk]))
                collected[k] = pred_k.cpu()
                collected[gk] = gt_k.cpu()
                preds[k].append(collected[k].cpu().numpy())
            
            for metric in metrics:
                metric.update(collected)

    log = {f"val/{k}":v for k, v in chain(*[metric.results().items() for metric in metrics])}
    logger.info(log)

    if accelerator.is_local_main_process:
        for k in preds:
            preds[k] = np.concatenate(preds[k], axis=0).astype(np.float32)
        pkl.dump(preds, open(f'{cfg.work_dir}/preds.pkl', 'wb'))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('config', type=Path)
    parser.add_argument('--test', action='store_true')
    parser.add_argument(
        "--options", nargs="+", action=DictAction, help="custom options"
    )
    parser.add_argument('--ckpt', type=Path, default=None)
    args = parser.parse_args()
    
    cfg_path: Path = args.config
    cfg = Config().fromfile(cfg_path)
    
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    
    if args.test:
        ckpt: Path = args.ckpt
        test(cfg, ckpt)
    else:
        main(cfg)
