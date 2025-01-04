import torch
import torch.nn as nn
import torch.nn.functional as F
import safetensors

from typing import Dict
from torch import Tensor

from pointcept.utils.config import Config
from pointcept.utils.registry import Registry
from pointcept.models.builder import MODELS, build_model
from accelerate.logging import get_logger


@MODELS.register_module()
class PoseNet(nn.Module):
    def __init__(self, 
                 backbone=None,
                 neck=None,
                 head=None,
                 supv_keys=['query'],
                 supv_weights=[1.0]) -> None:
        super().__init__()
        self.backbone = build_model(backbone)
        self.neck = build_model(neck) if neck else nn.Identity()
        self.head = build_model(head)
        self.supv_keys = supv_keys
        self.supv_weights = supv_weights

    def forward(self, samples: Dict[str, torch.Tensor], return_loss=False):
        feat = self.backbone(samples)
        feat = self.neck(feat)
        
        if isinstance(feat, torch.Tensor):
            feat = dict(query=feat)
        
        if return_loss:
            loss = 0
            for k, w in zip(self.supv_keys, self.supv_weights):
                pred = self.head(feat[k])
                loss += w * self.head.forward_loss(pred, samples)
            return loss
        else:
            k, w = self.supv_keys[0], self.supv_weights[0]
            pred = self.head(feat[k])
            return self.head.forward_decode(pred)

        
@MODELS.register_module()
class MultiTaskPoseNet(nn.Module):
    def __init__(self, 
                 backbone=None,
                 neck=None,
                 heads=None,
                 pose_key='query',
                 bifurcate=True,
                 pretrained=None) -> None:
        super().__init__()
        self.backbone = build_model(backbone)
        self.neck = build_model(neck) if neck else nn.Identity()
        self.heads = nn.ModuleDict(
            {k:build_model(h_cfg) for k, h_cfg in heads.items()}
        )
        self.pose_key = pose_key
        self.bifurcate = bifurcate
        
        if pretrained is not None:
            self.from_pretrained(pretrained)
        

    def forward(self, samples: Dict[str, torch.Tensor], return_loss=False):
        feat = self.backbone(samples)
        feat = self.neck(feat)
        
        if isinstance(feat, torch.Tensor):
            feat = {self.pose_key: feat}
        
        if return_loss:
            loss = {}
            for key in self.heads:
                preds = self.heads[key](feat[key] if self.bifurcate else feat)
                loss_key = self.heads[key].forward_loss(preds, samples)
                if isinstance(loss_key, dict):
                    loss.update(loss_key)
                else:
                    loss[key] = loss_key
            return loss
        else:
            pred = self.heads[self.pose_key](feat[self.pose_key] if self.bifurcate else feat)
            return self.heads[self.pose_key].forward_decode(pred)
        
    def from_pretrained(
            self, 
            path: str, 
            drop_keys=[
                'keypoint_embedding',
                'heads.query',
                'heads.inter_query',
                'heads.collected_feat',
                'heads.feat'
            ]):
        logger = get_logger(__name__)
        logger.info(f"Loading pretrained model from {path}")
        if path.endswith('.safetensors'):
            state_dict: Dict[Tensor] = safetensors.torch.load_file(path, device='cpu')
        else:
            state_dict: Dict[Tensor] = torch.load(path, map_location='cpu')
        model_state_dict = self.state_dict()
        for k in model_state_dict:
            if k in state_dict:
                for dk in drop_keys:
                    if dk in k:
                        state_dict.pop(k)
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        logger.info(f"Missing keys: {missing}\nUnexpected keys: {unexpected}")
        return self
    
    
