import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple, List, Dict

from typing import Union
from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import MLP
from pointcept.models.builder import MODELS, build_model
from .st_gcn import STGCN
from .smpl import SMPL
from .geometry import batch_rodrigues, rotation_matrix_to_angle_axis

HEADS = MODELS

@HEADS.register_module()
class Coord3dClsHead(nn.Module):
    def __init__(self, 
                 num_keypoints=14,
                 in_channel=64, 
                 grid_size=(72, 72, 120),
                 coord_range=[[-0.9, 0.9], 
                              [-0.9, 0.9], 
                              [-1.5, 1.5]],
                 beta=1.0,
                 sigma=2.0,
                 use_dark=True,
                 use_joint_wise_weight=False,
                 peak_func='argmax',
                 loss_weight=1.0) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.num_keypoints = num_keypoints
        joint_wise_weight = torch.ones(num_keypoints)
        if use_joint_wise_weight:
            joint_wise_weight[[4, 7, 10, 13]] = 2.0
        self.joint_wise_weight = nn.Parameter(joint_wise_weight, requires_grad=False)
        self.fcs = nn.ModuleList(MLP(in_channels=in_channel, out_channels=axis_grid_size)
                                 for axis_grid_size in grid_size)
        self.coord_range = nn.Parameter(torch.Tensor(coord_range), requires_grad=False)
        self.beta = beta
        self.sigma = sigma
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.kl_loss = nn.KLDivLoss(reduction='none')
        self.use_dark = use_dark
        self.peak_func = peak_func
        self.loss_weight = loss_weight

    def forward(self, query: torch.Tensor):
        B, K, C = query.shape
        coord_labels = [fc(query) for fc in self.fcs]            
        return coord_labels
    
    def criterion(self, dec_outs, labels):
        """Criterion function."""
        log_pt = self.log_softmax(dec_outs * self.beta)
        loss = torch.mean(self.kl_loss(log_pt, labels), dim=-1)
        return loss
    
    def forward_loss(self, 
                     preds: List[torch.Tensor], 
                     samples: Dict[str, torch.Tensor]):
        X, Y, Z = self.grid_size
        B, K = preds[0].shape[:2]
        targets = (
            samples['coord_label_x'].reshape(B, K, X),
            samples['coord_label_y'].reshape(B, K, Y),
            samples['coord_label_z'].reshape(B, K, Z),
        )
        weight = samples['keypoints_3d'].reshape(B, K, 4)[..., 3]
        loss = 0
        for pred, target in zip(preds, targets):
            loss = loss + self.criterion(pred, target).mul(weight).mul(self.joint_wise_weight[None, :]).sum() / \
                (weight * self.joint_wise_weight[None, :]).sum()
        return loss * self.loss_weight
    
    @staticmethod
    def _gaussian_blur_1d(input: torch.Tensor, 
                          blur_kernel_size: int):
        """
        input: [B, K, C]
        """
        sigma = 0.3*((blur_kernel_size-1)*0.5 - 1) + 0.8
        ksize_half = (blur_kernel_size - 1) * 0.5
        x = torch.linspace(-ksize_half, 
                           ksize_half, 
                           steps=blur_kernel_size,
                           device=input.device)
        pdf = torch.exp(-0.5 * (x / sigma).pow(2))
        kernel1d = pdf / pdf.sum()
        
        B, K, _ = input.shape
        blurred = F.conv1d(input.reshape(B*K, 1, -1), 
                            kernel1d[None, None, :], 
                            padding=blur_kernel_size//2)
        return blurred.reshape(B, K, -1)
    
    def _refine_simcc_dark(self,
                           keypoints: torch.Tensor, 
                           simcc: torch.Tensor,
                           blur_kernel_size: int) -> torch.Tensor:
        """
        keypoints: B K 1
        simcc: B K C
        """
        N = simcc.shape[0]

        # modulate simcc
        simcc = self._gaussian_blur_1d(simcc, blur_kernel_size)
        
        simcc = torch.clip(simcc, 1e-3, 50.).log()
        simcc = F.pad(simcc, (2, 2), 'replicate')

        px = (keypoints + 2.5).long().unsqueeze(-1)

        dx0 = torch.gather(simcc, dim=-1, index=px)
        dx1 = torch.gather(simcc, dim=-1, index=px+1)
        dx_1 = torch.gather(simcc, dim=-1, index=px-1)
        dx2 = torch.gather(simcc, dim=-1, index=px+2)
        dx_2 = torch.gather(simcc, dim=-1, index=px-2)

        dx = 0.5 * (dx1 - dx_1)
        dxx = 1e-9 + 0.25 * (dx2 - 2 * dx0 + dx_2)

        offset = dx / dxx
        if (offset > 1.0).any():
            return keypoints.float()

        return keypoints.float() - offset[..., 0]
    
    def forward_decode(self, preds: List[torch.Tensor]):
        coords = []
        for i, (pred, grid) in enumerate(zip(preds, self.grid_size)):
            coord = pred.argmax(dim=-1)
            if self.use_dark:
                sigma = int((self.sigma * 20 - 7) // 3)
                sigma -= int((sigma % 2) == 0)
                coord = self._refine_simcc_dark(coord, pred, sigma)
            coord = coord / grid * \
                (self.coord_range[i, 1] - self.coord_range[i, 0]) + self.coord_range[i, 0]
            coords.append(coord)
        return torch.stack(coords, axis=-1)


@HEADS.register_module()
class Coord3dHead(nn.Module):
    def __init__(self,
                 in_channel=64,
                 direct=False,
                 loss_weight=1.0) -> None:
        super().__init__()
        self.fc = nn.Identity() if direct else MLP(in_channels=in_channel, out_channels=3)
        self.loss_weight = loss_weight

    def forward(self, query: torch.Tensor):
        B, K, C = query.shape
        coord = self.fc(query)
        return coord
    
    def forward_decode(self, 
                       pred: torch.Tensor):
        return pred

    def forward_loss(self, 
                     pred: torch.Tensor,
                     samples: Dict[str, torch.Tensor]):
        """
        B K X Y Z
        """
        B, K = pred.shape[:2]
        keypoints_4d = samples['keypoints_3d'].reshape(B, K, 4)
        target = keypoints_4d[:, :, :3]
        weights = keypoints_4d[:, :, 3]
        loss = nn.functional.mse_loss(pred, target, reduction='none').sum(-1)
        loss = (loss * weights).sum() / weights.sum()
        return loss * self.loss_weight


@HEADS.register_module()
class SegmentationHead(nn.Module):
    def __init__(self, 
                 in_channel=64,
                 num_classes=23,
                 ignore_class=0,
                 loss_weight=1.0,
                 direct=False) -> None:
        super().__init__()
        self.fc = nn.Identity() if direct else nn.Sequential(
            nn.Linear(in_channel, in_channel),
            nn.BatchNorm1d(in_channel),
            nn.GELU(),
            nn.Linear(in_channel, num_classes),
        )
        self.loss_weight = loss_weight
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_class,
                                             reduction='mean')

    def forward(self, query: torch.Tensor):
        seg = self.fc(query)
        return seg
    
    def forward_decode(self, 
                       pred: torch.Tensor):
        return pred

    def forward_loss(self, 
                     pred: torch.Tensor,
                     samples: Dict[str, torch.Tensor]):
        if pred.ndim == 3:
            B, N, C = pred.shape
            pred = pred.reshape(B * N, C)
        target = samples['segment']
        loss = self.criterion(pred, target)
        return loss * self.loss_weight


@HEADS.register_module()
class InterCoord3dHead(nn.Module):
    def __init__(self, 
                 in_channel=12,
                 num_keypoints=14,
                 loss_weight=1.0) -> None:
        super().__init__()
        self.fc = MLP(in_channels=in_channel, out_channels=num_keypoints * 3)
        self.loss_weight = loss_weight
        
    def forward(self, inter_feat: torch.Tensor):
        N, C = inter_feat.shape
        coord = self.fc(inter_feat).reshape(N, -1, 3)
        return coord
    
    def forward_decode(self, 
                       pred: torch.Tensor):
        return pred

    def forward_loss(self, 
                     pred: torch.Tensor,
                     samples: Dict[str, torch.Tensor]):
        B, K = pred.shape[:2]
        keypoints_4d = samples['keypoints_3d'].reshape(B, K, 4)
        target = keypoints_4d[:, :, :3]
        weights = keypoints_4d[:, :, 3]
        loss = nn.functional.mse_loss(pred, target, reduction='none').sum(-1)
        loss = (loss * weights).sum() / weights.sum()
        return loss * self.loss_weight


@HEADS.register_module()
class SMPLOptimizerHead(nn.Module):
    def __init__(self,
                 loss_weight=1.0,
                 smpl_cfg=dict()) -> None:
        super().__init__()
        self.smpl = SMPL(**smpl_cfg)
        self.loss_weight = loss_weight
        self.criterion_param = nn.MSELoss()
        self.criterion_joints = nn.MSELoss()
        self.criterion_vertices = nn.MSELoss()
        
    def forward(self, x: Dict[str, torch.Tensor]):
        return x
    
    def forward_decode(self, 
                       preds: Dict[str, torch.Tensor]):
        decoded = dict()
        pred_rotmats = preds['pred_rotmats'].reshape(-1, 24, 3, 3)
        BT = pred_rotmats.shape[0]
        pred_human_vertices = self.smpl(
            pred_rotmats, 
            torch.zeros((BT, 10)).cuda()
        )
        pred_smpl_joints = self.smpl.get_full_joints(pred_human_vertices).reshape(BT, 24, 3)
        decoded['pred_keypoints_3d'] = pred_smpl_joints
        decoded['pred_pose'] = rotation_matrix_to_angle_axis(preds['pred_rotmats'].reshape(-1, 3, 3))
        return decoded

    def forward_loss(self, 
                     preds: Dict[str, torch.Tensor],
                     samples: Dict[str, torch.Tensor]):
        gt_rotmats = batch_rodrigues(samples['pose'].reshape(-1, 3)).reshape(-1, 24, 3, 3)
        gt_keypoints_3d = samples['keypoints_3d'].reshape(-1, 24, 4)
        gt_keypoints_weight = gt_keypoints_3d[..., 3:]
        gt_keypoints_3d = gt_keypoints_3d[..., :3]
        
        BT = gt_rotmats.shape[0]
        
        losses = dict()
        if 'pred_rotmats' in preds:
            # L_{\theta}
            pred_rotmats = preds['pred_rotmats'].reshape(-1, 24, 3, 3)
            loss_param = self.criterion_param(pred_rotmats, gt_rotmats)
            losses['param'] = loss_param

            # L_{J_{SMPL}}
            pred_human_vertices = self.smpl(
                pred_rotmats.reshape(-1, 24, 3, 3), torch.zeros((BT, 10)).cuda())
            pred_smpl_joints = self.smpl.get_full_joints(
                pred_human_vertices).reshape(BT, 24, 3)
            loss_smpl_joints = self.criterion_joints(
                pred_smpl_joints, gt_keypoints_3d)
            losses['smpl_joints'] = loss_smpl_joints

        if 'pred_keypoints_3d' in preds:
            # L_{J}
            pred_keypoints_3d = preds['pred_keypoints_3d']
            loss_keypoints_3d = self.criterion_joints(pred_keypoints_3d, gt_keypoints_3d)
            losses['keypoints_3d'] = loss_keypoints_3d
        
        return losses
