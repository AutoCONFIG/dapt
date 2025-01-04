from typing import Iterable, Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from einops import rearrange
from addict import Addict

from pointcept.models.modules import PointModule, PointSequential
from pointnet2_ops.pointnet2_modules import PointnetSAModule, PointnetFPModule
from pointcept.models.builder import MODELS
from pointcept.models.utils.structure import Point
from ..modules import TransformerDecoderLayerFlash

class KeypointExchange(PointModule):
    def __init__(self,
                 in_channels=128,
                 mem_channels=None,
                 out_channels=64,
                 nhead=4,
                 idx=0,
                 mixer='flash_attn'):
        super().__init__()
        mem_channels = mem_channels or in_channels
        self.fc_in = nn.Sequential(
            nn.Linear(in_channels, mem_channels),
            nn.LayerNorm(mem_channels)
        ) if in_channels != mem_channels else nn.Identity()
        
        if mixer == 'flash_attn':
            transformer = TransformerDecoderLayerFlash(
                d_model=mem_channels, 
                nhead=nhead,
                dim_feedforward=4 * mem_channels
            )
        
        self.transformer = transformer
        self.fc_out = nn.Sequential(
            nn.Linear(mem_channels, out_channels),
            nn.LayerNorm(out_channels)
        ) if out_channels != mem_channels else nn.Identity()
        self.idx = idx

    
    def forward(self, query, memory):
        query = self.fc_in(query)
        query = self.transformer(tgt=query, memory=memory)
        query = self.fc_out(query)
        return query


@MODELS.register_module()
class PointNet2Encoder(nn.Module):
    def __init__(self, 
                 num_keypoints=24,
                 num_points=512,
                 enc_pretrain=False):
        super().__init__()
        self.num_points = num_points
        self.enc_pretrain = enc_pretrain
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=256,
                radius=0.1,
                nsample=32,
                mlp=[0, 64, 64, 128],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=128,
                radius=0.2,
                nsample=32,
                mlp=[128, 128, 128, 256],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=64,
                radius=0.4,
                nsample=32,
                mlp=[256, 256, 256, 512],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=32,
                radius=0.8,
                nsample=32,
                mlp=[512, 512, 512, 512],
                use_xyz=True,
            )
        )

        
        if enc_pretrain:
            self.FP_modules = nn.ModuleList([
                PointnetFPModule(mlp=[128 + 0, 128, 64]),
                PointnetFPModule(mlp=[256 + 128, 256, 128]),
                PointnetFPModule(mlp=[256 + 256, 256, 256]),
                PointnetFPModule(mlp=[512 + 512, 256, 256]),
            ])

        self.KE_modules = nn.ModuleList([
            # KeypointExchange(in_channels=128, out_channels=256, nhead=8, idx=0),
            KeypointExchange(in_channels=256, out_channels=512, nhead=16, idx=1),
            KeypointExchange(in_channels=512, out_channels=512, nhead=32, idx=2),
            KeypointExchange(in_channels=512, out_channels=512, nhead=32, idx=3),
        ]) 
        self.keypoint_embedding = nn.Parameter(torch.zeros(1, num_keypoints, 256))

    def _break_up_pc(self, pc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        xyz = pc[..., :3].contiguous()
        features = pc[..., 3:].transpose(
            1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

    def forward(self, data):
        data = Addict(data)
        x = data.feat
        x = x.reshape(-1, self.num_points, 3)
        query = self.keypoint_embedding.expand(x.shape[0], -1, -1)
        
        # Encoding ~
        xyz, features = self._break_up_pc(x)
        l_xyz, l_features = [xyz], [features]
        for i, sa_module in enumerate(self.SA_modules):
            li_xyz, li_features = sa_module(l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)
        
        data.feat = li_features.transpose(1, 2).flatten(0, 1)
        data.feat_inter = li_features.transpose(1, 2).flatten(0, 1)
            
        # Keypoint Exchanging ~
        si = len(self.SA_modules) - len(self.KE_modules)
        for i, ke_module in enumerate(self.KE_modules):
            query = ke_module(query, l_features[i+si+1].transpose(1, 2))
            data[f'query_{ke_module.idx}'] = query
        data.query = query
        
        # Decoding ~
        if self.enc_pretrain:
            for i, fp_module in reversed([*enumerate(self.FP_modules)]):
                l_features[i] = fp_module(l_xyz[i], l_xyz[i+1], l_features[i], l_features[i+1])
            data.feat = l_features[i].transpose(1, 2).flatten(0, 1)
        return data