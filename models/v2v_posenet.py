

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
from einops import rearrange
import sys
from pointcept.models.modules import PointModule


class Basic3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super(Basic3DBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=1, padding=((kernel_size-1)//2)),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True)
        )
    
    def forward(self, x):
        return self.block(x)


class Res3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Res3DBlock, self).__init__()
        self.res_branch = nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True),
            nn.Conv3d(out_planes, out_planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_planes)
        )

        if in_planes == out_planes:
            self.skip_con = nn.Sequential()
        else:
            self.skip_con = nn.Sequential(
                nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm3d(out_planes)
            )
    
    def forward(self, x):
        res = self.res_branch(x)
        skip = self.skip_con(x)
        return F.relu(res + skip, True)

    
class Pool3DBlock(nn.Module):
    def __init__(self, pool_size):
        super(Pool3DBlock, self).__init__()
        self.pool_size = pool_size
    
    def forward(self, x):
        return F.max_pool3d(x, kernel_size=self.pool_size, stride=self.pool_size)
    

class Upsample3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride):
        super(Upsample3DBlock, self).__init__()
        assert(kernel_size == 2)
        assert(stride == 2)
        self.block = nn.Sequential(
            nn.ConvTranspose3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=0, output_padding=0),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)
    

class EncoderDecoder(nn.Module):
    def __init__(self):
        super(EncoderDecoder, self).__init__()

        self.encoder_pool1 = Pool3DBlock(2)
        self.encoder_res1 = Res3DBlock(32, 64)
        self.encoder_pool2 = Pool3DBlock(2)
        self.encoder_res2 = Res3DBlock(64, 128)

        self.mid_res = Res3DBlock(128, 128)

        self.decoder_res2 = Res3DBlock(128, 128)
        self.decoder_upsample2 = Upsample3DBlock(128, 64, 2, 2)
        self.decoder_res1 = Res3DBlock(64, 64)
        self.decoder_upsample1 = Upsample3DBlock(64, 32, 2, 2)

        self.skip_res1 = Res3DBlock(32, 32)
        self.skip_res2 = Res3DBlock(64, 64)

    def forward(self, x):
        skip_x1 = self.skip_res1(x)
        x = self.encoder_pool1(x)
        x = self.encoder_res1(x)
        skip_x2 = self.skip_res2(x)
        x = self.encoder_pool2(x)
        x = self.encoder_res2(x)

        x = self.mid_res(x)

        x = self.decoder_res2(x)
        x = self.decoder_upsample2(x)
        x = x + skip_x2
        x = self.decoder_res1(x)
        x = self.decoder_upsample1(x)
        x = x + skip_x1

        return x


class V2VModel(PointModule):
    def __init__(self, input_channels, output_channels):
        super(V2VModel, self).__init__()

        self.front_layers = nn.Sequential(
            Basic3DBlock(input_channels, 16, 7),
            Pool3DBlock(2),
            Res3DBlock(16, 32),
            Res3DBlock(32, 32),
            Res3DBlock(32, 32)
        )

        self.encoder_decoder = EncoderDecoder()

        self.back_layers = nn.Sequential(
            Res3DBlock(32, 32),
            Basic3DBlock(32, 32, 1),
            Basic3DBlock(32, 32, 1),
        )

        self.output_layer = nn.Conv3d(32, output_channels, kernel_size=1, stride=1, padding=0)

        self._initialize_weights()

    def forward(self, data_dict: Dict[str, torch.Tensor]):
        x = self.front_layers(x)
        x = self.encoder_decoder(x)
        x = self.back_layers(x)
        x = self.output_layer(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose3d):
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)


class HeatmapHead(nn.Module):
    def __init__(self, 
                 num_keypoints=14,
                 loss_weight=1.0,
                 grid_size=[36, 36, 60],
                 coord_range=[[-0.9, 0.9], 
                              [-0.9, 0.9], 
                              [-1.5, 1.5]],
                ) -> None:
        super().__init__()
        self.grid_size = nn.Parameter(torch.Tensor(grid_size), requires_grad=False)
        self.coord_range = nn.Parameter(torch.Tensor(coord_range), requires_grad=False)

    def forward(self, x):
        return x
    
    def forward_loss(self, 
                     pred: torch.Tensor, 
                     sample: Dict[str, torch.Tensor]):
        """
        Input:
            pred, gt: B K X Y Z
            gt_weight: B K
        """
        gt = sample['heatmap']
        gt_weight = sample['keypoints_3d_vis']
        loss = F.mse_loss(pred, gt, reduction='none').mean(dim=[-1,-2,-3])
        loss = (loss * gt_weight).sum() / gt_weight.sum()
        return loss
    
    def forward_decode(self, heatmap_3d: torch.Tensor):
        """
        Input:
            heatmap_3d: B K X Y Z
        Returns:
            coord: B K 3
        """
        X, Y, Z = heatmap_3d.shape[2:]
        spot_val, spot_idx = heatmap_3d.flatten(2).max(-1)
        spot_x = (spot_idx / (Y * Z)).to(dtype=torch.long)
        spot_y = ((spot_idx - spot_x * (Y * Z)) / Z).to(dtype=torch.long)
        spot_z = spot_idx % Z
        
        spot_coord = torch.stack([spot_x, spot_y, spot_z], dim=-1)
        spot_coord = spot_coord / self.grid_size * \
            (self.coord_range[None, None, :, 1] - self.coord_range[None, None, :, 0]) + self.coord_range[None, None, :, 0]
        return spot_coord