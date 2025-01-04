from functools import partial
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch_scatter import scatter, segment_csr
from einops import rearrange, repeat

import spconv.pytorch as spconv
from pointcept.models.point_prompt_training import PDNorm
from pointcept.models.builder import MODELS
from pointcept.models.utils.structure import Point
from pointcept.models.modules import PointModule, PointSequential
from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import (
    Embedding, SerializedPooling, SerializedUnpooling, MLP, SerializedAttention, DropPath
)

from ..modules import TransformerDecoderLayerFlash, TransformerDecoderLayerPad

def collect_inst_feat(point: Point, reduce='max'):
    features = point.feat
    index = point.batch
    inst_features = scatter(src=features, index=index, dim=0, reduce=reduce)
    return inst_features

class MDE(PointModule):
    def __init__(self,
                 in_channels=128,
                 mem_channels=128,
                 out_channels=64,
                 nhead=4,
                 idx=0,
                 mixer='flash_attn'):
        super().__init__()
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
        elif mixer == 'attn':
             transformer = TransformerDecoderLayerPad(
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

    
    def forward(self, point: Point):
        query = point.query
        query = self.fc_in(query)
        query = self.transformer(tgt=query, 
                                 memory=point.feat,
                                 cu_seqlens_memory=F.pad(point.offset, (1, 0)).to(torch.int32))
        query = self.fc_out(query)
        point.query = query
        point[f"query_{self.idx}_coord"] = point.coord
        point[f"query_{self.idx}"] = query
        return point


class SerializedStatedPooling(SerializedPooling):
    def forward(self, point: Point):
        point_dict = super().forward(point)
        for k in point.keys():
            if 'query' in k:
                point_dict[k] = point[k]
        return point_dict


class SerializedStatedUnpooling(SerializedUnpooling):
    def forward(self, point: Point):
        point_dict = super().forward(point)
        for k in point.keys():
            if 'query' in k:
                point_dict[k] = point[k]
        return point_dict


class Block(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size=48,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        pre_norm=True,
        order_index=0,
        cpe_indice_key=None,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,
    ):
        super().__init__()
        self.channels = channels
        self.pre_norm = pre_norm

        self.cpe = PointSequential(
            spconv.SubMConv3d(
                channels,
                channels,
                kernel_size=3,
                bias=True,
                indice_key=cpe_indice_key,
            ),
            nn.Linear(channels, channels),
            norm_layer(channels),
        )

        self.norm1 = PointSequential(norm_layer(channels))
        self.attn = SerializedAttention(
            channels=channels,
            patch_size=patch_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            order_index=order_index,
            enable_rpe=enable_rpe,
            enable_flash=enable_flash,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
        )
        self.norm2 = PointSequential(norm_layer(channels))
        self.mlp = PointSequential(
            MLP(
                in_channels=channels,
                hidden_channels=int(channels * mlp_ratio),
                out_channels=channels,
                act_layer=act_layer,
                drop=proj_drop,
            )
        )
        self.drop_path = PointSequential(
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

    def forward(self, point: Point):
        
        self.info_pooled_coord = point.coord
        
        shortcut = point.feat
        point = self.cpe(point)
        point.feat = shortcut + point.feat
        shortcut = point.feat
        if self.pre_norm:
            point = self.norm1(point)
        point = self.drop_path(self.attn(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm1(point)

        shortcut = point.feat
        if self.pre_norm:
            point = self.norm2(point)
        point = self.drop_path(self.mlp(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm2(point)
        point.sparse_conv_feat.replace_feature(point.feat)
        
        return point


@MODELS.register_module("PT-v3m1-dapt")
class PointTransformerV3(PointModule):
    def __init__(
        self,
        num_keypoints=14,
        num_extra=0,
        in_channels=6,
        order=("z", "z_trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(48, 48, 48, 48, 48),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_kp_channels=(128, 128, 128, 256, 256),
        dec_kp_mixer='flash_attn',
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(48, 48, 48, 48),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
    ):
        super().__init__()
        self.num_stages = len(enc_depths)
        self.order = [order] if isinstance(order, str) else order
        self.cls_mode = cls_mode
        self.shuffle_orders = shuffle_orders

        assert self.num_stages == len(stride) + 1
        assert self.num_stages == len(enc_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(enc_num_head)
        assert self.num_stages == len(enc_patch_size)
        # assert self.cls_mode or self.num_stages == len(dec_depths) + 1
        # assert self.cls_mode or self.num_stages == len(dec_channels) + 1
        # assert self.cls_mode or self.num_stages == len(dec_num_head) + 1
        # assert self.cls_mode or self.num_stages == len(dec_patch_size) + 1

        # norm layers
        if pdnorm_bn:
            bn_layer = partial(
                PDNorm,
                norm_layer=partial(
                    nn.BatchNorm1d, eps=1e-3, momentum=0.01, affine=pdnorm_affine
                ),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
            )
        else:
            bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        if pdnorm_ln:
            ln_layer = partial(
                PDNorm,
                norm_layer=partial(nn.LayerNorm, elementwise_affine=pdnorm_affine),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
            )
        else:
            ln_layer = nn.LayerNorm
        # activation layers
        act_layer = nn.GELU

        self.embedding = Embedding(
            in_channels=in_channels,
            embed_channels=enc_channels[0],
            norm_layer=bn_layer,
            act_layer=act_layer,
        )

        # encoder
        enc_drop_path = [
            x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))
        ]
        self.enc = PointSequential()
        for s in range(self.num_stages):
            enc_drop_path_ = enc_drop_path[
                sum(enc_depths[:s]) : sum(enc_depths[: s + 1])
            ]
            enc = PointSequential()
            if s > 0:
                enc.add(
                    SerializedStatedPooling(
                        in_channels=enc_channels[s - 1],
                        out_channels=enc_channels[s],
                        stride=stride[s - 1],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="down",
                )
            for i in range(enc_depths[s]):
                enc.add(
                    Block(
                        channels=enc_channels[s],
                        num_heads=enc_num_head[s],
                        patch_size=enc_patch_size[s],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=enc_drop_path_[i],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % len(self.order),
                        cpe_indice_key=f"stage{s}",
                        enable_rpe=enable_rpe,
                        enable_flash=enable_flash,
                        upcast_attention=upcast_attention,
                        upcast_softmax=upcast_softmax,
                    ),
                    name=f"block{i}",
                )
            if len(enc) != 0:
                self.enc.add(module=enc, name=f"enc{s}")

        # decoder
        if not self.cls_mode:
            num_dec_stages = len(dec_channels)
            dec_drop_path = [
                x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))
            ]
            self.dec = PointSequential()
            dec_channels = list(dec_channels) + [enc_channels[-1]]
            dec_kp_channels = list(dec_kp_channels)
            for s in reversed(range(num_dec_stages)):
                dec_drop_path_ = dec_drop_path[
                    sum(dec_depths[:s]) : sum(dec_depths[: s + 1])
                ]
                dec_drop_path_.reverse()
                dec = PointSequential()
                dec.add(
                    SerializedStatedUnpooling(
                        in_channels=dec_channels[s + 1],
                        skip_channels=enc_channels[s],
                        out_channels=dec_channels[s],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="up",
                )
                for i in range(dec_depths[s]):
                    dec.add(
                        Block(
                            channels=dec_channels[s],
                            num_heads=dec_num_head[s],
                            patch_size=dec_patch_size[s],
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            attn_drop=attn_drop,
                            proj_drop=proj_drop,
                            drop_path=dec_drop_path_[i],
                            norm_layer=ln_layer,
                            act_layer=act_layer,
                            pre_norm=pre_norm,
                            order_index=i % len(self.order),
                            cpe_indice_key=f"stage{s}",
                            enable_rpe=enable_rpe,
                            enable_flash=enable_flash,
                            upcast_attention=upcast_attention,
                            upcast_softmax=upcast_softmax,
                        ),
                        name=f"block{i}",
                    )
                self.dec.add(module=dec, name=f"dec{s}")
                if s+1 < len(dec_kp_channels):
                    self.dec.add(
                        module=MDE(
                            in_channels=dec_kp_channels[s+1],
                            mem_channels=dec_channels[s],
                            out_channels=dec_kp_channels[s],
                            mixer=dec_kp_mixer,
                            idx=s
                        ), 
                        name=f"dec_ke{s}")
        
        self.keypoint_embedding = nn.Parameter(torch.zeros(num_keypoints + num_extra, dec_kp_channels[-1]))
        self.num_keypoints = num_keypoints

    def forward(self, data_dict):
        point = Point(data_dict)
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()

        point = self.embedding(point)
        point = self.enc(point)
        
        B = point.batch[-1].item() + 1
        inter_feat = collect_inst_feat(point)
        point.query = self.keypoint_embedding.unsqueeze(0).repeat(B, 1, 1)

        if not self.cls_mode:
            point = self.dec(point)

        point.collected_feat = segment_csr(
            src=point.feat,
            indptr=nn.functional.pad(point.offset, (1, 0)),
            reduce="mean",
        )
        point.inter_feat = inter_feat
        return point