import torch
import math
import torch_scatter as tc
from torch import nn
from torch.nn import functional as F
from typing import Tuple, List, Dict
import spconv.pytorch as spconv
from pointcept.models.utils.structure import Point
from pointcept.models.modules import PointModule, PointSequential
from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import Block, MLP
from pointcept.models.builder import MODELS

from einops import rearrange

from .st_gcn import STGCN

NECKS = MODELS

def segment_softmax_csr(src, indprt, dim=0):
    seqlen = indprt[1:] - indprt[:-1]
    src_exp = torch.exp(src)
    src_exp_sum = torch.repeat_interleave(
        tc.segment_sum_csr(src=src_exp, indptr=indprt, dim=dim), 
        seqlen, 
        dim=0
    )
    return src_exp / src_exp_sum
    

@NECKS.register_module()
class SerializedKeypointsNeck(nn.Module):
    def __init__(self, 
                 num_keypoints=14,
                 initial=None,
                 depth=3,
                 channels=64,
                 num_heads=4,
                 patch_size=1024,
                 mlp_ratio=4,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.0,
                 proj_drop=0.0,
                 drop_path=0.0,
                 norm_layer=nn.LayerNorm,
                 act_layer=nn.GELU,
                 pre_norm=True,
                 enable_rpe=False,
                 enable_flash=True,
                 upcast_attention=False,
                 upcast_softmax=False,
        ) -> None:
        super().__init__()
        self.keypoint_embedding = nn.Parameter(torch.zeros(num_keypoints, 
                                                           channels))
        self.initial = initial or 'center'
        self.kp_dec = PointSequential()
        for i in range(depth):
            self.kp_dec.add(
                Block(
                    channels=channels,
                    num_heads=num_heads,
                    patch_size=patch_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    drop_path=drop_path,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    pre_norm=pre_norm,
                    order_index=0,
                    cpe_indice_key="stagek",
                    enable_rpe=enable_rpe,
                    enable_flash=enable_flash,
                    upcast_attention=upcast_attention,
                    upcast_softmax=upcast_softmax
                ),
                f"block{i}"
            )

    def insert_keypoint_query(self, 
                              point: Point,
                              coord: torch.Tensor):
        spconv_tensor = point.sparse_conv_feat
        feat: torch.Tensor = spconv_tensor.features
        indices: torch.Tensor = spconv_tensor.indices
        offset: torch.Tensor = point.offset

        B = spconv_tensor.batch_size
        K = self.keypoint_embedding.shape[0]
        N, C = feat.shape
        

        idx_q = torch.arange(B, device=feat.device) * K
        new_offset = offset + idx_q + K
        
        seq_len = offset.clone()
        seq_len[1:] = seq_len[1:] - seq_len[:-1]

        new_idx_p = torch.arange(N, device=feat.device) + idx_q.repeat_interleave(seq_len)
        new_idx_q = offset.repeat_interleave(K) + torch.arange(K, device=feat.device).repeat(B)

        # B*K C
        keypoint_queries = self.keypoint_embedding.repeat(B, 1)
        new_feat = torch.zeros(N + K * B, C, device=feat.device)
        new_feat[new_idx_p, :] = feat
        new_feat[new_idx_q, :] = keypoint_queries
        
        q_batch_indices = torch.arange(B, device=feat.device).repeat_interleave(K)
        keypoint_indices = torch.cat([q_batch_indices.unsqueeze(1), 
                                      coord.reshape(-1, 3)], dim=-1).to(dtype=indices.dtype)
        new_indices = torch.zeros(N + K * B, 4, device=indices.device, dtype=indices.dtype)
        new_indices[new_idx_p, :] = indices
        new_indices[new_idx_q, :] = keypoint_indices

        new_tensor = spconv.SparseConvTensor(
            features=new_feat,
            indices=new_indices,
            spatial_shape=spconv_tensor.spatial_shape,
            batch_size=spconv_tensor.batch_size
        )

        # reset the point order
        order = torch.arange(new_feat.shape[0], device=feat.device).unsqueeze(0)
        inverse = order
        
        point.sparse_conv_feat = new_tensor
        point.feat = new_tensor.features
        point.offset = new_offset
        point.query_indices = new_idx_q
        point.serialized_order = order
        point.serialized_inverse = inverse
        point.pop('pad')
        point.pop('unpad')
        point.pop('cu_seqlens_key')
        return point


    def gather_keypoint_query(self, point: Point):
        queries: torch.Tensor = point.feat
        queries = queries[point.query_indices, :].reshape(point.sparse_conv_feat.batch_size, 
                                                          self.keypoint_embedding.shape[0],
                                                          -1)
        return queries

    
    def forward(self, point: Point):
        if self.initial == 'center':
            init_coord = torch.tensor([c / 2 for c in point.sparse_shape], 
                                      device=point.feat.device)
            init_coord = init_coord.unsqueeze(0).repeat(self.keypoint_embedding.shape[0], 1) \
                                   .unsqueeze(0).repeat(point.sparse_conv_feat.batch_size, 1, 1)
        
        point = self.insert_keypoint_query(point, init_coord)
        point = self.kp_dec(point)
        queries = self.gather_keypoint_query(point)
        return queries


@NECKS.register_module()
class SerializedKeypointsCANeck(nn.Module):
    def __init__(self, 
                 num_keypoints=14,
                 channels=64,
                 mlp_ratio=4,
                 num_heads=4,
                 depth=4) -> None:
        super().__init__()
        self.keypoint_embedding = nn.Parameter(torch.zeros(num_keypoints, 
                                                           channels))
        self.keypoint_transformer = nn.ModuleList(
            nn.TransformerDecoderLayer(d_model=channels, 
                                       dim_feedforward=mlp_ratio * channels, 
                                       nhead=num_heads, 
                                       batch_first=True)
            for _ in range(depth)
        )
        self.keypoint_norm = nn.LayerNorm(channels)


    def forward(self, point: Point):
        memory_length = point.offset.clone()
        memory_length[1:] -= memory_length[:-1].clone()
        
        memory_key_padding_mask = torch.arange(memory_length.max().item(), 
                                               device=memory_length.device).unsqueeze(0) >= memory_length.unsqueeze(1)
        
        memory = torch.split_with_sizes(point.feat, memory_length.tolist())
        memory = torch.nn.utils.rnn.pad_sequence(memory, batch_first=True, padding_value=0)
        
        query = self.keypoint_embedding.expand(memory_length.shape[0], -1, -1)
        
        for layer in self.keypoint_transformer:
            query = layer(tgt=query, memory=memory, memory_key_padding_mask=memory_key_padding_mask)
        query = self.keypoint_norm(query)
        point.query = query
        return point


@NECKS.register_module()
class SerializedKeypointsStackedCANeck(nn.Module):
    def __init__(self, 
                 n_layers=4,
                 num_keypoints=15,
                 channels=128,
                 mlp_ratio=4,
                 num_heads=8,
                 depth=6) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.keypoint_embedding = nn.Parameter(torch.zeros(num_keypoints, 
                                                           channels))
        self.keypoint_transformer = nn.ModuleList(
            nn.TransformerDecoderLayer(d_model=channels, 
                                       dim_feedforward=mlp_ratio * channels, 
                                       nhead=num_heads, 
                                       batch_first=True)
            for _ in range(depth)
        )
        self.keypoint_norm = nn.LayerNorm(channels)

    def forward(self, point: Point):
        cu_seqlens_all = torch.stack([point.get(f"query_mem_{i}_cu_seqlens")
                                      for i in range(self.n_layers)])
        mem_all = torch.cat([point[f'query_mem_{i}'] for i in range(self.n_layers)], dim=0)
        cu_mem_len = F.pad(cu_seqlens_all[:, -1], (1, 0)).cumsum(0).to(dtype=torch.int32)
        
        indices = []
        for i in range(cu_seqlens_all.shape[1] - 1):
            for n in range(cu_seqlens_all.shape[0]):
                indices.append(torch.arange(cu_seqlens_all[n, i], 
                                            cu_seqlens_all[n, i+1], 
                                            device=cu_seqlens_all.device) + cu_mem_len[n])
        indices = torch.cat(indices)
        mem_all = mem_all[indices, :]
        
        cu_seqlens_all = cu_seqlens_all.sum(0)
        memory_length = cu_seqlens_all[1:] - cu_seqlens_all[:-1]
        memory_key_padding_mask = torch.arange(memory_length.max().item(), 
                                               device=memory_length.device).unsqueeze(0) >= memory_length.unsqueeze(1)
        memory = torch.split_with_sizes(mem_all, memory_length.tolist())
        memory = torch.nn.utils.rnn.pad_sequence(memory, batch_first=True, padding_value=0)
        
        query = self.keypoint_embedding.expand(memory_length.shape[0], -1, -1)
        
        for layer in self.keypoint_transformer:
            query = layer(tgt=query, memory=memory, memory_key_padding_mask=memory_key_padding_mask)
        query = self.keypoint_norm(query)
        point.query = query
        return point


@NECKS.register_module()
class STGCNNeck(nn.Module):
    def __init__(self, 
                 channels=128,
                 out_channels=3,
                 seq_len=16,
                 stgcn_use_coord=False,
                 stgcn_residual=False, 
                 graph_cfg=None) -> None:
        super(STGCNNeck, self).__init__()
        self.seq_len = seq_len
        self.stgcn_use_coord = stgcn_use_coord
        self.stgcn_residual = stgcn_residual
        self.out_channels = out_channels
        self.coord_fc = MLP(in_channels=channels, out_channels=3)
        self.stgcn = STGCN(channels if not stgcn_use_coord else channels + 3, 
                           out_channels, 
                           graph_cfg=graph_cfg)
    
    def forward(self, point: Point):
        BT, K, D = point.query.shape
        query: torch.Tensor = point.query.reshape(-1, self.seq_len, K, D)
        coord = self.coord_fc(point.query)
        point.query = coord
        if self.stgcn_use_coord:
            query = torch.cat([query, coord.reshape(-1, self.seq_len, K, self.out_channels)], dim=-1)
            point.query_stgcn = self.stgcn(query).reshape(BT, K, self.out_channels) + (coord if self.stgcn_residual else 0)
        else:
            point.query_stgcn = self.stgcn(query).reshape(BT, K, self.out_channels) + (coord if self.stgcn_residual else 0)
        return point


@NECKS.register_module()
class PRNNeck(nn.Module):
    def __init__(self, 
                 num_keypoints=14,
                 channels=64,
                 depth=3) -> None:
        super().__init__()
        
        self.num_keypoints = num_keypoints
        self.keypoint_embedding = nn.Parameter(torch.zeros(num_keypoints, channels))
        self.keypoint_fc = nn.Linear(channels + 3, channels)
        self.keypoint_fc_tr = nn.Linear(channels + 3, channels)
        self.keypoint_transformer = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model=channels, 
                nhead=4, 
                dim_feedforward=channels * 4,
                batch_first=True
            )
            for _ in range(depth)
        )
        self.keypoint_norm = nn.LayerNorm(channels)
        self.keypoint_coord_fc = nn.Sequential(
            nn.Linear(channels, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.Linear(8, 3),
        )
        
        self.seg_fc = nn.Sequential(
            nn.Linear(channels * 3, channels * 2),
            nn.BatchNorm1d(channels * 2),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(channels * 2, channels),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Linear(channels, num_keypoints + 1),
        )
        
        self.reg_fc = nn.Sequential(
            nn.Linear(channels * 3, channels * 2),
            nn.BatchNorm1d(channels * 2),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(channels * 2, channels),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Linear(channels, 3),
        )

    def forward(self, point: Point):
        memory_length = point.offset.clone()
        memory_length[1:] -= memory_length[:-1].clone()
        indptr = nn.functional.pad(point.offset, (1, 0))
        B = memory_length.shape[0]
        K = self.num_keypoints
        
        ### Vote
        feat_max, _ = tc.segment_max_csr(src=point.feat, indptr=indptr)
        feat_mean = tc.segment_mean_csr(src=point.feat, indptr=indptr)
        feat_all = torch.cat([
            point.feat,
            torch.repeat_interleave(feat_max, memory_length, dim=0),
            torch.repeat_interleave(feat_mean, memory_length, dim=0),
        ], dim=-1)
        
        seg = self.seg_fc(feat_all)
        seg_softmax = torch.softmax(seg, dim=-1)
        reg = self.reg_fc(feat_all)
        
        kpt_cls_max, _ = seg_softmax.max(dim=-1)
        kpt_cls_mask = (kpt_cls_max > 0.6) & (seg_softmax[:, 0] < 0.6)

        pose_r = (reg + point.coord)[:, None, :]
        pose_p = seg_softmax[:, :, None] * kpt_cls_mask[..., None, None]

        pose_r = tc.segment_sum_csr(src=pose_r * pose_p, indptr=indptr) / \
                 (tc.segment_sum_csr(src=pose_p, indptr=indptr) + 1e-8)
        point.seg = seg
        point.pose_r = pose_r[:, 1:, :]

        ### Refine
        attn_mask = torch.arange(memory_length.max().item() + K, 
                                 device=memory_length.device).unsqueeze(0) >= memory_length.unsqueeze(1) + K
        memory = torch.cat([point.coord, point.feat], dim=-1)
        memory = torch.split_with_sizes(memory, memory_length.tolist())
        memory = torch.nn.utils.rnn.pad_sequence(memory, batch_first=True, padding_value=0)
        
        query = self.keypoint_embedding.expand(memory_length.shape[0], -1, -1)
        query = self.keypoint_fc(torch.cat([pose_r[:, 1:, :], query], dim=-1))
        query = torch.cat([pose_r[:, 1:, :], query], dim=-1)

        x = torch.cat([query, memory], dim=1)
        x = self.keypoint_fc_tr(x)
        for layer in self.keypoint_transformer:
            x = layer(src=x, src_key_padding_mask=attn_mask)
        x = self.keypoint_norm(x)

        pose = self.keypoint_coord_fc(x[:, :K, :])
        point.pose = pose

        return point