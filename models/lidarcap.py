from .geometry import rot6d_to_rotmat
from .st_gcn import STGCN
from pointcept.models.builder import MODELS
from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import Embedding, Block
from pointcept.models.modules import PointModule
from pointcept.datasets.point import Point
from typing import Tuple
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F

def to_sequence(x: dict):
    B = x['frame_group'].max().item() + 1
    x['feat'] = rearrange(x['feat'], '(b t n) d -> b t n d', b=B, n=512)
    return x

class PointTransformerEncoder(PointModule):
    """使用Point Transformer V3编码器，具有注意力权重聚合"""
    
    def __init__(self, in_channels=3, embed_channels=256, depth=4, num_heads=8):
        super().__init__()
        
        # 特征嵌入层
        self.embedding = Embedding(
            in_channels=in_channels,
            embed_channels=embed_channels,
            norm_layer=nn.LayerNorm,
            act_layer=nn.GELU
        )
        
        # Transformer块
        self.blocks = nn.ModuleList([
            Block(
                channels=embed_channels,
                num_heads=num_heads,
                patch_size=48,  # 调整patch size以适应点云密度
                mlp_ratio=4.0,
                qkv_bias=True,
                drop_path=0.1 * i / max(depth - 1, 1),  # 修复除零错误
                enable_flash=True,
                enable_rpe=True  # 启用相对位置编码
            ) for i in range(depth)
        ])
        
        # 注意力权重聚合器
        self.feature_aggregator = nn.Sequential(
            nn.Linear(embed_channels, embed_channels // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_channels // 2, 1)  # 输出注意力权重
        )
        
        # 层归一化
        self.norm = nn.LayerNorm(embed_channels)
        
        # 输出头，映射到1024维
        self.head = nn.Sequential(
            nn.Linear(embed_channels, embed_channels * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_channels * 2, embed_channels * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_channels * 4, 1024)
        )
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                torch.nn.init.constant_(m.bias, 0)
                torch.nn.init.constant_(m.weight, 1.0)
        
    def forward(self, data):
        x = data['feat']  # (B, T, N, 3)
        B, T, N, _ = x.shape
        x = x.reshape(B * T, N, 3)

        # 创建Point对象
        point = Point(
            coord=x,
            feat=x,  # 初始特征就是坐标
            # offset=torch.arange(0, (B * T + 1) * N, N, device=x.device, dtype=torch.long)
            offset=torch.arange(0, (B * T + 1) * N, N, device=x.device)
        )
        
        # 特征嵌入
        point = self.embedding(point)
        
        # 通过Transformer块处理
        for block in self.blocks:
            point = block(point)
        
        # 注意力权重进行特征聚合
        features = point.feat
        # 按点云分组并池化
        pooled_features = []
        
        for i in range(B * T):
            start_idx = i * N
            end_idx = (i + 1) * N
            point_features = features[start_idx:end_idx]  # (N, embed_channels)
            
            # 计算注意力权重
            attention_logits = self.feature_aggregator(point_features).squeeze(-1)  # (N,)
            attention_weights = F.softmax(attention_logits, dim=0)  # (N,)
            
            # 加权聚合特征
            pooled = torch.sum(point_features * attention_weights.unsqueeze(-1), dim=0)  # (embed_channels,)
            pooled_features.append(pooled)
        
        # 堆叠所有聚合特征
        features = torch.stack(pooled_features)  # (B*T, embed_channels)
        
        # 层归一化
        features = self.norm(features)
        
        # 通过输出头映射到目标维度
        features = self.head(features)  # (B*T, 1024)
        
        return features.reshape(B, T, -1)  # (B, T, 1024)


class RNN(nn.Module):
    """改进的RNN模块，增加了正则化"""
    
    def __init__(self, n_input, n_output, n_hidden, n_rnn_layer=2, dropout_rate=0.1):
        super(RNN, self).__init__()
        
        self.rnn = nn.GRU(
            n_hidden, n_hidden, n_rnn_layer,
            batch_first=True, 
            bidirectional=True,
            dropout=dropout_rate if n_rnn_layer > 1 else 0
        )
        
        self.linear1 = nn.Linear(n_input, n_hidden)

        self.linear2 = nn.Linear(n_hidden * 2, n_output)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(n_hidden)
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化RNN权重"""
        for name, param in self.rnn.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                
    def forward(self, x):  # (B, T, D)
        # 输入层 + 层归一化
        x = self.layer_norm(self.linear1(x))
        x = F.relu(self.dropout(x), inplace=True)
        
        # RNN处理
        x, _ = self.rnn(x)  # (B, T, n_hidden * 2)
        
        # 输出层
        return self.linear2(x)


@MODELS.register_module()
class LiDARCap(nn.Module):
    """改进的LiDARCap模型，使用Point Transformer V3作为编码器"""
    
    def __init__(self, graph_cfg=None, encoder_cfg=None):
        super().__init__()
        
        # 编码器配置
        encoder_config = encoder_cfg or {
            'in_channels': 3,
            'embed_channels': 256,
            'depth': 4,
            'num_heads': 8
        }
        
        # 模型组件
        self.encoder = PointTransformerEncoder(**encoder_config)
        self.pose_s1 = RNN(1024, 24 * 3, 1024, dropout_rate=0.1)
        self.pose_s2 = STGCN(3 + 1024, 6, graph_cfg=graph_cfg)
        
    def forward(self, data):
        pred = {}
        
        # 转换为序列格式
        data = to_sequence(data)
        
        # 特征编码
        x = self.encoder(data)  # (B, T, 1024)
        B, T, _ = x.shape
        
        # 第一阶段：预测关节位置
        full_joints = self.pose_s1(x)  # (B, T, 24 * 3)
        
        # 第二阶段：预测旋转
        # 将关节位置和全局特征拼接
        joints_reshaped = full_joints.reshape(B, T, 24, 3)  # (B, T, 24, 3)
        global_feat_expanded = x.unsqueeze(-2).repeat(1, 1, 24, 1)  # (B, T, 24, 1024)
        
        combined_input = torch.cat((joints_reshaped, global_feat_expanded), dim=-1)  # (B, T, 24, 3+1024)
        
        # 预测6D旋转表示
        rot6ds = self.pose_s2(combined_input)  # (B, T, 24, 6)
        
        # 转换为旋转矩阵
        rot6ds = rot6ds.reshape(-1, rot6ds.size(-1))  # (B * T * 24, 6)
        rotmats = rot6d_to_rotmat(rot6ds).reshape(-1, 3, 3)  # (B * T * 24, 3, 3)
        
        # 输出预测结果
        pred['pred_rotmats'] = rotmats.reshape(B * T, 24, 3, 3)
        pred['pred_keypoints_3d'] = full_joints.reshape(B * T, 24, 3)
        
        return pred
