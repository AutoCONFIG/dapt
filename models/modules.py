import torch
from torch import nn, Tensor
from typing import Optional
from flash_attn import ( 
    flash_attn_varlen_kvpacked_func, 
    flash_attn_kvpacked_func
)
from einops import rearrange


class MultiHeadFlashAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, proj_drop=0.1,
                 bias=True, add_bias_kv=False, 
                 device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MultiHeadFlashAttention, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.dropout = dropout
        self.bias = bias
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.kv_proj = nn.Linear(embed_dim, embed_dim * 2, bias=bias, **factory_kwargs)
        
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.out_proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, 
                q: Tensor, 
                kv: Tensor = None, 
                cu_seqlens_q: Optional[Tensor] = None, 
                cu_seqlens_k: Optional[Tensor] = None,
                ):
        if kv is None: kv = q
        assert q.shape[-1] == kv.shape[-1], "The dimensions of q and k must be the same"
        
        no_varlen = cu_seqlens_k is None and cu_seqlens_q is None
        H, Dh = self.num_heads, self.head_dim
        if not no_varlen:
            if cu_seqlens_q is None:
                N, K, D = q.shape
                cu_seqlens_q = torch.arange(N + 1, device=q.device, dtype=torch.int32) * K
            if cu_seqlens_k is None:
                N, L, D = kv.shape
                cu_seqlens_k = torch.arange(N + 1, device=kv.device, dtype=torch.int32) * L
                
        q_proj = self.q_proj(q)
        kv_proj = self.kv_proj(kv)

        if no_varlen:
            N = q.shape[0]
            attn_output = flash_attn_kvpacked_func(
                q=rearrange(q_proj.to(torch.bfloat16), '... (h d) -> ... h d', h=H), 
                kv=rearrange(kv_proj.to(torch.bfloat16), '... (two h d) -> ... two h d', h=H, two=2), 
                dropout_p=self.dropout if self.training else 0
            )
        else:
            attn_output = flash_attn_varlen_kvpacked_func(
                q=q_proj.to(torch.bfloat16).reshape(-1, H, Dh), 
                kv=kv_proj.to(torch.bfloat16).reshape(-1, 2, H, Dh), 
                max_seqlen_q=(cu_seqlens_q[1:]-cu_seqlens_q[:1]).max(),
                max_seqlen_k=(cu_seqlens_k[1:]-cu_seqlens_k[:1]).max(),
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                dropout_p=self.dropout if self.training else 0
            )

        attn_output = attn_output.to(q.dtype).view(*q.shape)
        return self.out_proj_drop(self.out_proj(attn_output))


class TransformerDecoderLayerFlash(nn.TransformerDecoderLayer):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, 
                 dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5, norm_first: bool = False,
                 bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerDecoderLayerFlash, self).__init__(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            layer_norm_eps=layer_norm_eps, 
            norm_first=norm_first,
            bias=bias, 
            **factory_kwargs)
        self.self_attn = MultiHeadFlashAttention(d_model, nhead, dropout=dropout, **factory_kwargs)
        self.multihead_attn = MultiHeadFlashAttention(d_model, nhead, dropout=dropout, **factory_kwargs)

    def forward(
        self,
        tgt: Tensor,        
        memory: Tensor,
        cu_seqlens_tgt: Optional[Tensor] = None,
        cu_seqlens_memory: Optional[Tensor] = None,
    ) -> Tensor:
        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), cu_seqlens_tgt)
            x = x + self._mha_block(self.norm2(x), memory, cu_seqlens_tgt, cu_seqlens_memory)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, cu_seqlens_tgt))
            x = self.norm2(x + self._mha_block(x, memory, cu_seqlens_tgt, cu_seqlens_memory))
            x = self.norm3(x + self._ff_block(x))
        return x

    def _sa_block(self, x: Tensor, cu_seqlens_q: Optional[Tensor])  -> Tensor:
        x = self.self_attn(q=x,
                           cu_seqlens_q=cu_seqlens_q)
        return self.dropout1(x)

    def _mha_block(self, x: Tensor, mem: Tensor, 
                   cu_seqlens_q: Optional[Tensor], cu_seqlens_k: Optional[Tensor]) -> Tensor:
        x = self.multihead_attn(q=x, kv=mem,
                                cu_seqlens_q=cu_seqlens_q,
                                cu_seqlens_k=cu_seqlens_k)
        return self.dropout2(x)


class TransformerDecoderLayerPad(nn.TransformerDecoderLayer):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, 
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerDecoderLayerPad, self).__init__(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward,
            batch_first=True, 
            **factory_kwargs)

    def forward(
        self,
        tgt: Tensor,        
        memory: Tensor,
        cu_seqlens_tgt: Optional[Tensor] = None,
        cu_seqlens_memory: Optional[Tensor] = None,
    ) -> Tensor:
        memory_length = cu_seqlens_memory[1:] - cu_seqlens_memory[:-1]
        memory_key_padding_mask = torch.arange(memory_length.max().item(), 
                                               device=memory_length.device).unsqueeze(0) >= memory_length.unsqueeze(1)
        
        memory = torch.split_with_sizes(memory, memory_length.tolist())
        memory = torch.nn.utils.rnn.pad_sequence(memory, batch_first=True, padding_value=0)
        
        return super().forward(
            tgt=tgt, 
            memory=memory, 
            memory_key_padding_mask=memory_key_padding_mask
        )
