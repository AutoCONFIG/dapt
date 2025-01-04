import torch
from torch import Tensor
import torch.nn as nn
from torch import distributions


'''
pack function: convert hidden_states to packed_hidden_states (batch_size=1)
'''
def pack(hidden_states, cu_seqlens):
    batch_size, seq_len, hidden_dim = hidden_states.shape
    seq_len_list = cu_seqlens[1:] - cu_seqlens[:-1]
    seq_len_list_3d = seq_len_list.unsqueeze(1).unsqueeze(2)
    indices_3d = (
        torch.arange(seq_len, device=hidden_states.device)
        .unsqueeze(0)
        .unsqueeze(2)
        .repeat(batch_size, 1, hidden_dim)
    )
    mask_3d = indices_3d < seq_len_list_3d
    packed_hidden_states = hidden_states[mask_3d].view(-1, hidden_dim)
    return packed_hidden_states


'''
unpack function: convert packed_hidden_states (batch_size=1) to hidden_states
'''
def unpack(packed_hidden_states, cu_seqlens):
    batch_size = cu_seqlens.shape[0] - 1
    seq_len = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
    hidden_dim = packed_hidden_states.shape[2]
    hidden_states = torch.zeros(batch_size, seq_len, hidden_dim, dtype=packed_hidden_states.dtype, device=packed_hidden_states.device)
    for i in range(batch_size):
        hidden_states[i, : cu_seqlens[i + 1] - cu_seqlens[i], :] = packed_hidden_states[:, cu_seqlens[i] : cu_seqlens[i + 1], :]
    return hidden_states