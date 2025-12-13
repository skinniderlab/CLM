"""
Standalone Hyena components without registry dependencies.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def fftconv_ref(u, k, D, dropout_mask, gelu=True, k_rev=None):
    """Reference convolution with residual connection"""
    seqlen = u.shape[-1]
    fft_size = 2 * seqlen
    k_f = torch.fft.rfft(k, n=fft_size) / fft_size
    if k_rev is not None:
        k_rev_f = torch.fft.rfft(k_rev, n=fft_size) / fft_size
        k_f = k_f + k_rev_f.conj()
    u_f = torch.fft.rfft(u.to(dtype=k.dtype), n=fft_size)
    
    if len(u.shape) > 3:
        k_f = k_f.unsqueeze(1)

    y = torch.fft.irfft(u_f * k_f, n=fft_size, norm='forward')[..., :seqlen]

    out = y + u * D.unsqueeze(-1)
    if gelu:
        out = F.gelu(out)
    if dropout_mask is not None:
        return (out * rearrange(dropout_mask, 'b H -> b H 1')).to(dtype=u.dtype)
    else:
        return out.to(dtype=u.dtype)


@torch.jit.script
def mul_sum(q, y):
    return (q * y).sum(dim=1)


class Sin(nn.Module):
    """Sinusoidal activation function"""
    def __init__(self, dim, w=10, train_freq=True):
        super().__init__()
        self.freq = nn.Parameter(w * torch.ones(1, dim)) if train_freq else w * torch.ones(1, dim)

    def forward(self, x):
        return torch.sin(self.freq * x)


class PositionalEmbedding(nn.Module):
    """Complex exponential positional embeddings for Hyena filters"""
    def __init__(self, emb_dim: int, seq_len: int, **kwargs):
        super().__init__()
        
        self.seq_len = seq_len
        t = torch.linspace(0, 1, self.seq_len)[None, :, None]
        
        if emb_dim > 1:
            bands = (emb_dim - 1) // 2
        
        t_rescaled = torch.linspace(0, seq_len - 1, seq_len)[None, :, None]
        w = 2 * math.pi * t_rescaled / seq_len
        
        f = torch.linspace(1e-4, bands - 1, bands)[None, None]
        z = torch.exp(-1j * f * w)
        z = torch.cat([t, z.real, z.imag], dim=-1)
        
        self.register_buffer('z', z)
        self.register_buffer('t', t)
        
    def forward(self, L):
        return self.z[:, :L], self.t[:, :L]


class ExponentialModulation(nn.Module):
    """Exponential modulation for implicit filters"""
    def __init__(
        self,
        d_model,
        fast_decay_pct=0.3,
        slow_decay_pct=1.5,
        target=1e-2,
        modulate: bool = True,
        shift: float = 0.0,
        **kwargs
    ):
        super().__init__()
        self.modulate = modulate
        self.shift = shift
        max_decay = math.log(target) / fast_decay_pct
        min_decay = math.log(target) / slow_decay_pct
        deltas = torch.linspace(min_decay, max_decay, d_model)[None, None]
        self.register_buffer('deltas', deltas)
        
    def forward(self, t, x):
        if self.modulate:
            decay = torch.exp(-t * self.deltas.abs())
            x = x * (decay + self.shift)
        return x


class HyenaFilter(nn.Module):
    """Standalone Hyena filter without registry dependencies"""
    def __init__(
        self, 
        d_model,
        emb_dim=3,
        order=16,
        seq_len=1024,
        dropout=0.0,
        w=1,
        bias=True,
        num_inner_mlps=2,
        **kwargs
    ):
        super().__init__()
        self.d_model = d_model
        self.use_bias = bias
        self.bias = nn.Parameter(torch.randn(self.d_model))
        self.dropout = nn.Dropout(dropout)
        
        act = Sin(dim=order, w=w)
        self.emb_dim = emb_dim
        self.seq_len = seq_len
  
        self.pos_emb = PositionalEmbedding(emb_dim, seq_len)

        # Build MLP
        layers = [nn.Linear(emb_dim, order), act]
        for i in range(num_inner_mlps):
            layers.append(nn.Linear(order, order))
            layers.append(act)
        layers.append(nn.Linear(order, d_model, bias=False))
        
        self.implicit_filter = nn.Sequential(*layers)
        self.modulation = ExponentialModulation(d_model, **kwargs)

    def filter(self, L):
        z, t = self.pos_emb(L)
        h = self.implicit_filter(z)
        h = self.modulation(t, h)
        return h

    def forward(self, x, L, k=None, bias=None):
        if k is None:
            k = self.filter(L)
        
        k = k[0] if type(k) is tuple else k
        if bias is None:
            bias = self.bias
        bias = bias if self.use_bias else 0 * bias

        # Use reference implementation
        y = fftconv_ref(x, k, bias, dropout_mask=None, gelu=False)
        return y


class HyenaOperator(nn.Module):
    """Standalone Hyena operator without registry dependencies"""
    def __init__(
        self,
        d_model,
        l_max,
        order=2,
        filter_order=64,
        num_heads=1,
        inner_factor=1,
        num_blocks=1,
        dropout=0.0,
        filter_dropout=0.0,
        short_filter_order=3,
        **filter_args,
    ):
        super().__init__()
        
        assert d_model % num_heads == 0, f'Model dimension {d_model} must be divisible by num heads {num_heads}'
        assert l_max % num_blocks == 0, f'Maximum signal length {l_max} must be divisible by block dimension {num_blocks}'
        
        self.d_model = d_model
        self.order = order
        self.l_max = l_max
        self.num_heads = num_heads
        self.inner_factor = inner_factor
        self.num_blocks = num_blocks
        self.filter_order = filter_order
        self.short_filter_order = short_filter_order
        self.filter_dropout = filter_dropout
        
        self.block_dim = l_max // num_blocks
        self.head_dim = d_model // num_heads
        
        self.dropout = nn.Dropout(dropout)
        
        # Projections
        self.out_proj = nn.Linear(self.d_model * inner_factor, self.d_model)
        self.in_proj = nn.Linear(self.d_model, (self.order + 1) * self.d_model)
        
        # Short filter
        total_width = self.d_model * self.inner_factor * (self.order + 1)
        self.short_filter = nn.Conv1d(
            in_channels=total_width,
            out_channels=total_width,
            kernel_size=self.short_filter_order,
            groups=total_width,
            padding=self.short_filter_order - 1
        )
        
        # Long implicit filter
        self.filter_fn = HyenaFilter(
            self.head_dim * self.inner_factor * (self.order - 1),
            order=self.filter_order,
            seq_len=self.l_max,
            dropout=self.filter_dropout,
            **filter_args
        )

    def forward(self, u):
        l = u.size(-2)
        l_filter = min(l, self.l_max)
        
        u = self.in_proj(u)
        u = rearrange(u, 'b l d -> b d l')
        
        uc = self.short_filter(u)[..., :l_filter]
        
        uc = rearrange(
            uc, 'b (ho v) (z l) -> b ho v z l',
            z=self.num_blocks,
            ho=self.num_heads,
            v=self.head_dim * (self.order + 1)
        )

        *x, v = uc.split(self.d_model, dim=2)
        k = self.filter_fn.filter(l_filter)
        
        k = rearrange(k, 'c l (v o) -> c o v l', v=self.head_dim, o=self.order - 1)[0]
        bias = rearrange(self.filter_fn.bias, '(v o) -> o v', v=self.head_dim, o=self.order - 1)

        for o, x_i in enumerate(reversed(x[1:])):
            v = self.dropout(v * x_i)
            v = self.filter_fn(v, l_filter, k=k[o], bias=bias[o, None, :, None])

        y = rearrange(
            v * x[0], 'b h v z l -> b (z l) (h v)',
            z=self.num_blocks,
            h=self.num_heads
        )
        y = self.out_proj(y)
        
        return y

    @property
    def d_output(self):
        return self.d_model