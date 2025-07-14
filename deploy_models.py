from deploy_imports import *
from deploy_utils import *

# 3‑D Rotary Positional Embedding (RoPE) utility
def apply_rope_3d(q, k, shape, theta=10_000.0):
    """
    Vectorised 3D RoPE for flattened voxel grids.
    Parameters
    ----------
    q, k : tensors shaped (B*heads, head_dim, T)
    shape: tuple (H, W, L) where H*W*L == T
    theta: base frequency

    Returns
    -------
    (q_rot, k_rot) with the same shapes as the inputs.
    """
    H, W, L = shape
    _, d, N = q.shape
    if N != H * W * L:  # shape mismatch – skip
        return q, k

    rope_dims = (d // 6) * 6          # largest multiple of 6 ≤ head_dim
    if rope_dims == 0:
        return q, k                   # head_dim too small → nothing to do

    d_each = rope_dims // 3           # channels per axis (must be even)
    # frequency spectrum
    freq = 1.0 / (theta ** (torch.arange(0, d_each // 2,
                                         device=q.device,
                                         dtype=q.dtype) / (d_each // 2)))

    # voxel coordinates in raster order
    idx = torch.arange(N, device=q.device)
    z = idx % L
    y = (idx // L) % W
    x = idx // (L * W)
    coords = torch.stack((x, y, z))         # (3, N)

    def _rotate(t):
        rot, tail = t[:, :rope_dims, :], t[:, rope_dims:, :]
        rot = rot.view(-1, 3, d_each, N)    # (B*H, 3, d_each, N)
        for axis in range(3):
            coord = coords[axis]
            sin = torch.sin(coord * freq[:, None])
            cos = torch.cos(coord * freq[:, None])
            t1 = rot[:, axis, 0::2, :]
            t2 = rot[:, axis, 1::2, :]
            rot[:, axis, 0::2, :] = t1 * cos - t2 * sin
            rot[:, axis, 1::2, :] = t1 * sin + t2 * cos
        return torch.cat((rot.view(-1, rope_dims, N), tail), dim=1)

    return _rotate(q), _rotate(k)


class RelativePosition(nn.Module):
    """ https://github.com/evelinehong/Transformer_Relative_Position_PyTorch/blob/master/relative_position.py """

    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(th.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        device = self.embeddings_table.device
        range_vec_q = th.arange(length_q, device=device)
        range_vec_k = th.arange(length_k, device=device)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = th.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = final_mat.long()
        embeddings = self.embeddings_table[final_mat]
        return embeddings


class CrossAttention(nn.Module):
    """
    Cross-attention mechanism that lets a feature map attend to a sequence of token embeddings.
    Accepts:
      - x:     (B, C, ...)
      - embed: (B, T, embed_dim), i.e. token-wise embeddings (T tokens).
    """
    def __init__(self, feature_dim, embed_dim, num_heads=4, use_qk_norm=True):
        super().__init__()
        self.num_heads = num_heads
        self.use_qk_norm = use_qk_norm
        
        # We often match "feature_dim" for the final 'to_out',
        # so set each head to (feature_dim // num_heads).
        # If you prefer to match embed_dim, you can adjust accordingly.
        self.head_dim = feature_dim // num_heads  
        hidden_dim = self.num_heads * self.head_dim
        
        # Normalize the image features before projecting to Q
        self.norm = normalization(feature_dim)
        
        # Linear projections for Q (image), K (text), and V (text)
        self.to_q = nn.Linear(feature_dim, hidden_dim, bias=False)
        self.to_k = nn.Linear(embed_dim,   hidden_dim, bias=False)
        self.to_v = nn.Linear(embed_dim,   hidden_dim, bias=False)
        
        # QK normalization layers
        if use_qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)
        
        # Output projection back to 'feature_dim' channels
        self.to_out = nn.Sequential(
            nn.Linear(hidden_dim, feature_dim),
            # nn.Dropout(0.1)
        )

    def forward(self, x, embed):
        """
        x:     (B, C, D, H, W) or (B, C, height, width, etc.) -> any spatial shape
        embed: (B, T, embed_dim) -> *token-wise* text embeddings
        Returns:
            A residual: out + x, shaped (B, C, D, H, W).
        """

        # Flatten x to (B, L, C) so we can treat each spatial location as a "token."
        B, C, *spatial = x.shape
        L = 1
        for s in spatial:
            L *= s  # total spatial positions

        # Apply normalization before Q-projection:
        x_norm = self.norm(x).reshape(B, C, L).permute(0, 2, 1)  # (B, L, C)

        # Project Q, K, V
        #  Q from image features: (B, L, feature_dim) -> (B, L, hidden_dim)
        #  K, V from text tokens: (B, T, embed_dim)   -> (B, T, hidden_dim)
        q = self.to_q(x_norm)
        k = self.to_k(embed)
        v = self.to_v(embed)

        # Reshape for multi-heads
        # q: (B, L, num_heads, head_dim) -> (B, num_heads, L, head_dim)
        # k,v: (B, T, num_heads, head_dim) -> (B, num_heads, T, head_dim)
        q = q.reshape(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Apply QK normalization if enabled
        if self.use_qk_norm:
            # Normalize along the head dimension (last dimension)
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        # attn: (B, num_heads, L, T)
        attn = torch.matmul(q, k.transpose(-1, -2)) * scale
        attn = F.softmax(attn, dim=-1)

        # out: (B, num_heads, L, head_dim)
        out = torch.matmul(attn, v)

        # Merge heads back together -> (B, L, hidden_dim)
        out = out.permute(0, 2, 1, 3).reshape(B, L, self.num_heads * self.head_dim)

        # Final linear and dropout -> (B, L, feature_dim)
        out = self.to_out(out)

        # Reshape back to (B, C, ...) and add the residual
        out = out.permute(0, 2, 1).reshape(B, C, *spatial)
        return out + x


# Borrowed from https://github.com/openai/guided-diffusion
from abc import abstractmethod

import math
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        x = super().forward(x)
        return x


class WrappedConv3d(nn.Conv3d):
    def forward(self, x):
        B, C, T, H, W, L = x.shape
        x = super().forward(x)
        return x


class WrappedAvgPool3d(nn.AvgPool3d):
    def forward(self, x):
        B, C, T, H, W, L = x.shape
        x = super().forward(x)
        return x


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def checkpoint(func, inputs, enable_checkpoint):
    """
    Wrapper that delegates to torch.utils.checkpoint.checkpoint
    with use_reentrant=False so RNG state is preserved.
    `params` is ignored because PyTorch handles parameter gradients
    automatically.
    """
    if not enable_checkpoint:
        return func(*inputs)
    # built-in API: checkpoint(run_function, *args, use_reentrant=False)
    return checkpoint_fn(func, *inputs, use_reentrant=False)

# def checkpoint(func, inputs, params, flag):
#     """
#     Evaluate a function without caching intermediate activations, allowing for
#     reduced memory at the expense of extra compute in the backward pass.

#     :param func: the function to evaluate.
#     :param inputs: the argument sequence to pass to `func`.
#     :param params: a sequence of parameters `func` depends on but does not
#                    explicitly take as arguments.
#     :param flag: if False, disable gradient checkpointing.
#     """
#     if flag:
#         args = tuple(inputs) + tuple(params)
#         return CheckpointFunction.apply(func, len(inputs), *args)
#     else:
#         return func(*inputs)


# class CheckpointFunction(th.autograd.Function):
#     @staticmethod
#     def forward(ctx, run_function, length, *args):
#         ctx.run_function = run_function
#         ctx.input_tensors = list(args[:length])
#         ctx.input_params = list(args[length:])
#         with th.no_grad():
#             output_tensors = ctx.run_function(*ctx.input_tensors)
#         return output_tensors

#     @staticmethod
#     def backward(ctx, *output_grads):
#         ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
#         with th.enable_grad():
#             # Fixes a bug where the first op in run_function modifies the
#             # Tensor storage in place, which is not allowed for detach()'d
#             # Tensors.
#             shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
#             output_tensors = ctx.run_function(*shallow_copies)
#         input_grads = th.autograd.grad(
#             output_tensors,
#             ctx.input_tensors + ctx.input_params,
#             output_grads,
#             allow_unused=True,
#         )
#         del ctx.input_tensors
#         del ctx.input_params
#         del output_tensors
#         return (None, None) + input_grads


class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads_channels: int,
        output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            th.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5
        )
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """

class TimestepBlockConditioned(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, t_emb, cond):
        """
        Apply the module to `x` given `t_emb` timestep embeddings and `cond`.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x
    

class TimestepEmbedSequentialConditioned(nn.Sequential, TimestepBlock, TimestepBlockConditioned):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, cond):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, TimestepBlockConditioned):
                x = layer(x, cond)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            out = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2, x.shape[5] * 2), mode="nearest"
            )
        else:
            out = F.interpolate(x, scale_factor=2, mode="nearest")
        if x.shape[-1] == x.shape[-2] == 3:
            # upsampling layer transform [3x3] to [6x6]. Manually paddding it to make [7x7]
            out = F.pad(out, (1, 0, 1, 0))
        if self.use_conv:
            out = self.conv(out)
        return out


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else 2
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)

def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


class CheckpointFunction(th.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with th.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with th.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = th.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads


class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads_channels: int,
        output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            th.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5
        )
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads, num_heads_channels)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        B, C, T = x.shape[0], x.shape[1], x.shape[2]
        assert x.shape[1] == self.channels
        if self.dims == 3:
            out = F.interpolate(
                x, (x.shape[2] * 2, x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            out = F.interpolate(x, scale_factor=2, mode="nearest")
        if x.shape[-1] == x.shape[-2] == 3:
            # upsampling layer transform [3x3] to [6x6]. Manually paddding it to make [7x7]
            out = F.pad(out, (1, 0, 1, 0))
        if self.use_conv:
            out = self.conv(out)
        return out


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else 2
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.use_checkpoint
        )

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=True,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = QKVAttention(self.num_heads, channels, 32)
        # self.attention = QKVAttentionLegacy(self.num_heads)
        self.space_relative_pos_k = RelativePosition(num_units=32, max_relative_position=64)
        self.space_relative_pos_v = RelativePosition(num_units=32, max_relative_position=64)
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), True)

    def _forward(self, x):
        B, C, H, W, L = x.shape
        x = x.reshape(B, C, -1)
        z = self.norm(x)
        
        len_q_space = H * W * L
        k_rp_space = self.space_relative_pos_k(len_q_space, len_q_space)  # [H*W*L, H*W*L, head_dim]
        v_rp_space = self.space_relative_pos_v(len_q_space, len_q_space)  # [H*W*L, H*W*L, head_dim]

        h = self.attention(z, rp=(k_rp_space, v_rp_space))
        h = self.proj_out(h)
        return (x + h).reshape(B, C, H, W, L)


class AttentionBlockConditioned(TimestepBlockConditioned):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=True,
        embed_dim=768,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.embed_dim = embed_dim
        
        self.norm = normalization(channels)
        self.attention = QKVAttention(self.num_heads, channels, 32)
        self.mlp = nn.Sequential(
            nn.Conv1d(channels, channels*4, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(channels*4, channels, kernel_size=1),
        )

        self.norm_text = nn.LayerNorm(embed_dim)
        self.cross_attention = CrossAttention(feature_dim=channels, embed_dim=embed_dim, num_heads=num_heads)
        self.mlp_text = nn.Sequential(
            nn.Conv1d(channels, channels*4, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(channels*4, channels, kernel_size=1),
        )

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x, cond):
        return checkpoint(self._forward, (x, cond), True)

    def _forward(self, x, cond):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        
        h = self.norm(x)
        h = self.attention(h)
        h = h + self.mlp(h)

        cond = self.norm_text(cond)
        z = self.cross_attention(h, cond)

        h = h + self.mlp_text(z)
        h = self.proj_out(h)
        
        return (x + h).reshape(b, c, *spatial)


class RelativePosition3D(nn.Module):
    """
    Computes factorized 3D relative position embeddings:
      - an embedding table for height offsets
      - an embedding table for width offsets
      - an embedding table for depth (L) offsets
    Then sums them together for each voxel pair.
    """

    def __init__(self, num_units, max_h, max_w, max_l):
        super().__init__()
        self.num_units = num_units
        self.max_h = max_h
        self.max_w = max_w
        self.max_l = max_l


        # Separate embedding tables for each dimension
        self.table_h = nn.Parameter(torch.Tensor(2 * max_h + 1, num_units))
        self.table_w = nn.Parameter(torch.Tensor(2 * max_w + 1, num_units))
        self.table_l = nn.Parameter(torch.Tensor(2 * max_l + 1, num_units))

        # Init
        nn.init.xavier_uniform_(self.table_h)
        nn.init.xavier_uniform_(self.table_w)
        nn.init.xavier_uniform_(self.table_l)

    def forward(self, H, W, L):
        """
        Returns a [H*W*L, H*W*L, num_units] tensor of factorized 3D relative position embeddings.
        Uses a more memory-efficient implementation that avoids materializing the full coordinate grid.
        """

        device = self.table_h.device
        
        # Generate coordinate tensors for each dimension
        h_coords = torch.arange(H, device=device)
        w_coords = torch.arange(W, device=device)
        l_coords = torch.arange(L, device=device)
        
        # Compute relative positions for each dimension separately
        h_rel = h_coords.unsqueeze(1) - h_coords.unsqueeze(0)  # [H, H]
        w_rel = w_coords.unsqueeze(1) - w_coords.unsqueeze(0)  # [W, W]
        l_rel = l_coords.unsqueeze(1) - l_coords.unsqueeze(0)  # [L, L]
        
        # Clamp and offset the relative positions
        h_rel = h_rel.clamp(-self.max_h, self.max_h) + self.max_h  # [H, H]
        w_rel = w_rel.clamp(-self.max_w, self.max_w) + self.max_w  # [W, W]
        l_rel = l_rel.clamp(-self.max_l, self.max_l) + self.max_l  # [L, L]
        
        # Look up embeddings for each dimension
        h_emb = self.table_h[h_rel]  # [H, H, num_units]
        w_emb = self.table_w[w_rel]  # [W, W, num_units]
        l_emb = self.table_l[l_rel]  # [L, L, num_units]
        
        # Combine the embeddings using outer products
        # First combine h and w
        hw_emb = (
            h_emb.unsqueeze(2).unsqueeze(3) +  # [H, H, 1, 1, num_units]
            w_emb.unsqueeze(0).unsqueeze(1)    # [1, 1, W, W, num_units]
        )  # [H, H, W, W, num_units]
        
        # Then combine with l
        hwl_emb = (
            hw_emb.view(H * W, H * W, 1, 1, self.num_units) +  # [HW, HW, 1, 1, num_units]
            l_emb.unsqueeze(0).unsqueeze(0)                    # [1, 1, L, L, num_units]
        )  # [HW, HW, L, L, num_units]
        
        # Reshape to final form [H*W*L, H*W*L, num_units]
        N = H * W * L
        return hwl_emb.view(N, N, self.num_units)

class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A multi-head attention module. 
    RoPE is applied when rope_shape is provided.
    Flash attention is enabled when not using relative position (RP).
    """
    def __init__(self, n_heads=4, in_channels=1, head_dim=32, use_qk_norm=True):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.use_qk_norm = use_qk_norm

        # 1x1 Conv => projects from in_channels -> (3 * n_heads * head_dim)
        self.qkv_proj = nn.Conv1d(in_channels, 3 * n_heads * head_dim, kernel_size=1)

        # QK normalization layers
        if use_qk_norm:
            self.q_norm = nn.LayerNorm(head_dim)
            self.k_norm = nn.LayerNorm(head_dim)

        self.out_proj = nn.Conv1d(n_heads * head_dim, in_channels, kernel_size=1)

    def forward(self, x, rp=None, rope_shape=None):
        """
        x:  (B, in_channels, T)
          B = batch size
          in_channels = arbitrary number of input channels
          T = sequence length
        rp (optional): tuple of (k_rp, v_rp) relative position embeddings
          k_rp: (T, T, head_dim)
          v_rp: (T, T, head_dim)
        rope_shape (optional): (H, W, L)
        Returns: (B, n_heads * head_dim, T)
        """
        bs, width, length = x.shape

        # 1) Project input -> Q, K, V
        qkv = self.qkv_proj(x)  # (B, 3*n_heads*head_dim, T)
        q, k, v = qkv.chunk(3, dim=1)  # each: (B, n_heads*head_dim, T)

        # 2) Scale by sqrt(sqrt(dim)) following DDPM-ish approach
        scale = 1.0 / math.sqrt(math.sqrt(self.head_dim))

        # 3) Reshape for multi-head: (B*n_heads, head_dim, T)
        q = (q * scale).reshape(bs * self.n_heads, self.head_dim, length)
        k = (k * scale).reshape(bs * self.n_heads, self.head_dim, length)
        v = v.reshape(bs * self.n_heads, self.head_dim, length)

        # 4) Apply QK normalization if enabled
        if self.use_qk_norm:
            # Transpose to (B*n_heads, T, head_dim) for LayerNorm
            q = q.transpose(-1, -2)  # (B*n_heads, T, head_dim)
            k = k.transpose(-1, -2)  # (B*n_heads, T, head_dim)
            
            # Apply LayerNorm
            q = self.q_norm(q)
            k = self.k_norm(k)
            
            # Transpose back to (B*n_heads, head_dim, T)
            q = q.transpose(-1, -2)
            k = k.transpose(-1, -2)

        # 5) Optional dense 3D RoPE
        if rope_shape is not None:
            q, k = apply_rope_3d(q, k, rope_shape)

        # 6) faster — use PyTorch Flash / SDP kernel when not using RP
        if rp is None:
            # reshape to (B, heads, T, D) expected by SDPA
            q_ = q.reshape(bs, self.n_heads, self.head_dim, length).permute(0, 1, 3, 2)
            k_ = k.reshape(bs, self.n_heads, self.head_dim, length).permute(0, 1, 3, 2)
            v_ = v.reshape(bs, self.n_heads, self.head_dim, length).permute(0, 1, 3, 2)

            # Call scaled‑dot‑product attention with Flash kernels enabled
            with torch.backends.cuda.sdp_kernel(enable_flash=True,
                                                enable_math=True,
                                                enable_mem_efficient=True):
                out = F.scaled_dot_product_attention(q_, k_, v_, dropout_p=0.0)

            # back to (B, heads*D, T)
            out = out.permute(0, 1, 3, 2).reshape(bs, self.n_heads * self.head_dim, length)
            return self.out_proj(out)

        else:
            # 7) Basic attention scores: q * k
            weight = torch.einsum("bdt,bds->bts", q, k)  # (B*n_heads, T, T)

            # 8) Add relative position bias for keys, if provided
            if rp is not None:
                k_rp, v_rp = rp
                # k_rp: (T, T, head_dim)
                # q:    (B*n_heads, head_dim, T)
                # We want (b, t, s) out of "bdt, tsd -> bts"
                rel_weight = torch.einsum("bdt,tsd->bts", q, k_rp)
                weight = weight + rel_weight

            # 9) Normalize
            weight = F.softmax(weight.float(), dim=-1).to(weight.dtype)

            # 10) Apply attention to values
            out = torch.einsum("bts,bds->bdt", weight, v)  # (B*n_heads, head_dim, T)

            # 11) Add relative position values, if provided
            if rp is not None:
                # v_rp: (T, T, head_dim)
                # we want (b, d, t) out of "bts, tsd -> bdt"
                rel_out = torch.einsum("bts,tsd->bdt", weight, v_rp)
                out = out + rel_out

            # 12) Reshape back to (B, n_heads*head_dim, T) and project
            out = out.reshape(bs, self.n_heads * self.head_dim, length)
            return self.out_proj(out)

    @staticmethod
    def count_flops(model, x, y):
        # You can adapt your existing flop-count method here if needed
        return 0


class AttentionBlockInpainting(TimestepBlockConditioned):
    """
    An attention block that allows spatial positions to attend to each other and includes text conditioning.
    """
    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=True,
        embed_dim=768
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint

        # Space attention components
        self.norm_space = normalization(channels)
        self.attn_space = QKVAttention(self.num_heads, channels, 32)
        self.mlp_space = nn.Sequential(
            nn.Conv1d(channels, channels*4, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(channels*4, channels, kernel_size=1),
        )

        # Condition cross-attention components
        cond_channels = 32
        self.norm_cond = nn.LayerNorm(cond_channels)
        self.cross_attention = CrossAttention(feature_dim=channels, embed_dim=cond_channels, num_heads=num_heads)
        self.mlp_cond = nn.Sequential(
            nn.Conv1d(channels, channels*4, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(channels*4, channels, kernel_size=1),
        )

        # Relative position embeddings for space
        self.space_relative_pos_k = RelativePosition3D(num_units=32, max_h=16, max_w=16, max_l=16)
        self.space_relative_pos_v = RelativePosition3D(num_units=32, max_h=16, max_w=16, max_l=16)

        # Final projection
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x, cond):
        return checkpoint(self._forward, (x, cond), self.use_checkpoint)

    def _forward(self, x, cond):
        """
        x: (B, C, H, W, L)
        cond: (B, T, D) text embeddings
        """
        B, C, H, W, L = x.shape
        h = x

        # 1) Space attention
        h_space = self.norm_space(h)
        h_space = rearrange(h_space, 'b c h w l -> b c (h w l)')
        
        # Get spatial relative position embeddings
        k_rp_space = self.space_relative_pos_k(H, W, L)
        v_rp_space = self.space_relative_pos_v(H, W, L)
        
        # Apply space attention with relative positions
        h_space_attn = self.attn_space(h_space, rp=(k_rp_space, v_rp_space))
        h_space_attn = h_space_attn + self.mlp_space(h_space_attn)
        h_space_attn = rearrange(h_space_attn, 'b c (h w l) -> b c h w l', h=H, w=W, l=L)
        h = h + h_space_attn

        # 2) Condition cross attention
        h_cond = rearrange(h, 'b c h w l -> b c (h w l)')
        cond = rearrange(cond, 'b c h w l -> b (h w l) c')
        cond = self.norm_cond(cond)
        z = self.cross_attention(h_cond, cond)
        z = z + self.mlp_cond(z)
        z = rearrange(z, 'b c (h w l) -> b c h w l', h=H, w=W, l=L)
        h = h + z

        # Final projection
        out_reshaped = rearrange(h, 'b c h w l -> b c (h w l)')
        out_proj = self.proj_out(out_reshaped)
        out_proj = rearrange(out_proj, 'b c (h w l) -> b c h w l', h=H, w=W, l=L)
        return x + out_proj 


class UNetModel3DInpainting(nn.Module):
    """
    :param in_channels: channels in the input Tensor.
    :param emb_dim: base dimension of timestep embedding.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        time_emb_factor=4,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * time_emb_factor
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequentialConditioned(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlockInpainting(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequentialConditioned(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequentialConditioned(
                        *[ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        ),
                    ]
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch
        self.middle_block = TimestepEmbedSequentialConditioned(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlockInpainting(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlockInpainting(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequentialConditioned(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )

    def forward(self, x, timesteps, y):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        h = x.type(self.dtype)
        for i, module in enumerate(self.input_blocks):
            h = module(h, emb, y)
            hs.append(h)
        h = self.middle_block(h, emb, y)
        for i, module in enumerate(self.output_blocks):
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, y)
        h = h.type(x.dtype)
        return self.out(h)


def UNet3DInpaintingL(
    image_size,
    in_channels=3,
    out_channels=3,
    base_width=192,
    num_classes=None,
):
    if image_size == 128:
        channel_mult = (1, 1, 2, 3, 4)
    elif image_size == 64:
        channel_mult = (1, 2, 3, 4)
    elif image_size == 32:
        channel_mult = (1, 2, 2, 2)
    elif image_size == 28:
        channel_mult = (1, 2, 2, 2)
    elif image_size in [16, 8]:
        channel_mult = (1, 2, 3, 4)
    else:
        raise ValueError(f"unsupported image size: {image_size}")

    attention_ds = []
    if image_size == 28:
        attention_resolutions = "28,14,7"
    elif image_size == 8:
        attention_resolutions = "8,4,2"
    elif image_size == 16:
        attention_resolutions = "16,8,4,2"
    else:
        attention_resolutions = "32,16,8"
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return UNetModel3DInpainting(
        image_size=image_size,
        in_channels=in_channels,
        model_channels=base_width,
        out_channels=out_channels,
        num_res_blocks=3,
        attention_resolutions=tuple(attention_ds),
        dropout=0.0,
        channel_mult=channel_mult,
        num_classes=num_classes,
        use_checkpoint=True,
        use_fp16=False,
        num_heads=16,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_new_attention_order=True,
        dims=3,
    )



# ─────────────────────────────────────────────────────────────────────────────
# 1 · Core sparse blocks
# ─────────────────────────────────────────────────────────────────────────────
class SparseRes(nn.Module):
    """
    Conv3x3 stride s → SiLU with residual.
    Adds 1x1 proj when stride != 1 or c_in != c_out.
    """
    def __init__(self, c_in, c_out, kernel_size=3, downsample=False):
        super().__init__()
        self.downsample = downsample
        self.c_in = c_in
        self.c_out = c_out
        self.conv1 = fvnn.SparseConv3d(c_in, c_out, 3)
        self.conv2 = fvnn.SparseConv3d(c_out, c_out, 3)
        self.conv3 = fvnn.SparseConv3d(c_out, c_out, 1)
        self.act  = fvnn.ReLU()
        self.need_proj = (c_in != c_out)
        # if self.need_proj:
            # self.proj = fvnn.SparseConv3d(c_in, c_out, 1, 1)
        if downsample:
            # Try the simple convolution-based pooling first
            self.downsample_layer = SparseAvgPool3dConv(pool_factor=2, channels=c_out)

    def forward(self, x: fvnn.VDBTensor):
        h = self.act(self.conv1(x))
        h = self.act(self.conv2(h))
        h = self.act(self.conv3(h))
        if self.downsample:
            h = self.downsample_layer(h)
        return h


class SparseAttn(nn.Module):
    """
    X-CUBE latent self-attention (fully sparse) with the *batch-slice*
    indexing bug corrected.
    """
    def __init__(self, channels, num_heads=4, use_checkpoint=False):
        super().__init__()
        self.num_heads = num_heads
        self.channels = channels
        self.use_checkpoint = use_checkpoint
        self.norm = fvnn.GroupNorm(32, channels)
        self.qkv = fvnn.Linear(channels, channels * 3)
        self.proj_out = fvnn.Linear(channels, channels)
        self.groupnorm = fvnn.GroupNorm(8, channels)
        
    def _attention(self, qkv: torch.Tensor):
        # conduct attention for each batch
        length, width = qkv.shape
        assert width % (3 * self.num_heads) == 0
        ch = width // (3 * self.num_heads)
        qkv = qkv.reshape(length, self.num_heads, 3 * ch).unsqueeze(0)
        qkv = qkv.permute(0, 2, 1, 3) # (1, num_heads, length, 3 * ch)
        q, k, v = qkv.chunk(3, dim=-1) # (1, num_heads, length, ch)
        with torch.backends.cuda.sdp_kernel(enable_math=False):
            values = F.scaled_dot_product_attention(q, k, v)[0] # (1, num_heads, length, ch)
        values = values.permute(1, 0, 2) # (length, num_heads, ch)
        values = values.reshape(length, -1)
        return values
        
    def attention(self, qkv: fvnn.VDBTensor):
        values = []
        for batch_idx in range(qkv.grid.grid_count):
            feats = qkv.data[batch_idx].jdata
            if feats.numel() == 0:
                empty = torch.empty((0, self.channels), dtype=feats.dtype, device=feats.device)
                values.append(empty)
            else:
                values.append(self._attention(feats))         
        # Concatenate all values and use grid.jagged_like to maintain proper batch structure
        if len(values) == 1:
            concat_values = values[0]
        else:
            concat_values = torch.cat(values, dim=0)
        return qkv.grid.jagged_like(concat_values)

    def forward(self, x: fvnn.VDBTensor):
        return self._forward(x) # !: return None for feat_depth

    def _forward(self, x: fvnn.VDBTensor):
        x_norm = self.groupnorm(x)
        qkv = self.qkv(x_norm)
        
        original_grid = x.grid
        original_kmap = x.kmap
        feature = self.attention(qkv)
        feature = fvnn.VDBTensor(original_grid, feature, original_kmap)
        feature = self.proj_out(feature)
        return feature + x


class BinarySTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x[..., 0] > x[..., 1]
    @staticmethod
    def backward(ctx, g):
        return g



# ─────────────────────────────────────────────────────────────────────────────
# 2 · 2-channel structure head  (mask predictor)
# ─────────────────────────────────────────────────────────────────────────────
class StructHead(nn.Module):
    """Predict [ exist , empty ] logits for each voxel."""
    def __init__(self, c):
        super().__init__()
        self.block = nn.Sequential(
            fvnn.GroupNorm(32, c),
            fvnn.SiLU(),
            SparseRes(c, c),
            fvnn.SparseConv3d(c, 2, 1))

    def forward(self, x: fvnn.VDBTensor):
        return self.block(x)        # logits (2-ch)

def mask_from_logits(logits: fvnn.VDBTensor, min_voxels: int = 16) -> tuple[fvnn.VDBTensor, fvnn.VDBTensor]:
    exist_prob_tensor = logits.data.jdata[:,0] - logits.data.jdata[:,1] > 0.0
    # make sure there is at least 16 voxels in the mask
    min_voxels = min(min_voxels, logits.data.jdata.shape[0])
    if exist_prob_tensor.sum() < min_voxels:
        topk = (logits.data.jdata[:,0] - logits.data.jdata[:,1]).topk(min_voxels, dim=0)
        exist_prob_tensor[topk.indices] = True

    exist_prob = logits.grid.jagged_like(exist_prob_tensor)

    pred_mask_diff = logits.grid.jagged_like(logits.data.jdata[:,0:1] - logits.data.jdata[:,1:2])
    pred_mask_vdb = fvnn.VDBTensor(logits.grid, pred_mask_diff, logits.kmap)
    
    return exist_prob, pred_mask_vdb

class HierDecoder(nn.Module):
    """
    Args
    ----
    chs            list[int]   channel width at each level, coarse→fine
    up_factor      list[int]   (dz,dy,dx) per level   (usually [2,2,2])
    out_heads      dict{name →  nn.Module producing fvnn.VDBTensor}
    """
    def __init__(self, chs, up_factor):
        super().__init__()
        self.levels = len(chs) - 1
        self.up     = nn.ModuleList([fvnn.UpsamplingNearest(f)
                                     for f in up_factor])
        self.res    = nn.ModuleList([SparseRes(chs[i], chs[i+1])
                                      for i in range(self.levels)])
        self.struct = nn.ModuleList([StructHead(chs[i+1])
                                      for i in range(self.levels)])
        self.final_struct = StructHead(chs[-1])
        self.prune = fvnn.UpsamplingNearest(1)
    # -----
    def forward(self, x: fvnn.VDBTensor):
        mask_pyramid = []
        mask = None 
        for lvl in range(self.levels):
            if lvl > 0 and mask is not None:  
                x = self.up[lvl-1](x, mask)
            x = self.res[lvl](x)
            logits = self.struct[lvl](x)
            mask, mask_logits_diff = mask_from_logits(logits)
            mask_pyramid.append(mask_logits_diff)
            assert x.jdata.shape[0] == mask.jdata.shape[0]
        x = self.prune(x, mask)
        return x, mask_pyramid


class GeometryHead(nn.Module):
    def __init__(self, base_ch, geometry_channels):
        super().__init__()
        self.res_block = SparseRes(base_ch, base_ch)
        self.conv_active = fvnn.SparseConv3d(base_ch, geometry_channels, 2)
        self.conv_inactive = fvnn.SparseConv3d(base_ch, geometry_channels, 2)

    def forward(self, x: fvnn.VDBTensor):
        x = self.res_block(x)
        active = self.conv_active(x)
        inactive = self.conv_inactive(x)
        y = active - inactive
        return y
    
class ColorHead(nn.Module):
    def __init__(self, base_ch, color_channels):
        super().__init__()
        # self.res_block = SparseRes(base_ch, base_ch)
        self.conv = fvnn.SparseConv3d(base_ch, color_channels, 1)

    def forward(self, x: fvnn.VDBTensor):
        x = self.conv(x)
        x.data.jdata = torch.clamp(x.data.jdata, 0.0, 1.0)
        return x

class SparseStructureDecoder(nn.Module):
    def __init__(self, color_channels = 3,
                 geometry_channels = 1,
                 base_ch     = 32,
                 levels      = 3,
                 latent_ch   = 16,
                 latent_res  = 16,
                 use_attn    = True):
        super().__init__()

        self.levels = levels
        ch = base_ch * (2 ** levels)
        self.use_attn = use_attn


        # at latent level, we need another geom head to get rid of the unfilled voxels
        # self.latent_color_head = ColorHead(ch, color_channels)

        # Calculate channel counts for each level, goes from base_ch to base_ch * 2^levels in reverse order

        self.max_voxels = [(latent_res ** 3) * (4 ** (i + 1)) for i in range(levels)]
        self.res_blocks = nn.ModuleList([])
        self.geom_heads = nn.ModuleList([])
        self.color_heads = nn.ModuleList([])
        for i in range(levels):
            self.res_blocks.append(torch.nn.Sequential(SparseRes(ch, ch // 2), fvnn.GroupNorm(1, ch // 2), fvnn.SiLU(), SparseRes(ch // 2, ch // 2)))
            ch = ch // 2
            self.geom_heads.append(GeometryHead(ch, geometry_channels))
            self.color_heads.append(ColorHead(ch, color_channels))
        self.upsampling_blocks = nn.ModuleList([fvnn.UpsamplingNearest([2,2,2]) for _ in range(levels)])

    @staticmethod
    def apply_mask(x: fvnn.VDBTensor, mask_vdb: fvnn.VDBTensor, min_voxels: int = 16, max_voxels: int = 131072):
        # in practice max_voxels start at 2048 and should be grow by a factor of 4 each level

        """Apply mask but ensure minimum number of voxels remain and cap maximum to prevent OOM"""
        if mask_vdb is None:
            # mask_values = torch.ones_like(x.data.jdata).flatten()
            # mask = (mask_values > 0.0)
            return x
        elif mask_vdb.data.jdata.dtype != torch.bool:
            mask_values = mask_vdb.data.jdata.flatten()
            mask = (mask_values > 0.0)
        else:
            mask = mask_vdb.data.jdata.flatten()
            mask_values = mask_vdb.data.jdata.flatten().float()
        
        # Get batch indices for each voxel
        batch_indices = mask_vdb.grid.jidx
        batch_size = batch_indices.max().item() + 1
        
        # Process each batch item
        final_mask = torch.zeros_like(mask)
        
        for b in range(batch_size):
            # Get mask for this batch item
            batch_mask = (batch_indices == b)
            batch_voxel_mask = mask[batch_mask]
            batch_values = mask_values[batch_mask]
            
            # Apply min_voxels constraint
            if batch_voxel_mask.sum() < min_voxels and batch_mask.sum() > 0:
                # Keep top-k voxels for this batch item
                k = min(min_voxels, batch_mask.sum())
                _, topk_indices = torch.topk(batch_values, k, largest=True)
                batch_voxel_mask.zero_()
                batch_voxel_mask[topk_indices] = True
            
            # Apply max_voxels constraint
            if batch_voxel_mask.sum() > max_voxels:
                # Keep only top max_voxels for this batch item
                active_indices = torch.where(batch_voxel_mask)[0]
                active_values = batch_values[active_indices]
                _, top_indices = torch.topk(active_values, max_voxels, largest=True)
                keep_indices = active_indices[top_indices]
                
                batch_voxel_mask.zero_()
                batch_voxel_mask[keep_indices] = True
            
            # Update final mask
            final_mask[batch_mask] = batch_voxel_mask
        
        return fvnn.UpsamplingNearest(1)(x, final_mask)

    def forward(self, x: fvnn.VDBTensor):
        masks = []
        colors = []

        # latent level
        # color_latent = self.latent_color_head(x)
        # colors.append(color_latent)

        for i in range(self.levels):
            x = self.upsampling_blocks[i](x)
            x = self.res_blocks[i](x)
            mask = self.geom_heads[i](x)
            masks.append(mask)
            x = self.apply_mask(x, mask if i < self.levels - 1 else None, max_voxels=self.max_voxels[i])
            color = self.color_heads[i](x)
            colors.append(color)
            color.grid.set_global_voxel_size(1.0)
        final_color_masked = self.apply_mask(colors[-1], masks[-1], max_voxels=self.max_voxels[-1])
        final_color_masked.grid.set_global_voxel_size(1.0)
        return x, masks, colors, final_color_masked


class SparseColorVAE(nn.Module):
    def __init__(self, color_channels = 3,
                 geometry_channels = 1,
                 base_ch     = 16,
                 levels      = 3,        
                 latent_ch   = 8,
                 use_attn    = True):
        super().__init__()

        in_ch = color_channels
        self.levels = levels

        self.encoder = nn.Sequential(
            SparseRes(in_ch, base_ch),
        )

        self.pre_kl = nn.Sequential(
            SparseRes(base_ch, base_ch),
        )

        self.post_kl = nn.Sequential(
            fvnn.SparseConv3d(latent_ch, base_ch, 1),
            fvnn.ReLU()
        )

        self.decoder = nn.Sequential(
            SparseRes(base_ch, 2*base_ch),
            SparseRes(2*base_ch, base_ch),
        )

        self.down_levels = nn.ModuleList([])
        for _ in range(levels):
            self.down_levels.append(SparseRes(base_ch, base_ch, downsample=True))
            # self.down_levels.append(SparseRes(base_ch, base_ch, downsample=False))

        self.color_heads = nn.ModuleList([])
        for _ in range(levels):
            self.color_heads.append(fvnn.SparseConv3d(base_ch, color_channels, 1))
        # self.geom_head = fvnn.SparseConv3d(base_ch, geometry_channels, 1)

        # self.color_head = fvnn.SparseConv3d(base_ch, color_channels, 1)

        self.mu  = fvnn.SparseConv3d(base_ch, latent_ch, 1)
        self.log = fvnn.SparseConv3d(base_ch, latent_ch, 1)
        
    @staticmethod
    def reparametrize(mu: fvnn.VDBTensor, log: fvnn.VDBTensor):
        std = (0.5*log.data.jdata).exp()
        eps = torch.randn_like(std)
        return fvnn.VDBTensor(mu.grid, mu.data + eps*std)
    
    @staticmethod
    def upsample(x: fvnn.VDBTensor, factor: int = 8):
        return fvnn.UpsamplingNearest([factor, factor, factor])(x)

    def encode(self, x: fvnn.VDBTensor):
        # encoder
        h = x
        for block in self.encoder: 
            h = block(h)

        # bottleneck
        h = self.pre_kl(h)
        mu, log = self.mu(h), self.log(h)
        return mu, log
    
    def decode(self, z: fvnn.VDBTensor):
        h = self.post_kl(z)
        h = self.decoder(h)
        colors = []
        for i in range(self.levels):
            col  = self.color_heads[i](h)
            # col = self.color_head(h)
            # col.data.jdata = torch.clamp(col.data.jdata, 0.0, 1.0)
            # col.data.jdata = torch.sigmoid(col.data.jdata)
            colors.append(col)
            # h.data.jdata[: -3:] += col.data.jdata
            if i < self.levels - 1:
                h = self.down_levels[i](h)
        # geom = self.geom_head(h)
        # geom = self.sigmoid(geom)
        geom = None
        return {"color": colors[0], "color_levels": colors[::-1], "geometry": geom}

    # ------------------------------------------------------------
    def forward(self, x: fvnn.VDBTensor):
        mu, log = self.encode(x)
        z = self.reparametrize(mu, log)
        d = self.decode(z)
        
        return {**d, "mu": mu, "logvar": log, "z": z}
    
    @staticmethod
    def merge_features(out: dict) -> fvnn.VDBTensor:
        """
        Combine color-head and geometry-head outputs into a single feature
        tensor:  RGB (3)  +  alpha (1)  +  the geometry channels (≥1).

        Returns
        -------
        fvnn.VDBTensor
            grid : same as geometry head
            feature : JaggedTensor with concatenated channels
        """
        col_vdb : fvnn.VDBTensor = out["color"]            # RGB  -- (N,3)

        col_feat  = col_vdb.data.jdata                 # (N, 3)  
        grid = col_vdb.grid
        feat  = col_feat

        # wrap back into JaggedTensor aligned with the grid
        feat_jt = grid.jagged_like(feat)                   # preserves jidx
        return fvnn.VDBTensor(grid, feat_jt)
    
    def inference(self, z: fvnn.VDBTensor):
        dict = self.decode(z)
        return self.merge_features(dict)


class SparseAttn3D(nn.Module):
    """
    Improved 3D sparse self-attention with 3D RoPE positional encoding.
    Uses partial RoPE application.
    """
    def __init__(self, channels, num_heads=8, use_checkpoint=False, rope_theta=10000.0):
        super().__init__()
        self.num_heads = num_heads
        self.channels = channels
        self.head_dim = channels // num_heads
        self.use_checkpoint = use_checkpoint
        self.rope_theta = rope_theta
        
        assert channels % num_heads == 0, f"channels {channels} must be divisible by num_heads {num_heads}"
        
        self.norm = fvnn.GroupNorm(32, channels)
        self.qkv = fvnn.Linear(channels, channels * 3)
        self.proj_out = fvnn.Linear(channels, channels)
        
        # Pre-compute RoPE frequencies for 3D
        self._register_rope_frequencies()
        
    def _register_rope_frequencies(self):
        """Pre-compute RoPE frequency tensor for 3D coordinates using partial application"""
        # Use the largest multiple of 6 that fits in head_dim for 3D RoPE
        self.rope_dims = (self.head_dim // 6) * 6
        
        if self.rope_dims > 0:
            # Each spatial dimension gets rope_dims//3 dimensions, and each needs pairs
            rope_dim_per_axis = self.rope_dims // 3
            assert rope_dim_per_axis % 2 == 0, "rope_dim_per_axis must be even for sin/cos pairs"
            
            # Create frequency tensor: [rope_dim_per_axis//2] for each of x, y, z
            freqs = 1.0 / (self.rope_theta ** (torch.arange(0, rope_dim_per_axis, 2).float() / rope_dim_per_axis))
            self.register_buffer('rope_freqs', freqs, persistent=False)
            self.rope_dim_per_axis = rope_dim_per_axis
        else:
            # If head_dim < 6, skip RoPE entirely
            self.register_buffer('rope_freqs', torch.empty(0), persistent=False)
            self.rope_dim_per_axis = 0
    
    def _apply_rope_3d(self, x: torch.Tensor, coords: torch.Tensor):
        """
        Apply 3D RoPE to query/key tensors using partial application
        
        Args:
            x: tensor of shape [length, num_heads, head_dim]
            coords: 3D coordinates of shape [length, 3] (x, y, z)
        
        Returns:
            x with partial RoPE applied
        """
        if self.rope_dims == 0:
            return x  # Skip RoPE if head_dim too small
            
        length, num_heads, head_dim = x.shape
        
        # Split into RoPE and non-RoPE parts
        x_rope = x[:, :, :self.rope_dims]  # [length, heads, rope_dims]
        x_unchanged = x[:, :, self.rope_dims:]  # [length, heads, remaining_dims]
        
        # Reshape for 3D processing
        x_rope = x_rope.view(length, num_heads, 3, self.rope_dim_per_axis)  # [length, heads, 3, rope_dim_per_axis]
        
        # Apply RoPE to each spatial dimension
        result_parts = []
        for dim in range(3):
            x_dim = x_rope[:, :, dim, :]  # [length, heads, rope_dim_per_axis]
            coord_dim = coords[:, dim:dim+1]  # [length, 1]
            
            # Compute position encodings
            pos_enc = coord_dim * self.rope_freqs.unsqueeze(0)  # [length, rope_dim_per_axis//2]
            
            # Split into pairs for rotation
            x_pairs = x_dim.view(length, num_heads, self.rope_dim_per_axis // 2, 2)  # [length, heads, pairs, 2]
            
            # Apply rotation
            cos_pos = torch.cos(pos_enc).unsqueeze(1).unsqueeze(-1)  # [length, 1, pairs, 1]
            sin_pos = torch.sin(pos_enc).unsqueeze(1).unsqueeze(-1)  # [length, 1, pairs, 1]
            
            x_rotated = torch.stack([
                x_pairs[..., 0] * cos_pos.squeeze(-1) - x_pairs[..., 1] * sin_pos.squeeze(-1),
                x_pairs[..., 0] * sin_pos.squeeze(-1) + x_pairs[..., 1] * cos_pos.squeeze(-1)
            ], dim=-1)
            
            result_parts.append(x_rotated.view(length, num_heads, self.rope_dim_per_axis))
        
        # Concatenate RoPE-applied dimensions
        x_rope_applied = torch.cat(result_parts, dim=-1)  # [length, heads, rope_dims]
        
        # Concatenate with unchanged dimensions
        return torch.cat([x_rope_applied, x_unchanged], dim=-1)
    
    def _attention(self, qkv: torch.Tensor, coords: torch.Tensor):
        """
        Conduct attention for a single batch with 3D RoPE
        
        Args:
            qkv: tensor of shape [length, 3 * channels]
            coords: 3D coordinates of shape [length, 3]
        """
        length, width = qkv.shape
        assert width == 3 * self.channels
        
        # Reshape and split QKV
        qkv = qkv.reshape(length, self.num_heads, 3 * self.head_dim)
        q, k, v = qkv.chunk(3, dim=-1)  # Each: [length, num_heads, head_dim]
        
        # Apply 3D RoPE to queries and keys (partial application)
        q = self._apply_rope_3d(q, coords)
        k = self._apply_rope_3d(k, coords)
        
        # Reshape for attention: [1, num_heads, length, head_dim]
        q = q.permute(1, 0, 2).unsqueeze(0)
        k = k.permute(1, 0, 2).unsqueeze(0)
        v = v.permute(1, 0, 2).unsqueeze(0)
        
        # Scaled dot-product attention
        with torch.backends.cuda.sdp_kernel(enable_math=False):
            values = F.scaled_dot_product_attention(q, k, v)[0]  # [1, num_heads, length, head_dim]
        
        # Reshape back: [length, channels]
        values = values.permute(1, 0, 2).reshape(length, -1)
        return values
        
    def attention(self, qkv: fvnn.VDBTensor):
        """Apply attention across all batches in the VDBTensor"""
        values = []
        
        for batch_idx in range(qkv.grid.grid_count):
            feats = qkv.data[batch_idx].jdata
            
            if feats.numel() == 0:
                empty = torch.empty((0, self.channels), dtype=feats.dtype, device=feats.device)
                values.append(empty)
            else:
                # Get 3D coordinates for this batch
                coords = qkv.grid.ijk[batch_idx].jdata.float()  # [length, 3]
                values.append(self._attention(feats, coords))
                
        # Concatenate all values and use grid.jagged_like to maintain proper batch structure
        if len(values) == 1:
            concat_values = values[0]
        else:
            concat_values = torch.cat(values, dim=0)
        return qkv.grid.jagged_like(concat_values)

    def forward(self, x: fvnn.VDBTensor):
        # Normalize input
        x_norm = self.norm(x)
        
        # Compute QKV
        qkv = self.qkv(x_norm)
        
        # Apply attention with 3D RoPE
        feature = self.attention(qkv)
        feature = fvnn.VDBTensor(x.grid, feature, x.kmap)
        
        # Output projection
        feature = self.proj_out(feature)
        # Residual connection
        return feature + x


class SparseMLP(nn.Module):
    """
    Sparse MLP block for transformer
    """
    def __init__(self, channels, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        hidden_channels = int(channels * mlp_ratio)
        
        self.norm = fvnn.GroupNorm(32, channels)
        self.fc1 = fvnn.Linear(channels, hidden_channels)
        self.act = fvnn.SiLU()
        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc2 = fvnn.Linear(hidden_channels, channels)
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: fvnn.VDBTensor):
        # Normalize
        x_norm = self.norm(x)
        
        # MLP
        h = self.fc1(x_norm)
        h = self.act(h)
        
        # Apply dropout to the data tensor
        if hasattr(self.dropout1, 'p') and self.dropout1.p > 0:
            h_data = self.dropout1(h.data.jdata)
            h = fvnn.VDBTensor(h.grid, h.grid.jagged_like(h_data), h.kmap)
        
        h = self.fc2(h)
        
        # Apply dropout to the data tensor
        if hasattr(self.dropout2, 'p') and self.dropout2.p > 0:
            h_data = self.dropout2(h.data.jdata)
            h = fvnn.VDBTensor(h.grid, h.grid.jagged_like(h_data), h.kmap)
        
        # Residual connection
        return h + x


class SparseTransformerBlock(nn.Module):
    """
    Complete sparse transformer block with 3D RoPE attention and MLP
    
    Args:
        channels: Number of input/output channels
        num_heads: Number of attention heads
        mlp_ratio: Ratio of MLP hidden size to input size
        dropout: Dropout probability
        rope_theta: RoPE frequency base
        use_checkpoint: Whether to use gradient checkpointing
    """
    def __init__(self, 
                 channels, 
                 num_heads=8, 
                 mlp_ratio=4.0, 
                 dropout=0.0,
                 rope_theta=10000.0,
                 use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        
        self.attn = SparseAttn3D(
            channels=channels,
            num_heads=num_heads,
            use_checkpoint=use_checkpoint,
            rope_theta=rope_theta
        )
        
        self.mlp = SparseMLP(
            channels=channels,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )
    
    def _forward(self, x: fvnn.VDBTensor):
        # Self-attention with residual
        x = self.attn(x)
        
        # MLP with residual  
        x = self.mlp(x)
        
        return x
    
    def forward(self, x: fvnn.VDBTensor):
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)


class SparseAvgPool3dConv(nn.Module):
    """
    Simple sparse average pooling using strided convolution.
    Uses two passes: one to sum values, one to count valid voxels.
    """
    def __init__(self, pool_factor=2, channels=None):
        super().__init__()
        self.pool_factor = pool_factor
        self.kernel_size = pool_factor
        self.channels = channels
        
        # Only create conv layers if channels is specified
        if channels is not None:
            self.sum_conv = fvnn.SparseConv3d(
                channels, channels, self.kernel_size, stride=self.pool_factor
            )
            self.count_conv = fvnn.SparseConv3d(
                1, 1, self.kernel_size, stride=self.pool_factor
            )
            
            # Initialize weights
            with torch.no_grad():
                # Initialize sum_conv weights to identity mapping
                self.sum_conv.weight.data.zero_()
                for i in range(channels):
                    self.sum_conv.weight.data[i, i] = 1.0
                if self.sum_conv.bias is not None:
                    self.sum_conv.bias.data.zero_()
                
                # Initialize count_conv weights to sum
                self.count_conv.weight.data.fill_(1.0)
                if self.count_conv.bias is not None:
                    self.count_conv.bias.data.zero_()
            
            # Freeze parameters
            self.sum_conv.weight.requires_grad_(False)
            self.count_conv.weight.requires_grad_(False)
            if self.sum_conv.bias is not None:
                self.sum_conv.bias.requires_grad_(False)
            if self.count_conv.bias is not None:
                self.count_conv.bias.requires_grad_(False)
        else:
            self.sum_conv = None
            self.count_conv = None

    def forward(self, x: fvnn.VDBTensor):
        # Get number of channels from input
        C = x.data.jdata.shape[-1] if x.data.jdata.numel() > 0 else 1
        
        # Create conv layers if not already created (for backward compatibility)
        if self.sum_conv is None:
            self.channels = C
            device = x.data.jdata.device
            
            self.sum_conv = fvnn.SparseConv3d(
                C, C, self.kernel_size, stride=self.pool_factor
            ).to(device)
            self.count_conv = fvnn.SparseConv3d(
                1, 1, self.kernel_size, stride=self.pool_factor
            ).to(device)
            
            # Initialize weights
            with torch.no_grad():
                self.sum_conv.weight.data.zero_()
                for i in range(C):
                    self.sum_conv.weight.data[i, i] = 1.0
                if self.sum_conv.bias is not None:
                    self.sum_conv.bias.data.zero_()
                
                self.count_conv.weight.data.fill_(1.0)
                if self.count_conv.bias is not None:
                    self.count_conv.bias.data.zero_()
            
            # Freeze parameters
            self.sum_conv.weight.requires_grad_(False)
            self.count_conv.weight.requires_grad_(False)
            if self.sum_conv.bias is not None:
                self.sum_conv.bias.requires_grad_(False)
            if self.count_conv.bias is not None:
                self.count_conv.bias.requires_grad_(False)
        
        # Sum the feature values
        feature_sum = self.sum_conv(x)
        
        # Create a ones tensor for counting
        ones_data = torch.ones(x.data.jdata.shape[0], 1, dtype=x.data.jdata.dtype, device=x.data.jdata.device)
        ones_jt = x.grid.jagged_like(ones_data)
        ones_vdb = fvnn.VDBTensor(x.grid, ones_jt, x.kmap)
        
        # Count valid voxels
        voxel_count = self.count_conv(ones_vdb)
        
        # Compute average efficiently using broadcasting
        # Clamp count to avoid division by zero
        count_clamped = torch.clamp(voxel_count.data.jdata, min=1e-8)
        
        # Broadcast count to match feature dimensions and compute average
        count_broadcast = count_clamped.expand(-1, feature_sum.data.jdata.shape[1])
        avg_data = feature_sum.data.jdata / count_broadcast
        
        # Create result with same grid structure
        avg_jt = feature_sum.grid.jagged_like(avg_data)
        return fvnn.VDBTensor(feature_sum.grid, avg_jt, feature_sum.kmap)

    def state_dict(self, *args, **kwargs):
        """Exclude our conv layers from state dict"""
        state = super().state_dict(*args, **kwargs)
        return {k: v for k, v in state.items() 
                if '.sum_conv.' not in k 
                and '.count_conv.' not in k}

    def load_state_dict(self, state_dict, strict=True):
        """Prevent loading of our conv layer states"""
        filtered = {k: v for k, v in state_dict.items() 
                   if '.sum_conv.' not in k 
                   and '.count_conv.' not in k}
        return super().load_state_dict(filtered, strict=False)


class SparseFineVAE(nn.Module):
    def __init__(self, color_channels = 3,
                 geometry_channels = 1,
                 base_ch     = 32,
                 levels      = 2,        
                 latent_ch   = 16,
                 use_attn    = True):
        super().__init__()

        in_ch = color_channels

        self.encoder_blocks = nn.ModuleList([])
        self.encoder_blocks.append(SparseRes(in_ch, base_ch))
        channel_multiplier = 1
        ch = base_ch
        for _ in range(levels):
            self.encoder_blocks.append(nn.Sequential(
                SparseRes(ch, ch*2, downsample=True),
            ))
            ch *= 2
            channel_multiplier *= 2
        # scale with levels
        # self.encoder = nn.Sequential(
        #     SparseRes(in_ch, base_ch),
        #     SparseRes(base_ch, base_ch*2, downsample=True),
        #     SparseRes(base_ch*2, base_ch*4, downsample=True),
        #     # SparseRes(base_ch*4, base_ch*8, downsample=True),
        # )

        self.pre_kl = nn.Sequential(
            # SparseRes(base_ch*channel_multiplier, base_ch*6),
            # SparseTransformerBlock(base_ch*6, 8),
            # SparseRes(base_ch*6, base_ch),
            # SparseAttn(base_ch*channel_multiplier, 8),
            SparseRes(base_ch*channel_multiplier, base_ch*channel_multiplier),
            # SparseAttn(base_ch*channel_multiplier, 8),
            fvnn.GroupNorm(32, base_ch*channel_multiplier),
            fvnn.SiLU(),
            SparseRes(base_ch*channel_multiplier, base_ch),
        )

        self.post_kl = nn.Sequential(
            # SparseRes(latent_ch, base_ch*6),
            # SparseTransformerBlock(base_ch*6, 8),
            # SparseRes(base_ch*6, base_ch*channel_multiplier),
            SparseRes(latent_ch, base_ch*channel_multiplier),
            # SparseAttn(base_ch*channel_multiplier, 8),
            fvnn.GroupNorm(32, base_ch*channel_multiplier),
            fvnn.SiLU(),
            SparseRes(base_ch*channel_multiplier, base_ch*channel_multiplier),
            # SparseAttn(base_ch*channel_multiplier, 8),
        )

        self.decoder = SparseStructureDecoder(color_channels, geometry_channels, base_ch, levels, latent_ch, use_attn)

        self.mu  = fvnn.SparseConv3d(base_ch, latent_ch, 1)
        self.log = fvnn.SparseConv3d(base_ch, latent_ch, 1)
        
    @staticmethod
    def reparametrize(mu: fvnn.VDBTensor, log: fvnn.VDBTensor):
        std = (0.5*log.data.jdata).exp()
        eps = torch.randn_like(std)
        return fvnn.VDBTensor(mu.grid, mu.data + eps*std)
    
    @staticmethod
    def upsample(x: fvnn.VDBTensor, factor: int = 8):
        return fvnn.UpsamplingNearest([factor, factor, factor])(x)

    def encode(self, x: fvnn.VDBTensor):
        # encoder
        h = x
        for block in self.encoder_blocks: 
            h = block(h)

        # bottleneck
        h = self.pre_kl(h)
        mu, log = self.mu(h), self.log(h)
        return mu, log
    
    def decode(self, z: fvnn.VDBTensor):
        h = self.post_kl(z)
        h, masks, colors = self.decoder(h)

        return {"color": colors[-1], "color_levels": colors, "geometry": masks}

    # ------------------------------------------------------------
    def forward(self, x: fvnn.VDBTensor):
        # print("origins", x.grid.origins.max(), "voxel sizes", x.grid.voxel_sizes.max(), "num voxels",  x.grid.num_voxels.sum(), "grid max", x.grid.ijk.jdata.max())
        mu, log = self.encode(x)
        z = self.reparametrize(mu, log)
        d = self.decode(z)
        
        return {**d, "mu": mu, "logvar": log, "z": z}
    
    @staticmethod
    def apply_mask(x: fvnn.VDBTensor, mask_vdb: fvnn.VDBTensor, min_voxels: int = 16, max_voxels: int = 50000):
        """Apply mask but ensure minimum number of voxels remain and cap maximum to prevent OOM"""
        if mask_vdb.data.jdata.dtype != torch.bool:
            mask_values = mask_vdb.data.jdata.flatten()
            mask = (mask_values > 0.0)
        else:
            mask = mask_vdb.data.jdata.flatten()
            mask_values = mask_vdb.data.jdata.flatten().float()
        
        # Get batch indices for each voxel
        batch_indices = mask_vdb.grid.jidx
        batch_size = batch_indices.max().item() + 1
        
        # Process each batch item
        final_mask = torch.zeros_like(mask)
        
        for b in range(batch_size):
            # Get mask for this batch item
            batch_mask = (batch_indices == b)
            batch_voxel_mask = mask[batch_mask]
            batch_values = mask_values[batch_mask]
            
            # Apply min_voxels constraint
            if batch_voxel_mask.sum() < min_voxels and batch_mask.sum() > 0:
                # Keep top-k voxels for this batch item
                k = min(min_voxels, batch_mask.sum())
                _, topk_indices = torch.topk(batch_values, k, largest=True)
                batch_voxel_mask.zero_()
                batch_voxel_mask[topk_indices] = True
            
            # Apply max_voxels constraint
            if batch_voxel_mask.sum() > max_voxels:
                # Keep only top max_voxels for this batch item
                active_indices = torch.where(batch_voxel_mask)[0]
                active_values = batch_values[active_indices]
                _, top_indices = torch.topk(active_values, max_voxels, largest=True)
                keep_indices = active_indices[top_indices]
                
                batch_voxel_mask.zero_()
                batch_voxel_mask[keep_indices] = True
            
            # Update final mask
            final_mask[batch_mask] = batch_voxel_mask
        
        return fvnn.UpsamplingNearest(1)(x, final_mask)

    @staticmethod
    def merge_features(out: dict) -> fvnn.VDBTensor:
        """
        Combine color-head and geometry-head outputs into a single feature
        tensor:  RGB (3)  +  alpha (1)  +  the geometry channels (≥1).

        Returns
        -------
        fvnn.VDBTensor
            grid : same as geometry head
            feature : JaggedTensor with concatenated channels
        """
        col_vdb : fvnn.VDBTensor = out["color"]            # RGB  -- (N,3)

        col_feat  = col_vdb.data.jdata                 # (N, 3)  
        grid = col_vdb.grid
        feat  = col_feat

        # wrap back into JaggedTensor aligned with the grid
        feat_jt = grid.jagged_like(feat)                   # preserves jidx
        return fvnn.VDBTensor(grid, feat_jt)
    
    def inference(self, z: fvnn.VDBTensor):
        dict = self.decode(z)
        return self.merge_features(dict)


class SparseCoarseVAE(nn.Module):
    """
    Sparse coarse VAE.
    Same as SparseVAE but the latent space is a dense torch.Tensor.
    """
    def __init__(self, color_channels = 3,
                 geometry_channels = 1,
                 base_ch     = 128,
                 levels      = 3,        
                 latent_ch   = 32,
                 latent_res  = 16,
                 use_attn    = True):
        super().__init__()

        self.use_attn = use_attn
        self.latent_res = latent_res
        in_ch = color_channels

        self.encoder_blocks = nn.ModuleList([])
        self.encoder_blocks.append(SparseRes(in_ch, base_ch))
        channel_multiplier = 1
        ch = base_ch
        for _ in range(levels):
            self.encoder_blocks.append(nn.Sequential(
                SparseRes(ch, ch*2, downsample=True),
            ))
            ch *= 2
            channel_multiplier *= 2

        self.latent_max_voxels = latent_res ** 3
        # self.pre_kl = nn.Sequential(
        #     SparseRes(base_ch*channel_multiplier, base_ch*channel_multiplier),
        #     fvnn.GroupNorm(32, base_ch*channel_multiplier),
        #     fvnn.SiLU(),
        #     SparseRes(base_ch*channel_multiplier, base_ch),
        # )

        attn_channels = base_ch*6//4
        self.sparse_attn_pre_kl = nn.Sequential(
            # fvnn.SparseConv3d(base_ch*channel_multiplier, attn_channels, 1),
            SparseAttn3D(base_ch*channel_multiplier),  # Changed from attn_channels
            # fvnn.SparseConv3d(attn_channels, base_ch*channel_multiplier, 1),
        )
        self.sparse_attn_post_kl = nn.Sequential(
            # fvnn.SparseConv3d(base_ch*channel_multiplier, attn_channels, 1),
            SparseAttn3D(base_ch*channel_multiplier),  # Changed from attn_channels
            # fvnn.SparseConv3d(attn_channels, base_ch*channel_multiplier, 1),
        )

        self.pre_kl = nn.Sequential(
            nn.Conv3d(base_ch*channel_multiplier, base_ch, 1, stride=1, padding=0),
            nn.ReLU(),
            nn.GroupNorm(32, base_ch),
            nn.Conv3d(base_ch, base_ch, 3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv3d(base_ch, base_ch, 1, stride=1, padding=0),
        )

        self.post_kl = nn.Sequential(
            SparseRes(latent_ch, base_ch*channel_multiplier),
            fvnn.GroupNorm(32, base_ch*channel_multiplier),
            fvnn.SiLU(),
            SparseRes(base_ch*channel_multiplier, base_ch*channel_multiplier),
        )

        # self.post_kl = nn.Sequential(
        #     nn.Conv3d(latent_ch, base_ch, 3, stride=1, padding=1),
        #     nn.GroupNorm(32, base_ch),
        #     nn.SiLU(),
        #     nn.Conv3d(base_ch, base_ch, 3, stride=1, padding=1),
        #     nn.ReLU(),
        # )
        self.latent_geom_head = GeometryHead(latent_ch, geometry_channels)
        self.decoder = SparseStructureDecoder(color_channels, geometry_channels, base_ch, levels, latent_ch, latent_res, use_attn)

        # self.mu  = fvnn.SparseConv3d(base_ch, latent_ch, 1)
        # self.log = fvnn.SparseConv3d(base_ch, latent_ch, 1)

        self.mu = nn.Conv3d(base_ch, latent_ch, 1)
        self.log = nn.Conv3d(base_ch, latent_ch, 1)
        
    def reparametrize(self, mu: torch.Tensor, log: torch.Tensor):
        """Reparameterization for dense tensors in coarse VAE, using mean in eval mode."""
        if self.training:
            std = (0.5 * log).exp()
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    @staticmethod
    def upsample(x: fvnn.VDBTensor, factor: int = 8):
        return fvnn.UpsamplingNearest([factor, factor, factor])(x)

    def densify(self, x: fvnn.VDBTensor):
        dense = x.grid.write_to_dense(x.data, torch.tensor([0, 0, 0]), torch.tensor([self.latent_res, self.latent_res, self.latent_res]))
        dense = rearrange(dense, "b h w l c -> b c h w l")
        return dense

    def sparsify(self, v: torch.Tensor):
        """
        Returns a VDBTensor constructed from a dense tensor.
        Does not prune any voxels. (i.e. all voxels in the cube are active)
        """
        # breakpoint()
        v = rearrange(v, "b c h w l -> b h w l c")
        return fvdb.nn.vdbtensor_from_dense(v)
        # coords = torch.where(torch.ones(v.shape[:4], device=v.device).bool())
        # b, x, y, z = coords

        # xyz = torch.stack([x, y, z], dim=-1)
        # jagged_tensor = fvdb.JaggedTensor.from_data_and_indices(xyz.to(torch.float32), b.to(torch.int32), v.shape[0])

        # grid = fvdb.GridBatch(device=v.device)
        # grid.set_from_points(jagged_tensor)
        
        # feats = v[coords]
        # jidx = b.to(torch.int32).to(v.device)
        # feats_jagged = fvdb.JaggedTensor.from_data_and_indices(feats.to(torch.float32), jidx, v.shape[0])
        # return fvnn.VDBTensor(grid, feats_jagged)

    def encode(self, x: fvnn.VDBTensor):
        # encoder
        h = x
        for block in self.encoder_blocks: 
            h = block(h)

        # convert to dense
        if self.use_attn:
            h = self.sparse_attn_pre_kl(h)
        h_dense = self.densify(h)
        # bottleneck
        h_dense = self.pre_kl(h_dense)
        mu, log = self.mu(h_dense), self.log(h_dense)
        return mu, log
    
    def decode(self, z: torch.Tensor):
        z_sparse = self.sparsify(z)
        latent_mask = self.latent_geom_head(z_sparse)
        h = self.apply_mask(z_sparse, latent_mask, max_voxels=self.latent_max_voxels)
        h = self.post_kl(h)
        if self.use_attn:
            h = self.sparse_attn_post_kl(h)

        h, masks, colors, final_color_masked = self.decoder(h)

        return {"color": final_color_masked, "color_levels": colors, "geometry": [latent_mask] + masks}

    # ------------------------------------------------------------
    def forward(self, x: fvnn.VDBTensor):
        # print("origins", x.grid.origins.max(), "voxel sizes", x.grid.voxel_sizes.max(), "num voxels",  x.grid.num_voxels.sum(), "grid max", x.grid.ijk.jdata.max())
        mu, log = self.encode(x)
        z = self.reparametrize(mu, log)
        d = self.decode(z)
        
        return {**d, "mu": mu, "logvar": log, "z": z}
    
    @staticmethod
    def apply_mask(x: fvnn.VDBTensor, mask_vdb: fvnn.VDBTensor, min_voxels: int = 16, max_voxels: int = 131072):
        """Apply mask but ensure minimum number of voxels remain and cap maximum to prevent OOM"""
        if mask_vdb.data.jdata.dtype != torch.bool:
            mask_values = mask_vdb.data.jdata.flatten()
            mask = (mask_values > 0.0)
        else:
            mask = mask_vdb.data.jdata.flatten()
            mask_values = mask_vdb.data.jdata.flatten().float()
        
        # Get batch indices for each voxel
        batch_indices = mask_vdb.grid.jidx
        batch_size = batch_indices.max().item() + 1
        
        # Process each batch item
        final_mask = torch.zeros_like(mask)
        
        for b in range(batch_size):
            # Get mask for this batch item
            batch_mask = (batch_indices == b)
            batch_voxel_mask = mask[batch_mask]
            batch_values = mask_values[batch_mask]
            
            # Apply min_voxels constraint
            if batch_voxel_mask.sum() < min_voxels and batch_mask.sum() > 0:
                # Keep top-k voxels for this batch item
                k = min(min_voxels, batch_mask.sum())
                _, topk_indices = torch.topk(batch_values, k, largest=True)
                batch_voxel_mask.zero_()
                batch_voxel_mask[topk_indices] = True
            
            # Apply max_voxels constraint
            if batch_voxel_mask.sum() > max_voxels:
                # Keep only top max_voxels for this batch item
                active_indices = torch.where(batch_voxel_mask)[0]
                active_values = batch_values[active_indices]
                _, top_indices = torch.topk(active_values, max_voxels, largest=True)
                keep_indices = active_indices[top_indices]
                
                batch_voxel_mask.zero_()
                batch_voxel_mask[keep_indices] = True
            
            # Update final mask
            final_mask[batch_mask] = batch_voxel_mask
        
        return fvnn.UpsamplingNearest(1)(x, final_mask)

    @staticmethod
    def merge_features(out: dict) -> fvnn.VDBTensor:
        """
        Combine color-head and geometry-head outputs into a single feature
        tensor:  RGB (3)  +  alpha (1)  +  the geometry channels (≥1).

        Returns
        -------
        fvnn.VDBTensor
            grid : same as geometry head
            feature : JaggedTensor with concatenated channels
        """
        col_vdb : fvnn.VDBTensor = out["color"]            # RGB  -- (N,3)

        col_feat  = col_vdb.data.jdata                 # (N, 3)  
        grid = col_vdb.grid
        feat  = col_feat

        # wrap back into JaggedTensor aligned with the grid
        feat_jt = grid.jagged_like(feat)                   # preserves jidx
        return fvnn.VDBTensor(grid, feat_jt)
    
    def inference(self, z: fvnn.VDBTensor):
        dict = self.decode(z)
        return self.merge_features(dict)
