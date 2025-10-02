import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_, to_3tuple
from torch.distributions.normal import Normal
import torch.nn.functional as nnf
import numpy as np

import random
import math
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class InputEmbedding(nn.Module):
    def __init__(self, patch_size, input_channel, output_channel):
        super().__init__()
        self.output_channel = output_channel
        self.patch_size = patch_size
        self.input_channel = input_channel
        self.input_embedding = nn.Conv3d(
            input_channel, output_channel, kernel_size=(patch_size, patch_size, patch_size), stride=(patch_size, patch_size, patch_size)
        )

    def forward(self, input):
        batch_size, num_channels, depth, height, width = input.size()
        input = input.view(batch_size * num_channels, 1, depth, height, width)  # Reshape for Conv3D
        output = self.input_embedding(input)
        output = output.view(batch_size, num_channels, self.output_channel, -1)  # Reshape output
        return output


class LearnableTemporalPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len: int = 107520):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.empty(max_len, d_model), requires_grad=True)
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, input, indices):
        if indices is None:
            pe = self.pe[: input.size(1), :].unsqueeze(0)
        else:
            pe = self.pe[indices].unsqueeze(0)
        x = input + pe
        x = self.dropout(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.tem_pe = LearnableTemporalPositionalEncoding(hidden_dim, dropout)

    def forward(self, input, indices=None):
        batch_size, num_nodes, num_subseq, out_channels = input.size()
        input = self.tem_pe(input.view(batch_size * num_nodes, num_subseq, out_channels), indices=indices)
        input = input.view(batch_size, num_nodes, num_subseq, out_channels)
        return input


class TransformerLayers(nn.Module):
    def __init__(self, hidden_dim, num_layers, num_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = hidden_dim
        encoder_layers = TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim * 4, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

    def forward(self, input):
        batch_size, num_nodes, num_subseq, out_channels = input.size()
        x = input * math.sqrt(self.d_model)
        x = x.view(batch_size * num_nodes, num_subseq, out_channels)
        x = x.transpose(0, 1)
        output = self.transformer_encoder(x, mask=None)
        output = output.transpose(0, 1).view(batch_size, num_nodes, num_subseq, out_channels)
        return output


class MaskGenerator(nn.Module):
    def __init__(self, mask_size, mask_ratio):
        super().__init__()
        self.mask_size = mask_size
        self.mask_ratio = mask_ratio
        self.sort = True

    def uniform_rand(self):
        mask = list(range(int(self.mask_size)))
        random.shuffle(mask)
        mask_len = int(self.mask_size * self.mask_ratio)
        self.masked_tokens = mask[:mask_len]
        self.unmasked_tokens = mask[mask_len:]
        if self.sort:
            self.masked_tokens = sorted(self.masked_tokens)
            self.unmasked_tokens = sorted(self.unmasked_tokens)
        return self.unmasked_tokens, self.masked_tokens

    def forward(self):
        self.unmasked_tokens, self.masked_tokens = self.uniform_rand()
        return self.unmasked_tokens, self.masked_tokens


class Transformer(nn.Module):
    def __init__(
        self, patch_size, in_channel, out_channel, dropout, mask_size, mask_ratio, num_encoder_layers=6, mode="pretrain"
    ):
        super().__init__()
        self.patch_size = patch_size
        self.mode = mode
        self.patch = InputEmbedding(patch_size, in_channel, out_channel)
        self.pe = PositionalEncoding(out_channel, dropout=dropout)
        self.mask = MaskGenerator(mask_size, mask_ratio)
        self.encoder = TransformerLayers(out_channel, num_encoder_layers)
        self.encoder_2_decoder = nn.Linear(out_channel, out_channel)
        self.decoder = TransformerLayers(out_channel, 1)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, out_channel))
        trunc_normal_(self.mask_token, std=0.02)
        self.output_layer = nn.Linear(out_channel, patch_size ** 3)

    def _forward_pretrain(self, input):
        batch_size, num_channels, depth, height, width = input.size()

        patches = self.patch(input)
        patches = patches.transpose(-1, -2)
        patches = self.pe(patches)

        indices_not_masked, indices_masked = self.mask()
        repr_not_masked = patches[:, :, indices_not_masked, :]

        hidden_not_masked = self.encoder(repr_not_masked)
        hidden_not_masked = self.encoder_2_decoder(hidden_not_masked)

        hidden_masked = self.pe(
            self.mask_token.expand(batch_size, num_channels, len(indices_masked), hidden_not_masked.size(-1)),
            indices=indices_masked
        )

        hidden = torch.cat([hidden_not_masked, hidden_masked], dim=-2)
        hidden = self.decoder(hidden)

        output = self.output_layer(hidden)
        output_masked = output[:, :, len(indices_not_masked):, :]
        output_masked = output_masked.view(batch_size, num_channels, -1).transpose(1, 2)

        # Extracting labels based on patching
        patches_labels = input.unfold(2, self.patch_size, self.patch_size) \
                            .unfold(3, self.patch_size, self.patch_size) \
                            .unfold(4, self.patch_size, self.patch_size) \
                            .contiguous().view(batch_size, num_channels, -1, self.patch_size**3)

        # Ensure you are extracting the correct patches for labels
        # Reshape labels to match output_masked shape
        labels_masked = patches_labels[:, :, indices_masked, :].contiguous()  # Shape: [batch_size, num_channels, mask_size, patch_size^3]

        # Ensure labels_masked has the correct shape
        labels_masked = labels_masked.view(batch_size, num_channels, -1).transpose(1, 2)
        print(output.shape)
        print(patches_labels.shape)
        print(labels_masked.shape)
        print(output_masked.shape)
        return output_masked, labels_masked
    
    def _forward_backend(self, input):
        patches = self.patch(input)
        patches = patches.transpose(-1, -2)
        patches = self.pe(patches)
        hidden = self.encoder(patches)
        return hidden

    def forward(self, input_data):
        if self.mode == "pretrain":
            return self._forward_pretrain(input_data)
        else:
            return self._forward_backend(input_data)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, L, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, window_size, C)
    """
    B, H, W, L, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], L // window_size[2], window_size[2], C)

    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0], window_size[1], window_size[2], C)
    return windows


def window_reverse(windows, window_size, H, W, L):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
        L (int): Length of image
    Returns:
        x: (B, H, W, L, C)
    """
    B = int(windows.shape[0] / (H * W * L / window_size[0] / window_size[1] / window_size[2]))
    x = windows.view(B, H // window_size[0], W // window_size[1], L // window_size[2], window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, H, W, L, -1)
    return x

class LinearAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)

    def forward(self, x):
        B, N, C = x.size()
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        # Linear attention computation
        q = q * self.scale
        k = k.softmax(dim=-1)
        attn = (q @ k.transpose(-2, -1))  # This can be replaced with a linear approximation

        # Output calculation
        out = attn @ v
        return out.view(B, N, C)
    
class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, rpe=True, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1 * 2*Wt-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords_t = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w, coords_t]))  # 3, Wh, Ww, Wt
        coords_flatten = torch.flatten(coords, 1)  # 3, Wh*Ww*Wt
        self.rpe = rpe
        if self.rpe:
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wh*Ww*Wt, Wh*Ww*Wt
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww*Wt, Wh*Ww*Wt, 3
            relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 2] += self.window_size[2] - 1
            relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
            relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww*Wt, Wh*Ww*Wt
            self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww, Wt*Ww) or None
        """
        B_, N, C = x.shape #(num_windows*B, Wh*Ww*Wt, C)
        
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        if self.rpe:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1] * self.window_size[2],
                self.window_size[0] * self.window_size[1] * self.window_size[2], -1)  # Wh*Ww*Wt,Wh*Ww*Wt,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww*Wt, Wh*Ww*Wt
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=(7, 7, 7), shift_size=(0, 0, 0),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, rpe=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= min(self.shift_size) < min(self.window_size), "shift_size must in 0-window_size, shift_sz: {}, win_size: {}".format(self.shift_size, self.window_size)

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, rpe=rpe, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.H = None
        self.W = None
        self.T = None


    def forward(self, x, mask_matrix):
        H, W, T = self.H, self.W, self.T
        B, L, C = x.shape
        assert L == H * W * T, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, T, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_f = 0
        pad_r = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
        pad_b = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        pad_h = (self.window_size[2] - T % self.window_size[2]) % self.window_size[2]
        x = nnf.pad(x, (0, 0, pad_f, pad_h, pad_t, pad_b, pad_l, pad_r))
        _, Hp, Wp, Tp, _ = x.shape

        # cyclic shift
        if min(self.shift_size) > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2], C)  # nW*B, window_size*window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], self.window_size[2], C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp, Tp)  # B H' W' L' C

        # reverse cyclic shift
        if min(self.shift_size) > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0 or pad_h > 0:
            x = x[:, :H, :W, :T, :].contiguous()

        x = x.view(B, H * W * T, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm, reduce_factor=2):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(8 * dim, (8//reduce_factor) * dim, bias=False)
        self.norm = norm_layer(8 * dim)


    def forward(self, x, H, W, T):
        """
        x: B, H*W*T, C
        """
        B, L, C = x.shape
        assert L == H * W * T, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0 and T % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, T, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1) or (T % 2 == 1)
        if pad_input:
            x = nnf.pad(x, (0, 0, 0, T % 2, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, 0::2, :]  # B H/2 W/2 T/2 C
        x1 = x[:, 1::2, 0::2, 0::2, :]  # B H/2 W/2 T/2 C
        x2 = x[:, 0::2, 1::2, 0::2, :]  # B H/2 W/2 T/2 C
        x3 = x[:, 0::2, 0::2, 1::2, :]  # B H/2 W/2 T/2 C
        x4 = x[:, 1::2, 1::2, 0::2, :]  # B H/2 W/2 T/2 C
        x5 = x[:, 0::2, 1::2, 1::2, :]  # B H/2 W/2 T/2 C
        x6 = x[:, 1::2, 0::2, 1::2, :]  # B H/2 W/2 T/2 C
        x7 = x[:, 1::2, 1::2, 1::2, :]  # B H/2 W/2 T/2 C
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)  # B H/2 W/2 T/2 8*C
        x = x.view(B, -1, 8 * C)  # B H/2*W/2*T/2 8*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """
    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=(7, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 rpe=True,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 pat_merg_rf=2,):
        super().__init__()
        self.window_size = window_size
        self.shift_size = (window_size[0] // 2, window_size[1] // 2, window_size[2] // 2)
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.pat_merg_rf = pat_merg_rf
        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else (window_size[0] // 2, window_size[1] // 2, window_size[2] // 2),
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                rpe=rpe,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer, reduce_factor=self.pat_merg_rf)
        else:
            self.downsample = None

    def forward(self, x, H, W, T):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W*T, C).
            H, W: Spatial resolution of the input feature.
        """
        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size[0])) * self.window_size[0]
        Wp = int(np.ceil(W / self.window_size[1])) * self.window_size[1]
        Tp = int(np.ceil(T / self.window_size[2])) * self.window_size[2]
        img_mask = torch.zeros((1, Hp, Wp, Tp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size[0]),
                    slice(-self.window_size[0], -self.shift_size[0]),
                    slice(-self.shift_size[0], None))
        w_slices = (slice(0, -self.window_size[1]),
                    slice(-self.window_size[1], -self.shift_size[1]),
                    slice(-self.shift_size[1], None))
        t_slices = (slice(0, -self.window_size[2]),
                    slice(-self.window_size[2], -self.shift_size[2]),
                    slice(-self.shift_size[2], None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                for t in t_slices:
                    img_mask[:, h, w, t, :] = cnt
                    cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2])
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.blocks:
            blk.H, blk.W, blk.T = H, W, T
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)
        if self.downsample is not None:
            x_down = self.downsample(x, H, W, T)
            Wh, Ww, Wt = (H + 1) // 2, (W + 1) // 2, (T + 1) // 2
            return x, H, W, T, x_down, Wh, Ww, Wt
        else:
            return x, H, W, T, x, H, W, T

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=2, in_chans=5, embed_dim=16, norm_layer=None):
        super().__init__()
        patch_size = to_3tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W, T = x.size()
        if T % self.patch_size[2] != 0:
            x = nnf.pad(x, (0, self.patch_size[2] - T % self.patch_size[2]))
        if W % self.patch_size[1] != 0:
            x = nnf.pad(x, (0, 0, 0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = nnf.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # B C Wh Ww Wt
        if self.norm is not None:
            Wh, Ww, Wt = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww, Wt)
        return x

class SinusoidalPositionEmbedding(nn.Module):
    '''
    Rotary Position Embedding
    '''
    def __init__(self,):
        super(SinusoidalPositionEmbedding, self).__init__()

    def forward(self, x):
        batch_sz, n_patches, hidden = x.shape
        position_ids = torch.arange(0, n_patches).float().cuda()
        indices = torch.arange(0, hidden//2).float().cuda()
        indices = torch.pow(10000.0, -2 * indices / hidden)
        embeddings = torch.einsum('b,d->bd', position_ids, indices)
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = torch.reshape(embeddings, (1, n_patches, hidden))
        return embeddings

class SinPositionalEncoding3D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(SinPositionalEncoding3D, self).__init__()
        channels = int(np.ceil(channels/6)*2)
        if channels % 2:
            channels += 1
        self.channels = channels
        self.inv_freq = 1. / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        #self.register_buffer('inv_freq', inv_freq)

    def forward(self, tensor):
        """
        :param tensor: A 5d tensor of size (batch_size, x, y, z, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, z, ch)
        """
        tensor = tensor.permute(0, 2, 3, 4, 1)
        if len(tensor.shape) != 5:
            raise RuntimeError("The input tensor has to be 5d!")
        batch_size, x, y, z, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        pos_z = torch.arange(z, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        sin_inp_z = torch.einsum("i,j->ij", pos_z, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1).unsqueeze(1).unsqueeze(1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1).unsqueeze(1)
        emb_z = torch.cat((sin_inp_z.sin(), sin_inp_z.cos()), dim=-1)
        emb = torch.zeros((x,y,z,self.channels*3),device=tensor.device).type(tensor.type())
        emb[:,:,:,:self.channels] = emb_x
        emb[:,:,:,self.channels:2*self.channels] = emb_y
        emb[:,:,:,2*self.channels:] = emb_z
        emb = emb[None,:,:,:,:orig_ch].repeat(batch_size, 1, 1, 1, 1)
        return emb.permute(0, 4, 1, 2, 3)

class SwinTransformer(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (tuple): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, pretrain_img_size=224,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=(7, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 spe=False,
                 rpe=True,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 use_checkpoint=False,
                 pat_merg_rf=2,):
        super().__init__()
        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.spe = spe
        self.rpe = rpe
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            pretrain_img_size = to_3tuple(self.pretrain_img_size)
            patch_size = to_3tuple(patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1], pretrain_img_size[2] // patch_size[2]]

            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1], patches_resolution[2]))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        elif self.spe:
            self.pos_embd = SinPositionalEncoding3D(embed_dim).cuda()
            #self.pos_embd = SinusoidalPositionEmbedding().cuda()
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                depth=depths[i_layer],
                                num_heads=num_heads[i_layer],
                                window_size=window_size,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias,
                                rpe = rpe,
                                qk_scale=qk_scale,
                                drop=drop_rate,
                                attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                norm_layer=norm_layer,
                                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                                use_checkpoint=use_checkpoint,
                               pat_merg_rf=pat_merg_rf,)
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Forward function."""
        x = self.patch_embed(x)

        Wh, Ww, Wt = x.size(2), x.size(3), x.size(4)

        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = nnf.interpolate(self.absolute_pos_embed, size=(Wh, Ww, Wt), mode='trilinear')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww*Wt C
        elif self.spe:
            x = (x + self.pos_embd(x)).flatten(2).transpose(1, 2)
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, T, x, Wh, Ww, Wt = layer(x, Wh, Ww, Wt)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)

                out = x_out.view(-1, H, W, T, self.num_features[i]).permute(0, 4, 1, 2, 3).contiguous()
                outs.append(out)
        return outs

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer, self).train(mode)
        self._freeze_stages()

class Conv3dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        relu = nn.LeakyReLU(inplace=False)
        if not use_batchnorm:
            nm = nn.InstanceNorm3d(out_channels)
        else:
            nm = nn.BatchNorm3d(out_channels)

        super(Conv3dReLU, self).__init__(conv, nm, relu)

class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv3dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv3dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

    def forward(self, x, skip=None):
        # x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.up(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class RegistrationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        conv3d.weight = nn.Parameter(Normal(0, 1e-5).sample(conv3d.weight.shape))
        conv3d.bias = nn.Parameter(torch.zeros(conv3d.bias.shape))
        super().__init__(conv3d)

class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    Obtained from https://github.com/voxelmorph/voxelmorph
    """
    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=False, mode=self.mode)

class TransMorph(nn.Module):
    def __init__(self, config):
        '''
        TransMorph Model
        '''
        super(TransMorph, self).__init__()
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        self.transformer = SwinTransformer(patch_size=config.patch_size,
                                           in_chans=config.in_chans,
                                           embed_dim=config.embed_dim,
                                           depths=config.depths,
                                           num_heads=config.num_heads,
                                           window_size=config.window_size,
                                           mlp_ratio=config.mlp_ratio,
                                           qkv_bias=config.qkv_bias,
                                           drop_rate=config.drop_rate,
                                           drop_path_rate=config.drop_path_rate,
                                           ape=config.ape,
                                           spe=config.spe,
                                           rpe=config.rpe,
                                           patch_norm=config.patch_norm,
                                           use_checkpoint=config.use_checkpoint,
                                           out_indices=config.out_indices,
                                           pat_merg_rf=config.pat_merg_rf,
                                           )
        self.up0 = DecoderBlock(embed_dim*8, embed_dim*4, skip_channels=embed_dim*4 if if_transskip else 0, use_batchnorm=False)
        self.up1 = DecoderBlock(embed_dim*4, embed_dim*2, skip_channels=embed_dim*2 if if_transskip else 0, use_batchnorm=False)  # 384, 20, 20, 64
        self.up2 = DecoderBlock(embed_dim*2, embed_dim, skip_channels=embed_dim if if_transskip else 0, use_batchnorm=False)  # 384, 40, 40, 64
        self.up3 = DecoderBlock(embed_dim, config.reg_head_chan, skip_channels=embed_dim//2 if if_convskip else 0, use_batchnorm=False)  # 384, 80, 80, 128
        #self.up4 = DecoderBlock(embed_dim//2, config.reg_head_chan, skip_channels=config.reg_head_chan if if_convskip else 0, use_batchnorm=False)  # 384, 160, 160, 256
        self.c1 = Conv3dReLU(2, embed_dim//2, 3, 1, use_batchnorm=False)
        self.c2 = Conv3dReLU(2, config.reg_head_chan, 3, 1, use_batchnorm=False)
        self.reg_head = RegistrationHead(
            in_channels=config.reg_head_chan,
            out_channels=3,
            kernel_size=3,
        )
        self.spatial_trans = SpatialTransformer(config.img_size)
        self.avg_pool = nn.AvgPool3d(3, stride=2, padding=1)

    def forward(self, x):
        source = x[:, 0:1, :, :]
        if self.if_convskip:
            x_s1 = self.avg_pool(x)
            f4 = self.c1(x_s1)
        else:
            f4 = None

        out_feats = self.transformer(x)

        if self.if_transskip:
            f1 = out_feats[-2]
            f2 = out_feats[-3]
            f3 = out_feats[-4]
        else:
            f1 = None
            f2 = None
            f3 = None
        x = self.up0(out_feats[-1], f1)
        x = self.up1(x, f2)
        x = self.up2(x, f3)
        x = self.up3(x, f4)
        flow = self.reg_head(x)
        flow = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)(flow)
        out = self.spatial_trans(source, flow)
        return out, flow#, out_feats

class ResBlock(nn.Module):
    """Residual Block with optional downsampling"""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
            nn.BatchNorm3d(out_channels)
        ) if stride != 1 or in_channels != out_channels else None

    def forward(self, x):
        identity = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        if self.downsample:
            identity = self.downsample(identity)

        x += identity
        return self.relu(x)
    
class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, nsteps=7, img_size=(160, 160, 80)):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)

        self.spatial_trans = SpatialTransformer(img_size)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.spatial_trans(vec, vec)
        return vec

class HybridEncoder_v2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_embed = nn.ModuleList()
        self.patch_embed.append(PatchEmbed())
        self.pos_drop = nn.Dropout(p=0.)
        self.stages = nn.ModuleList()
        self.gated_fusions = nn.ModuleList()
        current_dim = config.embed_dim

        for i in range(config.num_layers):
            if i % 2 == 0:  # CNN stage
                if i == 0:
                    stage = nn.Sequential(
                        *[ResBlock(2 if j == 0 else current_dim, current_dim if j == 0 else current_dim,
                                   stride=2 if j == 0 else 1) for j in range(config.depths[i])]
                    )
                else:
                    stage = nn.Sequential(
                        *[ResBlock(current_dim, current_dim if j == 0 else current_dim * 2,
                                   stride=2 if j == 0 else 1) for j in range(config.depths[i])]
                    )
                current_dim = current_dim * 2 if i != 0 else current_dim
                self.patch_embed.append(PatchEmbed(in_chans=current_dim, embed_dim=current_dim * 2))
                self.gated_fusions.append(GatedFeatureFusion(current_dim))
            else:  # Transformer stage
                stage = BasicLayer(
                    dim=current_dim,
                    num_heads=config.num_heads[i],
                    window_size=config.window_size,
                    mlp_ratio=config.mlp_ratio,
                    drop_path=config.drop_path_rate,
                    depth=config.depths[i]
                )
            self.stages.append(stage)

    def forward(self, x):
        cnn_skips = []
        trans_skips = []
        print(x.shape)
        x_patches = self.patch_embed[0](x)

        for i, stage in enumerate(self.stages):
            if isinstance(stage, nn.Sequential):
                print(i)
                print(x.shape)
                x = stage(x)
                cnn_skips.append(x)
                print(x.shape)
            else:
                print(i)
                x = self.gated_fusions[(i // 2)](x, x_patches)
                C, Wh, Ww, Wt = x.size(1), x.size(2), x.size(3), x.size(4)
                print(Wh, Ww, Wt)
                x = x.flatten(2).transpose(1, 2)
                x, H, W, T, _, H, W, T = stage(x, Wh, Ww, Wt)
                print(x.shape)
                # if i==0:
                    # x = x + tran_repr
                x = x.reshape(1, C, H, W, T)  # Adjust the channels and dimensions as needed
                trans_skips.append(x)  # Adjust the channels and dimensions as needed
                x_patches = self.patch_embed[(i // 2) + 1](x)
                print(x.shape)

        return x, cnn_skips, trans_skips

def gradient(x):
    # Create 3D Sobel kernels for each axis
    sobel_x = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           [[-2, 0, 2], [-4, 0, 4], [-2, 0, 2]],
                           [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=torch.float32)
    
    sobel_y = sobel_x.transpose(0, 1)
    sobel_z = sobel_x.transpose(0, 2)
    
    # Stack kernels for 3D convolution
    kernel = torch.stack([sobel_x, sobel_y, sobel_z])  # Shape: [3, 3, 3, 3]
    kernel = kernel.unsqueeze(1).to(x.device)  # Shape: [3, 1, 3, 3, 3]
    
    # Pad and convolve
    x = F.pad(x, (1,1,1,1,1,1), mode='replicate')
    return F.conv3d(x, kernel, groups=x.shape[1])

class DifferenceEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.stages = nn.ModuleList()
        current_dim = config.embed_dim//2
        self.patch_embed = PatchEmbed(in_chans=4, embed_dim=current_dim)  # 2 channels for absolute difference + 2 for gradient difference
        
        for i in range(config.num_layers):
            if i % 2 == 0:  # CNN stage (local differences)
                stage = nn.Sequential(
                    *[ResBlock(
                        in_channels=current_dim if j == 0 else current_dim * 2,
                        out_channels=current_dim * 2,
                        stride=2 if j == 0 else 1
                    ) for j in range(config.depths[i])]
                )
                current_dim *= 2
            else:  # Transformer stage (global differences)
                stage = BasicLayer(
                    dim=current_dim,
                    num_heads=config.num_heads[i],
                    window_size=config.window_size,
                    mlp_ratio=config.mlp_ratio,
                    drop_path=config.drop_path_rate,
                    depth=config.depths[i]
                )
            self.stages.append(stage)

    def forward(self, x1, x2):
        # Input: Absolute difference + Gradient difference (local + structural)
        x_diff = torch.abs(x1 - x2)
        x_grad = torch.abs(gradient(x1) - gradient(x2))  # ∇x1 - ∇x2
        x_diff = torch.cat([x_diff, x_grad], dim=1)  # Combine
        x_diff = self.patch_embed(x_diff)
        
        # Process through hybrid stages
        for stage in self.stages:
            if isinstance(stage, nn.Sequential):  # CNN
                x_diff = stage(x_diff)
            else:  # Transformer
                B, C, H, W, D = x_diff.shape
                x_diff = x_diff.flatten(2).transpose(1, 2)  # (B, N, C)
                x_diff, H, W, T, _, H, W, T = stage(x_diff, H, W, D)
                x_diff = x_diff.reshape(B, C, H, W, D)
                
        
        return x_diff
    
class HybridEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_embed = PatchEmbed()
        self.pos_drop = nn.Dropout(p=0.)
        self.stages = nn.ModuleList()
        current_dim = config.embed_dim

        for i in range(config.num_layers):
            if i % 2 == 0:  # CNN stage
                if i == 0:
                    stage = nn.Sequential(
                        *[ResBlock(16 if j == 0 else current_dim, current_dim if j == 0 else current_dim,
                                   stride=1) for j in range(config.depths[i])]
                    )
                else:
                    stage = nn.Sequential(
                        *[ResBlock(current_dim, current_dim if j == 0 else current_dim * 2,
                                   stride=1) for j in range(config.depths[i])]
                    )
                    current_dim = current_dim * 2
                # Keep current_dim the same
            else:  # Transformer stage
                stage = BasicLayer(
                    dim=current_dim,
                    num_heads=config.num_heads[i],
                    window_size=config.window_size,
                    mlp_ratio=config.mlp_ratio,
                    drop_path=config.drop_path_rate,
                    depth=config.depths[i]
                )
            self.stages.append(stage)

    def forward(self, x):
        cnn_skips = []
        trans_skips = []
        x = self.patch_embed(x)
        cnn_skips.append(x)  # F_stem

        for i, stage in enumerate(self.stages):
            if isinstance(stage, nn.Sequential):
                x = stage(x)
                cnn_skips.append(x)  # Keep dimensions same
            else:
                C, Wh, Ww, Wt = x.size(1), x.size(2), x.size(3), x.size(4)
                x = x.flatten(2).transpose(1, 2)
                x, H, W, T, _, H, W, T = stage(x, Wh, Ww, Wt)
                x = x.reshape(1, C, H, W, T)  # Adjust the channels and dimensions as needed
                trans_skips.append(x)

        return x, cnn_skips, trans_skips

class CAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.gap(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)


class TransformerDecoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0.,
                 attn_drop=0., out_dim=None):
        super().__init__()
        self.out_dim = out_dim if out_dim is not None else dim
        self.norm1 = nn.LayerNorm(dim)
        '''
        self.cross_attn = WindowAttention(
            dim, window_size=(1, 1, 1),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop
        )
        '''
        self.cross_attn = LinearAttention(
            dim,
            num_heads=num_heads
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, hidden_features=int(dim * mlp_ratio), drop=drop)

        # Add channel reduction if needed
        if dim != self.out_dim:
            self.channel_reducer = nn.Linear(dim, self.out_dim)
        else:
            self.channel_reducer = nn.Identity()
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear')

    def forward(self, x):
        B, C, H, W, T = x.shape
        print(x.shape)
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        print(x.shape)

        # Transformer operations
        x = x + self.cross_attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        # Channel reduction if needed
        x = self.channel_reducer(x)

        # Reshape back
        x = x.transpose(1, 2).reshape(B, self.out_dim, H, W, T)
        # Upsample if needed
        if self.upsample:
            x = self.upsample(x)  # (B, C, 2H, 2W, 2T)

        return x
    
class LinearAttention_v2(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, context=None):
        B, N, C = x.size()
        context = x if context is None else context  # If no context, self-attention

        # Project queries, keys, values
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # (B, N, num_heads, head_dim)

        # Linear attention (simplified)
        q = q * self.scale
        k = k.softmax(dim=-2)  # Softmax over sequence length
        context = (k.transpose(-2, -1) @ v)  # (B, num_heads, head_dim, head_dim)
        out = q @ context  # (B, N, num_heads, head_dim)
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class TransformerDecoderBlock_v2(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.):
        super().__init__()
        # Self-attention
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = LinearAttention_v2(dim, num_heads)

        # Cross-attention (to encoder features)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = LinearAttention_v2(dim, num_heads)

        # FFN
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, hidden_features=int(dim * mlp_ratio), drop=drop)

    def forward(self, x, enc_feats=None):
        # x: (B, C, H, W, T) -> flatten to (B, N, C)
        B, C, H, W, T = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)

        # Self-attention
        x = x + self.self_attn(self.norm1(x))

        # Cross-attention (if encoder features provided)
        if enc_feats is not None:
            enc_feats = enc_feats.flatten(2).transpose(1, 2)  # (B, N, C)
            x = x + self.cross_attn(self.norm2(x), enc_feats)

        # FFN
        x = x + self.mlp(self.norm3(x))

        # Reshape back to (B, C, H, W, T)
        x = x.transpose(1, 2).reshape(B, C, H, W, T)
        return x
    
class ProgressiveFieldEstimation(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.conv1x1_cnn = None
        self.conv1x1_transformer = None
        self.dwconv3x3 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels)
        self.vec_int = VecInt()

    def initialize_convolutions(self, in_channels_cnn, in_channels_transformer):
        # 1x1 Convolution to adjust channel dimensions
        self.conv1x1_cnn = nn.Conv3d(in_channels_cnn, self.dwconv3x3.in_channels, kernel_size=1).to(self.dwconv3x3.weight.device)
        self.conv1x1_transformer = nn.Conv3d(in_channels_transformer, self.dwconv3x3.in_channels, kernel_size=1).to(self.dwconv3x3.weight.device)

    def forward(self, cnn_features, transformer_features, prev_vector_field=None):
        # Process CNN features
        cnn_processed = self.conv1x1_cnn(cnn_features)
        
        # Process Transformer features
        transformer_processed = self.conv1x1_transformer(transformer_features)
        
        # Merge both processed features
        merged_features = cnn_processed + transformer_processed
        
        # Apply depthwise convolution for refinement
        vector_field = self.dwconv3x3(merged_features)

        # Optionally include the previous deformation field
        if prev_vector_field is not None:
            # Upsample the previous deformation field
            prev_vector_field = F.interpolate(prev_vector_field, scale_factor=2, mode='trilinear')
            vector_field += prev_vector_field
        
        return vector_field

class HybridDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.stages = nn.ModuleList()
        dim = config.embed_dim * 4

        for i in range(3):
            if i == 2:
                stage = DecoderBlock(dim, 16)
            else:
                stage = DecoderBlock(dim, dim // 2)  # Adjust as needed
            dim //= 2
            self.stages.append(stage)

    def forward(self, x, cnn_skips, trans_skips, deformation_field):
        # Print shapes of cnn_skips
        print("CNN Skip Connections Shapes:")
        for i in range(len(cnn_skips)):
            print(f"cnn_skips[{i}]: {cnn_skips[i].shape}")

        # Print shapes of trans_skips
        print("Transformer Skip Connections Shapes:")
        for i in range(len(trans_skips)):
            print(f"trans_skips[{i}]: {trans_skips[i].shape}")
        for i, stage in enumerate(self.stages):
                if i==2:
                    x = stage(x)
                    x += cnn_skips[-(0 + 2)]  # Adding skip connections
                else:
    #           if isinstance(stage, nn.Sequential):
                    x = stage(x)  # Avoid index out of range
                    print(i, x.shape, trans_skips[-(0 + 2)].shape, cnn_skips[-(0+2)].shape)
                    x += cnn_skips[-(0 + 2)]  # Adding skip connections
                    x += trans_skips[-(0 + 2)]
                    del cnn_skips[-(0 + 1)], trans_skips[-(0 + 1)]

        # Use the final deformation field in the last stage if needed
        if deformation_field is not None:
            deformation_field = F.interpolate(deformation_field, scale_factor=2, mode='trilinear')
            x += deformation_field  # You can adjust how you want to incorporate it

        print(x.shape)
        return x

class FusionMorph(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = HybridEncoder(config)
        self.decoder = HybridDecoder(config)
        self.reg_head = RegistrationHead(
            in_channels=config.reg_head_chan,
            out_channels=3,
            kernel_size=3,
        )

        self.transformer_moving = Transformer(
            patch_size=4,
            in_channel=1,
            out_channel=48,
            dropout=0.1,
            mask_size=107520,
            mask_ratio=0.5,
            num_encoder_layers=6,
            mode="inference"
        )

        self.transformer_fixed = Transformer(
            patch_size=4,
            in_channel=1,
            out_channel=48,
            dropout=0.1,
            mask_size=107520,
            mask_ratio=0.5,
            num_encoder_layers=6,
            mode="inference"
        )
        
        # Initialize PFE without specific channels
        self.pfe1 = ProgressiveFieldEstimation(config.reg_head_chan)
        self.pfe2 = ProgressiveFieldEstimation(config.reg_head_chan)
        self.pfe3 = ProgressiveFieldEstimation(config.reg_head_chan)
        # Progressive refinement
        scales= [384, 192, 96]  # Define the number of scales
        self.pfe1.initialize_convolutions(scales[0], scales[0])
        self.pfe2.initialize_convolutions(scales[1], scales[1])
        self.pfe3.initialize_convolutions(scales[2], scales[2])
        # Initialize the spatial transformer
        self.spatial_trans = SpatialTransformer(config.img_size)
        self.vec_int = VecInt()
        self.apply(self._init_weights)
        # Load pretrained transformers once
        self._load_pretrained_transformers("/root/miccai2025/Checkpoint_trans/transformer_checkpoint_step_3.pth")

    def _load_pretrained_transformers(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.transformer_moving.load_state_dict(checkpoint['model_state_dict'])
        self.transformer_fixed.load_state_dict(checkpoint['model_state_dict'])
        
        # Freeze (optional)
        for param in self.transformer_moving.parameters():
            param.requires_grad = True
        for param in self.transformer_fixed.parameters():
            param.requires_grad = True
            
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        source = x[:, 0:1, :, :, :]
        target = x[:, 1:2, :, :, :]
        # Extract features from moving images
        moving_repr = self.transformer_moving(source).squeeze(1)

        # Extract features from fixed images
        fixed_repr = self.transformer_fixed(target).squeeze(1)

        tran_repr = torch.cat([moving_repr, fixed_repr], dim=2)
        print(tran_repr.shape)

        # Encoding and decoding
        x, cnn_skips, trans_skips = self.encoder(x, tran_repr)

        # Initialize the deformation field
        deformation_field = None

        # Progressive refinement
        deformation_field = self.pfe1(cnn_skips[-1], trans_skips[-1], deformation_field)
        deformation_field = self.pfe2(cnn_skips[-2], trans_skips[-2], deformation_field)
        deformation_field = self.pfe3(cnn_skips[-3], trans_skips[-3], deformation_field)

        # Decode using the current deformation field
        x = self.decoder(x, cnn_skips, trans_skips, deformation_field)

        # Keep these outside for numerical stability
        flow = self.reg_head(x)
        flow = F.interpolate(flow, scale_factor=2, mode='trilinear')
        flow = self.vec_int(flow)
        warped = self.spatial_trans(source, flow)
        return warped, flow
    
class GatedFeatureFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Mapping function 'g' (3x3 conv + channel reduction)
        self.gate_conv = nn.Conv3d(channels * 4, channels * 2, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.raw_alpha = nn.Parameter(torch.zeros(1))  # Learned in log-space
        
        # Additional projection to match output dimension
        self.output_proj = nn.Conv3d(channels * 2, channels, kernel_size=1) 

    def forward(self, Flocal, Fglobal, i=0):
        # Flocal: CNN features (local) [B, C, H, W, D]
        # Fglobal: Transformer features (global) [B, C, H, W, D]
        # Channel-wise concatenation
        x = torch.cat([Flocal, Fglobal], dim=1)  # [B, 2C, H, W, D]
        
        # Compute gating mask (Glocal)
        Glocal = self.sigmoid(self.gate_conv(x))  # [B, C, H, W, D]
        
        alpha = torch.exp(self.raw_alpha)  # Always positive
        # print("alpha:", alpha.item())
        # Adaptive fusion (Eq. 16)
        F_refined = alpha * (Glocal * Flocal) + Fglobal  # [B, C, H, W, D]
        F_refined = self.output_proj(F_refined)  # Project to output dimension
        return F_refined
    
class FusionMorph_dual(nn.Module):
    def __init__(self, config, upsampled=False):
        super().__init__()
        self.encoder = HybridEncoder(config)
        self.upsampled = upsampled

        # Decoders and Fusion Modules
        self.cnn_decoders = nn.ModuleList()
        self.trans_decoders = nn.ModuleList()
        self.gated_fusions = nn.ModuleList()
        
        dim = config.embed_dim * 4  # Initial dimension for the first decoder block
        for i in range(4):
            # CNN Decoder
            if i==3:
                dim=32
                cond_dim=16
                self.cnn_decoders.append(DecoderBlock(dim, 16, cond_dim))
            out_channels = dim // 2
            self.cnn_decoders.append(DecoderBlock(dim, out_channels*2, dim))
            
            # Transformer Decoder
            self.trans_decoders.append(TransformerDecoderBlock_v2(dim, out_channels))
            
            # Gated Fusion
            self.gated_fusions.append(GatedFeatureFusion(out_channels))
            
            dim = out_channels
            
        if upsampled:
            # If upsampled, add an additional cnn-trans decoder blocks to obtain the final output (with a higher resolution)
            self.cnn_decoders.append(DecoderBlock(24, config.reg_head_chan))
        # Registration Head
        self.reg_head = RegistrationHead(
            in_channels=config.reg_head_chan,
            out_channels=3,
            kernel_size=3,
        )
        '''
        # Transformers (Frozen)
        self.transformer_moving = Transformer(
            patch_size=4,
            in_channel=1,
            out_channel=48,
            dropout=0.1,
            mask_size=107520,
            mask_ratio=0.5,
            num_encoder_layers=6,
            mode="inference"
        )

        self.transformer_fixed = Transformer(
            patch_size=4,
            in_channel=1,
            out_channel=48,
            dropout=0.1,
            mask_size=107520,
            mask_ratio=0.5,
            num_encoder_layers=6,
            mode="inference"
        )
        '''
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        # Spatial Transformers
        self.spatial_trans = SpatialTransformer(config.img_size)
        self.vec_int = VecInt()
        self.reduce_channels = nn.Conv3d(in_channels=16, out_channels=2, kernel_size=1)
    '''     
        # Init and Load Pretrained
        self.apply(self._init_weights)
        self._load_pretrained_transformers("/root/miccai2025/Checkpoint_trans/transformer_checkpoint_step_3.pth")

    def _load_pretrained_transformers(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.transformer_moving.load_state_dict(checkpoint['model_state_dict'])
        self.transformer_fixed.load_state_dict(checkpoint['model_state_dict'])
        for param in self.transformer_moving.parameters():
            param.requires_grad = True
        for param in self.transformer_fixed.parameters():
            param.requires_grad = True
            
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    '''
    def forward(self, x, upsampled = False, pretraining=False, lambda_orth=1.0):
        # Feature Extraction
        source, target = x[:, 0:1], x[:, 1:2]
        # moving_repr = self.transformer_moving(source).squeeze(1)
        # fixed_repr = self.transformer_fixed(target).squeeze(1)
        # tran_repr = torch.cat([moving_repr, fixed_repr], dim=2)
        
        # Encoding
        x, cnn_skips, trans_skips = self.encoder(x)
        loss = 0
        if pretraining:
            # Decoding with Gated Fusion
            for i in range(len(cnn_skips)):
                if i==3:
                    x = self.cnn_decoders[i](x, cnn_skips[len(cnn_skips)-1-i])
                else:
                    x_cnn = self.cnn_decoders[i](x, cnn_skips[len(cnn_skips)-1-i]) if i < len(cnn_skips) else x
                    x_trans = self.trans_decoders[i](x, trans_skips[2-i]) if i < len(trans_skips) else x
                    x_trans = self.up(x_trans)
                    x = self.gated_fusions[i](x_cnn, x_trans)
            reconstructed_output = self.reduce_channels(x)
            print("Reconstructed Output Shape:", reconstructed_output.shape)
            return reconstructed_output  # Return the reconstructed image
        
        # Decoding with Gated Fusion
        for i in range(len(cnn_skips)):
            if i==3:
                print(x.shape, cnn_skips[len(cnn_skips)-1-i].shape)
                x = self.cnn_decoders[i](x, cnn_skips[len(cnn_skips)-1-i])
            else:
                x_cnn = self.cnn_decoders[i](x, cnn_skips[len(cnn_skips)-1-i]) if i < len(cnn_skips) else x
                x_trans = self.trans_decoders[i](x, trans_skips[2-i]) if i < len(trans_skips) else x
                x_trans = self.up(x_trans)
                x = self.gated_fusions[i](x_cnn, x_trans)
                
        if upsampled:
            x = self.cnn_decoders[-1](x)
            
        # Registration
        flow = self.vec_int(self.reg_head(x))
        if upsampled:
            # 2. Downsample to original resolution (preserves subpixel info)
            flow = F.interpolate(
                flow, 
                size=(160, 224, 192), 
                mode='trilinear',  # For 3D
                align_corners=False
            )
        warped = self.spatial_trans(source, flow)
        
        return warped, flow, loss

class Dual_FusionMorph(nn.Module):
    def __init__(self, config, upsampled=False):
        super().__init__()
        self.encoder = HybridEncoder(config)
        self.upsampled = upsampled

        # Decoders and Fusion Modules
        self.cnn_decoders = nn.ModuleList()
        self.trans_decoders = nn.ModuleList()
        self.gated_fusions = nn.ModuleList()
        
        dim = config.embed_dim * 8  # Initial dimension for the first decoder block
        for i in range(4):
            # CNN Decoder
            if i==3:
                dim=96
                cond_dim = 32
                out_channels=16
                self.cnn_decoders.append(DecoderBlock(dim, out_channels, cond_dim))
            else:
                out_channels = dim // 2
                self.cnn_decoders.append(DecoderBlock(dim, out_channels*2, dim))
            
            # Transformer Decoder
            self.trans_decoders.append(TransformerDecoderBlock_v2(dim, out_channels))
            
            # Gated Fusion
            self.gated_fusions.append(GatedFeatureFusion(out_channels))
            
            dim = out_channels
            
        if upsampled:
            # If upsampled, add an additional cnn-trans decoder blocks to obtain the final output (with a higher resolution)
            self.cnn_decoders.append(DecoderBlock(24, config.reg_head_chan))
        # Registration Head
        self.reg_head = RegistrationHead(
            in_channels=config.reg_head_chan,
            out_channels=3,
            kernel_size=3,
        )
        '''
        # Transformers (Frozen)
        self.transformer_moving = Transformer(
            patch_size=4,
            in_channel=1,
            out_channel=48,
            dropout=0.1,
            mask_size=107520,
            mask_ratio=0.5,
            num_encoder_layers=6,
            mode="inference"
        )

        self.transformer_fixed = Transformer(
            patch_size=4,
            in_channel=1,
            out_channel=48,
            dropout=0.1,
            mask_size=107520,
            mask_ratio=0.5,
            num_encoder_layers=6,
            mode="inference"
        )
        '''
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        # Spatial Transformers
        self.spatial_trans = SpatialTransformer(config.img_size)
        self.vec_int = VecInt()
        self.reduce_channels = nn.Conv3d(in_channels=16, out_channels=2, kernel_size=1)
    '''     
        # Init and Load Pretrained
        self.apply(self._init_weights)
        self._load_pretrained_transformers("/root/miccai2025/Checkpoint_trans/transformer_checkpoint_step_3.pth")

    def _load_pretrained_transformers(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.transformer_moving.load_state_dict(checkpoint['model_state_dict'])
        self.transformer_fixed.load_state_dict(checkpoint['model_state_dict'])
        for param in self.transformer_moving.parameters():
            param.requires_grad = True
        for param in self.transformer_fixed.parameters():
            param.requires_grad = True
            
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    '''
    def forward(self, x, upsampled = False, pretraining=False, lambda_orth=1.0):
        # Feature Extraction
        source, target = x[:, 0:1], x[:, 1:2]
        # moving_repr = self.transformer_moving(source).squeeze(1)
        # fixed_repr = self.transformer_fixed(target).squeeze(1)
        # tran_repr = torch.cat([moving_repr, fixed_repr], dim=2)
        
        # Get features from both images
        src_x, src_cnn_skips, src_trans_skips = self.encoder(source)
        tgt_x, tgt_cnn_skips, tgt_trans_skips = self.encoder(target)
        
        x = torch.cat([src_x, tgt_x], dim=1)
        del src_x, tgt_x
        cnn_skips = []
        trans_skips =[]
        # Fuse features with gated fusion
        for i, (src_cnn, tgt_cnn) in enumerate(zip(src_cnn_skips, tgt_cnn_skips)):
            # Concatenate features
            cnn_skip = torch.cat([src_cnn, tgt_cnn], dim=1)
            cnn_skips.append(cnn_skip)
            # Apply gated fusion
            trans_skip = torch.cat([src_trans_skips[i], tgt_trans_skips[i]], dim=1) if i < len(src_trans_skips) and i < len(tgt_trans_skips) else None
            trans_skips.append(trans_skip)
        
        # Decoding with Gated Fusion
        for i in range(len(cnn_skips)):
            if i==3:
                x = self.cnn_decoders[i](x, cnn_skips[len(cnn_skips)-1-i])
            else:
                x_cnn = self.cnn_decoders[i](x, cnn_skips[len(cnn_skips)-1-i]) if i < len(cnn_skips) else x
                x_trans = self.trans_decoders[i](x, trans_skips[2-i]) if i < len(trans_skips) else x
                x_trans = self.up(x_trans)
                x = self.gated_fusions[i](x_cnn, x_trans)
                
        if upsampled:
            x = self.cnn_decoders[-1](x)
            
        # Registration
        flow = self.vec_int(self.reg_head(x))
        if upsampled:
            # 2. Downsample to original resolution (preserves subpixel info)
            flow = F.interpolate(
                flow, 
                size=(160, 224, 192), 
                mode='trilinear',  # For 3D
                align_corners=False
            )
        warped = self.spatial_trans(source, flow)
        
        return flow, warped, target