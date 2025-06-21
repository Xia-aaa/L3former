import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Callable, Optional
import numpy as np

class LocalLinearLayer(nn.Module):
    """
    Local Linear Layer (L3): Applies a window linear transformation
    with shared weights across channels and independent weights across series,
    which has optional padding/stride behavior.

    Args:
        input_size (int): Number of time steps in the input sequence.
        window_size (int): Length of the local window. Default: 3.
        stride (int): Step between windows. Must be <= window_size. Default: 1.
        padding (Optional[int]): Explicit padding on each side. If None, auto-padding
            is computed when stride=1 and auto_padding=True. Default: None.
        auto_padding (bool): If True and stride=1, apply symmetric padding. Default: True.
        mod (int): Mode selector: 0 => output same length as input (symmetric padding),
            1 => valid-like output (pad only one side). Default: 0.
        use_pooling_init (bool): If True, initialize weights in window region to 1/window_size;
            otherwise retain default initialization. Default: False.
    """
    def __init__(
        self,
        input_size: int,
        window_size: int = 3,
        stride: int = 1,
        padding: int = None,
        auto_padding: bool = True,
        mod: int = 0,
        use_pooling_init: int = 0
    ):
        super().__init__()
        assert 1 <= stride <= window_size, "stride must be in [1, window_size]"

        # Save init parameters
        self.input_size = input_size
        self.window_size = window_size
        self.stride = stride
        self.mod = mod
        self.use_pooling_init = use_pooling_init
        self.auto_padding = auto_padding

        # Compute structural dimensions
        self.padding, self.padded_length, self.output_length = self._compute_dimensions(padding)

        # Define linear projection
        self.linear = nn.Linear(self.padded_length, self.output_length)

        # Define and register mask
        self.register_buffer('mask', self._make_mask())

        # Initialize weights according to mask
        self._init_weights()

        # Mask gradient flow
        self.linear.weight.register_hook(lambda grad: grad * self.mask.to(grad.device).float())

    def _compute_dimensions(self, padding: int):
        """
        Compute padding amount, padded input size, and output length.

        Returns:
            padding (int): Number of padding steps on left (and right if mod=0).
            padded_length (int): Effective input length after padding.
            output_length (int): Number of output positions (windows).
        """
        if padding is None:
            padding = (self.window_size - 1) // 2 if (self.auto_padding and self.stride == 1) else 0
        assert padding >= 0, "padding must be non-negative"

        if self.mod == 0:
            padded_length = self.input_size + 2 * padding
            output_length = self.input_size
        else:
            padded_length = self.input_size + padding
            output_length = (padded_length - self.window_size) // self.stride + 1

        return padding, padded_length, output_length

    def _make_mask(self) -> torch.BoolTensor:
        """
        Create a boolean mask where each row i has True over the window
        [i*stride : i*stride+window_size], else False.
        """
        mask = torch.zeros((self.output_length, self.padded_length), dtype=torch.bool)
        for i in range(self.output_length):
            start = i * self.stride
            end = start + self.window_size
            mask[i, start:end] = True
        return mask

    def _init_weights(self):
        """
        Initialize weights:
          - Zero out connections outside the local window (mask==False).
          - If use_pooling_init, set window weights to uniform average 1/window_size.
          - Else retain default initialization (e.g., Kaiming) inside window.
        """
        with torch.no_grad():
            self.linear.weight.data[~self.mask] = 0.0
            if self.use_pooling_init:
                self.linear.weight.data[self.mask] = 1.0 / self.window_size
            # otherwise keep original Kaiming initialization inside mask

    def _pad_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pad input by replicating edges:
          - mod=0: replicate `padding` steps at both front and end.
          - mod=1: replicate only at end for valid-like extraction.

        Args:
            x (Tensor): Input of shape (B, C, L)
        Returns:
            Padded tensor of shape (B, C, padded_length)
        """
        if self.padding > 0:
            if self.mod == 0:
                first_part = x[:, :, :self.padding]
                last_part = x[:, :, -self.padding:]
                x = torch.cat([first_part, x, last_part], dim=-1)
            else:
                last_part = x[:, :, -self.padding:]
                x = torch.cat([x, last_part], dim=-1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (Tensor): Input of shape (B, L, C).
            (B: batch size, L: number of time points, C: number of variables)
        Returns:
            Tensor of shape (B, output_length, C) after local-linear transforms.
        """
        x = x.permute(0, 2, 1)      # (B, L, C) -> (B, C, L)
        x = self._pad_input(x)
        weight = self.linear.weight * self.mask.float()
        out = F.linear(x, weight, bias=self.linear.bias)

        return out.permute(0, 2, 1)  # (B, C, L') -> (B, L', C)


class MultiScaleL3Extractor(nn.Module):
    """
    Extract multiscale local-linear features using multiple window sizes.

    Args:
        window_size_list (List[int]): List of window sizes for each scale.
        seq_len (int): Length of input sequence.
        dropout (float): Dropout probability applied after each scale.
        mod (int): Padding mode passed to LocalLinearLayer.
        use_pooling_init (bool): Whether to use uniform pooling init.
    """
    def __init__(self, window_size_list, seq_len, dropout=0., mod=0, use_pooling_init=0):
        super().__init__()
        self.window_size_list = window_size_list
        self.extractors = nn.ModuleList([
            LocalLinearLayer(input_size=seq_len,
                             window_size=w,
                             mod=mod,
                             use_pooling_init=use_pooling_init)
            for w in window_size_list
        ])
        self.L3_dropout = nn.ModuleList([
            nn.Dropout(dropout) for _ in range(len(window_size_list))
        ])

    def forward(self, x):
        """
        Forward multiscale extraction.

        Args:
            x (Tensor): Input sequence of shape (B, L, C).
        Returns:
            Tensor of shape (B, C, L, G), where G = len(window_size_list)+1.
        """
        seqs = [x]
        for extractor, drop in zip(self.extractors, self.L3_dropout):
            out = extractor(x)
            out = drop(out)
            seqs.append(out)
        stacked = torch.stack(seqs, dim=-1).permute(0,2,1,3)  # (B, C, L, G)

        return stacked


class Conv1dExtractor(nn.Module):
    """
    Single-scale 1D convolution extractor that preserves
    the number of channels and the sequence length.

    Args:
        in_channels (int): Number of input/output channels (C).
        kernel_size (int): Length of the convolution kernel (window size).
        padding (int): Padding on each side to keep output length == input length.
        dropout (float): Dropout probability applied after convolution.
    """
    def __init__(self, in_channels: int, kernel_size: int, padding: int, dropout: float):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
            bias=True
        )
        self.dropout = nn.Dropout(dropout)
        self.padding = (kernel_size - 1) // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C) -> (B, C, L)
        x = x.permute(0, 2, 1)
        first_part = x[:, :, :self.padding]
        last_part = x[:, :, -self.padding:]
        x = torch.cat([first_part, x, last_part], dim=-1)
        out = self.conv(x)           # (B, C, L)
        out = self.dropout(out)
        # back to (B, L, C)
        return out.permute(0, 2, 1)


class MultiScaleConvExtractor(nn.Module):
    """
    Extract multiscale convolutional features using multiple window sizes.
    Mirrors MultiScaleL3Extractor but replaces L3 with Conv1d.

    Args:
        window_size_list (List[int]): List of kernel sizes for each scale.
        seq_len (int): Length of input sequence (used to compute padding).
        in_channels (int): Number of input/output channels C.
        dropout (float): Dropout probability applied after each conv.
    """
    def __init__(self,
                 window_size_list: list,
                 in_channels: int,
                 dropout: float = 0.0):
        super().__init__()
        self.window_size_list = window_size_list
        self.extractors = nn.ModuleList([
            Conv1dExtractor(
                in_channels=in_channels,
                kernel_size=w,
                padding=(w - 1) // 2,
                dropout=dropout
            )
            for w in window_size_list
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input sequence of shape (B, L, C).
        Returns:
            Tensor of shape (B, C, L, G), where G = len(window_size_list)+1.
        """
        seqs = [x]
        for conv_ex in self.extractors:
            out = conv_ex(x)  # (B, L, C)
            seqs.append(out)
        stacked = torch.stack(seqs, dim=-1).permute(0, 3, 1, 2)
        # permute(0, G, L, C) -> (B, C, L, G) by swapping axes 1<->3
        stacked = stacked.permute(0, 3, 2, 1)
        return stacked


class ScaleSharedEmbedding(nn.Module):
    """
    Scale-Shared Embedding module.

    Projects the input sequence across multiple temporal scales
    into a shared embedding space (D-dim), where different scales are aligned.

    Args:
        seq_len (int): Length of input sequence (L).
        d_model (int): Dimension of output embedding (D).
        dropout (float): Dropout probability after projection. Default: 0.1
    """
    def __init__(self, seq_len, d_model, dropout=0.1):
        super(ScaleSharedEmbedding, self).__init__()
        self.series_wise_embedding = nn.Linear(seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (Tensor): Input tensor of shape (B, C, L, G),
                        where G is the number of scales.

        Returns:
            Tensor: Output embedding of shape (B*C, G, D)
        """
        emb = self.series_wise_embedding(x.permute(0, 1, 3, 2)).contiguous()  # (B, C, G, D)
        emb = self.dropout(emb)
        emb = torch.reshape(emb, (emb.shape[0] * emb.shape[1], emb.shape[2], emb.shape[3]))  # (B*C, G, D)

        return emb


class EnhancedTransformerLayer(nn.Module):
    """
    Enhanced Transformer Layer with optional components.

    This module implements three core operations for multiscale time-series:
      1. Scale-wise attention (SWAM) via multi-head attention
      2. Temporal-wise feed-forward network (TWFF)
      3. Variable-wise feed-forward network (VWFF)

    It also supports toggling of:
      - use_norm_in_former: whether to apply LayerNorm before/after TWFF
      - use_vwff: whether to include the VWFF branch
      - use_L3Linear: whether to skip attention and use custom L3Linear instead

    Args:
        d_model (int): Embedding dimension of model (D).
        n_heads (int): Number of attention heads for SWAM.
        n_vars (int): Number of variables (channels) in VWFF (C).
        vwff_dropout (float, optional): Dropout rate in VWFF branch. Default: 0.8.
        dropout (float, optional): Dropout rate after attention & TWFF. Default: 0.1.
        d_twff (Optional[int], optional): Hidden dimension for TWFF. Defaults to d_model.
        d_vwff (Optional[int], optional): Hidden dimension for VWFF. Defaults to 2 * n_vars.
        use_norm_in_former (bool, optional): LayerNorm flag around TWFF. Default: True.
        use_vwff (bool, optional): Enable VWFF branch. Default: True.
        use_L3Linear (bool, optional): Use L3Linear instead of SWAM. Default: False.
        init_residual_weight_list (List[float], optional):
            Initial scaling factors for residual branches [alpha1, alpha2, alpha3]. Default: [1.0, 1.0, 1.0].
    """
    def __init__(
            self,
            d_model: int,
            n_heads: int,
            n_vars: int,
            vwff_dropout: float = 0.8,
            dropout: float = 0.1,
            d_twff: Optional[int] = None,
            d_vwff: Optional[int] = None,
            use_norm_in_former: bool = True,
            use_vwff: bool = True,
            use_L3Linear: bool = False,
            init_residual_weight_list = [1.0, 1.0, 1.0],
            train_residual_weight: bool = True
    ):
        super().__init__()

        self.d_model = d_model
        self.d_mixing = d_vwff
        self.n_vars = n_vars
        self.use_norm_in_former = use_norm_in_former
        self.use_vwff = use_vwff
        self.use_L3Linear = use_L3Linear

        # Set internal hidden sizes
        d_twff = d_twff or d_model
        d_vwff = d_vwff or 2*n_vars

        # 1) Scale-wise Attention Mechanism (SWAM)
        # Only build if not using an external L3Linear alternative
        if not self.use_L3Linear:
            self.SWAM = _MultiheadAttention(d_model=d_model,
                                            n_heads=n_heads,
                                            proj_dropout=dropout)
            self.swam_dropout = nn.Dropout(dropout)

        # 2) Temporal-wise Feed-Forward Network (TWFF)
        self.TWFF = nn.Sequential(
            nn.Linear(d_model, d_twff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_twff, d_model)
        )
        self.twff_dropout = nn.Dropout(dropout)

        # 3) Variable-wise Feed-Forward Network (VWFF)
        # Only build if use_vwff=True
        if self.use_vwff:
            self.VWFF = nn.Sequential(
                nn.Linear(n_vars, d_vwff),
                nn.GELU(),
                nn.Dropout(vwff_dropout),
                nn.Linear(d_vwff, n_vars)
            )
            self.vwff_dropout = nn.Dropout(vwff_dropout)

        # Layer normalization layers, only if use_norm_in_former=True
        if self.use_norm_in_former:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)

        # Residual scaling parameters
        alpha_vals = init_residual_weight_list
        self.alpha1 = nn.Parameter(torch.tensor(alpha_vals[0]), requires_grad=train_residual_weight)
        self.alpha2 = nn.Parameter(torch.tensor(alpha_vals[1]), requires_grad=train_residual_weight)
        self.alpha3 = nn.Parameter(torch.tensor(alpha_vals[2]), requires_grad=train_residual_weight)

    def forward(self, x):
        """
        Forward pass through the enhanced transformer layer.

        Args:
            x (Tensor): Input features of shape (B*C, G, D).
            (B: batch size, C: number of variables, G: number of scales, D: embedding dimension)

        Returns:
            x (Tensor): Updated features of same shape.
            attn_weights (list[Tensor]): Collected attention matrices
                                         (empty if use_L3Linear=True).
        """

        attn_weights = []

        # --- 1) Scale-wise Attention (SWAM) ---
        if not self.use_L3Linear:
            x_out1, layer_attn, _ = self.SWAM(x)
            attn_weights.append(layer_attn[0, 7, :, :])
            x = x + self.alpha1 * self.swam_dropout(x_out1)

        # --- 2) Temporal-wise Feed-Forward (TWFF) ---
        if self.use_norm_in_former:
            x = self.norm1(x)
        x_out2 = self.TWFF(x)
        x = x + self.alpha2 * self.twff_dropout(x_out2)
        if self.use_norm_in_former:
            x = self.norm2(x)

        # --- 3) Variable-wise Feed-Forward (VWFF) ---
        if self.use_vwff:
            x = torch.reshape(x, (-1, self.n_vars, x.shape[-2], x.shape[-1]))
            x_out3 = x.permute(0, 3, 2, 1)
            # (B*C, G, D) -> (B, D, G, C)
            x_out3 = self.VWFF(x_out3)
            x = x + self.alpha3 * self.vwff_dropout(x_out3).permute(0, 3, 2, 1).contiguous()
            x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))

        return x, attn_weights


class Head_projector(nn.Module):
    """
    Prediction head for final output.

    Projects hidden representations into prediction horizon.
    Supports two modes:
        - Flatten mode: Flatten (D, G) into (D*G) and fully connect.
        - Separated mode: First collapse scales, then fully connect time.

    Args:
        d_model (int): Hidden dimension (D).
        pred_len (int): Output prediction length (S).
        n_scales (int): Number of scales (G).
        head_dropout (float): Dropout rate before final output.
        flatten_mod (bool): If True, flatten (D, G) before linear projection.
    """
    def __init__(self, d_model, pred_len, n_vars, n_scales, head_dropout=0., flatten_mod=1):
        super().__init__()
        self.d_model = d_model
        self.n_vars = n_vars
        self.n_scales = n_scales
        self.flatten_mode = flatten_mod
        flatten_size = n_scales * d_model

        if self.flatten_mode:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(flatten_size, pred_len)
            self.dropout = nn.Dropout(head_dropout)
        else:
            self.linear1 = nn.Linear(n_scales, 1)
            self.linear2 = nn.Linear(d_model, pred_len)
            self.dropout1 = nn.Dropout(head_dropout)
            self.dropout2 = nn.Dropout(head_dropout)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (Tensor): Input tensor of shape (B*C, D, G).

        Returns:
            Tensor: Predicted output of shape (B, C, F).
        """
        x = x.reshape(-1, self.n_vars, self.d_model, self.n_scales)  # (B, C, D, G)
        if self.flatten_mode:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        else:
            x = self.linear1(x).squeeze(-1)
            x = self.dropout2(x)
            x = self.linear2(x)
            x = self.dropout1(x)

        return x


class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=True, attn_dropout=0., proj_dropout=0.1, qkv_bias=True, lsa=False):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))

    def forward(self, Q:Tensor, K:Optional[Tensor]=None, V:Optional[Tensor]=None, prev:Optional[Tensor]=None,
                key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):

        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)       # q_s    : [bs x n_heads x max_q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)     # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)       # v_s    : [bs x n_heads x q_len x d_v]

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v) # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights


class _ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(self, q:Tensor, k:Tensor, v:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        '''
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        '''

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(q, k) * self.scale      # attn_scores : [bs x n_heads x max_q_len x q_len]

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None: attn_scores = attn_scores + prev

        # Attention mask (optional)
        if attn_mask is not None:                                     # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if key_padding_mask is not None:                              # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        # normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)                 # attn_weights   : [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)                        # output: [bs x n_heads x max_q_len x d_v]

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights


# code from https://github.com/ts-kim/RevIN, with minor modifications
class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False, use_std_in_revin=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.use_std_in_revin = use_std_in_revin
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        if self.use_std_in_revin:
            self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        if self.use_std_in_revin:
            x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        # x
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        if self.use_std_in_revin:
            x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x
