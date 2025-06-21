import torch
import torch.nn as nn
from layers.L3former_all_layers import *


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.n_scales = len(configs.window_size_list)+1  # number of total scales
        self.mask = configs.mask
        self.enc_in = configs.enc_in  # number of variables
        self.n_vars = configs.enc_in + 4 if self.mask else configs.enc_in  # number of variables with mask
        self.d_twff = configs.d_twff or configs.d_model
        self.d_vwff = configs.d_vwff or 2 * self.n_vars
        self.output_attention = configs.output_attention

        # RevIn
        self.revin = configs.revin
        if self.revin:
            self.revin_layer = RevIN(self.enc_in,
                                     affine=configs.affine,
                                     use_std_in_revin=configs.use_std_in_revin)

        # Scale-shared series-wise embedding
        self.embedding = ScaleSharedEmbedding(seq_len=configs.seq_len,
                                              d_model=configs.d_model,
                                              dropout=configs.dropout)

        # L3 based multiscale extractor
        self.multiscale_extractor = MultiScaleL3Extractor(window_size_list=configs.window_size_list,
                                                          seq_len=configs.seq_len,
                                                          dropout=configs.dropout,
                                                          mod=configs.mod,
                                                          use_pooling_init=configs.use_pooling_init)

        # Enhanced Transformer layers
        self.layers = nn.ModuleList([
            EnhancedTransformerLayer(
                d_model=configs.d_model,
                n_heads=configs.n_heads,
                n_vars=self.n_vars,
                vwff_dropout=configs.vwff_dropout,
                dropout=configs.dropout,
                d_twff=configs.d_twff,
                d_vwff=configs.d_vwff,
                use_norm_in_former=configs.use_norm_in_former,
                use_vwff=configs.use_vwff,
                use_L3Linear=configs.use_L3Linear,
                init_residual_weight_list=configs.init_residual_weight_list,
                train_residual_weight = bool(configs.train_residual_weight)
            ) for _ in range(configs.e_layers)
        ])

        # Head projector
        self.head = Head_projector(d_model=configs.d_model,
                                   pred_len=configs.pred_len,
                                   n_vars=self.n_vars,
                                   n_scales=self.n_scales,
                                   head_dropout=configs.head_dropout,
                                   flatten_mod=configs.flatten_mod)

    def forward(self, x, x_mark):
        # x: (B, L, C)
        if self.revin:
            x = self.revin_layer(x, 'norm')

        # Prepare multi-scale inputs
        if self.mask:
            if x_mark is not None:
                x = torch.cat([x, x_mark], dim=2)

        # L3 based multiscale extractor (B, C, L, G)
        x_stack = self.multiscale_extractor(x)

        # Scale-shared series-wise embedding (B*C, G, D)
        emb = self.embedding(x_stack)

        # # Enhanced transformer layers (B*C, G, D)
        x = emb
        for layer in self.layers:
            x, attn = layer(x)

        # Head projector (B, F, C)
        out = self.head(x)
        out = out.permute(0, 2, 1)[:, :, :self.enc_in]

        if self.revin:
            out = self.revin_layer(out, 'denorm')

        if not self.output_attention: return out
        else: return out, attn
