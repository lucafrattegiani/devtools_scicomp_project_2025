from __future__ import annotations
from line_profiler import profile
from typing import Tuple

import torch
from torch import nn

from .encoder import ConvNeXtEncoder, PositionalEmbedding, PerceiverEncoder
from .approximator import Approximator
from .decoder import PerceiverDecoder


class UPTSDF(nn.Module):
    """
    UPT ShapeNetCar forward pass.
    """

    def __init__(
        self,
        *,
        # --- Grid (SDF) encoder ---
        resolution: int,
        patch_size: int,
        dims: Tuple[int, ...],
        depths: Tuple[int, ...],
        kernel_size: int = 3,
        global_response_norm: bool = True,
        drop_path_rate: float = 0.0,
        drop_path_decay: bool = False,
        add_type_token: bool | None = None, 
        # --- Mesh encoding: positional embedding + Perceiver ---
        mesh_pos_embed_dim: int,
        mesh_pos_include_raw: bool = False,
        mesh_pos_upper: float = 200.0,
        mesh_num_tokens: int = 1024,
        mesh_num_heads: int = 4,
        mesh_init_weights: str = "truncnormal",
        # --- Approximator ---
        appr_depth: int = 12,
        appr_num_heads: int = 8,
        appr_drop_path_rate: float = 0.0,
        appr_drop_path_decay: bool = False,
        appr_init_weights: str = "truncnormal",
        appr_init_last_proj_zero: bool = False,
        # --- Decoder ---
        dec_num_heads: int | None = None,
        dec_ffn_ratio: int = 4,
        dec_dropout: float = 0.0,
        dec_query_in_dim: int = 3,       
        dec_use_query_mlp: bool = True,
        dec_query_mlp_ratio: int = 4,
    ) -> None:
        super().__init__()

        # ConvNeXt arguments
        convnext_kwargs = dict(
            resolution=resolution,
            patch_size=patch_size,
            dims=tuple(dims),
            depths=tuple(depths),
            kernel_size=kernel_size,
            global_response_norm=global_response_norm,
            drop_path_rate=drop_path_rate,
            drop_path_decay=drop_path_decay,
        )
        if add_type_token is not None:
            convnext_kwargs["add_type_token"] = add_type_token

        self.grid = ConvNeXtEncoder(**convnext_kwargs)
        H = int(getattr(self.grid, "hidden_dim", dims[-1]))  # last stage width

        # Mesh positional embedding
        self.mesh_pos_emb = PositionalEmbedding(
            hidden_dim=mesh_pos_embed_dim,
            include=mesh_pos_include_raw,
            upper=mesh_pos_upper,
        )

        # Mesh perceiver
        self.mesh_perceiver = PerceiverEncoder(
            hidden_dim=H,
            num_output_tokens=mesh_num_tokens,
            num_attn_heads=mesh_num_heads,
            init_weights=mesh_init_weights,
            add_type_token=True,
        )

        # Approximator 
        self.approximator = Approximator(
            dim=H,
            depth=appr_depth,
            num_attn_heads=appr_num_heads,
            drop_path_rate=appr_drop_path_rate,
            drop_path_decay=appr_drop_path_decay,
            init_weights=appr_init_weights,
            init_last_proj_zero=appr_init_last_proj_zero,
        )

        # Decoder 
        self.decoder = PerceiverDecoder(
            hidden_dim=H,
            query_in_dim=dec_query_in_dim,  
            num_attn_heads=dec_num_heads,
            use_query_mlp=dec_use_query_mlp,
            query_mlp_ratio=dec_query_mlp_ratio,
            ffn_ratio=dec_ffn_ratio,
            dropout=dec_dropout,
        )
    @profile
    def forward(self, mesh_pos: torch.Tensor, sdf: torch.Tensor) -> torch.Tensor:
        # Mesh branch
        q_mesh = self.mesh_pos_emb(mesh_pos)            # (B,P,H)
        mesh_tokens = self.mesh_perceiver(q_mesh)       # (B,Lm,H)

        # Grid branch
        sdf_tokens = self.grid(sdf)                     # (B,Ls,H)

        # Approximator
        tokens = torch.cat([mesh_tokens, sdf_tokens], dim=1)  # (B,Lm+Ls,H)
        latent = self.approximator(tokens)                    # (B,Lm+Ls,H)

        # Decoder
        pred = self.decoder(latent, mesh_pos)                 # (B,P,1)
        return pred.squeeze(-1)                               # (B,P)
    
##############################################################################################################

class UPT(nn.Module):
    """
    UPT ShapeNetCar forward pass.
    """

    def __init__(
        self,
        *,
        # --- Mesh encoding: positional embedding + Perceiver ---
        mesh_pos_embed_dim: int,
        mesh_pos_include_raw: bool = False,
        mesh_pos_upper: float = 200.0,
        mesh_num_tokens: int = 1024,
        mesh_num_heads: int = 4,
        mesh_init_weights: str = "truncnormal",
        # --- Approximator ---
        appr_depth: int = 12,
        appr_num_heads: int = 8,
        appr_drop_path_rate: float = 0.0,
        appr_drop_path_decay: bool = False,
        appr_init_weights: str = "truncnormal",
        appr_init_last_proj_zero: bool = False,
        # --- Decoder ---
        dec_num_heads: int | None = None,
        dec_ffn_ratio: int = 4,
        dec_dropout: float = 0.0,
        dec_query_in_dim: int = 3,        
        dec_use_query_mlp: bool = True,
        dec_query_mlp_ratio: int = 4,
    ) -> None:
        super().__init__()

        H = int(mesh_pos_embed_dim)

        # Mesh positional embedding
        self.mesh_pos_emb = PositionalEmbedding(
            hidden_dim=mesh_pos_embed_dim,
            include=mesh_pos_include_raw,
            upper=mesh_pos_upper,
        )

        # Mesh perceiver
        self.mesh_perceiver = PerceiverEncoder(
            hidden_dim=H,
            num_output_tokens=mesh_num_tokens,
            num_attn_heads=mesh_num_heads,
            init_weights=mesh_init_weights,
            add_type_token=True,
        )

        # Approximator 
        self.approximator = Approximator(
            dim=H,
            depth=appr_depth,
            num_attn_heads=appr_num_heads,
            drop_path_rate=appr_drop_path_rate,
            drop_path_decay=appr_drop_path_decay,
            init_weights=appr_init_weights,
            init_last_proj_zero=appr_init_last_proj_zero,
        )

        # Decoder 
        self.decoder = PerceiverDecoder(
            hidden_dim=H,
            query_in_dim=dec_query_in_dim,  # 3 by default
            num_attn_heads=dec_num_heads,
            use_query_mlp=dec_use_query_mlp,
            query_mlp_ratio=dec_query_mlp_ratio,
            ffn_ratio=dec_ffn_ratio,
            dropout=dec_dropout,
        )
    @profile
    def forward(self, mesh_pos: torch.Tensor) -> torch.Tensor:
        # Mesh branch
        q_mesh = self.mesh_pos_emb(mesh_pos)       # (B,P,H)
        tokens = self.mesh_perceiver(q_mesh)       # (B,Lm,H)

        # Approximator
        latent = self.approximator(tokens)                    # (B,Lm+Ls,H)

        # Decoder
        pred = self.decoder(latent, mesh_pos)                 # (B,P,1)
        return pred.squeeze(-1)                               # (B,P)
