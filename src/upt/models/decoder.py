from __future__ import annotations
from line_profiler import profile
from typing import Optional

import torch
from torch import nn


class PerceiverDecoder(nn.Module):
    r"""
    Perceiver decoder.

    Implements a single cross-attention layer where queries come from (embedded) output
    positions of width 'H_q' and keys come from the latent tokens of width 'H'.
    A small feed-forward network refines the querypathway, and a final linear head maps to pressure.

    Parameters
    ----------
    hidden_dim : int
        Model width 'H'; must match the last dimension of the latent tokens.
    query_in_dim : int
        Input width 'H_q' of the query embeddings to be projected to 'H'.
    num_attn_heads : int, optional
        Number of attention heads for cross-attention. If 'None', uses 'max(1, hidden_dim // 64)'.
    use_query_mlp : bool, default=True
        If 'True', use a shallow MLP for the query projection..
    query_mlp_ratio : int, default=4
        Expansion factor for the query projection MLP.
    ffn_ratio : int, default=4
        Expansion factor for the post-attention feed-forward network on the query pathway.
    dropout : float, default=0.0
        Dropout used on residual branches and inside the projection/FFN layers.

    Shapes
    ------
    Latent : '(B, T, H)'
    Queries : '(B, K, H_q)'
    Output : '(B, K, 1)'
    """

    def __init__(
        self,
        hidden_dim: int,
        query_in_dim: int,
        num_attn_heads: Optional[int] = None,
        use_query_mlp: bool = True,
        query_mlp_ratio: int = 4,
        ffn_ratio: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = int(hidden_dim)
        self.query_in_dim = int(query_in_dim)
        self.num_heads = int(num_attn_heads) if num_attn_heads is not None else max(1, self.dim // 64)

        # --- Query projection ---
        if use_query_mlp:
            q_hidden = self.dim * query_mlp_ratio
            self.query_proj = nn.Sequential(
                nn.Linear(self.query_in_dim, q_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(q_hidden, self.dim),
                nn.Dropout(dropout),
            )
        else:
            if self.query_in_dim == self.dim:
                self.query_proj = nn.Identity()
            else:
                self.query_proj = nn.Sequential(
                    nn.Linear(self.query_in_dim, self.dim),
                    nn.Dropout(dropout),
                )

        # --- Perceiver ---
        self.q_norm = nn.LayerNorm(self.dim)
        self.kv_norm = nn.LayerNorm(self.dim)
        self.attn = nn.MultiheadAttention(self.dim, self.num_heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)

        # --- Feed-forward projection ---
        ffn_hidden = self.dim * ffn_ratio
        self.ffn_norm = nn.LayerNorm(self.dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.dim, ffn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, self.dim),
            nn.Dropout(dropout),
        )

        # --- Project to scalar ---
        self.head = nn.Linear(self.dim, 1)

        # ---- Truncated-normal init (std=0.02) for all Linear layers in this module ----
        def _truncnormal_(m: nn.Module) -> None:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.apply(_truncnormal_)
    @profile
    def forward(self, latent: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
        r"""
        Decodes latent representations.

        Parameters
        ----------
        latent : torch.Tensor
            Latent tokens of shape '(B, T, H)'.
        queries : torch.Tensor
            Embedded query tokens of shape '(B, K, H_q)'.

        Returns
        -------
        torch.Tensor
            Scalar predictions per query: '(B, K, 1)'.
        """
        assert latent.ndim == 3 and queries.ndim == 3, "expected (B,T,H) and (B,K,H_q)"
        B, T, H_lat = latent.shape
        Bq, K, H_q = queries.shape
        assert B == Bq, "batch size mismatch between latent and queries"
        assert H_lat == self.dim, f"latent width mismatch: got {H_lat}, expected {self.dim}"
        assert H_q == self.query_in_dim, f"queries width mismatch: got {H_q}, expected {self.query_in_dim}"

        # Project queries to model width H
        q = self.query_proj(queries)                

        # Perceiver cross-attention
        qn = self.q_norm(q)
        kv = self.kv_norm(latent)
        attn_out, _ = self.attn(qn, kv, kv, need_weights=False)   
        x = q + self.drop(attn_out)                              

        # Feed-forward pass and residual add
        y = self.ffn(self.ffn_norm(x))
        x = x + y                                                

        # Project to scalar
        out = self.head(x)                                      
        return out
