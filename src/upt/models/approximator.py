from __future__ import annotations

import torch
from torch import nn
from kappamodules.transformer import PrenormBlock
from line_profiler import profile

class Approximator(nn.Module):
    r"""
    UPT approximator: stack of pre-norm Transformer blocks.

    Parameters
    ----------
    dim : int
        Model width for all blocks.
    depth : int
        Number of Transformer blocks.
    num_attn_heads : int
        Number of attention heads per block.
    drop_path_rate : float, default=0.0
        Maximum stochastic depth probability across the stack.
    drop_path_decay : bool, default=False
        Enable a linear schedule for drop-path from first to last block.
    init_weights : {"truncnormal", "xavier_uniform", ...}, default="truncnormal"
        Initialization scheme passed to the underlying blocks.
    init_last_proj_zero : bool, default=False
        If 'True', zero-initializes the last projection in residual branches for
        extra stability.

    Attributes
    ----------
    blocks : nn.ModuleList
        List of 'depth' pre-norm Transformer blocks.

    Shapes
    ------
    Input: '(B, K, dim)'
    Output: '(B, K, dim)'
    
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        num_attn_heads: int,
        drop_path_rate: float = 0.0,
        drop_path_decay: bool = False,
        init_weights: str = "truncnormal",
        init_last_proj_zero: bool = False,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.depth = int(depth)
        self.num_heads = int(num_attn_heads)

        if self.depth > 1 and drop_path_decay:
            dpr = [drop_path_rate * i / (self.depth - 1) for i in range(self.depth)]
        else:
            dpr = [drop_path_rate for _ in range(self.depth)]

        # Build the stack of pre-norm Transformer blocks
        def make_block(drop_path: float) -> nn.Module:
            return PrenormBlock(
                dim=self.dim,
                num_heads=self.num_heads,
                drop_path=drop_path,
                init_weights=init_weights,
                init_last_proj_zero=init_last_proj_zero,
            )

        self.blocks = nn.ModuleList([make_block(dpr[i]) for i in range(self.depth)])
    @profile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Apply the Transformer stack.

        Parameters
        ----------
        x : torch.Tensor
            Input tokens of shape '(B, K, dim)'.

        Returns
        -------
        torch.Tensor
            Output tokens of shape '(B, K, dim)'.
        """
        assert x.ndim == 3, f"expected (B, K, dim), got {tuple(x.shape)}"
        for blk in self.blocks:
            x = blk(x)
        return x