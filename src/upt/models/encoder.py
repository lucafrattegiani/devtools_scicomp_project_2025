from __future__ import annotations
import inspect
from typing import Tuple, Optional
from line_profiler import profile
import torch
import torch.nn as nn
import einops

from kappamodules.layers.continuous_sincos_embed import ContinuousSincosEmbed
from kappamodules.transformer import PerceiverPoolingBlock
from kappamodules.convolution import ConvNext


class PositionalEmbedding(nn.Module):
    r"""
    Rescale 3D positions to '[0, upper]' and embed with 'ContinuousSincosEmbed'.


    Parameters
    ----------
    hidden_dim : int
        Output dimension for embedding .
    include : bool, default=False
        If supported by the installed 'ContinuousSincosEmbed' signature, controls whether the
        raw coordinates are concatenated to the embedding.
    upper : float, default=200.0
        Upper bound used for rescaling (lower bound is 0).
    eps : float, default=1e-9
        Prevents division by zero.


    Attributes
    ----------
    pe : nn.Module
        Instantiated 'ContinuousSincosEmbed' with a signature compatible with the installed
        KappaModules version.
    upper : float
        Upper bound for rescaling.
    eps : float
        Epsilon used in division.


    Shapes
    ------
    Input: '(B, K, 3)' or '(K, 3)'
    Output: '(B, K, H)' where 'H = hidden_dim'
    """
    def __init__(
        self,
        hidden_dim: int = 256,
        include: bool = False,
        upper: float = 200.0,
        eps: float = 1e-9,
    ):
        super().__init__()
        if upper <= 0:
            raise ValueError("Upper bound should be greater than 0")
        else:
            self.upper = float(upper)
        self.eps = eps

        # CHECKOUT: Inspect the signature to adapt to the installed kappamodules version
        sig = inspect.signature(ContinuousSincosEmbed)
        params = sig.parameters
        kwargs = {"include": include} if "include" in params else {}

        if "ndim" in params:
            self.pe = ContinuousSincosEmbed(hidden_dim, ndim=3, **kwargs)
        elif "in_features" in params:
            self.pe = ContinuousSincosEmbed(hidden_dim, in_features=3, **kwargs)
        elif "input_dims" in params:
            self.pe = ContinuousSincosEmbed(hidden_dim, input_dims=3, **kwargs)
        else:
            self.pe = ContinuousSincosEmbed(hidden_dim)

    @staticmethod
    def rescale(x: torch.Tensor, hi: float = 200, eps: float = 10**-6) -> torch.Tensor:
        r"""
        Rescaling to '[0, hi]'.


        Parameters
        ----------
        x : torch.Tensor
            Input coordinates of shape '(B, K, 3)'.
        hi : float
            Upper bound for the rescaled range (lower bound is 0).
        eps : float
            Prevents division by zero.


        Returns
        -------
        torch.Tensor
        """
        xmin = x.amin(dim=1, keepdim=True)      # Extract the minimum per sample in all 3D
        xmax = x.amax(dim=1, keepdim=True)      # Extract the maximum per sample in all 3D
        span = (xmax - xmin).clamp_min(eps)     # Extract the span per sample in all 3D
        scale = span.amax(dim=2, keepdim=True)  # Extract the scale per sample in all 3D
        x01 = (x - xmin) / scale
        return x01 * hi
    
    @profile
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        r"""
        Embed rescaled 3D positions.


        Parameters
        ----------
        positions : torch.Tensor
            Coordinates with shape '(B, K, 3)' or '(K, 3)'.


        Returns
        -------
        torch.Tensor
            Positional embeddings of shape '(B, K, H)'.
        """
        # Adjust dimensions if one sample only is provided
        if positions.dim() == 2:  # (K,3) -> (1,K,3)
            positions = positions.unsqueeze(0)
        
        # Rescale
        x = self.rescale(positions, hi=self.upper, eps=self.eps)

        # Apply positional encoding
        return self.pe(x)  # Expect (B, K, H)

###########################################################################################################

class PerceiverEncoder(nn.Module):
    r"""
    Perceiver encoder.


    Pools a set of input tokens into learnable query tokens using cross-attention,
    producing a fixed-size latent set regardless of input length.


    Parameters
    ----------
    hidden_dim : int, default=256
        Model width 'H'; must match the last dimension of the input tokens.
    num_output_tokens : int, default=64
        Number of latent/query tokens 'L' to output.
    num_attn_heads : int or None, default=None
        Number of attention heads. If 'None', uses 'max(1, hidden_dim // 64)'.
    init_weights : {"truncnormal", "xavier_uniform", ...}, default="truncnormal"
        Passed to the underlying KappaModules Perceiver block.
    add_type_token : bool, default=True
        If 'True', adds a learned '(1, 1, H)' vector to each output token to encode type.


    Shapes
    ------
    Input 'tokens': '(B, K, H)'
    Output: '(B, L, H)'


    Raises
    ------
    AssertionError
        If the input does not have rank 3 or if 'H != hidden_dim'.
    """
    def __init__(
        self,
        hidden_dim: int = 256,
        num_output_tokens: int = 64,
        num_attn_heads: Optional[int] = None,
        init_weights: str = "truncnormal",
        add_type_token: bool = True,
    ):
        super().__init__()
        self.dim = int(hidden_dim)
        self.L = int(num_output_tokens)
        self.num_heads = int(num_attn_heads) if num_attn_heads is not None else max(1, hidden_dim // 64)

        # Perceiver pooling block
        self.block = PerceiverPoolingBlock(
            dim=self.dim,
            num_heads=self.num_heads,
            num_query_tokens=self.L,
            perceiver_kwargs=dict(
                init_weights=init_weights,
            ),
        )

        # Learnable vector to signal the token type
        self.type_token = nn.Parameter(torch.empty(1, 1, self.dim)) if add_type_token else None
        if self.type_token is not None:
            nn.init.trunc_normal_(self.type_token, std=0.02)
    @profile
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        r"""
        Perceiver step.


        Parameters
        ----------
        tokens : torch.Tensor
            Input token set of shape '(B, K, H)'.


        Returns
        -------
        torch.Tensor
            Latent/query tokens of shape '(B, L, H)'.
        """

        # Assert valid input dimensions
        assert tokens.ndim == 3, f"expected (B,K,H), got {tuple(tokens.shape)}"
        B, K, H = tokens.shape
        assert H == self.dim, f"hidden_dim mismatch: tokens H={H}, expected {self.dim}"

        # Single Perceiver pooling step
        Z = self.block(kv=tokens)   # Shape (B, L, H)

        # Add grid type embedding
        if self.type_token is not None:
            Z = Z + self.type_token

        return Z

###########################################################################################################

class ConvNeXtEncoder(nn.Module):
    r"""
    3D ConvNeXt-v2 encoder for SDF grids.


    Converts a cubic SDF grid into a sequence of grid tokens by applying a 3D ConvNeXt backbone
    and flattening the final feature volume.


    Parameters
    ----------
    resolution : int, default=32
        Input grid side length. Must be divisible by 'patch_size * 2**(S-1)' where
        'S = len(depths)'.
    patch_size : int, default=2
        Patch size given to the ConvNeXt stem. Controls the final token count together with the
        number of stages.
    dims : tuple of int, default=(192, 384, 768)
        Channel width per stage.
    depths : tuple of int, default=(2, 2, 2)
        Number of ConvNeXt blocks in each stage. Must have the same length as 'dims'.
    kernel_size : int, default=3
        Spatial kernel size of the depthwise convolution in each block (stride=1, padding=floor(k/2)).
    global_response_norm : bool, default=True
        Use GRN in the channel MLP path (ConvNeXt-v2).
    drop_path_rate : float, default=0.0
        Maximum stochastic depth probability across residual blocks (0 disables).
    drop_path_decay : bool, default=False
        If 'True', linearly increases drop-path from 0 to 'drop_path_rate' across blocks.
    add_type_token : bool, default=True
        If 'True', adds a learned '(1, 1, H)' vector to each grid token to encode type.


    Attributes
    ----------
    num_output_tokens : int
        Final number of tokens 'T = (R / (patch_size * 2**(S-1)))**3'.
    hidden_dim : int
        Token width (last stage channels).


    Shapes
    ------
    Input 'sdf': '(B, R, R, R)'
    Output: '(B, T, H)'.
    """

    def __init__(
        self,
        resolution: int = 32,
        patch_size: int = 2,
        dims: tuple[int, int, int] = (192, 384, 768),
        depths: tuple[int, int, int] = (2, 2, 2),
        kernel_size: int = 3,
        global_response_norm: bool = True,
        drop_path_rate: float = 0.0,
        drop_path_decay: bool = False,
        add_type_token: bool = True,
    ):
        super().__init__()
        self.resolution = int(resolution) # Used resolution
        self.patch_size = int(patch_size) # Determine the final number of tokens
        self.dims = list(dims) # Channel dimensionality per stage
        self.depths = list(depths) # Number of ConvNeXt blocks in each stage

        # ConvNeXt encoder
        self.model = ConvNext(
            patch_size=self.patch_size,
            input_dim=1,
            dims=self.dims,
            depths=self.depths,
            ndim=3,
            drop_path_rate=drop_path_rate,
            drop_path_decay=drop_path_decay,
            kernel_size=kernel_size,
            depthwise=False,
            global_response_norm=global_response_norm,
        )

        # Check if valid values have been provided
        num_stages = len(self.depths)
        denom = (2 ** (num_stages - 1)) * self.patch_size
        assert self.resolution % denom == 0, (
            f"resolution={self.resolution} must be divisible by patch_size*2**(S-1)={denom}"
        )

        self.out_side = self.resolution // denom
        self.num_output_tokens = self.out_side ** 3  # T
        self.hidden_dim = self.dims[-1]              # H 

        # Learnable vector to signal the token type
        self.type_token = nn.Parameter(torch.empty(1, 1, self.hidden_dim)) if add_type_token else None
        if self.type_token is not None:
            nn.init.trunc_normal_(self.type_token, std=0.02)
    @profile
    def forward(self, sdf: torch.Tensor) -> torch.Tensor:
        r"""
        Encode an SDF grid into grid tokens.


        Parameters
        ----------
        sdf : torch.Tensor
            SDF values with shape '(B, R, R, R)' where 'R == resolution'.


        Returns
        -------
        torch.Tensor
            Grid tokens of shape '(B, T, H)'.
        """
        # Validate shape and resolution
        assert sdf.ndim == 4, f"expected (B,R,R,R), got {tuple(sdf.shape)}"
        B, Rx, Ry, Rz = sdf.shape
        assert Rx == Ry == Rz == self.resolution, (
            f"expected cubic grid with side={self.resolution}, got {(Rx, Ry, Rz)}"
        )

        # 1) Add channel axis for 3D convs (channels-first)  
        x = sdf.unsqueeze(1)                                   # (B, 1, R, R, R)

        # 2) Apply ConvNeXt-v2
        x = self.model(x)                                      # (B, H, S, S, S), S=self.out_side

        # 3) Flatten spatial grid into tokens
        tokens = einops.rearrange(x, "b c d h w -> b (d h w) c")  # (B, T, H), T = S*S*S

        # 4) Add grid type embedding (broadcast add)
        if self.type_token is not None:
            tokens = tokens + self.type_token                   # (B, T, H)

        return tokens
