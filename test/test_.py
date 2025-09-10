import pytest
from upt.models.encoder import PositionalEmbedding, PerceiverEncoder, ConvNeXtEncoder
from upt.models.approximator import Approximator
from upt.models.decoder import PerceiverDecoder
import torch

def test_pe():

    # Upper bound should not be lower than zero
    upper = 0
    with pytest.raises(ValueError):
        pe = PositionalEmbedding(upper = upper)

    # Hidden dimension should not be lower than 6
    H = 1
    with pytest.raises(AssertionError):
        pe = PositionalEmbedding(hidden_dim=H)

    H = 2
    with pytest.raises(AssertionError):
        pe = PositionalEmbedding(hidden_dim=H)

    H = 3
    with pytest.raises(AssertionError):
        pe = PositionalEmbedding(hidden_dim=H)

    H = 4
    with pytest.raises(AssertionError):
        pe = PositionalEmbedding(hidden_dim=H)

    H = 5
    with pytest.raises(AssertionError):
        pe = PositionalEmbedding(hidden_dim=H)

    # Returns a tensor with 3 dimensions
    B = 2
    K = 2
    H = 6

    tensor = torch.randn(B, K, 3)
    pe = PositionalEmbedding(hidden_dim=H)
    rescaled = pe(tensor)
    with pytest.raises(IndexError):
        rescaled.shape[3]

def test_perceiver_enc():

    H = 150
    B = 2
    K = 2
    N = 64
    tensor = torch.randn(B, K, H)
    pe = PerceiverEncoder(hidden_dim=H, num_output_tokens=N)
    encoded = pe(tensor) 

    # Produce correct hidden dimension
    assert encoded.shape[2] == H

    # Produce correct number of tokens
    assert encoded.shape[1] == N

def test_convnet():

    # Length of 'dims' and 'depth' parameter must be matched
    dims = (89, 178, 256)
    depths = (2, 2)
    with pytest.raises(AssertionError):
        ConvNeXtEncoder(depths=depths, dims=dims)

    # Eccessive dimensionality for the input tensor
    tensor = torch.randn(1, 2, 2, 2, 2)
    cn = ConvNeXtEncoder()
    with pytest.raises(AssertionError):
        cn(tensor)

    # Non matching resolutions
    tensor = torch.randn(1, 32, 31, 32)
    with pytest.raises(AssertionError):
        cn(tensor)

    # Providing a tensor with different resolutions from the init one
    res = 40
    cn = ConvNeXtEncoder(resolution=res)
    tensor = torch.randn(1, 32, 32, 32)
    with pytest.raises(AssertionError):
        cn(tensor)

def test_approximator():

    H = 8
    depth = 4
    heads = 4
    ap = Approximator(dim=H, depth=depth, num_attn_heads=heads)

    # Expects rank 3 tensors
    tensor = torch.randn(2, 2, 2, 2)
    with pytest.raises(AssertionError):
        ap(tensor)

    tensor = torch.randn(2, 2)
    with pytest.raises(AssertionError):
        ap(tensor)

def test_decoder():

    H = 256
    H_q = 768

    dec = PerceiverDecoder(hidden_dim=H, query_in_dim=H_q)

    # Expects rank 3 tensors
    latent = torch.randn(2, 2)
    queries = torch.randn(2, 2, 2)
    with pytest.raises(AssertionError):
        dec(latent, queries)

    latent = torch.randn(2, 2, H)
    queries = torch.randn(2, 2)
    with pytest.raises(AssertionError):
        dec(latent, queries)

    # Batch sizes should match
    latent = torch.randn(3, 2, H)
    queries = torch.randn(2, 2, 2)
    with pytest.raises(AssertionError):
        dec(latent, queries)

    # Number of tokens can be different
    latent = torch.randn(2, 3, H)
    queries = torch.randn(2, 2, H_q)
    dec(latent, queries)

    # Hidden dimensions should coincide with the one declared
    latent = torch.randn(2, 2, 6)
    queries = torch.randn(2, 2, H_q)
    with pytest.raises(AssertionError):
        dec(latent, queries)

    latent = torch.randn(2, 2, H)
    queries = torch.randn(2, 2, 6)
    with pytest.raises(AssertionError):
        dec(latent, queries)