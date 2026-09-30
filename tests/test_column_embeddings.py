import torch

from tfmplayground.models.nanotabpfn import NanoTabPFNModel


def _model():
    torch.manual_seed(0)
    return NanoTabPFNModel(embedding_size=16, num_attention_heads=2, mlp_hidden_size=32, num_layers=1, num_outputs=3)


def test_column_embeddings_are_per_column_and_shared_across_rows():
    """Each column gets its own vector, added identically to every row of that column."""
    model = _model()
    x = torch.zeros(1, 3, 4, 16)  # (B, R, C, E)

    out = model.add_column_embeddings(x)

    assert out.shape == x.shape
    assert torch.equal(out[0, 0], out[0, 1])            # same vectors in every row
    assert not torch.allclose(out[0, 0, 0], out[0, 0, 1])  # different vectors per column


def test_column_embeddings_are_deterministic():
    """The vectors come from a fixed seed: two calls give exactly the same result."""
    model = _model()
    x = torch.randn(2, 5, 7, 16)

    assert torch.equal(model.add_column_embeddings(x), model.add_column_embeddings(x))
