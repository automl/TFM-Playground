import torch
import torch.nn.functional as F

from tfmplayground.models.nanotabpfn import NanoTabPFNModel


def test_gradient_checkpointing_same_loss_and_gradients():
    torch.manual_seed(0)
    model = NanoTabPFNModel(
        embedding_size=32, num_attention_heads=4, mlp_hidden_size=64, num_layers=2, num_outputs=3
    ).train()
    x = torch.randn(2, 30, 5)
    y = torch.randint(0, 3, (2, 20)).float()
    targets = torch.randint(0, 3, (2 * 10,))

    def run(use_checkpointing):
        model.zero_grad(set_to_none=True)
        model.gradient_checkpointing = use_checkpointing
        out = model((x, y), train_test_split_index=20)
        loss = F.cross_entropy(out.reshape(-1, 3), targets)
        loss.backward()
        return loss.detach(), [p.grad.clone() for p in model.parameters()]

    loss_plain, grads_plain = run(False)
    loss_ckpt, grads_ckpt = run(True)

    assert torch.allclose(loss_plain, loss_ckpt)
    for g_plain, g_ckpt in zip(grads_plain, grads_ckpt):
        assert torch.allclose(g_plain, g_ckpt, atol=1e-5)


def _record_block_input_dtypes(model):
    """Wraps every block's forward to record the dtype of the input it receives."""
    dtypes = []
    for block in model.transformer_blocks:
        original = block.forward

        def recording(src, *args, _original=original, **kwargs):
            dtypes.append(src.dtype)
            return _original(src, *args, **kwargs)

        block.forward = recording
    return dtypes


def _small_model():
    torch.manual_seed(0)
    return NanoTabPFNModel(
        embedding_size=32, num_attention_heads=4, mlp_hidden_size=64, num_layers=3, num_outputs=3
    ).train()


def test_checkpointing_under_autocast_stores_block_inputs_in_autocast_dtype():
    """With checkpointing and bf16 autocast, every block receives (and so checkpoints) a bf16
    input instead of the fp32 LayerNorm output, and the backward pass still works."""
    model = _small_model()
    model.gradient_checkpointing = True
    dtypes = _record_block_input_dtypes(model)
    x = torch.randn(2, 30, 5)
    y = torch.randint(0, 3, (2, 20)).float()

    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = model((x, y), train_test_split_index=20)

    assert len(dtypes) == 3 and all(d == torch.bfloat16 for d in dtypes)
    out.float().sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_checkpointing_without_autocast_keeps_fp32_block_inputs():
    """Without autocast nothing changes: block inputs stay fp32."""
    model = _small_model()
    model.gradient_checkpointing = True
    dtypes = _record_block_input_dtypes(model)
    x = torch.randn(2, 30, 5)
    y = torch.randint(0, 3, (2, 20)).float()

    model((x, y), train_test_split_index=20)

    assert len(dtypes) == 3 and all(d == torch.float32 for d in dtypes)
