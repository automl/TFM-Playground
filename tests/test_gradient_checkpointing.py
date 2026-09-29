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