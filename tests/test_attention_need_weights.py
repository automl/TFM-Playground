import torch

from tfmplayground.models.nanotabpfn import NanoTabPFNModel


def test_output_same_with_and_without_attention_weights():
    torch.manual_seed(0)
    model = NanoTabPFNModel(
        embedding_size=32, num_attention_heads=4, mlp_hidden_size=64, num_layers=2, num_outputs=3
    ).eval()
    x = torch.randn(2, 30, 5)
    y = torch.randint(0, 3, (2, 20)).float()

    with torch.no_grad():
        out_fast = model((x, y), train_test_split_index=20)       # need_weights=False
        for block in model.transformer_blocks:
            block.save_feature_attention = True
        out_weights = model((x, y), train_test_split_index=20)    # need_weights=True

    assert torch.allclose(out_fast, out_weights, atol=1e-5)
    for block in model.transformer_blocks:
        assert block.feature_attention.shape == (6,)              # 5 features + target