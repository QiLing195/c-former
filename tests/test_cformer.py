import torch

from cformer import CFormer, CFormerConfig, SyntheticViewTask
from cformer.model import SafeEgoProjection


def test_forward_shapes_and_finite_values() -> None:
    task = SyntheticViewTask()
    model = CFormer(CFormerConfig())
    tokens, observer_tokens, observer_ids, _ = task.sample_batch(4, "cpu")
    logits = model(tokens, observer_tokens, task.selected_positions(observer_ids))
    assert logits.shape == (4, task.token_vocab_size)
    assert torch.isfinite(logits).all()


def test_op_modulator_starts_as_identity() -> None:
    model = CFormer(CFormerConfig(n_layers=1))
    modulator = model.blocks[0].attn.modulator
    assert torch.count_nonzero(modulator.weight) == 0
    assert torch.count_nonzero(modulator.bias) == 0


def test_safe_projection_preserves_primary_token() -> None:
    projection = SafeEgoProjection()
    x = torch.randn(3, 5, 8)
    positions = torch.tensor([0, 2, 4])
    projected = projection(x, positions)
    rows = torch.arange(3)
    assert torch.equal(projected[rows, positions], x[rows, positions])


def test_contrast_logits_matches_definition() -> None:
    task = SyntheticViewTask()
    model = CFormer(CFormerConfig())
    model.eval()
    tokens = torch.randint(0, task.token_vocab_size, (2, task.seq_len))
    custom = task.observer_tokens(torch.tensor([1, 2]))
    default = task.observer_tokens(torch.tensor([0, 0]))
    tau = 0.4
    with torch.no_grad():
        custom_logits = model(tokens, custom)
        default_logits = model(tokens, default)
        expected = custom_logits + tau * (custom_logits - default_logits)
        actual = model.contrast_logits(tokens, custom, default, tau)
    assert torch.allclose(actual, expected)

