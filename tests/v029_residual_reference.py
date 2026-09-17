"""Independent eager residual arithmetic and value-level comparison helpers.

This is a white-box numerical oracle, not a public-API black-box assertion.
Observe norm INPUTS before fused in-place updates; never reuse a DMI HookPoint
output as the reference. GPU tensors are copied immediately to own the snapshot.
"""

import torch


def old_residual_expression(inputs):
    hidden = inputs[0]
    residual = inputs[1] if len(inputs) > 1 else None
    value = hidden if residual is None else hidden + residual
    return value.detach().cpu().clone()


def residual_key(row):
    return tuple(row[name] for name in (
        "request_id", "act_name", "layer_no", "start", "end"
    ))


def compare_residual_rows(expected, actual):
    """Require exact coverage, shape, dtype and bitwise-equal eager BF16 values."""
    def indexed(rows):
        result = {}
        for row in rows:
            key = residual_key(row)
            assert key not in result, f"duplicate residual key: {key}"
            result[key] = row["tensor"]
        assert result, "no residual reference rows"
        return result

    left, right = indexed(expected), indexed(actual)
    assert left.keys() == right.keys(), "residual key coverage mismatch"
    for key, reference in left.items():
        observed = right[key]
        assert reference.shape == observed.shape, f"residual shape mismatch: {key}"
        assert reference.dtype == observed.dtype, f"residual dtype mismatch: {key}"
        assert torch.isfinite(reference).all() and torch.isfinite(observed).all(), key
        assert torch.equal(reference, observed), f"residual values differ: {key}"
    return len(left)
