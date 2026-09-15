"""Regression oracles for semantic changes in the vLLM 0.29 port."""

from importlib import import_module

import pytest
import torch
from torch import nn

from dmi_vllm_integration.dmi_api import HookPoint


@pytest.mark.parametrize("family, cls", [
    ("qwen2", "Qwen2DecoderLayer"),
    ("qwen3", "Qwen3DecoderLayer"),
    ("llama", "LlamaDecoderLayer"),
])
@pytest.mark.parametrize("initial_residual", [False, True])
def test_residual_hooks_observe_authoritative_fused_norm_outputs(family, cls, initial_residual):
    """Kill duplicate additions before RMSNorm that perturb compiled BF16 math."""
    layer_type = getattr(import_module(f"dmi_vllm_integration.models.{family}"), cls)
    layer = layer_type.__new__(layer_type)
    nn.Module.__init__(layer)
    events = []
    first_residual = torch.full((1, 4), 123.0)
    second_residual = torch.full((1, 4), 456.0)

    class Norm(nn.Module):
        def __init__(self, name, returned_residual):
            super().__init__()
            self.name, self.returned_residual = name, returned_residual

        def forward(self, value, residual=None):
            events.append(self.name)
            if residual is None:
                return value * 2
            return value * 2, self.returned_residual

    class Attention(nn.Module):
        def forward(self, *, positions, hidden_states):
            return hidden_states

    layer.input_layernorm = Norm("norm1", first_residual)
    layer.post_attention_layernorm = Norm("norm2", second_residual)
    layer.self_attn = Attention()
    layer.mlp = nn.Identity()
    observed = {}
    for name in ("resid_pre", "ln1", "attn_out", "resid_mid", "ln2", "mlp_in", "mlp_out"):
        hook = HookPoint()
        def capture(_module, _args, output, name=name):
            events.append(name)
            observed[name] = output
        hook.register_forward_hook(capture)
        setattr(layer, f"hook_{name}", hook)
    hidden = torch.ones(1, 4)
    layer(torch.tensor([0]), hidden, hidden if initial_residual else None)
    assert observed["resid_pre"] is (first_residual if initial_residual else hidden)
    assert observed["resid_mid"] is second_residual
    assert events.index("norm1") < events.index("resid_pre")
    assert events.index("norm2") < events.index("resid_mid")
