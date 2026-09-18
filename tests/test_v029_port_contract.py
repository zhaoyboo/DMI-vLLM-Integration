"""M04/M11: version adaptation must preserve existing observation boundaries."""

import ast
from importlib import import_module
from pathlib import Path

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
@pytest.mark.parametrize("enabled", [False, True])
def test_residual_hooks_preserve_pre_port_placement(family, cls, initial_residual, enabled):
    """Reject moving pre-norm taps onto fused outputs during a version bump."""
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
        hook.enabled = enabled
        def capture(_module, _args, output, name=name):
            events.append(name)
            observed[name] = output
        hook.register_forward_hook(capture)
        setattr(layer, f"hook_{name}", hook)
    hidden = torch.ones(1, 4)
    result, residual = layer(torch.tensor([0]), hidden, hidden if initial_residual else None)
    torch.testing.assert_close(result, hidden * 4)
    assert residual is second_residual
    if enabled:
        torch.testing.assert_close(observed["resid_pre"], hidden * (2 if initial_residual else 1))
        assert events.index("resid_pre") < events.index("norm1")
        if initial_residual:
            assert observed["resid_pre"].data_ptr() != hidden.data_ptr()
        if family != "llama":
            expected_mid = hidden * 2 + (first_residual if initial_residual else hidden)
            torch.testing.assert_close(observed["resid_mid"], expected_mid)
            assert observed["resid_mid"] is not second_residual
            assert events.index("resid_mid") < events.index("norm2")
    else:
        assert "resid_pre" not in observed
        if family != "llama":
            assert "resid_mid" not in observed
    if family == "llama":
        # Llama's mid tap was already after norm at 420bafb. Do not change it
        # while reverting the pre/final tap changes introduced by this PR.
        assert observed["resid_mid"] is second_residual
        assert events.index("norm2") < events.index("resid_mid")


@pytest.mark.parametrize("family, cls", [
    ("qwen2", "Qwen2Model"),
    ("qwen3", "Qwen3Model"),
    ("llama", "LlamaModel"),
])
def test_final_residual_tap_keeps_guarded_add_before_norm(family, cls):
    """M11: cover final taps without invoking the model's compile decorator."""
    path = Path(__file__).parents[1] / "src/dmi_vllm_integration/models" / f"{family}.py"
    model = next(node for node in ast.parse(path.read_text()).body
                 if isinstance(node, ast.ClassDef) and node.name == cls)
    forward = next(node for node in model.body
                   if isinstance(node, ast.FunctionDef) and node.name == "forward")
    calls = [node for node in ast.walk(forward) if isinstance(node, ast.Call)]
    hook = next(node for node in calls if ast.unparse(node.func) == "self.hook_resid_final")
    norm = next(node for node in calls if ast.unparse(node.func) == "self.norm")
    assert ast.unparse(hook.args[0]) == "hidden_states + residual"
    assert hook.lineno < norm.lineno
    assert any(isinstance(node, ast.If)
               and ast.unparse(node.test) == "self.hook_resid_final.enabled"
               and hook in list(ast.walk(node)) for node in ast.walk(forward))
