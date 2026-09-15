"""Fail-closed contracts for OLMo3, removed from native vLLM in 0.29.

Legacy source remains for reference; it is not a supported registry target.
"""

from types import SimpleNamespace

import pytest

from dmi_vllm_integration.architectures import (
    ARCHITECTURE_REMAP,
    require_supported_architecture,
)
from dmi_vllm_integration.plugin import MODEL_REGISTRATIONS


@pytest.mark.parametrize("architecture", ["Olmo3ForCausalLM", "DMIOlmo3ForCausalLM"])
def test_removed_olmo3_native_architecture_fails_closed(architecture):
    assert "Olmo3ForCausalLM" not in ARCHITECTURE_REMAP
    assert "DMIOlmo3ForCausalLM" not in MODEL_REGISTRATIONS
    config = SimpleNamespace(
        architecture=architecture,
        hf_config=SimpleNamespace(architectures=[architecture]),
    )
    with pytest.raises(RuntimeError, match="does not support"):
        require_supported_architecture(config)
