"""Supported model-architecture boundary for the vLLM integration."""

from __future__ import annotations

from typing import Any


LLAMA_COMPAT_ARCHITECTURES = frozenset(
    {
        "CwmForCausalLM",
        "InternLM3ForCausalLM",
        "IQuestCoderForCausalLM",
        "LlamaForCausalLM",
        "LLaMAForCausalLM",
    }
)


ARCHITECTURE_REMAP: dict[str, str] = {
    "ApertusForCausalLM": "DMIApertusForCausalLM",
    "DeepseekV4ForCausalLM": "DMIDeepseekV4ForCausalLM",
    "Ernie4_5ForCausalLM": "DMIErnie4_5ForCausalLM",
    "GlmMoeDsaForCausalLM": "DMIGlmMoeDsaForCausalLM",
    "Gemma3ForCausalLM": "DMIGemma3ForCausalLM",
    "Gemma4ForConditionalGeneration": "DMIGemma4ForConditionalGeneration",
    "GPT2LMHeadModel": "DMIGPT2LMHeadModel",
    "GptOssForCausalLM": "DMIGptOssForCausalLM",
    "GraniteForCausalLM": "DMIGraniteForCausalLM",
    "KimiK3ForConditionalGeneration": "DMIKimiK3ForConditionalGeneration",
    "Llama4ForConditionalGeneration": "DMILlama4ForConditionalGeneration",
    "MiniCPMForCausalLM": "DMIMiniCPMForCausalLM",
    "MiniMaxM2ForCausalLM": "DMIMiniMaxM2ForCausalLM",
    "MistralForCausalLM": "DMIMistralForCausalLM",
    "Phi3ForCausalLM": "DMIPhi3ForCausalLM",
    "Qwen2ForCausalLM": "DMIQwen2ForCausalLM",
    "Qwen2MoeForCausalLM": "DMIQwen2MoeForCausalLM",
    "Qwen3ForCausalLM": "DMIQwen3ForCausalLM",
    "Qwen3MoeForCausalLM": "DMIQwen3MoeForCausalLM",
    "Qwen3_5ForConditionalGeneration": "DMIQwen3_5ForConditionalGeneration",
    **{
        architecture: "DMILlamaForCausalLM"
        for architecture in LLAMA_COMPAT_ARCHITECTURES
    },
}
SUPPORTED_CONFIG_ARCHITECTURES = frozenset(
    (*ARCHITECTURE_REMAP, *ARCHITECTURE_REMAP.values())
)


def require_supported_architecture(model_config: Any) -> tuple[str, ...]:
    """Validate and return the architecture vLLM actually resolved.

    vLLM resolves the declared list in order and records the selected entry on
    ``ModelConfig.architecture``.  Validating whether *any* later declaration
    is supported is unsafe: an earlier resolvable upstream architecture can
    win and bypass DMI's hooked model class.
    """

    hf_config = getattr(model_config, "hf_config", None)
    architectures = getattr(hf_config, "architectures", None)
    if (
        not isinstance(architectures, (list, tuple))
        or not architectures
        or any(not isinstance(arch, str) for arch in architectures)
    ):
        raise RuntimeError(
            "DMI vLLM requires the model config to declare a supported "
            "architecture"
        )
    # Real vLLM ModelConfig instances always expose ``architecture``.  The
    # first declaration is a conservative fallback for lightweight test
    # doubles and keeps malformed/mixed lists fail-closed.
    resolved = getattr(model_config, "architecture", architectures[0])
    if not isinstance(resolved, str) or resolved not in SUPPORTED_CONFIG_ARCHITECTURES:
        supported = ", ".join(sorted(ARCHITECTURE_REMAP))
        configured = ", ".join(architectures)
        raise RuntimeError(
            "DMI vLLM does not support the architecture resolved by vLLM: "
            f"{resolved!r} (declared: {configured}). Supported architectures: "
            f"{supported}"
        )
    return (resolved,)


__all__ = [
    "LLAMA_COMPAT_ARCHITECTURES",
    "ARCHITECTURE_REMAP",
    "SUPPORTED_CONFIG_ARCHITECTURES",
    "require_supported_architecture",
]
