"""vLLM general plugin for DMI-monitored model architectures."""

from __future__ import annotations

from collections.abc import Mapping


MODEL_REGISTRATIONS: Mapping[str, str] = {
    "DMIApertusForCausalLM": (
        "dmi_vllm_integration.models.apertus:ApertusPForCausalLM"
    ),
    "DMIDeepseekV4ForCausalLM": (
        "dmi_vllm_integration.models.deepseek_v4:DeepseekV4PForCausalLM"
    ),
    "DMIErnie4_5ForCausalLM": (
        "dmi_vllm_integration.models.ernie45:Ernie4_5PForCausalLM"
    ),
    "DMIGemma3ForCausalLM": ("dmi_vllm_integration.models.gemma3:Gemma3PForCausalLM"),
    "DMIGlmMoeDsaForCausalLM": (
        "dmi_vllm_integration.models.glm_moe_dsa:GlmMoeDsaPForCausalLM"
    ),
    "DMIGemma4ForConditionalGeneration": (
        "dmi_vllm_integration.models.gemma4:Gemma4PForConditionalGeneration"
    ),
    "DMIGPT2LMHeadModel": (
        "dmi_vllm_integration.models.gpt2:GPT2PLMHeadModel"
    ),
    "DMIGptOssForCausalLM": (
        "dmi_vllm_integration.models.gpt_oss:GptOssPForCausalLM"
    ),
    "DMIGraniteForCausalLM": (
        "dmi_vllm_integration.models.granite:GranitePForCausalLM"
    ),
    "DMIKimiK3ForConditionalGeneration": (
        "dmi_vllm_integration.models.kimi_k3:KimiK3PForConditionalGeneration"
    ),
    "DMILlama4ForConditionalGeneration": (
        "dmi_vllm_integration.models.mllama4:Llama4PForConditionalGeneration"
    ),
    "DMILlamaForCausalLM": (
        "dmi_vllm_integration.models.llama:LlamaPForCausalLM"
    ),
    "DMIMiniCPMForCausalLM": (
        "dmi_vllm_integration.models.minicpm:MiniCPMPForCausalLM"
    ),
    "DMIMiniMaxM2ForCausalLM": (
        "dmi_vllm_integration.models.minimax_m2:MiniMaxM2PForCausalLM"
    ),
    "DMIMistralForCausalLM": (
        "dmi_vllm_integration.models.mistral:MistralPForCausalLM"
    ),
    "DMIPhi3ForCausalLM": (
        "dmi_vllm_integration.models.phi3:Phi3PForCausalLM"
    ),
    "DMIQwen2ForCausalLM": (
        "dmi_vllm_integration.models.qwen2:Qwen2PForCausalLM"
    ),
    "DMIQwen2MoeForCausalLM": (
        "dmi_vllm_integration.models.qwen2_moe:Qwen2MoePForCausalLM"
    ),
    "DMIQwen3ForCausalLM": (
        "dmi_vllm_integration.models.qwen3:Qwen3PForCausalLM"
    ),
    "DMIQwen3MoeForCausalLM": (
        "dmi_vllm_integration.models.qwen3_moe:Qwen3MoePForCausalLM"
    ),
    "DMIQwen3_5ForConditionalGeneration": (
        "dmi_vllm_integration.models.qwen3_5:Qwen3_5PForConditionalGeneration"
    ),
}


def register() -> None:
    """Validate the runtime, then assert DMI's package-owned aliases."""

    from dmi_vllm_integration.compat import require_compatible_runtime

    require_compatible_runtime()

    from vllm import ModelRegistry

    for architecture, target in MODEL_REGISTRATIONS.items():
        # These names are DMI-owned aliases. Re-registering the same lazy
        # targets is idempotent, and deterministically replaces a conflicting
        # third-party registration instead of silently remapping to it.
        ModelRegistry.register_model(architecture, target)


def register_models() -> None:
    """Compatibility spelling for direct callers; prefer :func:`register`."""

    register()
