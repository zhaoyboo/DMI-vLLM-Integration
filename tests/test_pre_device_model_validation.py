"""CPU contracts for DMI-specific validation before CUDA initialization."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.worker.gpu_worker import Worker

from dmi_vllm_integration.adapter import DMXGPUWorker
import dmi_vllm_integration.model_validation as model_validation


pytestmark = pytest.mark.vllm


@pytest.mark.parametrize(
    (
        "architecture",
        "module_name",
        "validator_name",
        "positional_sources",
        "keyword_sources",
    ),
    [
        (
            "MiniCPMForCausalLM",
            "dmi_vllm_integration.models.minicpm",
            "_require_supported_minicpm_config",
            ("hf_config",),
            (),
        ),
        (
            "MistralForCausalLM",
            "dmi_vllm_integration.models.mistral",
            "_reject_unsupported_mistral_branches",
            ("vllm_config",),
            (),
        ),
        (
            "Qwen3MoeForCausalLM",
            "dmi_vllm_integration.models.qwen3_moe",
            "_require_supported_qwen3_moe_config",
            ("hf_config",),
            (),
        ),
        (
            "Qwen3_5ForConditionalGeneration",
            "dmi_vllm_integration.models.qwen3_5",
            "_require_supported_qwen36_config",
            ("hf_config",),
            (),
        ),
        (
            "GlmMoeDsaForCausalLM",
            "dmi_vllm_integration.models.glm_moe_dsa",
            "_require_supported_glm52_config",
            ("hf_config",),
            ("dtype", "use_mla"),
        ),
        (
            "Gemma4ForConditionalGeneration",
            "dmi_vllm_integration.models.gemma4",
            "_require_supported_gemma4_e2b_config",
            ("hf_config", "parallel_config"),
            ("kv_sharing_fast_prefill",),
        ),
        (
            "Llama4ForConditionalGeneration",
            "dmi_vllm_integration.models.llama4",
            "_require_supported_llama4_scout_config",
            ("hf_config",),
            (),
        ),
    ],
)
def test_dispatch_uses_only_inputs_needed_by_dmi_boundaries(
    monkeypatch,
    architecture,
    module_name,
    validator_name,
    positional_sources,
    keyword_sources,
):
    hf_config = object()
    parallel_config = object()
    dtype = object()
    calls = []
    imports = []

    def validator(*args, **kwargs):
        calls.append((args, kwargs))

    def fake_import(name):
        imports.append(name)
        assert name == module_name
        return SimpleNamespace(**{validator_name: validator})

    monkeypatch.setattr(model_validation, "import_module", fake_import)
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=hf_config,
            dtype=dtype,
            use_mla="mla",
        ),
        parallel_config=parallel_config,
        cache_config=SimpleNamespace(
            kv_sharing_fast_prefill="kv-sharing",
        ),
    )

    model_validation.validate_model_specific_config(config, [architecture])

    source_values = {
        "vllm_config": config,
        "hf_config": hf_config,
        "parallel_config": parallel_config,
        "dtype": dtype,
        "use_mla": "mla",
        "kv_sharing_fast_prefill": "kv-sharing",
    }
    expected_args = tuple(source_values[name] for name in positional_sources)
    expected_kwargs = {name: source_values[name] for name in keyword_sources}
    assert imports == [module_name]
    assert calls == [(expected_args, expected_kwargs)]


@pytest.mark.parametrize(
    "upstream",
    sorted(model_validation._UPSTREAM_VALIDATION_SPECS),
)
def test_dmi_alias_uses_the_same_validator_spec(upstream):
    alias = model_validation.ARCHITECTURE_REMAP[upstream]
    assert model_validation._VALIDATION_SPECS[alias] is (
        model_validation._VALIDATION_SPECS[upstream]
    )


_NO_DMI_SPECIFIC_VALIDATOR = {
    "GPT2LMHeadModel",
    "LlamaForCausalLM",
    "Qwen2ForCausalLM",
    "Qwen2MoeForCausalLM",
    "Qwen3ForCausalLM",
    "ApertusForCausalLM",
    "DeepseekV4ForCausalLM",
    "Ernie4_5ForCausalLM",
    "Gemma3ForCausalLM",
    "GptOssForCausalLM",
    "GraniteForCausalLM",
    "KimiK3ForConditionalGeneration",
    "MiniMaxM2ForCausalLM",
    "Phi3ForCausalLM",
}


def test_models_without_dmi_boundaries_do_not_import_validator_modules(
    monkeypatch,
):
    def fail_import(_name):
        raise AssertionError("validation imported a model without a DMI boundary")

    monkeypatch.setattr(model_validation, "import_module", fail_import)
    for architecture in sorted(_NO_DMI_SPECIFIC_VALIDATOR):
        model_validation.validate_model_specific_config(
            SimpleNamespace(),
            [architecture],
        )
        alias = model_validation.ARCHITECTURE_REMAP[architecture]
        model_validation.validate_model_specific_config(
            SimpleNamespace(),
            [alias],
        )


def test_exact_release_cell_machinery_is_removed():
    assert not hasattr(model_validation, "_PR1_CONFIG_FINGERPRINTS")
    assert not hasattr(model_validation, "_PR1_ROPE_FIELDS")
    assert not hasattr(model_validation, "_require_pr1_release_cell")


def _worker_config(hf_config, *, dtype=torch.bfloat16):
    architecture = hf_config.architectures[0]
    return SimpleNamespace(
        model_config=SimpleNamespace(
            runner_type="generate",
            model_impl="auto",
            enable_prompt_embeds=False,
            hf_config=hf_config,
            dtype=dtype,
            use_mla=True,
            architecture=architecture,
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            use_ubatching=False,
            enable_elastic_ep=False,
        ),
        cache_config=SimpleNamespace(kv_sharing_fast_prefill=False),
        additional_config={"dmx_hook_selection": "resid_pre"},
        speculative_config=None,
    )


@pytest.mark.parametrize(
    ("architecture", "hf_config", "match"),
    [
        (
            "MiniCPMForCausalLM",
            SimpleNamespace(
                architectures=["MiniCPMForCausalLM"],
                num_experts=1,
            ),
            "MoE branch",
        ),
        (
            "MistralForCausalLM",
            SimpleNamespace(
                architectures=["MistralForCausalLM"],
                llama_4_scaling={"beta": 1.0},
                ada_rms_norm_t_cond=False,
            ),
            "llama_4_scaling",
        ),
    ],
)
def test_real_model_boundary_rejection_precedes_worker_device_init(
    monkeypatch,
    architecture,
    hf_config,
    match,
):
    initialized = []
    monkeypatch.setattr(Worker, "init_device", lambda _self: initialized.append(True))
    worker = DMXGPUWorker.__new__(DMXGPUWorker)
    worker.use_v2_model_runner = False
    worker.vllm_config = _worker_config(hf_config)
    worker.vllm_config.model_config.architecture = architecture

    with pytest.raises(NotImplementedError, match=match):
        worker.init_device()

    assert initialized == []


@pytest.mark.parametrize(
    ("architecture", "hf_config", "dtype", "use_mla", "match"),
    [
        (
            "Qwen3MoeForCausalLM",
            SimpleNamespace(
                decoder_sparse_step=2,
                mlp_only_layers=[],
            ),
            torch.bfloat16,
            True,
            "every decoder layer",
        ),
        (
            "Qwen3_5ForConditionalGeneration",
            SimpleNamespace(
                text_config=SimpleNamespace(
                    model_type="qwen3_5_text",
                    layer_scale=True,
                )
            ),
            torch.bfloat16,
            True,
            "layer scaling",
        ),
        (
            "GlmMoeDsaForCausalLM",
            SimpleNamespace(),
            torch.float16,
            True,
            "FP16",
        ),
        (
            "GlmMoeDsaForCausalLM",
            SimpleNamespace(),
            torch.bfloat16,
            False,
            "MLA",
        ),
        (
            "GlmMoeDsaForCausalLM",
            SimpleNamespace(llama_4_scaling={"beta": 0.1}),
            torch.bfloat16,
            True,
            "llama_4_scaling=None",
        ),
        (
            "Gemma4ForConditionalGeneration",
            SimpleNamespace(
                text_config=SimpleNamespace(enable_moe_block=True),
            ),
            torch.bfloat16,
            True,
            "enable_moe_block=False",
        ),
        (
            "Llama4ForConditionalGeneration",
            SimpleNamespace(
                text_config=SimpleNamespace(interleave_moe_layer_step=2),
            ),
            torch.bfloat16,
            True,
            "every decoder layer",
        ),
    ],
)
def test_each_retained_model_boundary_has_a_negative_regression(
    architecture,
    hf_config,
    dtype,
    use_mla,
    match,
):
    config = _worker_config(
        SimpleNamespace(architectures=[architecture]),
        dtype=dtype,
    )
    config.model_config.hf_config = hf_config
    config.model_config.use_mla = use_mla

    with pytest.raises(NotImplementedError, match=match):
        model_validation.validate_model_specific_config(
            config,
            [architecture],
        )
