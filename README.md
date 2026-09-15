# DMI vLLM integration

`DMI-vLLM-Integration` connects DMI to an unmodified official vLLM
installation. This **unreleased 0.29.0 port** targets exactly vLLM `0.29.0` and requires
DMI integration API v1, first released with DMI `1.1.0`.

Choose a [DMI release tag](https://github.com/ProjectDMX/DMI/tags) in the
range `>=v1.1.0,<v2.0.0`, then follow the `docs/install.md` shipped in that
checkout, including its native-backend build. This port was tested with DMI
`c222f18c4db55f6f2d7136b1d6608c0b540b273a` (API v1). Build the native backend
against the **same PyTorch/CUDA as vLLM 0.29.0**, not a 0.27 environment.
From the `vllm-0.29-support` integration checkout, run:

```bash
python -m pip install 'vllm==0.29.0'
python -m pip install .
```

The integration package is distributed from this source repository and its
immutable tags; it is not published to PyPI or another package registry.

Both V1 and V2 adapters are ported. Local GPU qualification is bounded to
Qwen3-0.6B, BF16, TP1; see the [0.29 audit and evidence](docs/v029-port.md).
The V2 runner can be used through vLLM's normal default, or selected explicitly
with `VLLM_USE_V2_MODEL_RUNNER=1`; V1 remains supported with
`VLLM_USE_V2_MODEL_RUNNER=0`. V2 speculative decoding is rejected before CUDA
initialization because its computed-token accounting is not yet part of this
contract. Other unsupported architectures and parallel modes are also rejected
before model execution.

The bounded Qwen3 cell passes both eager and default compilation/CUDA graphs,
including V2 AOT-cache reload. Use a **fresh version-specific**
`VLLM_CACHE_ROOT` when migrating; old shared compilation artifacts are not a
valid baseline. The eager quickstart below is a convenience, not a V2 requirement.
V2 batch-sharded sampling is rejected when `final_logits` capture is selected.

The public `dmi_vllm_integration.worker.DMXGPUWorker` entry point imports and
inherits the existing V1 worker so its subclass behavior remains intact. At
construction it reads vLLM's resolved `use_v2_model_runner` setting; V1 takes
the existing implementation without importing V2, while V2 lazily imports and
constructs the implementation isolated under `dmi_vllm_integration.v2`.
Custom V2 workers must subclass
`dmi_vllm_integration.v2.worker.DMXV2GPUWorker` directly rather than subclassing
the dynamic entry point.

## Model support

The following model families have importable implementations in this source
tree. **Only Qwen3-0.6B has local 0.29 GPU evidence**; earlier version results
are not carried forward. Other entries are candidate coverage, not a claim that
their checkpoints, quantizations, multimodal inputs or distributed modes work:

- Apertus
- DeepSeek V4 Flash (experimental)
- ERNIE 4.5 (dense)
- Gemma 3 (text)
- Gemma 4 E2B (text)
- GLM-5.2 (experimental)
- GPT-2
- GPT-OSS
- Granite 4.1
- Kimi K3 (experimental)
- Llama
- Llama 4 (experimental)
- MiniCPM 4.1 (dense)
- MiniMax-M2.7 (experimental)
- Mistral
- Phi-3.5
- Qwen2
- Qwen2-MoE
- Qwen3
- Qwen3-MoE
- Qwen3.6 (text)

Gemma 3, Gemma 4 E2B, and Qwen3.6 currently claim text inference only.
An unsupported architecture is rejected before CUDA initialization. When
top-k routing capture is selected for an MoE model, the loaded routing
backend is validated after model load and before inference; it must expose the
modular routing result consumed by fused MoE.

OLMo 3's native module was removed in upstream 0.29. Its old DMI alias is
therefore no longer registered; the port does not silently switch it to the
Transformers backend.

For offline inference, select DMI's worker through the Python API:

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3-0.6B",
    worker_cls="dmi_vllm_integration.worker.DMXGPUWorker",
    enforce_eager=True,
    additional_config={"dmx_hook_selection": "resid_pre,final_ln,token_ids,final_logits"},
)

try:
    outputs = llm.generate(
        ["The answer is"],
        SamplingParams(temperature=0.0, max_tokens=16),
    )
    print(outputs[0].outputs[0].text)
finally:
    llm.collective_rpc("stop_monitoring")
```

The `dmi_models` general plugin registers the integration's model
architectures. Online serving also requires the opt-in finalization endpoint:

```bash
export VLLM_PLUGINS=dmi_models,dmi_stop_monitoring
vllm serve Qwen/Qwen3-0.6B \
    --worker-cls dmi_vllm_integration.worker.DMXGPUWorker \
    --enforce-eager
```

After stopping external request intake, call
`POST /v1/dmi/stop_monitoring` before terminating the server. The endpoint
pauses and drains generation, invokes `stop_monitoring` on every worker, and
leaves the engine terminally paused.

```bash
curl --fail-with-body -X POST \
  'http://127.0.0.1:8000/v1/dmi/stop_monitoring?timeout=30'
```

If the server uses an API key, add
`-H "Authorization: Bearer $VLLM_API_KEY"` to that request.

The official-vLLM behavior assumed by this release is documented in
[`docs/vllm_contract.md`](docs/vllm_contract.md).
The V2 integration points, block-table decision, and qualification evidence are
summarized in [`docs/v2_model_runner.md`](docs/v2_model_runner.md).
