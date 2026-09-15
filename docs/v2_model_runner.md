# vLLM V2 model-runner audit

Historical 0.27.1 evidence only. For the current port and changed signatures,
see [the 0.29.0 audit](v029-port.md).

This integration targets official vLLM `v0.27.1` at
`6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`. The V2 work was audited from
integration base `298956dcddb1a2937f4ff4cdb0703a86a80067b9`.

## Per-step field map

| DMI field | V1 source | V2 source | Validation |
|---|---|---|---|
| packed request IDs | `model_runner.input_batch.req_ids` | returned `InputBatch.req_ids` | unique strings; membership equals scheduler mapping |
| request count | `input_batch.num_reqs` | `InputBatch.num_reqs` | all aligned arrays have the same length |
| scheduled tokens | `_prepare_inputs` argument | `InputBatch.num_scheduled_tokens` | each count equals the scheduler value; sum equals scheduler total |
| computed tokens | `input_batch.num_computed_tokens_cpu` | `InputBatch.num_computed_tokens_np` | non-negative and snapshotted before forward |
| execution rows and graph mode | `_determine_batch_execution_and_padding` result | `cudagraph_manager.dispatch` result passed to `prepare_inputs` | descriptor rows are no shorter than real packed rows |
| token-ID dtype | `input_ids.gpu.dtype` | `input_buffers.input_ids.dtype` | absent only where the token-ID hook is not owned |

V2 sorts request IDs while preparing its `InputBatch`; scheduler dictionary
order is therefore never used as tensor order. DMI snapshots the returned CPU
and NumPy arrays, derives contiguous request row ranges, and commits the step
before `prepare_attn`, `model_state.prepare_attn`, or model forward can run.

The V1 `_is_uniform_decode` and `uniform_decode_query_len` helpers are not used
for V2. V2 computes `uniform_token_count` and selects its concrete
`BatchExecutionDescriptor` before `prepare_inputs`, so DMI observes that real
descriptor instead of reproducing the decision.

## Block-table decision

V2 stages block-table rows in GPU memory and retains only block counts on the
CPU. DMI 1.x does not put block IDs or block-table contents in `StepContext`,
record metadata, or any hook shape. No block-table mirror is needed for the
current integration, and official vLLM remains unmodified.

If a future DMI schema adds a block-dependent field, support is conditional on
adding a CPU shadow where V2 appends block IDs, checking its row lengths against
the authoritative counts at emit time, and degrading the new field on any
mismatch. It must not infer block contents from counts.

## Supported and excluded paths

The public worker entry point imports and inherits the existing V1 worker to
preserve its subclass behavior. It uses vLLM's resolved
`VllmConfig.use_v2_model_runner` property, preserving vLLM's architecture and
feature-dependent default. A V1 construction performs the neutral selection
check without importing V2; a V2 construction lazily imports the isolated
`dmi_vllm_integration.v2` implementation. Custom V2 workers must subclass
`dmi_vllm_integration.v2.worker.DMXV2GPUWorker` directly.

| Path | Status |
|---|---|
| V1 generation runner | Adapter and execution logic unchanged; public entry performs one neutral selection check |
| V2 generation runner, regular decoding | Supported |
| V2 CUDA graphs | Supported; DMI capacity fallback selects eager before input preparation |
| V2 prefix-cached regular decoding | Supported by computed-token range accounting |
| V2 speculative decoding | Rejected before CUDA initialization |
| data parallelism, DBO/ubatching, PCP, elastic EP | Rejected by the existing adapter contract |

V2 speculative decoding is deliberately fail-closed. Its asynchronous accepted
token updates can make the CPU computed-token counter an optimistic value at
the observation boundary, so it cannot be used as an exact DMI token range
without a separate accepted-token integration.

## Qualification

Portable tests cover V1 regression, V2 config validation, request reordering,
computed/scheduled count snapshots, method-signature drift, real descriptor
use, graph-to-eager capacity fallback, and wrapper teardown.

The GPU black-box test runs stock and DMI-enabled V2 in separate processes on
Qwen3-0.6B with a fixed ragged batch and CUDA graphs enabled. It requires exact
equality for generated token IDs, text, finish fields, prompt token IDs,
full-vocabulary logprobs, and the raw bytes of every real-request
`compute_logits` output.
