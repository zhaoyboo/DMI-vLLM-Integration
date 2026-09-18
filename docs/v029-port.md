# vLLM 0.29.0 port audit

Local port and bounded qualification, **not an immutable release or a claim of
GPU support for every registered model**. Audited on 2026-09-15.

**2026-09-18 scope correction:** nonessential model rewrites have been reverted.
GPU receipts below describe the earlier implementation, not qualification of
the corrected head. CPU regression evidence and outstanding GPU checks are
listed in the scope-correction section below. Compiled TP2 transparency remains
unqualified; no tolerance has been relaxed to accept the reported mismatch.

## Identity

| Component | Exact basis |
| --- | --- |
| Target official vLLM | [v0.29.0](https://github.com/vllm-project/vllm/releases/tag/v0.29.0), `98dff2a81d747d1dba01a47f939f48c3526d4206` |
| Previous upstream | v0.27.1, `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac` |
| Integration base | `420bafba42580291f1551ed129b5c55d5ac5799e` plus the changes in this PR |
| DMI core | `c222f18c4db55f6f2d7136b1d6608c0b540b273a`, package 1.2.0 / integration API v1 |
| Runtime | Official vLLM 0.29.0 wheel; Python 3.12.8, PyTorch 2.13.0 / CUDA 13.0, Transformers 5.17.0 |
| Native | Rebuilt from that DMI core against the target PyTorch/CUDA; SM89 |
| Accelerator / storage | RTX 4090, BF16, TP1/PP1; ClickHouse 25.12.2.54 |
| Checkpoint | Qwen/Qwen3-0.6B, revision `c1899de289a04d12100db370d81485cdf75e47ca` |
| Branch / packaging | `vllm-0.29-support`; separate editable integration/core worktrees and version-specific venv |

No existing supported worktree or runtime was overwritten. The PR records the
integration commit; this audit qualifies local test cells, not a release tag.
The DMI root and its integration gitlink are unchanged by this PR.

## Boundary coverage

Profile: [v029-audit-profile.json](v029-audit-profile.json).
After removing the obsolete OLMo3 module and elective Llama method, the inventory
contains 865 occurrences grouped into 555 candidate boundaries.
[v029-boundaries.tsv](v029-boundaries.tsv) maps all 555 group IDs to semantic
checklist IDs and review scope: **zero unmapped groups**. Mapping a boundary
is not the same as qualifying every runtime branch.

The grouped inventory includes 4 attribute-patch candidates, 93 configuration
accesses, 21 copied implementations, 16 DMI environment keys, 21 lazy targets,
3 private attributes, 81 inheritance boundaries, 223 potential overrides, and
93 imports. DMI-only hook-manifest methods and environment keys are explicitly
distinguished from upstream overrides. OLMo3 source, oracle, compare remap and
source-contract entry are removed; upstream 0.29 has no such native target.

## Changed contracts and evidence

| Checklist | Boundary | Discovery / implemented action | Final evidence scope |
| --- | --- | --- | --- |
| C01–C03, R01–R07 | package/plugin/registry | Pin exactly 0.29.0, including source fallback version. Drop OLMo3 alias because upstream removed its native module. | Package/CPU contracts; all 21 retained lazy targets import with the real native extension. |
| W01–W07 | worker lifecycle | Keep separate V1 and V2 workers, forwarding load/dummy/execute/shutdown arguments. | CPU lifecycle contracts and both runners' bounded GPU cells. |
| G01, G07, S01–S03 | V2 prepare_inputs | Forward new `batch_req_state` argument unchanged. Snapshot returned packed InputBatch, not scheduler dictionary order. | `tests_v2/test_v2_adapter.py`; actual storage request/token coverage. |
| G02–G04, G10 | V2 graph dispatch | Forward optional `max_query_len`; preserve the real descriptor and existing capacity fallback. | Signature/forwarding/overflow contracts; real FULL_AND_PIECEWISE replay. |
| C06, S03–S08 | V2 states / scheduling | Existing speculative, PCP, DCP, DP and DBO guards remain. No guessed counts for speculative verification. | CPU rejection/state-transition coverage; speculation/distributed GPU modes unqualified. |
| G09, M07, M13 | batch-sharded sampling | Reject final_logits selection when this mode bypasses compute_logits. Do not add new model sampling methods in this port. | CPU pre-device guard and forwarding contracts; sharded sampling GPU cell unqualified. |
| G09 | prompt logprobs | Read V2's actual in-progress prompt-logprobs worker state, not V1's num_prompt_logprobs attribute. | Regression proves rejection before upstream executes a capture-incompatible step. |
| M02, M10 | tied LM heads / loaders | Preserve existing constructors and weight ties. Remove retired AutoWeightsLoader skip_prefixes/skip_substrs; GPT2 uses WeightsMapper. Upstream automatically handles aliased parameters. | CPU loader contracts; GPU tied-head rerun pending after the scope correction. |
| M04, M11 | residual taps | Unchanged relative to 420bafb: restore pre-norm explicit additions where this PR moved them. Preserve Qwen3's original V views and Llama's already-post-norm mid tap. | CPU value/order/disabled-hook and final-tap source contracts; corrected-head GPU qualification pending. |
| M08 | MoE runner/router | Removed is_internal_router property: use runner.gate for backend validation; Qwen3-MoE and GLM pass hidden states to the runner-owned gate. Observe the single consumed routing result. | CPU route-call/order/value contracts; GPU MoE qualification not claimed. |
| M04, M07 | Kimi K3 | Preserve upstream SP ordering, avoid double reduce-scatter after fused GEMM-RS, use upstream auxiliary-stream helper and gather combined outputs. | CPU/source contracts only, not a real Kimi checkpoint test. |
| M04, M14 | Granite | Honor upstream use_rope/NoPE branch. Do not add a Granite SWA alias without qualification. | CPU/source contracts only. |
| N01–N13, L01–L02 | DMI native/storage | Core API unchanged; rebuild native instead of loading a 0.27 PyTorch/CUDA binary. Explicit stop must flush successfully. | Real native + ClickHouse readback in every monitored GPU cell. |
| G01, N04 | block tables | Current tensor metadata does not consume attention descriptors or block-table contents. | N/A: no CPU shadow is needed or fabricated. A future descriptor feature must add and validate one. |

Patch disposition relative to the integration base: **apply** unchanged
lifecycle/transport/hooks; **rewrite** the runner, loader, routing,
and model-forward boundaries listed above; **drop** OLMo3 registration. No
DMI patch is classified as upstreamed. This PR changes only the integration
repository, not the DMI root or an upstream vLLM checkout.

## Bounded GPU evidence

Historical evidence for the pre-correction implementation (through `5ce33c9`),
retained for reproducibility; do not attribute it to the corrected head.

Common workload: three fixed-composition requests, prompt lengths 5/12/14,
eight greedy output tokens each, BF16, max sequence length 256, max active
sequences 4, max batched tokens 64, prefix caching off, no quantization,
no speculative decoding, FLASH_ATTN, TP1/PP1. Graph capture sizes: 1/2/3/4.

Selected hooks: `resid_pre,final_ln,token_ids,final_logits`.

| Cell | Public output | Raw logits | DMI storage |
| --- | --- | --- | --- |
| V1 eager | Exact text/token/finish metadata match | 8 steps, bitwise equal | 744 rows validated |
| V2 eager, default runner selection | Exact match | 8 steps, bitwise equal | 744 rows validated |
| V1 default compilation + CUDA graphs | Exact match | 8 steps, bitwise equal | 744 rows validated |
| V2 default compilation + CUDA graphs, fresh cache | Exact match | 8 steps, bitwise equal | 744 rows validated |
| V2 persistent AOT cache reload, both processes | Exact match | 8 steps, bitwise equal | 744 new request rows validated |
| V2 graphs with explicit custom_ops=["all"] | Exact match | 8 steps, bitwise equal | 744 rows validated |

The default graph cells use upstream `custom_ops=["none"]`, compile mode 3,
FULL_AND_PIECEWISE. **No eager fallback or custom_ops override was required
to pass these cells.** AOT reload is checked in engine logs, not inferred from
running a workload twice. Reusing model_id keeps additional_config/cache keys
stable; fresh request IDs prevent old storage rows from masquerading as new
capture evidence.

The first comparison using an existing shared compile cache failed. Recompiling
**stock itself** changed its output; a fresh stock cache matched the monitored
runs bitwise, and fresh-cache stock cold/warm runs matched each other. This
isolates the observed mismatch to pre-existing compile artifacts, but does not
identify how they were created. They were not deleted. Use a fresh,
version-specific `VLLM_CACHE_ROOT` when migrating; do not carry over old
compiled artifacts as a qualification baseline.

The original six cells' storage checks cover request IDs, layer/activation coverage, no gaps/duplicates,
exclusive token ranges, BF16/int32 payload dtypes, shapes, finite hidden states,
exact input token IDs, and final-logit argmax matching the corresponding public
decision. Those graph cells do **not** have independent layer-by-layer residual
numerical reference coverage; the eager follow-up below does. The raw-logit tap observes
stock and monitored outputs identically in separate engine processes.

### PR review follow-up: residual values (2026-09-16)

`tests/v029_smoke.py --residual-reference --runner v1` now installs identical
test-only norm-input observers in separate stock and monitored engine processes.
Before the norm mutates any residual in place, the observer independently
computes the previous `hidden_states + residual` expression (or `hidden_states`
at the first layer), saving an owned CPU snapshot. It does not read the DMI hook
output or committed layout to construct its reference. This is a white-box
numerical diagnostic, explicitly restricted to V1 eager.

Qwen3-0.6B, BF16, TP1, the same 3 prompts and 8 generated tokens:

- All **1,368 residual rows** match the old-expression reference exactly:
  `(28 resid_pre + 28 resid_mid + 1 resid_final) × 8 × 3`.
- The monitored reference matches actual persisted tensors by request, hook,
  layer, token range, shape and dtype. The stock and monitored references also
  match each other; missing/duplicate keys and altered finite values fail.
- **1,416 total stored rows**, including token IDs and final logits; public
  output and all eight raw-logit steps remain equal.
- Raw evidence: `/tmp/v029-residual-review.b5g8ouqk/` (stock/monitored `.pt`,
  `.residuals.pt`, `.logits.pt`, and logs). See the compact committed receipt
  [v029-residual-review.json](evidence/v029-residual-review.json).

This does not numerically qualify Qwen2/Llama checkpoints or graph residuals;
their hook placement still has focused CPU/source evidence only. The bounded
release wrapper includes this eager value check after its existing cells.

Other review fixes: retained model/oracle provenance now names 0.29.0; real
`ParallelConfig` and CPU-instantiated `PromptLogprobsWorker` contracts pin both
V2 guard attributes; smoke private API/log-string dependencies and the unused
combined MoE gate boundary are documented in `vllm_contract.md`.

Follow-up validation: **680 passed, 4 skipped, 11 deselected** in the portable
gate; the three router-state skips also pass when isolated from earlier
monkey-patching tests. The optional layer-range module still requires a newer
DMI API than the pinned core. All 21 production lazy targets import without
the native stub. Wheel and sdist build from a clean temporary source snapshot;
the wheel contains no OLMo3 module. Shell syntax, diff checks and both opt-in
library-path wrapper tests pass. No graph GPU cells were rerun in this follow-up.

## CPU, import and negative-control evidence

- Portable suite: **668 passed, 4 skipped, 11 deselected** after comparator
  negative controls were added; no GPU result uses the native stub.
- Three router-isolation skips are additionally run in clean test processes;
  all pass. One optional layer-range module needs a newer DMI configuration API
  than the pinned core and remains explicitly unqualified.
- Wheel and source distribution build successfully. Every retained lazy target
  imports against the official vLLM wheel with the real native backend, including
  from an extracted integration wheel outside the editable source tree. This is
  **not** GPU model qualification.
- New negative controls reject changed public results, runner/config/runtime
  mismatch, altered logits even with identical argmax, missing logit steps,
  missing configuration provenance, and absent persisted rows.
- Residual-hook tests now enforce the original pre-port placement, including
  guarded explicit additions. V2 tests exercise changed prepare/dispatch arguments
  and incompatible sampling/logprob modes.

## Scope correction after hardware review (2026-09-18)

Discovery: M04/M11 observation placement and M02/M10 weight tying did not require
these rewrites for 0.29. Restore the existing contracts instead of optimizing
them during the port. The necessary AutoWeightsLoader, runner, routing, Granite
RoPE and Kimi SP changes remain.

- Reverted the residual-placement changes in Qwen2, Qwen3 and Llama. Their
  attention, decoder and model forward bodies match the integration base.
  Llama's `resid_mid` was already post-norm in that base and is left unchanged.
- Reverted Qwen3 V-view cleanup, the added Llama `compute_logits_local`, and
  optional `tie_weights` alignment in Qwen2/Qwen3/Qwen3.5, Qwen2/Qwen3 MoE,
  Granite, and the two Qwen3 test oracles. Existing Llama/GPT2 ties are untouched.
- Kept the independent eager residual-value oracle and strengthened the CPU
  placement tests. No new optimization PR or new model support is introduced.
- The reviewer reports a compiled TP2 token divergence at `5ce33c9`; the
  same-version old-placement control was still pending in that review.
  Reverting is **not** evidence that this divergence is fixed.
- All three local GPUs were occupied (about 21–22 GiB each). No GPU workload
  was launched and no existing process was stopped. Rerun the bounded V1/V2
  cells and the reviewer's same-version compiled TP2 control before claiming
  corrected-head GPU transparency. Existing token/logit thresholds and the
  compiled release gate are unchanged; their failures must remain visible.

Boundary inventory: 865 occurrences / 555 groups, zero unmapped. The only static
audit warning is the expected empty native source scope (DMI core is separate).
This is a focused rollback audit, not a new full-matrix qualification.

Corrected-head validation: **689 passed, 4 skipped, 11 deselected** in the
portable CPU gate. Focused model/loader/source/V2 checks: **107 passed, 1 skipped**.
The three router-state skips pass in separate fresh processes; the
remaining optional layer-range module needs a newer DMI API than the pinned
core. AST comparison against `420bafb` confirms the seven affected production
model files differ in method bodies only at the necessary Qwen3/Llama loaders,
Qwen3-MoE routing and Granite RoPE branch. All restored constructors, attention
views and residual forward bodies match the base. Diff and shell syntax checks
pass. GPU reruns remain pending as noted above.

## Reproduce

Build/install DMI native against the target vLLM environment first. A clean
library search path matters: this host's inherited LD_LIBRARY_PATH referenced
a different PyTorch/CUDA build. Unsetting it is now an explicit host-specific
opt-in, `DMI_V029_CLEAR_LD_LIBRARY_PATH=1`; normal runners preserve their paths.

From this integration checkout, with one available CUDA GPU and ClickHouse:

```bash
CUDA_VISIBLE_DEVICES=0 DMI_V029_PYTHON=.venv/bin/python \
  bash tests/run_v029_smoke.sh
```

The wrapper uses cached Qwen3-0.6B weights, fresh per-cell compilation caches,
separate stock/monitored processes, and writes logs, .pt provenance/results,
and JSON comparison receipts to a new temporary directory. It does not delete
storage data or caches. GPU commands were exercised individually locally;
the wrapper also passes shell syntax validation.

Portable gate:

```bash
env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES='' DMI_VLLM_TEST_NATIVE_STUB=1 \
  .venv/bin/python -m pytest tests tests_v2 -q -rs \
  -m 'not gpu and not clickhouse and not e2e and not slow'
```

Local raw evidence uses `/tmp/dmi-v029-qual-*` and
`/tmp/dmi-v029-customall-*`; these temporary files are not repository assets.
The manual GPU release workflow includes the bounded gate before the existing,
broader topology/online/model gates.

## Explicit limits

Only the cells above are locally GPU-qualified. Other retained model families
are candidate/source/CPU coverage. Not qualified here: 27B checkpoints,
multimodal inputs, TP2/TP4/PP/EP, quantization, MoE checkpoints, prefix-cache and
chunked-prefill interaction matrices, speculative decoding, online serving,
DMILLM, capture-toggle/storage-fault stress, and every available hook selected
simultaneously. Existing explicit rejection paths remain in force.

Release decision: usable local port for the qualified cells; no immutable
DMI/integration/vLLM pair or general model-support claim until separately
reviewed and published.
