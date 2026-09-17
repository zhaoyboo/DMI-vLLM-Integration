#!/usr/bin/env bash
# One GPU, cached Qwen3-0.6B, official vLLM 0.29.0, matching DMI native, and
# ClickHouse on localhost:9000 (override DMX_DB_HOST). No model downloads/deletes.
# Run from the integration checkout with a clean, version-specific Python env.
set -euo pipefail

smoke_python="${DMI_V029_PYTHON:-python}"
runtime_env=(env)
# Host-specific workaround; preserve required CUDA library paths by default.
if [[ "${DMI_V029_CLEAR_LD_LIBRARY_PATH:-0}" == 1 ]]; then
    runtime_env+=(-u LD_LIBRARY_PATH)
fi
artifact_root="${DMI_V029_ARTIFACT_ROOT:-${TMPDIR:-/tmp}}"
mkdir -p "$artifact_root"
run_root=$(mktemp -d "$artifact_root/v029-smoke.XXXXXX")
export HF_HUB_OFFLINE=1
printf 'Evidence directory: %s\n' "$run_root"

for runner in v1 v2; do
    for execution in eager graph; do
        cell="$runner-$execution"
        graph_args=()
        if [[ "$execution" == graph ]]; then graph_args=(--graph); fi
        capture_id="dmi-$cell-${run_root##*/}"
        for mode in stock monitored; do
            "${runtime_env[@]}" \
                VLLM_CACHE_ROOT="$run_root/cache/$cell/$mode" \
                "$smoke_python" -m tests.v029_smoke \
                --mode "$mode" --runner "$runner" "${graph_args[@]}" \
                --model-id "$capture_id" --db-host "${DMX_DB_HOST:-localhost}" \
                --output "$run_root/$cell-$mode.pt" \
                > "$run_root/$cell-$mode.log" 2>&1
        done
        "${runtime_env[@]}" "$smoke_python" -m tests.v029_smoke --mode compare \
            --stock "$run_root/$cell-stock.pt" --monitored "$run_root/$cell-monitored.pt" \
            | tee "$run_root/$cell.json"

        if [[ "$cell" == v2-graph ]]; then
            # additional_config (including model_id) contributes to vLLM's
            # cache key. Keep it identical, but use fresh request IDs so the
            # storage oracle cannot accidentally validate the previous run.
            for mode in stock monitored; do
                "${runtime_env[@]}" \
                    VLLM_CACHE_ROOT="$run_root/cache/$cell/$mode" \
                    "$smoke_python" -m tests.v029_smoke \
                    --mode "$mode" --runner v2 --graph \
                    --model-id "$capture_id" --request-offset 100 \
                    --db-host "${DMX_DB_HOST:-localhost}" \
                    --output "$run_root/$cell-$mode-reloaded.pt" \
                    > "$run_root/$cell-$mode-reloaded.log" 2>&1
                grep -q 'Directly load AOT compilation' "$run_root/$cell-$mode-reloaded.log"
            done
            "${runtime_env[@]}" "$smoke_python" -m tests.v029_smoke --mode compare \
                --stock "$run_root/$cell-stock-reloaded.pt" \
                --monitored "$run_root/$cell-monitored-reloaded.pt" \
                | tee "$run_root/$cell-reloaded.json"
        fi
    done
done

# Value-level residual regression: independent pre-norm old-expression snapshots
# versus real DMI storage, plus stock/monitored equality. Deliberately eager-only.
cell=v1-residual-eager
capture_id="dmi-$cell-${run_root##*/}"
for mode in stock monitored; do
    "${runtime_env[@]}" "$smoke_python" -m tests.v029_smoke \
        --mode "$mode" --runner v1 --residual-reference \
        --hooks resid_pre,resid_mid,resid_final,token_ids,final_logits \
        --model-id "$capture_id" --db-host "${DMX_DB_HOST:-localhost}" \
        --output "$run_root/$cell-$mode.pt" \
        > "$run_root/$cell-$mode.log" 2>&1
done
"${runtime_env[@]}" "$smoke_python" -m tests.v029_smoke --mode compare \
    --stock "$run_root/$cell-stock.pt" --monitored "$run_root/$cell-monitored.pt" \
    | tee "$run_root/$cell.json"
printf 'All bounded cells passed. Evidence: %s\n' "$run_root"
