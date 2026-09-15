"""Bounded stock/monitored workload for official vLLM 0.29.0.

Run stock and monitored in separate processes, then compare the saved results.
Requires a GPU, real model weights, DMI native built for this PyTorch, and
ClickHouse for the monitored run. No native stub or copied reference model.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def compare(stock_path: Path, monitored_path: Path) -> None:
    import torch

    stock = torch.load(stock_path, weights_only=False)
    monitored = torch.load(monitored_path, weights_only=False)
    assert stock["public"] == monitored["public"], "public completion mismatch"
    assert stock["runner"] == monitored["runner"], "runner mismatch"
    assert stock["configuration"] == monitored["configuration"], "configuration mismatch"
    assert stock["versions"] == monitored["versions"], "runtime mismatch"
    assert stock["logits"] and len(stock["logits"]) == len(monitored["logits"])
    for step, (left, right) in enumerate(zip(stock["logits"], monitored["logits"])):
        assert torch.equal(left, right), f"raw logits differ at step {step}"
    assert monitored["storage_rows"] > 0, "no persisted DMI evidence"
    print(json.dumps({"status": "passed", "requests": len(stock["public"]),
                      "logit_steps": len(stock["logits"]),
                      "storage_rows": monitored["storage_rows"],
                      "runner": stock["runner"]}, indent=2))


def check_storage(client, model_id: str, outputs, config, hooks: str) -> int:
    import torch
    from dmi.storage.clickhouse import CHClickhouseDriverReadOnly
    from tests.blackbox.storage_contracts import StorageRow, storage_contract_mismatches

    raw = client.execute(
        "SELECT request_id, act_name, layer_no, shard_rank, start_token_idx, "
        "end_token_idx, dtype, shape, bytes FROM default.offload "
        "WHERE model_id = %(model_id)s AND request_id IN %(request_ids)s",
        {"model_id": model_id, "request_ids": tuple(o.request_id for o in outputs)},
        settings={"strings_as_bytes": True},
    )
    # Decode schema strings only; tensor payloads must remain binary even
    # when a particular tensor happens to be valid UTF-8.
    raw = [tuple(value.decode() if index in (0, 1, 6) else value
                 for index, value in enumerate(row)) for row in raw]
    rows = [StorageRow(*row[:7], tuple(row[7])) for row in raw]
    contract = {
        "num_layers": config.num_hidden_layers,
        "expected_request_count": len(outputs),
        "hooks": [
            {"act_name": "blocks.hook_resid_pre", "layers": "all", "dtype": "torch.bfloat16",
             "shape_tail": [config.hidden_size]},
            {"act_name": "hook_final_ln", "layers": [-1], "dtype": "torch.bfloat16",
             "shape_tail": [config.hidden_size]},
            {"act_name": "token_ids", "layers": [-1], "dtype": "torch.int",
             "shape_tail": []},
            {"act_name": "final_logits", "layers": [-1], "dtype": "torch.bfloat16",
             "shape_tail": [config.vocab_size], "coverage": "decisions"},
        ],
    }
    selected = set(hooks.split(","))
    hook_names = {"blocks.hook_resid_pre": "resid_pre", "hook_final_ln": "final_ln",
                  "token_ids": "token_ids", "final_logits": "final_logits"}
    contract["hooks"] = [hook for hook in contract["hooks"]
                         if hook_names[hook["act_name"]] in selected]
    errors = storage_contract_mismatches(rows, contract)
    assert not errors, errors
    by_request = {output.request_id: output for output in outputs}
    for row in raw:
        request_id, act, _layer, _rank, start, end, dtype, shape, payload = row
        tensor = CHClickhouseDriverReadOnly.torch_decode(dtype, shape, payload)
        assert tuple(tensor.shape) == tuple(shape)
        if tensor.is_floating_point() and act != "final_logits":
            assert torch.isfinite(tensor).all(), (request_id, act, "nonfinite")
        if act == "token_ids":
            output = by_request[request_id]
            expected = output.prompt_token_ids + list(output.outputs[0].token_ids)
            assert tensor.reshape(-1).tolist() == expected[start:end]
        elif act == "final_logits":
            output = by_request[request_id]
            decision = end - len(output.prompt_token_ids)
            assert tensor.shape[0] == 1 and torch.isfinite(tensor).any()
            assert int(tensor[0].argmax()) == output.outputs[0].token_ids[decision]
    for request_id, output in by_request.items():
        # Every forwarded input token, excluding the final sampled token.
        expected_end = len(output.prompt_token_ids) + len(output.outputs[0].token_ids) - 1
        tokens = [r for r in rows if r.request_id == request_id and r.act_name == "token_ids"]
        if "token_ids" in selected:
            assert max(r.end_token_idx for r in tokens) == expected_end
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["stock", "monitored", "compare"], required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stock", type=Path)
    parser.add_argument("--monitored", type=Path)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--runner", choices=["v1", "v2"], default="v2")
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--hooks", default="resid_pre,final_ln,token_ids,final_logits")
    parser.add_argument("--custom-ops", choices=["all", "none"])
    parser.add_argument("--model-id", help="stable ID for AOT-cache replay tests")
    parser.add_argument("--request-offset", type=int, default=0,
                        help="fresh request IDs when reusing a model ID; use equally in both runs")
    parser.add_argument("--db-host", default="localhost")
    args = parser.parse_args()
    if args.mode == "compare":
        compare(args.stock, args.monitored)
        return
    assert args.output is not None
    if not set(args.hooks.split(",")) <= {"resid_pre", "final_ln", "token_ids", "final_logits"}:
        parser.error("this bounded storage oracle only covers resid_pre,final_ln,token_ids,final_logits")
    if os.environ.get("DMI_VLLM_TEST_NATIVE_STUB") == "1":
        raise RuntimeError("GPU evidence cannot use the native stub")
    if args.runner == "v1":
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    else:
        os.environ.pop("VLLM_USE_V2_MODEL_RUNNER", None)
    os.environ["VLLM_DISABLE_REQUEST_ID_RANDOMIZATION"] = "1"

    import torch
    from importlib.metadata import version
    from uuid import uuid4
    from vllm import LLM, SamplingParams

    assert version("vllm") == "0.29.0"
    assert torch.cuda.is_available()
    model_id = args.model_id or f"dmi-v029-smoke-{uuid4().hex}"
    os.environ["DMI_SMOKE_LOGITS_PATH"] = str(args.output.with_suffix(".logits.pt").resolve())
    kwargs = dict(model=args.model, dtype="bfloat16", seed=42,
                  max_model_len=256, max_num_seqs=4, max_num_batched_tokens=64,
                  enable_prefix_caching=False, gpu_memory_utilization=0.35,
                  enforce_eager=not args.graph,
                  worker_cls=f"tests.v029_smoke_workers.{args.mode.title()}{args.runner.upper()}Worker")
    if args.graph:
        kwargs["compilation_config"] = {"cudagraph_capture_sizes": [1, 2, 3, 4]}
    if args.custom_ops:
        kwargs.setdefault("compilation_config", {})["custom_ops"] = [args.custom_ops]
    if args.mode == "monitored":
        kwargs["additional_config"] = {
            "dmx_model_id": model_id,
            "dmx_hook_selection": args.hooks,
            "dmx_ring_payload_mb": 256, "dmx_ring_pinned_mb": 256,
            "dmx_db_host": args.db_host,
        }
    llm = LLM(**kwargs)
    resolved = llm.llm_engine.vllm_config
    configuration = {
        "model": resolved.model_config.model,
        "dtype": str(resolved.model_config.dtype),
        "enforce_eager": resolved.model_config.enforce_eager,
        "custom_ops": resolved.compilation_config.custom_ops,
        "compile_mode": int(resolved.compilation_config.mode),
        "graph_mode": str(resolved.compilation_config.cudagraph_mode),
        "max_model_len": 256, "max_num_seqs": 4, "max_num_batched_tokens": 64,
        "seed": 42, "temperature": 0, "max_tokens": 8,
        "enable_prefix_caching": False, "tp": 1, "pp": 1,
    }
    runner = llm.collective_rpc("smoke_describe_runner")
    expected_module = "vllm.v1.worker.gpu.model_runner" if args.runner == "v2" else "vllm.v1.worker.gpu_model_runner"
    assert runner == [expected_module], runner
    llm.collective_rpc("smoke_start")
    if args.request_offset < 0:
        parser.error("--request-offset must be nonnegative")
    # Test-only control of LLM's public request IDs, keeping additional_config
    # identical across AOT-cache reloads without mixing old storage rows.
    for _ in range(args.request_offset):
        next(llm.request_counter)
    prompts = ["The capital of France is", "Count from one to ten: 1, 2,",
               "Explain in one sentence why the sky appears blue during the day."]
    # The engine process can start while LLM.generate is still tokenizing the
    # next prompt. Hold its scheduler until all three public requests are
    # queued: otherwise a timing difference invalidates a bitwise comparison.
    # This test-only control-plane barrier does not alter model arithmetic.
    core = llm.llm_engine.engine_core
    core.call_utility("pause_scheduler", "wait", False)
    llm.enqueue(prompts, SamplingParams(temperature=0, max_tokens=8, ignore_eos=True))
    core.call_utility("resume_scheduler")
    outputs = llm.wait_for_completion()
    llm.collective_rpc("smoke_dump")
    storage_rows = 0
    if args.mode == "monitored":
        # Never swallow a flush failure; storage is part of this test's oracle.
        llm.collective_rpc("stop_monitoring")
        from clickhouse_driver import Client
        from transformers import AutoConfig
        storage_rows = check_storage(Client(args.db_host), model_id, outputs,
                                     AutoConfig.from_pretrained(args.model), args.hooks)
    public = [{"request_id": o.request_id, "prompt_token_ids": o.prompt_token_ids,
               "text": o.outputs[0].text, "token_ids": list(o.outputs[0].token_ids),
               "finish_reason": o.outputs[0].finish_reason,
               "stop_reason": o.outputs[0].stop_reason} for o in outputs]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"public": public, "runner": runner, "model_id": model_id,
                "configuration": configuration, "hooks": args.hooks,
                "versions": {name: version(name) for name in ("vllm", "torch", "DMI", "DMI-vLLM-Integration")},
                "storage_rows": storage_rows,
                "logits": torch.load(os.environ["DMI_SMOKE_LOGITS_PATH"], weights_only=True)}, args.output)
    print(json.dumps({"mode": args.mode, "output": str(args.output), "storage_rows": storage_rows}))


if __name__ == "__main__":
    main()
