from __future__ import annotations

import json
from pathlib import Path


def _write_timing(path: Path, steps: int, final_samples: int, elapsed_s: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[dict[str, float | int]] = []
    for step in range(1, steps + 1):
        frac = step / steps
        lines.append(
            {
                "step": step,
                "n_training_samples": int(final_samples * frac),
                "elapsed_s": elapsed_s * frac,
                "wall_time_s": elapsed_s / steps,
                "vllm_step_time_s": 0.0,
                "sae_time_s": 0.1,
                "step_time_s": 0.1,
            }
        )
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")


def _write_runner_cfg(
    run_dir: Path,
    *,
    hook_names: list[str],
    training_tokens: int,
    store_batch_size_prompts: int = 4,
    tensor_parallel_size: int = 1,
    sae_dp_mode: str = "manual",
) -> None:
    cfg = {
        "hook_name": hook_names[-1] if hook_names else "blocks.31.hook_resid_post",
        "hook_names": hook_names,
        "training_tokens": training_tokens,
        "store_batch_size_prompts": store_batch_size_prompts,
        "sae_dp_mode": sae_dp_mode,
        "model_from_pretrained_kwargs": {
            "tensor_parallel_size": tensor_parallel_size,
        },
    }
    (run_dir / "runner_cfg.json").write_text(json.dumps(cfg), encoding="utf-8")


def test_slot_assignment_keeps_failed_runs_to_prevent_label_shift(tmp_path: Path) -> None:
    from sae_lens.hooks_results_analysis import build_report

    results_root = tmp_path / "results"
    hook_names = [
        "blocks.16.hook_resid_post",
        "blocks.21.hook_resid_post",
        "blocks.26.hook_resid_post",
        "blocks.31.hook_resid_post",
    ]
    # 8 runs, including 3 failed/short runs that must keep their slots.
    run_specs = [
        ("260417_120001", 1, 1024, 2.0, None),  # slot 1 fail
        ("260417_120101", 32, 32768, 46.0, ("fsdp", 4, 1, 32768)),
        ("260417_120201", 32, 32768, 50.0, ("fsdp", 8, 1, 32768)),
        ("260417_120301", 32, 32768, 45.0, ("fsdp", 8, 2, 32768)),
        ("260417_120401", 1, 1024, 2.5, None),  # slot 5 fail
        ("260417_120501", 1, 1024, 2.8, None),  # slot 6 fail
        ("260417_120601", 32, 32768, 43.0, ("ddp", 8, 2, 32768)),
        ("260417_120701", 32, 65536, 39.0, ("manual", 4, 2, 65536)),
    ]

    for ts, steps, final_samples, elapsed_s, cfg in run_specs:
        run_dir = results_root / f"saelens_runner_gpu_{ts}"
        _write_timing(run_dir / "timing_history.jsonl", steps, final_samples, elapsed_s)
        if cfg is not None:
            mode, store_bs, tp_size, training_tokens = cfg
            _write_runner_cfg(
                run_dir,
                hook_names=hook_names,
                training_tokens=training_tokens,
                store_batch_size_prompts=store_bs,
                tensor_parallel_size=tp_size,
                sae_dp_mode=mode,
            )

    report = build_report(results_root)
    group = report.groups[4]
    slots = {row.slot: row for row in group.rows}

    assert slots[1].run_id.endswith("120001")
    assert slots[5].run_id.endswith("120401")
    assert slots[6].run_id.endswith("120501")
    # If failed runs were dropped, this would shift to slot 5/6.
    assert slots[8].scenario == "vTP"
    assert slots[8].run_id.endswith("120701")


def test_normalized_tokens_per_second_uses_common_token_budget(tmp_path: Path) -> None:
    from sae_lens.hooks_results_analysis import build_report

    results_root = tmp_path / "results"
    hook_names = ["blocks.16.hook_resid_post", "blocks.31.hook_resid_post"]

    run_ddp = results_root / "saelens_runner_gpu_260417_121001"
    _write_timing(run_ddp / "timing_history.jsonl", steps=32, final_samples=32768, elapsed_s=16.0)
    _write_runner_cfg(
        run_ddp,
        hook_names=hook_names,
        training_tokens=32768,
        store_batch_size_prompts=4,
        tensor_parallel_size=1,
        sae_dp_mode="ddp",
    )

    run_stp = results_root / "saelens_runner_gpu_260417_121101"
    _write_timing(run_stp / "timing_history.jsonl", steps=32, final_samples=65536, elapsed_s=16.0)
    _write_runner_cfg(
        run_stp,
        hook_names=hook_names,
        training_tokens=65536,
        store_batch_size_prompts=4,
        tensor_parallel_size=1,
        sae_dp_mode="manual",
    )

    report = build_report(results_root)
    group = report.groups[2]
    rows = sorted(group.rows, key=lambda row: row.run_id)

    assert rows[0].normalized_tokens_per_s == rows[1].normalized_tokens_per_s
