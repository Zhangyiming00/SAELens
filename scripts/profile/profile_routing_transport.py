"""Short real-vLLM routing benchmark, with fixed observed dead-count scenarios.

Only the benchmark replaces feature ages and switches auxk at phase boundaries.
Activations, optimizer updates, mixing, and transport use the production runner.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
HISTORY = REPO / "results/real_training_65536_20260928/full_tp4_1600/training/dead_history.jsonl"


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def scenarios():
    rows = {r["step"]: r for r in map(json.loads, HISTORY.read_text().splitlines())}
    result = [dict(name="noaux", auxk=0, source_step=None,
                   counts={h: [0] * 4 for h in rows[1600]["hooks"]})]
    for step in (1600, 1200, 832):
        result.append(dict(name=f"dead_s{step}", auxk=2048, source_step=step,
                           counts={h: v["dead_by_tp_rank"] for h, v in rows[step]["hooks"].items()}))
    return result


def worker(directory):
    import faulthandler

    import torch
    import torch.distributed as dist

    import run_sae_runner_gpu as entry
    from sae_lens.training.multi_sae_trainer import MultiSAETrainer
    from sae_lens.vllm_model import HookedVLLMModel

    spec = json.loads((directory / "spec.json").read_text())
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    faulthandler.dump_traceback_later(240, repeat=True)
    handle = (directory / f"rank{rank}.jsonl").open("w", buffering=1)
    warmup, measured = spec["warmup"], spec["measured"]
    phase_steps = warmup + measured
    state = dict(phase=None, measuring=False, captures=[], step=None)

    if spec.get("trace"):
        from functools import wraps

        from sae_lens.training.activations_store import ActivationsStore
        from sae_lens.training.async_routing_transport import (
            AsyncRoutingTransport,
            _Receiver,
            _Ring,
            _Sender,
        )

        def traced(method, label):
            @wraps(method)
            def call(*args, **kwargs):
                with torch.cuda.nvtx.range(label):
                    return method(*args, **kwargs)
            return call

        for cls, method in ((_Sender, "submit"), (_Receiver, "receive"), (_Ring, "publish"),
                            (_Ring, "read"), (AsyncRoutingTransport, "wait_until"),
                            (ActivationsStore, "_synchronized_serving_batches")):
            setattr(cls, method, traced(getattr(cls, method), f"routing:{cls.__name__}.{method}"))

    def log(event, **kw):
        handle.write(json.dumps(dict(event=event, rank=rank, **kw), allow_nan=False, default=str) + "\n")

    fit = MultiSAETrainer.fit

    def observed_fit(self, *args, **kwargs):
        assert self.units and self.cfg.gradient_accumulation_steps == 1
        assert not self.cfg.synchronize_timing
        log("fit", cfg=asdict(self.cfg), models={h: asdict(u.model.cfg) for h, u in self.units.items()},
            tp=self._tp_world_size(), dp=self._dp_world_size(), pp=self._pp_rank(),
            optimizer_overlap=self._runtime_optimizer_overlap,
            param_gather=self._runtime_param_gather_schedule)
        # Global feature ages are replicated even for TP. Match the observed
        # four-quarter counts, using deterministic artificial feature identities.
        self._bench_ages = []
        for scenario in spec["scenarios"]:
            ages_by_hook = {}
            for h in self.units:
                ages = torch.zeros(65536, dtype=self.n_forward_passes_since_fired_by_hook[h].dtype)
                generator = torch.Generator().manual_seed(9000 + int(h.split(".")[1]))
                for shard, count in enumerate(scenario["counts"][h]):
                    indices = torch.randperm(16384, generator=generator)[:count] + shard * 16384
                    ages[indices] = self.cfg.dead_feature_window + 1
                ages_by_hook[h] = ages.to(self.cfg.device)
            self._bench_ages.append(ages_by_hook)
        self._bench_events = [tuple(torch.cuda.Event(enable_timing=True) for _ in range(3))
                              for _ in range(phase_steps * len(spec["scenarios"]))]
        result = fit(self, *args, **kwargs)
        torch.cuda.synchronize()
        log("finished", steps=self.n_training_steps)
        return result

    MultiSAETrainer.fit = observed_fit
    start = MultiSAETrainer._maybe_start_memory_timeline
    stop = MultiSAETrainer._maybe_stop_memory_timeline
    train = MultiSAETrainer._train_step

    def on_start(self):
        step = self.n_training_steps
        phase, offset = divmod(step, phase_steps)
        scenario = spec["scenarios"][phase]
        state.update(phase=phase, step=step)
        for h, unit in self.units.items():
            unit.model.cfg.auxk = scenario["auxk"]
            self.n_forward_passes_since_fired_by_hook[h].copy_(self._bench_ages[phase][h])
        if offset == 0:
            log("phase", name=scenario["name"], step=step, counts=scenario["counts"], auxk=scenario["auxk"])
        if offset == warmup:
            torch.cuda.synchronize()
            dist.barrier(group=self.runtime.training_group)
            torch.cuda.synchronize()
            if spec.get("trace"):
                torch.cuda.profiler.start()
            state.update(measuring=True, captures=[], started=time.perf_counter())
            torch.cuda.reset_peak_memory_stats()
        self._bench_events[step][0].record()
        start(self)

    def observed_train(self, *args, **kwargs):
        self._bench_events[self.n_training_steps][1].record()
        self._bench_last_inputs = args[0]
        result = train(self, *args, **kwargs)
        # Diagnostics are read after the timed phase; avoid extra device sync.
        self._bench_last_outputs = result[0]
        return result

    def on_stop(self):
        stop(self)
        step = self.n_training_steps
        phase, offset = divmod(step, phase_steps)
        self._bench_events[step][2].record()
        assert self._last_global_tokens == 4096
        assert self._last_window_microbatches == 1
        if offset != phase_steps - 1:
            return
        torch.cuda.synchronize()
        dist.barrier(group=self.runtime.training_group)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - state["started"]
        state["measuring"] = False
        if spec.get("trace"):
            torch.cuda.profiler.stop()
        events = self._bench_events[phase * phase_steps + warmup:(phase + 1) * phase_steps]
        losses = {h: {k: float(v) for k, v in out.losses.items()}
                  for h, out in self._bench_last_outputs.items()}
        assert all(math.isfinite(v) for row in losses.values() for v in row.values()), losses
        auxiliary = {h: dict(getattr(u.model, "_last_auxk_execution", {})) for h, u in self.units.items()}
        scenario = spec["scenarios"][phase]
        for h, diag in auxiliary.items():
            if scenario["auxk"] == 0:
                assert diag["selection"] == "disabled", diag
            else:
                assert diag["num_dead"] == sum(scenario["counts"][h]), diag
        transport = getattr(self.runtime, "routing_transport_instance", None)
        # A sampled input check outside the timed interval. The trainer already
        # retains these tensors through this boundary, so no additional GPU copy.
        input_hashes = {h: hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                        for h, t in self._bench_last_inputs.items()}
        log("measurement", name=scenario["name"], seconds=seconds, updates=measured,
            ms_per_update=seconds * 1000 / measured, tokens_per_second=4096 * measured / seconds,
            update_cuda_ms=[a.elapsed_time(c) for a, b, c in events],
            data_cuda_ms=[a.elapsed_time(b) for a, b, c in events],
            sae_cuda_ms=[b.elapsed_time(c) for a, b, c in events],
            capture_seconds=state["captures"], losses=losses, auxiliary=auxiliary,
            final_input_sha256=input_hashes,
            peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved(),
            transport_stats=getattr(transport, "stats", None))

    MultiSAETrainer._maybe_start_memory_timeline = on_start
    MultiSAETrainer._maybe_stop_memory_timeline = on_stop
    MultiSAETrainer._train_step = observed_train
    capture = HookedVLLMModel.run_with_cache

    def observed_capture(self, *args, **kwargs):
        began = time.perf_counter()
        result = capture(self, *args, **kwargs)
        if state["measuring"]:
            state["captures"].append(time.perf_counter() - began)
        return result

    HookedVLLMModel.run_with_cache = observed_capture
    sys.argv = [str(REPO / "run_sae_runner_gpu.py"), *spec["runner_args"]]
    try:
        entry.main()
    except BaseException as exc:
        log("failed", error=repr(exc))
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()
        handle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layout", choices=("tp4", "dp4", "pp3"), default="tp4")
    parser.add_argument("--transport", choices=("nccl", "shm_async"), default="shm_async")
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--measured", type=int, default=24)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--only", choices=("noaux", "dead_s1600", "dead_s1200", "dead_s832"))
    parser.add_argument("--trace", action="store_true", help="NVTX and CUDA profiler API around the measured window")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    directory = args.output.resolve()
    if args.worker:
        return worker(directory)
    assert args.warmup >= 1 and args.measured >= 2
    directory.mkdir(parents=True, exist_ok=False)
    source = REPO / "results/real_training_65536_20260928/run.py"
    module_spec = importlib.util.spec_from_file_location("original_real_bench", source)
    original = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(original)
    cases = scenarios()
    if args.only:
        cases = [case for case in cases if case["name"] == args.only]
    options = SimpleNamespace(case=args.layout, updates=(args.warmup + args.measured) * len(cases),
                              optimizer_overlap="on", context=1024, prompts=1)
    argv = original.runner_args(options, directory)
    # Keep timed windows free of periodic disk/metric synchronization.
    for name in ("--save-dead-every-n-steps", "--save-mse-every-n-steps", "--save-timing-every-n-steps"):
        argv[argv.index(name) + 1] = "0"
    argv += ["--routing-transport", args.transport, "--routing-shm-slots", "2"]
    tracked = [Path(__file__).resolve(), source, HISTORY,
               REPO / "sae_lens/training/async_routing_transport.py",
               REPO / "sae_lens/training/activations_store.py"]
    spec = dict(layout=args.layout, transport=args.transport, warmup=args.warmup,
                measured=args.measured, scenarios=cases, runner_args=argv,
                trace=args.trace,
                history=str(HISTORY), artificial_mask_seed_base=9000,
                notes="Fixed observed counts/quarter distribution, artificial mask identities; sequential training phases.",
                sha256={str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked})
    dump(directory / "spec.json", spec)
    (directory / "harness_source.py").write_text(Path(__file__).read_text())
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", WANDB_MODE="disabled", TOKENIZERS_PARALLELISM="false",
               NCCL_LAUNCH_ORDER_IMPLICIT="1", VLLM_ENABLE_V1_MULTIPROCESSING="0", PYTORCH_ALLOC_CONF="expandable_segments:True")
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4",
               str(Path(__file__).resolve()), "--worker", "--output", str(directory)]
    dump(directory / "command.json", command)
    began = time.perf_counter()
    with (directory / "run.log").open("w") as handle:
        proc = subprocess.Popen(command, cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = proc.wait(timeout=args.timeout)
        except BaseException:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            raise
    dump(directory / "status.json", dict(exit_code=code, elapsed_seconds=time.perf_counter() - began))
    if code:
        raise SystemExit(code)
    rows = [json.loads(line) for p in sorted(directory.glob("rank*.jsonl")) for line in p.read_text().splitlines()]
    training_ranks = 3 if args.layout == "pp3" else 4
    measurements = [r for r in rows if r["event"] == "measurement"]
    assert len(measurements) == training_ranks * len(cases), len(measurements)
    summary = []
    for scenario in cases:
        ranks = [r for r in measurements if r["name"] == scenario["name"]]
        seconds = max(r["seconds"] for r in ranks)
        summary.append(dict(name=scenario["name"], ms_per_update=1000 * seconds / args.measured,
                            tokens_per_second=4096 * args.measured / seconds,
                            peak_allocated=max(r["peak_allocated"] for r in ranks)))
    dump(directory / "summary.json", summary)
    print(json.dumps(dict(layout=args.layout, transport=args.transport, results=summary)), flush=True)
    return None


if __name__ == "__main__":
    main()
