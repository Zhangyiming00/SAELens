"""Real non-streaming runner: vLLM topology and activation-buffer memory.

Instrumentation only: the normal CLI/runner performs all model generation,
routing, mixing and native Megatron updates. Never substitutes activations.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import runpy
import signal
import subprocess
import sys
import threading
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from profile_static_memory_factors import dump, memory, sha


def cases():
    base = dict(vtp=4, vdp=1, stp=2, sdp=2, spp=1, h=3, buffers=16, global_prompts=16, mix=.5)
    changes = dict(vtp4={}, vtp2=dict(vtp=2), vtp1=dict(vtp=1),
        vdp2=dict(vtp=1, vdp=2), vdp4=dict(vtp=1, vdp=4),
        vtp2dp2=dict(vtp=2, vdp=2), buffer2x=dict(buffers=32),
        capture_half=dict(global_prompts=8), mix_zero=dict(mix=0.0),
        verify_buffer24=dict(buffers=24),
        verify_capture12=dict(global_prompts=12),
        verify_t2d2_buffer24_capture24=dict(vtp=2, vdp=2, buffers=24, global_prompts=24),
        verify_d2_buffer24_capture12=dict(vtp=1, vdp=2, buffers=24, global_prompts=12))
    return {name: dict(base, **change) for name, change in changes.items()}


def worker(args):
    import faulthandler
    import torch
    faulthandler.dump_traceback_later(120, repeat=True)
    sys.path.insert(0, str(REPO))
    from sae_lens.vllm_model import HookedVLLMModel
    from sae_lens.training.multi_sae_trainer import MultiSAETrainer
    from sae_lens.vllm_memory_snapshot import collect_worker_weight_kv_bytes
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    directory = args.output / args.case
    c = cases()[args.case]
    init = HookedVLLMModel.__init__

    def model_init(self, *a, **kw):
        torch.cuda.memory._record_memory_history(enabled="all", context="all", stacks="python", max_entries=200000)
        init(self, *a, **kw)
        torch.cuda.synchronize()
        torch.cuda.memory._dump_snapshot(str(directory / f"model_init_rank{rank}.pickle"))
        torch.cuda.memory._record_memory_history(enabled=None)
        rows = self.llm.collective_rpc(collect_worker_weight_kv_bytes)
        assert rows and all(r["weight_bytes_param_scan"] > 0 for r in rows), rows
        dump(directory / f"model_init_rank{rank}.json", dict(config=c, memory=memory(), workers=rows, dtype=str(self.dtype)))

    HookedVLLMModel.__init__ = model_init
    start = MultiSAETrainer._maybe_start_memory_timeline
    stop = MultiSAETrainer._maybe_stop_memory_timeline
    fit = MultiSAETrainer.fit

    def on_start(self):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start(self)
        if self._memory_timeline_active:
            snapshot = torch.cuda.memory._snapshot()
            dump(directory / f"start_rank{rank}.json", dict(segments=snapshot["segments"], trace_index=len(snapshot["device_traces"][rank])))
        self._audit_start = time.perf_counter()

    def on_stop(self):
        torch.cuda.synchronize()
        self._audit_steps.append(dict(step=self.n_training_steps, ms=1000*(time.perf_counter()-self._audit_start),
            traced=self._memory_timeline_active, **memory()))
        stop(self)

    def audited_fit(self, *a, **kw):
        self._audit_steps = []
        result = fit(self, *a, **kw)
        dump(directory / f"rank{rank}.json", dict(config=c, steps=self._audit_steps,
            local_hooks=self.hook_names, effective_wavefront=self._runtime_tp_wavefront,
            optimizers=[type(u.optimizer).__name__ for u in self.units.values()],
            model_configs={h: u.model.cfg.to_dict() for h, u in self.units.items()},
            global_update_batch=self.global_update_batch_size, final=memory()))
        return result

    MultiSAETrainer._maybe_start_memory_timeline = on_start
    MultiSAETrainer._maybe_stop_memory_timeline = on_stop
    MultiSAETrainer.fit = audited_fit
    sys.argv = [str(REPO / "run_sae_runner_gpu.py"),
        "--model-name", "/root/models/Llama-3.1-8B",
        "--dataset-path", "/root/datasets/wikitext2_tokenized_llama31_ctx2048",
        "--context-size", "256", "--max-model-len", "257", "--max-num-batched-tokens", "4096",
        "--hook-names", "blocks.16.hook_resid_post,blocks.21.hook_resid_post,blocks.26.hook_resid_post",
        "--d-sae", "8192", "--k", "128", "--train-batch-size-tokens", "4096",
        "--training-tokens", "40960", "--gradient-accumulation-steps", "1",
        "--vllm-tp-size", str(c["vtp"]), "--vllm-dp-size", str(c["vdp"]),
        "--sae-tp-size", str(c["stp"]), "--sae-dp-size", str(c["sdp"]), "--sae-pp-size", str(c["spp"]),
        "--store-batch-size-prompts", str(c["global_prompts"] // c["vdp"]),
        "--no-auto-scale-store-batch-size-prompts", "--n-batches-in-buffer", str(c["buffers"]),
        "--activations-mixing-fraction", str(c["mix"]), "--dtype", "float32",
        "--sae-topk-backend", "sharded_dense", "--no-use-sparse-activations",
        "--dead-feature-window", "1000", "--multi-sae-tp-wavefront-schedule", "bounded",
        "--multi-sae-tp-wavefront-max-live-hooks", "2", "--sae-runtime-output-retention", "summary",
        "--multi-sae-optimizer-overlap", "on", "--multi-sae-param-gather-schedule", "one_hook_lag",
        "--save-mse-every-n-steps", "0", "--save-memory-every-n-steps", "0", "--save-timing-every-n-steps", "0",
        "--record-memory-timeline-step", "8", "--n-checkpoints", "0", "--no-save-final-checkpoint",
        "--output-path", str(directory / "runner")]
    dump(directory / f"argv_rank{rank}.json", sys.argv)
    try:
        runpy.run_path(str(REPO / "run_sae_runner_gpu.py"), run_name="__main__")
    finally:
        faulthandler.cancel_dump_traceback_later()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=REPO / "results/static_memory_factors_20260927/online")
    p.add_argument("--cases", nargs="+", choices=list(cases()), default=[k for k in cases() if not k.startswith("verify_")])
    p.add_argument("--case", choices=list(cases()))
    p.add_argument("--worker", action="store_true")
    args = p.parse_args()
    args.output = args.output.resolve()
    if args.worker:
        worker(args)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    dump(args.output / "cases.json", cases())
    dump(args.output / "source_sha256.json", {str(f.relative_to(REPO)): sha(f) for f in [*list((REPO / "sae_lens").rglob("*.py")), REPO / "run_sae_runner_gpu.py", REPO / "third_party/vllm/vllm/distributed/parallel_state.py", Path(__file__)]})
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", HF_HUB_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false", NCCL_LAUNCH_ORDER_IMPLICIT="1", VLLM_ENABLE_V1_MULTIPROCESSING="0",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    for name in args.cases:
        directory = args.output / name
        if (directory / "result.json").exists():
            if json.loads((directory / "result.json").read_text())["returncode"] == 0:
                continue
            directory.rename(args.output / f"{name}_failed_{time.time_ns()}")
        directory.mkdir(exist_ok=True)
        c = cases()[name]
        world = max(c["vtp"]*c["vdp"], c["stp"]*c["sdp"]*c["spp"])
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={world}",
            str(Path(__file__).resolve()), "--worker", "--case", name, "--output", str(args.output)]
        dump(directory / "command.json", dict(argv=command, harness_sha256=sha(__file__),
            vllm_parallel_state_sha256=sha(REPO / "third_party/vllm/vllm/distributed/parallel_state.py")))
        samples, done = [], threading.Event()

        def sample():
            import pynvml
            pynvml.nvmlInit()
            handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(world)]
            while not done.is_set():
                samples.append(dict(time=time.time(), used=[pynvml.nvmlDeviceGetMemoryInfo(h).used for h in handles]))
                done.wait(.05)

        sampler = threading.Thread(target=sample, daemon=True)
        sampler.start()
        begin = time.time()
        print("START", name, flush=True)
        with (directory / "run.log").open("w") as log:
            proc = subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                rc = proc.wait(timeout=600)
            except subprocess.TimeoutExpired:
                import psutil
                children = psutil.Process(proc.pid).children(recursive=True)
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                # torchrun workers may own separate process sessions. They
                # must not survive a failed case and pollute the next sample.
                for child in children:
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
                psutil.wait_procs(children, timeout=10)
                rc = 124
        done.set()
        sampler.join(timeout=5)
        dump(directory / "device_samples.json", samples)
        dump(directory / "result.json", dict(returncode=rc, elapsed_s=time.time()-begin,
            sampled_device_peak=[max(s["used"][i] for s in samples) for i in range(world)] if samples else None))
        print("END", name, rc, flush=True)
        if rc:
            raise RuntimeError(f"Case failed: {directory / 'run.log'}")


if __name__ == "__main__":
    main()
