"""Bounded real-model DP2<->DP3 watermark experiment, without forced switches.

Normal vLLM inference, SHM mixing and native Megatron updates run unchanged.
The existing auto-controller alone makes switching decisions. No sleep is
inserted into either training or production to manufacture a rate difference.
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def dump(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")


def predict(args):
    """Optional strict native estimates; real online measurements need no model."""
    selected = None
    if args.native_profile is not None:
        from sae_lens.autoconfig.execution_time_model import predict_native
        from scripts.profile.simulate_megatron_execution_time import check_sources

        profile = json.loads(Path(args.native_profile).read_text())
        check_sources(profile)
        configs = json.loads(Path(args.native_configs).read_text())
        estimates = []
        for dp in (2, 3):
            cfg = configs[f"dp{dp}"]
            required = dict(tp=1, dp=dp, pp=1, h=3, d_in=4096,
                            d_sae=args.d_sae, batch=args.microbatch*args.ga, ga=args.ga)
            if any(cfg.get(key) != value for key, value in required.items()):
                raise ValueError(f"dp{dp} prediction configuration differs from the requested run")
            estimate = predict_native(cfg, profile)
            estimates.append(dict(dp=dp, step_ms=estimate["total_ms"],
                                  tokens_per_second=estimate["tokens_per_second"],
                                  native_peak_allocated_gib=estimate["peak_allocated_bytes"]/2**30))
        selected = dict(d_sae=args.d_sae, microbatch=args.microbatch, ga=args.ga, estimates=estimates)
    local_capacity = max(math.ceil(args.microbatch/2), 2*args.chunk_tokens)
    extra_rows = 3*2*local_capacity + 2*args.microbatch + 3*args.chunk_tokens
    kv = (math.ceil(args.prompts*args.context/16)+1)*16*32*8*256*2
    return dict(created_unix=time.time(), selected=selected,
                prediction_status="calibrated" if selected is not None else "not_requested",
                source_provider_extra_budget_gib=extra_rows*3*4096*4/2**30,
                vllm_kv_gib=kv/2**30,
                shm_host_gib=args.chunks*args.chunk_tokens*3*4096*4/2**30,
                notes=["Optional native predictions require --native-profile and --native-configs with calibrated dp2/dp3 families",
                       "The run uses its real dead-feature schedule; supplied predictions describe only the configured dead state",
                       "Provider extra is a conservative storage budget, not calibrated SHM peak memory",
                       "Native estimates exclude vLLM and transport; online rates and peaks are measured"])


def worker(directory):
    import faulthandler
    import torch
    faulthandler.dump_traceback_later(600,repeat=True)
    sys.path.insert(0,str(REPO))
    import run_sae_runner_gpu as entry
    from sae_lens.llm_sae_training_runner import LanguageModelSAETrainingRunner as Runner
    from sae_lens.training.multi_sae_trainer import MultiSAETrainer
    from sae_lens.vllm_model import HookedVLLMModel
    from scripts.profile.elastic_memory_probe import training_weakrefs
    rank=int(os.environ["RANK"])
    torch.set_num_threads(1);torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    handle=(directory/f"observations_rank{rank}.jsonl").open("a",buffering=1)
    state={}
    pending=deque()

    def mem():
        return dict(allocated=torch.cuda.memory_allocated(),peak_allocated=torch.cuda.max_memory_allocated(),
                    reserved=torch.cuda.memory_reserved())

    def log(event,**kw):
        handle.write(json.dumps(dict(timestamp=time.time(),rank=rank,event=event,**kw),allow_nan=False)+"\n")

    def drain():
        while pending and pending[0][1].query():
            a,b,record=pending.popleft()
            log("step",cuda_ms=a.elapsed_time(b),**record)

    build=Runner._build_elastic_multi_trainer
    def observed_build(self,*a,**kw):
        trainer=build(self,*a,**kw)
        state["runner"]=self
        ctx=trainer.runtime.require_local()
        assert all(u.ddp._sae_megatron_ddp and u.ddp.dp_group is ctx.dp_group for u in trainer.units.values())
        assert all(getattr(u.optimizer,"_sae_distributed_optimizer",False) for u in trainer.units.values())
        log("build",dp=ctx.dp_group.size(),ga=trainer.cfg.gradient_accumulation_steps,
            global_update_batch=trainer.global_update_batch_size,
            step=trainer.n_training_steps,tokens=trainer.n_training_samples,
            dead_window=trainer.cfg.dead_feature_window,**mem())
        return trainer
    Runner._build_elastic_multi_trainer=observed_build
    start=MultiSAETrainer._maybe_start_memory_timeline
    stop=MultiSAETrainer._maybe_stop_memory_timeline
    def on_start(self):
        drain();torch.cuda.reset_peak_memory_stats()
        self._watermark_a=torch.cuda.Event(enable_timing=True)
        self._watermark_a.record()
        self._watermark_start=time.perf_counter()
        self._watermark_micro_start=self.data_provider.step_index
        start(self)
    def on_stop(self):
        stop(self)
        b=torch.cuda.Event(enable_timing=True);b.record()
        runner=state["runner"]
        micros=self.data_provider.step_index-self._watermark_micro_start
        assert micros==self._last_window_microbatches
        assert 1<=micros<=self.cfg.gradient_accumulation_steps
        assert all(u.update_count==self.n_training_steps+1 for u in self.units.values())
        pending.append((self._watermark_a,b,dict(epoch=runner._elastic_epoch,
            dp=self.runtime.require_local().dp_group.size(),step=self.n_training_steps+1,
            tokens=self.n_training_samples,micro_index=self.data_provider.step_index,
            microbatches=micros,update_tokens=self._last_global_tokens,
            auxk={h:dict(getattr(u.model,"_last_auxk_execution",{})) for h,u in self.units.items()},
            cpu_window_ms=1000*(time.perf_counter()-self._watermark_start),**mem())))
        drain()
    MultiSAETrainer._maybe_start_memory_timeline=on_start
    MultiSAETrainer._maybe_stop_memory_timeline=on_stop
    switch=Runner._elastic_switch_sae_topology
    def on_switch(self,*a,**kw):
        torch.cuda.synchronize();drain()
        old=kw.get("old_trainer")
        old_refs=training_weakrefs(old) if old else {}
        previous=(old.n_training_steps,old.n_training_samples,old.data_provider.step_index) if old else None
        old_ages={h:t.detach().cpu().clone() for h,t in old.n_forward_passes_since_fired_by_hook.items()} if old else {}
        log("switch_begin",epoch=kw["epoch"],target=kw["target_sae_dp"],progress=previous,**mem())
        started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        trainer,provider,logical=switch(self,*a,**kw)
        retained=[name for name,reference in old_refs.items() if reference() is not None]
        assert not retained,retained
        progress=(trainer.n_training_steps,trainer.n_training_samples,provider.step_index) if trainer else None
        if previous and progress:
            assert previous==progress,(previous,progress)
            assert all(torch.equal(age,trainer.n_forward_passes_since_fired_by_hook[h].cpu())
                       for h,age in old_ages.items()),"Dead-feature ages changed during migration"
        log("switch_end",epoch=kw["epoch"],target=kw["target_sae_dp"],progress=progress,
            duration_s=time.perf_counter()-started,retired_objects_checked=len(old_refs),
            dead_ages_preserved=True if previous and progress else None,**mem())
        return trainer,provider,logical
    Runner._elastic_switch_sae_topology=on_switch
    init=HookedVLLMModel.__init__
    def on_model_init(self,*a,**kw):
        torch.cuda.reset_peak_memory_stats();init(self,*a,**kw)
        log("vllm_loaded",**mem())
    HookedVLLMModel.__init__=on_model_init
    # Capture finite training losses at the normal metric cadence. This uses
    # the existing logger; no extra per-update synchronization is inserted.
    train=MultiSAETrainer._train_step
    def on_train(self,*a,**kw):
        outputs,timing=train(self,*a,**kw)
        step=self.n_training_steps+1
        if step%16==0 or self.cfg.dead_feature_window <= step <= self.cfg.dead_feature_window+4:
            losses={h:float(o.loss) for h,o in outputs.items()}
            components={h:{k:float(v) for k,v in o.losses.items()} for h,o in outputs.items()}
            assert all(math.isfinite(v) for v in losses.values()),losses
            assert all(math.isfinite(v) for values in components.values() for v in values.values()),components
            log("loss",step=step,tokens=self.data_provider.global_tokens_consumed,
                dp=self.runtime.require_local().dp_group.size(),losses=losses,components=components,
                auxk={h:dict(getattr(u.model,"_last_auxk_execution",{})) for h,u in self.units.items()})
        return outputs,timing
    MultiSAETrainer._train_step=on_train
    sys.argv=[str(REPO/"run_sae_runner_gpu.py"),*json.loads((directory/"runner_args.json").read_text())]
    try:
        entry.main()
        torch.cuda.synchronize();drain();log("finished",**mem())
    except BaseException as exc:
        log("failed",error=repr(exc));raise
    finally:
        faulthandler.cancel_dump_traceback_later();handle.close()


def terminate(proc):
    if proc.poll() is not None:return
    os.killpg(proc.pid,signal.SIGTERM)
    try:proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid,signal.SIGKILL);proc.wait()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--worker",action="store_true")
    p.add_argument("--d-sae",type=int,default=16384)
    p.add_argument("--microbatch",type=int,default=6144)
    p.add_argument("--ga",type=int,default=2)
    p.add_argument("--updates",type=int,default=480)
    p.add_argument("--context",type=int,default=1024)
    p.add_argument("--prompts",type=int,default=1)
    p.add_argument("--chunks",type=int,default=256)
    p.add_argument("--chunk-tokens",type=int,default=2048)
    p.add_argument("--dead-window","--dead-feature-window",type=int,default=1000)
    p.add_argument("--low-watermark",type=float,default=.25)
    p.add_argument("--high-watermark",type=float,default=.70)
    p.add_argument("--switches",type=int,default=4)
    p.add_argument("--timeout",type=int,default=1200)
    p.add_argument("--native-profile", help="Optional current native interpolation profile")
    p.add_argument("--native-configs", help="JSON mapping dp2/dp3 to complete calibrated configurations")
    args=p.parse_args();args.output=args.output.resolve()
    if args.worker:return worker(args.output)
    if not 0 < args.low_watermark < args.high_watermark < 1:
        p.error("Require 0 < low-watermark < high-watermark < 1")
    if args.dead_window < 0 or min(args.chunks,args.chunk_tokens,args.updates,args.ga) < 1:
        p.error("dead-window must be nonnegative; buffer and update sizes must be positive")
    if (args.native_profile is None) != (args.native_configs is None):
        p.error("--native-profile and --native-configs must be supplied together")
    args.output.mkdir(parents=True,exist_ok=False)
    dump(args.output/"predictions.json",predict(args))
    tokens=args.updates*args.microbatch*args.ga
    control=args.output/"control.json"
    runner_args=["--elastic-streaming","--no-cleanup","--elastic-permanent-vllm-dp-size","1",
        "--elastic-permanent-sae-dp-size","2","-vtp","1","-vdp","2","-stp","1","-sdp","2","-spp","1",
        "--elastic-streaming-control-path",str(control),"--streaming-dp-batch-mode","exact",
        "--hook-names","blocks.16.hook_resid_post,blocks.21.hook_resid_post,blocks.26.hook_resid_post",
        "--model-name","/root/models/Llama-3.1-8B","--dataset-path","/root/datasets/wikitext2_tokenized_llama31_ctx2048",
        "--context-size",str(args.context),"--max-model-len",str(args.context+1),
        "--store-batch-size-prompts",str(args.prompts),"--n-batches-in-buffer",str(max(2,math.ceil(args.microbatch/args.context))),
        "--d-sae",str(args.d_sae),"--k","128","--dtype","float32",
        "--train-batch-size-tokens",str(args.microbatch),"--gradient-accumulation-steps",str(args.ga),
        "--training-tokens",str(tokens),"--dead-feature-window",str(args.dead_window),"--seed","42",
        "--sae-topk-backend","sharded_dense","--sae-topk-keys","torch","--no-use-sparse-activations",
        "--ddp-zero-optimizer","--multi-sae-optimizer-overlap","on","--multi-sae-param-gather-schedule","one_hook_lag",
        "--sae-runtime-output-retention","summary","--streaming-mixing-streams","2",
        "--streaming-chunk-size-tokens",str(args.chunk_tokens),"--streaming-num-chunks",str(args.chunks),
        "--streaming-prefetch-chunks","2","--streaming-mix-chunks","2","--streaming-mix-fraction","0.5",
        "--save-memory-every-n-steps","0","--save-mse-every-n-steps","16","--save-timing-every-n-steps","1",
        "--step-window-profile-start-step","0","--step-window-profile-window-steps","0",
        "--step-window-profile-window-count","0","--n-checkpoints","0","--no-save-final-checkpoint",
        "--output-path",str(args.output/"training"),"--checkpoint-path",str(args.output/"checkpoints")]
    # Worker instrumentation files live outside the training output directory.
    dump(args.output/"runner_args.json",runner_args)
    env=dict(os.environ,OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",
             HF_HUB_OFFLINE="1",HF_DATASETS_OFFLINE="1",WANDB_MODE="disabled",TOKENIZERS_PARALLELISM="false",
             NCCL_LAUNCH_ORDER_IMPLICIT="1",VLLM_ENABLE_V1_MULTIPROCESSING="0",PYTORCH_ALLOC_CONF="expandable_segments:True")
    cmd=[sys.executable,"-m","torch.distributed.run","--standalone","--nproc_per_node=4",str(Path(__file__).resolve()),"--worker","--output",str(args.output)]
    auto=[sys.executable,str(REPO/"scripts/elastic_streaming_control.py"),str(control),"auto",
          "--low-watermark",str(args.low_watermark),"--high-watermark",str(args.high_watermark),"--poll-interval","0.5",
          "--rate-window-seconds","4","--stable-samples","3","--min-rate-ratio","1.05",
          "--min-rate-gap","1000","--cooldown-seconds","12","--max-switches",str(args.switches),
          "--log-path",str(args.output/"controller.jsonl")]
    dump(args.output/"command.json",dict(argv=cmd,auto=auto,options=vars(args)|{"output":str(args.output)},
        source_sha256={str(f.relative_to(REPO)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (REPO/"sae_lens").rglob("*.py") if "autoconfig" not in f.parts}))
    started=time.monotonic()
    with (args.output/"run.log").open("w") as log,(args.output/"auto.log").open("w") as auto_log:
        proc=subprocess.Popen(cmd,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        ctl=subprocess.Popen(auto,cwd=REPO,env=env,stdout=auto_log,stderr=subprocess.STDOUT,start_new_session=True)
        print("START",args.output,"trainer_pid",proc.pid,"controller_pid",ctl.pid,flush=True)
        try:
            rc=proc.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            terminate(proc);rc=124
        finally:
            try:ctl.wait(timeout=5)
            except subprocess.TimeoutExpired:terminate(ctl)
    result=dict(returncode=rc,controller_returncode=ctl.returncode,elapsed_s=time.monotonic()-started)
    if control.exists():
        result["control"]=json.loads(control.read_text())
        name=result["control"].get("buffer_name","")
        if name.startswith("sae_buf_") and "/" not in name:
            for f in Path("/dev/shm").glob(name+"*"):
                if f.is_file():f.unlink()
    dump(args.output/"result.json",result)
    print(json.dumps(result,indent=2),flush=True)
    if rc:raise SystemExit(rc)


if __name__=="__main__":main()
