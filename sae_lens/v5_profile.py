"""Opt-in SAE-only profiling, dispatched by run_sae_runner_gpu.py --sae-profile.

Fresh CUDA worker subprocess per case/repeat, no contamination of a real
training run. A CPU/Gloo parent group coordinates multi-rank failures and ports;
it never initializes CUDA. Each worker uses the V4-derived native validator.
No LLM is loaded. Synthetic or explicitly supplied cached inputs are labelled.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta
import itertools
import json
import os
from pathlib import Path
import socket
import subprocess
import sys


def integer_list(text):
    try:
        result = [int(x) for x in text.split(',')]
    except ValueError as exc:
        raise argparse.ArgumentTypeError('Expected comma-separated integers') from exc
    if not result:
        raise argparse.ArgumentTypeError('Empty list')
    return result


def parse(argv=None):
    from sae_lens.v5_cli import (
        add_v5_arguments, resolve_experiment_policy_defaults,
        set_experiment_policy_defaults,
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sae-tp-size','-stp',type=int,default=1)
    p.add_argument('--sae-dp-size','-sdp',type=int,default=1)
    p.add_argument('--train-batch-size-tokens',type=int,default=2048)
    p.add_argument('--gradient-accumulation-steps',type=int,default=1)
    p.add_argument('--d-sae',type=int,default=32768)
    p.add_argument('--k',type=int,default=128)
    p.add_argument('--hook-names','--hooks',default='h0,h1,h2')
    p.add_argument('--sae-topk-backend',default='sharded_ragged',choices=['legacy','sharded_dense','sharded_ragged'])
    p.add_argument('--sae-profile-d-in',type=int,default=4096)
    p.add_argument('--sae-profile-dead-counts',type=integer_list,default=[0,512,2048])
    p.add_argument('--sae-profile-dead-by-rank',default=None,
                   help='Semicolon-separated TP count vectors, e.g. "2048,0;1024,1024"; replicated across DP')
    p.add_argument('--sae-profile-placement',choices=['front','round_robin'],default='round_robin')
    p.add_argument('--sae-profile-aux-ks',type=integer_list,default=None)
    p.add_argument('--sae-profile-main-ks',type=integer_list,default=None)
    p.add_argument('--sae-profile-policies',default=None,
                   help='Semicolon-separated Main/Aux pairs, e.g. "sparse/sparse;auto/auto;local_dense/compact_dense"')
    p.add_argument('--sae-profile-warmup',type=int,default=8)
    p.add_argument('--sae-profile-steps',type=int,default=16)
    p.add_argument('--sae-profile-repeats',type=int,default=3)
    p.add_argument('--sae-profile-schedule',choices=['eager','lazy','bounded'],default='lazy')
    p.add_argument('--sae-profile-max-live-hooks',type=int,default=2)
    p.add_argument('--sae-profile-no-wavefront',action='store_true')
    p.add_argument('--sae-profile-optimizer-overlap',choices=['on','off'],default='on')
    p.add_argument('--sae-profile-no-fusion',action='store_true')
    p.add_argument('--sae-profile-no-norm',action='store_true')
    p.add_argument('--sae-profile-autocast',action='store_true')
    p.add_argument('--sae-profile-distributed-optimizer',action='store_true')
    p.add_argument('--sae-profile-save-state',action='store_true')
    p.add_argument('--sae-profile-trace',action='store_true')
    p.add_argument('--sae-profile-keys',choices=['torch','triton'],default='torch')
    p.add_argument('--sae-profile-cache-dir',type=Path,default=None)
    p.add_argument('--sae-profile-output',type=Path,default=None)
    p.add_argument('--sae-profile-dry-run',action='store_true')
    p.add_argument('--sae-profile-case-timeout',type=int,default=1800)
    add_v5_arguments(p)
    set_experiment_policy_defaults(p)
    argv = sys.argv[1:] if argv is None else argv
    return resolve_experiment_policy_defaults(p.parse_args(argv), argv)


def cases(args):
    from sae_lens.v5_cli import v5_config_kwargs
    tp,dp=args.sae_tp_size,args.sae_dp_size
    if min(tp,dp,args.d_sae,args.sae_profile_d_in,args.k,args.train_batch_size_tokens,
           args.gradient_accumulation_steps,args.sae_profile_steps,args.sae_profile_repeats)<1:
        raise ValueError('Shapes, TP/DP, GA and steps must be positive')
    if args.d_sae%tp or args.sae_profile_warmup<0:
        raise ValueError('d_sae must divide TP and warmup must be nonnegative')
    if args.sae_profile_dead_by_rank:
        layouts = [integer_list(v) for v in args.sae_profile_dead_by_rank.split(';')]
    else:
        layouts=[]
        for n in args.sae_profile_dead_counts:
            if not 0<=n<=args.d_sae:raise ValueError('Dead count outside [0,d_sae]')
            if args.sae_profile_placement=='round_robin':
                q,r=divmod(n,tp);layouts.append([q+(i<r) for i in range(tp)])
            else:
                width=args.d_sae//tp
                layouts.append([min(width,max(0,n-i*width)) for i in range(tp)])
    if any(len(v)!=tp or any(c<0 or c>args.d_sae//tp for c in v) for v in layouts):
        raise ValueError('Each TP count vector must have TP entries within [0,local_width]')
    policies = [(args.sae_main_compute,args.sae_aux_compute)]
    if args.sae_profile_policies:
        policies=[tuple(x.split('/')) for x in args.sae_profile_policies.split(';')]
        allowed={'inherit','sparse','local_dense','compact_dense','auto'}
        if any(len(v)!=2 or any(x not in allowed for x in v) for v in policies):
            raise ValueError('Policies must be Main/Aux pairs of known computation modes')
    ks=args.sae_profile_main_ks or [args.k]
    aks=args.sae_profile_aux_ks or [args.sae_aux_k or args.sae_profile_d_in//2]
    if any(k<1 or k>args.d_sae for k in ks) or any(k<1 for k in aks):
        raise ValueError('Invalid Main/Aux K')
    common=v5_config_kwargs(args)
    plans=[]
    for i,(layout,k,ak,policy,repeat) in enumerate(itertools.product(layouts,ks,aks,policies,range(args.sae_profile_repeats))):
        opts={key:value for key,value in common.items() if key.startswith('v5_') or key=='topk_tie_policy'}
        opts.update(v5_main_compute=policy[0],v5_aux_compute=policy[1])
        if args.sae_topk_backend!='sharded_ragged' and any(v!='inherit' for v in policy):
            raise ValueError('V5 policies require sharded_ragged; profile legacy/dense in a separate invocation')
        name=f'{i:04d}_k{k}_aux{ak}_dead-'+'-'.join(map(str,layout))+f'_{policy[0]}-{policy[1]}_r{repeat}'
        plans.append(dict(name=name,main_k=k,aux_k=ak,dead_by_tp_rank=layout,repeat=repeat,options=opts))
    return plans


def command(args,case,output,options_path):
    root=Path(__file__).resolve().parents[1]
    cmd=[sys.executable,str(root/'tools/validate_wave_aux_gpu.py'),
         '--tp',str(args.sae_tp_size),'--dp',str(args.sae_dp_size),
         '--batch',str(args.train_batch_size_tokens),'--ga',str(args.gradient_accumulation_steps),
         '--d-in',str(args.sae_profile_d_in),'--d-sae',str(args.d_sae),'--k',str(case['main_k']),
         '--aux-k',str(case['aux_k']),'--hooks',args.hook_names,
         '--fixed-dead-by-tp-rank',*map(str,case['dead_by_tp_rank']),
         '--warmup',str(args.sae_profile_warmup),'--steps',str(args.sae_profile_steps),
         '--schedule',args.sae_profile_schedule,'--max-live-hooks',str(args.sae_profile_max_live_hooks),
         '--optimizer-overlap',args.sae_profile_optimizer_overlap,'--keys',args.sae_profile_keys,
         '--topk-backend',args.sae_topk_backend,'--ragged-engine',args.sae_ragged_engine,
         '--ragged-main-compute',args.sae_ragged_main_compute,'--ragged-aux-compute',args.sae_ragged_aux_compute,
         '--ragged-wgrad-split',str(args.sae_ragged_wgrad_split),'--ragged-index-backend',args.sae_ragged_index_backend,
         '--ragged-openai-page-k',str(args.sae_ragged_openai_page_k),
         '--ragged-openai-workspace-mib',str(args.sae_ragged_openai_workspace_mib),
         '--ragged-openai-forward',args.sae_ragged_openai_forward,
         '--auxk-selection',args.sae_auxk_selection,'--auxk-decoder',args.sae_auxk_decoder,
         '--auxk-complement',args.sae_auxk_complement,
         '--output-retention','summary','--v5-options',str(options_path),'--output',str(output)]
    flags={'sae_profile_no_wavefront':'--no-wavefront','sae_profile_no_fusion':'--no-gradient-fusion',
           'sae_profile_no_norm':'--no-norm','sae_profile_autocast':'--autocast',
           'sae_profile_distributed_optimizer':'--distributed-optimizer',
           'sae_profile_save_state':'--save-state','sae_profile_trace':'--trace'}
    cmd += [flag for name,flag in flags.items() if getattr(args,name)]
    if args.sae_profile_cache_dir:cmd += ['--cache-dir',str(args.sae_profile_cache_dir.resolve())]
    return cmd


def summarize_records(records):
    import statistics
    steps=records[0]['steps']
    aligned=[max(r['steps'][i]['cuda_ms'] for r in records) for i in range(len(steps))]
    return dict(mean_max_rank_ms=statistics.mean(aligned), step_max_rank_ms=aligned,
                ranks=[dict(rank=r['rank'],gpu=r.get('gpu'),
                            mean_ms=statistics.mean(s['cuda_ms'] for s in r['steps']),
                            peak_allocated_bytes=r['peak_allocated_bytes'],
                            effective_optimizer_overlap=r.get('effective_optimizer_overlap'),
                            effective_decoder_fusion=r.get('effective_decoder_fusion'),
                            main_execution=r['steps'][-1].get('main_execution'),
                            aux_execution=r['steps'][-1].get('aux_execution')) for r in records])


def main_from_runner(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    args=parse(argv); plans=cases(args)
    if args.sae_profile_dry_run:
        print(json.dumps(dict(workload='SAE-only; synthetic unless cache-dir supplied',
                              tp=args.sae_tp_size,dp=args.sae_dp_size,cases=plans),indent=2))
        return
    world=args.sae_tp_size*args.sae_dp_size
    if 'RANK' not in os.environ:
        root=Path(__file__).resolve().parents[1]
        cmd=[sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={world}',
             str(root/'run_sae_runner_gpu.py'),*argv]
        subprocess.run(cmd,check=True)
        return
    import torch.distributed as dist
    from tempfile import TemporaryDirectory
    rank=int(os.environ['RANK'])
    if int(os.environ['WORLD_SIZE'])!=world:raise ValueError('WORLD_SIZE must equal SAE TP*DP in profile mode')
    if dist.is_initialized():raise RuntimeError('Profile needs fresh CPU parent processes, not live training workers')
    dist.init_process_group('gloo',timeout=timedelta(seconds=args.sae_profile_case_timeout+120))
    output=[str(args.sae_profile_output.resolve()) if args.sae_profile_output else
            str((Path('results')/('sae_v5_profile_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+str(os.getpid()))).resolve())]
    dist.broadcast_object_list(output,src=0);out=Path(output[0])
    try:
        if rank==0:
            out.mkdir(parents=True,exist_ok=False)
            (out/'plan.json').write_text(json.dumps(dict(arguments=vars(args),cases=plans),default=str,indent=2)+'\n')
        dist.barrier()
        for case in plans:
            port=[0]
            if rank==0:
                with socket.socket() as sock:
                    sock.bind(('',0));port[0]=sock.getsockname()[1]
            dist.broadcast_object_list(port,src=0)
            env=dict(os.environ,MASTER_PORT=str(port[0]))
            # A torchrun parent points env:// at its agent store. The isolated
            # child group must instead let child rank0 create a NEW store.
            env['TORCHELASTIC_USE_AGENT_STORE']='False'
            env.setdefault('NCCL_LAUNCH_ORDER_IMPLICIT','1')
            case_out=out/case['name']
            # Each node/rank has its own options temp file; output should be a
            # shared directory on multi-node runs, just as the native validator.
            with TemporaryDirectory(prefix='sae-v5-options-') as tmp:
                opts=Path(tmp)/'options.json';opts.write_text(json.dumps(case['options']))
                log=out/f"{case['name']}.rank{rank}.log"
                status=0
                with log.open('w') as stream:
                    try:
                        result=subprocess.run(command(args,case,case_out,opts),env=env,
                                              stdout=stream,stderr=subprocess.STDOUT,
                                              timeout=args.sae_profile_case_timeout)
                        status=result.returncode
                    except subprocess.TimeoutExpired:
                        status=124
                statuses=[None]*world;dist.all_gather_object(statuses,status)
                if any(statuses):
                    raise RuntimeError(f"Profile case {case['name']} failed {statuses}; inspect per-rank logs. No timings fabricated.")
            # No attempt to continue after OOM/NCCL failure with corrupted workers.
            record=json.loads((case_out/f'rank{rank}.json').read_text())
            records=[None]*world;dist.all_gather_object(records,record)
            if rank==0:
                summary=dict(case=case,**summarize_records(records))
                with (out/'summary.jsonl').open('a') as f:f.write(json.dumps(summary)+'\n')
                rank_text=' '.join(f"r{r['rank']}={r['mean_ms']:.3f}ms" for r in summary['ranks'])
                print(f"{case['name']}: max-rank mean={summary['mean_max_rank_ms']:.3f}ms; {rank_text}",flush=True)
                for r in summary['ranks']:
                    print(f"  r{r['rank']} Main={r['main_execution']} Aux={r['aux_execution']}",flush=True)
        if rank==0:
            write_aggregate(out)
            print('Profile results:',out,flush=True)
    finally:
        dist.destroy_process_group()


def write_aggregate(out):
    """Comparable repeated-run summaries; no pooled pseudo-confidence intervals."""
    import statistics
    summaries=[json.loads(line) for line in (out/'summary.jsonl').read_text().splitlines() if line.strip()]
    table=[]; groups={}
    for s in summaries:
        c=s['case'];layout=','.join(map(str,c['dead_by_tp_rank']))
        policy=json.dumps(c['options'],sort_keys=True)
        key=(c['main_k'],c['aux_k'],layout,policy)
        groups.setdefault(key,[]).append(s)
        table.append(dict(case=c['name'],main_k=c['main_k'],aux_k=c['aux_k'],dead_by_tp_rank=layout,
                          repeat=c['repeat'],mean_max_rank_ms=s['mean_max_rank_ms'],
                          max_peak_allocated_gib=max(r['peak_allocated_bytes'] for r in s['ranks'])/2**30,
                          policy=policy))
    with (out/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    result=[]
    for (k,ak,layout,policy),ss in groups.items():
        times=[s['mean_max_rank_ms'] for s in ss]
        result.append(dict(main_k=k,aux_k=ak,dead_by_tp_rank=layout,policy=json.loads(policy),runs=len(times),
                           mean_ms=statistics.mean(times),stdev_run_mean_ms=statistics.stdev(times) if len(times)>1 else None,
                           max_peak_allocated_gib=max(r['peak_allocated_bytes'] for s in ss for r in s['ranks'])/2**30))
    (out/'aggregate.json').write_text(json.dumps(result,indent=2)+'\n')
    best={}
    for row in result:
        key=(row['main_k'],row['aux_k'],row['dead_by_tp_rank'])
        if key not in best or row['mean_ms']<best[key]['mean_ms']:best[key]=row
    (out/'best_measured.json').write_text(json.dumps(list(best.values()),indent=2)+'\n')
    print('Best measured policy PER workload (not a universal optimum):',flush=True)
    for row in best.values():
        p=row['policy']
        print(f"  K={row['main_k']} Aux={row['aux_k']} dead=[{row['dead_by_tp_rank']}] "
              f"{row['mean_ms']:.3f} ms: Main={p['v5_main_compute']} Aux={p['v5_aux_compute']}",flush=True)


if __name__=='__main__':
    main_from_runner()
