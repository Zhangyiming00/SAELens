"""Static runtime acceptance using the existing activation transport and trainers."""

import json
import os
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import load_file

from sae_lens import distributed_v2 as routing
from sae_lens.sae_runtime import SAETrainingDomain
from sae_lens.training.activations_store import ActivationsStore
from tests.saes.test_megatron_sae_trainers import REFERENCE, _exercise_trainers

HOOKS = ("blocks.0.hook_resid_post", "blocks.1.hook_resid_post")


def _routing_worker(rank, rendezvous, output, transport="shm_async"):
    os.environ["SAE_ADAM_IMPL"] = "forloop"
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=4,
        init_method=rendezvous,
        timeout=timedelta(seconds=180),
    )
    inputs = load_file(REFERENCE / "inputs.safetensors")
    cases = [
        ("tp2_dp2_h1", 2, 2, HOOKS[:1], None),
        ("tp2_dp2_h2", 2, 2, HOOKS, None),
        ("tp1_dp4_h2", 1, 4, HOOKS, None),
        ("empty_replica_h2", 1, 4, HOOKS, None),
        ("empty_replica_h1", 1, 4, HOOKS[:1], None),
        ("tp4_dp1_h2", 4, 1, HOOKS, None),
        ("prefix_h2", 2, 1, HOOKS, (SAETrainingDomain("prefix", (0, 1), 2, HOOKS),)),
        (
            "noncontiguous_h2",
            1,
            2,
            HOOKS,
            (SAETrainingDomain("odd", (1, 3), 1, HOOKS),),
        ),
        (
            "placement_h2",
            2,
            1,
            HOOKS,
            (
                SAETrainingDomain("first", (0, 1), 2, HOOKS[:1]),
                SAETrainingDomain("second", (2, 3), 2, HOOKS[1:]),
            ),
        ),
    ]
    reports = []
    try:
        for name, tp, dp, hooks, domains in cases:
            progress_path = Path(output) / f"routing_progress_rank{rank}.json"
            progress_path.write_text(json.dumps({"case": name, "stage": "initializing"}))
            runtime = routing.initialize_sae_routing(
                P=1,
                Q=dp,
                vllm_tp_size=4,
                sae_tp_size=tp,
                batch_size=11,
                hook_names=hooks,
                training_domains=domains,
                sae_pp_size=len(domains) if domains else 1,
                routing_transport=transport,
            )
            context = runtime.local
            local_hooks = context.domain.hooks if context is not None else hooks
            store = object.__new__(ActivationsStore)
            store.device = torch.device(f"cuda:{rank}")
            store.dtype = torch.float32
            store.d_in = 16
            store.hook_names = list(local_hooks)
            store._all_hook_names = list(hooks)
            store.is_multi_hook = len(hooks) > 1
            store._add_data_timing = lambda **_kwargs: None

            def receive(step, distinguish_hooks=False):
                progress_path.write_text(json.dumps({"case": name, "stage": "receive", "step": step}))
                batch = inputs["batches"][step].to(store.device)
                payload = (
                    {
                        h: batch + i * 100 if distinguish_hooks else batch
                        for i, h in enumerate(hooks)
                    }
                    if store.is_multi_hook
                    else batch
                )
                # Only generation is supplied by this deterministic test source.
                # Slicing, P2P, assembly and TP input broadcast are production code.
                store._get_raw_llm_batch_with_epoch_restart = lambda: (payload, None)
                assembled = store._produce_one_v2_assembled_batch()
                if context is None:
                    assert assembled is None
                    return None
                route = next(
                    r
                    for r in routing.get_routing_table()
                    if r.consumer_idx == context.dp_rank
                )
                selection = slice(route.row_start, route.row_end)
                if isinstance(assembled, dict):
                    assert set(assembled) == set(local_hooks)
                    for hook, value in assembled.items():
                        torch.testing.assert_close(
                            value, payload[hook][selection], rtol=0, atol=0
                        )
                    assembled = assembled[local_hooks[0]]
                else:
                    torch.testing.assert_close(
                        assembled, batch[selection], rtol=0, atol=0
                    )
                if name.startswith("empty_replica_") and not distinguish_hooks:
                    # Exercise the existing filtered/buffered input boundary:
                    # one empty replica, with the fixed global oracle batch
                    # distributed among the remaining replicas.
                    selection = (slice(0, 0), slice(0, 4), slice(4, 8), slice(8, 11))[
                        context.dp_rank
                    ]
                    assembled = batch[selection]
                return assembled, selection

            receive(0, distinguish_hooks=True)
            if context is not None:
                directory = Path(output) / name
                directory.mkdir(parents=True, exist_ok=True)
                errors = _exercise_trainers(
                    context.tp_group,
                    context.dp_group,
                    f"cuda:{rank}",
                    directory,
                    "single" if len(hooks) == 1 else "legacy_per_hook_wrapper",
                    runtime=runtime,
                    batch_provider=receive,
                )
                reports.append(
                    {
                        "case": name,
                        "hooks": local_hooks,
                        "tp_ranks": context.tp_ranks,
                        "dp_ranks": context.dp_ranks,
                        "root": context.receive_rank,
                        "max_abs_error": errors,
                    }
                )
            else:
                for step in range(6):
                    receive(step)
                reports.append({"case": name, "producer_only": True})
            dist.barrier()
            if transport == "shm_async":
                stats = runtime.routing_transport_instance.stats
                reports[-1]["transport"] = dict(stats)
                if rank == 0 and name.startswith("tp4_"):
                    assert stats["received_bytes"] == 0
                    assert stats["local_routes"] == 7
                if rank != 0 and context is not None:
                    assert stats["received_bytes"] > 0
            routing._reset()
            progress_path.write_text(json.dumps({"case": name, "stage": "complete"}))
        (Path(output) / f"routing_rank{rank}.json").write_text(
            json.dumps(reports, indent=2)
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="Requires four CUDA GPUs")
@pytest.mark.parametrize("transport", ["shm_async", "nccl"])
def test_static_routing_train_units_and_resume(tmp_path, transport):
    mp.spawn(
        _routing_worker, args=(f"file://{tmp_path / 'rdzv'}", str(tmp_path), transport), nprocs=4
    )
    for rank in range(4):
        reports = json.loads((tmp_path / f"routing_rank{rank}.json").read_text())
        assert len(reports) == 9


def _routing_payload_worker(rank, rendezvous, output):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=4, init_method=rendezvous,
                            timeout=timedelta(seconds=180))
    cases = [
        ("matched", 4, 4, 1, 1, 1, None),
        ("fan_in", 4, 2, 1, 2, 1, None),
        ("nondivisible_fan_out", 3, 4, 1, 1, 1, None),
        ("pp_hooks", 2, 1, 2, 2, 2, None),
        ("source_is_tp_follower", 2, 1, 2, 2, 1,
         (SAETrainingDomain("shifted", (1, 2), 2, HOOKS),)),
        ("noncontiguous_dp", 2, 3, 2, 1, 1,
         (SAETrainingDomain("odd", (0, 1, 3), 1, HOOKS),)),
    ]
    reports = []
    try:
        for name, p, q, vtp, stp, pp, domains in cases:
            previous = []
            for transport in ("nccl", "shm_async"):
                runtime = routing.initialize_sae_routing(
                    P=p, Q=q, vllm_tp_size=vtp, sae_tp_size=stp, sae_pp_size=pp,
                    batch_size=11, hook_names=HOOKS, training_domains=domains,
                    routing_transport=transport, routing_shm_slots=1,
                )
                directory = getattr(runtime, "routing_shm_directory", None)
                store = object.__new__(ActivationsStore)
                store.device = torch.device(f"cuda:{rank}")
                store.dtype = torch.bfloat16
                store.d_in = 7
                store.hook_names = list(runtime.local.domain.hooks if runtime.local else HOOKS)
                store._all_hook_names = list(HOOKS)
                store.is_multi_hook = True
                store.exclude_special_tokens = torch.tensor([1000003], device=store.device)
                store._add_data_timing = lambda **_kw: None
                rows = torch.arange(11, device=store.device)

                def payload(producer, step):
                    values = (rows[:, None] * 7 + torch.arange(7, device=store.device)
                              + producer * 91 + step * 13).to(store.dtype)
                    tokens = torch.where(rows % 4 == 0, 1000003, 2000009)
                    if step == 3 and producer == 1:
                        tokens = torch.full_like(tokens, 1000003)
                    return {h: values + i * 400 for i, h in enumerate(HOOKS)}, tokens

                for step in range(9):
                    store._get_raw_llm_batch_with_epoch_restart = lambda: payload(routing.get_producer_idx(), step)
                    if transport == "shm_async":
                        store._run_nccl_p2p_exchange_v2 = lambda *_: pytest.fail("SHM used NCCL activation exchange")
                    actual = store._produce_one_v2_assembled_batch()
                    if runtime.local is None:
                        assert actual is None
                        continue
                    routes = [r for r in routing.get_routing_table()
                              if r.consumer_idx == runtime.local.dp_rank]
                    for hook in store.hook_names:
                        expected = []
                        for route in routes:
                            data, tokens = payload(route.producer_idx, step)
                            sl = slice(route.row_start, route.row_end)
                            expected.append(data[hook][sl][tokens[sl] != 1000003])
                        expected = torch.cat(expected)
                        torch.testing.assert_close(actual[hook], expected, atol=0, rtol=0)
                    snapshot = {h: x.cpu() for h, x in actual.items()}
                    if transport == "nccl":
                        previous.append(snapshot)
                    else:
                        torch.testing.assert_close(snapshot, previous[step], atol=0, rtol=0)
                report = dict(case=name, transport=transport, steps=9)
                if transport == "shm_async":
                    stats = runtime.routing_transport_instance.stats
                    report.update(stats)
                    if name == "matched":
                        assert stats["sent_bytes"] == stats["received_bytes"] == 0
                        assert stats["local_routes"] == 9
                    if name == "source_is_tp_follower" and rank == 2:
                        assert stats["local_routes"] == 9
                reports.append(report)
                dist.barrier()
                routing.close_sae_routing()
                dist.barrier()
                if directory:
                    assert not Path(directory).exists(), "transport leaked SHM files"
        (Path(output) / f"payload_rank{rank}.json").write_text(json.dumps(reports, indent=2))
    finally:
        if routing._initialized:
            routing.close_sae_routing()
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="Requires four CUDA GPUs")
def test_routing_transports_fanin_fanout_pp_filtering(tmp_path):
    mp.spawn(_routing_payload_worker, args=(f"file://{tmp_path / 'rdzv'}", str(tmp_path)), nprocs=4)
