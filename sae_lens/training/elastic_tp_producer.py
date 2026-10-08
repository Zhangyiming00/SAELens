"""Releasable or resident vLLM producers for online elastic TP."""

from __future__ import annotations

import json
import time

from sae_lens.training.elastic_tp_config import (
    activation_dtype,
    open_buffer,
    write_json,
)
from sae_lens.training.elastic_tp_handoff import (
    cuda_memory,
    read_status,
    release_cuda_memory,
    released,
)


def load_tokens(args):
    from datasets import load_from_disk

    dataset = load_from_disk(args.dataset).with_format("torch")
    tokens = dataset[:]["tokens"].long()
    if tokens.ndim != 2 or not tokens.numel() or tokens.shape[1] % args.context:
        raise ValueError("Tokenized rows must contain whole context windows")
    return tokens.reshape(-1, args.context).contiguous()


def token_batch(tokens, sequence, rows):
    import torch

    indices = (torch.arange(rows) + sequence * rows) % len(tokens)
    return tokens[indices]


def load_llm(args):
    import torch

    from sae_lens.vllm_model import HookedVLLMModel

    return HookedVLLMModel(
        args.model,
        tokenizer=None,
        dtype=getattr(torch, args.vllm_dtype),
        capture_batch_size=args.prompts,
        capture_context_size=args.context,
        tensor_parallel_size=1,
        max_model_len=getattr(args, "max_model_len", None) or args.context + 1,
        max_num_batched_tokens=getattr(args, "max_num_batched_tokens", None)
        or args.context * args.prompts,
        gpu_memory_utilization=getattr(args, "gpu_memory_utilization", 0.55),
        **(
            {"limit_mm_per_prompt": {"image": 0, "video": 0}}
            if getattr(args, "vllm_text_only", False)
            else {}
        ),
        skip_tokenizer_init=True,
    )


def capture_chunk_gpu(model, tokens, args, sequence):
    """Pack one chunk on the producer GPU, retaining the managed input dtype.

    Shape checks are cheap and unconditional. Full finite-value validation is
    a diagnostic opt-in: it must not scan a CPU activation chunk every step.
    The returned storage is owned by this chunk, so a subsequent vLLM capture
    cannot overwrite data still being copied by AsyncVLLMShmWriter.
    """
    import torch

    rows = args.batch_size // args.context
    batch = token_batch(tokens, sequence, rows)
    pieces = []
    for offset in range(0, rows, args.prompts):
        _, cache = model.run_with_cache(
            batch[offset : offset + args.prompts], [args.hook]
        )
        pieces.append(cache[args.hook].reshape(-1, args.d_in))
    result = torch.cat(pieces) if len(pieces) > 1 else pieces[0]
    # This is the explicit user-selected input dtype boundary. Transport and
    # mixing preserve it, including true FP32 sources under the auto policy.
    result = result.to(dtype=getattr(torch, activation_dtype(args)))
    if result.shape != (args.batch_size, args.d_in):
        raise RuntimeError("Invalid real activation chunk")
    if (
        getattr(args, "validate_activations", False)
        and not torch.isfinite(result).all()
    ):
        raise RuntimeError("Non-finite real activation chunk")
    return result


class ProducerModel:
    """Keep CPU control/data state alive while destroying GPU model state."""

    def __init__(self, args, device):
        self.args, self.device = args, device
        self.model = None
        self.baseline = cuda_memory(device)["allocated"]
        self.release_report = None

    def load(self):
        if self.model is None:
            self.release_report = None
            self.model = load_llm(self.args)

    def quiesce(self, writer, *, release):
        import torch

        began = time.perf_counter()
        writer.drain()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        drain_s = time.perf_counter() - began
        if not release:
            return dict(memory_released=False, memory=cuda_memory(self.device),
                        drain_s=drain_s, close_s=0.0, quiesce_s=drain_s)
        close_started = time.perf_counter()
        if self.model is not None:
            self.model.close()
            self.model = None
        close_s = time.perf_counter() - close_started
        if self.release_report is None:
            self.release_report = release_cuda_memory(
                self.device, baseline_allocated=self.baseline,
                tolerance_mib=self.args.release_tolerance_mib,
            )
        return dict(memory_released=True, memory=self.release_report,
                    drain_s=drain_s, close_s=close_s,
                    quiesce_s=time.perf_counter() - began)


def producer(args):
    import torch

    from sae_lens.training.async_vllm_shm_writer import (
        AsyncJsonlWriter,
        AsyncVLLMShmWriter,
    )

    torch.set_num_threads(1)
    tokens = load_tokens(args)
    buffer = lifecycle = log = writer = None
    status_path = args.output / f"producer{args.producer_id}_status.json"
    control_path = args.output / "producer_control.json"
    sae_status_path = args.output / f"sae{args.producer_id}_status.json"
    control = dict(epoch=0)
    try:
        buffer = open_buffer(args)
        device = torch.device("cuda", torch.cuda.current_device())
        lifecycle = ProducerModel(args, device)
        previous = None
        log = AsyncJsonlWriter(args.output / f"producer{args.producer_id}.jsonl")
        in_flight = {}

        def record(event, **data):
            log.log(dict(event=event, timestamp=time.time(), **data))

        def on_written(written):
            sequence = written["seq_no"]
            metadata = in_flight.pop(sequence)
            record(
                "produced",
                sequence=sequence,
                tokens=args.batch_size,
                # Submission-to-publication latency overlaps the next inference;
                # it is NOT the producer's per-chunk serial service time.
                duration_s=time.perf_counter() - metadata["began"],
                capture_s=written["inference_time_s"],
                d2h_s=written["d2h_time_s"],
                shm_write_s=written["write_time_s"],
                staging_wait_s=written["staging_wait_s"],
                async_write=True,
                epoch=metadata["epoch"],
                dataset_first_window=(sequence * args.batch_size // args.context)
                % len(tokens),
            )

        writer = AsyncVLLMShmWriter(
            buffer=buffer,
            device=device,
            dtype=getattr(torch, activation_dtype(args)),
            max_rows=args.batch_size,
            d_model=args.d_in,
            producer_id=args.producer_id,
            staging_slots=2,
            on_written=on_written,
        )
        if args.vllm_residency == "resident":
            lifecycle.load()
        active = terminal = False

        def publish(state, *, ready, **data):
            nonlocal previous
            payload = dict(epoch=control["epoch"], state=state, ready=ready,
                           mode=args.vllm_residency, **data)
            write_json(status_path, payload)
            record("role", **payload)
            previous = (control["epoch"], state)

        while True:
            control = json.loads(control_path.read_text())
            if control["stop"]:
                break
            if int(buffer._header[3]) >= args.steps or terminal:
                if previous != (control["epoch"], "done"):
                    # Close and release BEFORE publishing terminal readiness.
                    # Stay alive to acknowledge later epochs without reloading.
                    memory = lifecycle.quiesce(writer, release=True)
                    if not terminal:
                        buffer.signal_done()
                    terminal, active = True, False
                    publish("done", ready=True, **memory)
                time.sleep(0.02)
                continue
            paused = args.producer_id < control["tp"]
            if paused:
                state = "released" if args.vllm_residency == "release" else "paused"
                if previous != (control["epoch"], state):
                    publish("releasing" if state == "released" else "draining",
                            ready=False, memory_released=False)
                    memory = lifecycle.quiesce(writer, release=state == "released")
                    active = False
                    publish(state, ready=True, **memory)
                time.sleep(0.02)
                continue

            if not active:
                # During startup there is no SAE process yet. Every later
                # resume needs the departing SAE worker's matching release ACK,
                # even in resident mode; the two allocators are process-local.
                if not control.get("startup", False) and not released(
                    read_status(sae_status_path), control["epoch"]
                ):
                    if previous != (control["epoch"], "waiting_for_sae_release"):
                        publish("waiting_for_sae_release", ready=False,
                                memory_released=lifecycle.model is None)
                    time.sleep(0.02)
                    continue
                publish("loading", ready=False, memory_released=False)
                began = time.perf_counter()
                lifecycle.load()
                torch.cuda.synchronize(device)
                loaded_control = json.loads(control_path.read_text())
                if loaded_control != control:
                    # A pause/stop can arrive during a cold load. Never publish
                    # stale running readiness or allocate a chunk for that epoch.
                    continue
                active = True
                publish("running", ready=True, memory_released=False,
                        load_s=time.perf_counter() - began)
            elif previous != (control["epoch"], "running"):
                publish("running", ready=True, memory_released=False, load_s=0.0)

            def interrupted():
                current = json.loads(control_path.read_text())
                return current["stop"] or args.producer_id < current["tp"]

            ticket = buffer.allocate_write_chunk(stop_check=interrupted)
            if ticket is None:
                continue
            slot, sequence = ticket
            began = time.perf_counter()
            chunk = capture_chunk_gpu(lifecycle.model, tokens, args, sequence)
            capture_s = time.perf_counter() - began
            in_flight[sequence] = dict(began=began, epoch=control["epoch"])
            submit_started = time.perf_counter()
            staging_wait_s = writer.submit(
                chunk_idx=slot,
                seq_no=sequence,
                step=sequence + 1,
                activations_gpu=chunk,
                valid_rows=args.batch_size,
                valid_tokens_per_hook=args.batch_size,
                inference_time_s=capture_s,
            )
            del (
                chunk
            )  # writer records the copy stream; no GPU payload held while paused
            record(
                "submitted",
                sequence=sequence,
                epoch=control["epoch"],
                capture_s=capture_s,
                submit_s=time.perf_counter() - submit_started,
                cycle_s=time.perf_counter() - began,
                staging_wait_s=staging_wait_s,
            )
        memory = lifecycle.quiesce(writer, release=True)
        if not terminal:
            buffer.signal_done()
        publish("done", ready=True, **memory)
    except BaseException as exc:
        write_json(status_path, dict(epoch=control["epoch"], state="failed",
                                    ready=False, memory_released=False, error=repr(exc)))
        raise
    finally:
        try:
            if writer is not None:
                writer.close()
        finally:
            try:
                if log is not None:
                    log.close()
            finally:
                try:
                    if lifecycle is not None and writer is not None:
                        lifecycle.quiesce(writer, release=True)
                finally:
                    if buffer is not None:
                        buffer.close()
