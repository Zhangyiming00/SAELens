"""Warm vLLM producers and asynchronous activation transport for elastic TP."""

from __future__ import annotations

import json
import time

from sae_lens.training.elastic_tp_config import (
    activation_dtype,
    open_buffer,
    write_json,
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


def producer(args):
    import torch

    from sae_lens.training.async_vllm_shm_writer import (
        AsyncJsonlWriter,
        AsyncVLLMShmWriter,
    )

    torch.set_num_threads(1)
    tokens = load_tokens(args)
    buffer = model = log = writer = None
    try:
        buffer = open_buffer(args)
        model = load_llm(args)
        status_path = args.output / f"producer{args.producer_id}_status.json"
        control_path = args.output / "producer_control.json"
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
            device=torch.device("cuda", torch.cuda.current_device()),
            dtype=getattr(torch, activation_dtype(args)),
            max_rows=args.batch_size,
            d_model=args.d_in,
            producer_id=args.producer_id,
            staging_slots=2,
            on_written=on_written,
        )
        while True:
            control = json.loads(control_path.read_text())
            if control["stop"] or int(buffer._header[3]) >= args.steps:
                break
            paused = args.producer_id < control["tp"]
            state = (control["epoch"], "paused" if paused else "running")
            if state != previous:
                # A pause acknowledgement hands this GPU to SAE. Publish all
                # submitted chunks before ACK, including CPU SHM writes; CUDA
                # synchronization alone does not drain the background writer.
                writer.drain()
                torch.cuda.synchronize()
                write_json(
                    status_path, dict(epoch=state[0], state=state[1], ready=True)
                )
                record("role", epoch=state[0], state=state[1])
                previous = state
            if paused:
                time.sleep(0.02)
                continue

            def interrupted():
                current = json.loads(control_path.read_text())
                return current["stop"] or args.producer_id < current["tp"]

            ticket = buffer.allocate_write_chunk(stop_check=interrupted)
            if ticket is None:
                continue
            slot, sequence = ticket
            began = time.perf_counter()
            chunk = capture_chunk_gpu(model, tokens, args, sequence)
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
        writer.close()
        buffer.signal_done()
        write_json(status_path, dict(epoch=control["epoch"], state="done", ready=True))
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
                    if model is not None:
                        model.close()
                finally:
                    if buffer is not None:
                        buffer.close()
