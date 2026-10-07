"""Memory accounting for the refactored FP32 Megatron SAE runtime.

Independent of phase_memory_model.py (the legacy estimator). The analytic
part predicts tensor payloads, not allocator reservation or CUDA/NCCL memory.
Peak activation lifetime is reconstructed from a measured allocator trace;
it is deliberately not extrapolated using old forward/backward coefficients.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class MegatronMemoryConfig:
    d_in: int = 4096
    d_sae: int = 16384
    hooks: int = 3
    global_batch: int = 8192
    tp: int = 1
    dp: int = 1
    ga: int = 1

    def __post_init__(self):
        if min(self.d_in, self.d_sae, self.hooks, self.global_batch, self.tp, self.dp, self.ga) < 1:
            raise ValueError("dimensions, topology and GA must be positive")
        if self.tp > self.d_sae or self.global_batch % (self.dp * self.ga):
            raise ValueError("TP must not exceed d_sae; global batch must divide DP*GA exactly")

    @property
    def microbatch(self):
        return self.global_batch // (self.dp * self.ga)


def estimate_tensor_payloads(cfg: MegatronMemoryConfig) -> dict[str, int]:
    """Maximum-rank payload terms for fused Adam and fully cached inputs.

    Unequal feature ownership uses ceil(d_sae/TP); no padded parameters exist.

    DP=1 direct gradients are transient; DP>1 native main_grad is persistent.
    DDP bucket padding, small metadata, allocator slack, runtime/context and
    transient tensors are excluded and must be measured separately.
    """
    width = (cfg.d_sae + cfg.tp - 1) // cfg.tp
    params = cfg.hooks * 4 * (2 * cfg.d_in * width + width + cfg.d_in)
    inputs = cfg.hooks * 4 * (cfg.global_batch // cfg.dp) * cfg.d_in
    grads = params if cfg.dp > 1 else 0
    return dict(parameters=params, adam_moments=2 * params, gradients=grads,
        cached_inputs=inputs, persistent_payload=params * 3 + grads + inputs,
        direct_gradient_capacity=params if cfg.dp == 1 else 0,
        full_feature_tensor=4 * cfg.microbatch * cfg.d_sae,
        local_feature_tensor=4 * cfg.microbatch * width,
        # sae_in aliases cached inputs; hidden_pre, feature_acts and sae_out
        # are the additional detached output payload (with norm rescaling on).
        retained_output_set=cfg.hooks * 4 * cfg.microbatch * (2 * cfg.d_sae + cfg.d_in))


def allocation_category(frames):
    """Allocation-site attribution, not an exclusive CUDA execution phase."""
    names = {f.get("name", "") for f in frames}
    paths = " ".join(f.get("filename", "") for f in frames)
    if "_engine_run_backward" in names or "backward" in names:
        return "backward_allocations"
    if "_fused_adam" in names or "_init_group" in names and "adam.py" in paths:
        return "optimizer_allocations"
    if "training_forward_pass" in names or any("tp_wavefront" in n for n in names):
        return "forward_and_retained_outputs"
    if "train_runtime_window" in names:
        return "window_statistics_and_control"
    return "other_runtime"


def replay_allocator_window(snapshot, start, rank, storages):
    """Replay alloc/free_requested/free_completed and expandable-segment maps.

    Native traces record requested sizes. Allocation sizes use the native
    default 512-byte rounding; callers MUST compare the resulting peak with
    max_memory_allocated. Nondefault allocator rounding is not supported.
    Active-but-awaiting-free bytes remain separate from allocated tensors.
    """
    owners = {e["address"]: e["category"] for e in storages}
    live, active = {}, {}
    reserved = 0
    for segment in start["segments"]:
        if segment["device"] != rank:
            continue
        reserved += segment["total_size"]
        for block in segment["blocks"]:
            if block["state"] == "inactive":
                continue
            address = block["address"]
            entry = dict(size=block["size"], requested=block["requested_size"],
                category=owners.get(address, allocation_category(block.get("frames", []))),
                frames=block.get("frames", []), address=address)
            active[address] = entry
            if block["state"] == "active_allocated":
                live[address] = entry
    allocated = sum(v["size"] for v in live.values())
    active_bytes = sum(v["size"] for v in active.values())
    baseline = allocated
    categories = defaultdict(int)
    for value in live.values():
        categories[value["category"]] += value["size"]
    peak = allocated
    peak_active, peak_reserved = active_bytes, reserved
    peak_live, peak_categories = list(live.values()), dict(categories)
    peak_index = start["trace_index"]
    pending_at_peak, reserved_at_peak = active_bytes - allocated, reserved
    series = []
    events = snapshot["device_traces"][rank]
    for index in range(start["trace_index"], len(events)):
        e = events[index]
        action = e["action"]
        address = e.get("addr")
        if action == "alloc":
            if address in active:
                raise ValueError(f"allocating an active address: {address}")
            size = ((e["size"] + 511) // 512) * 512
            entry = dict(size=size, requested=e["size"], address=address,
                category=allocation_category(e.get("frames", [])), frames=e.get("frames", []))
            live[address] = active[address] = entry
            allocated += size
            active_bytes += size
            categories[entry["category"]] += size
        elif action == "free_requested":
            value = live.pop(address)
            allocated -= value["size"]
            categories[value["category"]] -= value["size"]
        elif action == "free_completed":
            active_bytes -= active.pop(address)["size"]
        elif action in ("segment_alloc", "segment_map"):
            reserved += e["size"]
        elif action in ("segment_free", "segment_unmap"):
            reserved -= e["size"]
        elif action == "oom":
            raise ValueError("OOM in measurement interval")
        if allocated > peak:
            peak = allocated
            peak_index = index
            peak_live = list(live.values())
            peak_categories = dict(categories)
            pending_at_peak = active_bytes - allocated
            reserved_at_peak = reserved
        peak_active = max(peak_active, active_bytes)
        peak_reserved = max(peak_reserved, reserved)
        series.append(dict(index=index, time_us=e.get("time_us"), allocated=allocated,
            active=active_bytes, reserved=reserved))
    final_blocks = [b for s in snapshot["segments"] if s["device"] == rank
                    for b in s["blocks"] if b["state"] == "active_allocated"]
    final_expected = sum(b["size"] for b in final_blocks)
    groups = defaultdict(lambda: dict(bytes=0, count=0))
    for entry in peak_live:
        frames = [f for f in entry["frames"] if "/sae_lens/" in f.get("filename", "")
                  or "/megatron/" in f.get("filename", "")]
        first = frames[0] if frames else (entry["frames"][0] if entry["frames"] else {})
        key = (entry["category"], first.get("filename", "").split("/sources/")[-1],
               first.get("line", 0), first.get("name", "unknown"))
        groups[key]["bytes"] += entry["size"]
        groups[key]["count"] += 1
    sites = [dict(category=k[0], filename=k[1], line=k[2], function=k[3], **v) for k, v in groups.items()]
    return dict(baseline_allocated=baseline, peak_allocated=peak, peak_active=peak_active,
        peak_reserved=peak_reserved, peak_trace_index=peak_index,
        pending_free_at_allocated_peak=pending_at_peak, reserved_at_allocated_peak=reserved_at_peak,
        categories_at_allocated_peak=peak_categories,
        largest_sites_at_allocated_peak=sorted(sites, key=lambda x: -x["bytes"]),
        end_allocated=allocated, snapshot_end_allocated=final_expected,
        final_accounting_error=allocated - final_expected,
        trace_events=len(events), measured_trace_events=len(series), series=series)
