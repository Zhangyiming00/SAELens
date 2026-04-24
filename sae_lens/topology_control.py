"""
Persistent control state for topology switching.

The supervisor writes control_state.json atomically before every state
transition. Workers read it at startup to discover their topology.
"""

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class TopologySpec:
    vllm_tp: int
    vllm_dp: int
    sae_tp: int
    sae_dp: int = 1  # 0 = no SAE (vLLM-only), 1 = has SAE


@dataclass
class BufferParams:
    num_chunks: int
    chunk_size_tokens: int
    d_model: int
    dtype: str  # "bfloat16" or "float32"
    num_hooks: int = 1


@dataclass
class ControlState:
    phase: str  # "RUNNING" | "QUIESCING" | "RESTARTING"
    topology: TopologySpec
    buffer_name: str
    buffer_params: BufferParams
    checkpoint_path: str | None
    next_claim_seq_at_quiesce: int
    target_chunks: int
    epoch: int = 0


def write_control_state(path: Path | str, state: ControlState) -> None:
    """Atomically write control state to disk (write-then-rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    data = {
        "phase": state.phase,
        "topology": asdict(state.topology),
        "buffer_name": state.buffer_name,
        "buffer_params": asdict(state.buffer_params),
        "checkpoint_path": state.checkpoint_path,
        "next_claim_seq_at_quiesce": state.next_claim_seq_at_quiesce,
        "target_chunks": state.target_chunks,
        "epoch": state.epoch,
    }
    tmp.write_text(json.dumps(data, indent=2))
    os.rename(tmp, path)


def read_control_state(path: Path | str) -> ControlState:
    """Read control state from disk."""
    data = json.loads(Path(path).read_text())
    topo_data = data["topology"]
    topo_data.setdefault("sae_dp", 1)
    return ControlState(
        phase=data["phase"],
        topology=TopologySpec(**topo_data),
        buffer_name=data["buffer_name"],
        buffer_params=BufferParams(**data["buffer_params"]),
        checkpoint_path=data.get("checkpoint_path"),
        next_claim_seq_at_quiesce=data["next_claim_seq_at_quiesce"],
        target_chunks=data["target_chunks"],
        epoch=data.get("epoch", 0),
    )
