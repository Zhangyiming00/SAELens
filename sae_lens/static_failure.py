"""Out-of-band failure notification for a fixed, synchronous runner world.

The rendezvous store remains usable when a rank is blocked in an SAE/vLLM
collective. A monitor aborts runner-owned NCCL communicators; control waits
poll the same failure record. This is not rank replacement or elastic recovery.
"""

import ctypes
import json
import sys
import threading
import time
import uuid
from functools import wraps

import torch
import torch.distributed as dist


class StaticPeerFailure(RuntimeError):
    pass


class StaticFailureMonitor:
    def __init__(self, runtime, owned_groups):
        token = [uuid.uuid4().hex if dist.get_rank() == 0 else None]
        dist.broadcast_object_list(token, src=0, group=runtime.control_group)
        self.store = dist.PrefixStore(
            f"sae-static/{token[0]}/", dist.distributed_c10d._get_default_store()
        )
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.owned_groups = tuple(
            sorted(
                owned_groups,
                key=lambda group: dist.distributed_c10d._world.pg_names[group],
                reverse=True,
            )
        )
        self.error = None
        self._communication_guards = []
        self.vllm_comms = []
        vllm_state = sys.modules.get("vllm.distributed.parallel_state")
        if vllm_state is not None:
            # vLLM's PyNccl handles are not torch ProcessGroups. Match their
            # owning device group so unrelated vLLM instances stay untouched.
            for reference in tuple(vllm_state._groups.values()):
                coordinator = reference()
                if (
                    coordinator is None
                    or getattr(coordinator, "device_group", None) not in owned_groups
                ):
                    continue
                device_comm = getattr(coordinator, "device_communicator", None)
                if device_comm is not None:
                    for name in (
                        "all_reduce",
                        "all_gather",
                        "all_gatherv",
                        "reduce_scatter",
                        "reduce_scatterv",
                        "broadcast",
                        "send",
                        "recv",
                        "batch_isend_irecv",
                    ):
                        method = getattr(device_comm, name, None)
                        if method is None:
                            continue
                        self._communication_guards.append((device_comm, name, method))
                        setattr(device_comm, name, self._guard_communication(method))
                comm = getattr(device_comm, "pynccl_comm", None)
                if comm is not None and comm.available and comm not in self.vllm_comms:
                    abort = comm.nccl.lib.ncclCommAbort
                    abort.argtypes = [ctypes.c_void_p]
                    abort.restype = ctypes.c_int
                    self.vllm_comms.append(comm)
        self._stop = threading.Event()
        self._send_sequence = {}
        self._recv_sequence = 0
        self._training_sequences = {}
        # Optional profiling callback. Disabled in ordinary training, so no
        # clock reads or per-poll timing are added unless explicitly observed.
        self.backward_wait_observer = None
        self._thread = threading.Thread(
            target=self._monitor, name="sae-failure", daemon=True
        )
        self._thread.start()

    def _guard_communication(self, method):
        @wraps(method)
        def guarded(*args, **kwargs):
            # No store access in vLLM's hot path. The monitor publishes the
            # local error before aborting, preventing fallback/reinitialization
            # of another communicator after one has already been aborted.
            if self.error is not None:
                raise StaticPeerFailure(f"Static SAE world failed: {self.error}")
            return method(*args, **kwargs)

        return guarded

    def fail(self, exc):
        message = json.dumps(dict(rank=self.rank, error=f"{type(exc).__name__}: {exc}"))
        # Keep the first cause instead of replacing it with downstream NCCL errors.
        self.store.compare_set("failure", "", message)
        self._read_failure()

    def _read_failure(self):
        if self.error is None and self.store.check(["failure"]):
            self.error = self.store.get("failure").decode()
            self.store.set(f"observed/{self.rank}", "1")
        return self.error

    def check(self):
        if self._read_failure() is not None:
            raise StaticPeerFailure(f"Static SAE world failed: {self.error}")

    def _monitor(self):
        while not self._stop.wait(0.01):
            if self._read_failure() is None:
                continue
            backends = []
            for group in self.owned_groups:
                if dist.get_backend(group) == "nccl":
                    backends.append(group._get_backend(torch.device("cuda")))
            if backends:
                # Grouping aborts prevents cyclic waits among overlapping NCCL
                # communicators. Keep Python's group registry for normal close().
                backends[0]._group_start()
                try:
                    for comm in self.vllm_comms:
                        result = comm.nccl.lib.ncclCommAbort(comm.comm)
                        comm.comm = ctypes.c_void_p()
                        comm.available = False
                        comm.nccl.NCCL_CHECK(result)
                    for backend in backends:
                        backend.abort()
                    for comm in self.vllm_comms:
                        comm.disabled = True
                finally:
                    backends[0]._group_end()
            return

    def _wait(self, keys):
        while True:
            self.check()
            if self.store.check(keys):
                return
            self._stop.wait(0.005)

    def send_command(self, rank, command, path):
        self.check()
        sequence = self._send_sequence.get(rank, 0)
        self.store.set(f"command/{rank}/{sequence}", json.dumps([command, path]))
        self._send_sequence[rank] = sequence + 1
        if command == "checkpoint":
            key = f"saved/{rank}/{sequence}"
            self._wait([key])
            self.store.delete_key(key)

    def receive_command(self):
        key = f"command/{self.rank}/{self._recv_sequence}"
        self._wait([key])
        result = json.loads(self.store.get(key))
        self.store.delete_key(key)
        self._recv_sequence += 1
        return result

    def acknowledge_checkpoint(self):
        self.check()
        self.store.set(f"saved/{self.rank}/{self._recv_sequence - 1}", "1")

    def complete_backward(self, domain, hook):
        # A CUDA collective can return to Python before its GPU work finishes.
        # Keep peers on the CPU until every member has returned from backward:
        # otherwise one member can launch the next hook or Adam after another
        # member's autograd hook raised, racing communicator abort.
        # Use monotonically increasing counters so the store has bounded size.
        name = json.dumps([domain.name, hook])
        sequence = self._training_sequences.get(name, 0) + 1
        self._training_sequences[name] = sequence
        observer = self.backward_wait_observer
        started = time.perf_counter() if observer is not None else 0.0
        polls = 0
        sleep_s = 0.0
        completed = False
        try:
            self.check()
            self.store.set(f"backward/{name}/{self.rank}", str(sequence))
            keys = [f"backward/{name}/{rank}" for rank in domain.ranks]
            while True:
                self.check()
                if self.store.check(keys) and all(
                    int(self.store.get(key)) >= sequence for key in keys
                ):
                    completed = True
                    return
                if observer is not None:
                    polls += 1
                    before_sleep = time.perf_counter()
                self._stop.wait(0.005)
                if observer is not None:
                    sleep_s += time.perf_counter() - before_sleep
        finally:
            if observer is not None:
                observer(
                    dict(domain=domain.name, hook=hook, sequence=sequence,
                         elapsed_s=time.perf_counter() - started,
                         poll_sleeps=polls, sleep_s=sleep_s, completed=completed)
                )

    def finish(self):
        self.check()
        self.store.set(f"finished/{self.rank}", "1")
        self._wait([f"finished/{rank}" for rank in range(self.world_size)])

    def close(self):
        if self.error is not None:
            # Give every live peer a chance to observe the cause before the
            # launcher closes a rank-0-owned TCPStore. This wait is bounded.
            deadline = time.monotonic() + 5
            keys = [f"observed/{rank}" for rank in range(self.world_size)]
            while not self.store.check(keys) and time.monotonic() < deadline:
                time.sleep(0.01)
            # Let the monitor finish communicator aborts before destroying them.
            self._thread.join(timeout=10)
        self._stop.set()
        self._thread.join(timeout=10)
        if self._thread.is_alive():
            raise RuntimeError(
                "Static failure monitor did not finish communicator aborts"
            )
        for device_comm, name, method in self._communication_guards:
            setattr(device_comm, name, method)
        self._communication_guards.clear()
