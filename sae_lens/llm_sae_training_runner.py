import json
import math
import os
import signal
import sys
import threading
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic

import torch
import torch.distributed as dist
import wandb
from safetensors.torch import save_file
from simple_parsing import ArgumentParser
from transformer_lens.hook_points import HookedRootModule
from typing_extensions import deprecated

from sae_lens import logger
from sae_lens.config import HfDataset, LanguageModelSAERunnerConfig
from sae_lens.constants import (
    ACTIVATIONS_STORE_STATE_FILENAME,
    RUNNER_CFG_FILENAME,
    SAE_CFG_FILENAME,
    SAE_WEIGHTS_FILENAME,
    SPARSITY_FILENAME,
    TRAINER_STATE_FILENAME,
)
from sae_lens.distributed import (
    get_dp_group,
    get_sae_dp_size,
    get_sae_root_rank,
    get_tp_group,
    get_vllm_dp_p2p_group,
    get_vllm_dp_rank,
    get_vllm_dp_size,
    get_vllm_world_ranks,
    init_distributed,
    is_sae_active,
    is_vllm_active,
    is_vllm_dp_root,
    preinit_vllm_distributed,
)
from sae_lens.evals import EvalConfig, run_evals
from sae_lens.load_model import load_model, load_tokenizer_only_model
from sae_lens.registry import SAE_TRAINING_CLASS_REGISTRY
from sae_lens.saes.sae import (
    T_TRAINING_SAE,
    T_TRAINING_SAE_CONFIG,
    TrainingSAE,
    TrainingSAEConfig,
)
from sae_lens.training.activation_scaler import ActivationScaler
from sae_lens.training.activations_store import ActivationsStore
from sae_lens.training.multi_hook_sae import MultiHookSAE
from sae_lens.training.multi_sae_trainer import MultiSAETrainer, sanitize_hook_name_for_path
from sae_lens.training.sae_trainer import SAETrainer
from sae_lens.training.types import DataProvider
from sae_lens.util import temporary_seed


# GPU direct streaming control message types
_MSG_REQUEST_DATA = 1
_MSG_DATA_READY = 2
_MSG_READY_TO_RECV = 3
_MSG_EOF = 4
_MSG_CONSUMER_DONE = 5
_GPU_DIRECT_DTYPE_CODES = {
    torch.float32: 1,
    torch.bfloat16: 2,
    torch.float16: 3,
}
_GPU_DIRECT_CODE_DTYPES = {
    code: dtype for dtype, code in _GPU_DIRECT_DTYPE_CODES.items()
}


class VLLMProducerStagingQueue:
    """GPU staging buffer for GPU direct streaming."""

    def __init__(self, capacity: int, chunk_shape: tuple, device: torch.device):
        self.capacity = capacity
        self.chunk_shape = chunk_shape
        self.device = device
        self._queue: list[tuple[torch.Tensor, int]] = []

    def try_push(self, tensor: torch.Tensor, valid_tokens_per_hook: int) -> bool:
        """Try to push tensor to queue. Returns False if full. Must clone tensor."""
        if len(self._queue) >= self.capacity:
            return False
        stored = tensor.detach().contiguous().clone()
        self._queue.append((stored, valid_tokens_per_hook))
        return True

    def pop(self) -> tuple[torch.Tensor, int]:
        """Pop from queue. Raises IndexError if empty."""
        return self._queue.pop(0)

    @property
    def count(self) -> int:
        return len(self._queue)

    @property
    def is_full(self) -> bool:
        return len(self._queue) >= self.capacity


class VLLMGPUHandler:
    """Persistent irecv state machine for GPU direct streaming control messages."""

    def __init__(self, sae_dp_root_rank: int, gloo_ctrl_group: dist.ProcessGroup):
        self._sae_dp_root_rank = sae_dp_root_rank
        self._gloo_ctrl_group = gloo_ctrl_group
        self._ctrl_recv_buf = torch.zeros(4, dtype=torch.int32)
        self._pending_irecv: dist.Work | None = None
        self._post_irecv()

    def _post_irecv(self):
        self._pending_irecv = dist.irecv(
            self._ctrl_recv_buf,
            src=self._sae_dp_root_rank,
            group=self._gloo_ctrl_group,
        )

    def consume_pending_request(self) -> int | None:
        """Check if pending irecv completed. Returns msg_type or None."""
        if self._pending_irecv is None:
            return None
        if not self._pending_irecv.is_completed():
            return None
        self._pending_irecv.wait()
        msg_type = int(self._ctrl_recv_buf[0])
        self._pending_irecv = None
        return msg_type


class GpuDirectReceiver:
    """Background Gloo/NCCL receiver for GPU direct streaming.

    The receiver owns all control handshakes and NCCL recv calls. The trainer
    thread only waits for local provider state to become available.
    """

    def __init__(
        self,
        inner,
        gloo_ctrl_group: dist.ProcessGroup,
        nccl_group: dist.ProcessGroup,
        producer_global_rank: int,
        d_model: int,
        dtype: torch.dtype,
        device: torch.device,
        cuda_log_path: Path | None = None,
    ):
        self._inner = inner
        self._gloo_ctrl = gloo_ctrl_group
        self._nccl_group = nccl_group
        self._producer_rank = producer_global_rank
        self._d_model = d_model
        self._dtype = dtype
        self._device = device
        self._ctrl_buf = torch.zeros(4, dtype=torch.int32)
        self._cuda_log_path = cuda_log_path
        self._t_ready = time.time()
        self._cv = threading.Condition()
        self._thread: threading.Thread | None = None
        self._closed = False
        self._eof = False
        self._done_sent = False
        self._error: BaseException | None = None
        if self._cuda_log_path is not None:
            self._cuda_log_path.parent.mkdir(parents=True, exist_ok=True)
            self._cuda_log_path.write_text("")

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="gpu-direct-receiver",
            daemon=True,
        )
        self._thread.start()
        self._cuda_log({"event": "receiver_started"})

    def wait_for_data(self) -> None:
        with self._cv:
            t0 = time.perf_counter()
            while (
                self._inner_min_pool_tokens() == 0
                and not self._eof
                and self._error is None
            ):
                self._cv.wait(timeout=0.1)
            if self._error is not None:
                raise RuntimeError("GPU direct receiver thread failed") from self._error
            wait_s = time.perf_counter() - t0
        if wait_s > 0:
            self._cuda_log({"event": "trainer_wait_for_data", "wait_time_s": wait_s})

    def wait_for_prefill(
        self,
        requested_target_tokens: int,
        timeout_s: float = 120.0,
    ) -> None:
        """Block until the inner provider has post-mixing serving tokens."""
        prefill_target = getattr(self._inner, "prefill_target_tokens", None)
        effective_target = (
            int(prefill_target(requested_target_tokens))
            if prefill_target is not None
            else int(requested_target_tokens)
        )
        if effective_target <= 0:
            return

        prefill_satisfied = getattr(self._inner, "prefill_satisfied", None)
        t0 = time.perf_counter()
        with self._cv:
            while (
                not self._inner_prefill_satisfied(prefill_satisfied, effective_target)
                and not self._eof
                and self._error is None
            ):
                elapsed = time.perf_counter() - t0
                if elapsed >= timeout_s:
                    logger.warning(
                        "[gpu-direct-receiver] Prefill timeout after %.1fs "
                        "(serving=%d, target=%d)",
                        elapsed,
                        self._inner_serving_tokens(),
                        effective_target,
                    )
                    break
                self._cv.wait(timeout=0.5)
            if self._error is not None:
                raise RuntimeError("GPU direct receiver thread failed") from self._error

        self._cuda_log({
            "event": "prefill_complete",
            "requested_target_tokens": int(requested_target_tokens),
            "effective_target_tokens": effective_target,
            "serving_tokens": self._inner_serving_tokens(),
            "storage_tokens": self._inner_storage_tokens(),
            "available_tokens": self._inner_available_tokens(),
            "elapsed_s": time.perf_counter() - t0,
        })

    def next_batch(self) -> torch.Tensor | dict[str, torch.Tensor]:
        while True:
            with self._cv:
                try:
                    batch = next(self._inner)
                    pool_tokens = self._inner_min_pool_tokens()
                    available_tokens = self._inner_available_tokens()
                    self._cv.notify_all()
                except StopIteration:
                    if self._eof:
                        raise
                    if self._error is not None:
                        raise RuntimeError("GPU direct receiver thread failed") from self._error
                    self._cv.wait(timeout=0.1)
                    continue
            self._cuda_log({
                "event": "batch_served",
                "pool_tokens": pool_tokens,
                "available_tokens": available_tokens,
            })
            return batch

    def notify_consumed(self) -> None:
        with self._cv:
            self._cv.notify_all()

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify_all()
        if self._thread is None:
            self._send_consumer_done()
        else:
            self._thread.join()

    @property
    def eof(self) -> bool:
        return self._eof

    def _run(self) -> None:
        try:
            if self._device.type == "cuda":
                torch.cuda.set_device(self._device)
            while True:
                with self._cv:
                    while (
                        not self._closed
                        and not self._eof
                        and not self._inner_needs_refill()
                    ):
                        self._cv.wait(timeout=0.1)
                    if self._closed or self._eof:
                        return
                    desired_chunks = self._desired_refill_chunks()
                received = self._request_and_recv(desired_chunks)
                if not received:
                    return
                for _ in range(desired_chunks - 1):
                    if not self._recv_one_chunk():
                        return
        except BaseException as exc:
            with self._cv:
                self._error = exc
                self._cv.notify_all()
            self._cuda_log({"event": "receiver_error", "error": repr(exc)})
        finally:
            if self._closed and not self._eof and self._error is None:
                self._send_consumer_done()

    def _send_consumer_done(self) -> None:
        if self._done_sent or self._eof:
            return
        ctrl = torch.zeros(4, dtype=torch.int32)
        ctrl[0] = _MSG_CONSUMER_DONE
        dist.send(ctrl, dst=self._producer_rank, group=self._gloo_ctrl)
        self._done_sent = True
        self._cuda_log({"event": "consumer_done"})

    def _request_and_recv(self, requested_chunks: int = 1) -> bool:
        self._ctrl_buf.zero_()
        self._ctrl_buf[0] = _MSG_REQUEST_DATA
        self._ctrl_buf[1] = max(1, int(requested_chunks))
        self._cuda_log({
            "event": "request_data",
            "producer_rank": self._producer_rank,
            "requested_chunks": int(self._ctrl_buf[1]),
        })
        dist.send(self._ctrl_buf, dst=self._producer_rank, group=self._gloo_ctrl)
        return self._recv_one_chunk()

    def _recv_one_chunk(self) -> bool:
        t0 = time.perf_counter()
        dist.recv(self._ctrl_buf, src=self._producer_rank, group=self._gloo_ctrl)
        wait_s = time.perf_counter() - t0
        msg_type = int(self._ctrl_buf[0])

        if msg_type == _MSG_EOF:
            with self._cv:
                self._eof = True
                self._inner.mark_eof()
                self._cv.notify_all()
            self._cuda_log({"event": "eof", "wait_time_s": wait_s})
            return False

        assert msg_type == _MSG_DATA_READY
        valid_tokens_per_hook = int(self._ctrl_buf[1])
        dtype_code = int(self._ctrl_buf[2])
        dtype = _GPU_DIRECT_CODE_DTYPES.get(dtype_code, self._dtype)
        self._cuda_log({
            "event": "data_ready",
            "valid_tokens_per_hook": valid_tokens_per_hook,
            "dtype_code": dtype_code,
            "wait_time_s": wait_s,
        })

        self._ctrl_buf.zero_()
        self._ctrl_buf[0] = _MSG_READY_TO_RECV
        dist.send(self._ctrl_buf, dst=self._producer_rank, group=self._gloo_ctrl)

        num_hooks = len(self._inner._pp_hook_names)
        recv_rows = valid_tokens_per_hook * num_hooks
        recv_buf = torch.empty(
            recv_rows, self._d_model, dtype=dtype, device=self._device
        )
        t0 = time.perf_counter()
        dist.broadcast(recv_buf, src=self._producer_rank, group=self._nccl_group)
        nccl_time_s = time.perf_counter() - t0

        with self._cv:
            self._inner.receive_chunk(recv_buf)
            pool_tokens = self._inner_min_pool_tokens()
            available_tokens = self._inner_available_tokens()
            self._cv.notify_all()
        self._cuda_log({
            "event": "nccl_recv_complete",
            "rows": recv_rows,
            "d_model": self._d_model,
            "dtype": str(dtype).removeprefix("torch."),
            "nccl_time_s": nccl_time_s,
            "pool_tokens": pool_tokens,
            "available_tokens": available_tokens,
        })
        return True

    def _inner_needs_refill(self) -> bool:
        receiver_needs_refill = getattr(self._inner, "receiver_needs_refill", None)
        if receiver_needs_refill is not None:
            return bool(receiver_needs_refill())
        needs_refill = getattr(self._inner, "needs_refill", None)
        if needs_refill is not None:
            return bool(needs_refill())
        return self._inner_min_pool_tokens() == 0

    def _inner_available_tokens(self) -> int:
        return int(getattr(self._inner, "available_tokens", self._inner_min_pool_tokens()))

    def _inner_serving_tokens(self) -> int:
        serving_tokens = getattr(self._inner, "serving_tokens", None)
        if serving_tokens is not None:
            return int(serving_tokens())
        return self._inner_min_pool_tokens()

    def _inner_storage_tokens(self) -> int:
        storage_tokens = getattr(self._inner, "storage_tokens", None)
        if storage_tokens is not None:
            return int(storage_tokens())
        return -1

    def _inner_prefill_satisfied(self, prefill_satisfied: Any, target: int) -> bool:
        if prefill_satisfied is not None:
            return bool(prefill_satisfied(target))
        return self._inner_min_pool_tokens() >= target

    def _inner_min_pool_tokens(self) -> int:
        return int(self._inner._min_pool_tokens())

    def _desired_refill_chunks(self) -> int:
        desired = getattr(self._inner, "desired_receiver_refill_chunks", None)
        if desired is None:
            desired = getattr(self._inner, "desired_refill_chunks", None)
        if desired is None:
            return 1
        return max(1, int(desired()))

    def _cuda_log(self, record: dict) -> None:
        if self._cuda_log_path is None:
            return
        record["elapsed_s"] = time.time() - self._t_ready
        with open(self._cuda_log_path, "a") as f:
            json.dump(record, f)
            f.write("\n")


class GpuDirectDataProvider:
    """Wraps GpuStreamingActivationProvider with GPU direct receive logic.

    Blocks in __next__ to perform the control handshake when the inner
    provider's pool is empty unless a background receiver is provided.
    """

    def __init__(
        self,
        inner,
        gloo_ctrl_group: dist.ProcessGroup | None = None,
        nccl_group: dist.ProcessGroup | None = None,
        producer_global_rank: int | None = None,
        d_model: int | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        cuda_log_path: Path | None = None,
        receiver: GpuDirectReceiver | None = None,
    ):
        self._inner = inner
        self._receiver = receiver
        self._gloo_ctrl = gloo_ctrl_group
        self._nccl_group = nccl_group
        self._producer_rank = producer_global_rank
        self._d_model = d_model
        self._dtype = dtype
        self._device = device
        self._eof = False
        self._closed = False
        self._ctrl_buf = torch.zeros(4, dtype=torch.int32)
        self._cuda_log_path = cuda_log_path
        self._t_ready = time.time()
        if self._cuda_log_path is not None:
            self._cuda_log_path.parent.mkdir(parents=True, exist_ok=True)
            self._cuda_log_path.write_text("")

    def __iter__(self):
        return self

    def __next__(self) -> torch.Tensor | dict[str, torch.Tensor]:
        if self._receiver is not None:
            return self._receiver.next_batch()

        while True:
            try:
                batch = next(self._inner)
                self._refill_inner_to_high_watermark()
                return batch
            except StopIteration:
                if self._eof:
                    raise
                self._refill_inner_to_high_watermark(force_one=True)

    def close(self) -> None:
        """Tell the producer that training ended before upstream EOF."""
        if self._receiver is not None:
            self._receiver.close()
            self._closed = True
            return
        if self._closed or self._eof:
            return
        self._ctrl_buf.zero_()
        self._ctrl_buf[0] = _MSG_CONSUMER_DONE
        dist.send(self._ctrl_buf, dst=self._producer_rank, group=self._gloo_ctrl)
        self._closed = True
        self._cuda_log({"event": "consumer_done"})

    def _request_and_recv(self, requested_chunks: int = 1) -> bool:
        self._ctrl_buf.zero_()
        self._ctrl_buf[0] = _MSG_REQUEST_DATA
        self._ctrl_buf[1] = max(1, int(requested_chunks))
        self._cuda_log({
            "event": "request_data",
            "producer_rank": self._producer_rank,
            "requested_chunks": int(self._ctrl_buf[1]),
        })
        dist.send(self._ctrl_buf, dst=self._producer_rank, group=self._gloo_ctrl)

        return self._recv_one_chunk()

    def _recv_one_chunk(self) -> bool:
        t0 = time.perf_counter()
        dist.recv(self._ctrl_buf, src=self._producer_rank, group=self._gloo_ctrl)
        wait_s = time.perf_counter() - t0
        msg_type = int(self._ctrl_buf[0])

        if msg_type == _MSG_EOF:
            self._eof = True
            self._inner.mark_eof()
            self._cuda_log({"event": "eof", "wait_time_s": wait_s})
            return False

        assert msg_type == _MSG_DATA_READY
        valid_tokens_per_hook = int(self._ctrl_buf[1])
        dtype_code = int(self._ctrl_buf[2])
        dtype = _GPU_DIRECT_CODE_DTYPES.get(dtype_code, self._dtype)
        self._cuda_log({
            "event": "data_ready",
            "valid_tokens_per_hook": valid_tokens_per_hook,
            "dtype_code": dtype_code,
            "wait_time_s": wait_s,
        })

        self._ctrl_buf.zero_()
        self._ctrl_buf[0] = _MSG_READY_TO_RECV
        dist.send(self._ctrl_buf, dst=self._producer_rank, group=self._gloo_ctrl)

        num_hooks = len(self._inner._pp_hook_names)
        recv_rows = valid_tokens_per_hook * num_hooks
        recv_buf = torch.empty(
            recv_rows, self._d_model, dtype=dtype, device=self._device
        )
        t0 = time.perf_counter()
        dist.broadcast(recv_buf, src=self._producer_rank, group=self._nccl_group)
        nccl_time_s = time.perf_counter() - t0

        self._inner.receive_chunk(recv_buf)
        self._cuda_log({
            "event": "nccl_recv_complete",
            "rows": recv_rows,
            "d_model": self._d_model,
            "dtype": str(dtype).removeprefix("torch."),
            "nccl_time_s": nccl_time_s,
            "pool_tokens": self._inner._min_pool_tokens(),
        })
        return True

    def _refill_inner_to_high_watermark(self, *, force_one: bool = False) -> None:
        requested = 0
        while not self._eof and (
            force_one or self._inner_needs_refill()
        ):
            force_one = False
            desired_chunks = self._desired_refill_chunks()
            received = self._request_and_recv(desired_chunks)
            if not received:
                break
            requested += 1
            for _ in range(desired_chunks - 1):
                if self._eof or not self._inner_needs_refill():
                    break
                if not self._recv_one_chunk():
                    break
                requested += 1
        if requested > 0:
            self._cuda_log({
                "event": "refill_to_high_watermark",
                "chunks": requested,
                "available_tokens": self._inner_available_tokens(),
                "pool_tokens": self._inner._min_pool_tokens(),
            })

    def _inner_needs_refill(self) -> bool:
        receiver_needs_refill = getattr(self._inner, "receiver_needs_refill", None)
        if receiver_needs_refill is not None:
            return bool(receiver_needs_refill())
        needs_refill = getattr(self._inner, "needs_refill", None)
        if needs_refill is not None:
            return bool(needs_refill())
        return self._inner._min_pool_tokens() == 0

    def _inner_available_tokens(self) -> int:
        return int(getattr(self._inner, "available_tokens", self._inner._min_pool_tokens()))

    def _desired_refill_chunks(self) -> int:
        desired = getattr(self._inner, "desired_receiver_refill_chunks", None)
        if desired is None:
            desired = getattr(self._inner, "desired_refill_chunks", None)
        if desired is None:
            return 1
        return max(1, int(desired()))

    def _cuda_log(self, record: dict) -> None:
        if self._cuda_log_path is None:
            return
        record["elapsed_s"] = time.time() - self._t_ready
        with open(self._cuda_log_path, "a") as f:
            json.dump(record, f)
            f.write("\n")


class InterruptedException(Exception):
    pass


def interrupt_callback(sig_num: Any, stack_frame: Any):  # noqa: ARG001
    raise InterruptedException()


@dataclass
class LLMSaeEvaluator(Generic[T_TRAINING_SAE]):
    model: HookedRootModule
    activations_store: ActivationsStore
    eval_batch_size_prompts: int | None
    n_eval_batches: int
    model_kwargs: dict[str, Any]

    def __call__(
        self,
        sae: T_TRAINING_SAE,
        data_provider: DataProvider,
        activation_scaler: ActivationScaler,
    ) -> dict[str, Any]:
        exclude_special_tokens = False
        if self.activations_store.exclude_special_tokens is not None:
            exclude_special_tokens = (
                self.activations_store.exclude_special_tokens.tolist()
            )

        eval_config = EvalConfig(
            batch_size_prompts=self.eval_batch_size_prompts,
            n_eval_reconstruction_batches=self.n_eval_batches,
            n_eval_sparsity_variance_batches=self.n_eval_batches,
            compute_ce_loss=True,
            compute_l2_norms=True,
            compute_sparsity_metrics=True,
            compute_variance_metrics=True,
        )

        eval_metrics, _ = run_evals(
            sae=sae,
            activation_store=self.activations_store,
            model=self.model,
            activation_scaler=activation_scaler,
            eval_config=eval_config,
            exclude_special_tokens=exclude_special_tokens,
            model_kwargs=self.model_kwargs,
        )  # not calculating featurwise metrics here.

        # Remove eval metrics that are already logged during training
        eval_metrics.pop("metrics/explained_variance", None)
        eval_metrics.pop("metrics/explained_variance_std", None)
        eval_metrics.pop("metrics/l0", None)
        eval_metrics.pop("metrics/l1", None)
        eval_metrics.pop("metrics/mse", None)

        # Remove metrics that are not useful for wandb logging
        eval_metrics.pop("metrics/total_tokens_evaluated", None)

        return eval_metrics


class LanguageModelSAETrainingRunner:
    """
    Class to run the training of a Sparse Autoencoder (SAE) on a TransformerLens model.
    """

    cfg: LanguageModelSAERunnerConfig[Any]
    model: HookedRootModule
    sae: TrainingSAE[Any] | None
    activations_store: ActivationsStore

    def __init__(
        self,
        cfg: LanguageModelSAERunnerConfig[T_TRAINING_SAE_CONFIG],
        override_dataset: HfDataset | None = None,
        override_model: HookedRootModule | None = None,
        override_sae: TrainingSAE[Any] | None = None,
        resume_from_checkpoint: Path | str | None = None,
        tp_size: int = 1,
        shared_tp_size: int | None = None,
        vllm_tp_size: int | None = None,
        sae_tp_size: int | None = None,
        sae_dp_size: int = 1,
        vllm_dp_size: int = 1,
        use_shard_routing: bool = True,
        streaming_mode: bool = False,
        quiesce_dir: Path | None = None,
        sae_pp_size: int = 1,
    ):
        if override_dataset is not None:
            logger.warning(
                f"You just passed in a dataset which will override the one specified in your configuration: {cfg.dataset_path}. As a consequence this run will not be reproducible via configuration alone."
            )
        if override_model is not None:
            logger.warning(
                f"You just passed in a model which will override the one specified in your configuration: {cfg.model_name}. As a consequence this run will not be reproducible via configuration alone."
            )

        self.cfg = cfg
        self.cached_activations_only = bool(cfg.use_cached_activations)
        self.hook_names = (
            list(cfg.hook_names)
            if cfg.hook_names is not None and len(cfg.hook_names) > 0
            else [cfg.hook_name]
        )
        if resume_from_checkpoint is not None:
            self.cfg.resume_from_checkpoint = str(resume_from_checkpoint)
        self.sae_dp_size = sae_dp_size
        self.sae_pp_size = sae_pp_size
        self.cfg.sae_pp_size = sae_pp_size
        self.is_multi_sae = (
            len(self.hook_names) > 1
            or self.sae_pp_size > 1
            or os.environ.get("SAELENS_FORCE_MULTI_SAE_TRAINER", "0") == "1"
        )
        if self.is_multi_sae and self.cfg.sae_dp_mode == "manual":
            logger.warning(
                "Multi-layer/PP SAE training does not use manual DP sync; "
                "defaulting sae_dp_mode to 'ddp'. With sae_dp_size=1 this "
                "runs without data-parallel communication."
            )
            self.cfg.sae_dp_mode = "ddp"
        self.vllm_dp_size = vllm_dp_size
        self.use_shard_routing = use_shard_routing

        # Cached-only mode: no producers, all ranks are SAE.
        if self.cached_activations_only:
            self.vllm_dp_size = 0
            vllm_dp_size = 0
            self.use_shard_routing = True
            use_shard_routing = True

        inferred_cfg_vllm_tp_size = int(
            self.cfg.model_from_pretrained_kwargs.get("tensor_parallel_size", 1)
        )
        self.shared_tp_size = shared_tp_size
        if (
            self.shared_tp_size is None
            and sae_tp_size is None
            and vllm_tp_size is None
            and tp_size > 1
            and vllm_dp_size == 1
        ):
            # Backward-compatible shared-TP semantics.
            self.shared_tp_size = tp_size

        if self.shared_tp_size is not None:
            self.sae_tp_size = self.shared_tp_size
            self.vllm_tp_size = self.shared_tp_size
            if (
                "tensor_parallel_size" in self.cfg.model_from_pretrained_kwargs
                and inferred_cfg_vllm_tp_size != self.shared_tp_size
            ):
                raise ValueError(
                    "shared_tp_size does not match "
                    "cfg.model_from_pretrained_kwargs['tensor_parallel_size']"
                )
        else:
            self.sae_tp_size = tp_size if sae_tp_size is None else sae_tp_size
            self.vllm_tp_size = (
                inferred_cfg_vllm_tp_size if vllm_tp_size is None else vllm_tp_size
            )
            if (
                "tensor_parallel_size" in self.cfg.model_from_pretrained_kwargs
                and inferred_cfg_vllm_tp_size != self.vllm_tp_size
            ):
                raise ValueError(
                    "vllm_tp_size does not match "
                    "cfg.model_from_pretrained_kwargs['tensor_parallel_size']"
                )

        self._quiesce_dir = quiesce_dir
        self.streaming_mode = streaming_mode or cfg.streaming_mode

        # GPU direct streaming validation (must run before _streaming_init)
        one_side_absent = vllm_dp_size == 0 or sae_dp_size == 0
        if cfg.streaming_use_gpu_direct:
            if not cfg.streaming_mode:
                raise ValueError("streaming_use_gpu_direct requires streaming_mode=True")
            if one_side_absent:
                logger.warning(
                    "streaming_use_gpu_direct=True but vllm_dp_size=%d sae_dp_size=%d; "
                    "one side absent, falling back to shm path",
                    vllm_dp_size, sae_dp_size,
                )
                self._use_gpu_direct = False
            elif vllm_dp_size != 1 or sae_dp_size != 1:
                raise ValueError(
                    "streaming_use_gpu_direct requires vllm_dp_size=1 and sae_dp_size=1 "
                    f"(got vllm_dp_size={vllm_dp_size}, sae_dp_size={sae_dp_size})"
                )
            elif (
                self.vllm_tp_size != 1
                or self.sae_tp_size != 1
                or self.sae_pp_size != 1
            ):
                raise ValueError(
                    "streaming_use_gpu_direct background receiver MVP requires "
                    "vllm_tp_size=1, sae_tp_size=1, and sae_pp_size=1 "
                    f"(got vllm_tp_size={self.vllm_tp_size}, "
                    f"sae_tp_size={self.sae_tp_size}, sae_pp_size={self.sae_pp_size})"
                )
            else:
                if cfg.streaming_staging_queue_capacity < 1:
                    raise ValueError(
                        "streaming_staging_queue_capacity must be >= 1"
                    )
                self._use_gpu_direct = True
        else:
            self._use_gpu_direct = False

        if self.streaming_mode:
            self._streaming_init(cfg)
            return

        # Cached-only mode: no producers; force vllm_tp=1 regardless of cfg.
        if self.cached_activations_only:
            self.vllm_tp_size = 1

        # Initialize distributed process groups for SAE TP/DP.
        # With torchrun, dist.init_process_group is already called; skip it.
        if (
            self.vllm_tp_size > 1
            or self.sae_tp_size > 1
            or sae_dp_size > 1
            or vllm_dp_size > 1
            or self.sae_pp_size > 1
        ):
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl")
            if use_shard_routing:
                from sae_lens.distributed_v2 import init_distributed_v2
                batch_size = cfg.store_batch_size_prompts * len(
                    range(cfg.context_size)[slice(*cfg.seqpos_slice)]
                )
                init_distributed_v2(
                    P=vllm_dp_size,
                    Q=sae_dp_size,
                    vllm_tp_size=self.vllm_tp_size,
                    sae_tp_size=self.sae_tp_size,
                    batch_size=batch_size,
                    sae_pp_size=self.sae_pp_size,
                    use_gpu_direct=cfg.streaming_use_gpu_direct,
                )
            elif self.shared_tp_size is not None:
                init_distributed(
                    shared_tp_size=self.shared_tp_size,
                    sae_dp_size=sae_dp_size,
                )
            else:
                init_distributed(
                    sae_tp_size=self.sae_tp_size,
                    vllm_tp_size=self.vllm_tp_size,
                    vllm_dp_size=vllm_dp_size,
                    sae_dp_size=sae_dp_size,
                )
        self._sync_run_paths_across_ranks()

        if self.cached_activations_only:
            self.sae_active = True
            self.vllm_active = False
            self.uses_split_roles = False
        elif use_shard_routing and dist.is_initialized():
            import sae_lens.distributed_v2 as v2_mod
            self.sae_active = v2_mod.is_consumer()
            self.vllm_active = v2_mod.is_producer()
            self.uses_split_roles = v2_mod.is_producer() != v2_mod.is_consumer()
        else:
            self.sae_active = is_sae_active() if dist.is_initialized() else True
            self.vllm_active = is_vllm_active() if dist.is_initialized() else True
            self.uses_split_roles = (
                self.vllm_tp_size != self.sae_tp_size
                or self.vllm_dp_size != self.sae_dp_size
            )
        self.uses_vllm_dp_fan_in = self.vllm_dp_size > self.sae_dp_size
        self.uses_matched_dp = self.vllm_dp_size == self.sae_dp_size and self.vllm_dp_size > 1

        if dist.is_initialized():
            if use_shard_routing:
                import sae_lens.distributed_v2 as v2_mod
                vllm_world_ranks = sorted(
                    r for ranks in v2_mod._producer_world_ranks.values() for r in ranks
                )
            else:
                vllm_world_ranks = get_vllm_world_ranks()
            os.environ["SAELENS_VLLM_WORLD_RANKS"] = ",".join(
                str(rank) for rank in vllm_world_ranks
            )

        # Pre-initialize vLLM parallel state when needed to avoid deadlock
        # on dist.new_group() calls that non-vLLM ranks would never enter.
        if (
            not self.cached_activations_only
            and dist.is_initialized()
            and (
                use_shard_routing
                or self.sae_tp_size > self.vllm_tp_size
                or vllm_dp_size > 1
            )
        ):
            if use_shard_routing and vllm_world_ranks:
                preinit_vllm_distributed(vllm_world_ranks, self.vllm_tp_size)
            elif not use_shard_routing:
                preinit_vllm_distributed(get_vllm_world_ranks(), self.vllm_tp_size)

        if self.uses_split_roles:
            if self.cfg.logger.log_to_wandb:
                raise ValueError(
                    "Prefix-overlap training currently requires log_to_wandb=False."
                )
            if self.cfg.n_eval_batches > 0:
                raise ValueError(
                    "Prefix-overlap training currently requires n_eval_batches=0."
                )
            if self.cfg.sae.normalize_activations == "expected_average_only_in":
                raise ValueError(
                    "Prefix-overlap training does not yet support "
                    "normalize_activations='expected_average_only_in'."
                )

        if self.is_multi_sae:
            requires_dp_wrapper = (
                self.cached_activations_only and self.sae_dp_size > 1
            ) or not self.cached_activations_only
            if requires_dp_wrapper and self.cfg.sae_dp_mode not in ("ddp", "fsdp"):
                raise ValueError("Multi-layer SAE training requires sae_dp_mode='ddp' or 'fsdp'.")
            if self.cfg.sae.normalize_activations == "expected_average_only_in":
                raise ValueError(
                    "Multi-layer SAE training does not yet support "
                    "normalize_activations='expected_average_only_in'."
                )
            if self.cfg.sae_dp_mode == "fsdp" and not dist.is_initialized():
                raise ValueError(
                    "Multi-layer SAE training with sae_dp_mode='fsdp' requires "
                    "torch distributed. Use the default/ddp mode for sae_dp_size=1."
                )
            if self.cfg.n_eval_batches > 0:
                raise ValueError("Multi-layer SAE training requires n_eval_batches=0 in v1.")
            if override_sae is not None or self.cfg.from_pretrained_path is not None:
                raise ValueError("Multi-layer SAE training does not support override/pretrained SAE in v1.")

        if (
            vllm_dp_size > 1
            and self.cfg.resume_from_checkpoint is not None
            and self.cfg.sae_dp_mode not in ("ddp", "fsdp")
        ):
            raise ValueError(
                "resume_from_checkpoint with vllm_dp_size > 1 is only "
                "supported with sae_dp_mode='ddp' or 'fsdp'"
            )
        if vllm_dp_size > 1 and self.cfg.use_cached_activations:
            raise ValueError(
                "use_cached_activations is not supported with vllm_dp_size > 1"
            )

        # Compute per-PP-stage hook subset BEFORE model/cache loading so that
        # cached mode only loads the hooks assigned to this PP stage.
        if self.is_multi_sae and self.sae_pp_size > 1 and dist.is_initialized():
            import sae_lens.distributed_v2 as v2_mod
            from sae_lens.distributed_v2 import hooks_for_pp_rank
            self._pp_hook_names = hooks_for_pp_rank(
                v2_mod.get_sae_pp_rank(), self.sae_pp_size, self.hook_names
            )
        else:
            self._pp_hook_names = list(self.hook_names)

        if override_model is None:
            if self.cached_activations_only:
                excl = self.cfg.exclude_special_tokens
                if excl is True:
                    try:
                        self.model = load_tokenizer_only_model(
                            self.cfg.model_name,
                            self.cfg.device,
                        )
                    except Exception as e:
                        raise RuntimeError(
                            f"--use-cached-activations with exclude_special_tokens=True "
                            f"requires a loadable tokenizer at model_name={self.cfg.model_name!r}. "
                            f"Either provide the tokenizer, or set exclude_special_tokens to False "
                            f"or to an explicit list of token ids."
                        ) from e
                else:
                    self.model = None  # type: ignore[assignment]
            elif self.vllm_active:
                self.model = load_model(
                    self.cfg.model_class_name,
                    self.cfg.model_name,
                    device=self.cfg.device,
                    model_from_pretrained_kwargs=self.cfg.model_from_pretrained_kwargs,
                )
            else:
                self.model = load_tokenizer_only_model(
                    self.cfg.model_name,
                    self.cfg.device,
                )
        else:
            self.model = override_model

        # Determine dataset shard params for vLLM DP.
        ds_shard_index = 0
        ds_shard_count = 1
        mixing_shard_index = 0
        if dist.is_initialized() and vllm_dp_size > 1:
            if use_shard_routing:
                import sae_lens.distributed_v2 as v2_mod
                if v2_mod.is_producer():
                    ds_shard_index = v2_mod.get_producer_idx()
                    ds_shard_count = vllm_dp_size
                if v2_mod.is_consumer():
                    mixing_shard_index = v2_mod.get_sae_dp_idx()
            else:
                ds_shard_index = get_vllm_dp_rank()
                ds_shard_count = vllm_dp_size
                mixing_shard_index = ds_shard_index

        # Cached-mode: shard rows across SAE DP only. PP rank does NOT participate so
        # all PP stages within a DP replica read identical row indices (cross-hook alignment).
        cached_shard_index = 0
        cached_shard_count = 1
        if (
            self.cached_activations_only
            and dist.is_initialized()
            and self.sae_dp_size > 1
        ):
            import sae_lens.distributed_v2 as v2_mod
            cached_shard_index = v2_mod.get_sae_dp_idx()
            cached_shard_count = self.sae_dp_size
            mixing_shard_index = cached_shard_index

        if self.cached_activations_only:
            consumer_only = False
        else:
            consumer_only = use_shard_routing and self.sae_active and not self.vllm_active

        self.activations_store = ActivationsStore.from_config(
            self.model,
            self.cfg,
            override_dataset=override_dataset,
            dataset_shard_index=ds_shard_index,
            dataset_shard_count=ds_shard_count,
            consumer_only=consumer_only,
            mixing_shard_index=mixing_shard_index,
            cached_shard_index=cached_shard_index,
            cached_shard_count=cached_shard_count,
            hook_names_override=(
                self._pp_hook_names if self.cached_activations_only else None
            ),
            skip_raw_dataset_load=self.cached_activations_only,
        )

        # Multi-SAE cached mode: ensure dict-shaped batches by setting is_multi_hook
        # eagerly (the F3 fix in activations_store keys the dict branch on this flag).
        if self.cached_activations_only and self.is_multi_sae:
            self.activations_store.is_multi_hook = True
            self.activations_store._all_hook_names = list(self.hook_names)
            self.activations_store.hook_names = list(self._pp_hook_names)

        self.sae_by_hook: dict[str, Any] = {}
        self.base_sae_by_hook: dict[str, TrainingSAE[Any]] = {}
        self.multi_hook_sae: Any | None = None
        if self.is_multi_sae:
            if self.sae_active:
                self._init_multi_saes()
            self.sae = None
            self._base_sae = None
            return

        if self.sae_active:
            if override_sae is None:
                if self.sae_tp_size > 1:
                    if self.use_shard_routing:
                        import sae_lens.distributed_v2 as v2_mod
                        tp_group = v2_mod.get_sae_tp_group()
                    else:
                        tp_group = get_tp_group()
                else:
                    tp_group = None
                self.sae = self._create_training_sae(
                    seed=self.cfg.seed,
                    tp_group=tp_group,
                    device=self.cfg.device,
                    from_pretrained_path=self.cfg.from_pretrained_path,
                )
            else:
                self.sae = override_sae
                self.sae.to(self.cfg.device)
        else:
            self.sae = None

        # _create_training_sae already shards under TP>1; the legacy
        # "init full then shard_weights" branch only applies to override_sae,
        # which we still allow callers to pass in pre-built (e.g. tests).
        if (
            self.sae is not None
            and self.sae_tp_size > 1
            and override_sae is not None
        ):
            if self.use_shard_routing:
                import sae_lens.distributed_v2 as v2_mod
                tp_group = v2_mod.get_sae_tp_group()
            else:
                tp_group = get_tp_group()
            if (
                tp_group is not None
                and hasattr(self.sae, "shard_weights")
                and getattr(self.sae, "_tp_group", None) is None
            ):
                self.sae.shard_weights(tp_group)

        # _base_sae is always the raw module before any torch DP wrapper.
        # SAE compilation must happen on this module before FSDP wrapping; training
        # still enters through the wrapper so FSDP owns parameter all-gather/reshard.
        self._base_sae = self.sae
        if (
            self._base_sae is not None
            and self.cfg.sae_dp_mode == "fsdp"
            and self.cfg.resume_from_checkpoint is not None
        ):
            self._base_sae.load_weights_from_checkpoint(
                self.cfg.resume_from_checkpoint
            )

        # Wrap after TP sharding and optional raw-module compile when sae_dp_mode
        # requests a torch DP wrapper. self.sae may become FSDP/DDP.
        if self.sae is not None and self.cfg.sae_dp_mode in ("ddp", "fsdp"):
            if not dist.is_initialized():
                raise ValueError(
                    f"sae_dp_mode='{self.cfg.sae_dp_mode}' requires an initialized "
                    "distributed process group."
                )
            sae_dp_group = get_dp_group()
            if self.use_shard_routing:
                import sae_lens.distributed_v2 as v2_mod
                sae_dp_group = v2_mod.get_sae_dp_group()
            if sae_dp_group is None or dist.get_world_size(sae_dp_group) <= 1:
                raise ValueError(
                    f"sae_dp_mode='{self.cfg.sae_dp_mode}' requires sae_dp_size > 1."
                )
            if self.cfg.sae_dp_mode == "fsdp":
                if sae_tp_size > 1:
                    raise ValueError(
                        "sae_dp_mode='fsdp' with sae_tp_size > 1 is not supported. "
                        "Use sae_tp_size=1 with FSDP."
                    )
                self._compile_sae_if_needed()
                from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                from torch.distributed.fsdp.api import ShardingStrategy
                self.sae = FSDP(
                    self._base_sae,
                    process_group=sae_dp_group,
                    sharding_strategy=ShardingStrategy.FULL_SHARD,
                    use_orig_params=True,
                    backward_prefetch=self._resolve_fsdp_backward_prefetch(),
                    forward_prefetch=self.cfg.fsdp_forward_prefetch,
                )
            else:
                from torch.nn.parallel import DistributedDataParallel as DDP
                device_ids = None
                output_device = None
                device = torch.device(self.cfg.device)
                if device.type == "cuda":
                    device_index = (
                        device.index
                        if device.index is not None
                        else torch.cuda.current_device()
                    )
                    device_ids = [device_index]
                    output_device = device_index
                self._compile_sae_if_needed()
                ddp_kwargs = self._resolve_ddp_kwargs()
                self.sae = DDP(
                    self._base_sae,
                    process_group=sae_dp_group,
                    device_ids=device_ids,
                    output_device=output_device,
                    **ddp_kwargs,
                )
        else:
            self._compile_sae_if_needed()

    def _create_training_sae(
        self,
        *,
        seed: int,
        tp_group: "dist.ProcessGroup | None",
        device: str,
        from_pretrained_path: str | None = None,
        resume_checkpoint_path: str | None = None,
        hook_metadata_overrides: dict[str, Any] | None = None,
    ) -> TrainingSAE[Any]:
        """Single SAE-construction code path used by every runner entry point.

        TP=1: behaves exactly like the legacy ``temporary_seed +
        TrainingSAE.from_dict/load_from_disk + .to(device)`` sequence.

        TP>1: TopK-only. Calls ``TopKTrainingSAE.from_config_sharded`` so each
        rank only ever materializes its local shard of W_enc/W_dec/b_enc; the
        legacy "full init then shard_weights" path is not used. Pretrained or
        resume checkpoints are read via the TP slice loader (only the local
        rank's slice ever lands on the device).
        """
        from sae_lens.saes.topk_sae import TopKTrainingSAE, TopKTrainingSAEConfig

        cfg_dict = self.cfg.get_training_sae_cfg_dict()
        sae_cfg = TrainingSAEConfig.from_dict(cfg_dict)

        is_tp = tp_group is not None and dist.get_world_size(tp_group) > 1
        if is_tp and not isinstance(sae_cfg, TopKTrainingSAEConfig):
            raise NotImplementedError(
                f"sae_tp_size>1 only supports TopK; got {type(sae_cfg).__name__}"
            )

        sae_cfg.device = device
        with temporary_seed(seed):
            if is_tp:
                assert isinstance(sae_cfg, TopKTrainingSAEConfig)
                sae = TopKTrainingSAE.from_config_sharded(sae_cfg, tp_group)  # type: ignore[arg-type]
            elif from_pretrained_path is not None:
                sae = TrainingSAE.load_from_disk(from_pretrained_path, device)
            else:
                sae = TrainingSAE.from_dict(sae_cfg.to_dict())

        if not is_tp:
            sae.to(device)

        if hook_metadata_overrides:
            for key, value in hook_metadata_overrides.items():
                setattr(sae.cfg.metadata, key, value)

        if is_tp and from_pretrained_path is not None:
            sae.load_weights_from_checkpoint(from_pretrained_path)
        if resume_checkpoint_path is not None:
            sae.load_weights_from_checkpoint(resume_checkpoint_path)

        return sae

    def _multi_sae_tp_group(self) -> "dist.ProcessGroup | None":
        if self.sae_tp_size <= 1:
            return None
        if self.use_shard_routing:
            import sae_lens.distributed_v2 as v2_mod

            return v2_mod.get_sae_tp_group()
        return get_tp_group()

    def _create_multi_sae_for_hook(
        self,
        idx: int,
        hook_name: str,
    ) -> TrainingSAE[Any]:
        seed = (
            self.cfg.seed
            if self.cfg.multi_sae_seed_mode == "same"
            else self.cfg.seed + idx
        )
        return self._create_training_sae(
            seed=seed,
            tp_group=self._multi_sae_tp_group(),
            device=self.cfg.device,
            hook_metadata_overrides={
                "hook_name": hook_name,
                "hook_head_index": self.cfg.hook_head_index,
                "dataset_path": self.cfg.dataset_path,
                "model_name": self.cfg.model_name,
                "model_class_name": self.cfg.model_class_name,
                "context_size": self.cfg.context_size,
                "seqpos_slice": self.cfg.seqpos_slice,
                "prepend_bos": self.cfg.prepend_bos,
                "exclude_special_tokens": self.cfg.exclude_special_tokens,
            },
        )

    def _init_multi_saes(self) -> None:
        if self.cfg.sae_dp_mode == "fsdp" and not dist.is_initialized():
            raise ValueError("Multi-layer SAE training with FSDP requires torch distributed.")

        sae_dp_group = get_dp_group() if dist.is_initialized() else None
        if self.use_shard_routing and dist.is_initialized():
            import sae_lens.distributed_v2 as v2_mod

            sae_dp_group = v2_mod.get_sae_dp_group()
        sae_dp_world_size = (
            dist.get_world_size(sae_dp_group)
            if sae_dp_group is not None and dist.is_initialized()
            else 1
        )
        if self.cfg.sae_dp_mode == "fsdp" and self.sae_tp_size > 1:
            raise ValueError(
                "sae_dp_mode='fsdp' with sae_tp_size > 1 is not supported. "
                "Use sae_tp_size=1 with FSDP."
            )
        if self.cfg.sae_dp_mode == "fsdp" and sae_dp_group is None:
            raise ValueError("Multi-layer SAE training with FSDP requires an SAE DP group.")
        ddp_kwargs_multi = (
            self._resolve_ddp_kwargs()
            if self.cfg.sae_dp_mode == "ddp" and sae_dp_world_size > 1
            else {}
        )

        # PP hook subsetting: only create SAEs for this stage's hooks.
        if self.sae_pp_size > 1 and dist.is_initialized():
            import sae_lens.distributed_v2 as v2_mod
            from sae_lens.distributed_v2 import hooks_for_pp_rank

            pp_rank = v2_mod.get_sae_pp_rank()
            self._pp_hook_names = hooks_for_pp_rank(pp_rank, self.sae_pp_size, self.hook_names)
        else:
            self._pp_hook_names = list(self.hook_names)

        # Store all hooks for producer-side data generation
        if hasattr(self, 'activations_store') and self.activations_store is not None:
            self.activations_store._all_hook_names = list(self.hook_names)
            self.activations_store.hook_names = list(self._pp_hook_names)
            self.activations_store.is_multi_hook = True

        if self.cfg.multi_sae_distributed_architecture == "unified_multi_hook":
            self._init_multi_saes_unified(sae_dp_group, sae_dp_world_size)
            return

        for idx, hook_name in enumerate(self._pp_hook_names):
            sae = self._create_multi_sae_for_hook(idx, hook_name)
            self.base_sae_by_hook[hook_name] = sae

            wrapped: Any
            if self.cfg.sae_dp_mode == "fsdp":
                from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                from torch.distributed.fsdp.api import ShardingStrategy

                wrapped = FSDP(
                    sae,
                    process_group=sae_dp_group,
                    sharding_strategy=ShardingStrategy.FULL_SHARD,
                    use_orig_params=True,
                    backward_prefetch=self._resolve_fsdp_backward_prefetch(),
                    forward_prefetch=self.cfg.fsdp_forward_prefetch,
                )
            elif sae_dp_world_size > 1:
                from torch.nn.parallel import DistributedDataParallel as DDP

                device_ids = None
                output_device = None
                device = torch.device(self.cfg.device)
                if device.type == "cuda":
                    device_index = (
                        device.index
                        if device.index is not None
                        else torch.cuda.current_device()
                    )
                    device_ids = [device_index]
                    output_device = device_index
                wrapped = DDP(
                    sae,
                    process_group=sae_dp_group,
                    device_ids=device_ids,
                    output_device=output_device,
                    **ddp_kwargs_multi,
                )
            else:
                wrapped = sae
            self.sae_by_hook[hook_name] = wrapped

    def _init_multi_saes_unified(
        self,
        sae_dp_group: "dist.ProcessGroup | None",
        sae_dp_world_size: int,
    ) -> None:
        raw_sae_by_hook: dict[str, TrainingSAE[Any]] = {}
        for idx, hook_name in enumerate(self._pp_hook_names):
            sae = self._create_multi_sae_for_hook(idx, hook_name)
            self.base_sae_by_hook[hook_name] = sae
            raw_sae_by_hook[hook_name] = sae

        raw_multi_hook_sae = MultiHookSAE(list(self._pp_hook_names), raw_sae_by_hook)
        self.sae_by_hook = dict(raw_sae_by_hook)
        self.multi_hook_sae = raw_multi_hook_sae

        if self.cfg.sae_dp_mode == "fsdp":
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp.api import ShardingStrategy
            from torch.distributed.fsdp.wrap import ModuleWrapPolicy

            self.multi_hook_sae = FSDP(
                raw_multi_hook_sae,
                process_group=sae_dp_group,
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                auto_wrap_policy=ModuleWrapPolicy({TrainingSAE}),
                use_orig_params=True,
                backward_prefetch=self._resolve_fsdp_backward_prefetch(),
                forward_prefetch=self.cfg.fsdp_forward_prefetch,
            )
        elif sae_dp_world_size > 1:
            from torch.nn.parallel import DistributedDataParallel as DDP

            device_ids = None
            output_device = None
            device = torch.device(self.cfg.device)
            if device.type == "cuda":
                device_index = (
                    device.index
                    if device.index is not None
                    else torch.cuda.current_device()
                )
                device_ids = [device_index]
                output_device = device_index
            self.multi_hook_sae = DDP(
                raw_multi_hook_sae,
                process_group=sae_dp_group,
                device_ids=device_ids,
                output_device=output_device,
                **self._resolve_ddp_kwargs(),
            )

    def _resolve_ddp_kwargs(self) -> dict[str, Any]:
        ddp_kwargs: dict[str, Any] = {}
        if self.cfg.ddp_broadcast_buffers is not None:
            ddp_kwargs["broadcast_buffers"] = self.cfg.ddp_broadcast_buffers
        if self.cfg.ddp_find_unused_parameters is not None:
            ddp_kwargs["find_unused_parameters"] = self.cfg.ddp_find_unused_parameters
        if self.cfg.ddp_gradient_as_bucket_view is not None:
            ddp_kwargs["gradient_as_bucket_view"] = self.cfg.ddp_gradient_as_bucket_view
        if self.cfg.ddp_static_graph is not None:
            ddp_kwargs["static_graph"] = self.cfg.ddp_static_graph
        if self.cfg.ddp_bucket_cap_mb is not None:
            ddp_kwargs["bucket_cap_mb"] = self.cfg.ddp_bucket_cap_mb

        if (
            ddp_kwargs.get("static_graph") is True
            and ddp_kwargs.get("find_unused_parameters") is True
        ):
            message = (
                "DDP config conflict: static_graph=True with "
                "find_unused_parameters=True is not recommended."
            )
            if self.cfg.ddp_config_strict:
                raise ValueError(message)
            logger.warning(message + " Overriding find_unused_parameters=False.")
            ddp_kwargs["find_unused_parameters"] = False

        effective = {
            "broadcast_buffers": ddp_kwargs.get("broadcast_buffers", "torch_default"),
            "find_unused_parameters": ddp_kwargs.get(
                "find_unused_parameters", "torch_default"
            ),
            "gradient_as_bucket_view": ddp_kwargs.get(
                "gradient_as_bucket_view", "torch_default"
            ),
            "static_graph": ddp_kwargs.get("static_graph", "torch_default"),
            "bucket_cap_mb": ddp_kwargs.get("bucket_cap_mb", "torch_default"),
        }
        logger.info(f"Effective DDP config: {effective}")
        return ddp_kwargs

    def _resolve_fsdp_backward_prefetch(self) -> Any:
        if self.cfg.fsdp_backward_prefetch == "none":
            value = None
        else:
            from torch.distributed.fsdp.api import BackwardPrefetch

            value = (
                BackwardPrefetch.BACKWARD_PRE
                if self.cfg.fsdp_backward_prefetch == "backward_pre"
                else BackwardPrefetch.BACKWARD_POST
            )
        logger.info(
            "Effective FSDP config: "
            f"{{'backward_prefetch': {self.cfg.fsdp_backward_prefetch!r}, "
            f"'forward_prefetch': {self.cfg.fsdp_forward_prefetch!r}}}"
        )
        return value

    def _sync_run_paths_across_ranks(self) -> None:
        if not dist.is_initialized() or dist.get_world_size() <= 1:
            return
        run_paths = [self.cfg.checkpoint_path, self.cfg.output_path]
        dist.broadcast_object_list(run_paths, src=0)
        self.cfg.checkpoint_path = run_paths[0]
        self.cfg.output_path = run_paths[1]

    def run(self):
        """
        Run the training of the SAE.
        """
        if self.streaming_mode:
            import sae_lens.distributed_streaming as ds
            if ds.is_producer():
                if self._use_gpu_direct:
                    self._run_gpu_direct_producer_loop()
                else:
                    self._run_streaming_producer_loop()
                return None
            else:
                if self._use_gpu_direct:
                    return self._run_gpu_direct_consumer_loop()
                else:
                    return self._run_streaming_consumer_loop()

        if self.use_shard_routing and self.vllm_active and not self.sae_active:
            self._load_producer_resume_state_if_needed()
            self._run_producer_helper_loop_v2()
            return None

        if not self.sae_active:
            if os.environ.get("SAELENS_DEBUG_PREFIX_TP") == "1":
                rank = dist.get_rank() if dist.is_initialized() else -1
                line = f"[prefix-debug rank{rank}] entering helper loop\n"
                with open(f"/tmp/saelens_debug_rank{rank}.log", "a") as f:
                    f.write(line)
                print(line, end="", flush=True)
            self._run_vllm_helper_loop()
            return None

        if self.is_multi_sae and len(getattr(self, "_pp_hook_names", [])) == 0:
            if self.use_shard_routing and dist.is_initialized():
                self._load_producer_resume_state_if_needed()
                self._run_producer_helper_loop_v2()
            return {}

        if self.is_multi_sae:
            return self._run_multi_sae()

        assert self.sae is not None
        self._set_sae_metadata()
        if self.cfg.logger.log_to_wandb:
            wandb.init(
                project=self.cfg.logger.wandb_project,
                entity=self.cfg.logger.wandb_entity,
                config=self.cfg.to_dict(),
                name=self.cfg.logger.run_name,
                id=self.cfg.logger.wandb_id,
            )

        evaluator = LLMSaeEvaluator(
            model=self.model,
            activations_store=self.activations_store,
            eval_batch_size_prompts=self.cfg.eval_batch_size_prompts,
            n_eval_batches=self.cfg.n_eval_batches,
            model_kwargs=self.cfg.model_kwargs,
        )

        sae_dp_group = get_dp_group()
        if self.use_shard_routing:
            import sae_lens.distributed_v2 as v2_mod
            sae_dp_group = v2_mod.get_sae_dp_group()
        # In FSDP mode the dp_group is already embedded in the FSDP wrapper; we
        # still pass it so the trainer can use it for sparsity/firing sync and rank checks.
        trainer = SAETrainer(
            sae=self.sae,
            base_sae=self._base_sae,
            data_provider=self.activations_store,
            evaluator=evaluator,
            save_checkpoint_fn=self.save_checkpoint,
            cfg=self.cfg.to_sae_trainer_config(),
            dp_group=(
                sae_dp_group
                if sae_dp_group is not None and dist.get_world_size(sae_dp_group) > 1
                else None
            ),
            token_count_weighted_dp=self.use_shard_routing,
            append_logs=self.cfg.resume_from_checkpoint is not None
            or self.cfg.append_history_logs,
        )

        if self.cfg.resume_from_checkpoint is not None:
            logger.info(f"Resuming from checkpoint: {self.cfg.resume_from_checkpoint}")
            trainer.load_trainer_state(self.cfg.resume_from_checkpoint)
            if self.cfg.sae_dp_mode != "fsdp":
                self._base_sae.load_weights_from_checkpoint(
                    self.cfg.resume_from_checkpoint
                )
            self.activations_store.load_from_checkpoint(self.cfg.resume_from_checkpoint)

        self._compile_if_needed()
        sae = self.run_trainer_with_interruption_handling(trainer)

        if self.cfg.output_path is not None:
            self.save_final_sae(
                sae=sae,
                output_path=self.cfg.output_path,
                log_feature_sparsity=trainer.log_feature_sparsity,
            )

        if self.cfg.logger.log_to_wandb:
            wandb.finish()

        return sae

    def _run_multi_sae(self) -> dict[str, TrainingSAE[Any]]:
        sae_dp_group = get_dp_group() if dist.is_initialized() else None
        if self.use_shard_routing and dist.is_initialized():
            import sae_lens.distributed_v2 as v2_mod

            sae_dp_group = v2_mod.get_sae_dp_group()

        pp_hooks = self._pp_hook_names if hasattr(self, "_pp_hook_names") else self.hook_names
        trainer = MultiSAETrainer(
            hook_names=pp_hooks,
            sae_by_hook=self.sae_by_hook,
            base_sae_by_hook=self.base_sae_by_hook,
            multi_hook_sae=self.multi_hook_sae,
            data_provider=self.activations_store,
            save_checkpoint_fn=self.save_checkpoint,
            cfg=self.cfg.to_sae_trainer_config(),
            dp_group=sae_dp_group,
            token_count_weighted_dp=self.use_shard_routing,
            sae_dp_mode=self.cfg.sae_dp_mode,
            backward_mode=self.cfg.multi_sae_backward_mode,
            seed_mode=self.cfg.multi_sae_seed_mode,
            append_logs=self.cfg.resume_from_checkpoint is not None
            or self.cfg.append_history_logs,
        )
        if self.cfg.resume_from_checkpoint is not None:
            trainer.load_trainer_state(self.cfg.resume_from_checkpoint)
            self.activations_store.load_from_checkpoint(self.cfg.resume_from_checkpoint)

        result = self.run_multi_trainer_with_interruption_handling(trainer)
        if self.cfg.output_path is not None:
            trainer.save_final(self.cfg.output_path)
        return result

    def _run_vllm_helper_loop(self) -> None:
        """Pump activations for helper-only ranks (vllm_active and not sae_active).

        With m:1 fan-in, helper DP root ranks (vllm_dp_rank > 0, vllm_tp_rank == 0)
        also send their raw batch to the cluster SAE root via Gloo P2P.
        """
        import math

        vllm_dp_size = get_vllm_dp_size() if dist.is_initialized() else 1
        sae_dp_size = get_sae_dp_size() if dist.is_initialized() else 1
        is_dp_root = is_vllm_dp_root() if dist.is_initialized() else False
        vllm_dp_rank = get_vllm_dp_rank() if dist.is_initialized() else 0
        batch_size = self.cfg.store_batch_size_prompts
        ctx_size = self.activations_store.training_context_size

        # n = vLLM replicas per cluster; each cluster feeds one SAE replica.
        n = vllm_dp_size // sae_dp_size if sae_dp_size > 0 else vllm_dp_size

        # Approximate number of raw batches this helper should produce.
        if self.uses_vllm_dp_fan_in:
            tokens_per_batch = batch_size * ctx_size
            # Each helper produces total_tokens / (n * tokens_per_batch) batches,
            # because the cluster's SAE root will yield n batches per outer iteration.
            target_batches = math.ceil(
                self.cfg.total_training_tokens / (tokens_per_batch * n)
            )
        else:
            target_batches = None  # Legacy: use token count

        n_batches_done = 0
        n_training_samples = 0
        while True:
            if target_batches is not None and n_batches_done >= target_batches:
                break
            if target_batches is None and n_training_samples >= self.cfg.total_training_tokens:
                break

            if os.environ.get("SAELENS_DEBUG_PREFIX_TP") == "1":
                rank = dist.get_rank() if dist.is_initialized() else -1
                line = (
                    f"[prefix-debug rank{rank}] helper batch start n={n_batches_done}\n"
                )
                with open(f"/tmp/saelens_debug_rank{rank}.log", "a") as f:
                    f.write(line)
                print(line, end="", flush=True)

            if self.uses_vllm_dp_fan_in:
                # Directly call get_raw_llm_batch to participate in intra-group
                # broadcasts. Do NOT go through mixing buffer.
                raw_acts, raw_tokens = self.activations_store.get_raw_llm_batch()
                n_batches_done += 1

                # Non-root helper DP root: send raw batch to cluster SAE root.
                # The cluster SAE root is the rank of the first vLLM DP replica
                # in this cluster (vllm_dp_rank % n == 0 → cluster root).
                cluster_first_vllm_dp = (vllm_dp_rank // n) * n
                is_cluster_sae_root_vllm = vllm_dp_rank == cluster_first_vllm_dp
                if is_dp_root and not is_cluster_sae_root_vllm:
                    p2p_group = get_vllm_dp_p2p_group()
                    if p2p_group is None:
                        raise RuntimeError(
                            "vLLM DP P2P group is not initialized; refusing to fall back to the default process group."
                        )
                    cluster_sae_root = get_sae_root_rank()
                    raw_acts = raw_acts.to("cpu").contiguous()
                    dist.send(raw_acts, dst=cluster_sae_root, group=p2p_group)
                    if raw_tokens is not None:
                        raw_tokens = raw_tokens.to("cpu").contiguous()
                        dist.send(raw_tokens, dst=cluster_sae_root, group=p2p_group)
            else:
                # Legacy: go through mixing buffer to stay in sync with
                # existing split-role broadcasts.
                batch = next(self.activations_store)
                n_training_samples += batch.shape[0]

            if os.environ.get("SAELENS_DEBUG_PREFIX_TP") == "1":
                rank = dist.get_rank() if dist.is_initialized() else -1
                line = (
                    f"[prefix-debug rank{rank}] helper batch done n={n_batches_done}\n"
                )
                with open(f"/tmp/saelens_debug_rank{rank}.log", "a") as f:
                    f.write(line)
                print(line, end="", flush=True)

    def _run_producer_helper_loop_v2(self) -> None:
        """Producer-only loop for shard-routing mode (use_shard_routing=True).

        All producer TP ranks participate in get_raw_llm_batch() each step.
        Producer TP roots then participate in the same per-consumer NCCL P2P
        exchange phase as consumer ranks.
        Exits after total_producer_steps steps to stay in lockstep with consumers.
        """
        ctx_size = self.activations_store.training_context_size
        remaining_training_tokens = max(
            self.cfg.total_training_tokens - self._resume_training_samples(),
            0,
        )
        rows_per_consumer_step = self._v2_rows_per_consumer_step(ctx_size)
        buffer_size = self.cfg.n_batches_in_buffer * ctx_size
        total_producer_steps = max(
            self._mixing_buffer_source_steps_needed(
                target_samples=remaining_training_tokens,
                source_batch_size=rows_per_step,
                buffer_size=buffer_size,
                train_batch_size=self.cfg.train_batch_size_tokens,
                mix_fraction=self.cfg.activations_mixing_fraction,
            )
            for rows_per_step in rows_per_consumer_step
        )

        for _ in range(total_producer_steps):
            _local_slices, outgoing = self.activations_store._run_producer_phase2_v2()
            self.activations_store._run_nccl_p2p_exchange_v2(outgoing)

    def _v2_rows_per_consumer_step(self, ctx_size: int) -> list[int]:
        try:
            import sae_lens.distributed_v2 as v2_mod

            if v2_mod._initialized:
                from sae_lens.shard_routing import routes_for_consumer

                return [
                    sum(
                        route.row_end - route.row_start
                        for route in routes_for_consumer(v2_mod.get_routing_table(), c)
                    )
                    for c in range(v2_mod.get_sae_dp_size())
                ]
        except ImportError:
            pass

        total_rows = self.cfg.store_batch_size_prompts * ctx_size * self.vllm_dp_size
        rows_per_consumer = (total_rows + self.sae_dp_size - 1) // self.sae_dp_size
        return [rows_per_consumer]

    @staticmethod
    def _mixing_buffer_source_steps_needed(
        *,
        target_samples: int,
        source_batch_size: int,
        buffer_size: int,
        train_batch_size: int,
        mix_fraction: float,
    ) -> int:
        if target_samples <= 0:
            return 0
        if source_batch_size <= 0:
            raise ValueError("source_batch_size must be > 0")
        if buffer_size < train_batch_size:
            raise ValueError(
                "buffer_size must be greater than or equal to train_batch_size"
            )
        if not 0 <= mix_fraction <= 1:
            raise ValueError("mix_fraction must be in [0, 1]")

        source_steps = 0
        yielded_samples = 0
        storage_samples = 0
        while yielded_samples < target_samples:
            source_steps += 1
            storage_samples += source_batch_size
            if storage_samples < buffer_size:
                continue

            keep_for_mixing = int(buffer_size * mix_fraction)
            num_to_serve = storage_samples - keep_for_mixing
            num_serving_batches = max(1, num_to_serve // train_batch_size)
            serving_cutoff = num_serving_batches * train_batch_size
            yielded_samples += serving_cutoff
            storage_samples -= serving_cutoff

        return source_steps

    def _resume_checkpoint_path(self) -> Path | None:
        if self.cfg.resume_from_checkpoint is None:
            return None
        return Path(self.cfg.resume_from_checkpoint)

    def _resume_training_samples(self) -> int:
        checkpoint_path = self._resume_checkpoint_path()
        if checkpoint_path is None:
            return 0
        state_dict = torch.load(
            checkpoint_path / TRAINER_STATE_FILENAME,
            map_location="cpu",
        )
        return int(state_dict.get("n_training_samples", 0))

    def _load_producer_resume_state_if_needed(self) -> None:
        checkpoint_path = self._resume_checkpoint_path()
        if checkpoint_path is None:
            return
        self.activations_store.load_from_checkpoint(checkpoint_path)

    def save_final_sae(
        self,
        sae: TrainingSAE[Any],
        output_path: str,
        log_feature_sparsity: torch.Tensor | None = None,
    ):
        tp_group = getattr(sae, "_tp_group", None)
        tp_rank = dist.get_rank(tp_group) if tp_group is not None else 0
        dp_group = get_dp_group()
        if self.use_shard_routing:
            import sae_lens.distributed_v2 as v2_mod
            dp_group = v2_mod.get_sae_dp_group()
        dp_rank = (
            dist.get_rank(dp_group)
            if dp_group is not None and dist.get_world_size(dp_group) > 1
            else 0
        )

        base_output_path = Path(output_path)
        base_output_path.mkdir(exist_ok=True, parents=True)

        if self.cfg.sae_dp_mode == "fsdp":
            # FSDP state dict gather is a collective — all DP ranks must call it.
            # rank0_only=True means only dp_rank 0 receives a non-empty result.
            from torch.distributed.fsdp import FullStateDictConfig, StateDictType
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            fsdp_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self.sae, StateDictType.FULL_STATE_DICT, fsdp_cfg):
                state_dict = self.sae.state_dict()
            # Non-rank-0 processes have an empty dict; nothing to save.
            if dp_rank != 0:
                return
            sae.process_state_dict_for_saving_inference(state_dict)
            weights_path = base_output_path / SAE_WEIGHTS_FILENAME
            cfg_path = base_output_path / SAE_CFG_FILENAME
            if tp_rank == 0:
                save_file(state_dict, weights_path)
                config = sae.cfg.get_inference_sae_cfg_dict()
                with open(cfg_path, "w") as f:
                    json.dump(config, f)
            if tp_group is not None:
                dist.barrier(group=tp_group)
        else:
            if dp_rank != 0:
                return
            weights_path, cfg_path = sae.save_inference_model(str(base_output_path))
            if tp_rank != 0:
                return

        sparsity_path = None
        if log_feature_sparsity is not None:
            sparsity_path = base_output_path / SPARSITY_FILENAME
            save_file({"sparsity": log_feature_sparsity}, sparsity_path)

        runner_config = self.cfg.to_dict()
        with open(base_output_path / RUNNER_CFG_FILENAME, "w") as f:
            json.dump(runner_config, f)

        if self.cfg.logger.log_to_wandb:
            self.cfg.logger.log(
                self,
                weights_path,
                cfg_path,
                sparsity_path=sparsity_path,
                wandb_aliases=["final_model"],
            )

    def _set_sae_metadata(self):
        assert self._base_sae is not None
        self._base_sae.cfg.metadata.dataset_path = self.cfg.dataset_path
        self._base_sae.cfg.metadata.hook_name = self.cfg.hook_name
        self._base_sae.cfg.metadata.model_name = self.cfg.model_name
        self._base_sae.cfg.metadata.model_class_name = self.cfg.model_class_name
        self._base_sae.cfg.metadata.hook_head_index = self.cfg.hook_head_index
        self._base_sae.cfg.metadata.context_size = self.cfg.context_size
        self._base_sae.cfg.metadata.seqpos_slice = self.cfg.seqpos_slice
        self._base_sae.cfg.metadata.model_from_pretrained_kwargs = (
            self.cfg.model_from_pretrained_kwargs
        )
        self._base_sae.cfg.metadata.prepend_bos = self.cfg.prepend_bos
        self._base_sae.cfg.metadata.exclude_special_tokens = self.cfg.exclude_special_tokens
        self._base_sae.cfg.metadata.sequence_separator_token = (
            self.cfg.sequence_separator_token
        )
        self._base_sae.cfg.metadata.disable_concat_sequences = (
            self.cfg.disable_concat_sequences
        )

    def _compile_sae_if_needed(self):
        if not self.cfg.compile_sae or self._base_sae is None:
            return

        backend = "aot_eager" if self.cfg.device == "mps" else "inductor"
        compiled_training_forward = torch.compile(
            self._base_sae.training_forward_pass,
            mode=self.cfg.sae_compilation_mode,
            backend=backend,
        )
        self._base_sae.training_forward_pass = (  # type: ignore[method-assign]
            compiled_training_forward
        )

    def _compile_if_needed(self):
        # Compile model. SAE compilation is done before FSDP/DDP wrapping so it
        # targets the raw module, not the distributed wrapper.
        #  torch.compile can provide significant speedups (10-20% in testing)
        # using max-autotune gives the best speedups but:
        # (a) increases VRAM usage,
        # (b) can't be used on both SAE and LM (some issue with cudagraphs), and
        # (c) takes some time to compile
        # optimal settings seem to be:
        # use max-autotune on SAE and max-autotune-no-cudagraphs on LM
        # (also pylance seems to really hate this)
        if self.cfg.compile_llm:
            self.model = torch.compile(
                self.model,
                mode=self.cfg.llm_compilation_mode,
            )  # type: ignore

    def run_trainer_with_interruption_handling(
        self, trainer: SAETrainer[TrainingSAE[TrainingSAEConfig], TrainingSAEConfig]
    ):
        try:
            # signal handlers (if preempted)
            signal.signal(signal.SIGINT, interrupt_callback)
            signal.signal(signal.SIGTERM, interrupt_callback)

            # train SAE
            sae = trainer.fit()

        except (KeyboardInterrupt, InterruptedException):
            if self.cfg.checkpoint_path is not None:
                logger.warning("interrupted, saving progress")
                checkpoint_path = Path(self.cfg.checkpoint_path) / str(
                    trainer.n_training_samples
                )
                self.save_checkpoint(checkpoint_path)
                logger.info("done saving")
            raise

        return sae

    def run_multi_trainer_with_interruption_handling(
        self, trainer: MultiSAETrainer
    ) -> dict[str, TrainingSAE[Any]]:
        try:
            signal.signal(signal.SIGINT, interrupt_callback)
            signal.signal(signal.SIGTERM, interrupt_callback)
            return trainer.fit()
        except (KeyboardInterrupt, InterruptedException):
            if self.cfg.checkpoint_path is not None:
                logger.warning("interrupted, saving multi-SAE progress")
                trainer.save_checkpoint(checkpoint_name=str(trainer.n_training_samples))
                logger.info("done saving")
            raise

    # ------------------------------------------------------------------
    # Streaming mode (v1) — sae_dp=1 only
    # ------------------------------------------------------------------

    def _streaming_init(self, cfg: LanguageModelSAERunnerConfig[T_TRAINING_SAE_CONFIG]) -> None:
        import sae_lens.distributed_streaming as ds
        from sae_lens.training.shared_activation_buffer import SharedActivationBuffer
        from sae_lens.util import str_to_dtype

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        self.device = torch.device(f"cuda:{local_rank}")

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

        ds.init_distributed_streaming(
            vllm_tp=self.vllm_tp_size,
            vllm_dp=self.vllm_dp_size,
            sae_tp=self.sae_tp_size,
            sae_dp=self.sae_dp_size,
            sae_pp_size=self.sae_pp_size,
            use_gpu_direct=getattr(self, "_use_gpu_direct", False),
        )

        # Buffer name: rank 0 generates, all ranks receive via CUDA tensor broadcast (NCCL)
        buffer_name = cfg.streaming_buffer_name
        if not buffer_name:
            name_buf = torch.zeros(32, dtype=torch.int32, device=self.device)
            if dist.get_rank() == 0:
                uid = uuid.uuid4().hex[:24]
                for i, c in enumerate(uid):
                    name_buf[i] = ord(c)
            dist.broadcast(name_buf, src=0)
            buffer_name = "sae_buf_" + "".join(
                chr(int(c)) for c in name_buf.tolist() if c > 0
            )

        self._streaming_buffer_name = buffer_name
        self._streaming_num_hooks = len(self.hook_names)
        target_chunks = math.ceil(cfg.training_tokens / cfg.streaming_chunk_size_tokens)

        # Multi-hook: each chunk stores all hooks' activations concatenated.
        buf_chunk_size = cfg.streaming_chunk_size_tokens * self._streaming_num_hooks

        if getattr(self, "_use_gpu_direct", False):
            self._streaming_buffer = None
            dist.barrier()
        else:
            # True when no existing buffer was provided — we just generated a fresh name.
            # False on restart — cfg.streaming_buffer_name was set from control state.
            is_new_buffer = not cfg.streaming_buffer_name

            # On first run: rank 0 creates the buffer, others attach after barrier.
            # On restart: all ranks attach to the existing buffer (READY chunks preserved).
            if is_new_buffer and dist.get_rank() == 0:
                buffer_dtype = str_to_dtype(cfg.dtype)
                SharedActivationBuffer.assert_has_space(
                    num_chunks=cfg.streaming_num_chunks,
                    chunk_size_tokens=buf_chunk_size,
                    d_model=cfg.sae.d_in,
                    dtype=buffer_dtype,
                )
                self._streaming_buffer = SharedActivationBuffer(
                    name=buffer_name,
                    num_chunks=cfg.streaming_num_chunks,
                    chunk_size_tokens=buf_chunk_size,
                    d_model=cfg.sae.d_in,
                    num_producers=self.vllm_dp_size,
                    target_chunks=target_chunks,
                    create=True,
                    dtype=buffer_dtype,
                )
            dist.barrier()
            if not (is_new_buffer and dist.get_rank() == 0):
                self._streaming_buffer = SharedActivationBuffer(
                    name=buffer_name,
                    num_chunks=cfg.streaming_num_chunks,
                    chunk_size_tokens=buf_chunk_size,
                    d_model=cfg.sae.d_in,
                    num_producers=self.vllm_dp_size,
                    target_chunks=target_chunks,
                    create=False,
                )

        self.sae_active = ds.is_consumer()
        self.vllm_active = ds.is_producer()
        self.uses_split_roles = True
        self.uses_vllm_dp_fan_in = False
        self.uses_matched_dp = False

        # Pre-initialize vLLM parallel state on ALL ranks before any rank calls LLM().
        # vLLM creates 5+ process groups (WORLD, TP, DCP, PCP, PP, DP) even with tp=1;
        # these are collective, so all ranks must participate — even non-vLLM consumers.
        # SAELENS_VLLM_WORLD_RANKS tells vLLM's init_distributed_environment the subset
        # of world ranks that belong to vLLM, so its world-size check passes when the
        # process group was pre-initialized with fewer ranks than torch world_size.
        vllm_world_ranks = list(range(self.vllm_dp_size * self.vllm_tp_size))
        os.environ["SAELENS_VLLM_WORLD_RANKS"] = ",".join(str(r) for r in vllm_world_ranks)
        # Skip preinit when there are no vLLM ranks (vllm_dp=0 topology).
        # No LLM() is constructed in that case, so no deadlock can occur.
        if vllm_world_ranks:
            preinit_vllm_distributed(vllm_world_ranks, self.vllm_tp_size)

        self._sync_run_paths_across_ranks()
        self._init_streaming_logger()

        if ds.is_producer():
            self._streaming_init_producer(cfg)
        else:
            self._streaming_init_consumer(cfg)

    def _streaming_init_producer(
        self, cfg: LanguageModelSAERunnerConfig[T_TRAINING_SAE_CONFIG]
    ) -> None:
        import sae_lens.distributed_streaming as ds

        self.model = load_model(  # type: ignore[assignment]
            cfg.model_class_name,
            cfg.model_name,
            device=str(self.device),
            model_from_pretrained_kwargs=cfg.model_from_pretrained_kwargs,
        )
        self.activations_store = ActivationsStore.from_config(
            self.model,
            cfg,
            dataset_shard_index=ds.get_producer_idx(),
            dataset_shard_count=ds.get_vllm_dp_size(),
        )
        if self._quiesce_dir is not None:
            best_state = self._find_best_producer_dataset_state(self._quiesce_dir)
            if best_state is not None:
                self.activations_store.load_from_checkpoint(best_state)
        self.sae = None
        self._base_sae = None

    @staticmethod
    def _find_best_producer_dataset_state(qdir: Path) -> Path | None:
        """Find the producer dataset state with the highest n_dataset_processed.

        When vllm_dp changes across switches, multiple per-producer state dirs
        may exist. We pick the one that advanced furthest so the new producer(s)
        fast-forward past all data that was already consumed.
        """
        from safetensors.torch import load_file
        best_path: Path | None = None
        best_n = -1
        for p in sorted(qdir.glob("producer_dataset_state_*")):
            state_file = p / ACTIVATIONS_STORE_STATE_FILENAME
            if not state_file.exists():
                continue
            try:
                sd = load_file(str(state_file))
                n = int(sd["n_dataset_processed"].item())
                if n > best_n:
                    best_n = n
                    best_path = p
            except Exception:
                continue
        return best_path

    def _checkpoint_tp_size_changed(self, checkpoint_path: str) -> bool:
        """Return True if the checkpoint was saved with a different sae_tp_size."""
        state_path = Path(checkpoint_path) / TRAINER_STATE_FILENAME
        if not state_path.exists():
            return False
        state_dict = torch.load(str(state_path), map_location="cpu")
        saved_tp = state_dict.get("sae_tp_size", 1)
        return saved_tp != self.sae_tp_size

    def _streaming_init_consumer(
        self, cfg: LanguageModelSAERunnerConfig[T_TRAINING_SAE_CONFIG]
    ) -> None:
        import sae_lens.distributed_streaming as ds

        self.model = None  # type: ignore[assignment]
        self.activations_store = None  # type: ignore[assignment]

        if self.is_multi_sae:
            self._streaming_init_consumer_multi(cfg)
            return

        if self.sae_tp_size > 1:
            tp_group = ds.get_sae_tp_group()
        else:
            tp_group = None

        if cfg.resume_from_checkpoint is not None:
            logger.info(
                f"[streaming-consumer] Loading weights from checkpoint: "
                f"{cfg.resume_from_checkpoint}"
            )
        else:
            logger.info("[streaming-consumer] No checkpoint to resume from — using fresh weights")

        sae = self._create_training_sae(
            seed=cfg.seed,
            tp_group=tp_group,
            device=str(self.device),
            resume_checkpoint_path=cfg.resume_from_checkpoint,
        )

        self._base_sae = sae
        self.sae = sae

    def _streaming_init_consumer_multi(
        self, cfg: LanguageModelSAERunnerConfig[T_TRAINING_SAE_CONFIG]
    ) -> None:
        import sae_lens.distributed_streaming as ds

        self.sae_by_hook: dict[str, Any] = {}
        self.base_sae_by_hook: dict[str, TrainingSAE[Any]] = {}
        self.multi_hook_sae = None
        if self.sae_pp_size > 1 and dist.is_initialized():
            import sae_lens.distributed_v2 as v2_mod
            from sae_lens.distributed_v2 import hooks_for_pp_rank

            self._pp_hook_names = hooks_for_pp_rank(
                v2_mod.get_sae_pp_rank(), self.sae_pp_size, self.hook_names
            )
        else:
            self._pp_hook_names = list(self.hook_names)

        for idx, hook_name in enumerate(self._pp_hook_names):
            seed = (
                cfg.seed
                if cfg.multi_sae_seed_mode == "same"
                else cfg.seed + idx
            )
            tp_group = ds.get_sae_tp_group() if self.sae_tp_size > 1 else None

            hook_resume = None
            if cfg.resume_from_checkpoint is not None:
                hook_ckpt = (
                    Path(cfg.resume_from_checkpoint)
                    / sanitize_hook_name_for_path(hook_name)
                )
                if hook_ckpt.exists():
                    hook_resume = str(hook_ckpt)

            sae = self._create_training_sae(
                seed=seed,
                tp_group=tp_group,
                device=str(self.device),
                resume_checkpoint_path=hook_resume,
                hook_metadata_overrides={"hook_name": hook_name},
            )

            self.base_sae_by_hook[hook_name] = sae
            self.sae_by_hook[hook_name] = sae

        self._base_sae = None  # type: ignore[assignment]
        self.sae = None

    def _init_streaming_logger(self) -> None:
        import logging
        self._logger = logging.getLogger("saelens.streaming.null")
        self._logger.addHandler(logging.NullHandler())
        self._logger.propagate = False

    def _run_streaming_producer_loop(self) -> None:
        import sae_lens.distributed_streaming as ds

        vllm_tp_group = ds.get_vllm_tp_group()
        is_tp_root = ds.is_vllm_tp_root()
        tp_root_world = ds.get_producer_tp_root()
        chunk_size = self.cfg.streaming_chunk_size_tokens
        total_tokens = self.cfg.training_tokens
        buf = self._streaming_buffer
        store = self.activations_store

        # Set up per-chunk timing and shm management logs (TP root only)
        timing_path: Path | None = None
        shm_log_path: Path | None = None
        t_ready = time.time()
        if is_tp_root and self.cfg.output_path is not None:
            out_dir = Path(self.cfg.output_path)
            out_dir.mkdir(parents=True, exist_ok=True)
            if self.cfg.save_timing_every_n_steps > 0:
                timing_path = out_dir / "timing_history_vllm.jsonl"
                timing_path.write_text("")
            shm_log_path = out_dir / "shm_log_vllm.jsonl"
            shm_log_path.write_text("")

        def _shm_log(record: dict) -> None:
            if shm_log_path is None:
                return
            record["elapsed_s"] = time.time() - t_ready
            with open(shm_log_path, "a") as f:
                json.dump(record, f)
                f.write("\n")

        ctrl = torch.zeros(1, dtype=torch.int32, device=self.device)
        chunk_idx = -1
        seq_no = -1
        chunk_step = 0

        # Quiesce signal paths — use quiesce_dir if provided (supervisor mode),
        # otherwise fall back to checkpoint_path (standalone mode).
        vllm_stop_request_path: Path | None = None
        vllm_stopped_ack_path: Path | None = None
        vllm_finished_ack_path: Path | None = None
        _qdir = self._quiesce_dir or (
            Path(self.cfg.checkpoint_path) if self.cfg.checkpoint_path is not None else None
        )
        if _qdir is not None and is_tp_root:
            producer_idx = ds.get_producer_idx()
            vllm_stop_request_path = _qdir / "vllm_stop_produce_request"
            vllm_stopped_ack_path = (
                _qdir / f"vllm_stopped_produce_ack_producer_{producer_idx}"
            )
            vllm_finished_ack_path = _qdir / f"vllm_finished_ack_producer_{producer_idx}"

        def _producer_stop_requested() -> bool:
            return (
                vllm_stop_request_path is not None
                and vllm_stop_request_path.exists()
            )

        def _save_producer_state_and_ack() -> None:
            if not is_tp_root:
                return
            if _qdir is not None:
                store.save_to_checkpoint(
                    _qdir / f"producer_dataset_state_{ds.get_producer_idx()}"
                )
            _shm_log({"event": "vllm_stopped_produce_ack", "total_chunks": chunk_step})
            if vllm_stopped_ack_path is not None:
                vllm_stopped_ack_path.parent.mkdir(parents=True, exist_ok=True)
                vllm_stopped_ack_path.touch()

        while True:
            # Quiesce check: after at least one chunk written, check for stop signal.
            # Root checks the file; result is broadcast to all TP ranks.
            if chunk_step > 0:
                if is_tp_root:
                    ctrl[0] = 1 if _producer_stop_requested() else 0
                if vllm_tp_group is not None:
                    dist.broadcast(ctrl, src=tp_root_world, group=vllm_tp_group)
                if int(ctrl[0]) == 1:
                    if is_tp_root:
                        _save_producer_state_and_ack()
                    break
            # Outer Phase 1: quota check — root tries to allocate a chunk slot.
            # Pass a stop_check so allocate_write_chunk can bail out when quiesce
            # is requested (avoids deadlock when buffer is full and consumer has stopped).
            if is_tp_root:
                def _quiesce_stop() -> bool:
                    return _producer_stop_requested()
                result = buf.allocate_write_chunk(stop_check=_quiesce_stop if chunk_step > 0 else None)
                ctrl[0] = 0 if result is None else 1
                if result is not None:
                    chunk_idx, seq_no = result
                    _shm_log({"event": "chunk_allocated", "chunk_idx": chunk_idx, "seq_no": seq_no})
            if vllm_tp_group is not None:
                dist.broadcast(ctrl, src=tp_root_world, group=vllm_tp_group)
            if int(ctrl[0]) == 0:
                if is_tp_root:
                    if _producer_stop_requested():
                        _save_producer_state_and_ack()
                    else:
                        _shm_log({"event": "quota_exhausted", "total_chunks": chunk_step})
                break  # quota exhausted or quiesce; all TP ranks exit together

            # Compute exact token limit for this chunk (root knows seq_no; non-root uses full chunk_size)
            if is_tp_root:
                max_this_chunk = max(1, min(chunk_size, total_tokens - seq_no * chunk_size))
            else:
                max_this_chunk = chunk_size  # non-root: ignored, root drives the internal loop

            # All TP ranks participate in inference (required by vLLM external_launcher semantics)
            t_infer_start = time.perf_counter()
            acts_cpu, valid_tokens = store.get_streaming_activations(max_this_chunk)
            t_infer_end = time.perf_counter()

            # Outer Phase 2: EOF check — did dataset run dry?
            if is_tp_root:
                ctrl[0] = 0 if acts_cpu is None else 1
            if vllm_tp_group is not None:
                dist.broadcast(ctrl, src=tp_root_world, group=vllm_tp_group)
            if int(ctrl[0]) == 0:
                if is_tp_root:
                    buf.abort_write_chunk(chunk_idx)  # WRITING → FREE
                    _shm_log({"event": "dataset_exhausted", "chunk_idx": chunk_idx, "total_chunks": chunk_step})
                break  # dataset exhausted; all TP ranks exit together

            # Write to buffer (TP root only)
            if is_tp_root:
                t_write_start = time.perf_counter()
                buf.write_chunk(
                    chunk_idx,
                    acts_cpu,
                    valid_tokens,
                    producer_id=ds.get_producer_idx(),
                )
                buf.mark_ready(chunk_idx)
                t_write_end = time.perf_counter()

                chunk_step += 1
                inference_time_s = t_infer_end - t_infer_start
                write_time_s = t_write_end - t_write_start
                _shm_log({
                    "event": "chunk_written",
                    "chunk_idx": chunk_idx,
                    "seq_no": seq_no,
                    "step": chunk_step,
                    "valid_tokens": valid_tokens,
                    "total_tokens": seq_no * chunk_size + valid_tokens,
                    "inference_time_s": inference_time_s,
                    "write_time_s": write_time_s,
                    "buffer_state": buf.queue_counts(),
                })

                if timing_path is not None and chunk_step % self.cfg.save_timing_every_n_steps == 0:
                    record = {
                        "step": chunk_step,
                        "elapsed_s": time.time() - t_ready,
                        "inference_time_s": inference_time_s,
                        "write_time_s": write_time_s,
                        "chunk_time_s": inference_time_s + write_time_s,
                        "valid_tokens": valid_tokens,
                        "total_tokens": seq_no * chunk_size + valid_tokens,
                    }
                    with open(timing_path, "a") as f:
                        json.dump(record, f)
                        f.write("\n")

        if is_tp_root:
            buf.signal_done()
            _shm_log({"event": "producer_done", "total_chunks": chunk_step})
            if vllm_finished_ack_path is not None:
                vllm_finished_ack_path.parent.mkdir(parents=True, exist_ok=True)
                vllm_finished_ack_path.touch()
        buf.close()

    def _run_gpu_direct_producer_loop(self) -> None:
        import sae_lens.distributed_streaming as ds

        vllm_tp_group = ds.get_vllm_tp_group()
        is_tp_root = ds.is_vllm_tp_root()
        tp_root_world = ds.get_producer_tp_root()
        gloo_ctrl = ds.get_gloo_ctrl_group()
        nccl_group = ds.get_streaming_nccl_group(0)

        store = self.activations_store
        chunk_size = self.cfg.streaming_chunk_size_tokens
        num_hooks = len(self.hook_names)
        consumer_global_rank = ds.get_consumer_tp_root()

        ctrl = torch.zeros(1, dtype=torch.int32, device=self.device)
        requested_chunks_ctrl = torch.zeros(1, dtype=torch.int32, device=self.device)
        ctrl_msg = torch.zeros(4, dtype=torch.int32)

        staging = VLLMProducerStagingQueue(
            capacity=self.cfg.streaming_staging_queue_capacity,
            chunk_shape=(chunk_size * num_hooks, self.cfg.sae.d_in),
            device=self.device,
        )
        dataset_exhausted = False
        chunks_sent = 0
        cuda_log_path: Path | None = None
        t_ready = time.time()
        if is_tp_root and self.cfg.output_path is not None:
            out_dir = Path(self.cfg.output_path)
            out_dir.mkdir(parents=True, exist_ok=True)
            cuda_log_path = out_dir / "cuda_log_vllm.jsonl"
            cuda_log_path.write_text("")

        def _cuda_log(record: dict) -> None:
            if cuda_log_path is None:
                return
            record["elapsed_s"] = time.time() - t_ready
            with open(cuda_log_path, "a") as f:
                json.dump(record, f)
                f.write("\n")

        def _generate_one() -> bool:
            nonlocal dataset_exhausted
            acts_gpu, valid_rows = store.get_streaming_activations_gpu(chunk_size)
            if is_tp_root:
                if acts_gpu is None:
                    dataset_exhausted = True
                    return False
                vph = valid_rows // num_hooks
                staging.try_push(acts_gpu, vph)
                return True
            return True

        prefill_generated = self._refill_gpu_staging_until_full(
            staging=staging,
            is_tp_root=is_tp_root,
            is_dataset_exhausted=lambda: dataset_exhausted,
            generate_one=_generate_one,
            ctrl=ctrl,
            vllm_tp_group=vllm_tp_group,
            tp_root_world=tp_root_world,
        )

        logger.info(
            "[gpu-direct-producer] Pre-fill done: staging=%d exhausted=%s",
            staging.count if is_tp_root else -1,
            dataset_exhausted,
        )
        if is_tp_root:
            _cuda_log({
                "event": "prefill_complete",
                "staging_count": staging.count,
                "dataset_exhausted": dataset_exhausted,
                "generated_chunks": prefill_generated,
            })

        # Main request-response loop
        while True:
            # TP root: block waiting for REQUEST_DATA
            requested_chunks = 1
            if is_tp_root:
                t0 = time.perf_counter()
                dist.recv(ctrl_msg, src=consumer_global_rank, group=gloo_ctrl)
                request_wait_s = time.perf_counter() - t0
                msg_type = int(ctrl_msg[0])
                if msg_type == _MSG_CONSUMER_DONE:
                    _cuda_log({
                        "event": "consumer_done_received",
                        "chunks_sent": chunks_sent,
                        "request_wait_s": request_wait_s,
                    })
                    ctrl[0] = 4  # consumer stopped before producer EOF
                else:
                    assert msg_type == _MSG_REQUEST_DATA
                    ctrl[0] = 0
                    requested_chunks = max(1, int(ctrl_msg[1]))
                _cuda_log({
                    "event": "request_received",
                    "consumer_rank": consumer_global_rank,
                    "request_wait_s": request_wait_s,
                    "requested_chunks": requested_chunks,
                    "staging_count": staging.count,
                    "dataset_exhausted": dataset_exhausted,
                })
                requested_chunks_ctrl[0] = requested_chunks

                if int(ctrl[0]) == 4:
                    pass
                elif staging.count > 0:
                    ctrl[0] = 1  # SEND
                elif dataset_exhausted:
                    ctrl[0] = 2  # EOF
                else:
                    ctrl[0] = 3  # NEED_GENERATE
            if vllm_tp_group is not None:
                dist.broadcast(
                    requested_chunks_ctrl, src=tp_root_world, group=vllm_tp_group
                )
            requested_chunks = int(requested_chunks_ctrl[0])
            if vllm_tp_group is not None:
                dist.broadcast(ctrl, src=tp_root_world, group=vllm_tp_group)
            action = int(ctrl[0])

            if action == 4:
                break

            if action == 3:
                _generate_one()
                if is_tp_root:
                    if staging.count > 0:
                        action = 1
                    elif dataset_exhausted:
                        action = 2
                    ctrl[0] = action
                if vllm_tp_group is not None:
                    dist.broadcast(ctrl, src=tp_root_world, group=vllm_tp_group)
                action = int(ctrl[0])

            if action == 2:
                if is_tp_root:
                    ctrl_msg[0] = _MSG_EOF
                    ctrl_msg[1] = 0
                    ctrl_msg[2] = 0
                    ctrl_msg[3] = 0
                    dist.send(ctrl_msg, dst=consumer_global_rank, group=gloo_ctrl)
                    _cuda_log({
                        "event": "eof_sent",
                        "chunks_sent": chunks_sent,
                    })
                break

            # action == 1: Send data
            for _ in range(requested_chunks):
                generated = self._refill_gpu_staging_until_full(
                    staging=staging,
                    is_tp_root=is_tp_root,
                    is_dataset_exhausted=lambda: dataset_exhausted,
                    generate_one=_generate_one,
                    ctrl=ctrl,
                    vllm_tp_group=vllm_tp_group,
                    tp_root_world=tp_root_world,
                )
                if is_tp_root and generated > 0:
                    _cuda_log({
                        "event": "refill_complete",
                        "generated_chunks": generated,
                        "staging_count": staging.count,
                        "dataset_exhausted": dataset_exhausted,
                    })

                if is_tp_root:
                    if staging.count == 0:
                        ctrl[0] = 2 if dataset_exhausted else 3
                    else:
                        ctrl[0] = 1
                if vllm_tp_group is not None:
                    dist.broadcast(ctrl, src=tp_root_world, group=vllm_tp_group)
                send_action = int(ctrl[0])
                if send_action == 2:
                    if is_tp_root:
                        ctrl_msg[0] = _MSG_EOF
                        ctrl_msg[1] = 0
                        ctrl_msg[2] = 0
                        ctrl_msg[3] = 0
                        dist.send(ctrl_msg, dst=consumer_global_rank, group=gloo_ctrl)
                        _cuda_log({
                            "event": "eof_sent",
                            "chunks_sent": chunks_sent,
                        })
                    action = 2
                    break
                if send_action != 1:
                    break

                if is_tp_root:
                    chunk_tensor, valid_tph = staging.pop()
                    ctrl_msg[0] = _MSG_DATA_READY
                    ctrl_msg[1] = valid_tph
                    ctrl_msg[2] = _GPU_DIRECT_DTYPE_CODES.get(chunk_tensor.dtype, 0)
                    ctrl_msg[3] = 0
                    dist.send(ctrl_msg, dst=consumer_global_rank, group=gloo_ctrl)
                    t_ready_to_recv = time.perf_counter()
                    dist.recv(ctrl_msg, src=consumer_global_rank, group=gloo_ctrl)
                    ready_wait_s = time.perf_counter() - t_ready_to_recv
                    assert int(ctrl_msg[0]) == _MSG_READY_TO_RECV
                    t_nccl = time.perf_counter()
                    dist.broadcast(chunk_tensor, src=dist.get_rank(), group=nccl_group)
                    nccl_time_s = time.perf_counter() - t_nccl
                    chunks_sent += 1
                    _cuda_log({
                        "event": "chunk_sent",
                        "chunk": chunks_sent,
                        "valid_tokens_per_hook": valid_tph,
                        "rows": int(chunk_tensor.shape[0]),
                        "d_model": int(chunk_tensor.shape[1]),
                        "dtype": str(chunk_tensor.dtype).removeprefix("torch."),
                        "ready_wait_s": ready_wait_s,
                        "nccl_time_s": nccl_time_s,
                        "staging_count_after_pop": staging.count,
                    })
                elif nccl_group is not None:
                    # Non-root producer TP ranks do not join the streaming NCCL
                    # group in the vllm_tp=1 MVP.
                    pass

            if action == 2:
                break

        logger.info("[gpu-direct-producer] Done: sent %d chunks", chunks_sent)

    @staticmethod
    def _refill_gpu_staging_until_full(
        *,
        staging: VLLMProducerStagingQueue,
        is_tp_root: bool,
        is_dataset_exhausted: Any,
        generate_one: Any,
        ctrl: torch.Tensor,
        vllm_tp_group: dist.ProcessGroup | None,
        tp_root_world: int,
    ) -> int:
        """Generate chunks until the GPU staging queue is full or data ends."""
        generated = 0
        while True:
            if is_tp_root:
                ctrl[0] = 0 if (staging.is_full or is_dataset_exhausted()) else 1
            if vllm_tp_group is not None:
                dist.broadcast(ctrl, src=tp_root_world, group=vllm_tp_group)
            if int(ctrl[0]) == 0:
                break
            if generate_one():
                generated += 1
        return generated

    def _run_streaming_consumer_loop(self) -> TrainingSAE[Any]:
        import sae_lens.distributed_streaming as ds
        from sae_lens.training.streaming_activation_provider import StreamingActivationProvider

        sae_tp_group = ds.get_sae_tp_group()
        sae_tp_size = ds.get_sae_tp_size()

        shm_log_path: Path | None = None
        buffer_monitor_path: Path | None = None
        if ds.is_sae_tp_root() and self.cfg.output_path is not None:
            out_dir = Path(self.cfg.output_path)
            out_dir.mkdir(parents=True, exist_ok=True)
            shm_log_path = out_dir / "shm_log_sae.jsonl"
            buffer_monitor_path = out_dir / "buffer_monitor.jsonl"

        qdir = self._quiesce_dir or (
            Path(self.cfg.checkpoint_path) if self.cfg.checkpoint_path is not None else None
        )
        sae_stop_acquire_request = (
            qdir / "sae_stop_acquire_request" if qdir is not None else None
        )

        from sae_lens.util import str_to_dtype
        # PP coordination: PP-0 + TP-0 of each DP replica claims chunks; siblings
        # read the same chunks from /dev/shm and slice their own hook subset.
        from sae_lens import distributed_v2
        pp_rank = distributed_v2.get_sae_pp_rank() if distributed_v2.is_consumer() else 0
        dp_replica_group = distributed_v2.get_sae_dp_replica_group()
        dp_replica_root = distributed_v2.get_sae_dp_replica_root_global_rank()
        pp_hook_names = (
            self._pp_hook_names
            if self.is_multi_sae and hasattr(self, "_pp_hook_names")
            else None
        )

        provider = StreamingActivationProvider(
            buffer=self._streaming_buffer,
            train_batch_size_tokens=self.cfg.train_batch_size_tokens,
            prefetch_chunks=self.cfg.streaming_prefetch_chunks,
            device=self.device,
            sae_tp_group=sae_tp_group if sae_tp_size > 1 else None,
            sae_tp_rank=ds.get_sae_tp_rank(),
            sae_tp_root_global_rank=ds.get_consumer_tp_root(),
            d_model=self.cfg.sae.d_in,
            dtype=str_to_dtype(self.cfg.dtype),
            shm_log_path=shm_log_path,
            shuffle=self.cfg.streaming_shuffle,
            random_chunks=self.cfg.streaming_random_chunks,
            mix_chunks=self.cfg.streaming_mix_chunks,
            mix_fraction=self.cfg.streaming_mix_fraction,
            mixing_seed=self.cfg.seed,
            mixing_shard_index=0,
            stop_acquire_check=(
                (lambda: sae_stop_acquire_request.exists())
                if sae_stop_acquire_request is not None
                else None
            ),
            buffer_monitor_path=buffer_monitor_path,
            hook_names=self.hook_names if self.is_multi_sae else None,
            select_hook_names=pp_hook_names,
            dp_replica_group=dp_replica_group if self.sae_pp_size > 1 else None,
            dp_replica_root_global_rank=dp_replica_root if self.sae_pp_size > 1 else None,
            sae_pp_size=self.sae_pp_size,
            pp_rank=pp_rank,
        )

        if self.is_multi_sae:
            return self._run_streaming_consumer_multi(provider, ds)
        return self._run_streaming_consumer_single(provider, ds)

    def _run_gpu_direct_consumer_loop(self) -> TrainingSAE[Any]:
        import sae_lens.distributed_streaming as ds
        from sae_lens import distributed_v2
        from sae_lens.training.gpu_streaming_activation_provider import (
            GpuStreamingActivationProvider,
        )
        from sae_lens.util import str_to_dtype

        gloo_ctrl = ds.get_gloo_ctrl_group()
        nccl_group = ds.get_streaming_nccl_group(0)
        producer_global_rank = distributed_v2.get_producer_tp_root(0)
        cuda_log_path: Path | None = None
        if self.cfg.output_path is not None:
            out_dir = Path(self.cfg.output_path)
            out_dir.mkdir(parents=True, exist_ok=True)
            cuda_log_path = out_dir / "cuda_log_sae.jsonl"

        pp_hook_names = (
            self._pp_hook_names
            if self.is_multi_sae and hasattr(self, "_pp_hook_names")
            else self.hook_names
        )

        inner = GpuStreamingActivationProvider(
            pp_hook_names=pp_hook_names,
            is_multi_sae=self.is_multi_sae,
            train_batch_size_tokens=self.cfg.train_batch_size_tokens,
            d_model=self.cfg.sae.d_in,
            shuffle=self.cfg.streaming_shuffle,
            buffer_size=self.cfg.n_batches_in_buffer
            * len(range(self.cfg.context_size)[slice(*self.cfg.seqpos_slice)]),
            mix_fraction=self.cfg.activations_mixing_fraction,
            mixing_seed=self.cfg.seed,
            mixing_shard_index=0,
            device=self.device,
        )

        receiver = GpuDirectReceiver(
            inner=inner,
            gloo_ctrl_group=gloo_ctrl,
            nccl_group=nccl_group,
            producer_global_rank=producer_global_rank,
            d_model=self.cfg.sae.d_in,
            dtype=str_to_dtype(self.cfg.dtype),
            device=self.device,
            cuda_log_path=cuda_log_path,
        )
        receiver.start()

        prefill_chunks = self.cfg.streaming_consumer_prefill_chunks
        if prefill_chunks > 0:
            requested_target_tokens = (
                prefill_chunks * self.cfg.streaming_chunk_size_tokens
            )
            logger.info(
                "[gpu-direct-consumer] Waiting for prefill: %d chunks "
                "(%d requested tokens)",
                prefill_chunks,
                requested_target_tokens,
            )
            receiver.wait_for_prefill(requested_target_tokens)

        provider = GpuDirectDataProvider(
            inner=inner,
            receiver=receiver,
        )

        if self.is_multi_sae:
            return self._run_streaming_consumer_multi(provider, ds)
        return self._run_streaming_consumer_single(provider, ds)

    def _run_streaming_consumer_single(self, provider: Any, ds: Any) -> TrainingSAE[Any]:
        trainer = SAETrainer(
            sae=self.sae,
            base_sae=self._base_sae,
            data_provider=provider,
            evaluator=None,
            save_checkpoint_fn=self._streaming_save_checkpoint,
            cfg=self.cfg.to_sae_trainer_config(),
            dp_group=None,
            token_count_weighted_dp=False,
            append_logs=self.cfg.resume_from_checkpoint is not None
            or self.cfg.append_history_logs,
        )

        if self.cfg.resume_from_checkpoint is not None:
            trainer.load_trainer_state(self.cfg.resume_from_checkpoint)
            logger.info(
                f"[streaming-consumer] Resumed trainer: "
                f"n_samples={trainer.n_training_samples} n_steps={trainer.n_training_steps}"
            )

        (
            consumer_quiesce_request,
            consumer_drain_ack,
            consumer_finished_ack,
        ) = self._streaming_quiesce_paths(ds)

        try:
            signal.signal(signal.SIGINT, interrupt_callback)
            signal.signal(signal.SIGTERM, interrupt_callback)
            sae = trainer.fit(
                quiesce_request_path=consumer_quiesce_request,
                quiesce_ack_path=consumer_finished_ack,
                quiesce_drain_ack_path=consumer_drain_ack,
                quiesce_finished_ack_path=consumer_finished_ack,
            )
        except StopIteration:
            sae = trainer.sae
        except (KeyboardInterrupt, InterruptedException):
            if self.cfg.checkpoint_path is not None:
                checkpoint_path = Path(self.cfg.checkpoint_path) / str(
                    trainer.n_training_samples
                )
                self._streaming_save_checkpoint(checkpoint_path)
            raise
        finally:
            if isinstance(provider, GpuDirectDataProvider):
                provider.close()

        if self._streaming_buffer is not None:
            self._streaming_buffer.close()

        if self.cfg.output_path is not None:
            self._streaming_save_final(sae, self.cfg.output_path, trainer.log_feature_sparsity)

        return sae

    def _run_streaming_consumer_multi(self, provider: Any, ds: Any) -> TrainingSAE[Any]:
        # In PP mode each rank only trains hooks in self._pp_hook_names; sae_by_hook /
        # base_sae_by_hook were already restricted in _setup_streaming_consumer.
        local_hooks = (
            self._pp_hook_names if hasattr(self, "_pp_hook_names") else self.hook_names
        )
        trainer = MultiSAETrainer(
            hook_names=local_hooks,
            sae_by_hook=self.sae_by_hook,
            base_sae_by_hook=self.base_sae_by_hook,
            multi_hook_sae=None,
            data_provider=provider,
            save_checkpoint_fn=self._streaming_save_checkpoint,
            cfg=self.cfg.to_sae_trainer_config(),
            dp_group=None,
            token_count_weighted_dp=False,
            sae_dp_mode="ddp",
            backward_mode=self.cfg.multi_sae_backward_mode,
            seed_mode=self.cfg.multi_sae_seed_mode,
            append_logs=self.cfg.resume_from_checkpoint is not None
            or getattr(self.cfg, "append_history_logs", False),
        )

        if self.cfg.resume_from_checkpoint is not None:
            trainer.load_trainer_state(self.cfg.resume_from_checkpoint)

        (
            consumer_quiesce_request,
            consumer_drain_ack,
            consumer_finished_ack,
        ) = self._streaming_quiesce_paths(ds)

        try:
            signal.signal(signal.SIGINT, interrupt_callback)
            signal.signal(signal.SIGTERM, interrupt_callback)
            trainer.fit(
                quiesce_request_path=consumer_quiesce_request,
                quiesce_ack_path=consumer_finished_ack,
                quiesce_drain_ack_path=consumer_drain_ack,
                quiesce_finished_ack_path=consumer_finished_ack,
            )
        except StopIteration:
            pass
        except (KeyboardInterrupt, InterruptedException):
            if self.cfg.checkpoint_path is not None:
                trainer.save_checkpoint(
                    checkpoint_name=str(trainer.n_training_samples)
                )
            raise
        finally:
            if isinstance(provider, GpuDirectDataProvider):
                provider.close()

        if self._streaming_buffer is not None:
            self._streaming_buffer.close()

        if self.cfg.output_path is not None:
            trainer.save_final(self.cfg.output_path)

        first_hook = local_hooks[0]
        return self.base_sae_by_hook[first_hook]

    def _streaming_quiesce_paths(
        self, ds: Any
    ) -> tuple[Path | None, Path | None, Path | None]:
        consumer_quiesce_request: Path | None = None
        consumer_drain_ack: Path | None = None
        consumer_finished_ack: Path | None = None
        _qdir = self._quiesce_dir or (
            Path(self.cfg.checkpoint_path) if self.cfg.checkpoint_path is not None else None
        )
        if _qdir is not None:
            consumer_quiesce_request = _qdir / "sae_stop_acquire_request"
            if ds.is_sae_tp_root():
                # Each PP-stage TP-root acks independently so the supervisor can
                # wait for every PP rank in every DP replica.
                from sae_lens import distributed_v2
                dp_idx = distributed_v2.get_sae_dp_idx()
                pp_rank = distributed_v2.get_sae_pp_rank()
                if dp_idx < 0:
                    dp_idx = 0
                if pp_rank < 0:
                    pp_rank = 0
                consumer_drain_ack = (
                    _qdir / f"sae_drain_ack_consumer_d{dp_idx}_pp{pp_rank}"
                )
                consumer_finished_ack = (
                    _qdir / f"sae_finished_ack_consumer_d{dp_idx}_pp{pp_rank}"
                )
        return consumer_quiesce_request, consumer_drain_ack, consumer_finished_ack

    def _streaming_save_checkpoint(self, checkpoint_path: Path | None) -> None:
        """Called by TP root only (from SAETrainer.save_checkpoint's save_checkpoint_fn guard).

        SAE weights are already saved by the trainer's _save_model (collective).
        We only need to persist the runner config here.
        """
        if checkpoint_path is None:
            return
        import sae_lens.distributed_streaming as ds
        checkpoint_path.mkdir(exist_ok=True, parents=True)
        if ds.is_sae_tp_root():
            runner_config = self.cfg.to_dict()
            with open(checkpoint_path / RUNNER_CFG_FILENAME, "w") as f:
                json.dump(runner_config, f)
            (checkpoint_path / "COMPLETED").write_text("ok\n")

    def _streaming_save_final(
        self,
        sae: TrainingSAE[Any],
        output_path: str,
        log_feature_sparsity: torch.Tensor | None,
    ) -> None:
        """Called by ALL SAE TP ranks — save_inference_model is collective when sae_tp > 1."""
        import sae_lens.distributed_streaming as ds
        base = Path(output_path)
        base.mkdir(exist_ok=True, parents=True)
        sae.save_inference_model(str(base))
        if ds.is_sae_tp_root():
            if log_feature_sparsity is not None:
                save_file({"sparsity": log_feature_sparsity}, base / SPARSITY_FILENAME)
            runner_config = self.cfg.to_dict()
            with open(base / RUNNER_CFG_FILENAME, "w") as f:
                json.dump(runner_config, f)

    def save_checkpoint(
        self,
        checkpoint_path: Path | None,
    ) -> None:
        if checkpoint_path is None:
            return
        sae = self._base_sae
        tp_group = getattr(sae, "_tp_group", None) if sae is not None else None
        tp_rank = dist.get_rank(tp_group) if tp_group is not None else 0
        dp_group = get_dp_group()
        if self.use_shard_routing:
            import sae_lens.distributed_v2 as v2_mod
            dp_group = v2_mod.get_sae_dp_group()
        dp_rank = (
            dist.get_rank(dp_group)
            if dp_group is not None and dist.get_world_size(dp_group) > 1
            else 0
        )
        if tp_rank != 0 or dp_rank != 0:
            return

        self.activations_store.save_to_checkpoint(checkpoint_path)

        runner_config = self.cfg.to_dict()
        with open(checkpoint_path / RUNNER_CFG_FILENAME, "w") as f:
            json.dump(runner_config, f)


def _parse_cfg_args(
    args: Sequence[str],
) -> LanguageModelSAERunnerConfig[TrainingSAEConfig]:
    """
    Parse command line arguments into a LanguageModelSAERunnerConfig.

    This function first parses the architecture argument to determine which
    concrete SAE config class to use, then parses the full configuration
    with that concrete type.
    """
    if len(args) == 0:
        args = ["--help"]

    # First, parse only the architecture to determine which concrete class to use
    architecture_parser = ArgumentParser(
        description="Parse architecture to determine SAE config class",
        exit_on_error=False,
        add_help=False,  # Don't add help to avoid conflicts
    )
    architecture_parser.add_argument(
        "--architecture",
        type=str,
        choices=["standard", "gated", "jumprelu", "topk", "batchtopk"],
        default="standard",
        help="SAE architecture to use",
    )

    # Parse known args to extract architecture, ignore unknown args for now
    arch_args, remaining_args = architecture_parser.parse_known_args(args)
    architecture = arch_args.architecture

    # Remove architecture from remaining args if it exists
    filtered_args = []
    skip_next = False
    for arg in remaining_args:
        if skip_next:
            skip_next = False
            continue
        if arg == "--architecture":
            skip_next = True  # Skip the next argument (the architecture value)
            continue
        filtered_args.append(arg)

    # Create a custom wrapper class that simple_parsing can handle
    def create_config_class(
        sae_config_type: type[TrainingSAEConfig],
    ) -> type[LanguageModelSAERunnerConfig[TrainingSAEConfig]]:
        """Create a concrete config class for the given SAE config type."""

        # Create the base config without the sae field
        from dataclasses import field as dataclass_field
        from dataclasses import fields, make_dataclass

        # Get all fields from LanguageModelSAERunnerConfig except the generic sae field
        base_fields = []
        for field_obj in fields(LanguageModelSAERunnerConfig):
            if field_obj.name != "sae":
                base_fields.append((field_obj.name, field_obj.type, field_obj))

        # Add the concrete sae field
        base_fields.append(
            (
                "sae",
                sae_config_type,
                dataclass_field(
                    default_factory=lambda: sae_config_type(d_in=512, d_sae=1024)
                ),
            )
        )

        # Create the concrete class
        return make_dataclass(
            f"{sae_config_type.__name__}RunnerConfig",
            base_fields,
            bases=(LanguageModelSAERunnerConfig,),
        )

    # Map architecture to concrete config class
    sae_config_map: dict[str, type[TrainingSAEConfig]] = {
        name: cfg for name, (_, cfg) in SAE_TRAINING_CLASS_REGISTRY.items()
    }

    sae_config_type = sae_config_map[architecture]
    concrete_config_class = create_config_class(sae_config_type)

    # Now parse the full configuration with the concrete type
    parser = ArgumentParser(exit_on_error=False)
    parser.add_arguments(concrete_config_class, dest="cfg")

    # Parse the filtered arguments (without --architecture)
    parsed_args = parser.parse_args(filtered_args)

    # Return the parsed configuration
    return parsed_args.cfg


# moved into its own function to make it easier to test
def _run_cli(args: Sequence[str]):
    cfg = _parse_cfg_args(args)
    LanguageModelSAETrainingRunner(cfg=cfg).run()


if __name__ == "__main__":
    _run_cli(args=sys.argv[1:])


@deprecated("Use LanguageModelSAETrainingRunner instead")
class SAETrainingRunner(LanguageModelSAETrainingRunner):
    pass
