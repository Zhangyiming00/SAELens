import pytest

from sae_lens.autoconfig.phase_memory_model import (
    SAEPhaseMemoryConfig,
    estimate_phase_memory,
)

MB = 1024**2


def _w_bytes(d_in: int, d_sae: int, tp: int, db: int) -> int:
    d_sae_local = d_sae // tp
    return (2 * d_in * d_sae_local + d_sae_local + d_in) * db


def test_forward_backward_optimizer_match_closed_form_single_hook_fp32():
    d_in, d_sae, B, k, db = 4096, 65536, 2048, 128, 4
    cfg = SAEPhaseMemoryConfig(
        d_in=d_in,
        d_sae=d_sae,
        num_hooks=1,
        k=k,
        train_batch_size_tokens=B,
        dtype="fp32",
        optimizer_impl="fused",
    )
    e = estimate_phase_memory(cfg)

    W = _w_bytes(d_in, d_sae, 1, db)
    batch = B * d_in * db
    retained = (2 * B * d_sae + 2 * B * d_in) * db + 3 * 512
    topk = B * k * db + B * k * 8

    fwd_current = (2 * B * d_sae + B * d_in) * db
    expected_fwd = 3 * W + batch + retained + fwd_current + topk
    assert e.forward_bytes == pytest.approx(expected_fwd)

    bracket = (d_in * d_sae + 2 * B * d_sae + B * d_in) * db
    t_full = 2 * B * d_sae * db
    expected_bwd = 4 * W + batch + retained + bracket + t_full
    assert e.backward_bytes == pytest.approx(expected_bwd)

    # fused optimizer transient is 0 → just resident P+G+M+V + batch
    assert e.optimizer_bytes == pytest.approx(4 * W + batch)


def test_optimizer_impl_transient_scaling():
    d_in, d_sae, B, db = 4096, 65536, 2048, 4
    common = dict(
        d_in=d_in,
        d_sae=d_sae,
        train_batch_size_tokens=B,
        dtype="fp32",
    )
    W = _w_bytes(d_in, d_sae, 1, db)
    p_big = d_in * d_sae * db

    # fused: 0, hook-independent
    for H in (1, 2, 4):
        e = estimate_phase_memory(
            SAEPhaseMemoryConfig(num_hooks=H, optimizer_impl="fused", **common)
        )
        assert e.components["optimizer"]["optim_transient"] == 0

    # foreach: H * W (linear in H)
    for H in (1, 2, 4):
        e = estimate_phase_memory(
            SAEPhaseMemoryConfig(num_hooks=H, optimizer_impl="foreach", **common)
        )
        assert e.components["optimizer"]["optim_transient"] == pytest.approx(H * W)

    # for_loop: 3 * P_big (hook-independent)
    for H in (1, 2, 4):
        e = estimate_phase_memory(
            SAEPhaseMemoryConfig(num_hooks=H, optimizer_impl="for_loop", **common)
        )
        assert e.components["optimizer"]["optim_transient"] == pytest.approx(3 * p_big)


def test_tp_shards_weights_and_batch_transients_along_d_sae():
    d_in, d_sae, B, db = 4096, 65536, 2048, 4
    common = dict(
        d_in=d_in, d_sae=d_sae, train_batch_size_tokens=B, dtype="fp32", num_hooks=1
    )
    e1 = estimate_phase_memory(SAEPhaseMemoryConfig(tp_size=1, **common))
    e2 = estimate_phase_memory(SAEPhaseMemoryConfig(tp_size=2, **common))

    # weights: b_dec (d_in) is replicated so the sharded W is slightly more than
    # half; the P_big block inside the backward bracket halves exactly.
    assert e2.components["backward"]["weights"] == pytest.approx(
        _w_bytes(d_in, d_sae, 2, db) * 4
    )
    # backward bracket P_big term halves under tp=2
    bracket1 = e1.components["backward"]["bracket"]
    bracket2 = e2.components["backward"]["bracket"]
    # bracket = (d_in*d_sae/tp + 2*B*d_sae/tp + B*d_in)*db ; only B*d_in survives tp
    assert bracket2 == pytest.approx(
        (d_in * d_sae // 2 + 2 * B * d_sae // 2 + B * d_in) * db
    )
    assert bracket2 < bracket1


def test_ddp_bucket_only_without_gradient_as_bucket_view():
    d_in, d_sae, B, db, H = 4096, 65536, 2048, 4, 2
    common = dict(
        d_in=d_in,
        d_sae=d_sae,
        num_hooks=H,
        train_batch_size_tokens=B,
        dtype="fp32",
        dp_mode="ddp",
        dp_size=2,
    )
    with_bucket = estimate_phase_memory(
        SAEPhaseMemoryConfig(gradient_as_bucket_view=False, **common)
    )
    no_bucket = estimate_phase_memory(
        SAEPhaseMemoryConfig(gradient_as_bucket_view=True, **common)
    )
    # DDP replicates params (B is local = 1024), bucket adds exactly +H*W_local
    W = _w_bytes(d_in, d_sae, 1, db)
    assert with_bucket.backward_bytes - no_bucket.backward_bytes == pytest.approx(H * W)
    # bucket does not appear in forward (no grad yet)
    assert with_bucket.forward_bytes == pytest.approx(no_bucket.forward_bytes)


def test_ddp_fsdp_split_local_batch():
    d_in, d_sae = 4096, 32768
    manual = estimate_phase_memory(
        SAEPhaseMemoryConfig(
            d_in=d_in, d_sae=d_sae, train_batch_size_tokens=4096, dtype="fp32"
        )
    )
    ddp = estimate_phase_memory(
        SAEPhaseMemoryConfig(
            d_in=d_in,
            d_sae=d_sae,
            train_batch_size_tokens=4096,
            dtype="fp32",
            dp_mode="ddp",
            dp_size=2,
        )
    )
    # local batch halves under dp=2 → batch term halves
    assert ddp.components["backward"]["batch"] == pytest.approx(
        manual.components["backward"]["batch"] / 2
    )
    # retained output (batch-scaled) also halves
    assert (
        ddp.components["backward"]["retained_output"]
        < manual.components["backward"]["retained_output"]
    )


def test_fsdp_phase_weight_formulas():
    d_in, d_sae, B, db, H, dp = 4096, 65536, 2048, 4, 2, 2
    e = estimate_phase_memory(
        SAEPhaseMemoryConfig(
            d_in=d_in,
            d_sae=d_sae,
            num_hooks=H,
            train_batch_size_tokens=B,
            dtype="fp32",
            dp_mode="fsdp",
            dp_size=dp,
        )
    )
    W = _w_bytes(d_in, d_sae, 1, db)
    HW = H * W
    # forward: 3HW/dp + HW
    assert e.components["forward"]["weights"] == pytest.approx(3 * HW / dp + HW)
    # backward: 3HW/dp + (H+1)W + (H-1)W/dp
    assert e.components["backward"]["weights"] == pytest.approx(
        3 * HW / dp + (H + 1) * W + (H - 1) * W / dp
    )
    # optimizer: 4HW/dp (full grad released, no comm)
    assert e.components["optimizer"]["weights"] == pytest.approx(4 * HW / dp)


def test_resident_weights_are_phase_aware():
    d_in, d_sae, H, db = 4096, 65536, 2, 4
    e = estimate_phase_memory(
        SAEPhaseMemoryConfig(
            d_in=d_in,
            d_sae=d_sae,
            num_hooks=H,
            train_batch_size_tokens=2048,
            dtype="fp32",
            optimizer_impl="fused",
        )
    )
    HW = H * _w_bytes(d_in, d_sae, 1, db)
    # forward has no gradient yet (P+M+V), backward/optimizer hold P+G+M+V
    assert e.components["forward"]["weights"] == pytest.approx(3 * HW)
    assert e.components["backward"]["weights"] == pytest.approx(4 * HW)
    assert e.components["optimizer"]["weights"] == pytest.approx(4 * HW)


def test_forward_peak_holds_retained_and_current_output_together():
    d_in, d_sae, B, db = 4096, 65536, 2048, 4
    e = estimate_phase_memory(
        SAEPhaseMemoryConfig(
            d_in=d_in,
            d_sae=d_sae,
            num_hooks=1,
            train_batch_size_tokens=B,
            dtype="fp32",
        )
    )
    retained = e.components["forward"]["retained_output"]
    current = e.components["forward"]["current_output"]
    # both coexist at the forward peak: retained = 2·B·d_sae + 2·B·d_in,
    # current = 2·B·d_sae + B·d_in → together the measured 4·B·d_sae + 3·B·d_in.
    assert retained == pytest.approx((2 * B * d_sae + 2 * B * d_in) * db + 3 * 512)
    assert current == pytest.approx((2 * B * d_sae + B * d_in) * db)
    assert retained > 0 and current > 0


def test_dead_feature_aux_forward_term():
    d_in, d_sae, B = 4096, 65536, 2048
    off = estimate_phase_memory(
        SAEPhaseMemoryConfig(
            d_in=d_in, d_sae=d_sae, train_batch_size_tokens=B, dtype="fp32"
        )
    )
    assert off.components["forward"]["aux"] == 0

    F = 1000
    on = estimate_phase_memory(
        SAEPhaseMemoryConfig(
            d_in=d_in,
            d_sae=d_sae,
            train_batch_size_tokens=B,
            dtype="fp32",
            num_dead_features=F,
        )
    )
    # max(B*F*1 + B*F*4, B*F*4 + r_sum) with r_sum=0 → 5*B*F
    assert on.components["forward"]["aux"] == pytest.approx(5 * B * F)


def test_bf16_halves_dtype_scaled_terms():
    common = dict(
        d_in=4096,
        d_sae=65536,
        num_hooks=1,
        train_batch_size_tokens=2048,
        optimizer_impl="foreach",
    )
    fp32 = estimate_phase_memory(SAEPhaseMemoryConfig(dtype="fp32", **common))
    bf16 = estimate_phase_memory(SAEPhaseMemoryConfig(dtype="bf16", **common))
    # optimizer transient (H*W) is pure dtype-scaled → exactly halves
    assert bf16.components["optimizer"]["optim_transient"] == pytest.approx(
        fp32.components["optimizer"]["optim_transient"] / 2
    )


def test_invalid_config_raises():
    with pytest.raises(ValueError):
        SAEPhaseMemoryConfig(d_in=4096, d_sae=65535, tp_size=2)  # not divisible
    with pytest.raises(ValueError):
        SAEPhaseMemoryConfig(d_in=0, d_sae=100)
    with pytest.raises(ValueError):
        SAEPhaseMemoryConfig(d_in=100, d_sae=100, num_hooks=0)
