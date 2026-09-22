"""Validate memory pickles and produce the new Megatron-only report/figures."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import shutil
import sys
import zipfile

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("megatron_memory_model", REPO / "sae_lens/autoconfig/megatron_memory_model.py")
MODEL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODEL
SPEC.loader.exec_module(MODEL)
GIB = 1024**3
LAYOUTS = ["single", "tp2", "tp4", "dp2", "dp4", "tp2dp2"]


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analyze(root):
    roots = [root, *sorted(root.glob("batch*"))]
    ranks, cases, artifacts, validations = [], [], [], []
    for experiment in roots:
        if not (experiment / "runs").is_dir():
            continue
        before = json.loads((experiment / "legacy_sha256_before.json").read_text())
        assert all(sha(REPO / name) == value for name, value in before.items()), "legacy file changed"
        production = json.loads((experiment / "source_sha256.json").read_text())
        assert all(sha(experiment / "sources" / name) == value == sha(REPO / name)
                   for name, value in production.items()), "source drift"
        validations.append(dict(experiment=str(experiment.relative_to(root)), legacy_unchanged=True,
                                frozen_source_matches_current=True, source_files=len(production)))
        for result_path in sorted((experiment / "runs").glob("*.result.json")):
            assert json.loads(result_path.read_text())["returncode"] == 0
            directory = result_path.with_name(result_path.name.removesuffix(".result.json"))
            case_ranks = []
            for path in sorted(directory.glob("rank*.json")):
                record = json.loads(path.read_text())
                rank = record["rank"]
                snap_path = directory / f"memory_timeline_rank{rank}.pickle"
                with snap_path.open("rb") as f:
                    snapshot = pickle.load(f)
                start = json.loads((directory / f"start_state_rank{rank}.json").read_text())
                replay = MODEL.replay_allocator_window(snapshot, start, rank, record["storages"])
                assert replay["peak_allocated"] == record["memory"]["peak_allocated"]
                assert replay["peak_reserved"] == record["memory"]["peak_reserved"]
                assert replay["final_accounting_error"] == 0
                cfg = MODEL.MegatronMemoryConfig(d_in=record["config"]["d_in"], d_sae=record["config"]["d_sae"],
                    hooks=3, global_batch=record["global_update_batch"], tp=record["tp"], dp=record["dp"], ga=record["ga"])
                estimate = MODEL.estimate_tensor_payloads(cfg)
                inventory = defaultdict(int)
                for item in record["storages"]:
                    inventory[item["category"]] += item["bytes"]
                for key in ("parameters", "adam_moments", "cached_inputs"):
                    assert inventory[key] == estimate[key], (key, inventory[key], estimate[key])
                assert inventory["gradients"] >= estimate["gradients"]
                assert record["flags"]["wavefront"] == (cfg.tp > 1)
                assert record["flags"]["gradient_accumulation_fusion"] == (cfg.dp > 1)
                assert record["flags"]["optimizer_overlap"] and record["flags"]["fused_adam"]
                phases = json.loads((directory / f"host_phase_markers_rank{rank}.json").read_text())
                max_phase = next((x["phase"] for x in phases if x["running_peak"] == replay["peak_allocated"]), None)
                series = replay.pop("series")
                label = f"b{cfg.global_batch}_{directory.name}_rank{rank}"
                analysis = root / "analysis" / label
                analysis.mkdir(parents=True, exist_ok=True)
                dump(analysis / "allocator_replay.json", replay)
                dump(analysis / "tensor_payload_model.json", dict(config=asdict(cfg), estimate=estimate, measured=dict(inventory)))
                # Preserve every event; plotting alone downsamples for readability.
                with (analysis / "timeline.csv").open("w") as f:
                    writer = csv.DictWriter(f, fieldnames=list(series[0]))
                    writer.writeheader()
                    writer.writerows(series)
                entry = dict(case=directory.name, batch=cfg.global_batch, rank=rank, tp=cfg.tp, dp=cfg.dp, ga=cfg.ga,
                    microbatch=cfg.microbatch, peak_allocated_gib=replay["peak_allocated"] / GIB,
                    peak_reserved_gib=replay["peak_reserved"] / GIB,
                    sampled_device_peak_gib=record["memory"]["sampled_device_peak"] / GIB,
                    baseline_allocated_gib=replay["baseline_allocated"] / GIB,
                    persistent_payload_gib=estimate["persistent_payload"] / GIB,
                    parameters_gib=inventory["parameters"] / GIB, adam_moments_gib=inventory["adam_moments"] / GIB,
                    gradient_buffer_gib=inventory["gradients"] / GIB, cached_inputs_gib=inventory["cached_inputs"] / GIB,
                    peak_extra_above_baseline_gib=(replay["peak_allocated"] - replay["baseline_allocated"]) / GIB,
                    pending_free_at_allocated_peak_gib=replay["pending_free_at_allocated_peak"] / GIB,
                    inactive_at_allocated_peak_gib=(replay["reserved_at_allocated_peak"] - replay["peak_allocated"]
                                                   - replay["pending_free_at_allocated_peak"]) / GIB,
                    peak_observed_after_host_phase=max_phase,
                    replay_peak_error_bytes=0, replay_final_error_bytes=0,
                    snapshot=str(snap_path.relative_to(root)), analysis=str(analysis.relative_to(root)))
                ranks.append(entry)
                case_ranks.append(entry)
                artifacts.append(dict(path=str(snap_path.relative_to(root)), bytes=snap_path.stat().st_size, sha256=sha(snap_path)))
            assert len(case_ranks) == cfg.tp * cfg.dp
            worst = max(case_ranks, key=lambda r: r["peak_allocated_gib"])
            case = {**worst, "layout": directory.name.split("_ga")[0], "ranks": len(case_ranks)}
            for key in ("peak_allocated_gib", "peak_reserved_gib", "sampled_device_peak_gib"):
                case[key] = max(r[key] for r in case_ranks)
            case["min_rank_allocated_gib"] = min(r["peak_allocated_gib"] for r in case_ranks)
            case["allocated_peak_rank"] = worst["rank"]
            cases.append(case)
    assert cases
    cases.sort(key=lambda r: (r["batch"], r["ga"], LAYOUTS.index(r["layout"])))
    dump(root / "summary.json", cases)
    dump(root / "rank_summary.json", ranks)
    dump(root / "snapshot_manifest.json", artifacts)
    dump(root / "validation.json", dict(cases=len(cases), ranks=len(ranks), pickles=len(artifacts),
        all_allocated_and_reserved_peaks_replayed_exactly=True, all_end_allocations_replayed_exactly=True,
        analytic_parameters_adam_inputs_exact=True, legacy_and_source_checks=validations))
    for name, rows in (("summary.csv", cases), ("rank_summary.csv", ranks)):
        with (root / name).open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return cases


def plots(root, cases):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    out = root / "figures"
    out.mkdir(exist_ok=True)
    batches = sorted({r["batch"] for r in cases})
    colors = {1: "#4056a1", 2: "#d28b27", 4: "#168a80"}
    fig, axes = plt.subplots(len(batches), 3, figsize=(16, 4.2 * len(batches)), squeeze=False)
    for i, batch in enumerate(batches):
        labels = [l for l in LAYOUTS if any(r["batch"] == batch and r["layout"] == l for r in cases)]
        for j, (key, title) in enumerate((("peak_allocated_gib", "Allocated"), ("peak_reserved_gib", "Reserved"),
                                         ("sampled_device_peak_gib", "Device used (10 ms samples)"))):
            ax = axes[i, j]
            for g, ga in enumerate((1, 2, 4)):
                values = [next((r[key] for r in cases if r["batch"] == batch and r["layout"] == l and r["ga"] == ga), float("nan")) for l in labels]
                ax.bar(np.arange(len(labels)) + (g - 1) * .25, values, width=.23, label=f"GA={ga}", color=colors[ga])
            ax.set_xticks(range(len(labels)), labels, rotation=25)
            ax.set_title(f"B_eff={batch:,} / {title}")
            ax.set_ylabel("GiB / GPU (max across ranks)")
            ax.grid(axis="y", alpha=.2)
            ax.set_axisbelow(True)
            if i == 0 and j == 0:
                ax.legend()
    fig.suptitle("Current Megatron SAE: H=3, d_in=4096, d_sae=16384, FP32, optimizations enabled")
    fig.tight_layout()
    for ext in ("png", "svg", "pdf"):
        fig.savefig(out / f"01_memory_comparison.{ext}", dpi=180)
    plt.close(fig)
    subset = [r for r in cases if r["batch"] == 8192]
    fig, ax = plt.subplots(figsize=(15, 5))
    bottom = np.zeros(len(subset))
    keys = [("parameters_gib", "Parameters"), ("adam_moments_gib", "Adam m/v"),
            ("gradient_buffer_gib", "Persistent main_grad"), ("cached_inputs_gib", "Cached inputs"),
            ("other", "Other baseline"), ("peak_extra_above_baseline_gib", "Peak above baseline")]
    for key, label in keys:
        values = np.array([r[key] if key != "other" else r["baseline_allocated_gib"] - sum(r[k] for k, _ in keys[:4]) for r in subset])
        ax.bar(range(len(subset)), values, bottom=bottom, label=label)
        bottom += values
    ax.set_xticks(range(len(subset)), [f"{r['layout']}\nGA{r['ga']}" for r in subset], rotation=40, ha="right")
    ax.set_ylabel("GiB / rank with highest allocated peak")
    ax.set_title("B_eff=8192: measured residency and transient peak (not sums of separate phase peaks)")
    ax.legend(ncol=3, fontsize=9)
    fig.tight_layout()
    for ext in ("png", "svg", "pdf"):
        fig.savefig(out / f"02_memory_components.{ext}", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True)
    for ax, layout in zip(axes, ("tp4", "dp4", "tp2dp2")):
        row = next(r for r in cases if r["batch"] == 8192 and r["ga"] == 1 and r["layout"] == layout)
        with (root / row["analysis"] / "timeline.csv").open() as f:
            values = list(csv.DictReader(f))
        t0 = int(values[0]["time_us"])
        x = [(int(v["time_us"]) - t0) / 1e6 for v in values]
        for key, label in (("allocated", "Allocated"), ("active", "Active incl. pending free"), ("reserved", "Reserved")):
            ax.plot(x, [int(v[key]) / GIB for v in values], label=label, linewidth=1)
        ax.set_title(f"{layout}, GA=1, B_eff=8192")
        ax.set_xlabel("Seconds from measured allocator event")
    axes[0].set_ylabel("GiB")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "svg", "pdf"):
        fig.savefig(out / f"03_allocator_timeline.{ext}", dpi=180)
    plt.close(fig)


def report(root, cases):
    validation = json.loads((root / "validation.json").read_text())
    lines = ["# 新版 Megatron 离线 SAE 内存报告", "",
        f"已完成 **{len(cases)} 个配置、{validation['pickles']} 份 rank 级 memory pickle**。旧版代码与结果未修改。",
        "", "H=3，d_in=4096，d_sae=16384，K=128，dense FP32；GA=1/2/4 表示每个 DP replica 的本地累积次数。",
        "固定各实验的全局有效 batch，local microbatch = global effective batch / (DP × GA)。",
        "使用已有真实 Llama 激活缓存，各拓扑/GA 使用相同 global_row_order 前缀；全部输入预先驻留 GPU，报告包含这部分显存。",
        "不同 batch 使用嵌套 token 前缀，每次 update 重放同一集合；未启动 vLLM、未包含 LLM、在线混合缓冲、磁盘或 H2D 峰值。",
        "", "每配置独立进程，6 次 update 预热 + 3 次 update 采集。仅在 update 边界同步，阶段内部不插入同步，不调用 empty_cache。",
        "开启 TP wavefront、跨 hook optimizer overlap、fused Adam、single-replica fast path、梯度累积 fusion 和 GA1 mean-loss fast path；按适用拓扑检查生效。",
        "参数 gather overlap 开关开启，但本次使用普通非分片 Adam，没有 optimizer 参数 gather；不是 distributed optimizer/ZeRO 实验。",
        "TF32/AMP 关闭；decoder norm rescale 开启；dead-feature window=1000，本次没有进入辅助重构分支。",
        "", "## 实测峰值", "",
        "均为每配置所有 rank 的最大值，单位 GiB。allocated/reserved/设备峰值分别取最大值，未假定发生在同一时刻。",
        "设备占用由 NVML 每约 10 ms 采样，包含 CUDA/NCCL 等非 PyTorch 开销；采样最大值不保证捕获更短暂的设备峰值。",
        "", "| Effective batch | 配置 | GA | Local microbatch | Allocated | Reserved | Device sampled |",
        "|---:|---|---:|---:|---:|---:|---:|"]
    for r in cases:
        lines.append(f"| {r['batch']} | {r['layout']} | {r['ga']} | {r['microbatch']} | {r['peak_allocated_gib']:.3f} | {r['peak_reserved_gib']:.3f} | {r['sampled_device_peak_gib']:.3f} |")
    lines += ["", "![Memory comparison](figures/01_memory_comparison.png)", "", "## TP、DP、GA 的作用与配置变化", "",
        "当前 Megatron 的 encoder 输出先按 feature 切分，再 all-gather 到每个 TP rank。hidden_pre 和 dense feature_acts 都是完整 [B_local,d_sae]。decoder narrow 的局部视图不会释放被输出或 autograd 引用的完整 storage。",
        "TrainStepOutput 保留 hidden_pre、feature_acts、sae_out；detach 解除图关系但不释放 storage。GA 期间每个 hook 保留上一份输出至替换，图不会跨全部 GA microbatch 累积。TP wavefront 同时构建多个 hook 的图，TP1 采用逐 hook forward/backward。",
        "", "令 I=d_in、F=d_sae、H=本地 hook 数、T=TP、D=DP、G=本地 GA、B_eff=全局有效 batch。主要尺度为：", "",
        "```text", "b = B_eff / (D * G)", "P = 4 * H * (2 * I * F/T + F/T + I)   # FP32 参数字节",
        "Adam moments = 2P                    # 本次未分片",
        "persistent gradients = 0 (D=1 fast path) / P + padding (D>1)",
        "cached inputs = 4 * H * (B_eff / D) * I  # 全窗口预加载，不随 G 缩小",
        "one full feature tensor per hook = 4 * b * F  # 没有 /T",
        "one detached output set ≈ 4 * H * b * (2F + I)  # input 另计，已共享缓存",
        "```", "",
        "DP=1 的参数梯度在 backward 产生，并在各 hook 更新后释放，因此不能把 H 个 hook 的梯度都当作整个窗口的固定驻留。DP>1 的 main_grad 桶持续驻留；梯度 fusion 避免另建完整矩阵 wgrad 后再累加。",
        "GA 不减少参数、moment 或 main_grad 桶；固定 B_eff 时通过缩小 b 减少激活。若固定 microbatch、只增大 GA，则没有同样的激活节省，完整缓存还可能变大。",
        "", "**不能仅凭 d_sae 很大判定 TP 更省。** 权重状态和完整激活都随 F 线性增长，状态/激活比例中的 F 大致抵消；更重要的是 b/I、TP/DP、hook 调度及计算 dtype。",
        "状态主导时 TP 的收益强；完整激活主导时 DP 或 GA 更有效，TP×DP 可以同时降低两项。提高 TP 无法消除当前实现完整激活的每卡下限。",
        "普通 DP 不切分 Adam；切换 distributed optimizer 会改变静态项和通信临时项，需要另测。BF16 autocast 可降低部分激活而本 runtime 保留 FP32 参数/状态；稀疏 TopK 也不能直接假定消除完整 hidden_pre，当前 Megatron decoder 仍会 to_dense。以上配置未在本报告实测。",
        "", "算法上全局 TopK 不要求永久保存全量 hidden_pre：可探索 local TopK 候选交换、全局索引选择和分片 decoder，但必须同时处理 backward、并列值、辅助损失和输出统计；本次没有修改训练算法。",
        "", "**本次已观察到配置变化造成优势反转：** B_eff=8192、GA1 时 TP4 allocated 为 6.927 GiB、DP4 为 8.115 GiB；B_eff=16384、GA1 时 TP4 为 12.713 GiB、DP4 为 9.586 GiB。",
        "B_eff=16384、TP4、GA1 的峰值分配栈中，三个 hook 的完整 gather 输出共 3 GiB，dense TopK 输出另 3 GiB；forward/保留输出来源合计约 10.829 GiB。完整张量复制已经是实测中的主要项。",
        "候选交换和分片训练输出的详细实现思路见 [分布式 TopK 方案](code/docs/megatron_distributed_topk_memory_plan.md)，尚未修改生产训练算法。",
        "", "## 同卡数下能否减少 GA", "",
        "下表比较 TP4/GA2、DP4/GA4、TP2DP2/GA1；应同时检查 reserved 和设备采样，不能只按 allocated 判断容量。",
        "", "| Batch | 配置 | GA | Allocated | Reserved | Device sampled |", "|---:|---|---:|---:|---:|---:|"]
    for batch in sorted({r["batch"] for r in cases}):
        for layout, ga in (("tp4", 2), ("dp4", 4), ("tp2dp2", 1)):
            for r in cases:
                if (r["batch"], r["layout"], r["ga"]) == (batch, layout, ga):
                    lines.append(f"| {batch} | {layout} | {ga} | {r['peak_allocated_gib']:.3f} | {r['peak_reserved_gib']:.3f} | {r['sampled_device_peak_gib']:.3f} |")
    lines += ["", "上述实测仅证明对应配置的显存关系，不是任意 batch/d_sae 的保证；内存采集窗口不作为吞吐 benchmark，也没有验证训练收敛。",
        "", "## 新版内存模型与 pickle 核对", "",
        "新版代码分为独立的解析驻留模型与 allocator 历史重放，不调用或覆盖旧版经验系数。解析部分预测参数、Adam、缓存输入和 DP 梯度 payload；动态峰值按实际 alloc/free 生命周期计算，避免把彼此不同时的 forward/backward/optimizer 峰值相加。",
        "allocated 在 free_requested 时下降；active 还包括跨 CUDA stream 等待完成的释放，直到 free_completed 才下降；reserved 包含 active 和缓存空闲块。支持本次 expandable_segments 的 segment_map/unmap。",
        "本次使用 native allocator 默认 512-byte rounding；重放峰值与 runtime 数值逐字节一致。改变 allocator rounding 时必须重新核验，不应直接套用。",
        "**该解析模型不是任意配置的总峰值预测器。** 激活、算子 workspace、跨流存活和 reserved slack 由本次 pickle 量化，不能把这一个尺寸的残差外推为普适常数。",
        "", f"验收：{validation['pickles']} 个 rank 的 allocated 峰值、reserved 峰值及结束 allocated 均重放一致；参数、Adam moments 和缓存输入的解析 payload 与 storage inventory 一致。输入 storage 按地址去重，避免 main_grad view 或张量 view 重复计数。",
        "host_phase_markers 记录的是 CPU 调用边界及全窗口累计峰值；优化器可能在另一条 stream 执行，不能将其宣称为互斥的 CUDA 阶段峰值。allocation-site 分类仅说明张量来源。",
        "", "![Components](figures/02_memory_components.png)", "", "![Timeline](figures/03_allocator_timeline.png)",
        "", "## 复现与文件", "", "```bash",
        ".venv/bin/python scripts/profile_megatron_sae_memory.py --gas 1 2 4",
        ".venv/bin/python scripts/profile_megatron_sae_memory.py \\",
        "  --output results/memory_model_megatron_20260922/batch16384 \\",
        "  --layouts tp4 dp4 tp2dp2 --gas 1 2 4 --batch 16384",
        ".venv/bin/python scripts/analyze_megatron_sae_memory.py", "```", "",
        "已有成功结果会保留；重新独立采集请提供新的 --output。生产源码在每个实验目录 sources/ 冻结，并核对源文件 hash。",
        "本报告的源码与旧文件校验见 [validation.json](validation.json)，逐 rank 数值见 [rank_summary.csv](rank_summary.csv)，汇总见 [summary.csv](summary.csv)。",
        "[memory pickles 压缩包](memory_pickles.zip) / [代码、报告、图和诊断包](report_and_code.zip)。",
        "pickle 为 PyTorch memory snapshot 格式，可在 PyTorch memory_viz 中打开；仅打开可信来源的 pickle。",
        "采集代码：code/scripts/profile_megatron_sae_memory.py；分析代码：code/scripts/analyze_megatron_sae_memory.py；新版模型：code/sae_lens/autoconfig/megatron_memory_model.py。",
        "每个 run 保存真实命令、优化生效检查、loss、调用次数、memory pickle、初始 allocator state 和 NVML 样本；analysis/ 保存峰值分配栈及逐事件内存曲线。",
        "旧版 scripts/predict_sae_memory.py、scripts/profile_sae_phase_v5.py、sae_lens/autoconfig/phase_memory_model.py 的 SHA256 前后一致。本工作区没有旧 results/memory_model 原始结果目录，因此未声称完成相同配置的新旧实测差分。",
        "首个采集尝试在 storage inventory 处遇到脚本错误，未纳入汇总；已修复后完整重跑，失败日志保留在 failed_attempts/。", ""]
    (root / "README.md").write_text("\n".join(lines))


def archive(root):
    files = ["scripts/profile_megatron_sae_memory.py", "scripts/analyze_megatron_sae_memory.py",
             "docs/megatron_distributed_topk_memory_plan.md",
             "sae_lens/autoconfig/megatron_memory_model.py", "tests/test_megatron_memory_model.py"]
    for relative in files:
        path = REPO / relative
        if path.exists():
            target = root / "code" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    with zipfile.ZipFile(root / "memory_pickles.zip", "w", compression=zipfile.ZIP_DEFLATED) as z:
        for item in json.loads((root / "snapshot_manifest.json").read_text()):
            z.write(root / item["path"], item["path"])
        z.write(root / "snapshot_manifest.json", "snapshot_manifest.json")
    with zipfile.ZipFile(root / "report_and_code.zip", "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix in (".zip", ".pickle", ".pyc") or "__pycache__" in path.parts:
                continue
            z.write(path, str(path.relative_to(root)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=REPO / "results/memory_model_megatron_20260922")
    args = p.parse_args()
    root = args.output.resolve()
    cases = analyze(root)
    plots(root, cases)
    report(root, cases)
    archive(root)
    print(f"Validated {len(cases)} cases; report: {root / 'README.md'}")


if __name__ == "__main__":
    main()
