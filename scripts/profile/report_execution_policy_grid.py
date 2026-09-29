"""Build and report the measured TP/k/AuxK execution-policy grid."""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from pathlib import Path

POLICIES = {
    "S_SC": ("sharded_dense", "sparse", "compact"),
    "R_SC": ("sharded_ragged", "sparse", "compact"),
    "F_SC": ("full", "sparse", "compact"),
    "F_DD": ("full", "dense", "dense"),
    "S_DC": ("sharded_dense", "dense", "compact"),
    "R_CC": ("sharded_ragged", "compact", "compact"),
    "R_SS": ("sharded_ragged", "sparse", "sparse"),
}
BASE = dict(tp=1, dp=1, pp=1, h=3, batch=8192, ga=1, d_in=4096,
            d_sae=16384, backend="sharded_dense", wave="lazy", live=2,
            zero=False, overlap="off", aux="auto", dead=2048,
            main_compute="inherit", aux_compute="inherit")


def build(root):
    configs, rows, equivalent = {}, [], {}
    for k in (128, 1, 512):
        for auxk in (2048, 0, 128, 512):
            for tp in (1, 2):
                for policy, (storage, main, aux) in POLICIES.items():
                    key = (tp, k, auxk, storage, main, aux if auxk else None)
                    case = equivalent.get(key)
                    if case is None:
                        case = f"tp{tp}_k{k}_a{auxk}_{policy}"
                        equivalent[key] = case
                        configs[case] = dict(BASE, tp=tp, k=k, auxk=auxk, execution=dict(
                            main_representation=storage, aux_representation=storage,
                            main_compute=main, aux_compute=aux))
                    rows.append(dict(tp=tp, k=k, auxk=auxk, policy=policy,
                                     storage=storage, main_compute=main, aux_compute=aux,
                                     case=case))
    root.mkdir(parents=True, exist_ok=True)
    (root / "cases.json").write_text(json.dumps(configs, indent=2) + "\n")
    (root / "rows.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(f"{len(rows)} logical rows; {len(configs)} independent runs")


def report(root, markdown):
    rows = json.loads((root / "rows.json").read_text())
    summary, pending = [], set()
    for row in rows:
        directory = root / "runs" / row["case"]
        result_path = directory / "result.json"
        if not result_path.exists():
            pending.add(row["case"])
            continue
        result = json.loads(result_path.read_text())
        if result["returncode"]:
            summary.append(dict(row, status="FAILED"))
            continue
        ranks = [json.loads((directory / f"rank{i}.json").read_text()) for i in range(row["tp"])]
        for rank in ranks:
            assert len(rank["steps"]) == 20, directory
            for branch in ("main", "aux"):
                for dispatch in rank[branch + "_execution"]:
                    if branch == "aux" and row["auxk"] == 0:
                        assert dispatch["selection"] == "disabled", (directory, dispatch)
                        continue
                    for stage in ("forward", "value_gradient", "weight_gradient"):
                        actual = dispatch[stage].removesuffix("_dense")
                        assert actual == row[branch + "_compute"], (directory, branch, stage, dispatch)
                    assert dispatch["representation"] == row["storage"], (directory, dispatch)
        samples = [max(r["steps"][i]["ms"] for r in ranks) for i in range(len(ranks[0]["steps"]))]
        quantiles = statistics.quantiles(samples, n=10, method="inclusive")
        spreads = [max(s["post_output_allocated_bytes"] for s in r["steps"]) -
                   min(s["post_output_allocated_bytes"] for s in r["steps"]) for r in ranks]
        summary.append(dict(row, status="OK", median_ms=statistics.median(samples),
                            p10_ms=quantiles[0], p90_ms=quantiles[-1], steps=len(samples),
                            peak_allocated_gib=max(r["peak_allocated_bytes"] for r in ranks)/2**30,
                            poststep_spread_bytes=max(spreads),
                            main_dispatch=[r["main_execution"] for r in ranks],
                            aux_dispatch=[r["aux_execution"] for r in ranks]))
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fields = list(rows[0]) + ["status", "median_ms", "p10_ms", "p90_ms", "steps",
                             "peak_allocated_gib", "poststep_spread_bytes", "main_dispatch", "aux_dispatch"]
    with (root / "summary.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in summary:
            writer.writerow({k: json.dumps(v) if isinstance(v, list) else v for k, v in row.items()})
    print(f"{len(summary)}/{len(rows)} rows; pending {len(pending)} independent cases")
    if markdown is None:
        return
    by_key = {(r["tp"], r["k"], r["auxk"], r["policy"]): r for r in summary}
    lines = ["# TP1/TP2：k × AuxK × 存储/计算路径实测", "",
             "本报告为当前 Megatron SAE 训练实测，不是模拟器预测。使用已修正的分片 TopK（Torch key 工作区预算 256 MiB）；性能原因与前后对照见 [选择实现复查](sharded_selection_performance_audit.md)。", "",
             f"测量进度：{len(summary)}/{len(rows)} 行，剩余 {len(pending)} 次独立运行。" if pending else
             f"已完成全部 {len(rows)} 行（{len({r['case'] for r in rows})} 次独立运行；AuxK=0 的等价行共享实测）。", "",
             "## 测量口径", "",
             "- RTX 5090 32 GiB；FP32，TF32/autocast 关闭。d_in=4096，d_sae=16384，batch=8192，H=3（3 个 hook 均训练）。",
             "- TP=1/2，DP=PP=GA=1；tp-overlap=lazy（TP1 自动关闭）；optimizer overlap 关闭；原生 fused Adam，ZeRO 关闭。",
             "- dead=2048，均匀分布；每步重置同一 dead mask，使辅助任务量固定。AuxK=0 明确禁用 AuxK；128/512 需要选择，2048 选中全部 dead。",
             "- 同一组真实缓存激活、相同初始化；每个配置独立进程，串行运行。5 次 warmup + 20 次 measured update；时间取每步最慢 rank，再取 20 步中位数。",
             "- step 包含 3 个 SAE 的 forward/backward/optimizer；不含 vLLM、SHM I/O、读取数据、保存或日志。不是 LLM 全网络推理时间。",
             "- 内存为 warmup 后 torch.cuda.max_memory_allocated() 的 rank 最大值，单位 GiB；含模型、Adam、梯度、GPU batch、计算及通信张量。不含 reserved、CUDA context、NCCL 内部非 PyTorch 分配。",
             "- 下表为 7 组明确配置的完整 3×4 参数网格。Main/Aux 使用同一存储；不穷举独立 Main/Aux 存储混搭或逐阶段混合计算。none 是默认策略别名，不重复测量。",
             "- Main compact 的宽度是本批次选中列的并集，依赖激活分布与训练状态，不等于 k；Aux compact 的列集合来自 dead mask。本表固定激活、初始化和 dead 分布，不能仅根据 k 外推其他数据的 compact 时间。",
             "- AuxK=0 时 R_SS 与 R_SC 具有同一有效计算路径，复用同一实测行；CSV 的 case 字段可追溯。", "",
             "## 路径定义", "", "|编号|Main/Aux 存储|Main 计算|Aux 计算|", "|---|---|---|---|"]
    for name, values in POLICIES.items():
        lines.append(f"|{name}|{'|'.join(values)}|")
    lines += ["", "每格为 **step ms / peak allocated GiB**。共享/分片存储规范名称为 sharded_dense、sharded_ragged（shared_* 是 CLI 别名）。", ""]
    for tp in (1, 2):
        lines += [f"## TP{tp}", "", "|k|AuxK|" + "|".join(POLICIES) + "|",
                  "|---:|---:|" + "---:|" * len(POLICIES)]
        for k in (1, 128, 512):
            for aux in (0, 128, 512, 2048):
                cells = []
                for policy in POLICIES:
                    row = by_key.get((tp, k, aux, policy))
                    cells.append("PENDING" if row is None else "FAILED" if row["status"] != "OK" else
                                 f"{row['median_ms']:.2f} / {row['peak_allocated_gib']:.3f}")
                lines.append(f"|{k}|{aux}|" + "|".join(cells) + "|")
        lines.append("")
    lines += ["## 最快配置", "", "只在本次 7 组已测路径中排序。接近测量波动的差异不代表稳定优势。", "",
              "|k|AuxK|TP1 最快|ms / GiB|TP2 最快|ms / GiB|", "|---:|---:|---|---:|---|---:|"]
    for k in (1, 128, 512):
        for aux in (0, 128, 512, 2048):
            cells = []
            for tp in (1, 2):
                candidates = [r for r in summary if (r["tp"], r["k"], r["auxk"]) == (tp, k, aux) and r["status"] == "OK"]
                if len(candidates) < len(POLICIES):
                    cells += ["PENDING", "PENDING"]
                else:
                    best = min(candidates, key=lambda r: r["median_ms"])
                    cells += [best["policy"], f"{best['median_ms']:.2f} / {best['peak_allocated_gib']:.3f}"]
            lines.append(f"|{k}|{aux}|" + "|".join(cells) + "|")
    if not pending and all(r["status"] == "OK" for r in summary):
        spread = max(r["poststep_spread_bytes"] for r in summary)
        lines += ["", "## 数据检查", "",
                  f"全部独立运行成功，loss 有限、每步更新次数和 token 数通过运行断言；Main/Aux 各阶段 dispatch 与指定策略一致。20 个测量步内，释放输出后的 allocated 最大波动为 {spread / 2**20:.3f} MiB。此结论限定于该测量窗口。", "",
                  f"[完整 CSV（含 p10/p90）]({os.path.relpath(root / 'summary.csv', markdown.parent)}) · "
                  f"[热图 PNG]({os.path.relpath(root / 'grid.png', markdown.parent)}) · "
                  f"[热图 PDF]({os.path.relpath(root / 'grid.pdf', markdown.parent)})", ""]
    lines += ["", "## 复现与原始数据", "", f"结果目录：`{root}`。`cases.json` 保存实际配置，`summary.csv/json` 含 p10/p90、rank dispatch、步末 allocated 范围；`runs/<case>` 含逐步 loss、耗时、allocated 与日志，`runs/source_sha256.json` 保存运行源码哈希。", "",
              "```bash", ".venv/bin/python scripts/profile/profile_megatron_execution_time_v2.py \\",
              f"  --output {root}/runs --config-file {root}/cases.json \\",
              "  --warmup 5 --steps 20", "```", ""]
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text("\n".join(lines))


def plot(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rows = json.loads((root / "summary.json").read_text())
    assert len(rows) == 168 and all(r["status"] == "OK" for r in rows), "Complete the grid first"
    lookup = {(r["tp"], r["k"], r["auxk"], r["policy"]): r for r in rows}
    points = [(k, a) for k in (1, 128, 512) for a in (0, 128, 512, 2048)]
    fig, axes = plt.subplots(1, 4, figsize=(23, 8), constrained_layout=True)
    for ax, (tp, field, label) in zip(axes, [
        (1, "median_ms", "TP1 step (ms)"), (2, "median_ms", "TP2 step (ms)"),
        (1, "peak_allocated_gib", "TP1 allocated (GiB)"),
        (2, "peak_allocated_gib", "TP2 allocated (GiB)"),
    ]):
        data = np.array([[lookup[tp, k, a, p][field] for p in POLICIES] for k, a in points])
        im = ax.imshow(data, aspect="auto", cmap="YlOrRd",
                       vmin=min(r[field] for r in rows), vmax=max(r[field] for r in rows))
        ax.set_title(label)
        ax.set_xticks(range(len(POLICIES)), POLICIES, rotation=45, ha="right")
        ax.set_yticks(range(len(points)), [f"k={k}, a={a}" for k, a in points])
        for i in range(len(points)):
            for j in range(len(POLICIES)):
                value = data[i, j]
                ax.text(j, i, f"{value:.0f}" if field == "median_ms" else f"{value:.2f}",
                        ha="center", va="center", fontsize=8,
                        color="white" if im.norm(value) > .65 else "black")
        for boundary in (3.5, 7.5):
            ax.axhline(boundary, color="black", linewidth=.8)
        fig.colorbar(im, ax=ax, shrink=.65)
    fig.suptitle("Measured Megatron SAE: D=4096, F=16384, B=8192, H=3, dead=2048\n"
                 "5 warmup + 20 updates; median slowest-rank wall time; max-rank allocated", fontsize=14)
    fig.supxlabel("Storage prefix: S=sharded_dense, R=sharded_ragged, F=full. "
                  "Main/Aux suffix: S=sparse, C=compact, D=dense. a=AuxK.", fontsize=11)
    fig.savefig(root / "grid.png", dpi=180)
    fig.savefig(root / "grid.pdf")
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--build", action="store_true")
    p.add_argument("--markdown", type=Path)
    p.add_argument("--plot", action="store_true")
    args = p.parse_args()
    if args.build:
        build(args.root)
    else:
        report(args.root, args.markdown)
        if args.plot:
            plot(args.root)
