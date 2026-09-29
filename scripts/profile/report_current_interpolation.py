"""Render independently measured interpolation validation and scope."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    args = p.parse_args()
    root = args.root
    profile = json.loads((root/"profile.json").read_text())
    validation = json.loads((root/"validation.json").read_text())
    sparse = json.loads((root/"sparse/validation.json").read_text())
    sparse_profile = json.loads((root/"sparse/profile.json").read_text())
    refined = sparse_profile.get("refinement")
    hold_cfg = json.loads((root/"sparse/holdout_configs.json").read_text())
    sparse_hold_shapes = sorted({(c["rows"], c["width"]) for c in hold_cfg.values()})
    rows = validation["rows"]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, metric, unit, divisor in ((axes[0], "ms", "Native update (ms)", 1),
                                      (axes[1], "peak_bytes", "Peak allocated (GiB)", 2**30)):
        xs = [r["actual_"+metric]/divisor for r in rows]
        ys = [r["predicted_"+metric]/divisor for r in rows]
        lim = max(xs+ys)*1.06
        ax.plot([0, lim], [0, lim], color="gray", lw=1, label="Exact prediction")
        ax.scatter(xs, ys, s=36, color="#147d92")
        ax.set(xlim=(0, lim), ylim=(0, lim), xlabel="Measured: "+unit, ylabel="Interpolated: "+unit)
        ax.grid(alpha=.2)
    fig.suptitle("Independent holdouts: current native execution and allocator peaks")
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(root/f"validation.{suffix}", dpi=160)
    plt.close(fig)
    lines = ["# 当前执行路径的插值 profiler、模拟器与显存模型", "",
             f"原生校准 **{len(profile['rows'])}** 点，先冻结预测，再运行 **{len(rows)}** 个独立验证点。每点独立进程，4 次预热、12 次测量；每步取最慢 rank，再取步骤中位数。",
             "稀疏阶段另外调用生产 `sparse_parts` 的 forward、dvalues、dweight；包含 adapter、分页、排序和输出分配，不使用 dense 替代。", "",
             f"原生时间 MAPE **{validation['time_mape_pct']:.2f}%**，最大绝对百分比误差 **{validation['time_max_error_pct']:.2f}%**。",
             f"峰值 allocated 显存 MAPE **{validation['memory_mape_pct']:.2f}%**，最大绝对百分比误差 **{validation['memory_max_error_pct']:.2f}%**。", "",
             "![Validation](validation.png)", "", "## 原生独立验证", "",
             "| 配置 | 预测/实测 ms | 时间误差 | 预测/实测 GiB | 显存误差 |",
             "|---|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['name']} | {r['predicted_ms']:.2f} / {r['actual_ms']:.2f} | {r['time_error_pct']:+.2f}% | {r['predicted_peak_bytes']/2**30:.3f} / {r['actual_peak_bytes']/2**30:.3f} | {r['memory_error_pct']:+.2f}% |")
    lines += ["", "## 插值与内存语义", "",
              "- 原生整次更新在 Batch × d_sae 的完整矩形中做双线性插值；单轴退化为线性插值，完全相同的点直接使用测量中位数。没有拟合系数或理论 FLOPS 缩放。",
              "- TP/DP/PP、H、GA、wavefront、optimizer overlap、ZeRO、k/AuxK、dead 状态、engine、workspace、forward/dvalues/dweight 策略均为精确匹配的离散条件。Python null 与字符串 none 不合并。",
              "- 插值的角点必须有相同的实测执行路径；遇到缺角、越界、TopK 算法边界或不同实际 dispatch，会报错要求补测。未校准的 scheduler 不从串行时间推算。",
              "- 显存先按目标配置和 rank 计算 FP32 参数、梯度 buffer、Adam 状态、缓存输入与 feature statistics，再插值 resident 和 peak 的剩余分配。DP/ZeRO 的持有关系沿用真实 bucket 布局。",
              "- peak allocated、resident allocated 和 transient 分别输出。没有把各稀疏阶段的独立峰值相加，也没有把 reserved 或 CUDA/NCCL 的外部占用算成张量内存。容量判断还需要为这些外部占用留空间。",
              "- 显存峰值在预热后的训练更新窗口内采集，初始化和保存 checkpoint 的峰值不在此表中。",
              "- 微基准用于查询和比较稀疏阶段；原生 update 时间直接插值完整更新，包含实际通信和调度。独立阶段的耗时不直接相加来替代有并行重叠的训练更新。", "",
              "## 稀疏阶段重测", "",
              f"校准 {len(sparse_profile['rows'])} 个形状/路径，独立验证 {len({r['name'] for r in sparse['rows']})} 个形状。每阶段 3 次预热、9 次测量。",
              "覆盖 OpenAI/Triton、局部每行 entries=32/64/128/256、等长和交替长短行、默认 256 MiB workspace，以及 OpenAI k128 的 64 MiB workspace。selected zeros 保留；两种引擎的三阶段结果均与 dense 数学参考检查。",
              (f"初轮验证用于发现转折，随后增加中间 rows/width 和 OpenAI row-tile 饱和边界两侧，再补 B=5120/6144/7168。最终独立验证使用 (rows,width)={sparse_hold_shapes}；前两轮诊断验证点不计入最终独立验证。"
               if refined else "校准 rows=1024/8192、local width=16384/65536；独立验证 rows=4096、width=32768。"),
              "k 是局部真实 entry 数的基准均值，不把 TP 下全局 k/TP 当成实际所有行的长度。合成 feature IDs 使用固定互素步长；不代表任意热点分布。", "",
              "| 微基准指标 | MAPE | 最大绝对百分比误差 |", "|---|---:|---:|"]
    for key, stats in sparse["summary"].items():
        lines.append(f"| {key} | {stats['mape_pct']:.2f}% | {stats['max_error_pct']:.2f}% |")
    memory_worst = max((r for r in sparse["rows"] if r["metric"] == "peak_extra_bytes"), key=lambda r: abs(r["error_pct"]))
    lines += ["", f"局部阶段仍有约 {sparse['summary']['wall_ms']['max_error_pct']:.1f}% 的耗时偏差，不能用其区分性能接近的 kernel 方案；本轮最差值也不是未来误差上界。主模型直接使用完整 native 更新的插值，因此这些局部误差不会累加进入原生时间预测。",
              f"临时显存最大百分比偏差对应 `{memory_worst['name']}/{memory_worst['stage']}`，实测 {memory_worst['actual']/2**20:.2f} MiB，预测 {memory_worst['predicted']/2**20:.2f} MiB。小分配的 allocator 块复用/粒度会放大百分比误差，不能将该比例套用到整个模型的 GiB 级峰值。"]
    if refined:
        coarse = json.loads((root/"sparse_coarse/validation.json").read_text())
        lines += ["", f"保留 {refined['reused_calibration_rows']} 个原校准点，新增 {refined['new_calibration_rows']} 个校准点。初轮粗网格的 wall MAPE 为 {coarse['summary']['wall_ms']['mape_pct']:.2f}%，局部临时显存最大误差为 {coarse['summary']['peak_extra_bytes']['max_error_pct']:.2f}%；粗表完整保存在 `sparse_coarse/`。最终误差来自不同验证形状，不是同一测试集调参后的分数。"]
    lines += ["", "CUDA event elapsed 包含该 stream 上的发射间隙；它不是 nsys kernel duration 之和。wall 是同步后主机时间。微基准的合成行分布不代替真实训练的 nnz 分布。", "",
              "## nsys 阶段归因抽查", "",
              "两组 B8192/F65536/H1/noaux 各预热 3 次、捕获 3 次；核对真实 native 路径。下面是每次更新的 GPU kernel 时长之和，不是 wall 时间。", "",
              "| decoder 策略 | sparse forward ms | sparse dvalues ms | sparse dweight ms | 未找到 launch 的 kernel |",
              "|---|---:|---:|---:|---:|"]
    for name, label in (("tp1_b8192_f65536", "sparse/sparse/sparse"),
                        ("mixed_tp1_b8192_f65536", "sparse/sparse/dense")):
        trace = json.loads((root/"trace"/name/"sparse_stages.json").read_text())
        stages = trace["stages_by_rank"]["0"]
        values = [stages[k]["kernel_ms_per_update"] for k in ("sparse_forward", "sparse_dvalues", "sparse_dweight")]
        lines.append(f"| {label} | {values[0]:.3f} | {values[1]:.3f} | {values[2]:.3f} | {sum(trace['unmatched_kernel_launches'].values())} |")
    lines += ["", "mixed 路径的 dweight 不再落入稀疏 range，forward/dvalues 仍正确归因。捕获数据只用于核验，未混入无 profiler 的 wall 校准。", "",
              "## 范围与文件", "",
              "硬件：4× RTX 5090，Torch 2.10.0+cu128，FP32/TF32 off。使用已有真实 activation cache，未加载 vLLM；这些结果不更新 producer、SHM 传输或共置争用模型。Llamascopium 的优化器、TopK 与 AuxK 语义不同，其历史数据没有混入 SAELens 校准。",
              "rank 固定映射本机 GPU0..world_size-1；更换 GPU 放置、硬件、软件或 dead 分布需要对应的新校准。",
              "原生主体覆盖 H1 的 TP1/TP2/TP4/DP2/DP4/TP2DP2、DP4 与 TP2DP2 ZeRO，另有 H3 TP4 lazy、TP2 AuxK=2048/dead=2048、单卡 Triton 和单卡 sparse/sparse/dense 混合 decoder。主体范围 B=4096..8192、d_sae=32768..65536；单卡 OpenAI noaux 另有 B=18432/F65536 和 B4096/F163840 的一维边界，不能据此填补未测二维角点。", "",
              "- `profile.json` / `predictions.json` / `validation.json`：原生表、测量前预测、独立验证。",
              "- `calibration/` / `holdout/`：完整配置、命令、rank 结果和源码指纹。",
              "- `sparse/`：稀疏三阶段表、数值检查、独立验证。",
              "- `trace/`：两组 nsys 原文件、SQLite、逐阶段 kernel 归因。",
              "- `tests.xml`：32 项模型与内存测试通过；`hardware.csv` / `topology.txt` 记录硬件。",
              "- `sparse_first_attempt/`：保留首次 JSON 字段顺序比较误报的数据，不参与最终结果。修正后完整重跑得到粗表，再进行边界加密。",
              "- 使用说明：[docs/current_interpolation.md](../../docs/current_interpolation.md)。", ""]
    (root/"README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
