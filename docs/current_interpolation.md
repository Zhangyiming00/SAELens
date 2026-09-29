# 当前原生训练与稀疏阶段的插值模型

唯一执行模型入口是 `sae_lens/autoconfig/execution_time_model.py`，统一提供原生完整
更新的时间/显存预测和稀疏阶段预测，保持多线性插值。校准 schema 仍为
`native_interpolation_v3` / `sparse_parts_interpolation_v1`，已有数值和插值规则不变。
`allocated_peak_model.predict_native` 对当前 schema 委托同一入口。

旧 v1/v2 phase 缩放、k/AuxK 加法近似及对应模拟入口已移除；旧 schema=1/2、
`k_aux_v1` 不再作为当前执行预测输入。原生运行 profiler
`profile_megatron_execution_time_v2.py` 仍用于实际测量；它的名称不代表存在第二个预测模型。
本机校准结果保存在 `results/current_interpolation_20260928/README.md`。

## 查询

在仓库根目录执行；`--configs` 是名字到完整配置的 JSON 映射，可参考结果中的
`holdout_configs.json`。输出路径必须不存在，避免覆盖冻结预测。
以下命令使用按当前源码重新冻结的 profile（生成步骤见“重新校准”）。历史表的
数值可直接通过 Python 接口回放，但旧源码指纹不能直接通过当前 CLI 的检查。

```bash
.venv/bin/python scripts/profile/simulate_megatron_execution_time.py predict \
  --profile results/new_profile/profile.json \
  --configs results/new_profile/holdout_configs.json \
  --output /tmp/sae_predictions.json
```

Python 接口：

```python
from sae_lens.autoconfig.execution_time_model import predict_native, predict_sparse
from sae_lens.autoconfig.allocated_peak_model import predict_native as predict_memory

prediction = predict_native(config, native_profile)
rank0_memory = predict_memory(config, native_profile, rank=0)
stages = predict_sparse(sparse_config, sparse_profile)
```

`config` 使用 profiler 的完整格式；不省略 None/字符串 none，不把旧 backend 名称
当作当前计算策略。精确匹配离散字段后，只对 profile 声明的数值轴做多线性插值。
`corners` 列出源测量和权重。未测执行族、缺角、越界和跨路径边界明确报错。
同一 CLI 的 `predict` 也识别 `sparse_parts_interpolation_v1`，将 `--profile` 和
`--configs` 指向 `sparse/` 内的对应文件即可查询三个独立阶段。

完整更新的时间包含真实调度，因此 H、GA、并行拓扑、ZeRO 和 overlap 配置均不能
未经测量随意改变。`timeline=True` 仅导出完整服务区间，不伪造 CUDA stream 时间线。
该模型可以提供 streaming 模拟的 SAE 消费时间；producer、传输和共置模型仍需其
自己的校准，不能把本次 SAE-only 数据当成在线端到端测量。

## 显存

每个 rank 的结构部分是 FP32 参数、DP 持续梯度、Adam 状态、预载 GA 输入窗口和
统计向量。ZeRO 按实际参数注册顺序、bucket padding 和 DP 所有权计算。
在目标形状重新计算结构部分，对已测 resident/peak 减去结构部分的余量分别插值。
训练阶段的临时空间不独立相加；返回的 `transient_above_resident_bytes` 是峰值与
稳定驻留的差值。reserved、allocator 碎片以及 CUDA/NCCL 外部占用不在 allocated
预测中，所以单凭预测低于物理显存不能保证不会 OOM。
这里的峰值是预热后的训练更新窗口；初始化、保存 checkpoint 和 vLLM 加载峰值需要
各自测量，不由这张表代替。

## 重新校准

每个新实验使用新输出目录，依次运行以下 phase：

```bash
.venv/bin/python scripts/profile/profile_current_interpolation.py --output results/new_profile --phase prepare
.venv/bin/python scripts/profile/profile_current_interpolation.py --output results/new_profile --phase calibration
.venv/bin/python scripts/profile/profile_current_interpolation.py --output results/new_profile --phase freeze
.venv/bin/python scripts/profile/profile_current_interpolation.py --output results/new_profile --phase holdout
.venv/bin/python scripts/profile/profile_current_interpolation.py --output results/new_profile --phase validate
.venv/bin/python scripts/profile/profile_sparse_parts.py --output results/new_profile/sparse_coarse
.venv/bin/python scripts/profile/refine_sparse_parts_profile.py --base results/new_profile/sparse_coarse --output results/new_profile/sparse_tiled
.venv/bin/python scripts/profile/extend_sparse_profile.py --base results/new_profile/sparse_tiled --output results/new_profile/sparse --rows 5120 6144 7168 --holdouts 4608:24576 6656:49152
.venv/bin/python scripts/profile/profile_megatron_execution_time_v2.py --output results/new_profile/trace --config-file results/new_profile/calibration_configs.json --cases tp1_b8192_f65536 mixed_tp1_b8192_f65536 --warmup 3 --steps 3 --trace
.venv/bin/python scripts/profile/analyze_sparse_parts_trace.py results/new_profile/trace/tp1_b8192_f65536
.venv/bin/python scripts/profile/analyze_sparse_parts_trace.py results/new_profile/trace/mixed_tp1_b8192_f65536
.venv/bin/python scripts/profile/report_current_interpolation.py results/new_profile
```

实际执行仍使用 `profile_megatron_execution_time_v2.py` 的 native trainer；该 profiler
现在记录完整 resolved model config 和 optimizer 类型，支持 execution 中覆盖 engine，
并为 sparse forward/dvalues/dweight 增加独立 NVTX range。不同运行框架或优化器不混表。

要扩充网格，可给原生 profiler 提供新的完整配置映射，再用统一 CLI 的 `calibrate`
冻结表。校准和验证目录必须分开。校准表保存输入与源码 SHA；预测 CLI 检查指纹，
运行时实现改变后必须重新 profile。未经采样的新 dead 分布、k、AuxK 路径、workspace
或调度组合不由临近配置猜测。

稀疏微基准直接执行生产函数，分别测 forward、dvalues、dweight；dvalues 复用
forward 的 page groups。CUDA event elapsed、同步 wall 时间和新增 allocated 峰值
分别记录，不把 event elapsed 称为纯 kernel 时间。局部 k、行长模式、engine、分页、
workspace 为离散键；输入含 selected zeros。主模型不把三个微基准时间直接相加来
替代完整更新；这两张表分别服务于原生配置预测和局部 kernel 路径比较。
微基准在 feature-major 连续 decoder 权重和已选中 entries 上运行，不包含上游
encoder、TopK、权重转置准备和 optimizer；原生完整更新的测量包含这些工作。
`extend_sparse_profile.py` 可用 `--rows` / `--widths` 扩充已有网格，保留已有 tile
边界；`--holdouts` 指定新的 `rows:width` 验证形状。它检查源码和运行环境，避免将
不同实现的旧数据拼入新表。前一轮用来指导加密的验证点按诊断数据处理，最终报告
只统计重新留出的形状。

本机实验将 rank 映射到 GPU0..world_size-1。GPU 放置、拓扑、Torch/CUDA、输入分布
改变后应提供新的校准表；CPU 预测接口本身不访问 GPU，也不会自动检测硬件变化。

模型合并没有改写历史 profile 或冻结预测。Python 接口仍可读取原有当前格式的数值表；
CLI 保持源码指纹检查，旧文件路径消失或源码已变更时会拒绝。仅预测代码重构可从保留的
测量结果重新冻结校准表；如果实际训练/kernel 源码也发生变化，应重新测量，不能只更新摘要。

`profile_elastic_watermarks.py` 的实际在线实验不再自动使用旧 v2 估计。
需要预估时同时传入 `--native-profile` 和 `--native-configs`（包含 `dp2`、`dp3` 完整配置）；
它调用同一个严格插值入口，未校准的 DP3、GA、dead 或优化器组合会明确报错。
