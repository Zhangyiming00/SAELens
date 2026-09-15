# 静态 routing：Megatron DDP 与梯度累计

静态 CUDA TopK SAE 按 hook 建立独立 Megatron DDP、梯度 buffer、fused
`torch.optim.Adam` 和 LR scheduler。DP1 使用同一条路径。FP32 参数可配合
BF16 autocast；CPU 测试使用兼容路径，不作为 Megatron DDP 验收。

## 配置和 batch 定义

| 配置 | 含义 |
| --- | --- |
| `train_batch_size_tokens` | 原有 provider microbatch 大小，含义和数据切分方式保持不变 |
| `gradient_accumulation_steps` | 每个更新窗口最多读取的 microbatch 数，默认 `1`，DP1 同样有效 |
| `ddp_bucket_cap_mb` | 每个 hook 的梯度通信 bucket 目标大小，单位 MiB，与累计次数独立；一个参数不会被拆开 |

**每个 hook 的名义全局更新 batch = 累计次数 × 所有 DP replica 的本地
microbatch 大小之和。** TP 复制和 hook 数量不乘入 batch。

- 直接构造 `LanguageModelSAERunnerConfig` 时，`routing_dp_batch_mode="equal"`
  保留原有本地 batch 语义；`"exact"` 保留全局 microbatch 语义。
- `run_sae_runner_gpu.py` 的 `--train-batch-size-tokens` 原本就是全局
  microbatch 参数；equal 模式的 CLI 仍先按 DP 数转换到本地配置。
- 例如 exact 的全局 microbatch 为 `32`、累计 `3` 次：DP1、DP3、TP2DP2
  的名义全局更新 batch 均为 `96`。DP3 每轮本地配额仍为 `10/11/11`。
- `training_tokens`、数据切分、token 排除、buffer 容量、混合比例和 shuffle
  都保留原有定义。不会为了累计把 provider batch 或 buffer 扩大。

命令行增加：

```bash
--routing-dp-batch-mode exact \
--train-batch-size-tokens 32 \
--gradient-accumulation-steps 3 \
--ddp-bucket-cap-mb 25
```

## 窗口语义

对于 hook `h`，令 `n[h,r,m]` 为 replica `r` 在 microbatch `m` 的有效 token
数。窗口梯度为：

```text
N[h] = sum_r sum_m n[h,r,m]
gradient[h] = sum_r sum_m (n[h,r,m] * gradient(mean_loss[h,r,m])) / N[h]
```

Megatron 的 FP32 `main_grad` 累计 token loss 的梯度之和。窗口前面的
microbatch 在 `no_sync()` 内；已知的最后一个 microbatch 退出 `no_sync()`，
由 Megatron 的 backward hook 在 bucket 就绪时自动发起 DP 通信，继续当前
hook 的其余 backward 及后续 hook 计算。GA1/2/3 使用同一机制。所有 hook
完成计算后才等待归约、按 `N[h]` 归一化，分别以 `max_norm=1` 裁剪，再分别执行 fused Adam 和
scheduler。TP 梯度仍由 Megatron 线性层完成；裁剪只把复制的 `b_dec` 计入一次。
Megatron 的 `overlap_grad_reduce=True` 启用 bucketing 和就绪触发；每个 bucket
只在其本窗口所有梯度写入完成后发起。Megatron 0.16.1 首窗口先记录就绪计数，
通过原生 `finish_grad_sync()` 发起，后续窗口才在 backward 内自动发起。

不预读 provider 来判断最后一批。完整配置窗口和显式有长度的输入可以识别
最后一个 microbatch；流式数据提前耗尽或预算截断形成的未知长度短窗口，
继续使用全程 `no_sync()` 加窗口末显式发起。该保守路径保留原有数据读取、
buffer/shuffle、尾批与中断语义。恢复后新建 DDP 同样先经历一次就绪计数学习。

### 原生 bucket 自动归约与通信组顺序

`SAETrainUnit.backward(..., sync_gradients=True)` 在最后一批 backward 前将
本窗口梯度所有权交给 Megatron，允许它逐 bucket 自动发起，同时禁止后续
另一次 backward。即使 backward 中途失败，也不能清零可能仍在归约的 buffer。
`SAETrainUnit.start_grad_sync()` 是窗口末回退接口；原生模式已接管时为空操作，
不会重复发起已经就绪的 bucket。显式模式每窗口幂等；
`finish_grad_sync()` 要求本窗口已交给原生自动或显式发起路径，通过 Megatron 的完成接口建立当前 CUDA
stream 对归约的依赖，然后将 `parameter.grad` 指向 `main_grad`。
最后一批 backward 内，Megatron 保证已归约 bucket 不再写入，其他未就绪
bucket 仍可计算；随后禁止新增 microbatch 的 backward。未完成依赖前禁止
清零、裁剪和 unit optimizer 更新。开始/结束的 buffer 清零次数保持原样。
CPU 兼容路径仍在窗口末归约。

TP1 只使用一个实际通信组，可以直接提前发起。TP>1 且 DP>1 时，后续 hook
会使用另一个 NCCL group，必须满足跨组发起顺序要求。当前实现只在
**NCCL >= 2.26 且启动前设置 `NCCL_LAUNCH_ORDER_IMPLICIT=1`** 时启用此路径；
否则保留窗口末发起，并在启动日志中报告每 hook 是否启用。不要在已经建立
NCCL communicator 后修改该环境变量；运行库不会代替调用者在初始化后设置它。
初始化时在整个训练 domain 对可用性取交集；任一 rank 不支持则所有 rank
一起退回晚发起，避免各 rank 选择不同的 collective 顺序。

```bash
NCCL_LAUNCH_ORDER_IMPLICIT=1 torchrun ... run_sae_runner_gpu.py ...
```

通用 [PyTorch 2.10 多 process group 说明](https://docs.pytorch.org/docs/2.10/distributed.html#torch.distributed.new_group)
要求调用者建立通信之间的同步关系。此处 TP+DP 提前路径使用
[NCCL 2.27.5 的 implicit launch ordering 契约](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2275/user-guide/docs/usage/communicators.html#using-multiple-nccl-communicators-concurrently)：
所有 rank 按相同 hook、microbatch 和 autograd 图顺序发起。一个 hook 的
backward 返回后才进入下一 hook；本次允许原生 DDP 在该 backward 的参数
梯度 hook 中发起，TP/DDP 的交错由相同计算图及 NCCL ordering 契约约束。
该功能由 NCCL 对跨 communicator 的执行建立顺序；CUDA runtime/driver >=12.3
允许内核并发，较早版本会序列化。窗口末仍使用 Megatron 的真实 wait 建立梯度
读写依赖，不读取或伪造其内部归约 handle，也不自写 all-reduce。

局部 token 为零的 rank 仍执行 dummy backward 和同样的发起序列；每 hook
的 dead-feature mask 在 TP/DP 内一致。provider/routing 通信只出现在窗口
计算前；下一窗口的数据获取发生在本窗口所有归约依赖与更新之后。

短 microbatch 和不足累计次数的最后窗口都按实际有效 token 数归一化。
空 replica 仍参与相同 forward/backward 和 collective 序列，dummy token 不计数。
整个窗口没有有效 token 的 hook 不更新参数、Adam 或 scheduler；全空窗口也不
推进 scaler。AMP 跳过的 hook 不推进 scheduler。

dead-feature mask 在窗口内固定，窗口末合并 firing 信息并推进一次计数，
与原生 SAELens 把相同 token 拼成一个大 batch 的行为一致。
LR 的训练步数按 `ceil(训练预算 / 名义更新 batch)` 规划，warmup/decay/
feature window 的步数均以更新窗口计。既有 exact 模式对训练预算的整
microbatch 限制保留；尾窗口不要求整更新 batch。

启动日志和 MSE 记录包含名义 `global_update_batch_size`；MSE 记录还包含
`window_microbatches` 和 `global_valid_tokens_by_hook`。整体训练进度继续
使用原有本地预算及 exact 的余数记账，hook 有效 token 数独立用于梯度归一化。

## 检查点

周期检查点在窗口的优化器、scheduler 和进度提交后保存。恢复包含模型、
各 hook Adam/scheduler、scaler、统计、原有 provider/buffer/shuffle 状态和
token 进度。检查点身份包含累计次数，禁止无提示地改变更新 batch；旧检查点
按累计 `1` 解释。通信 bucket 可在恢复时改变。
旧检查点的 Adam 学习率、betas、eps 和 moments 继续恢复；CUDA 静态路径的
执行方式固定为 fused Adam，不受旧的 foreach/for-loop 标记影响。

窗口中途失败或中断不会生成声称完整的检查点，须从上一个完成的检查点恢复。
同 bucket 的确定性验收要求逐位一致；改变 bucket 可能改变 NCCL 求和顺序，
因此允许规定的浮点误差。

## 验收方法

`tests/native_reference/accumulation.py` 在 `python -I` 子进程中校验并解压
上游 `sae-lens==6.37.6` wheel，断言导入来自该 wheel。原生模型把相同 token
拼成大 batch，使用普通 Adam，独立生成裁剪前后梯度、梯度范数、参数、Adam
状态、LR 和 dead-feature 计数；本仓库 CUDA 路径使用 Megatron DDP 和 fused Adam。

```bash
.venv/bin/pytest -q tests/training/test_gradient_accumulation.py
.venv/bin/pytest -q tests/training/test_sae_resume_boundaries.py
.venv/bin/python scripts/validate_sae_runner_static.py \
  --output results/accumulation_check --layout dp3 --batch-mode exact \
  --hooks 1 2 --accumulation-steps 3 --ddp-bucket-cap-mb 0.125
```

原生对照覆盖 DP1、DP3、TP2DP2 × 单/双 hook × 累计 `1/2/3`，其中累计 `3`
用两种 bucket 重复验证，共 24 组；包含不同 replica/hook token 数、空 replica、
全空 hook、短 microbatch、尾窗口，以及保持/改变 bucket 的磁盘续训。
判定容差为 `atol=4e-6, rtol=4e-5`，同 bucket 续训使用零容差。

真实 runner 验收用本地 Llama-3.1-8B，经 vLLM、静态 routing、token 排除和
混合 buffer，核对恢复前后每一个 microbatch 的内容哈希，并比较模型、Adam、
LR、scaler、进度和进程组回收结果。结果保存在 `results/accumulation_20260915/`。

### 2026-09-15 验收结果

| 拓扑 | 裁剪前梯度最大绝对误差 | 参数最大绝对误差 | 同 bucket 续训 |
| --- | ---: | ---: | --- |
| DP1 | `4.77e-7` | `2.98e-8` | 逐位一致 |
| DP3 | `9.54e-7` | `2.98e-8` | 逐位一致 |
| TP2DP2 | `9.54e-7` | `2.98e-8` | 逐位一致 |

24 组原生对照全部通过，改变 bucket 的续训结果也在容差内。真实 runner
的三种拓扑各自单/双 hook，以及 equal 模式的不等过滤 token 回归，均通过
逐位续训检查。现有回归为 104 通过、1 跳过；独立窗口边界测试 7 项通过，
包括 TP 单分片溢出同步跳步、scaler、旧 Adam 执行标记迁移和半窗口中断。

汇总报告：`results/accumulation_20260915/summary.json`。每个真实 runner
验收目录记录当时的调用参数、代码哈希和逐 rank 结果；汇总记录最终代码哈希。

### GA=1 提前归约的专项验收

新增 early/late 调度的逐位对照，检查每次更新的裁剪前后梯度、范数、参数、
Adam 状态和 LR；同时检查每 hook 每窗口只发起一次。原生大 batch 的 24 组
矩阵、空 replica/全空 hook、尾窗口、同/异 bucket 续训，以及 AMP 溢出和
半窗口检查点限制继续通过。额外覆盖 pending buffer 清零/写入保护、重复
start/finish、部分 bucket 发起失败，以及训练 domain 的启用条件一致性。

性能只比较冻结的优化前代码与本次改动，并分别记录 CUDA trace 和关闭 trace
的配对窗口。H2 小配置的结果与 H3/d32768/K128/稠密/context2048 分开报告；
H3 另有 GA 改动前 `13b0858` 的同配置历史参考。
完整结果、源码指纹和复现命令见
[提前归约专项报告](../results/early_hook_reduce_20260915/README.md)。

后续原生 bucket 就绪调度的实现、GA1/2/3 逐位对照和同配置性能结果见
[原生 bucket 归约报告](../results/native_bucket_sync_20260915/README.md)。
