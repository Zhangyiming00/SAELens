# Sharded TopK 本地集成与验收（2026-09-22）

已将 `/root/saelens_sharded_topk_delivery.zip` 的补丁应用到本地
`afebe72` 基础上的工作树，`git apply --check` 通过。保留了原有未提交文件和
`third_party/vllm` 修改。交付包原始实现说明见 [sharded_topk.md](sharded_topk.md)；
该文件中“未做 GPU 验证”描述的是原交付状态，本次验证结果如下。

## 启用方式

现有 `run_sae_runner_gpu.py` 训练命令追加：

```bash
--sae-topk-backend sharded_dense --sae-topk-protocol auto
```

这是本次大 batch 测量中更省显存、耗时更少的分片配置。配置类对应
`TopKTrainingSAEConfig(topk_backend="sharded_dense")`，由 runner 创建
`MegatronTopKSAE`。默认仍为 `legacy`，**不加参数不会启用本次优化**。
无需改变 batch、TP、GA 或 hook 数。

需要试验本地稀疏 decoder 时：

```bash
--sae-topk-backend sharded_sparse --sae-sparse-decoder torch
```

`--sae-topk-keys triton --sae-sparse-decoder triton` 也已通过本机小规模数值验证；
尚无本次真实规模的 Triton 性能测量，不默认启用。

## 实现内容和边界

- Encoder 只生成本地 `[B, d_sae/TP]`。精确全局 TopK 使用本地 `min(k, shard_width)`
  候选及 int64 比较键；候选接收区过大时切换到分布式 radix 直方图归约。
  两条路径均不回退全量 activation gather。
- `sharded_dense` 输出本地 dense 激活，沿用原生 RowParallelLinear；
  `sharded_sparse` 输出本地 COO，通过 weighted embedding_bag 或 Triton 解码。
  `[B, d_in]` 重构结果仍在 TP 内归约。
- AuxK 保留负激活、全局 dead mask 语义、可微 decoder norm 和 detached residual。
  GA、单/多 trainer、评估和日志使用局部激活或复制的一维 `[d_sae]` 统计。
- wavefront 保留分阶段 encode/decode/reduce 调度；参数、分片轴、Adam 和 checkpoint
  格式不变。Sparse decoder 使用普通 dense 参数梯度，单独关闭其 linear fusion。
- **没有实现 sparse encoder backward**；本地 dense scores、梯度和工作区仍可能共存。
  参数和 optimizer 状态仍是 dense 分片。TP1 时本地宽度仍等于 `d_sae`。
- 并列分数按全局 feature ID 升序打破并列；与旧 `torch.topk` 未规定的 tie 顺序
  不保证一致。`encode()` 返回本地 feature；TP>1 下会拒绝全局 post-latent hooks。

## 本次修复

1. PyTorch `2.10.0+cu128` 的 CUDA `embedding_bag` 未实现 BF16
   per-sample-weight backward。Torch sparse decoder 保留 BF16 输入舍入后，
   使用 FP32 本地参数/稀疏值计算，再转回输出 dtype，保持 autograd 连通。
   额外内存仅涉及本地参数和稀疏值，没有全局 latent；该成本计入测量。
2. 删除交付验收脚本强制设置的 `CUDA_MODULE_LOADING=EAGER`。该设置在本机
   NCCL 初始化时导致 `invalid resource handle`，包括 legacy 控制组。
   使用环境默认 LAZY 后相同测试通过；没有修改生产 runtime 的 CUDA 设置。
3. 将已有独立 upstream oracle 的 TP1 数值和 trainer 续训测试扩展到两种分片后端；
   清理新增代码格式及 trainer 中不再使用的局部变量。

## 实际验证

环境：4 × RTX 5090 32 GiB，PyTorch `2.10.0+cu128`，NCCL `2.27.5`，
当前虚拟环境中实际安装的 Megatron Core。

| 范围 | 结果 |
|---|---|
| 交付独立 CPU/Gloo + CUDA/Triton 测试 | 96 passed，包含原先跳过的 6 个 CUDA 测试 |
| 本地模型、参数边界、独立 oracle、trainer 续训和 train-unit 回归 | 44 passed，3 个 CUDA 测试在沙箱 CPU 运行中 skipped，5 个分布式测试 deselected |
| 原生 GPU TP2×DP2、双 hook、GA2、wavefront、AuxK | 三后端通过 loss、梯度、参数和 Adam 数值比较 |
| 原生 GPU TP1 无 norm；TP4 强制 radix | 通过 |
| 原生 GPU TP2×DP2：Triton；BF16 autocast；Triton + BF16 | 通过 |
| 真实单/多 trainer，TP2×DP2，GA2，空/不均匀 batch、尾窗口、特征年龄 | 两种分片后端通过独立 oracle |
| 原生 bucket 提前同步、FP32Optimizer 裁剪/更新、checkpoint 恢复及 bucket 大小变更 | 两种分片后端通过 |

FP32 小规模 A/B 中 sharded_dense 快照差异为 0，Torch sparse 最大绝对差异
`8.41e-7`。BF16 sparse 最大快照差异约 `9.93e-4`，通过脚本的
`atol=rtol=5e-3`；这不是 BF16 bitwise 等价声明。
真实 trainer 报告逐 rank 记录梯度、裁剪、参数、Adam 的误差和恢复结果。

最初的通用 `test_sae_trainer.py` 扩大回归因无关的 TinyStories 模型下载失败而中止，
没有算作通过；上述 44 项回归和真实 trainer 验证均使用本地数据/独立 oracle。

## 大 batch 显存测量

合成 SAE 训练；FP32、TP4×DP1、每个 hook 的 B=16384、d_in=4096、d_sae=16384、
k=128、3 hooks、GA1、wavefront、decoder norm 开启、AuxK 关闭、Torch keys/decoder。
1 个 warmup 窗口，2 个测量窗口；表中取各 rank 的最大 allocated 峰值及最大平均窗口时间。

| 后端 | 每卡峰值 allocated | 合成更新窗口时间 |
|---|---:|---:|
| legacy | 12.712 GiB | 231.9 ms |
| sharded_dense | 7.640 GiB | 210.4 ms |
| sharded_sparse / Torch | 8.905 GiB | 714.9 ms |

此负载下 sharded_dense 节省 **5.07 GiB / 39.9%**。Sparse 表示不保证总峰值更低，
Torch decoder 的内部工作区及参数布局开销也计入总量。
时间包含输入生成、前反向、归一化、裁剪和 Adam，只有两个测量窗口，不能当作
在线 LLM→SAE 吞吐结论。大规模使用 `--benchmark-only`，没有保存参数副本做 A/B；
数值正确性来自前述小规模验收。

两种分片模式在首次 warmup 前向和反向开启 TorchDispatch guard，未出现
`[B, d_sae]` 算子输出。它结合 allocator 峰值支持本次内存结论，不能视为对任意
CUDA 内部临时分配的完整追踪。

原始日志、逐 rank JSON、命令脚本及汇总在
[`results/sharded_topk_20260922`](../results/sharded_topk_20260922/)，
其中 [`summary.json`](../results/sharded_topk_20260922/summary.json) 汇总 A/B 与显存。

复现大 batch：

```bash
.venv/bin/torchrun --standalone --nproc_per_node=4 tools/validate_sharded_topk_gpu.py \
  --tp 4 --dp 1 --global-batch 16384 --d-in 4096 --d-sae 16384 --k 128 \
  --hooks 3 --ga 1 --wavefront --warmup 1 --steps 2 --benchmark-only \
  --output results/sharded_topk_recheck
```

未验收：原生 ZeRO/distributed optimizer、提前 optimizer overlap、FSDP、
CUDA Graph/compile、在线 LLM streaming、elastic 切换。已验证的提前梯度 bucket 同步
不等于提前 optimizer overlap。现有独立 memory predictor 仍按 legacy 建模，
本次未修改用户已有的 memory-model 文件；新后端容量判断以实测为准。
