# 精确 Sharded TopK + 本地稀疏 Decoder

> 本文件保留交付包的实现说明与原始验证状态。2026-09-22 本地集成、GPU 修复和
> RTX 5090 实测见 [sharded_topk_integration.md](sharded_topk_integration.md)。

## 交付状态与源码基线

本补丁根据用户归档 `full_fixed_megatron_20260918_diagnostics.zip` 的
`sources/megatron` 完整源码快照制作，快照日期为 **2026-09-18**。
本次 GitHub Git 拉取遇到 DNS 失败，网页/原始文件访问也未能取得该 fork 的最新源码。
**没有拉取或验证最新 main；本补丁不是针对已确认的当前远端 HEAD。**
`docs/sharded_topk_baseline.json` 记录被修改原文件的 SHA256。其中本地 Git
commit 仅为制作补丁而创建，绝不是远端提交 ID。

当前已实现可执行源码和测试，不是伪代码。CPU/Gloo 的验证结果见交付包中的
`TEST_REPORT.md`、原始 pytest 日志和 JUnit XML。
CUDA、Triton JIT、原生 Megatron Core/NCCL、真实 SAE runner 和吞吐尚未实测。
Triton 后端是显式开启的实验路径，不是默认路径。

## 1. 内存不变量

记 `B` 为某一个 DP 副本本次 microbatch 的 token 数，`D=d_sae`，`p=TP`，
`S=D/p`。`B` 不是所有 DP 副本 batch 的总和；同一 TP 组必须处理相同 token。

启用 `sharded_dense` 或 `sharded_sparse` 时：

- encoder 只产生本地 `hidden_pre[..., S]`。
- `feature_acts` 的逻辑末维始终为 `S`。稀疏模式是本地 COO，不是全局 COO。
- 选择只交换比较键/直方图，不聚合任何完整的 `hidden_pre[..., D]` 或
  `feature_acts[..., D]`；不存在先聚合再切片、先 densify 全局再转 sparse 的步骤。
- encoder backward 仍由原生 dense linear 执行，但其 latent 梯度仅为 `[B,S]`。
- decoder/reconstruction 的 `[B,d_in]` AllReduce 保留；这不是 latent 聚合。
- firing/dead 统计允许复制一维 `[D]` 向量；不复制 `[B,D]`。
- checkpoint/优化器仍可聚合参数，这不属于本补丁禁止的 activation gather。

**这个约束不是“总峰值最多等于一份 `[B,S]`”。** 本地 scores、局部梯度、比较键、
候选、COO 坐标、排序 workspace、参数和 optimizer 状态可以共存；多 Hook 也仍可能
同时保留各自的本地 scores。承诺是 token×feature 数据不再在 TP rank 上恢复全局宽度。
TP=1 时 `S=D`，此约束自然退化为单卡完整宽度。

## 2. 精确选择协议

### 2.1 正常 k：唯一比较键的候选交换

每个 rank 对本地 scores 选出 `min(k,S)` 个候选。不能用固定 `k/p`：全局 winner
可能全部落在同一个 shard。

比较键采用 int64：高位编码 FP32 的有序数值，低 32 位编码反向全局 feature ID。
FP16/BF16 可无损转成 FP32 参与比较。所有选择使用同一个总序：

```
score 降序；score 相等时 global feature index 升序
```

不会给 score 加 epsilon，不改变接近阈值的浮点数。正负零统一排序；NaN 统一排在
+inf 之前，但选中的实际 NaN 值不会被消毒，仍可传播到 loss。
本路径拒绝 FP64 score 和超过 uint32 feature ID 范围的布局。

每 rank 只 AllGather int64 候选键，再计算唯一第 k 大键。通信始终 detached。
本地通过 winner mask 从 **仍有 autograd 连接的本地 scores** 取出值。
因此 encoder 梯度不会断开，也不需要候选通信的 backward collective。

与 dense reference 的精确性是相对于上述 tie policy 而言；旧 `torch.topk` 不稳定
并列顺序不属于等价保证。无并列且浮点计算路径相同时选择一致；不同 GEMM 实现或
混合精度舍入引起的临界排名差异，需按数值容差和阈值间隔分析。

### 2.2 大 k / AuxK：有界 radix 阈值选择

只有满足以下条件时，auto 才允许候选 AllGather：

```
p * min(k,S) * 8 <= S * scores.element_size()
```

即每 token 的候选接收区字节数不得超过一份本地 score 行。
不满足时，`auto` 转为精确 distributed radix selection：逐段统计比较键的直方图，
对直方图 AllReduce，缩小包含第 k 大键的前缀。每轮最多 256 个 bins；所有 TP rank
采用同一轮数和 collective 顺序。低位 tie-break 也参与最终选择，因此不会在并列时
留下多于 k 个 eligible winner。

这是确定正确性的保守后端，不宣称 radix 多轮通信更快。`--sae-topk-protocol candidates`
若超出约束会直接报错，不允许偷偷 full gather。`radix` 可强制用于独立测试。

Torch 比较键生成使用按 token 分块，按临时张量估算将 scratch 控制在约 32 MiB；
这不是 allocator 实测峰值的硬上限。显式 Triton key kernel 融合逐元素转换，避免
多次分配中间位运算张量。局部 TopK 仍使用 `torch.topk(int64)`，不是一个已经调优
完成的自研 GPU TopK kernel。

## 3. 稀疏计算选择

### 默认：复用 PyTorch weighted embedding_bag

`sae_lens/sharded_sparse.py` 将本地 COO 展平为 row offsets、local feature IDs 和 values，
调用：

```python
F.embedding_bag(
    feature_ids, decoder_weight.T,
    offsets=offsets, per_sample_weights=values,
    mode="sum", include_last_offset=True, sparse=False,
)
```

`decoder_weight` 保持现有 Megatron `[d_in,S]` Parameter，不额外维护第二套权重。
不在 Python 中创建 `[B,k,d_in]` embedding 展开张量。
`Sparse=False` 指返回普通 **dense 参数梯度**，不是把 latent 变成 dense 再 GEMM。
PyTorch 的内部实现可能复制本地 weight layout，CUDA 梯度也不保证 bitwise 确定性。
这种成本必须计入 GPU 实测，不能根据少做乘法推导必然更快。

### 显式实验后端：Triton

`sae_lens/sharded_triton.py` 独立实现：

1. 稀疏激活×本地 decoder weight 的前向加权求和；
2. 只在本地 sparse entries 处计算 value gradient 的 sampled dot product；
3. 按 feature 分组的 `sparse.T × grad_output` 参数梯度。

接受现有 decoder strides；前向不会为了 kernel 先转置整张 weight 并 contiguous。
weight gradient 用连续的转置存储后返回普通 dense view，由 autograd/DDP 处理。
没有直接写 Megatron `main_grad`，不自己派发 DP bucket，也不修改 Adam 状态。
仅支持一阶 backward；无 CUDA/JIT 或性能通过的声明。

当前 COO 采用每 token `min(k,S)` 的固定候选槽位，未胜出位置保存零值。
这避免了动态 nnz 的主机同步/形状协调，但不是完全 packed 的平均 `k/p` 表示。
kernel 可跳过零值的部分访存，排序、value-gradient 等开销仍可能按候选槽位计算。

**本次没有实现 sparse encoder backward。** 原生 encoder 的 dense 局部 backward
符合内存约束；换成 sparse encoder backward 是另一项性能工作，而不是这份补丁
已经具备的能力。权重、参数梯度和 Adam 状态仍是 dense 本地分片。

## 4. 与现有训练的连接

`MegatronTopKSAE` 保留参数身份、分片轴、显式 runtime TP/DP 组和 checkpoint layout。
新 selector 不会使用隐式 WORLD；`group=None` 被定义为纯本地操作。
候选只在 TP 组内交换，DP 各副本独立选择各自 token 的 TopK。

AuxK 同样调用本地 exact selector，但使用本地 dead mask、**不做 ReLU**，保留负的
auxiliary activation、原有全局 k_aux/scale、detached residual target 和无 decoder
bias 的辅助重建。decoder norm 路径仍然可微，不通过 detach 丢弃其梯度。

新增 `TrainStepOutput.feature_firing_counts`：它是已完成 TP 汇总的 `[D]` detached
向量。GA、单/多 SAE trainer 和日志直接使用它；不会在只有 rank0 记录日志时
额外发起 collective。评估函数仅对本地 latent densify，L0/L1 归约每 token 的标量，
feature density 汇总一维向量。

wavefront 仍使用 `encode_launch → decode_launch → finish`：候选通信提交到现有共享
通信 stream，decode 等待事件、计算本地 partial，再提交 reconstruction AllReduce。
CPU 测试验证了分阶段方法的数学组合；CUDA stream/event 生命周期和多 communicator
并发仍待原生 GPU 测试。没有更改原有每 Hook 参数等待和 optimizer 更新编排。

稀疏 decoder 返回普通参数梯度，其 native linear gradient-accumulation fusion 被
关闭；encoder 的 fusion 仍可由原 runtime 开启。单独复制两个 linear 的 config，
避免共享 config 时关掉 decoder fusion 顺便改变 encoder。

TP>1 的 sharded 路径拒绝已有的全局 post-latent 修改 hooks，不会静默将全局语义
变成本地语义。新的 `encode()` 返回本地 features；调用者不能再假设其末维为 D。
需要全局分析的调用者必须另行设计局部接口或显式选用 legacy，不会自动聚合。

未验证：原生 distributed optimizer/ZeRO、FSDP、CUDA Graph、torch.compile、真实
streaming/elastic 切换、原生 checkpoint 续训与 early optimizer overlap。这些不是
CPU/Gloo 测试能证明的性质。本补丁未主动改动其数据流和参数格式，但仍需集成验收。

## 5. 开关与应用

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--sae-topk-backend` | `legacy` | `legacy` / `sharded_dense` / `sharded_sparse` |
| `--sae-topk-keys` | `torch` | Torch 比较键或实验性 `triton` 融合 key kernel |
| `--sae-topk-protocol` | `auto` | `auto` / `candidates` / `radix`，均精确 |
| `--sae-sparse-decoder` | `torch` | weighted embedding_bag 或实验性 `triton` |

旧路径保持默认；新路径不需要加 `--use-sparse-activations`，其 representation 由
`--sae-topk-backend` 明确决定。旧标志仍用于 legacy 语义。

在现有 `run_sae_runner_gpu.py` 命令末尾添加以下任意一组即可，不改变原 batch、ctx、
训练 token 数、TP、DP 或 hook 设置。

**先独立验证通信与本地 latent：**

```bash
--sae-topk-backend sharded_dense --sae-topk-protocol auto
```

**启用有 CPU 数值验证的本地 sparse decoder：**

```bash
--sae-topk-backend sharded_sparse \
--sae-topk-keys torch --sae-topk-protocol auto --sae-sparse-decoder torch
```

**实验性融合 key + Triton decoder，必须先过 GPU 验收：**

```bash
--sae-topk-backend sharded_sparse \
--sae-topk-keys triton --sae-topk-protocol auto --sae-sparse-decoder triton
```

**原行为对照：**

```bash
--sae-topk-backend legacy
```

应用前在目标仓库执行，PATCH 设置为实际保存路径：

```bash
PATCH=/absolute/path/saelens_sharded_topk_base_20260918.patch
git status --short
git apply --check "$PATCH"
git apply "$PATCH"
```

只有检查通过才执行第二个 apply。若 main 已发生冲突，不要使用 `--reject`、丢弃
用户改动或直接用旧源码覆盖。补丁能 apply 也不等于当前版本的 GPU runtime 已被验证。
交付包中的 `patch_files/tools/check_sharded_topk_base.py --repo /path/to/SAELens`
可以在应用前核对归档基线 SHA256；脚本不修改目标仓库。

## 6. 复现测试

依赖轻量 CPU 测试仅需要 torch 和 pytest，不执行 SAELens 的 LLM 初始化。
它们将真实被修改模型方法加载到一个 F.linear/Gloo 依赖替身中；其中 Torch DDP 和
Adam 是实际实现，不是 native Megatron Core 的 CUDA 实现。请单独运行该目录，
不要与需要真实 SAELens 初始化的其他测试混在同一 pytest session：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 GLOO_SOCKET_IFNAME=lo \
PYTHONPATH=tests/sharded_topk_standalone:$PYTHONPATH \
python -m pytest -q --confcutdir=tests/sharded_topk_standalone \
  tests/sharded_topk_standalone
```

测试覆盖 FP32/FP16/BF16 选择、ties、相邻浮点数、负 AuxK、集中在一个 rank 的
winner、零 winner/空 batch、norm、前反向和普通参数梯度、TP2/TP4、TP2×DP2、
TP1×DP3 不均匀 tokens、GA=2、两步 Adam、非 SAE spectator rank。关键前反向测试
使用 TorchDispatchMode 检查新路径算子返回值，不允许出现全局 `[B,D]`。
这个 guard 不是 CUDA 驱动内分配的完整追踪，不能替代 GPU memory snapshot/nsys。

### 原生 GPU 验收脚本（已提供，未在本环境执行）

在已经安装 SAELens/Megatron 的 GPU 环境运行；不需要下载 LLM/数据集：

```bash
torchrun --standalone --nproc_per_node=4 tools/validate_sharded_topk_gpu.py \
  --tp 2 --dp 2 --global-batch 65 --hooks 2 --ga 2 --wavefront --aux \
  --output sharded_topk_validation/native_torch
```

该脚本默认顺序比较 legacy、sharded_dense、sharded_sparse；检查 loss、最终参数、
梯度和 Adam 状态，记录每 rank 合成训练窗口时间和峰值 allocated bytes。
计时包含合成输入生成、forward/backward、窗口归一化、clip 和 Adam，不是在线
LLM→SAE end-to-end benchmark。取各 rank 最大值评估步时/显存，不比较不同 global batch。
内存形状 guard 只在第一次 warmup 开启，不计入测量窗口。

再验证实验后端：

```bash
torchrun --standalone --nproc_per_node=4 tools/validate_sharded_topk_gpu.py \
  --tp 2 --dp 2 --global-batch 65 --hooks 2 --ga 2 --wavefront --aux \
  --keys triton --decoder triton \
  --output sharded_topk_validation/native_triton
```

脚本使用真实 Megatron linears/DDP/runtime 和 dense Torch Adam，但不验证 native
ZeRO/distributed optimizer，也不验证提前 bucket/optimizer overlap。需要另在原 runner
上验收这些功能。`--autocast`、`--no-norm`、`--no-fusion` 支持分项验证。

真实规模只测性能/显存且不保留 CPU 权重参考副本可加 `--benchmark-only`；先通过
小规模数值验证，再用原定 global batch/d_in/d_sae/k/hook 数测试。初次 GPU 验收不得
仅因为吞吐提升就跳过梯度与两步 optimizer 一致性检查。

## 7. 理论数据大小，不是性能测量

FP32、单个 DP 副本 B=4096、D=81920、k=128：

| 单份数据 | TP2 | TP4 |
|---|---:|---:|
| 旧完整 `[B,D]` score | 1280 MiB | 1280 MiB |
| 新本地 `[B,S]` score | 640 MiB | 320 MiB |
| candidate receive 的 int64 key | 8 MiB | 16 MiB |

这里不包含多份 buffer、COO 坐标、局部 TopK 工作区、参数、梯度和 optimizer 状态；
不能将这些比值称为整体峰值降低倍数或训练加速倍数。

新路径可能慢的因素包括 int64 exact selection、Torch 分块 launch、COO coalesce、
原生 decoder stride 导致的局部权重 copy、较大的 AuxK、radix 多轮 collective，以及
GPU 上 dense GEMM 的优势。当前没有 H100/5090 实测，不能承诺最快。
显存严格受限而 sparse decoder 较慢时，选择 `sharded_dense`；只有明确允许恢复旧
latent 内存行为时才选择 `legacy`，不存在自动破坏内存约束的回退。

## 8. 参考与复用边界

- OpenAI `sparse_autoencoder/kernels.py` 的三类稀疏 forward/backward 运算结构：
  https://github.com/openai/sparse_autoencoder/blob/main/sparse_autoencoder/kernels.py
- OpenAI 分片 SAE 训练/TopK 的组织：
  https://github.com/openai/sparse_autoencoder/blob/main/sparse_autoencoder/train.py
- 直接调用的 PyTorch `embedding_bag` 官方 API：
  https://docs.pytorch.org/docs/2.10/generated/torch.nn.functional.embedding_bag.html
- Megatron DDP 的 `main_grad` / `grad_added_to_main_grad` 普通参数梯度契约：
  https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/distributed/distributed_data_parallel.py

本补丁没有复制整个第三方训练框架，也没有引入 Meta/FlexSAE 的非商业 kernel。
默认后端直接复用 PyTorch 运算；Triton 文件为按现有参数布局独立实现的实验后端。
不会把“算法上采用现有结构”宣传为“现成高性能 kernel 已在当前硬件验证”。
