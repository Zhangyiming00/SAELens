# 不等分 TP 与相邻动态 TP：实验性实现

基线：2026-10-07 fetch 并快进至 `origin/main@061f9e1e473773010db98b54f22866da1ddec2b1`。在 `dynamic_tp_v1_061f9e1_20261007.zip` 的参考实现上应用并补齐真实通信、激活缓存和暂停时间验证。

## 结论与交付边界

方案可行。SAE 的 feature 维可以置换，只要 encoder 行、encoder bias、decoder 列、Adam moments、dead/firing 状态使用相同的稳定 feature ID。现有 Megatron 布局中，`encoder.weight=[local_features,d_in]`、`decoder.weight=[d_in,local_features]`。它不同于原生 SAELens checkpoint 的矩阵朝向。

**补充验收结论：大尺寸迁移本身已通过逐字节 SHA-256 对照；但 65536 维真实激活的训练严格数值等价测试未通过。不能将“小尺寸 oracle 通过”或“迁移无损”解释为切换前后训练轨迹逐位一致，详见后文的耐久与数值审计。**

实现内容：

- 原 `MegatronTopKSAE` 自动按真实宽度分片：`4096/TP3 -> 1366,1365,1365`，没有 padding features。
- 不等宽且可非连续的 feature ownership；TopK/AuxK 按稳定 ID 处理分数相同的情况。
- 参数、Adam/AMSGrad 状态、step、学习率等 optimizer group options、dead/firing counters 的直接内存迁移。
- 两套训练通信域轮换：切换后旧组作为备用组，反向切换可复用；其他目标在下一次 prepare 时回收重建。另有固定控制组和迁移组。
- 固定 worker pool 内任意一个 rank 加入/退出，TP 在 `1..min(pool_size,d_sae)` 内做相邻转换，包括非末尾 rank 退出、多次重排和 TP1。
- 单个 DP 副本内多个 hook 的动态 TP；不同 hook 可以使用不同 d_sae，API 逐 hook 建计划。
- `DynamicTPSession` 训练 API、读取真实 safetensors 激活的 `run_dynamic_tp_sae.py`，以及真实 Gloo/NCCL 验收脚本。

边界必须保留：动态 session 为 **DP=1、FP32 SAE、每 hook 一组 torch Adam**。CUDA 入口使用 fused Adam。它不是现有 `elastic_streaming` DP/ZeRO 角色切换的开关；没有把该控制器改造成同时调 DP/TP 的实现，也没有实现世界进程数量变化、FSDP/ZeRO 状态重分片、AMP scaler 迁移、部分梯度累计窗口迁移。现有静态 runner 获得不等分 TP；在线动态 TP 已接入 `run_sae_runner_gpu.py --elastic-tp`，支持 vLLM close/reload 与保留权重两种交接模式。

动态 session 使用原生 Megatron encoder/decoder 与现有 sharded TopK/AuxK。`tp_overlap=off/eager/lazy/bounded` 复用原有 `forward_tp_wavefront` / `PendingWavefrontOutputs` 调度；bounded 的窗口由 `tp_overlap_max_live_hooks` 指定，每个完整 step 后才能切换。它尚未接入原 MultiSAETrainer 的 DDP optimizer-overlap、完整日志/调度器体系，不支持动态 DP/ZeRO、AMP 或部分 GA 窗口。不能把这个独立入口的耗时直接当成旧 runner 的性能对照。已有训练数据应按原有方式先做 activation scaling，入口要求 `normalize_activations=none`。

## 2026-10-08 H100 v29 数值与卸载修复

`v29-sparse-colocated-20261008-external.tar.gz` 和 `v29-sparse-elastic-tp-20261008-external.tar.gz` 的 `sae_lens` 源码与当时远端 `origin/main@1e02754` 一致。该版本存在以下两个已在当前工作区修复的问题：

- **非二次幂输入宽度的 OpenAI 稀疏前向漏掉 TP 求和。** 独立计算策略路径把 `d_in=5120` 补到 8192 后，切回原宽度时留下非连续输出。Megatron 的 reduce mapping 在 `input.contiguous()` 副本上求和，却返回原输入，各 rank 因而看到自己的局部重建。修复在稀疏前向出口恢复连续布局，覆盖 bucketed 和 COO 两种适配。`d_in=4096` 的此前测试没有触发此问题；TP1 也不涉及这个跨 rank 求和错误。该问题影响 Main，不能通过调整 AuxK、学习率或日志缩放解决。
- **vLLM 采样器的全局 CUDA 缓存未随模型卸载。** 只清理 engine/model runner 无法释放 `_TRITON_BUFFER_CACHE` 和 `_TRITON_TABLE_CACHE`。`HookedVLLMModel.close()` 现在调用 vLLM 自带 `reset_buffer_cache()`。没有提高 `release_tolerance_mib`；冷加载会按需重建缓存。

当前集群为 RTX 5090，未运行 H100 原 Qwen3.6-27B 模型。原维度 SAE 的 TP4、GA2 固定状态对照覆盖 sharded_dense/sharded_ragged、eager/bounded、前向、梯度、参数、Adam 和 firing ages；在线本地 Llama-3.1-8B 则使用同样的 `context=2048`、4 prompts 和 `max-num-batched-tokens=8192`，完成 TP2→TP3→TP2→TP3、GA2 以及 13 个微批的内容/消费审计。冷加载前后激活逐位相同，残留活跃内存为 8.125 MiB，低于原 64 MiB 限制。详细证据位于 `results/h100_v29_20261008/`。

固定状态算子/更新一致不等于不同 FP32 算法的长期训练轨迹逐位相同。大维度独立轨迹在第五次更新出现一对近边界 TopK winner 交换，报告保留了该严格对照失败；逐更新对齐参数及 Adam 状态的对照通过。H100 原共置任务停在第 893 步且未保存退出原因，现有包仍不足以解释其终止原因。受错误 TP 重建影响的训练结果应重新运行，不能作为等价训练的性能结论。

## 主入口：在线 elastic TP

在线 producer、异步 SHM 写入、水位控制、训练循环、迁移验证和进程管理均由 `sae_lens` 包提供。主入口为 `run_sae_runner_gpu.py`，运行和子进程启动均不依赖 `scripts/`、Git 元数据或仓库相对路径。用普通 Python 启动一次，入口自行创建固定的 worker pool；不要在外面再套 `torchrun`。

### 准备本地输入

- 模型目录需已准备好权重及 `config.json`。输入维度从模型配置自动读取，每个 vLLM producer 使用 TP1，因此完整模型必须能放入单张卡。
- 数据目录需是 `datasets.save_to_disk` 保存的单个 Dataset，包含非空、等长的二维 `tokens` 列。数据应事先使用该模型对应的 tokenizer 处理。每行 token 数必须是 `context-size` 的整数倍；运行时切为 context 窗口并循环读取至训练结束，不做下载或在线分词。
- `train-batch-size-tokens` 是微批 token 数，必须是 `context-size` 的整数倍；`training-tokens` 必须是微批的整数倍。微批数为两者相除，optimizer 更新数为 `ceil(微批数 / GA)`。`max-model-len` 必须大于 `context-size`。
- `/dev/shm` 要有足够空间。激活主体占用约为 `streaming-num-chunks × train-batch-size-tokens × d_in × dtype字节数`，另有少量元数据。例如 BF16、96 块、4096 tokens、d_in=4096 约需 3 GiB。
- 当前在线模式支持单机、单 hook、SAE FP32、DP=PP=1、任意正整数 GA。模型与数据之外，Python 环境需已安装兼容的 PyTorch、vLLM、Transformers、Datasets 等依赖。

入口在导入 Transformers 前默认设置 `HF_HUB_OFFLINE=1`、`HF_DATASETS_OFFLINE=1`、`HF_HUB_DISABLE_TELEMETRY=1`、`VLLM_NO_USAGE_STATS=1`，所有子进程继承这些值，用户无需手写环境变量。加载前会拒绝远程模型标识和不存在的本地模型/数据目录。使用这些库读取本地文件不需要 Hugging Face 服务、账号或联网。

### 日常启动：选卡、卡池和初始 TP

以下 Bash 示例先定义一次公共配置；模型、数据、hook 和训练规模按本地任务修改。学习率等训练算法配置保持默认，日常系统使用不需要额外设置。

```bash
COMMON=(
  --model-name /root/models/Llama-3.1-8B
  --dataset-path /root/datasets/wikitext2_tokenized_llama31_ctx2048
  --hook-name blocks.21.hook_resid_post
  --d-sae 65536 --k 128 --dead-feature-window 1500
  --training-tokens 16384000 --train-batch-size-tokens 4096
  --context-size 1024 --max-model-len 1025
  --store-batch-size-prompts 1
  --streaming-num-chunks 96 --streaming-prefetch-chunks 2
  --tp-overlap bounded
)

# 两卡：默认 SAE TP1 + 一个活跃 producer，自动在 TP1/TP2 间调整。
CUDA_VISIBLE_DEVICES=3,1 python run_sae_runner_gpu.py \
  "${COMMON[@]}" --elastic-tp-size 2 --output-path results/elastic_2gpu

# 三卡：默认 SAE TP1 + 两个活跃 producer，自动在 TP1..TP3 间调整。
CUDA_VISIBLE_DEVICES=3,0,2 python run_sae_runner_gpu.py \
  "${COMMON[@]}" --elastic-tp-size 3 --output-path results/elastic_3gpu

# 四卡：显式从 SAE TP2 + 两个活跃 producer 开始。
CUDA_VISIBLE_DEVICES=3,1,0,2 python run_sae_runner_gpu.py \
  "${COMMON[@]}" --elastic-tp-size 4 --sae-tp-size 2 \
  --output-path results/elastic_4gpu
```

每条启动命令是一次独立运行，输出目录必须未用于先前运行。`--elastic-tp-size N` 同时启用 elastic TP 并设置卡池大小，不必额外传 `--elastic-tp`。单独 `--elastic-tp` 默认使用四卡，`--elastic-tp-pool-size` 是卡池参数的别名。

GPU 顺序由 `CUDA_VISIBLE_DEVICES` 决定，池内 rank 对应其前 N 张卡。上面的两卡示例中，rank 0 是物理卡 3、rank 1 是物理卡 1；rank 0 始终训练，rank 1 可在 SAE 与活跃生产之间切换。不设置可见卡时使用前 N 张可见卡；程序不会自动寻找空闲 GPU，启动前应选择空闲且显存足够的卡。

`--sae-tp-size N`（或 `-stp N`）只设置初始 SAE TP，**不会限制后续缩容**。省略时从最低允许 TP 启动，默认下限为 1。高低水位自动决定相邻切换，无需传入“从什么切换到什么”的列表。

### 梯度累积：GA > 1

沿用普通 runner 的 `--gradient-accumulation-steps K`，默认 1。`--train-batch-size-tokens B` 始终是 provider 微批大小，通常每次 optimizer 更新消费 `B × K` tokens，**不会自动把 B 除以 K**。例如保留原先有效 batch=4096、降低单次前后向的激活显存：

```bash
CUDA_VISIBLE_DEVICES=0,1 python run_sae_runner_gpu.py \
  "${COMMON[@]}" --elastic-tp-size 2 \
  --train-batch-size-tokens 1024 --gradient-accumulation-steps 4 \
  --output-path results/elastic_ga4
```

若保留 `B=4096` 并设置 `GA=4`，有效 batch 将变为 16384。`training-tokens` 是总输入预算，不会乘以 GA。例如 9 个微批、GA4，执行 4+4+1 三个 optimizer 更新；末尾窗口按实际 token 数归一化，不丢弃、不补齐样本。

- 各微批前后向结束即释放计算图，只保留累积梯度和小型统计；缓存可以在窗口中途补充，`streaming-prefetch-chunks` 不要求大于等于 GA。
- 整个窗口固定参数和 dead mask；梯度、loss components 按微批 token 数加权，窗口末尾才裁剪梯度、执行一次 Adam、更新 firing/dead counters。`dead-feature-window` 与 step 日志按 optimizer 更新计数。
- TP prepare、迁移和 checkpoint 只允许完整窗口之间；不迁移部分梯度。GA>1 的已准备切换在 producer ACK 后的 optimizer 边界提交，不额外强制运行一个旧 TP 窗口。
- 暂停最后一个 producer 前，必须有足够的 READY+本地缓存微批覆盖下一整个窗口。全卡 SAE 若储备不足，会在窗口开始前恢复 producer（`reason=ga_input_reserve`），可越过 cooldown；不在累积中途缩容。GA 大于缓存总容量时保留 producer，并边训练边补充。
- 全卡启动预填充尽量覆盖一个 GA 窗口，仍受 SHM 容量限制；容量不足时允许在第一个 optimizer 更新之前缩容。Adam 尚未初始化的迁移诊断也支持此路径。

`run_config.json` 中 `steps` 保留为微批/生产 chunk 数，`gradient_accumulation_steps` 单独保存。`train_rank*.jsonl` 的 `step` 和报告的 `steps` 是 optimizer 更新数；`microbatches`、`consumed_microbatches`、`tokens` 用于区分窗口大小与累计输入。`step_s/input_s` 是窗口内各微批对应阶段的时间之和，完整墙钟时间还包含供给等待和控制开销。GA1 行为保持原有语义。

离线 `run_dynamic_tp_sae.py` 同样接受 `--gradient-accumulation-steps`，其 `--steps` 和 TP schedule 仍表示 optimizer 更新；激活文件需至少 `steps × GA × batch-size` 行，checkpoint 保存并恢复 GA，旧 checkpoint 默认 GA1。在线 checkpoint 的原有限制不变。

Python API：`train_step()` / `train_cached_step()` 保持一次输入完成一次更新；配置 `DynamicTPSession(..., gradient_accumulation_steps=K)` 后，用 `train_microbatch()` / `train_cached_microbatch()` 累积。非末尾调用返回 `None`；末尾返回窗口平均 loss。短尾通过每次调用相同的 `window_microbatches=实际微批数` 指定。

CPU 数学与调度回归：

```bash
python -m pytest --confcutdir=tests/sharded_topk_standalone \
  tests/sharded_topk_standalone/test_dynamic_tp_ga.py -q
# 支持本机 TCP 通信的环境，可额外运行三进程真实 Gloo 测试：
SAE_TEST_GLOO=1 python -m pytest --confcutdir=tests/sharded_topk_standalone \
  tests/sharded_topk_standalone/test_dynamic_tp_ga.py -q
```

CPU 测试使用真实 SAE/TopK/AuxK 方法、PyTorch Adam 与 F.linear 替代 Megatron 线性层；线程通信测试覆盖 1→2→3→2→1、非末尾 rank 退出、Adam 状态与 checkpoint。它们不替代原生 Megatron、fused Adam、NCCL、vLLM 的 GPU 端到端验收。

### 显存放不下 TP1：限制最低 SAE TP

如果已知 SAE 至少需要 TP2，设置下限；四卡的例子如下：

```bash
CUDA_VISIBLE_DEVICES=3,1,0,2 python run_sae_runner_gpu.py \
  "${COMMON[@]}" --elastic-tp-size 4 --elastic-tp-min-size 2 \
  --output-path results/elastic_4gpu_min2
```

此时默认从 TP2 开始，只允许 TP2、TP3、TP4。也可另传 `--sae-tp-size 3` 或 `4` 指定更大的初始 TP。在 TP2 且缓存不足时，训练等待 producer 补充数据，不会继续缩到 TP1。这个参数是显式边界，不会自动探测最低显存需求或在 OOM 后自动回滚重试；如果 TP2 本身仍然放不下，运行仍会失败。

参数须满足 `pool_size >= 2`、`1 <= min_tp <= initial_tp <= pool_size <= d_sae`，并且 **`min_tp < pool_size`**。初始值低于下限会在启动模型前报错。下限等于卡池大小也会报错，因为在线模式需要保留恢复至少一个 producer 的可能性：例如两卡且 SAE 至少需要 TP2，就不能使用当前在线角色切换模式持续补充输入，需要增加卡数或另用预存激活训练入口。

默认 `release` 模式先释放 vLLM 模型与 KV cache，再让对应卡加入 SAE；`resident` 模式暂停时仍保留权重。最低 TP 应根据 SAE 分片及迁移临时存储的峰值设置；resident 还需容纳 vLLM。卡池大小固定，控制器不会临时调用池外 GPU。

### vLLM 显存交接：release 与 resident

2026-10-08 的交接补丁基于 `origin/main@7b47ebf`，同时保留 GA 支持。默认使用：

```bash
--elastic-tp-vllm-residency release
```

扩容 `m→m+1`：控制器发布新 epoch 和目标 TP，加入卡上的 producer 停止领取新 chunk；已经领取的请求完成推理、D2H 和 SHM 发布后，调用现有 `HookedVLLMModel.close()`。随后同步、清理引用、运行 GC 和 `empty_cache()`，检查存活 PyTorch 分配量，再发布该 epoch 的 `released / ready=true / memory_released=true`。SAE 只在收到有效确认并抵达完整 optimizer 更新边界后迁移参数、Adam、dead/firing 状态及激活缓存，提交备用训练通信域。旧 SAE ranks 可在 producer 排空期间继续训练；GA>1 暂停最后一个 producer 时保留完整窗口输入。

缩容 `m→m−1`：先在更新边界迁走退出 rank 的状态并提交新 TP，再在退出 rank 上释放 SAE 张量引用、同步、GC 和 `empty_cache()`，写出 `saeN_status.json` 的同 epoch `released` 确认。所有 worker 确认交接成功后，控制器才发布较小 TP。producer 核对 SAE release ACK，重新加载模型，CUDA 同步完成后发布 `running / ready=true`，随后继续领取输入。冷加载期间剩余 SAE ranks 可以消耗缓存；控制器等到 producer ready 后才允许下一次拓扑切换，超时或失败会广播给全部 SAE workers 并中止本次运行。

显存充足时可显式保留原来的暖驻留路径：

```bash
--elastic-tp-vllm-residency resident
```

resident 扩容仅排空并确认 `paused`，不销毁模型；缩容仍先清理 SAE allocator，再恢复推理，无需重新加载权重。不会因为 release 失败自动切回 resident。

交接约束：

- `done` 只在输出排空且模型关闭、缓存归还后发布。输入预算耗尽后 producer 保留 CPU 控制循环，为后续 epoch 重新确认 `done`，直到 stop；不会重新加载。旧 epoch 的 `done/released` 不能放行新交接。
- 初始 SAE ranks 在 release 模式下不会先加载 vLLM；全卡启动预填充结束也必须先收到全部当前 epoch 的释放确认，才启动 SAE worker。startup 特例只用于尚未启动 SAE 的阶段，训练开始后关闭。
- CUDA context、固定控制/迁移通信域和至多两套训练通信域仍驻留。`memory_released` 表示模型/训练张量及可归还 allocator 缓存的交接，不代表 NVML 为 0。默认允许比建模前 PyTorch allocated 基线多 64 MiB；`--elastic-tp-release-tolerance-mib` 可调。超过阈值即失败，日志保留分配量，不能用空闲状态替代释放确认。此检查不能验证第三方 allocator 的所有分配。
- release 复用当前 fork 的 `close()`，恢复采用冷加载。没有实现或声称验证 vLLM level-1 sleep/wake，也不使用 checkpoint/resume 搬运 SAE 状态。冷加载可能是显著成本，缓存不足时会反映为训练等待。

计时应同时检查 `switches.json` 的 `prepare_s`、`pause_s_max`、`handoff_s`、`request_to_commit_s`，以及 `producerN.jsonl` 的 `drain_s/close_s/quiesce_s/load_s` 和 `train_rank0.jsonl` 的 `producer_resumed.resume_s`。其中 `pause_s_max` 仅为核心迁移暂停，`handoff_s` 含诊断与退出 SAE 的释放，`request_to_commit_s` 还包括等待 producer 释放和更新边界；`resume_s` 从缩容释放完成后的恢复请求开始，不属于核心迁移时间。不要只用 `pause_s_max` 宣称释放模式没有切换损失。

CPU 回归命令：

```bash
python -m pytest --confcutdir=tests/sharded_topk_standalone \
  tests/sharded_topk_standalone/test_dynamic_tp_ga.py \
  tests/sharded_topk_standalone/test_elastic_tp_handoff.py -q
```

测试覆盖 epoch、防提前 ACK、关闭/重新加载失败、残留分配、加载期间 stop、全卡启动预填充、三 rank 的 `1→2→3→2→1`、GA1/3、不等宽分片、Adam 状态和恢复超时。CUDA/vLLM/SHM 在协议测试中被模拟；CPU 数学/线程通信测试不替代 H100/5090 原生验收。GPU 验收应分别跑两种模式、普通与全卡启动，并核对同卡的 producer released → SAE 加入、SAE released → producer running 顺序、allocated/reserved 与 NVML、唯一输入数和全部退出码。本文下方的历史 GPU 记录属于原驻留实现，不能当作本次 release 模式的性能证据。

#### 本次补丁的 GPU 验收（2026-10-08）

在 RTX 5090 32 GiB 上，使用本地 Llama-3.1-8B 和已分词的 WikiText-2，通过 `run_sae_runner_gpu.py` 完成以下四项真实运行。每个微批 256 tokens，`d_sae=1025`，开启输入哈希核对和 bounded overlap；包含非整除分片、AuxK 和保存模型。下面的 TP 序列均由控制器自动触发。

| 模式 | 卡数 / 初始 SAE TP | GA | 微批数 → optimizer 更新数 | 实际切换 |
|---|---|---|---|---|
| release | 2 / 1 | 3 | 49 → 17，末尾 1 个微批 | TP1 → TP2 → TP1 |
| release | 3 / 3，初始零活跃 vLLM | 4 | 17 → 5，末尾 1 个微批 | TP3 → TP2 → TP3 |
| resident | 2 / 1 | 1 | 17 → 17 | TP1 ↔ TP2，累计 14 次切换 |
| resident | 2 / 2，初始零活跃 vLLM | 3 | 17 → 6，末尾 2 个微批 | TP2 → TP1 |

三卡 release 用例特意只留 2 个 SHM chunks、本地缓存 1 个微批，均小于 GA4：系统在第一个 optimizer 更新之前以 `ga_input_reserve` 缩容，释放退出 rank 的 SAE 状态后恢复 producer，并在窗口中补充输入；剩余最后一个微批时重新进入 TP3。所有运行的生产和消费序号恰好覆盖输入预算，无重复或遗漏；切换全部发生在完整更新边界，交接 epoch 和释放顺序通过日志审计，所有子进程退出码为 0，SHM 正常清理。

释放效果使用 producer 进程级 NVML 采样交叉验证：加载后约 16194 MiB，release 确认后稳定为 728 MiB；同期 PyTorch allocated 为 39.44 MiB、reserved 为 62 MiB。全卡启动的 resident 用例在暂停期间仍约 16194 MiB。这验证了真实显存归还，但不会清空 CUDA context 等常驻开销。两次 release 恢复耗时分别约 2.73 和 2.76 秒；resident 恢复约 22–38 ms。这里使用小 SAE，数据用于功能与交接验收，不是大 SAE 的容量或吞吐结论，也不是 H100 测量。

额外的三卡原生 NCCL/fused Adam 测试位于 `tests/training/test_elastic_tp_ga_native.py`：两个 hook、不等宽分片、BF16 输入，比较 GA3 的不等长微批与拼接后的完整 batch，覆盖 TP1/2/3 以及非连续 active ranks、短尾窗口、off/bounded 调度。损失、参数和 Adam 状态在容差内一致，进度及 dead/firing 统计完全一致。

```bash
# 真实三进程 Gloo 的 GA 迁移与 checkpoint 对照
SAE_TEST_GLOO=1 python -m pytest --confcutdir=tests/sharded_topk_standalone \
  tests/sharded_topk_standalone/test_dynamic_tp_ga.py::test_uneven_tp_migration_and_checkpoint -q

# 需要三张可见 GPU；使用项目兼容的本地依赖环境
python -m pytest --confcutdir=tests/training \
  tests/training/test_elastic_tp_ga_native.py -q
```

本次共通过 159 项测试（补丁 CPU 60、真实 Gloo 1、原生 GPU 2、runner/精度/显存回归 96），另有上述 4 项真实 vLLM 端到端运行。本机完整命令、NVML 采样、worker 日志和审计结果保存在 `results/elastic_handoff_20261008/`，汇总为 `summary.json`；`run_e2e.py --case <用例名> --tag <新目录名>` 可按记录的本地路径复现，`audit.py` 检查原始四组结果。

#### 大 SAE 显存、迁移时间与恢复专项验收（2026-10-08）

进一步使用三张 RTX 5090、本地 Llama-3.1-8B、`d_in=4096`、`d_sae=262145`、微批 128 tokens、Main K32、AuxK64、最低 SAE TP2。测试辅助 worker 记录显存和事件时间、保存并核对激活，调度与迁移仍调用原生产代码；这部分插桩只存在于测试目录，不改变正常启动流程。NVML 每 50 ms 采样，另记录 PyTorch 的分配峰值；短于采样间隔的驱动峰值可能没有被 NVML 捕获。

| 用例 | 结果 |
|---|---|
| release，TP2 启动，GA3，49 微批 | 17 次更新，末尾 1 微批；6 次 TP2↔TP3 切换全部通过 |
| release，TP3/零活跃 vLLM 启动，GA4，37 微批 | 10 次更新，末尾 1 微批；缓存仅 2 chunks，第一个更新前 TP3→TP2，通过 |
| resident，其余配置同第一行 | OOM：常驻 vLLM 约 15.81 GiB，SAE 申请额外 1.33 GiB 时只剩约 1.1–1.2 GiB；未完成训练 |

**显存归还。** 第一组的 producer 进程从约 16192 MiB 降到 728 MiB；四次真实关闭后的平台值相同，未观察到逐轮增长。释放后 PyTorch allocated 为 39.44 MiB、reserved 为 62 MiB。退出 SAE rank 在三次缩容后的 allocated 均为 16.25 MiB、reserved 均为 40 MiB。SAE 迁移峰值为 20.07 GiB，训练峰值为 24.16 GiB；两者是 PyTorch 分配量，不能代替驱动总占用。保留的训练 ranks 仍会缓存 allocator 块，NVML 不会随每次扩容立即降到新分片的理论大小。

**卸载位置与顺序。** `elastic_tp_producer.ProducerModel.quiesce()` 在对应 producer 进程的本地 GPU 上先 drain 异步输出，再调用 `HookedVLLMModel.close()`，最后发布释放 ACK。权重 storage 被销毁，CPU 只留下空占位张量，不把完整权重 offload 到 CPU；该次关闭前后 RSS 变化不超过 0.12 MiB。三次加入事件中，producer 的释放 ACK 均早于任一 SAE rank 开始迁移，间隔约 12–37 ms。缩容由 `elastic_tp_trainer` 先完成迁移，再调用 `release_inactive_sae()`；三次 producer 重新加载均晚于同卡 SAE release ACK，间隔约 20–27 ms。全卡启动用例中，两张 producer 卡都在 SAE 初始化前约 5.85 秒完成释放。恢复读取已有本地模型；缓存中的激活和输入游标继续使用。

**迁移时间。** 第一组中，初始化 Adam 后的五次迁移测量如下；第 0 步尚无 Adam moments，核心迁移为 217 ms，未混入表中。正式测量包含诊断开销，不是关闭诊断后的吞吐基准。

| 阶段 | 范围 | 中位数 |
|---|---:|---:|
| 迁移前检查 | 41.5–47.7 ms | 44.8 ms |
| 目标状态分配 | 29.8–82.1 ms | 59.7 ms |
| 参数/Adam 等传输，约 8 GiB | 245.9–261.4 ms | 252.2 ms |
| 提交通信域与状态 | 1.9–2.8 ms | 2.3 ms |
| 核心迁移 `pause_s_max` | 321–393 ms | 360 ms |
| 含诊断和 SAE 释放的 `handoff_s` | 381–833 ms | 739 ms |
| 从请求到提交 | 479–863 ms | 766 ms |
| 缩容后 vLLM 冷恢复，单独计时 | 3.43–3.56 s | 3.55 s |

**恢复和数据完整性。** 两组成功运行共 86 个微批、11008 tokens。验证生产、发布、加载的 chunk 序号恰好覆盖预算，原始 token 窗口索引互不重复；反转 shuffle 后，SHM 数据与 producer 捕获数据逐字节相同；每次实际训练的微批哈希与缓存消费顺序相同，且每个 active rank 一致。另外从重载前、三次重载后以及末尾选取五个块，用另一个新加载且未切换的 vLLM 重算，激活逐位一致。

新增 `tests/training/test_elastic_tp_handoff_native.py` 使用始终 TP1 的独立参考，以两个 hook（`d_in=64`，`d_sae=257/263`）检查完整状态，覆盖 BF16 输入、不等长 GA3 微批、短尾窗口、不等宽分片、非连续 ranks 和 off/bounded 两种调度，共比较 18 次更新。迁移前后检查全部参数、Adam 一阶/二阶矩与 step、dead/firing 状态、进度和剩余缓存；TP3 保存后在 TP2 加载 checkpoint 也逐位一致。与固定 TP1 参考的训练结果最大绝对误差为 `4.66e-8`。大 SAE 运行仍使用生产路径中的参数/Adam 坐标采样检查，不能把小规模全状态对照说成大 SAE 全量参数对照。

**异常退出修复。** 压力测试曾发现 resident OOM 后，`elastic_tp_trainer.train()` 的 `finally` 无条件调用 `TPGroupPair.close()`；OOM rank 在控制 barrier 等待，另一 rank 仍处于 CUDA collective，导致额外等待约 180 秒，整次任务约 293 秒才结束。

现已将失败清理与正常清理分开：训练初始化或训练期间发生异常时，先在 CPU 上写出 `train_rankN_error.json`，保存 rank、PID、时间、原始异常类型/信息和 traceback；失败路径不等待输入 loader 的 CUDA event，也不再调用 CUDA synchronize、barrier 或 communicator destruction。supervisor 在轮询中检测这些错误文件，保留最早的原始错误，立即终止整个训练进程组，然后停止 producer 并销毁 SHM。即使 worker 的解释器/CUDA 析构尚未退出，也能通过错误文件启动清理。

使用完全相同的 resident 大 SAE 配置复测，仍如预期触发 OOM，但从错误文件发布到最终失败结果仅 **2.12 秒**，完整任务 **41.62 秒**；日志没有 180 秒 collective/barrier 超时，原始 `OutOfMemoryError` 出现在最终 `result.json`。全部进程退出、SHM 删除，四卡显存均回到约 1 MiB。另有正常 release、GA4、三卡全 SAE 启动及 TP 迁移回归通过。CPU 协议/GA/失败测试 69 项、runner 回归 63 项通过；其中新增 9 项覆盖初始化/训练失败、取消、错误报告写入失败以及训练进程未退出时的及时终止。证据汇总为 `results/elastic_tp_failure_20261008/summary.json`。修复解决的是失败退出等待，不会让原本显存不足的 resident 配置自动恢复训练。

本机证据目录为 `results/elastic_handoff_deep_20261008/`：`summary.json` 汇总通过和失败用例，`memory_and_handoff.png/.svg` 展示显存与阶段耗时，各用例的 `deep_audit.json` 保存逐次检查结果。复现方式：

```bash
# 本机脚本中的模型/数据均为已有本地目录；--tag 必须使用新目录名
python results/elastic_handoff_deep_20261008/run.py --case pressure_release --tag repeat_release
python results/elastic_handoff_deep_20261008/run.py --case pressure_full_start --tag repeat_full
# resident 用例用于复现上述 OOM 与异常退出问题，预期失败
python results/elastic_handoff_deep_20261008/run.py --case pressure_resident --tag repeat_resident

# 需要三张可见 GPU；使用项目兼容的本地依赖环境
python -m pytest --confcutdir=tests/training \
  tests/training/test_elastic_tp_handoff_native.py -q
```

### 零活跃 vLLM：训练中进入，或从全卡 SAE 启动

活跃 producer 数为 `pool_size - 当前 SAE TP`。SAE TP 达到池大小时，所有 producer 暂停，SAE 使用已有缓存；低水位后自动缩容一个 TP rank，并恢复该卡的生产。SAE 至少保留一个 rank，不支持 SAE TP0。

从零活跃 vLLM 状态开始 SAE 训练，可使用：

```bash
CUDA_VISIBLE_DEVICES=3,1 python run_sae_runner_gpu.py \
  "${COMMON[@]}" --elastic-tp-size 2 --sae-tp-size 2 \
  --output-path results/elastic_2gpu_full_start

CUDA_VISIBLE_DEVICES=3,0,2 python run_sae_runner_gpu.py \
  "${COMMON[@]}" --elastic-tp-size 3 --sae-tp-size 3 \
  --output-path results/elastic_3gpu_full_start
```

不要传 `--vllm-dp-size 0`；活跃 producer 数由卡池与当前 TP 推导。全卡启动时 supervisor 先在 SAE worker 启动前临时运行 producer，预填充目标为 `min(streaming-num-chunks, max(streaming-prefetch-chunks, GA), 总微批数)` 块，然后等待所有在途写入完成并确认 producer 已释放（release）或暂停（resident），再初始化请求的全卡 SAE TP。若储备不足一个 GA 窗口，先恢复 producer 再训练。排空期间已生成的完整块也会保留，因此实际预填充数可能多于目标。预填充期间没有 SAE 更新，最低 SAE TP 限制不受影响。控制 epoch 从启动 0、预填充 1、交还全卡 SAE 2 单调递增，trainer 不会重置为 1。

“零活跃 vLLM”时 producer 控制进程仍存活；release 模式释放模型，resident 模式保留模型。完全不加载 vLLM 的预存激活训练不属于这个在线模式；已有独立文件输入入口 `run_dynamic_tp_sae.py`，用法见后文。

### AuxK 的默认解码策略

普通 `run_sae_runner_gpu.py` 和 elastic TP 共用 Aux 解码策略。省略
`--aux-compute`（或传 `none`）时使用 `auto`，**正常情况始终 compact**，
无需另外调节计算方式；TopK、近全选和 select-all 都遵循这个默认值。
`--sae-aux-k K` 设置每个 token 在所有 TP rank 上合计的选择预算；省略时
为 `d_in // 2`，设为 `0` 则关闭 Aux。

全量、分片与列压缩是三个不同概念。设 batch 为 `B`，本卡拥有 `S_r`
个特征，本批次在本卡实际选中的 dead 列并集为 `C_r`：

| 路径 | 激活形状或计算宽度 |
| --- | --- |
| full 存储 | 全局 `[B, d_sae]`，可能涉及 TP all-gather |
| local dense | 本地 `[B, S_r]`，仍包含本卡未选中的列 |
| compact 解码 | 本地 `[B, |C_r|]`，只对选中列及其 decoder 权重做 GEMM |

`--aux-storage full|sharded_dense|sharded_ragged` 控制存储与选择路径；
compact 解码在本卡进行，不把压缩后的激活或权重重新聚合到全局。
各 rank 的 `C_r` 可以大小不同，切换 TP 后按新的实际归属重新计算。

每个 token 全部 rank 的有效选中数之和是 `min(K, dead)`。不同 token
可能选择不同特征，因此本批次 compact 宽度之和 `sum(|C_r|)` 可能大于
K，最多达到全局 dead 数；它不是固定的 K，也不是 `max(K, dead)`。
需要筛选时用本地 feature bitmap 求准确并集，不排序整个 `B*K` ID 数组。
选中值恰好为零的项仍保留，以保证梯度正确。

全选条件是 **全局 `dead <= K`，包含相等**：同步与异步入口都跳过 TopK
比较。此时 `C_r` 就是本卡 dead 列；workspace 足够时直接 compact GEMM，
避免逐项 ragged 元数据，超出 workspace 时使用分块 compact。

默认 auto 的 dense 例外是保守的：用户显式设置的 K 至少为全局
`d_sae / 2`，过滤后的 `min(K, dead)` 仍至少为 `d_sae / 2`，且本卡的
实际选中列并集覆盖整个本地 shard。三者同时满足时，列压缩已经不再减少
该卡矩阵宽度，直接使用 local dense。缺少全局选择元数据时保留 compact。
普通 AuxK 不因本地 K 越过 512、本地 dead 比例或密度而自动改走 sparse。

显式 `--aux-compute sparse|compact|dense` 和阶段覆盖项仍然生效。
旧的 Aux 本地 K 阈值字段仅保留配置兼容性，不再控制 Aux auto；列宽比例
和密度阈值用于 Main auto。普通 Main TopK 默认仍是 sparse，可用
`--main-compute sparse` 显式指定，与 Aux 的 compact 策略独立。

### 自动切换与可调系统参数

水位为 SHM 已占用槽位占总槽位的比例，包含正在写入的块；训练侧已取走的本地缓存不计入该水位。高水位说明输入积压，控制器尝试增加一个 SAE TP rank；低水位说明生产不足，尝试减少一个 SAE TP rank。扩容前必须先等待将加入 SAE 的 producer 完成所选模式的交接确认，切换发生在完整 optimizer step 之间。

| 参数 | 默认值 | 用途 |
|---|---|---|
| `--elastic-tp-size` | 4（使用 `--elastic-tp` 时） | 固定 GPU 池大小；显式传此项即启用 elastic TP |
| `--elastic-tp-min-size` | 1 | SAE TP 硬下限，也是未指定初始 TP 时的启动值 |
| `--sae-tp-size` | 最低允许 TP | 初始 SAE TP，允许等于卡池大小 |
| `--elastic-tp-low-watermark` | 0.05 | 低于或等于此水位时考虑缩容 |
| `--elastic-tp-high-watermark` | 0.85 | 高于或等于此水位时考虑扩容 |
| `--elastic-tp-poll-interval` | 0.5 秒 | 水位采样间隔 |
| `--elastic-tp-watermark-samples` | 3 | 连续满足阈值所需的采样次数 |
| `--elastic-tp-cooldown` | 12 秒 | 每次切换后的冷却时间 |
| `--streaming-num-chunks` | 96 | SHM 槽位总数，每块一个训练 batch |
| `--streaming-prefetch-chunks` | 2 | 训练侧一次补充、打乱的最大块数 |
| `--elastic-tp-startup-timeout` | 300 秒 | producer 初始化 / 全卡启动预填充的超时限制 |
| `--elastic-tp-pause-timeout` | 120 秒 | 切换等待 producer 排空与确认的超时限制 |
| `--elastic-tp-resume-timeout` | 300 秒 | 缩容后等待 producer 重新加载/ready 的超时；也作为关闭时等待冷加载返回的宽限期 |
| `--elastic-tp-vllm-residency` | `release` | `release` 关闭/重载，`resident` 保留暖权重 |
| `--elastic-tp-release-tolerance-mib` | 64 MiB | 释放后比模型初始化前基线额外存活的 PyTorch 分配上限 |

一般保留水位、采样和冷却默认值。短验收为了观察切换，可以缩小缓存并缩短采样/冷却，但这些不是日常启动必需参数。是否发生切换取决于实际生产、消费速率与水位；短任务可能在满足连续采样条件前就结束，处于 TP 下限或上限时也不会越界切换。

通用参数仍由 `run_sae_runner_gpu.py` 配置，无需使用 elastic TP 专属副本：

| 类别 | 可复用参数 |
|---|---|
| 本地输入 | `--model-name`、`--dataset-path`、`--hook-name`、`--context-size`、`--store-batch-size-prompts`、`--max-model-len`、`--max-num-batched-tokens`、`--vllm-text-only` |
| 训练规模 | `--d-sae`、`--k`、`--training-tokens`、`--train-batch-size-tokens`、`--gradient-accumulation-steps`、`--dead-feature-window`、`--seed` |
| 精度与执行策略 | `--vllm-dtype`、`--activation-dtype`、`--activation-conversion`、FP32 `--dtype`、Main/Aux storage/compute/stage 参数、TopK/AuxK 参数、`--tp-overlap`、`--tp-overlap-max-live-hooks` |
| 缓存与输出 | `--streaming-num-chunks`、`--streaming-prefetch-chunks`、`--output-path`、`--no-save-final-sae`、`--no-save-final`、`--performance-only` |

首次加载缓存时自动估计固定 activation scale，一般无需设置；专用诊断选项包括 `--elastic-tp-activation-scale`、`--elastic-tp-audit-inputs`（完整输入 SHA-256 校验）和 `--elastic-tp-validate-activations`（检查 producer 激活是否有限），后两者有额外开销。

显式设置 `--streaming-chunk-size-tokens` 时必须等于训练 batch。传输固定为异步 SHM；普通 streaming 的 rolling mix 与 exact 后台队列、DDP/FSDP optimizer overlap、普通 runner 的 profiling 尚未接入。`--elastic-streaming` 是另一套 elastic DP 模式，不能同时启用。尚不支持在线 checkpoint 恢复或定期训练状态保存；显式传入未支持参数会报错。

`--hook-name` 控制在线 hook；显式 `--hook-names` 只接受一个 hook，未指定时不采用静态 runner 的四 hook 默认值。默认导出 canonical feature 顺序的最终 SAE 至 `model/`，`--no-save-final-sae` 或 `--performance-only` 可关闭导出。

### 如何确认运行和切换结果

启动终端打印卡池大小、初始/最低 SAE TP、初始活跃 vLLM 数和离线状态。详细信息写入输出目录：

| 文件 | 查看内容 |
|---|---|
| `startup.json`、`run_config.json` | 实际 GPU 映射、初始角色、最低 TP、离线环境与完整配置 |
| `startup_prefill.json` | 仅全卡启动时生成；预填充目标、实际块数与耗时 |
| `progress.json`、`train_rank0.jsonl` | 最新定期进度，以及完整 step、TP、水位、loss 和输入事件 |
| `switches.json` | 每次已提交切换的 old/new ranks、触发原因、水位和迁移验证；未切换时可能不存在 |
| `producerN.jsonl`、`producerN_status.json` | producer 暂停/恢复、生产 chunk、当前角色与 epoch |
| `saeN_status.json` | 退出 SAE rank 的同 epoch 张量/allocator 释放确认或失败 |
| `training_report.json` | 完成步数、tokens、唯一 chunk 数、切换记录与模型导出状态 |
| `result.json` | supervisor 返回码、错误和所有子进程退出码 |
| `train.log`、`producerN.log` | 初始化日志和失败 traceback |

例如 `tail -f results/elastic_3gpu/train.log` 可查看训练日志。正常完成应有 `training_report.json` 中 `passed=true`，`result.json` 中 `returncode=0` 且所有 `process_returncodes` 为 0。确认零活跃 vLLM 应同时查看全卡 TP 下的 step 和各 producer 的 `released`（release）或 `paused`（resident）记录，再检查低水位缩容后的 `running` 与新 `produced` 事件。输入已耗尽时允许同 epoch、已释放的 `done`，无需重新生产。训练完成后关闭 producer 失败也会令 `result.json` 非零；不能单看训练报告判断整个运行成功。

结束或失败后 supervisor 负责停止其启动的进程并清理本次 SHM。失败时保留输出目录中的日志；重跑请换一个新输出目录，已有 `run_config.json` 的目录不会覆盖。库调用入口为 `sae_lens.elastic_tp_runner.ElasticTPSAETrainingRunner`，配置类为 `sae_lens.training.elastic_tp_config.ElasticTPConfig`；库调用设置 `min_tp` 时需同时保证 `initial_tp >= min_tp`。

### 实测记录

本轮两卡/三卡、零活跃 vLLM 与最低 TP 验收保存在 [`results/elastic_tp_2_3gpu_20261007/`](../results/elastic_tp_2_3gpu_20261007/)。使用本地 Llama-3.1-8B、RTX 5090、d_sae=1025、k=32、batch=256、context=32，每组均为 64 步 / 16384 tokens：

| 目录 | 卡池 / 初始 / 最低 SAE TP | 实际切换 | 切换次数 | 全卡 SAE 训练步数 |
|---|---|---|---|---|
| `pool2_initial1` | 2 / 1 / 1 | 1→2→1 | 2 | 5 |
| `pool2_initial2` | 2 / 2 / 1 | 2→1 | 1 | 2 |
| `pool3_initial1` | 3 / 1 / 1 | 1↔2↔3，反复切换 | 28 | 34 |
| `pool3_initial3` | 3 / 3 / 1 | 3→2→1 后反复扩缩容 | 32 | 40 |
| `pool3_initial3_min2` | 3 / 3 / 2 | 3→2，此后保持 TP2 | 1 | 2 |
| `pool4_min2` | 4 / 2 / 2 | 2↔3↔4，未进入 TP1 | 30 | 40 |

这些短验收将 SHM 设置为 8 块，并使用 low=0.25、high=0.5、poll=0.01 秒、连续采样 1 次、cooldown=0，以覆盖切换路径；**不是使用默认控制器时延的吞吐测试，也不是大尺寸显存容量或训练数值等价证明**。启动命令未设置学习率，也未手写四个离线环境变量；不同案例使用不同的非连续 GPU 顺序。

全卡启动案例的首个训练 step 确实使用请求的 TP2 / TP3。各案例都核对了 producer 的暂停确认、全卡 SAE 训练，以及缩容后恢复 `running` 并发布新 chunk；producer 确认暂停后没有继续写入。`pool3_initial3_min2` 在 TP2 观察到 181 次低水位采样，其中 118 次无 READY 块且尚未完成生产，没有准备或提交 TP1 切换，随后完成全部训练。四卡下限案例未显式设置初始 TP，首个 step 自动为 TP2。

六组均完整消费 64 个唯一 chunk，无重复或遗漏；共 282 次完整输入 SHA-256 校验和 94 次切换缓存校验通过，loss 有限，模型成功导出，所有子进程退出码为 0，SHM 与 GPU 资源已释放。参数与入口回归 63 项通过，4 项独立 CUDA 调度单测在沙箱中跳过；上表为在可访问 GPU 的环境实际运行的在线测试。

完整命令见 [`commands.json`](../results/elastic_tp_2_3gpu_20261007/commands.json)，逐项验收结果见 [`verification.json`](../results/elastic_tp_2_3gpu_20261007/verification.json)。可用 `python3 results/elastic_tp_2_3gpu_20261007/verify.py` 重查日志；若重跑训练命令，先更换输出目录。两卡全卡启动后只发生缩容是此次生产/消费速率下的正常行为，不要求每个任务都反复扩容。

整合验收：66 项入口/传输/精度回归、16 项原生 Megatron 回归、23 项动态 TP 独立测试通过。`results/elastic_tp_integration_20261007/` 保存新主入口的两卡真实在线测试：本地 Llama-3.1-8B、d_sae=1025、32 步/8192 tokens，完成 TP1→TP2→TP1、28 次完整输入 SHA-256 检查、最终模型导出和 SHM 清理。另有 CUDA 测试覆盖 off/eager/lazy/bounded 四种调度的分项 loss 输出。此验收不扩大前述大尺寸训练数值等价的结论。

全卡启动与离线默认值的后续验收保存在 `results/elastic_tp_full_start_20261007/`：使用 `CUDA_VISIBLE_DEVICES=3,1,0,2 --elastic-tp-size 4 --sae-tp-size 4` 对应的启动配置，命令中没有设置上述四个离线环境变量。首个训练 step 为 TP4；64 步/16384 tokens 完整消费 64 个 chunk，自动完成 28 次切换（七轮 4→3→2→3→4），通过 32 次完整输入校验并导出模型，所有子进程正常退出且 SHM 已清理。参数、源路径检查、预填充握手和既有 streaming/精度回归共 98 项通过，2 项 CUDA 用例在沙箱内跳过。

当前工作区 `.venv` 的 Transformers 与全局 huggingface-hub 版本不匹配，以上验收沿用已有的 `PYTHONPATH=results/dynamic_tp_e2e_20261007/dependencies:.` 兼容环境。正常安装的兼容依赖环境无需该设置，包内运行代码也不引用该目录。

## 扩缩容算法

设总特征数 F，现有 m 个 owner，宽度相差至多 1。

扩容 m→m+1：先算目标真实宽度 floor/ceil(F/(m+1))，再让旧 rank 保留前缀、捐出尾部；新 rank 按 donor rank 顺序拼接收到的片段。余数优先分配给原本较大的幸存分片，避免因整数余数造成幸存 rank 间的额外迁移。

缩容 m→m−1：幸存 rank 保留全部原特征；将退出 rank 的本地序列按各幸存 rank 的缺额分块，追加到相应 rank 尾部。没有必要恢复原来的全局连续区间。

例：F=12，3→4→3，最后退出的可以是中间 rank 1：

| rank | TP3 初始 | TP4 扩容后 | rank 1 退出后 |
|---|---|---|---|
| 0 | 0,1,2,3 | 0,1,2 | 0,1,2,4 |
| 1 | 4,5,6,7 | 4,5,6 | 退出 |
| 2 | 8,9,10,11 | 8,9,10 | 8,9,10,5 |
| 3 | 未参与 | 3,7,11 | 3,7,11,6 |

每个 ID 始终恰好有一个 owner。相同 ID 的所有参数、moments 和计数保持对应。全局 dead/firing 摘要继续使用 canonical feature ID 顺序；不把 rank 拼接顺序误当作 canonical 顺序。最终 checkpoint/export 也还原 canonical 顺序。

在此平衡约束下，扩容仅迁移新 rank 最终持有的特征；缩容仅迁移退出 rank 的特征，达到相应所有权变更的必要网络字节数。测试直接统计传输字节，排除了隐式全模型汇集。

## 激活应如何处理

切换点放在完整 optimizer step 之后，且 `zero_grad(set_to_none=True)` 已完成。

1. `[B,d_sae]` logits / feature_acts / AuxK 中间张量属于已结束的 autograd 图。此时应释放，不需要迁移。不能在 backward 进行到一半时简单切片这些张量来恢复图。
2. `[B,d_in]` 是 TP 各 rank 共同消费的输入，不随 feature owner 切片。新 rank 需要同一批完整输入；可以按 token 行分块从多个已有副本拼装，但不能按 d_sae 的新 ownership 来切它。
3. 未消费的 provider/reservoir 输入不能随意丢弃。新入口按统一 step 游标读取同一份数据，切换不推进游标，不跳过或重复 token。外部接入时 provider 应留在 session 外，保持其游标和待消费队列。
4. `session.stage_inputs(batches)` 挂载所有旧 TP rank 内容相同、已冻结的连续 BF16/FP32 `[tokens,d_in]` 缓存（保留激活管理 dtype），存放在 `TPTrainState.activation_caches`。扩容时旧 rank 按 rank 顺序各发送不同的连续行段，新 rank 按原行位置拼装；行数不能整除 donor 数时使用真实长度，允许零行 donor。幸存者保留原存储，缩容不传激活，退出者释放引用。`train_cached_step(batch_size)` 只消费缓存前端；切换不消费任何行。CLI 默认缓存四批，可通过 `--cache-batches` 调节。
5. `replicated` 用于 canonical 顺序的计数等复制状态；非复制的、token 分片式 reservoir 仍需单独的行所有权协议。外部 provider 要保持每个 TP 副本的待消费内容和顺序一致，不能在迁移期间修改这些张量。

## 通信域与提交协议

`TPGroupPair` 的固定 pool 在启动时加入 WORLD；此后 WORLD 和工作进程都不销毁。

- `prepare(target_ranks)`：所有 WORLD worker（包括 idle 及 pool 外协调进程）按同一顺序调用；建立并 warm up 新 TP group。可以提前若干 step 调用，此后旧组仍可训练。
- 不用后台线程并发发起 NCCL 建组，避免多 communicator 的集体通信顺序冲突。准备本身仍有成本；驱动把它放在迁移计时之外，并不声称这部分完全被隐藏。
- `switch_tp`：确认不在训练窗口、无旧 gradient/main_grad；核对计划摘要、optimizer schema 和 progress；等待旧 GPU streams 完成。
- 暂存新模型和 Adam buffers；所有 rank 对分配/验证结果投票。失败时不提交，旧状态仍在。
- 按统一 hook→parameter→moment→move→chunk 顺序执行 P2P。大矩阵不经 CPU，不写临时 checkpoint。
- 全部迁移完成后 barrier，交换 active/prepared handle，将旧组留作 spare；旧组销毁移到下次 prepare 或 close。新参数对象没有旧 reducer/autograd hooks。

PyTorch 标准 ProcessGroup 不能通过重命名直接更改成员。本实现更换显式 group handle，使用 WORLD 同序的 `new_group(..., use_local_synchronization=False)`。本机 PyTorch 2.10 的 local-sync 名称散列包含进程本地 `pg_names` 数量，经历不同成员的组后会不一致，真实测试曾在 TP1→TP2 卡住；全局建组顺序解决了这一问题。固定迁移组包含加入者和退出者，NCCL 组绑定 `device_id` 并提前初始化，避免暂停中再建立懒初始化的 P2P 通信器。所有 worker 在 session 启动时预热 Megatron/Adam 依赖，避免新 rank 首次加入时才导入。

迁移采用固定顺序、分块 P2P。当前实现串行提交各 donor 和 tensor 的传输，NCCL 的 eager P2P serialization 提示与这个调度一致；未声称多 donor 传输并发。

分配失败与通信失败不同：分配/schema 错误在传输前一致拒绝；通信中途失败会把 session 标记为 failed，不承诺跨 GPU 故障的透明回滚。

## 显存和暂停时间

FP32 encoder/decoder/b_enc 加普通 Adam 两个 moments，单个 feature 的可迁移状态字节为：

`4 × (2*d_in + 1) × 3`。

F=32768、d_in=4096 时，2→3 每 hook 转移约 1 GiB 的分片状态，另加很小的复制 bias/计数和按需输入缓存。H 个同尺寸 SAE 乘 H。AMSGrad 还有一个 moment。这里是字节推导，不是实测毫秒数。

暂停包括 drain、计划/schema 核对、目标分配、本地保留数据拷贝、P2P 和 commit。通信域准备与旧组回收耗时另外记录；prepare 当前仍为同步操作，不能把它从应用总开销中忽略。`switch()` 返回总传输字节数、移动 feature 数及分阶段耗时。性能脚本汇总各 rank 的最大暂停和 allocator 峰值，详见下面的实测。

为了在迁移前分配失败时保留旧模型，本版暂存新旧两套本地参数/Adam。每卡需预算 old_local_state + new_local_state + 至少一个传输 chunk（默认 16 MiB；单个 feature 超过该值时至少容纳一个 feature）及上下文/allocator。复制的计数/输入缓存在幸存者上复用，不额外复制。没有执行本地 parameter storage 的原地缩容或 reserve-capacity arena，这可以作为后续降低峰值/拷贝成本的工作。

`megatron_memory_model.py` 已对静态 TP 使用最大真实分片 ceil(F/TP)，避免用 floor 低估最大 rank。它不估计本版切换峰值。旧的其他 legacy profiler/simulator 不在本次改造范围。

## 应用与运行

先在四卡机器运行真实通信验收：

```bash
python -m torch.distributed.run --standalone --nproc_per_node=4 scripts/validate_dynamic_tp.py --backend nccl --report dynamic_tp_acceptance.json
```

脚本覆盖 `1→2→3→4→3→2→1` 及中间 rank 退出、重新加入，逐步比较 loss、权重、Adam/AMSGrad moments、step 和 dead counters。CPU 机器可把 backend 改成 gloo。本地已完成真实 NCCL/Gloo 验收；可加 `--topk-backend sharded_sparse`，或 `--topk-backend sharded_ragged --key-backend triton --protocol radix`，以及 `--wavefront` 覆盖相关分支。

真实训练的输入文件需包含每 hook 一个 `[tokens,d_in]` tensor，已经按训练配置完成 scaling。以下需要每个 hook 至少 `70*8192` 行：

```bash
python -m torch.distributed.run --standalone --nproc_per_node=4 run_dynamic_tp_sae.py \
  --activations activations.safetensors \
  --hooks blocks.16 blocks.21 \
  --d-sae 32768 --k 128 --batch-size 8192 --steps 70 \
  --tp-schedule '0:1,10:2,20:3,30:4,40:3,50:2,60:1' \
  --backend nccl --output dynamic_tp_output
```

schedule 的 key 表示已经完成多少个 optimizer step。计划中的所有 GPU 都需在 worker pool 内；它们空闲时仍留在进程中。默认最终 `save_model` 只导出模型。完整训练 checkpoint 需显式启用 `--checkpoint-every N` 或 `--checkpoint-final`，用 `--resume <checkpoint_dir>` 恢复；这与切换中的纯显存迁移相互独立。`--profile-only` 完全跳过模型导出，并拒绝与 checkpoint 写入选项同时使用。

`session.save_checkpoint(path, provider_state=...)` 保存参数、Adam/AMSGrad、optimizer options、feature ownership、firing/dead counters、进度、待消费输入，以及每个 WORLD worker 的 Torch CPU/CUDA、Python、NumPy RNG。文件逐个校验，完成后才将 `.incomplete` 目录原子改名。`session.load_checkpoint(path)` 支持同一固定 WORLD pool 内恢复为不同的 TP 数量/成员，并返回 provider state。外部 provider 的游标、shuffle generator 的 `get_state()`、排列和 scaling 需要由调用者以 tensor/基础 Python 类型传入；它不自动序列化在线 vLLM producer 或外部 SHM 队列。CLI 会校验激活文件 SHA-256、batch size 和 hooks。

任意 rank 成员使用 `--schedule-json schedule.json`，文件例如：

```json
{"0":[0,1],"10":[0,1,2],"20":[0,2],"30":[2]}
```

在线应用可通过 `DynamicTPSession.prepare(ranks)` 与 `switch()` 自行决定切换时刻，不局限于预先列出的 schedule。调用必须由全部 WORLD rank 同序执行。输入 producer 可在 session 外继续运行；session 本身不管理 producer/consumer 角色。

### 四卡在线端到端测试

`scripts/validate_dynamic_tp_e2e.py` 提供独立的真实输入测试驱动。默认使用本地 Llama-3.1-8B、Wikitext2 tokenized、`blocks.21.hook_resid_post`，d_in=4096、k=128、FP32 SAE、fused Adam、sharded ragged Main/AuxK。强制离线模式，不下载模型或数据。

```bash
python scripts/validate_dynamic_tp_e2e.py capture --output results/dynamic_tp_speed --producer-id 3
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  scripts/validate_dynamic_tp_e2e.py benchmark --output results/dynamic_tp_speed
python scripts/validate_dynamic_tp_e2e.py run --output results/dynamic_tp_online \
  --d-sae 65536 --batch-size 4096 --steps 4000 --dead 1500 --low 0.05 --high 0.85
```

测速复用一批**真实**激活；其中 `aux_stress` 会人为增加未触发 feature 的年龄，只用于测 AuxK 开销。正式 `run` 不修改年龄或 dead mask，按真实训练累计到 dead window。

四个 SAE worker 固定存活，初始只有 rank 0 训练，GPU 1/2/3 各有一个 vLLM producer。TP=m 时 GPU 0..m-1 训练，其余 GPU 生成激活。扩容前等待加入 GPU 的 producer 完成当前 chunk 并确认暂停；收缩后再恢复其推理。暂停的 vLLM 权重保留在原 GPU，不与 SAE 同时执行推理，不进行模型重载；因此显存预算必须包含这些常驻权重。这是固定四卡池的测试角色调度，不是旧 elastic DP/ZeRO 控制器的替代品。

水位定义为 SHM 中 writing+ready+consuming 占所有 slots 的比例。连续三个 0.5 秒采样达到上/下阈值时请求相邻 TP，切换间默认冷却 12 秒。prepare 后旧 TP 至少继续一个训练 step，再在完整 optimizer 边界切换；暂时无输入时可直接在该边界提交。prepare 同步耗时、迁移 pause、请求到提交时间分别记录。默认 256 个 BF16 chunks，每 chunk 4096 tokens；GPU 待消费缓存最多四批。每次取缓存将行随机打乱，第一批估计固定 activation scaling，此后不随 TP 改变。数据集不足时显式循环；每个全局 chunk sequence 仍恰好生产、消费一次。

每次切换检查每个 feature 的参数和 Adam 坐标抽样、完整 firing/dead counters、progress，以及激活缓存的行/列抽样。该大尺寸运行不是全部参数/moments 逐元素的固定 TP 对照；完整 oracle 仍由小尺寸 `validate_dynamic_tp.py` 执行。每步记录 loss 组成、dead 数量、输入耗时、训练耗时、TP 和显存，结束后导出 canonical 顺序模型。`devices.jsonl` 记录包含 producer、allocator、CUDA context 的设备总显存。`training_report.json`、`switches.json` 和 `result.json` 保存结果。

2026-10-07 四卡实测保存在 `results/dynamic_tp_e2e_20261007/`：

- `speed/`：真实输入单批重放，TP1/2/3/4 主路径分别约 158/83/59/46 ms，AuxK 压力测试约 294/157/113/90 ms；单个 vLLM producer 约 14,300 tokens/s。
- `smoke/`：160 步、dead=32、16 chunks 的短测，自动 1→2→3，包含 AuxK 启用后的 2→3 迁移。
- `full/`：1H、F=65536、batch=4096、4000 steps、dead=1500、阈值 0.05/0.85，学习率 3e-4、aux_loss_coefficient=1。1638.4 万 tokens，在线训练及导出 612.96 秒，约 26,729 tokens/s；从测试配置写入到所有进程退出约 651.16 秒。
- 正式运行按水位完成 7 次转换：1→2、2→3、3→2、2→1、1→2、2→3、3→2。迁移最大 rank pause 106.8–153.0 ms，中位数 123.5 ms；通信组 prepare 14.0–576.8 ms，累计 1.537 秒。pause 累计 0.887 秒；包括切换验证后为 1.051 秒。prepare 仍有同步开销，不能当成零中断。
- 全部 4000 个 chunk sequence 恰好生产和消费一次；各 active TP rank 的 loss 和参与步数一致；最终全部权重有限且形状准确。设备总显存峰值最高 26.14 GiB。进程正常退出并释放全部四卡资源。
- 覆盖边界：正式运行未自然触发 TP4，最后一次切换在 step 1437，AuxK 从 step 1502 开始，之后稳定在 TP2。因此正式长跑不构成 AuxK 启用后的迁移或 TP4 在线运行证明；TP4 测速及小尺寸完整迁移验收、短测中的 AuxK 迁移分别保留。
- **数值观察：step 1955 的 AuxK loss 峰值为 387442.16，主 MSE 为 6026.76，dead=2104；step 1957 的 dead 降至 2047。随后恢复，最终 MSE=779.91、AuxK=61.97。** 尖峰发生在持续 TP2 区间，距最后一次切换 518 步。这只是时间关系，尚无固定 TP 的真实数据重放对照，不能确定归因，也不能把“运行/迁移核验通过”解释为训练数值没有异常。

`full/audit.json` 保存独立核验和上述数值观察，`full/training.png` / `.svg` 是训练曲线；`full/model/` 为导出模型。依赖环境与源码摘要见 `full/provenance.json`。本机测试通过 `--dependency-dir` 使用已有的本地兼容导入目录，不更改共享虚拟环境。

绘图入口 `scripts/plot_dynamic_tp_e2e.py <run_dir>` 直接读取现有日志，无需重跑训练或权重核验。输出 PNG 和 SVG：`training` 为总览；`mse` 为线性坐标、普通十进制刻度的全范围与 0–2000 局部图；`buffer` 包含 5%/85% 水位、slots 状态、生产/训练速率；`tp_switches` 标出每次切换及各 GPU 的 SAE 分配；`switch_costs` 展示建组、迁移、验证耗时与传输量；`resources` 展示设备显存及利用率。总览中的 MSE、AuxK 也均采用线性坐标。`figures.json` 列出完整图件。

## 本地验证（2026-10-07）

环境：Python 3.10、PyTorch 2.10.0+cu128、Megatron Core 0.16.1，4 × RTX 5090（每卡约 32 GiB）。测试未修改依赖版本。

- standalone 数学、CUDA kernel、Gloo 回归与动态迁移：176 项通过。动态测试包含超过 12,000 次随机 ownership 变化、多 donor 缓存分段、短/空缓存、精确通信字节、Adam/AMSGrad 续训、预检拒绝与双通信组复用。
- 原生 Megatron CPU 和显存模型：24 项通过，覆盖 TP1/2/3/5/7 的真实分配、非连续 ID、参数/Adam 续训和 canonical checkpoint。
- 四卡 NCCL：dense、sparse、ragged+Triton radix；分别逐元素比较 loss、全部参数、Adam/AMSGrad moments、step、dead counters 和待消费输入。覆盖任意中间 rank 离开、重新加入，以及 Adam 尚未初始化时切换。
- 原有 MultiSAETrainer：新增一项四卡测试通过，覆盖 F=49、TP3/DP1 与 TP2/DP2、GA2、wavefront、分片 Adam，以及存盘恢复后再训练。
- 命令行入口：合成激活、两 hook、14 步、98 tokens/hook，与固定 TP1 的最终模型逐元素对照，最大差 `7.45e-9`；缓存四批、两步一次切换，覆盖非空缓存跨迁移继续消费。
- 五进程 Gloo：四个 eligible worker 加一个 pool 外协调进程，完成同一切换序列。

结果文件保存在 `results/dynamic_tp_20261007/`；`nccl*.json` 和 `gloo.json` 是真实通信验收，不能与线程模拟测试混淆。

单次实测（两个 hook，F=32768/32771，d_in=4096，batch=64，每 hook 缓存 256 行）：

| 切换 | 暂停 ms | prepare ms | 传输 GiB | 最大 allocated GiB |
|---|---:|---:|---:|---:|
| 1→2 | 148.5 | 55.9 | 3.009 | 9.05 |
| 2→3 | 121.3 | 507.0 | 2.009 | 5.06 |
| 3→4 | 114.4 | 150.5 | 1.509 | 3.55 |
| 4→3 | 111.6 | 7.9 | 1.500 | 3.57 |
| 3→2 | 114.4 | 215.8 | 2.000 | 5.07 |
| 2→1 | 139.4 | 270.2 | 3.000 | 9.07 |

这里只报告实测值，没有 checkpoint/resume 基线，因此不报告加速倍数。峰值是 PyTorch allocated memory，不包括 NCCL/context 或 allocator reserved；新旧状态共存的成本已包含。

性能复现：

```bash
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  scripts/profile_dynamic_tp.py --d-in 4096 --widths 32768 32771 \
  --batch-size 64 --cache-rows 256 --report dynamic_tp_profile.json
```

该性能脚本训练真实 SAE 并分配完整 Adam 状态；在计时区间外检查每个 feature 的参数/moments 抽样及完整输入缓存。它不是逐元素模型验收，逐元素比较由小尺寸验收脚本完成。训练 batch、硬件拓扑和内存压力会影响结果，报告中的 prepare 耗时与 pause 耗时分列。

## 2026-10-07 耐久、恢复与严格数值审计

完整结果：`results/dynamic_tp_audit_20261007/`。本次测试没有把严格数值失败改成更宽的容差，也没有把只有模型权重的导出当作完整训练恢复。

- **通信域和显存：** `stress/` 在 1H、d_in=4096、d_sae=65536、batch=4096 下训练并做 120 次相邻切换，包含中间 rank 退出、rank 0 退出、重新加入。存活进程组最多 5 个，即 WORLD、control、transfer 和两套训练组；close 后回到 WORLD。allocated、reserved、NVML 显存、文件描述符和线程数进入稳定周期，没有持续增长，OOM 和 allocator retry 均为 0。存在碎片：`inactive_split_bytes` 周期峰值约 2567 MiB，而非零碎片；它没有随循环累积。空闲 rank 的历史 reserved 可保留约 26.2 GiB，实际存活张量约 16.25 MiB，这是 allocator/库缓存，不代表模型或旧通信组仍存活。若要把空闲卡交给其他进程，需要另行处理缓存释放和显存预算。
- **反向复用：** `final_matrix/stress_reuse/` 在 bounded overlap 下反复 1↔2，共 60 次。除第一次创建 TP2 组外，之后 59 次 prepare 均复用备用组，没有调用 `new_group`。
- **逐字节迁移：** `state_proof/full_state_migration/` 对 65536 维完整模型及全部 Adam moments，按 canonical feature ID 哈希每个元素，另对所有 dead/firing 状态和待消费缓存哈希。10 次迁移前后的 SHA-256 均一致。该测试 batch=64，状态大小仍为完整 65536×4096；另一个 batch=4096 的 120 次压力测试使用参数/moments 坐标抽样。
- **overlap：** off/eager/lazy/bounded、dense/sparse/ragged、原生 Triton，以及 full Main/sharded Aux 和 sharded Main/full Aux 的小尺寸固定 TP1 oracle 均通过；另测了三个 hook、窗口 2 的滚动调度。原 static runtime 的 TP/DP、GA、optimizer overlap、AMP/空 shard、保存恢复等回归 41 项通过，CPU 回归 42 项通过。这不表示动态 session 已支持 DP/ZeRO 或 DDP optimizer-overlap。
- **checkpoint：** `checkpoint/` 在非连续成员 `[1,3]`、rank 0 空闲、缓存尚未消费完时保存，退出进程，再启动新进程。相同 TP 恢复并续训逐位一致；恢复为 TP3 后续训最大状态差为 0.0000000149。参数、Adam/AMSGrad、计数器、缓存、provider 的排列/游标、四种 RNG 均验证。损坏文件在提交前被一致拒绝，当前状态不变。canonical `save_model` 重新加载也逐位一致。
- **FineWeb 在线：** `final_matrix/fineweb_online/` 使用 `/root/datasets/fineweb_tokenized_llama31_ctx2048`，1H、65536、4096 batch、512 steps、dead=32、上下阈值 0.05/0.85、cooldown=1 s。512 个 chunk 恰好生产和消费一次，共 2,097,152 tokens，源数据不循环。每次补充缓存逐字节哈希并核对所有 active TP rank；切换前后也检查完整缓存 SHA-256。每一步所有 active rank 的 loss、组成和 token 进度完全相同。自然切换发生在 step 7（1→2）和 19（2→3）；这次在线运行不覆盖 AuxK 启用后的切换，后者由压力和 oracle 测试覆盖。
- **profiling 退出：** CLI 的 `--profile-only` 没有创建输出模型目录，导出分支约 1 微秒，随后约 2.8 秒进程全部退出。FineWeb 在线测试同样设置此标志，保留日志/报告而未生成 `.pt` 或 `.safetensors`。关闭后的资源检查与进程退出时间均不包含 checkpoint 保存。

120 次测试同时包装真实 `dist.send/recv`，发送量、接收量和计划字节逐次完全相等。以下为每次切换的 **tensor 有效载荷**，不包含控制通信和 NCCL 协议开销。增长时范围不同来自剩余输入缓存长度；收缩不传输入缓存。

| 方向 | 有效载荷 GiB | migration 中位 ms | prepare 中位 ms |
|---|---:|---:|---:|
| 1→2 | 3.064–3.189 | 137 | 106 |
| 2→3 | 2.001–2.126 | 104 | 328 |
| 3→4 | 1.563–1.688 | 90 | 579 |
| 4→3 | 1.500 | 86 | 159 |
| 3→2 | 2.000 | 102 | 327 |
| 2→1 | 3.000 | 131 | 87 |

prepare 当前仍同步阻塞，不能只用 migration 时间代表总切换影响；同两种成员来回切换可以避开重新建组。切换不写 checkpoint，也不重新加载模型。`memory_and_groups.png`、`switch_costs.png`、`numerical_replay.png` 及 SVG 保存上述数据的可视化。

### 严格数值等价未通过的部分

数学上的 feature 置换不改变 SAE，但 FP32 的训练轨迹不能仅凭这个性质得到保证。大尺寸 oracle 使用同一批既有真实 Llama 激活重放（`dynamic_tp_e2e_20261007/speed/probe.safetensors`，来源为此前的 Wikitext2 probe），每一步逐元素比较参数、Adam 和计数器，容差保持 `atol=0.00002, rtol=0.0001`，计数器要求精确相等。

- 正常学习率 0.0003、普通 Adam、dead=1500，固定 TP1 自对照及固定 TP2 对 TP1 通过。
- 动态 TP 经历不等分 TP3 后出现少量超差坐标。10 步后 encoder/decoder 权重最大差约 0.000852/0.000769，权重相对 L2 差约 0.0185%/0.0245%；loss 最大差 0.00390625。关闭 overlap 仍复现基本相同的差异，累计 firing counts 也出现差异。
- 固定不等分 TP3 对 TP1 同样未通过，且首步即出现 firing counts 差异。因此该现象不依赖动态迁移，也不是新加 overlap 调度独有的问题。上述逐字节迁移检查排除了已测路径上的状态搬错，但尚未完全定位大尺寸不等分执行下的选取/归约/梯度数值差异，**不能宣称严格训练等价已经验收**。
- 更激进的 lr=0.002、dead=0、AMSGrad 压力配置，连固定 TP1 两次原生稀疏执行也出现小幅分歧，动态轨迹的差异更大。这些失败原始日志和 JSON 均保留。

在线数据层也需区分两种保证：同一次运行的 TP ranks 消费相同输入、无重复/漏消费已通过；不同在线运行的 producer 完成顺序和 refill 大小可能不同。相同 seed 并不保证不同切换计划下的在线 minibatch 顺序完全一样，数学对照使用固定输入重放；完整在线 resume 还需调用者持久化外部 producer/SHM provider 状态。

复现入口为 `scripts/audit_dynamic_tp_suite.py`，使用新输出目录，`--continue-on-failure` 会继续其余独立测试并保留失败状态。`scripts/plot_dynamic_tp_audit.py <audit_dir>` 重绘压力和数值图；`scripts/plot_dynamic_tp_e2e.py <online_run_dir>` 生成在线 MSE、buffer、水位切换、资源和耗时图，MSE 使用普通十进制线性坐标。

### 首步分数路径隔离与连续 decoder norm

后续验证保存在 `results/dynamic_tp_norm_20261007/`，原审计目录保留修改前的结果。此次只调整 decoder norm 的计算布局，同时补齐原来绕过 `_decoder_norm()` 的 wavefront encode 入口；没有修改 feature 所有权、迁移方案、TopK 实现或验收容差。

`_decoder_norm()` 现在通过 `_decoder_vectors().norm(dim=1)`，沿每个 feature 连续的 4096 个元素计算范数。wavefront 保存并复用这份可微的 decoder vectors，Main/Aux 继续共享当前 forward 的图，不跨 step 缓存。PyTorch 2.10 的 [Reduce.cuh](https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/native/cuda/Reduce.cuh) 根据输出宽度、地址和 stride 选择归约配置；这提供了旧范数可能随 TP 宽度变化的具体机制，但机制本身不能证明它触发了已观察的胜者变化。

`scripts/audit_tp_score_path.py` 使用同一份真实激活、seed=47、FP32、TF32 关闭，让 TP1 与固定不等分 TP3 分别计算，并按稳定 feature ID 对齐。在修改后的代码上同时计算旧/新范数，做可控替换：

| 对照 | 首步实测 |
|---|---|
| 初始 encoder/decoder 参数 | 全元素一致 |
| 旧 strided norm，TP3 对 TP1 | 13071 个 feature 的范数不同，最大差约 0.0000000149 |
| 新 contiguous norm，TP3 对 TP1 | 全部范数完全一致 |
| encoder 原始输出，TP3 对 TP1 | 最大差 0.00002384185791015625 |
| 原始输出各自计算 + 旧 norm | 6 个 token 的胜者变化，12 个 feature 的 firing count 不同 |
| 原始输出各自计算 + 新 norm | 同样的 6 个 token、12 个 feature 不同 |
| 强制使用 TP1 原始输出 + 旧 norm | 胜者和 firing count 一致，选中值仍有微差 |
| 强制使用 TP1 原始输出 + 新 norm | 胜者、选中值、firing count 全部完全一致 |

这是对该批输入首步的隔离证据：旧 norm 确实存在分片相关舍入差异，但它不是此处 12 个 firing-count 差异的触发点；encoder 原始输出的差异已足以解释这次胜者变化。不能据此宣称整个训练的所有后续差异都已定位。

`score_path_confirmed.json` 进一步核对：各 rank 的 encoder 输入完全一致，相同形状重复 GEMM 完全一致。在 TP1 所在的同一张 GPU 上，用相同权重分别计算 21846/21845/21845 宽度的 `matmul + bias`，结果与真正 TP3 的三个分片逐元素完全相等，却与该卡的 65536 宽度 GEMM 有 259402245 个 raw 元素不同、最大差仍为 0.00002384185791015625。这把本例首步 raw 分歧进一步隔离到 GEMM 的输出宽度，而非跨卡输入或权重不同；具体 cuBLAS 内部算法/累加实现尚未剖析。

“相同分数”的独立检查将 TP1 的 TopK 前分数原样按稳定 ID 分给 TP3。balanced 与迁移后的不连续布局、Torch/Triton key backend、auto/candidates/radix 协议全部组合，以及额外的全相同分数 tie，共 24 项，胜者、选中值、计数均精确相等。真实输入检查使用完整 batch=4096；全 tie 检查使用 8 行、65536 features。这里不使用近似容差。

TP1 在真实分数和全 tie 输入上的胜者还与独立的 `torch.argsort(descending=True, stable=True)` 逐项完全相等，避免只比较同一个 TopK 实现的两条路径。

修改后重新运行 10 步原生训练 oracle：固定 TP1、固定 TP2 仍通过；固定 TP3、动态 bounded、动态 off 仍超出原容差。动态 encoder/decoder 最大参数差约 0.000852/0.000769，loss 最大差 0.00390625，与修改前相近。固定 TP3 最大参数差约 0.005705/0.005769，loss 最大差 0.01611328125。相同 TP1 对照的最大权重差约 0.000000183，因此不能仅以“正常重复运行也有差异”就宣布训练质量无影响。10 步重放不是长期质量评估。

验收应分开记录：相同分数下 TopK/所有权/计数的语义一致性必须精确通过；跨 TP 的 FP32 权重轨迹逐位相同不是已实现的保证。若后续要强制一致轨迹，需要单独评估固定 encoder GEMM 累加方式、decoder 与反向归约的性能代价；若目标是训练语义和质量，则继续保留数值诊断，并用足够长的受控训练比较质量及同配置重复运行基线，不能直接放宽本次失败容差。

复现首步隔离（四卡、已有兼容依赖环境；不需要模型或数据集下载）：

```bash
PYTHONPATH=results/dynamic_tp_e2e_20261007/dependencies:. \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
NCCL_LAUNCH_ORDER_IMPLICIT=1 OMP_NUM_THREADS=1 \
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  scripts/audit_tp_score_path.py \
  --output results/score_path_replay.json
```

## 输入精度与默认 BF16 管理（2026-10-07）

vLLM、输入激活管理、SAE 参数、Main/Aux representation 独立配置。正常入口
`run_sae_runner_gpu.py` 默认 `--vllm-dtype bfloat16 --sae-dtype float32`，
`--activation-dtype none --activation-conversion auto`。`--dtype` 是
`--sae-dtype` 的兼容名称；Main/Aux 配置没有改变。

默认链路为 vLLM BF16 → hook BF16 → BF16 store/routing/SHM/mixing/pending cache
→ 当前 batch 转 FP32 → FP32 SAE。转换在 SAE 消费边界、输入缩放之前完成，避免先在
BF16 中缩放造成舍入；新分配的 FP32 batch 原地缩放，并同时供 forward 和 MSE target
使用。`process_sae_in()` 内已有的 `.to(self.dtype)` 此时不复制、不分配。未消费缓存
保持原始 dtype 和数值，不会因训练或 TP 迁移变为 FP32。

| 设置 | 激活存储与传输 dtype |
|---|---|
| 显式 `--activation-dtype float32/bfloat16` | 始终遵守显式值 |
| `none` + `auto` | producer 与 SAE compute dtype 中更小者 |
| `none` + `vllm` | 在 producer 侧转换成 SAE compute dtype |
| `none` + `sae` | 保留 producer dtype，到 SAE 消费时转换 |

FP32 producer + FP32 SAE 在 auto 下仍为 FP32，绝不默认量化成 BF16。这里不增加 SAE
低精度训练能力；现有 Megatron SAE/Adam 仍要求 FP32 参数。已有 SAE autocast 是独立
选项，默认关闭。`--autocast-lm` 也不替代 vLLM 模型 dtype 配置。

上述管理策略覆盖正常共置、routing NCCL/SHM、streaming 和 DP elastic；其 buffer、
staging、provider、scatter 使用解析后的配置 dtype。GPU-direct 空混合池跟随输入
类型，避免空 FP32 张量参与 cat 提升整个 BF16 池。SHM attach 验证实际 dtype 与
配置相同。用户从 cached activations 训练时，根据缓存 metadata 推断源 dtype，
未知或混合 metadata 保守用 FP32，不套用未启用的 vLLM 默认值。

在线 TP 入口 `scripts/validate_dynamic_tp_e2e.py` 也默认 vLLM BF16、activation auto、
SAE FP32；文件入口 `run_dynamic_tp_sae.py` 从各 hook safetensors 实际 dtype 解析
管理 dtype。动态 session 的 `input_scale` 保存在 checkpoint 中；缺少该字段的旧
checkpoint 按 1.0 加载，与旧的已缩放 FP32 缓存兼容。provider dtype metadata 变化时
恢复会拒绝，需显式选择旧 checkpoint 使用的 dtype。

`DynamicTPInputLoader.load_raw()` 由固定输入 owner 使用：SHM 槽位直接复制到可复用
pinned buffer，释放槽位后按原 dtype H2D，在 GPU 上 shuffle。CPU RNG 和排列顺序
保持不变，host 重用前等待前次 DMA 完成。原 dtype 的返回缓存由 session 负责广播、
迁移和 checkpoint。只有 `train_cached_step()` 消费的 batch 转 FP32；保留的
`load(scale=...)` 是显式请求整池 FP32 的兼容接口，在线默认路径不使用它。

缩放系数仅首次估计时同步一次；指定现成系数时无需额外同步。首次 norm 校准按 batch
转换计算，不分配整池 FP32 临时量。不会新增 NCCL 通信域或后台 collective。

16 × 4096 × 4096 输入，TP2、单 CPU 线程，对照上一次已优化的 pinned refill（整池
FP32 + FP32 broadcast），包含广播和全部 16 批的必要转换/缩放；交替运行 6 次，
丢弃首轮后平均：

| 输入 | 之前每 batch | 当前每 batch | 当前 pending cache |
|---|---:|---:|---:|
| BF16 | 6.91 ms | 6.18 ms | 512 MiB（之前 1024 MiB） |
| FP32 | 12.23 ms | 12.28 ms | 1024 MiB |

BF16 owner 的操作增量显存峰值由 1536 MiB 降至 576 MiB（不含两版本共有的固定
staging 和验证 reference）。GPU upload staging、pinned staging 各 512 MiB 固定复用，
没有新增 FP32 全池。BF16 当前 batch 只分配一次 64 MiB FP32 转换结果。
两张卡全部元素与参考结果精确相同，FP32 样本保留 BF16 无法表示的尾数。

这是一项输入路径隔离测试，不等于完整训练吞吐提升比例，也没有重新运行 8000 步。
此前跨 TP encoder GEMM 舍入差异仍存在；输入传输精确不代表不同 TP 的训练权重轨迹
逐位一致。结果与脚本见 `results/precision_policy_20261007`。

## 在线 producer 与 elastic DP 共用异步传输

在线 TP 入口使用与正常 streaming/elastic DP 相同的 `AsyncVLLMShmWriter`：
hook 激活留在 GPU，按配置 dtype 打包，通过独立 CUDA stream 复制到两个固定的
pinned host 槽位，后台线程写入 SHM 并发布 READY。生产线程可继续下一次 vLLM
推理。默认路径没有逐块 CPU 拼接或全量有限值扫描；`--validate-activations` 可在
诊断时启用 GPU 有限值检查（会增加同步），形状校验始终保留。

producer 发出 paused/done 确认前必须 drain writer，保证已申请序号的激活全部发布；
仅执行 CUDA synchronize 不代表 CPU 后台写入已经结束。两槽位固定复用，copy stream
记录 GPU 源张量的生命周期，暂停期间不额外持有已提交的 GPU chunk。

生产日志的 `submitted.cycle_s` 是捕获与提交所占生产线程时间；`produced.duration_s`
是从捕获开始到 SHM 发布完成的单块延迟，包含可与下一块重叠的 D2H/写入。
`capture_s`、`d2h_s`、`shm_write_s`、`staging_wait_s` 单独保留，异步阶段不能直接
相加作为端到端时间。SAE 的 `input_s` 只覆盖消费侧准备，不代表生产传输成本。

## 参考工作与实现取舍

- HotSPa，SOSP 2024，*Enabling Parallelism Hot Switching for Efficient Training of Large Language Models*：多策略图共享参数存储、运行中搬迁参数/梯度、规划策略转换。https://doi.org/10.1145/3694715.3695969 ，实现 https://github.com/PKU-DAIR/Hetu 。
- DynaTrain，2026，*Fast Online Parallelism Switching for Elastic LLM Training*：用统一逻辑参数空间表达状态所有权，生成 retain/send/receive 计划，将通信域准备与状态迁移分离。https://arxiv.org/abs/2605.18815 ，实现 https://github.com/infinigence/ElasticMegatron 。
- AnchorTP，2025，推理系统的不等宽 TP 和最小迁移规划；其 KV cache/推理约束不能直接当成 SAE Adam 训练迁移。https://arxiv.org/abs/2511.11617 。
- PyTorch 多进程组顺序约束与 `new_group` 文档：https://docs.pytorch.org/docs/stable/distributed.html 。

本实现针对 SAE 利用 feature permutation 保留既有 owner 的数据；没有直接引入这些系统的整个运行时，也不把“预建组 + P2P 状态迁移”当成首次提出的通用技术。
