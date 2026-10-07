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

边界必须保留：动态 session 为 **DP=1、FP32 SAE、每 hook 一组 torch Adam**。CUDA 入口使用 fused Adam。它不是现有 `elastic_streaming` DP/ZeRO 角色切换的开关；没有把该控制器改造成同时调 DP/TP 的实现，也没有实现 vLLM 冷启动/停止、世界进程数量变化、FSDP/ZeRO 状态重分片、AMP scaler 迁移、部分梯度累计窗口迁移。现有静态 runner 获得不等分 TP，动态训练走新入口。

动态 session 使用原生 Megatron encoder/decoder 与现有 sharded TopK/AuxK。`tp_overlap=off/eager/lazy/bounded` 复用原有 `forward_tp_wavefront` / `PendingWavefrontOutputs` 调度；bounded 的窗口由 `tp_overlap_max_live_hooks` 指定，每个完整 step 后才能切换。它尚未接入原 MultiSAETrainer 的 DDP optimizer-overlap、完整日志/调度器体系，不支持动态 DP/ZeRO、AMP 或部分 GA 窗口。不能把这个独立入口的耗时直接当成旧 runner 的性能对照。已有训练数据应按原有方式先做 activation scaling，入口要求 `normalize_activations=none`。

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
