# 静态 SAE routing 与独立训练单元

这一入口统一了 SAE 通信组来源，并为每个 hook 提供独立的模型、DDP reducer、
梯度裁剪和 Adam 状态。保留原有 activation slicing、过滤、buffer、shuffle、
token 加权和 SAE loss；本阶段不实现异步 routing、streaming/elastic 调度，
也不实现 Megatron DDP、Distributed Optimizer 或跨 hook optimizer overlap。

## 训练域初始化

Megatron Core 固定为 0.16.1。其 `initialize_model_parallel()` 没有父进程组参数，
始终读取默认 world。因此不能先 `dist.new_group([0, 1])` 再直接调用该全局初始化器。

`sae_lens.sae_runtime.SAERuntime` 采用显式 process-group collection 路径：

1. 运行时根据配置建立 `SAETrainingDomain`，指定训练 rank、TP 大小和 hook。
2. 使用 Megatron `parallel_state.RankGenerator` 在域内划分 TP/DP；使用 Megatron
   `create_group` 建组。CP、EP、Megatron PP 均为 1。
3. 构造 `ProcessGroupCollection`，由模型和训练器持有同一个运行时；以后接入
   Megatron DDP/optimizer 时传入这一 collection。
4. TP root 从实际 process group 查询全局成员、组内 rank 和接收 rank，再向
   routing world 交换 endpoint metadata。producer-only rank 不属于 SAE TP/DP。

不替换默认 world、不修改 `torch.distributed`，也不修改 Megatron 全局模型并行
状态。当前适配针对固定拓扑、互不重叠的训练域；同设备上的多个 hook 放在同一个域。
默认布局仍由配置指定，支持显式非连续 rank 域。

例如 SAE TP2×DP2：

| 副本 | TP 成员 | 接收 rank |
| --- | --- | --- |
| 0 | 0、1 | 0 |
| 1 | 2、3 | 2 |

两个参数分片对应的 DP 组为 `{0, 2}` 和 `{1, 3}`。routing 读取运行时报告，
将不同数据送到两个 root；原有 TP broadcast 将同一副本的输入复制给 TP follower。

## 入口与所有权

`distributed_v2.initialize_sae_routing(...) -> SAERuntime` 是新的静态入口。
所有 routing-world rank 按相同顺序进入初始化；`close_sae_routing()` 释放本入口的
transport/SAE 组，保留默认 world。实际静态 runner 只使用这一入口，
拒绝 `use_shard_routing=False`；不再进入旧 prefix-overlap 建组、统一多 hook
模型或按行数预估生成次数的 helper 训练分支。旧模块仍供尚未迁移的 streaming
调用及文件兼容使用，不属于静态生产入口。

```python
runtime = initialize_sae_routing(
    P=1, Q=2, vllm_tp_size=4, sae_tp_size=2,
    batch_size=11,
    hook_names=("blocks.21.hook_resid_post", "blocks.31.hook_resid_post"),
)
```

可以传入 `training_domains` 明确指定子域，例如两个 hook 分别放在 `{0, 1}` 和
`{2, 3}`。这些是两个独立 SAE，没有 SAE 输出互传，也没有串行 pipeline 调度。
当前所有 placement 使用相同 DP 大小，以复用现有 row routing 和数据步进协议。

`LanguageModelSAETrainingRunner` 的静态 shard-routing 路径会自动使用此入口，
并将运行时传给 `MegatronTopKSAE`、`SAETrainer` / `MultiSAETrainer`。
当前静态训练入口支持 `TopKTrainingSAEConfig`；每个 hook 的模型配置和种子沿用 runner。
`MultiSAETrainer.units[hook]` 和单 SAE 的 `SAETrainer.unit` 暴露独立训练单元。
可通过 runner 的 `sae_training_domains` 传入显式 placement。单进程 TopK 也自动
建立 runtime；runner 仅销毁自己初始化的默认 world。初始化失败、训练失败和正常
返回都会释放已有的 SAE/transport 组及自己预初始化的 vLLM 组，`close()` 可重复调用。

producer-only rank 等待同步的 `batch`、`checkpoint`、`finish` 控制消息。
消费者需要补 buffer 时才请求生成，因此过滤后的行数及恢复的待消费 buffer 不会
导致 helper 提前退出。控制端取自 runtime 报告的首个接收 root。

同步顺序是：按 hook 顺序 forward/loss、按相同顺序 backward、完成梯度同步、
逐 SAE 裁剪、逐 SAE 更新。空 DP 副本使用零权重 dummy backward，并参与相同的
统计通信；dummy 激活不计入 firing/token 统计。
稀疏空副本直接构造零 firing/count 向量，不切片 COO 张量。全局空步完成 backward
和 TP 同步后清空梯度，跳过 Adam、LR scheduler 及 GradScaler 的更新；下一正常步
可以正常 `unscale_()`。runtime 下训练进度按 DP 的实际总 token 数累计，避免一个
空副本因进度落后多进入训练循环。

不同 hook placement 共用检查点根目录。运行时保存全体 SAE rank 的训练组，
发布 checkpoint 完成标记前等待各 placement 写完；producer-only rank 不参与
这个完成屏障，其状态保存另有同步确认。完成标记在 runner 的数据状态保存后发布。
根 manifest 包含所有 placement 的 hook，各 hook 的状态文件独立。

`SAETrainUnit` 持有自己的模型、DDP wrapper、Adam 和 runtime 引用。
`UnitOptimizers` 仅向现有 scheduler/scaler/checkpoint 代码提供兼容视图，
不持有额外 Adam 状态。各 hook 的 optimizer state 和梯度存储独立，裁剪不跨 hook。
每个训练单元有自己的 LR scheduler，参数组按参数名保存、原位恢复全部选项，包含
当前 LR、initial LR、betas 和 eps；同时恢复 scheduler 和 GradScaler 状态。
旧命名 checkpoint 缺少参数组时，从 scheduler 恢复当前 LR，其余缺失选项沿用配置。

固定拓扑沿用原有模型与 Adam moments 文件，并新增每 rank 的
`activation_stream_rankN.pt`：保存未消费的 serving buffer、保留的混合 buffer、
shuffle RNG 和精确序列位置（缓存模式为缓存行位置）。恢复不会丢弃已生成的激活。
真实 token 的排除标记随路由 payload 传输，以精确的 0/1 列表示，接收端在进入混合
buffer 前过滤；不会把词表 ID 转成低精度浮点数。

静态 `routing_dp_batch_mode="exact"` 支持在相同拓扑与 batch 配置下续训。
例如 global batch 4096、DP3 的本地 batch 为 1365/1365/1366。各 rank 保存
`trainer_runtime_rankN.pt`，恢复自己的 token 进度与计数余数；checkpoint 目录名
和触发阈值统一使用 global token 数，runner 配置也保存原始 global batch。
exact 模式每次 buffer 补充后对可提供的训练批次数取全体 SAE rank 的最小值，
保证本地 batch 大小或过滤行数不同的副本仍同时进入下一次静态路由。
shuffle 和保留区的算法不变，暂时多出的激活保留在本地 buffer。

每个 placement 的 DP0/TP0 另存 `placement_state_N.pt`，包含该 placement
的 GradScaler 状态。恢复不再用 placement 0 的 scaler 覆盖其他 placement。
每个 placement 内的多个 hook 仍共享该训练器的 scaler；这不是逐 hook AMP 调度。
缺少 placement AMP 状态的旧多 placement AMP checkpoint 会明确拒绝恢复。

静态 `run()` 使用独立于训练 collective 的 rendezvous Store 传递首次失败原因、
producer 命令及保存确认。每个 rank 的监控线程收到失败后中止 runner 自己创建的
NCCL 通信组，以及归属这些组的 vLLM PyNccl communicator；等待 producer 命令
或保存确认的线程会检查相同失败记录并抛错。
每个 hook 的 backward 返回后，同一 placement 的 TP/DP 成员通过 Store 确认
均已返回，再进入后续 hook 或梯度处理；避免一个成员抛出 autograd 异常时，
其他成员仍继续提交 CUDA 更新。该同步边界不包含 producer-only rank，
不同 placement 不为此互相等待。vLLM 通信入口和训练单元在失败后拒绝继续通信。
成功结束使用可检查失败的完成握手。此机制处理 run 阶段捕获到的 Python 异常，
不提供进程被 SIGKILL、节点失联、初始化中途失效或硬件故障后的弹性恢复保证。

CLI 默认使用每 hook wrapper，`--multi-sae-optimizer-overlap` 默认为 `off`。
新的静态路径拒绝 FSDP、ZeRO 和 optimizer overlap。已有 streaming 入口保留其
原有调度；本次没有将它迁移为训练单元调度，也不声明其 elastic 行为已经验收。

## 实际 runner 验收

```bash
.venv/bin/python scripts/validate_sae_runner_static.py --output results/runner_new
.venv/bin/python scripts/validate_sae_runner_static.py --output results/runner_empty_new --empty-steps
.venv/bin/python scripts/validate_sae_runner_static.py --output results/runner_prefix_new --layout prefix --hooks 2
.venv/bin/python scripts/validate_sae_runner_static.py --output results/runner_placement_new --layout placement --hooks 2
```

均要求四卡和新输出目录，内部启动进程；`--model` 默认为本机
`/root/models/Llama-3.1-8B`。使用固定本地 token 数据集，经真实 vLLM prefill/激活提取、
token 过滤、routing、buffer/shuffle 和 runner 训练循环。H=1/H=2 的 TP2×DP2、
部分训练域、独立 placement 都比较第 3 步恢复后的输入、学习率、最终参数、Adam 和
scheduler 状态。`--empty-steps` 在真实 mixer 输出之后注入一个空副本及两个全局空步，
启用 autocast/GradScaler，检查空步不改变 Adam/scaler，以及后续正常更新与 DP 一致性。
空批次是受控注入，未声称由自然文本过滤恰好产生。该小规模功能验收不衡量吞吐。

新增边界验收命令：

```bash
.venv/bin/python scripts/validate_sae_runner_static.py --output results/exact_dp3_new --layout dp3 --batch-mode exact --global-batch 4096
.venv/bin/python scripts/validate_sae_runner_static.py --output results/placement_amp_new --layout placement --hooks 2 --placement-amp
.venv/bin/python scripts/validate_sae_runner_static.py --output results/failure_new --layout prefix --hooks 2 --failure consumer_backward
```

故障点还可选 `consumer_wait`、`producer_generate`。验收要求所有进程自行报告
失败原因、释放 owned groups 并在时限内退出；launcher 超时强制回收判定为失败。
每次运行保存 `invocation.json` 中的配置、设备及源码 SHA256。
报告以 `routing_dp_batch_mode` 明确区分 equal/exact，以
`checkpoint_state_bitwise_equal`、`resume_bitwise_equal` 和最大数值误差分别记录
恢复瞬间状态与后续更新，不再使用有歧义的 `exact_resume` 字段。DP3 后续更新
允许 `rtol=1e-6, atol=1e-7` 的浮点误差，输入、LR、scaler 与恢复瞬间状态仍要求
逐位一致；这些数值验收不能替代性能测试。

## 原生参考与拓扑回归

```bash
.venv/bin/python scripts/validate_sae_routing_runtime.py \
  --output results/sae_routing_runtime_acceptance_new
```

输出必须为新目录。脚本自行启动四个 worker，外层不使用 torchrun。
依次执行上游独立 CUDA 参考重放、真实 Megatron 建组和静态 routing/trainer 验收，
成功后写入 `summary.json` 及每 rank 报告。

覆盖 H=1/H=2、TP1×DP4 / TP2×DP2 / TP4×DP1、仅部分 rank 训练、非连续 rank、
独立 hook placement、不同 hook 数据内容、空 DP 副本以及第 3 步磁盘保存后续训到
第 6 步。检查每步权重、重建和全部 Adam 状态。

数据源为固定合成激活；数据切片、NCCL P2P、组装、TP broadcast 和 SAE 训练器均
使用生产代码。该验收没有加载大语言模型，也不替代完整 vLLM 生成任务的吞吐验收。
