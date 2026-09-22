# Megatron SAE：用分布式 TopK 避免完整 hidden_pre gather

这是基于当前源码的实现方案，尚未改变生产训练路径。

当前 `MegatronTopKSAE.encode_with_hidden_pre` 在局部 encoder 后 all-gather
完整 `[B,F]`；dense TopK 又创建同形状输出。wavefront 路径有相同布局。
decoder 只使用 `[B,F/T]`，但完整 storage 仍被 forward state、autograd
或 `TrainStepOutput` 引用。`detach()` 不释放这份 storage。

本次 H3/I4096/F16384 的实测中，B_eff16384、TP4、GA1 的 allocated
峰值为 12.713 GiB。该峰值的分配栈归因显示，三个 hook 的 gather 输出
共 3 GiB、dense TopK 输出另 3 GiB；forward/保留输出来源合计约
10.829 GiB。这是当前负载里值得针对的内存项，不等于全部都能直接节省。

## 第一版方案：分片 dense 激活，继续使用原生 Megatron Linear

设本地 encoder 输出 `z_r: [B,F/T]`，局部 decoder norm 为 `n_r`。

1. 按原顺序计算 `s_r = z_r * n_r`，若 rescale 关闭则直接使用 `z_r`。
2. 对 `s_r` 求 `min(K,F/T)` 个局部候选，形成分数和全局 feature 索引。
3. all-gather 候选分数及索引，得到最多 `[B,T*K]`。只将候选通信视作
   **选索引的元数据**，不让它成为被选特征值的梯度路径。
4. 各 rank 对候选集合做一致的全局 TopK，得到赢家全局索引。
5. 每个 rank 从本地原始 `s_r` 取出属于自己的赢家，按原顺序应用 ReLU，
   scatter 成局部 dense `a_r: [B,F/T]`。没有本地赢家的 token 保持零值，
   仍参加相同的 collective 序列。
6. 按原实现计算 `a_r / n_r`，交给现有 RowParallelLinear；继续 all-reduce
   `[B,I]` 重构结果，保留 b_dec、输入梯度和 loss 的现有 TP 语义。

候选充分性：如果某个特征在本分片中严格排在第 K 名以后，本分片内就至少有
K 个特征优于它，因此它不可能是全局前 K。并列分数需要各级使用同一个全序。

梯度通过“本地原始分数 → local gather/scatter → decoder”传播，未选特征
梯度为零，和普通 TopK 的分段导数相同。不要把 detached 的候选值直接填到
decoder 输入，那会截断 encoder 和 decoder norm 的梯度。norm 仍须共享
当前 forward 的可微节点，以保留 rescale 的各条梯度分支。

该方案仍使用局部 dense GEMM；无需第一步就实现稀疏 GEMM、稀疏 wgrad
或改动 optimizer。主要 feature storage 从 `[B,F]` 变为 `[B,F/T]`，
仍存在本地反向临时张量、梯度桶和通信 workspace，不能承诺总峰值精确 `/T`。

## 通信与显存量级

完整 FP32 gather 的每 rank 输出 payload 是 `4*B*F` 字节。
候选收集是至多 `B*T*K*(4+index_bytes)` 字节；包含索引，不能只按分数数量
比值宣传通信节省，实际传输量还取决于 collective 算法。

F=65536、T=4、K=128 时，每 token 收集 512 个候选。使用 int64 索引时
候选输出 payload 约为完整 FP32 分数的 1/42.7；用合法范围内的 int32
索引约为 1/64。局部候选与临时排序 workspace 另计。

K 较大、TP 较大或 F/T 很小时，候选方案收益减小。可令局部候选数为
`min(K,F/T)`，并基于 payload 选择是否回退完整 gather；所有 rank 必须
根据相同配置选择同一路径。

## 必须同时调整的消费端

- **训练输出**：不能继续让 `hidden_pre` 和 `feature_acts` 默认代表完整
  全局 feature 轴。使用明确的分片类型或携带 feature offset/global width；
  普通训练不重建 `[B,F]`。需要完整表示的调试、导出或自定义 hook 再显式 gather。
- **firing/dead-feature 统计**：本地 `[F/T]` feature 在 DP 组归约；TP 只交换
  必需的标量或 `[F]` 统计向量，避免交换 `[B,F]`。L0 等指标可由各 TP 分片
  求和获得。模型及统计 checkpoint 保持既有全局 feature 编号与重分片协议。
- **辅助重构**：对本地 dead mask 下的分数独立做 candidate TopK，
  `K_aux=min(I//2, global_num_dead)`，全局 dead 数由 TP 标量归约得到。
  本分片 dead 数不足时按一致方式处理无效候选。原辅助分支没有普通 TopK 的
  ReLU，必须保留负值语义，不能直接套用普通激活函数。
- **wavefront**：encode launch 从完整 feature gather 改为候选 gather；
  decode launch 等待候选并生成局部激活。保留 stream/event、record_stream、
  failure monitor 和 TP/DP collective 启动顺序。

GA 期间，当前训练只保留每 hook 最近一份 detached 输出，但它仍持有完整
storage。即使只做提前释放，也需审计日志、loss 汇总、firing 和用户 hook
是否还需要该字段；这可以是独立的小优化，不能代替上述 feature 分片。

## 并列分数与验收

旧 `torch.topk(sorted=False)` 不提供稳定的并列索引契约，分布式候选排序
不能自然保证复现旧 kernel 对所有相同分数的选择。应明确约定
`(分数降序, 全局 feature index 升序)` 等确定性规则，或明示并列处与旧实现
只保证 TopK 条件、不保证相同特征选择；并列处选择不同可能改变重构，不能
仅因分数一样就宣称数值相同。

实现时建议新增可选路径，保留旧路径作为 oracle，分两层验收：

1. 独立 TopK 比较：无并列时的赢家、激活和梯度；并列、全零、负值、
   K 大于局部分片宽度、极端候选分布；候选选择顺序必须在所有 rank 一致。
2. 完整训练比较：TP1/2/4、TP2DP2、GA1/2/4，主 loss 与辅助 loss、
   rescale 开关、encoder/decoder/bias 梯度、裁剪、Adam 状态、更新参数、
   feature 年龄、空 batch/尾窗口、checkpoint 恢复、wavefront 与 overlap。

随后用相同缓存和 memory pickle 对比实际 allocated/reserved/设备占用，
再单独进行不带 memory tracing 的吞吐测试。不要用显存理论降幅替代实测。
