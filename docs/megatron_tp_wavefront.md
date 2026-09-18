# Megatron 跨 hook TP wavefront

固定 TopK routing（缓存、静态在线和固定 SHM streaming 的公共训练单元）沿用
旧非 Megatron 路径的开关：

```bash
# 开启跨 hook TP forward overlap
--multi-sae-distributed-architecture unified_multi_hook

# 关闭（默认）
--multi-sae-distributed-architecture legacy_per_hook_wrapper
```

与旧路径相同，`unified_multi_hook` 要求配置
`--multi-sae-backward-mode combined`（CLI 默认值）；如果原命令显式选择了
`sequential`，开启时也需改回 `combined`。Megatron runtime 内部仍按独立
unit 的固定顺序 backward，以保留 native reducer 与逐 hook 更新的所有权。

无需新增 TP overlap 开关。runner 保留用户选择并传到 trainer；启动日志打印
`SAE TP wavefront requested_architecture=... effective=... reason=...`。
CUDA、TP>1 且同一 placement 有至少两个 hook 时启用；TP1、单本地 hook、
不支持的模型/设备回退串行。是否启用在 placement 的 TP×DP 域内取交集，
避免不同 rank 采用不同 collective 序列。

`--multi-sae-tp-phase-fence auto|always|off` 和 `SAE_TP_PHASE_FENCE` 环境覆盖
保留原语义。`auto` 在存在独立 DP communicator 或共置 producer 时，于整个
wavefront 后、backward 前做一次阶段同步；不在相邻 hook 间同步。
`always` 强制该边界，`off` 关闭边界。TP forward wavefront 本身不要求
`NCCL_LAUNCH_ORDER_IMPLICIT=1`；提前 DP bucket 归约和多 rank optimizer overlap
仍遵守其原有 NCCL 版本及启动环境要求。

## 调度和原生语义

旧 MultiHookSAE 和 Megatron runtime 调用同一个 `forward_tp_wavefront`：

```text
encode A → launch AG A
encode B → launch AG B → wait AG A → decode A → launch AR A
encode C → launch AG C → wait AG B → decode B → launch AR B
wait AG C → decode C → launch AR C
wait AR A/B/C → 各 hook 的重构后处理及 loss
```

相邻 hook 的本地计算可覆盖上一 hook 的 TP 通信。实现使用单独的 TP CUDA
stream、完成 event 和 allocator `record_stream` 依赖，不在每次 collective
后阻塞计算 stream。Megatron 自己的 gather/reduce autograd 保持不变；没有
自定义 collective backward。encoder 仍调用 ColumnParallelLinear。
decoder 使用已安装 Megatron 0.16.1 RowParallelLinear 的 `_forward_impl`
计算本地结果，仅把尾部的原生 TP reduce 放到通信 stream；不复制 Linear
backward、不改权重布局或安装的 Megatron 源码。升级 Megatron 时需回归这个
版本相关接口。

全局 TopK、稀疏返回、decoder norm rescale、辅助重构、input/bias 梯度语义
与串行路径一致。辅助重构仍走普通 native decoder。每个 microbatch 完成
全部 phased forward 后，按固定 hook 顺序 backward；图不跨 GA microbatch
保留。开启时会同时持有本 placement 的多 hook forward 图，显存可能增加。

每 hook 的 native DDP/bucket、裁剪范数、optimizer、scheduler、AMP 及状态
所有权保持独立。runtime 不构造新的统一参数 owner；内部 checkpoint 的
ownership 字段仍为 `legacy_per_hook_wrapper`，runner config 保存所请求的
forward 开关。已有同拓扑 checkpoint 可继续使用。

`--multi-sae-optimizer-overlap` 控制的是另一个阶段，可以独立开关；开启 TP
wavefront 后，它覆盖后续 hook 的 backward，而后续 forward 已在 TP 阶段
提交。已有参数 gather schedules 也保持原定义。PP/SPP 只在本 placement
内调度，例如 H3/PP2 的双 hook placement 可开启，单 hook placement 回退。

## 验证

`tests/training/test_megatron_tp_wavefront.py` 验证真实 TP2、TP4、TP2DP2、
TP2SPP2、TP1 回退；普通/分片、optimizer overlap on/off、GA1/3、AMP、
不等 token、空 replica/hook/window、overflow、短尾窗口、磁盘恢复及后续更新。
独立模型对照包含辅助损失、稀疏激活、normalization、rescale、bias 开关和
输入梯度。另有无 implicit ordering 的 TP2DP2 验证和 TP 通信在途故障注入。

```bash
SAE_TP_WAVEFRONT_REPORT_DIR=results/tp_wavefront_check \
  .venv/bin/pytest -q tests/training/test_megatron_tp_wavefront.py
```

测试使用四张 CUDA GPU。已执行结果及真实 runner/短 trace 记录见
[`results/tp_wavefront_20260918`](../results/tp_wavefront_20260918/README.md)。
该功能不增加动态弹性、跨拓扑 reshard 或 Megatron pipeline parallel 支持。
