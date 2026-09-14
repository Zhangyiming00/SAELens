# Megatron SAE v1 检查与静态训练器接入

基线：`f7d566a26e1a37db08102ce4b53dcf2bc6380229` + 上传的
`staged_changes9.14v1.patch`；依赖 Megatron Core 0.16.1。

## 检查结论

编码器 ColumnParallelLinear、解码器 RowParallelLinear 的参数布局、
全局 TopK、decoder norm 的计算轴以及 b_dec 的梯度方案相互一致。
编码器已对输入梯度做 TP 求和，因此 b_dec 不应再乘 1/TP 或在 backward
后重复求和。实际 Megatron TP1 模块已经与独立的原生 SAELens 参考对齐。
这不等于 TP2/TP4 或 NCCL 验收通过。

## 已修复问题

| 问题 | 影响 | 处理 |
| --- | --- | --- |
| TP1 普通 runner 未初始化 distributed | torchrun 只设置环境变量，模型构造仍会失败 | 在解析 TP 组时初始化 worker，复用现有组 |
| 多 SAE 的 Adam 保存名与加载名不同 | 原生 W_enc/W_dec/b_enc 被加载后静默跳过，续训重置动量 | 按磁盘原生名称与轴读取本地切片，再转换到 Megatron 名称与布局，TP1 同样转换 |
| 优化器切片加载用 torch.empty(shape) 推断维度 | 为大 SAE 临时分配完整 CPU 动量大小的张量 | 使用 meta 张量，只读取形状 |
| 上传补丁缺少参考 wheel、manifest 与 safetensors | 无法复现参考测试 | 补齐固定版本 wheel 和独立生成的 CPU 参考数据，记录哈希与生成环境 |
| 测试/诊断脚本调用已删除的 shard_weights/from_config_sharded | 测试收集或执行失败；旧验收脚本不能证明新模型正确 | 替换为新模型与真实训练器测试，迁移诊断和 profile 入口 |
| FSDP 配置仍能进入未适配的路径 | 首次裁剪失败，导出格式也未完成适配 | 在 runner 构造 TopK 时明确拒绝 FSDP |
| gather 依赖当前 CUDA device | 多设备调用时可能把接收缓冲区放错设备 | 使用带设备上下文的 Megatron gather 包装；TP1 直接返回输入 |

保留的普通 TopKTrainingSAE 服务于单进程库 API 和派生架构；训练 runner 的
TopK TP 只有 Megatron 实现。没有恢复旧 TP backend。

## 本轮后续实现

新增真实 SAETrainer 和 MultiSAETrainer 的静态验收入口：

- 单 SAE、两条独立 hook；per-hook DDP 和 unified MultiHook DDP。
- TP1×DP4、TP2×DP2、TP4×DP1。
- 固定全局 batch=11：DP4 分成 3/3/3/2，DP2 分成 6/5，按 token 加权。
- 每条 SAE 单独裁剪；验证更新后权重和所有 Adam 状态。
- 第 3 步通过真实训练器保存，重建模型和优化器、从磁盘恢复，继续到第 6 步。
- 各 hook 使用不同的归一化/偏置配置，防止混用 hook 状态。

验收脚本：

```bash
.venv/bin/python scripts/validate_megatron_sae_static.py \
  --output results/megatron_v2_acceptance --with-trainers
```

脚本自行启动 4 个进程，外层无需 torchrun。output 必须是新目录。
它先在 CUDA 上重放上游参考，再运行 TP 模块和训练器矩阵，最后写 summary.json。
没有 4 张 CUDA 卡时会报错，不能用 skip 冒充通过。

## 已执行验证与限制

在 PyTorch 2.10.0+cpu、Megatron Core 0.16.1 环境执行：

```bash
python -m pytest -q \
  tests/saes/test_megatron_sae_boundaries.py \
  tests/saes/test_megatron_sae_native_reference.py \
  tests/saes/test_megatron_sae_trainers.py \
  tests/training/test_load_weights_tp.py
```

结果：28 passed，4 skipped（CUDA 测试）。
上游独立进程参考重放通过（4 种配置 × 6 步）；修改/新增的小型模块与脚本
通过 Ruff 检查，git diff --check 通过。

当前环境没有 GPU，且禁止本地 Gloo socket。CPU 数值测试使用实际 Megatron
模块和不通信的 singleton fake process group；参数切片测试用模拟 rank
验证真实文件切片。这些测试不证明跨进程通信正确。没有运行完整项目测试集。

## 下一阶段

1. 在目标服务器执行上述 4 卡验收，确认数值和续训一致。
2. 接入 Megatron DDP + Distributed Optimizer：完整处理 main_grad、归约、
   按 SAE 裁剪、参数 all-gather 和优化器检查点；当前 DP 仍为 PyTorch DDP。
3. 接回 hook placement、streaming、wavefront 与 optimizer overlap，分别验收。
4. 最后验收固定 TP 下的弹性 DP 连续状态迁移。当前 checkpoint/resume 测试
   只覆盖固定拓扑，不是 elastic 实现。
