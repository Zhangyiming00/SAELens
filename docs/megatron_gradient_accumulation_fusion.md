# Megatron weight-gradient accumulation fusion

固定 Megatron runtime 默认设置 `sae_gradient_accumulation_fusion=True`。
CLI 用 `--no-sae-gradient-accumulation-fusion` 关闭，
`--sae-gradient-accumulation-fusion` 显式开启。
这个开关独立于 `gradient_accumulation_steps`：GA=1 也可以避免一次大梯度复制/相加。

原生 DDP 分配 `weight.main_grad` 后，encoder 的 `ColumnParallelLinear`
和 decoder 的 `RowParallelLinear` 都开启 `gradient_accumulation_fusion`。
原生 CUDA 扩展在 wgrad GEMM 内执行 `main_grad += dout.T @ input`；
TP wavefront 的局部 decoder 调用使用同一个开关。
standalone SAE 初始化时仍关闭，因为此时还没有 DDP gradient buffer。

实际启用条件：CUDA、native DDP gradient buffer、FP32 计算，且
`fused_weight_gradient_mlp_cuda` 可用。扩展不可用时警告并回退。
默认 DP1 direct-gradient fast path 没有 `main_grad`，保持关闭；
CPU fallback 和 AMP/autocast 也保持关闭。
当前 autocast 路径保存的 input 与 dout 可能分别为 FP32/BF16，
不满足原生融合扩展要求；不能仅设置配置位就假设支持 AMP。

## Decoder 的额外梯度

`rescale_acts_by_decoder_norm=True` 时，decoder 权重还参与 encode/decode
的 `weight.norm(dim=0)` 求导，且主重建和 auxiliary reconstruction 共用 decoder。
设置 fusion 后，原生 linear 会将 `param.grad_added_to_main_grad` 设为 True。
若直接让 DDP 跳过整个 `param.grad`，norm 分支的普通 autograd 梯度也会丢失。

因此在 rescale 开启时，同时使用 Megatron 原生的
`decoder.weight.zero_out_wgrad=True`：linear 返回全零 dummy wgrad，
真实 GEMM 梯度已累计到 main_grad；DDP 再将 norm 分支的残余梯度加入 main_grad。
encoder 不需要这个兼容处理。rescale 关闭时 decoder 也不需要它。
不要将 `grad_added_to_main_grad=True` 单独视为“所有 add_ 已消失”的证据。

不 detach norm、不改变 TopK/auxiliary loss 的求导，不改 GEMM 为低精度，
不启用 half optimizer 或 optimizer sharding。

## 同一次 forward 共享 decoder norm

encode、主 decode、aux decode 使用相同的 decoder 权重，现在复用一个可微的
norm 节点。三条分支先在长度为 `d_sae / TP` 的向量上累计梯度，然后只展开一次
decoder 矩阵梯度，保留全部 norm 导数。普通 forward 使用局部作用域；TP wavefront
把 norm 放在该次 forward 的 state 中，finish 时提供给 aux 分支。
不跨 microbatch/step 缓存；异常和嵌套调用会恢复原作用域。

H3、DP2、aux 开启时，norm backward 从每步 9 次降为 3 次；6 次大矩阵
gradient addition 变成向量 addition。原生 DDP 的 decoder `main_grad.add_`
仍为 3 次，native dummy 清零也仍保留，不能与这 6 次普通 autograd addition 混淆。

同配置配对实验：DP2 SAE step 210.36→187.87 ms（-10.69%），
TP2DP2 + wavefront 134.67→128.33 ms（-4.71%）。数据与 Nsight
见 `results/decoder_norm_20260918/README.md`。
这次主要节省重复计算；DP2 未覆盖的通信时间仅由 22.96 降到 22.28 ms。
实验性的“仅 encoder fusion”另省约 1.56 ms，但默认仍保留两块 linear 的原生 fusion。

## 2026-09-18 仅开启 fusion 的初始验证

结果与原始 trace 位于 `results/gradient_fusion_20260918/`。
配置沿用 H3、global batch 2048、4096→32768、K128、dense、FP32、
普通 fused Adam、optimizer overlap on。TP wavefront 开关开启，DP2 的 TP1 无可用 TP overlap。

数值测试覆盖 DP2、TP2DP2、TP2/DP1、GA1/3、rescale on/off、auxiliary loss、
AMP fallback，以及额外的非抵消 norm loss；比较梯度、参数、Adam 状态。

配对性能采用 off/on/on/off，每次 16 个 warmup + 30 个稳态 steps。
DP2 SAE step 平均 211.80 ms → 210.69 ms（约 0.52%），不能视为显著加速。
这是 cached-activation SAE step，不是包含 vLLM 生成的在线端到端耗时。

Nsight 独立采集 off/on 各一次，30 个稳态 steps，时间为 211.89 → 210.91 ms。
每步 512 MiB 的 DDP main_grad addition 从 6 次变成 3 次，
累计 GPU kernel 时间 6.46 → 3.42 ms；encoder 的 3 次全部消失，
decoder 的 3 次保留。native dummy wgrad 另外增加 6 次清零，抵消部分收益。
CPU AllReduce enqueue 提前，但 GPU 实际开始并非所有 hook 都提前。
因此原生 fusion 已默认启用，但“所有大 addition 消失”和“通信全面提前”未达成。
