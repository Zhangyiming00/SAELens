# Routing 的异步 SHM 传输

普通 routing 默认使用 `--routing-transport shm_async`。回到原有 NCCL P2P
和 SAE TP 广播路径，使用 `--routing-transport nccl`。Python runner 配置对应
`routing_transport="shm_async"|"nccl"`；`routing_shm_slots` / `--routing-shm-slots`
控制每条远端边的有界缓冲深度，默认 2，最小 1。

```bash
# 默认新路径
... --routing-transport shm_async --routing-shm-slots 2

# 原路径，用于兼容和对照
... --routing-transport nccl
```

这项设置独立于 `routing_dp_batch_mode=equal|exact`，也独立于 streaming 的配置。
SHM 要求所有 routing rank 位于同一共享 `/dev/shm` 命名空间；不满足时明确报错，
不静默更改模式。跨节点可显式使用 `nccl`。

每条 producer→SAE endpoint 边有一个固定大小的 SHM ring，多个远端 TP rank
共同读取。只打包该 endpoint 需要的 hook。生产者源 rank 如果也是消费者，
直接使用自己的 GPU 切片，包括源 rank 恰好是 SAE TP follower 的情况。
无远端读者的边不创建 ring、staging buffer 或传输线程。单来源本地 assembly
直接复用 tensor；多来源按原 routing 表顺序拼接。特殊 token 过滤仍在 assembly
之后执行，随后使用原有 mixing 和训练流程。

远端路径为 GPU→pinned host→SHM→pinned host→目标 GPU。D2H 在独立 CUDA
stream 排队，后台线程发布 SHM；读者在独立 stream 预取 H2D。后台没有
torch.distributed 调用，也没有 SAE TP activation broadcast。保留原来的
vLLM 生成线程、生产周期和 producer helper 控制协议；不会额外预生成未请求的
LLM batch，因此 dataset cursor、mixing RNG 和 checkpoint 语义保持不变。

每个 slot 带单调递增序号与读者 ACK。源端只有在全部远端读者已复制 SHM 数据后
才能覆写 slot；慢消费者会产生背压，不丢弃或替换数据。接收 GPU 队列有相同深度
上限。CPU worker 异常写入该作业私有的错误记录，并在调用线程传播；等待有超时。
关闭时停止并 join worker，完成已发出的 D2H，再释放映射和本作业拥有的文件。

CUDA event 按源端 staging slot 复用，源 GPU tensor 由提交线程保留到 D2H 完成，
再在该线程复用 slot 时释放。这样避免后台 CUDA 对象析构与训练线程等待 collective
期间的 CUDA 驱动锁形成循环等待。H2D 完成后才把输入交给消费者，并登记消费 stream。

新增内存随远端边的 payload 大小和 slots 增长：包括 SHM ring、发送 pinned staging、
接收 pinned staging、有界 GPU 预取，以及尚待安全释放的源 tensor。它们不属于之前
SAE-only profiler 的峰值。异步描述的是传输的执行方式，不承诺所有拓扑都比 NCCL 快。

正确性验证包括有界多读者 ring、错误传播、配置检查，以及四卡训练/恢复和
多生产者、非整除 fan-in/fan-out、TP、PP、非连续 rank、BF16 特殊 token 过滤对照。
真实 vLLM runner 验证入口为 `scripts/validate_sae_runner_static.py`，同样接受
`--routing-transport`；比较不同路径时应固定 batch 模式、输入、seed 和并行配置。

四张 RTX 5090、vLLM TP4、H=3、d_sae=65536、全局 batch=4096 的短跑中，
SHM 相比 NCCL 的吞吐下降：SAE TP4 为 17.7%～22.8%，DP4 为 4.0%～5.6%，
PP3 为 4.3%～6.4%，覆盖无 Aux 和三个历史 dead 数量状态。
nsys 显示额外 CPU payload copy 延迟了远端 rank 参与 mixing buffer 就绪计数 AllReduce；
后台传输没有在该生产周期下消除关键路径上的等待。
详见 [完整性能报告](../results/routing_perf_20260928/README.md)。
复现入口为 `scripts/profile/profile_routing_transport.py`，支持固定 dead 状态的
短时真实训练以及独立 nsys 捕获窗口；不应把这些 routing 数值直接用于 streaming。
