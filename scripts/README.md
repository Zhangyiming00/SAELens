# 脚本入口

从仓库根目录运行，Python 使用 `.venv/bin/python`。GPU 实验需显式选择输出目录；查看参数使用 `--help`。

| 用途 | 入口 |
|---|---|
| 当前 native 时间 / allocated 峰值内存插值 | `profile/simulate_megatron_execution_time.py` |
| 插值校准、冻结、独立 holdout | `profile/profile_current_interpolation.py`、`profile/report_current_interpolation.py` |
| 真实 SAE-only 计时及 Nsight 采集 | `profile/profile_megatron_execution_time_v2.py` |
| 稀疏 kernel profile、细化及扩展网格 | `profile/profile_sparse_parts.py`、`profile/refine_sparse_parts_profile.py`、`profile/extend_sparse_profile.py` |
| native / sparse Nsight 归因 | `profile/analyze_megatron_execution_trace.py`、`profile/analyze_megatron_execution_trace_v2.py`、`profile/analyze_sparse_parts_trace.py` |
| 存储 / 主项 / AuxK 策略实测矩阵 | `profile/report_execution_policy_grid.py` |
| routing SHM / NCCL 性能与归因 | `profile/profile_routing_transport.py`、`profile/summarize_routing_transport.py`、`profile/analyze_routing_trace.py` |
| routing 训练、checkpoint 与 transport 等价性 | `validate_sae_runner_static.py`、`validate_sae_routing_runtime.py`、`compare_routing_transports.py` |
| native 数值验收 | `validate_megatron_sae_static.py` |
| 在线静态基线与 GA=1 Nsight | `benchmark_sae_runner_static.py`、`profile_sae_ga1.py`、`analyze_sae_ga1_trace.py` |
| 静态内存因子、在线 buffer / vLLM 内存、allocator replay | `profile/profile_static_memory_factors.py`、`profile/profile_static_online_memory.py`、`profile/analyze_static_memory_factors.py`、`profile/replay_allocated_memory.py` |
| Megatron 内存矩阵及报告 | `profile_megatron_sae_memory.py`、`analyze_megatron_sae_memory.py` |
| elastic 正确性、watermark 实测与模拟 | `profile/validate_elastic_megatron.py`、`profile/elastic_memory_probe.py`、`profile/profile_elastic_watermarks.py`、`profile/report_elastic_watermarks.py`、`profile/simulate_elastic_watermarks.py` |
| 在线 elastic 控制 | `elastic_streaming_control.py` |
| 拓扑进程重启控制 | `topology_supervisor.py`、`demo_topology_switch.py` |
| 激活缓存 | `run_cache_activations_runner_gpu.py`、`cache_extra_hooks_aligned.py`、`merge_extra_hooks_into_cache.py` |
| vLLM capture / KV 内存 | `profile_hooked_vllm_memory.py`、`verify_minimized_kv_pool.py`、`aggregate_kv_cases.py` |

完整的当前插值流程见 [current_interpolation.md](../docs/current_interpolation.md)，routing 说明见 [routing_async_shm.md](../docs/routing_async_shm.md)。主训练入口是仓库根目录的 `run_sae_runner_gpu.py`。

`profile_megatron_execution_time_v2.py` 仍是当前采集实现，名称中的 v2 不代表已过期。两个 native trace analyzer 分别负责 GPU 活动归因和 CPU/stream 审计，后者复用前者。

`predict_sae_memory.py` 是旧 v4 phase/driver 内存估算器，仅保留历史回归用途，不能代替当前 native 插值模型。`profile_sae_phase_v5.py` 仍被 Megatron 内存对照流程记录源码指纹，因此一并保留。通用训练示例、Ansible 配置、数据工具和 CI 使用的 `huggingface_sae_sync.py` 也保留。

2026-09-29 清理了旧 step-window/gap 实验链路、旧绘图/扫描脚本和已被替代的校准入口。逐文件删除原因见本地 [清理记录](../results/scripts_cleanup_20260929/README.md)；历史实验数据保留在 `results*` 中。
