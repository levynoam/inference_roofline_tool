# Inference Performance Guide

A comprehensive guide to using the `inference_performance.py` module for calculating LLM inference resource requirements.

## Overview

The `InferencePerformance` class calculates compute, memory, and network requirements for LLM inference, supporting various parallelism strategies. It models two key phases:
- **Prefill (TTFT)**: Processing the input prompt to generate the first token
- **Decode**: Generating subsequent tokens one at a time

## Core Classes

### `InferencePerformance`

The main calculator class for inference performance analysis.

**Constructor:**
```python
InferencePerformance(model: LLMArchitecture)
```

**Key Methods:**

#### 1. `calculate_achievable_ttft()`
Calculates the achievable Time-To-First-Token given system constraints.

**Parameters:**
- `system_constraints`: `SystemConstraints` - GPU specifications (memory, bandwidth, compute)
- `batch_size`: `int` - Number of sequences to process
- `sequence_length`: `int` - Length of input prompts
- `parallelism_config`: `Optional[ParallelismConfig]` - Parallelism strategy (None for single GPU)
- `kernel_launch_latency`: `float` - Kernel launch overhead in seconds (default: 5e-6)
- `dtype_override`: `Optional[str]` - Override model dtype ("float32", "float16", "bfloat16", "int8", "int4")

**Returns:** `ResourceUtilization` containing:
- `achievable_ttft`: Time to first token in seconds
- `bottleneck`: Which resource limits performance ("compute", "memory_bandwidth", "network_bandwidth")
- `compute_utilization`: Fraction of compute capacity used (0-1)
- `memory_bandwidth_utilization`: Fraction of memory bandwidth used (0-1)
- `network_bandwidth_utilization`: Fraction of network bandwidth used (0-1)
- `memory_utilization`: Fraction of memory capacity used (0-1)
- `prefill_resources`: Detailed resource breakdown

**Example:**
```python
from llm_configs import LLAMA_3_8B
from inference_performance import InferencePerformance, SystemConstraints

perf = InferencePerformance(LLAMA_3_8B)
gpu = SystemConstraints.from_gpu_spec("H100")

result = perf.calculate_achievable_ttft(
    system_constraints=gpu,
    batch_size=16,
    sequence_length=4096,
    parallelism_config=None  # Single GPU
)

print(f"TTFT: {result.achievable_ttft * 1000:.2f} ms")
print(f"Bottleneck: {result.bottleneck}")
print(f"Compute Utilization: {result.compute_utilization * 100:.1f}%")
```

#### 2. `calculate_decode_performance()`
Calculates decode phase performance including TPS and throughput.

**Parameters:**
- `system_constraints`: `SystemConstraints` - GPU specifications
- `batch_size`: `int` - Number of sequences
- `prefill_length`: `int` - Length of prefill context
- `decode_steps`: `int` - Number of decode steps to analyze
- `parallelism_config`: `Optional[ParallelismConfig]` - Parallelism strategy
- `kernel_launch_latency`: `float` - Kernel launch overhead in seconds
- `dtype_override`: `Optional[str]` - Override model dtype
- `return_step_details`: `bool` - Return per-step resource breakdown (default: False)

**Returns:** `DecodePerformance` containing:
- `tps`: Tokens per second per user
- `total_throughput`: Total tokens per second across all users
- `step_time`: Time per decode step in seconds
- `bottleneck`: Which resource limits performance
- Utilization metrics (compute, memory bandwidth, network bandwidth, memory)
- `step_details`: List of `DecodeStepResources` if `return_step_details=True`

**Example:**
```python
result = perf.calculate_decode_performance(
    system_constraints=gpu,
    batch_size=32,
    prefill_length=2048,
    decode_steps=512,
    parallelism_config=None,
    return_step_details=True
)

print(f"TPS per user: {result.tps:.2f}")
print(f"Total throughput: {result.total_throughput:.2f} tokens/sec")
print(f"Step time: {result.step_time * 1000:.2f} ms")
print(f"Bottleneck: {result.bottleneck}")
```

### `SystemConstraints`

Defines GPU hardware specifications.

**Constructor:**
```python
SystemConstraints(
    memory_capacity: float,                # Bytes
    memory_bandwidth: float,               # Bytes/sec
    compute_throughput: float,             # FLOPS
    network_bandwidth: float,              # Bytes/sec
    persistent_storage_bandwidth: float    # Bytes/sec (default: 20 GB/s)
)
```

**Factory Method:**
```python
SystemConstraints.from_gpu_spec(gpu_name: str)
```

Supported GPUs: "H100", "A100", "MI300X", "B200", "GB200"

**Example:**
```python
# Using factory method (includes default 20 GB/s storage bandwidth)
gpu = SystemConstraints.from_gpu_spec("H100")

# Manual specification
custom_gpu = SystemConstraints(
    memory_capacity=80e9,                      # 80 GB
    memory_bandwidth=3.35e12,                  # 3.35 TB/s
    compute_throughput=1979e12,                # 1979 TFLOPS
    network_bandwidth=900e9,                   # 900 GB/s
    persistent_storage_bandwidth=20e9          # 20 GB/s (NVMe SSD)
)
```

### `ParallelismConfig`

Configures distributed execution strategy.

**Constructor:**
```python
ParallelismConfig(
    parallelism_type: ParallelismType,
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1
)
```

**Parallelism Types:**
- `ParallelismType.NONE`: Single GPU
- `ParallelismType.DATA_PARALLEL`: Data parallelism (DP)
- `ParallelismType.TENSOR_PARALLEL`: Tensor parallelism (TP)
- `ParallelismType.PIPELINE_PARALLEL`: Pipeline parallelism (PP)

**Example:**
```python
from inference_performance import ParallelismConfig, ParallelismType

# 4-way tensor parallelism
tp_config = ParallelismConfig(
    parallelism_type=ParallelismType.TENSOR_PARALLEL,
    tensor_parallel_size=4
)

# 2-way tensor + 4-way pipeline = 8 GPUs
hybrid_config = ParallelismConfig(
    parallelism_type=ParallelismType.TENSOR_PARALLEL,
    tensor_parallel_size=2,
    pipeline_parallel_size=4
)

total_gpus = hybrid_config.total_gpus  # Returns 8
```

### `PrefillResources`

Contains detailed resource breakdown for prefill phase.

**Attributes:**
- `memory_per_gpu`: Total memory required per GPU (bytes)
  - `memory_model_weights`: Model parameter memory
  - `memory_kv_cache`: KV cache memory
  - `memory_activations`: Activation memory
- `memory_bandwidth_per_gpu`: Required memory bandwidth (bytes/sec)
- `memory_bandwidth_used`: Actual memory bandwidth consumed (bytes/sec)
- `network_bandwidth_per_gpu`: Required network bandwidth (bytes/sec)
- `compute_per_gpu`: Total FLOPs per GPU
- `compute_throughput_required`: Required FLOP/s per GPU
- `arithmetic_intensity`: FLOPs per byte (compute/memory ratio)

### `DecodeStepResources`

Contains resource requirements for a single decode step.

**Attributes:**
- `context_length`: Current context length
- `memory_per_gpu`: Memory required
- `memory_bandwidth_per_gpu`: Memory bandwidth required
- `network_bandwidth_per_gpu`: Network bandwidth required
- `compute_per_gpu`: Compute required (FLOPs)
- `weights_memory_traffic`: Memory traffic for model weights (bytes)
- `kv_cache_memory_traffic`: Memory traffic for KV cache (bytes)
- `activations_memory_traffic`: Memory traffic for activations (bytes)

Access via `decode_result.step_details[i]` when `return_step_details=True`.

### `ResourceUtilization`

Result from TTFT calculation showing utilization metrics.

**Attributes:**
- `achievable_ttft`: Achievable time to first token (seconds)
- `bottleneck`: Limiting resource ("compute", "memory_bandwidth", "network_bandwidth", "storage_bandwidth")
- `compute_utilization`: Compute usage (0-1)
- `memory_bandwidth_utilization`: Memory bandwidth usage (0-1)
- `network_bandwidth_utilization`: Network bandwidth usage (0-1)
- `persistent_storage_bandwidth_utilization`: Storage bandwidth usage (0-1, MoE only)
- `memory_utilization`: Memory capacity usage (0-1)
- `prefill_resources`: PrefillResources object with detailed breakdown
- `kernel_overhead_fraction`: Fraction of time spent on kernel launches

### `DecodePerformance`

Result from decode calculation showing performance metrics.

**Attributes:**
- `tps`: Tokens per second per user
- `total_throughput`: Total tokens/sec across all users
- `step_time`: Time per decode step (seconds)
- `bottleneck`: Limiting resource
- Utilization metrics (compute, memory bandwidth, network bandwidth, memory)
- `kernel_overhead_fraction`: Fraction of time spent on kernel launches
- `step_details`: Optional list of DecodeStepResources

## Common Usage Patterns

### 1. Find Maximum Batch Size
```python
from llm_configs import LLAMA_3_70B
from inference_performance import InferencePerformance, SystemConstraints

perf = InferencePerformance(LLAMA_3_70B)
gpu = SystemConstraints.from_gpu_spec("H100")

for batch_size in [1, 2, 4, 8, 16, 32, 64]:
    result = perf.calculate_achievable_ttft(
        system_constraints=gpu,
        batch_size=batch_size,
        sequence_length=4096
    )
    if result.memory_utilization > 1.0:
        print(f"OOM at batch size {batch_size}")
        break
    print(f"Batch {batch_size}: TTFT={result.achievable_ttft*1000:.2f}ms, "
          f"Util={result.compute_utilization*100:.1f}%")
```

### 2. Compare Parallelism Strategies
```python
configs = {
    "1 GPU": None,
    "2-way TP": ParallelismConfig(ParallelismType.TENSOR_PARALLEL, 2),
    "4-way TP": ParallelismConfig(ParallelismType.TENSOR_PARALLEL, 4),
    "2x2 TP+PP": ParallelismConfig(ParallelismType.TENSOR_PARALLEL, 2, 2)
}

for name, config in configs.items():
    result = perf.calculate_decode_performance(
        system_constraints=gpu,
        batch_size=16,
        prefill_length=2048,
        decode_steps=512,
        parallelism_config=config
    )
    print(f"{name}: TPS={result.tps:.2f}, Throughput={result.total_throughput:.2f}")
```

### 3. Analyze Bandwidth Breakdown
```python
result = perf.calculate_decode_performance(
    system_constraints=gpu,
    batch_size=32,
    prefill_length=2048,
    decode_steps=100,
    return_step_details=True
)

# Examine step details
for i, step in enumerate(result.step_details[::10]):  # Every 10th step
    total_traffic = (step.weights_memory_traffic + 
                     step.kv_cache_memory_traffic + 
                     step.activations_memory_traffic)
    weights_pct = step.weights_memory_traffic / total_traffic * 100
    kv_pct = step.kv_cache_memory_traffic / total_traffic * 100
    
    print(f"Step {i*10}, Context {step.context_length}: "
          f"Weights={weights_pct:.1f}%, KV={kv_pct:.1f}%")
```

### 4. Test Different Dtypes
```python
dtypes = ["float32", "float16", "bfloat16", "int8"]

for dtype in dtypes:
    result = perf.calculate_achievable_ttft(
        system_constraints=gpu,
        batch_size=16,
        sequence_length=2048,
        dtype_override=dtype
    )
    print(f"{dtype}: TTFT={result.achievable_ttft*1000:.2f}ms, "
          f"Memory={result.memory_utilization*100:.1f}%")
```

## Key Concepts

### Bottleneck Analysis
The calculator identifies which resource limits performance:
- **compute**: Compute throughput is limiting
- **memory_bandwidth**: Memory bandwidth is limiting (most common for LLMs)
- **network_bandwidth**: Network communication is limiting (for multi-GPU)
- **storage_bandwidth**: Persistent storage bandwidth is limiting (MoE models that don't fit in DRAM)

### Utilization Metrics
All utilization values range from 0 to 1 (or >1 if resources exceeded):
- `< 0.5`: Underutilized
- `0.5 - 0.9`: Good utilization
- `> 0.9`: Near capacity
- `> 1.0`: Over-subscribed (OOM or impossible)

### Kernel Overhead
The `kernel_launch_latency` parameter (default: 5 microseconds) models the overhead of launching GPU kernels. This becomes significant for small batch sizes or short sequences where kernel launch time is comparable to execution time.

### Memory Components
Memory usage is broken down into:
1. **Model Weights**: Parameters stored on GPU
2. **KV Cache**: Cached keys/values for attention
3. **Activations**: Intermediate computation results

### Arithmetic Intensity
Ratio of compute operations to memory transfers (FLOPs/byte). Higher values indicate compute-bound workloads, lower values indicate memory-bound workloads. LLM inference is typically memory-bound (low arithmetic intensity).

## Advanced Features

### MLA (Multi-head Latent Attention)
If your model uses MLA, the calculator automatically adjusts KV cache calculations:
```python
# Model already configured with MLA
result = perf.calculate_decode_performance(
    system_constraints=gpu,
    batch_size=32,
    prefill_length=2048,
    decode_steps=512
)
# KV cache size is automatically reduced based on mla_kv_lora_rank
```

### Hybrid Architectures

The framework supports three levels of hybrid architecture modeling:

**1. Approximate hybrid (conv + attention, e.g., LFM2)**

Some models like **LFM2** combine convolution layers with attention layers. The calculator treats all layers uniformly as full transformer layers (conservative upper-bound estimate). KV cache is overestimated; compute is slightly overestimated for conv layers.

**2. Exact Mamba-2 / Attention hybrid (e.g., Nemotron-3-30B)**

Models that designate each layer as `MAMBA`, `ATTENTION`, or `MLP` via `hybrid_layer_types` are handled exactly:
- KV cache only from attention layers
- Mamba state (constant size) only from Mamba layers
- Attention FLOPs only from attention-type layers
- Mamba SSM FLOPs only from Mamba-type layers

**3. Sublayer-style hybrid (e.g., Nemotron-3-Super-120B)**

In these models, each entry in `hybrid_layer_types` is a _single standalone operation_ (`MAMBA_ONLY`, `LATENT_MOE`, or `ATTENTION_ONLY`). Each sublayer does exactly one thing — no combined blocks:
- `MAMBA_ONLY`: Mamba-2 SSM, no FFN
- `LATENT_MOE`: Latent-space MoE FFN, no attention or Mamba
- `ATTENTION_ONLY`: Standard GQA attention, no FFN

Memory effects:
- KV cache only from `ATTENTION` / `ATTENTION_ONLY` sublayers
- Mamba state only from `MAMBA` / `MAMBA_ONLY` sublayers
- `LATENT_MOE` sublayers contribute only weight traffic (no stateful memory)

```python
from llm_configs import NEMOTRON_3_SUPER_120B
from inference_performance import InferencePerformance, SystemConstraints

perf = InferencePerformance(NEMOTRON_3_SUPER_120B)
gpu = SystemConstraints.from_gpu_spec("H100")

# 88 sublayers: 40 Mamba, 40 LatentMoE, 8 Attention
# KV cache: from 8 attention sublayers only
# Mamba state: from 40 Mamba sublayers only
# Compute: 3 separate components

result = perf.calculate_achievable_ttft(
    system_constraints=gpu,
    batch_size=1,
    sequence_length=4096
)

print(f"TTFT: {result.achievable_ttft * 1000:.2f} ms")
print(f"Bottleneck: {result.bottleneck_resource}")
```

### Latent MoE

For `LATENT_MOE` sublayers (Nemotron-3-Super-120B), the calculator calls `LatentMoEConfig.get_prefill_flops()` / `get_decode_flops()` and `get_weight_params()` to compute exact FLOPs and memory traffic. See `LatentMoEConfig` in `LLM_STRUCTS.md` and the corresponding algorithm in `ALGORITHMS.md` for the full derivation.

### DSA (Dynamic Sparse Attention)
For models with DSA (top-K KV selection), calculations reflect reduced KV cache access:
```python
# Model configured with DSA
result = perf.calculate_decode_performance(
    system_constraints=gpu,
    batch_size=32,
    prefill_length=8192,  # Long context
    decode_steps=512
)
# Only top-K KV pairs are accessed, reducing memory traffic
```

### MoE (Mixture of Experts)
For MoE models, calculations account for sparse expert activation:
```python
from llm_configs import DEEPSEEK_V3
perf = InferencePerformance(DEEPSEEK_V3)

result = perf.calculate_achievable_ttft(
    system_constraints=gpu,
    batch_size=16,
    sequence_length=2048
)
# Only active experts are counted in compute and memory
```

### Persistent Storage Offloading (MoE Models)

When MoE models don't fit in GPU DRAM, the system models expert offloading from persistent storage (NVMe SSD). During decode, active experts are loaded on-demand from storage.

**When Storage Offloading Activates:**
- Model type: MoE only (not dense models)
- Phase: Decode only (not prefill/TTFT)
- Condition: Total model parameters exceed DRAM capacity

**Storage Traffic Calculation:**
```python
# For each decode iteration when model doesn't fit in DRAM:
storage_traffic = (num_active_experts × expert_size × num_moe_layers) / tensor_parallel_size
```

**Bottleneck Detection:**
Storage bandwidth is included in bottleneck analysis. If storage is too slow, it becomes the limiting factor:
```python
bottleneck_time = max(compute_time, memory_bw_time, network_time, storage_bw_time)
```

**Example:**
```python
from llm_configs import MIXTRAL_8X7B
from inference_performance import InferencePerformance, SystemConstraints

perf = InferencePerformance(MIXTRAL_8X7B)

# Model is 93.4 GB, use 40 GB GPU with slow storage
gpu = SystemConstraints(
    memory_capacity=40e9,                      # 40 GB DRAM
    memory_bandwidth=2e12,                     # 2 TB/s
    compute_throughput=1000e12,                # 1000 TFLOPS
    network_bandwidth=400e9,                   # 400 GB/s
    persistent_storage_bandwidth=5e9           # 5 GB/s (slow SSD)
)

result = perf.calculate_decode_performance(
    system_constraints=gpu,
    batch_size=16,
    prefill_length=2048,
    decode_steps=512
)

print(f"Bottleneck: {result.bottleneck}")  # Will show "storage_bandwidth"
print(f"Storage Utilization: {result.avg_storage_bw_utilization * 100:.1f}%")

# With fast storage, it won't bottleneck
fast_gpu = SystemConstraints(
    memory_capacity=40e9,
    memory_bandwidth=2e12,
    compute_throughput=1000e12,
    network_bandwidth=400e9,
    persistent_storage_bandwidth=1000e9        # 1 TB/s (fast NVMe)
)

result2 = perf.calculate_decode_performance(
    system_constraints=fast_gpu,
    batch_size=16,
    prefill_length=2048,
    decode_steps=512
)

print(f"Bottleneck: {result2.bottleneck}")  # Won't be storage_bandwidth
print(f"Storage Utilization: {result2.avg_storage_bw_utilization * 100:.1f}%")  # Low %
```

**Key Points:**
- Default storage bandwidth: 20 GB/s (typical NVMe SSD)
- Only affects decode phase performance
- Dense models (LLAMA, GPT) never use storage offloading
- Storage utilization shown in results when >0%
- Can become bottleneck if storage is slower than compute/memory

## Troubleshooting

### OOM Errors
If `memory_utilization > 1.0`:
1. Reduce batch size
2. Use tensor parallelism to split model across GPUs
3. Enable quantization (int8 or int4 dtype)
4. For long sequences, consider DSA to reduce KV cache

### Low Performance
If utilization is low but performance is poor:
1. Check `bottleneck` - optimize the limiting resource
2. Examine `kernel_overhead_fraction` - high values indicate too many small kernels
3. For multi-GPU, check if network bandwidth is bottleneck
4. Consider increasing batch size to improve utilization

### Unrealistic Results
If results seem incorrect:
1. Verify model configuration (use `print(model)`)
2. Check GPU specifications (use `print(gpu)`)
3. Ensure parallelism config matches your setup
4. Validate dtype_override matches your model precision

## Best Practices

1. **Start Simple**: Begin with single GPU, no parallelism
2. **Profile First**: Use `return_step_details=True` to understand resource usage patterns
3. **Match Hardware**: Use accurate GPU specs via `from_gpu_spec()` or custom values
4. **Validate**: Compare calculated values with actual measurements when possible
5. **Iterate**: Test multiple batch sizes and parallelism strategies to find optimum
