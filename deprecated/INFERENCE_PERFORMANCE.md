# Inference Performance Library

Calculate resource requirements for LLM inference operations with support for various parallelism strategies.

## Overview

This library provides tools to calculate:
- **Memory requirements** per GPU (model weights, KV cache, activations)
- **Memory bandwidth** requirements per GPU
- **Compute requirements** (FLOPs and FLOP/s) per GPU
- **Network bandwidth** requirements for inter-GPU communication
- **Arithmetic intensity** and bottleneck analysis

## Key Components

### `inference_performance.py`

**Main Classes:**
- `InferencePerformance` - Main calculator class
- `ParallelismConfig` - Configuration for distributed execution
- `PrefillResources` - Results containing all resource metrics

**Parallelism Support:**
- None (single GPU)
- Data Parallel (DP)
- Tensor Parallel (TP)
- Pipeline Parallel (PP)
- Combinations (TP+PP, TP+DP, TP+PP+DP)

## Prefill Phase

The prefill phase processes the entire input prompt in one forward pass to generate the first token.

### Function: `calculate_prefill_resources()`

**Inputs:**
- `batch_size`: Number of sequences in batch
- `sequence_length`: Input sequence length (prompt length)
- `time_to_first_token`: Target TTFT in seconds
- `num_gpus`: Total number of GPUs
- `parallelism_config`: Configuration for parallelism strategy

**Outputs (PrefillResources):**
- `memory_per_gpu`: Total memory required per GPU (bytes)
  - `memory_model_weights`: Model parameter memory
  - `memory_kv_cache`: KV cache memory
  - `memory_activations`: Activation memory
- `memory_bandwidth_per_gpu`: Required memory bandwidth (bytes/sec)
- `network_bandwidth_per_gpu`: Required network bandwidth (bytes/sec)
- `compute_per_gpu`: Total FLOPs per GPU
- `compute_flops_per_sec`: Required FLOP/s per GPU
- `arithmetic_intensity`: FLOPs per byte (higher = more compute-bound)
- `compute_bound`: Boolean indicating compute vs memory bound

## Quick Start

### Basic Example: Single GPU

```python
from llm_configs import get_model
from inference_performance import InferencePerformance, ParallelismConfig, ParallelismType

# Load model
model = get_model("llama-3-8b")
perf = InferencePerformance(model)

# Configure parallelism (single GPU)
parallelism = ParallelismConfig(
    parallelism_type=ParallelismType.NONE,
    tensor_parallel_size=1,
    pipeline_parallel_size=1,
    data_parallel_size=1
)

# Calculate prefill resources
resources = perf.calculate_prefill_resources(
    batch_size=1,
    sequence_length=2048,
    time_to_first_token=0.5,  # 500ms
    num_gpus=1,
    parallelism_config=parallelism
)

# Print summary
print(resources.summary())
```

### Tensor Parallelism Example

```python
# Split Llama 3 70B across 4 GPUs with tensor parallelism
model = get_model("llama-3-70b")
perf = InferencePerformance(model)

parallelism = ParallelismConfig(
    parallelism_type=ParallelismType.TENSOR_PARALLEL,
    tensor_parallel_size=4,
    pipeline_parallel_size=1,
    data_parallel_size=1
)

resources = perf.calculate_prefill_resources(
    batch_size=4,
    sequence_length=4096,
    time_to_first_token=1.0,
    num_gpus=4,
    parallelism_config=parallelism
)
```

### 3D Parallelism Example

```python
# DeepSeek V3 with TP=4, PP=2, DP=2 (16 GPUs total)
model = get_model("deepseek-v3")
perf = InferencePerformance(model)

parallelism = ParallelismConfig(
    parallelism_type=ParallelismType.FULL_3D,
    tensor_parallel_size=4,
    pipeline_parallel_size=2,
    data_parallel_size=2
)

resources = perf.calculate_prefill_resources(
    batch_size=8,
    sequence_length=4096,
    time_to_first_token=1.5,
    num_gpus=16,
    parallelism_config=parallelism
)
```

## Calculation Details

### Compute (FLOPs)

For each transformer layer:
- **Attention:**
  - QKV projections: `2 * batch * seq * hidden * (num_heads * head_dim)`
  - Attention scores: `2 * batch * heads * seq^2 * head_dim`
  - Attention output: `2 * batch * heads * seq^2 * head_dim`
  - Output projection: `2 * batch * seq * hidden * hidden`
  
- **FFN:**
  - Up projection: `2 * batch * seq * hidden * intermediate`
  - Gating (if used): `2 * batch * seq * hidden * intermediate`
  - Down projection: `2 * batch * seq * intermediate * hidden`

- **MoE (if applicable):**
  - Router: `2 * batch * seq * hidden * num_experts`
  - Per-expert computation scaled by active experts

Total FLOPs accounts for all layers, embeddings, and final projection.

### Memory

**Model Weights:**
- Single GPU: Full model parameters
- Tensor Parallel: Parameters / TP_size
- Pipeline Parallel: Parameters / PP_size
- Combined: Parameters / (TP_size * PP_size)

**KV Cache:**
- Size: `2 * batch * kv_heads * seq * head_dim * layers * bytes_per_element`
- Distributed based on TP and DP (KV heads split across TP, batch split across DP)

**Activations:**
- Temporary buffers for attention and FFN outputs
- Peak memory for ~2 layers worth of activations

### Memory Bandwidth

Data movement includes:
- Reading model weights once
- Writing KV cache once
- Reading/writing activations multiple times (~4x estimate)

`Memory BW = total_data_movement / time_to_first_token`

### Network Bandwidth

**Tensor Parallelism:**
- All-reduce operations after each layer
- Data size: `batch * seq * hidden * bytes`
- Ring all-reduce factor: `2 * (TP_size - 1) / TP_size`

**Pipeline Parallelism:**
- Send activations between pipeline stages
- Data size: `batch * seq * hidden * bytes`
- Number of transfers: `PP_size - 1`

## Understanding Arithmetic Intensity

**Arithmetic Intensity (AI)** = FLOPs / Bytes moved from memory

- **Low AI (< 10)**: Memory-bound - limited by memory bandwidth
- **Medium AI (10-100)**: Balanced
- **High AI (> 100)**: Compute-bound - limited by compute throughput

Modern GPUs typically have roofline around AI = 50-200 depending on architecture.

## Hardware Requirements Validation

Compare calculated requirements against hardware specs:

```python
# Calculate requirements
resources = perf.calculate_prefill_resources(...)

# Compare against hardware (e.g., A100 80GB)
gpu_memory = 80 * (1024**3)  # 80 GB
gpu_memory_bw = 2000 * (1024**3)  # 2 TB/s
gpu_compute = 312e12  # 312 TFLOP/s (FP16)

memory_ok = resources.memory_per_gpu <= gpu_memory
mem_bw_ok = resources.memory_bandwidth_per_gpu <= gpu_memory_bw
compute_ok = resources.compute_flops_per_sec <= gpu_compute

# Utilization percentages
mem_util = resources.memory_per_gpu / gpu_memory * 100
mem_bw_util = resources.memory_bandwidth_per_gpu / gpu_memory_bw * 100
compute_util = resources.compute_flops_per_sec / gpu_compute * 100
```

## Example Results

### Llama 3 8B - Single GPU (batch=1, seq=2048, TTFT=500ms)
- Memory: 15.55 GB
- Memory BW: 33.17 GB/s
- Compute: 65.88 TFLOP/s
- Network BW: 0 GB/s
- AI: 1849.88 (Compute-bound)

### Llama 3 70B - Tensor Parallel 4 GPUs (batch=4, seq=4096, TTFT=1s)
- Memory: 39.60 GB per GPU
- Memory BW: 56.10 GB/s per GPU
- Compute: 2453.38 TFLOP/s per GPU
- Network BW: 60.00 GB/s per GPU
- AI: 40725.34 (Compute-bound)

### DeepSeek V3 - 3D Parallel 16 GPUs (batch=8, seq=4096, TTFT=1.5s)
- Memory: 20.11 GB per GPU
- Memory BW: 21.41 GB/s per GPU
- Compute: 2495.35 TFLOP/s per GPU
- Network BW: 13.27 GB/s per GPU
- AI: 108547.28 (Compute-bound)

## Scaling Characteristics

### Sequence Length Scaling
- Memory: ~O(seq_len) for KV cache
- Compute: ~O(seq_len²) for attention
- Higher sequence lengths increase AI (more compute-bound)

### Batch Size Scaling
- Memory: ~O(batch) for KV cache
- Compute: ~O(batch) linear scaling
- AI relatively stable with batch size

### Parallelism Trade-offs
- **Tensor Parallel**: Lower memory per GPU, high network BW
- **Pipeline Parallel**: Lower memory per GPU, low network BW, higher latency
- **Data Parallel**: High memory per GPU (full model), no network for inference

## Use Cases

1. **Capacity Planning**: Determine GPU requirements for target workload
2. **Bottleneck Analysis**: Identify if limited by compute, memory BW, or network
3. **Parallelism Selection**: Compare strategies to choose optimal configuration
4. **Performance Prediction**: Estimate TTFT for given hardware
5. **Cost Optimization**: Find most cost-effective hardware/parallelism combo

## Limitations & Assumptions

- Activation memory is estimated (implementation-dependent)
- Perfect efficiency assumed (no idle time, perfect overlap)
- Network communication modeled ideally (no congestion)
- Does not account for kernel launch overhead
- Assumes full FLOPs utilization (real hardware has efficiency < 100%)

## Future Extensions

Planned additions:
- **Decode phase** resource calculation
- **Continuous batching** support
- **KV cache compression** (quantization, GQA optimization)
- **Attention optimizations** (Flash Attention, Paged Attention)
- **Hardware-specific models** (A100, H100, AMD MI300, Intel Gaudi)
- **Real performance data** calibration
- **Multi-query/multi-turn** scenarios
- **Speculative decoding** modeling

## See Also

- `llm_architecture.py` - Model architecture definitions
- `llm_configs.py` - Pre-configured model examples
- `example_prefill.py` - Detailed usage examples
