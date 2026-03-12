# Kernel Launch Overhead

## Overview

The performance library now includes kernel launch latency modeling to account for the overhead of launching GPU kernels during inference.

## Model Parameter

**`kernel_launch_latency`** (float, default: 5e-6 seconds = 5 microseconds)
- Represents the overhead to launch a single kernel on the GPU
- Typical values:
  - **1-3 µs**: Highly optimized implementations with kernel fusion
  - **5-10 µs**: Moderately optimized implementations
  - **10-20 µs**: Less optimized or older GPU architectures

## Kernel Count Calculation

For a **moderately optimized implementation** with basic fusion:

### Per Transformer Layer (Dense):
1. Pre-attention layer norm: 1 kernel
2. QKV projection (fused): 1 kernel  
3. Attention score computation: 1 kernel
4. Softmax: 1 kernel
5. Attention output matmul: 1 kernel
6. Attention output projection: 1 kernel
7. Post-attention layer norm: 1 kernel
8. FFN up projection + activation (fused): 1 kernel
9. FFN down projection: 1 kernel

**Total: ~9 kernels per dense layer**

### Per Transformer Layer (MoE):
- All dense operations: 9 kernels
- Router computation: 1 kernel
- Expert selection/dispatch: 1 kernel
- Expert combine: 1 kernel

**Total: ~12 kernels per MoE layer**

### Additional Operations:
- Embedding lookup: 1 kernel
- Final layer norm: 1 kernel
- LM head projection: 1 kernel

### Total Formula:
```
Dense Model: 3 + (num_layers * 9)
MoE Model:   3 + (num_layers * 12)
```

With pipeline parallelism, only the layers on each GPU are counted:
```
kernels_per_gpu = 3 + (num_layers / pipeline_parallel_size) * kernels_per_layer
```

## Impact on Performance

The kernel launch overhead reduces the effective compute time available:

```
effective_compute_time = TTFT - (num_kernels * kernel_launch_latency)
```

This increases the required compute throughput:
```
required_FLOP/s = total_FLOPs / effective_compute_time
```

## Examples

### Llama 3 8B (32 layers, dense)
- Kernel launches: 291
- Overhead @ 5µs: 1.46 ms
- Impact on 100ms TTFT: 1.5%
- Impact on 10ms TTFT: 14.6%

### Llama 3 70B (80 layers, dense)  
- Kernel launches: 723
- Overhead @ 5µs: 3.62 ms
- Impact on 100ms TTFT: 3.6%

### DeepSeek V3 (61 layers, MoE)
- Kernel launches: 735
- Overhead @ 5µs: 3.68 ms
- Impact on 100ms TTFT: 3.7%

## Usage

```python
from llm_configs import get_model
from inference_performance import InferencePerformance, ParallelismConfig, ParallelismType

# Load model (default kernel_launch_latency = 5µs)
model = get_model("llama-3-8b")

# Or create custom model with different latency
from llm_architecture import LLMArchitecture

custom_model = LLMArchitecture(
    # ... other params ...
    kernel_launch_latency=10e-6  # 10 µs for less optimized implementation
)

# Calculate resources (automatically includes kernel overhead)
perf = InferencePerformance(model)
resources = perf.calculate_prefill_resources(
    batch_size=1,
    sequence_length=2048,
    time_to_first_token=0.1,  # 100ms
    num_gpus=1,
    parallelism_config=ParallelismConfig(ParallelismType.NONE, 1, 1, 1)
)

# Access kernel overhead metrics
print(f"Kernel launches: {resources.num_kernel_launches}")
print(f"Launch overhead: {resources.kernel_launch_overhead * 1000:.2f} ms")
print(f"Effective compute time: {resources.effective_compute_time * 1000:.2f} ms")
```

## Key Insights

1. **Larger models** have more kernel launches due to more layers
2. **MoE models** have ~33% more kernel launches per layer than dense models
3. **Pipeline parallelism** reduces kernels per GPU (layers are distributed)
4. **Tensor/Data parallelism** don't reduce kernel count (all layers still execute)
5. **Impact increases** with shorter TTFT targets (overhead becomes larger %)
6. **Highly optimized** implementations can reduce kernels through aggressive fusion

## Comparison: Kernel Launch Latencies

For Llama 3 8B (batch=1, seq=2048, TTFT=200ms):

| Latency | Overhead | % of TTFT | Required Compute |
|---------|----------|-----------|------------------|
| 1 µs    | 0.29 ms  | 0.1%      | 164.93 TFLOP/s   |
| 5 µs    | 1.46 ms  | 0.7%      | 165.90 TFLOP/s   |
| 10 µs   | 2.91 ms  | 1.5%      | 167.13 TFLOP/s   |
| 20 µs   | 5.82 ms  | 2.9%      | 169.63 TFLOP/s   |

## Error Handling

If kernel launch overhead exceeds the target TTFT, an error is raised:

```python
ValueError: Kernel launch overhead (15.00ms) exceeds target TTFT (10.00ms). 
Cannot meet timing requirements with 291 kernel launches.
```

This indicates that either:
- The TTFT target is too aggressive
- More kernel fusion is needed
- Hardware with lower launch latency is required
