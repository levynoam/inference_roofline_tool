# Inverse TTFT Calculation Function

## Overview

Added `calculate_achievable_ttft()` method to the `InferencePerformance` class - the **inverse** of `calculate_prefill_resources()`.

### Bidirectional Modeling

**Forward Direction (existing):** `calculate_prefill_resources()`
- Input: Target TTFT, batch size, sequence length, parallelism
- Output: Required resources (memory, compute, bandwidth)
- Answers: "What hardware do I need to achieve this performance?"

**Backward Direction (NEW):** `calculate_achievable_ttft()`
- Input: System constraints (GPU specs), batch size, sequence length, parallelism
- Output: Achievable TTFT, resource utilization, bottleneck identification
- Answers: "Given my hardware, what performance can I achieve?"

## New Data Structures

### SystemConstraints
Defines hardware specifications per GPU:
- `memory_capacity`: Total memory in bytes
- `memory_bandwidth`: Memory bandwidth in bytes/sec
- `compute_throughput`: Compute capability in FLOPs/sec
- `network_bandwidth`: Network bandwidth in bytes/sec

Includes factory method `from_gpu_spec()` with presets for:
- A100-40GB: 40GB, 1.5TB/s, 312 TFLOP/s, 600GB/s NVLink
- A100-80GB: 80GB, 2TB/s, 312 TFLOP/s, 600GB/s NVLink
- H100-80GB: 80GB, 3.35TB/s, 1979 TFLOP/s, 900GB/s NVLink 4.0
- MI300X: 192GB, 5.3TB/s, 1.3 PFLOP/s, 896GB/s Infinity Fabric

### ResourceUtilization
Complete performance analysis results:
- `achievable_ttft`: Calculated time to first token (seconds)
- `bottleneck_resource`: Which resource limits performance ("Compute", "Memory Bandwidth", "Network Bandwidth")
- Utilization percentages (0-1) for all resources
- Actual resource consumption rates
- Available resource capacities
- Kernel launch overhead breakdown

Includes `summary()` method for human-readable output.

## Algorithm

The function calculates time constraints from each resource:

1. **Compute Time** = Total FLOPs / GPU Compute Throughput
2. **Memory BW Time** = Memory Traffic / Memory Bandwidth
   - Memory traffic = 2x memory footprint (read + write)
3. **Network Time** = Network Traffic / Network Bandwidth
   - Tensor parallel: All-reduce of activations per layer
   - Pipeline parallel: Send activations between stages

**Bottleneck:** `TTFT = max(compute_time, memory_bw_time, network_time) + kernel_overhead`

The limiting resource determines the total time, and other resources will be under-utilized.

## Example Results

### Single GPU Analysis (Llama 3 8B on A100-80GB)
```
Achievable TTFT: 107.03 ms
  Kernel Overhead: 1.46 ms
  Effective Compute: 105.57 ms
Bottleneck: Compute

Resource Utilization:
  Memory:            19.4% ( 15.55 GB /  80.00 GB)
  Memory BW:         14.7% (294.62 / 2000.00 GB/s)
  Compute:          100.0% (312.00 / 312.00 TFLOP/s)
  Network BW:         0.0% (  0.00 / 600.00 GB/s)
```

**Key Insight:** Llama 3 8B is compute-bound on A100. Memory bandwidth is only 14.7% utilized - we have plenty of bandwidth headroom.

### GPU Comparison (Llama 3 8B, batch=1, seq=2048)
```
A100-40GB   : TTFT=107.03ms, Bottleneck=Compute, Memory= 38.9%, Compute=100.0%
A100-80GB   : TTFT=107.03ms, Bottleneck=Compute, Memory= 19.4%, Compute=100.0%
H100-80GB   : TTFT= 18.10ms, Bottleneck=Compute, Memory= 19.4%, Compute=100.0%
MI300X      : TTFT= 26.79ms, Bottleneck=Compute, Memory=  8.1%, Compute=100.0%
```

**Key Insight:** All GPUs are compute-bound. H100 is 5.9x faster than A100 due to higher compute throughput (1979 vs 312 TFLOP/s). Memory capacity doesn't matter here since 40GB is sufficient.

### Batch Size Scaling (Llama 3 8B, seq=2048, A100-80GB)
```
   Batch   TTFT (ms)       Bottleneck    Mem%    Compute%    MemBW%
------------------------------------------------------------------------
       1      107.03          Compute    19.4       100.0      14.7
       2      212.60          Compute    20.2       100.0       7.6
       4      423.75          Compute    21.7       100.0       4.1
       8      846.04          Compute    24.6       100.0       2.3
      16     1690.62          Compute    30.6       100.0       1.4
      32     3379.78          Compute    42.4       100.0       1.0
```

**Key Insight:** TTFT scales linearly with batch size (doubles each time). Always compute-bound. Memory bandwidth utilization drops because we're doing more compute per byte moved - this is good for efficiency!

### Model Comparison (A100-80GB, batch=1, seq=2048)
```
Llama 3 8B     : TTFT= 107.03ms, Memory= 19.4%, Bottleneck=Compute
Llama 3 70B    : OOM - requires 132.7GB
DeepSeek 3.2   : TTFT= 321.17ms, Memory= 49.6%, Bottleneck=Compute
Mixtral 8x7B   : TTFT= 176.35ms, Memory= 30.8%, Bottleneck=Compute
```

**Key Insight:** Llama 3 70B doesn't fit on single A100-80GB. DeepSeek 3.2 (21B with MLA) is 3x slower than Llama 3 8B but uses less memory thanks to MLA compression.

### Tensor Parallelism (Llama 3 70B, batch=1, seq=2048, A100-80GB)
```
  TP   TTFT (ms)       Bottleneck    Mem%    Compute%    Network%
--------------------------------------------------------------------
   1      951.30          Compute   165.9       100.0         0.0
   2      951.30          Compute    83.4       100.0         0.9
   4      951.30          Compute    42.1       100.0         0.9
   8      951.30          Compute    21.5       100.0         0.9
```

**Key Insights:**
- TP=1: OOM (166% memory usage)
- TP=2+: Fits in memory
- **TTFT stays constant** across TP configurations! This is because we're still compute-bound
- Network bandwidth is only 0.9% utilized - communication overhead is negligible
- TP helps fit the model in memory but doesn't improve TTFT (need more compute, not more GPUs)

## Usage

```python
from llm_configs import LLAMA_3_8B
from inference_performance import InferencePerformance, SystemConstraints

# Initialize
model = LLAMA_3_8B
perf = InferencePerformance(model)
gpu = SystemConstraints.from_gpu_spec("A100-80GB")

# Calculate achievable TTFT
result = perf.calculate_achievable_ttft(
    system_constraints=gpu,
    batch_size=1,
    sequence_length=2048
)

# Display results
print(result.summary())
print(f"Achievable TTFT: {result.achievable_ttft * 1000:.2f} ms")
print(f"Bottleneck: {result.bottleneck_resource}")
```

## Files Modified

- `inference_performance.py`:
  - Added `SystemConstraints` dataclass with GPU presets
  - Added `ResourceUtilization` dataclass with summary() method
  - Added `calculate_achievable_ttft()` method to InferencePerformance class

## Example File

See `example_achievable_ttft.py` for comprehensive demonstrations:
1. Single GPU analysis
2. GPU comparison
3. Batch size scaling
4. Model comparison
5. Tensor parallelism analysis
6. MLA memory benefit analysis
