"""Test pipeline parallelism scaling in decode"""
from llm_configs import LLAMA_3_8B
from inference_performance import InferencePerformance, SystemConstraints, ParallelismConfig, ParallelismType

model = LLAMA_3_8B
perf = InferencePerformance(model)

# System constraints (B200)
gpu = SystemConstraints(
    memory_capacity=192e9,
    memory_bandwidth=8000e9,
    compute_throughput=1250e12,
    network_bandwidth=900e9
)

batch_size = 1
prefill_length = 2048
output_length = 10

print("=" * 70)
print("Testing Pipeline Parallelism Scaling")
print("=" * 70)
print(f"Model: Llama 3 8B")
print(f"Batch Size: {batch_size}")
print(f"Prefill Length: {prefill_length}")
print(f"Output Length: {output_length}")
print()

# Test with no parallelism
print("1. No Parallelism (1 GPU)")
print("-" * 70)
result1 = perf.calculate_decode_performance(
    system_constraints=gpu,
    batch_size=batch_size,
    prefill_length=prefill_length,
    output_length=output_length,
    parallelism_config=None
)
print(f"Total Decode Time: {result1.total_decode_time*1000:.2f} ms")
print(f"Avg Step Time:     {result1.avg_step_time*1000:.3f} ms")
print(f"TPS per user:      {result1.tokens_per_second_per_user:.2f} tokens/sec")
print(f"Primary Bottleneck: {result1.primary_bottleneck}")
print()

# Test with Pipeline Parallel = 2
print("2. Pipeline Parallel (2 GPUs)")
print("-" * 70)
pp_config = ParallelismConfig(
    parallelism_type=ParallelismType.PIPELINE_PARALLEL,
    pipeline_parallel_size=2,
    tensor_parallel_size=1,
    data_parallel_size=1
)
result2 = perf.calculate_decode_performance(
    system_constraints=gpu,
    batch_size=batch_size,
    prefill_length=prefill_length,
    output_length=output_length,
    parallelism_config=pp_config
)
print(f"Total Decode Time: {result2.total_decode_time*1000:.2f} ms")
print(f"Avg Step Time:     {result2.avg_step_time*1000:.3f} ms")
print(f"TPS per user:      {result2.tokens_per_second_per_user:.2f} tokens/sec")
print(f"Primary Bottleneck: {result2.primary_bottleneck}")
print(f"Speedup vs 1 GPU:  {result1.total_decode_time/result2.total_decode_time:.2f}x")
print()

# Test with Pipeline Parallel = 4
print("3. Pipeline Parallel (4 GPUs)")
print("-" * 70)
pp_config4 = ParallelismConfig(
    parallelism_type=ParallelismType.PIPELINE_PARALLEL,
    pipeline_parallel_size=4,
    tensor_parallel_size=1,
    data_parallel_size=1
)
result4 = perf.calculate_decode_performance(
    system_constraints=gpu,
    batch_size=batch_size,
    prefill_length=prefill_length,
    output_length=output_length,
    parallelism_config=pp_config4
)
print(f"Total Decode Time: {result4.total_decode_time*1000:.2f} ms")
print(f"Avg Step Time:     {result4.avg_step_time*1000:.3f} ms")
print(f"TPS per user:      {result4.tokens_per_second_per_user:.2f} tokens/sec")
print(f"Primary Bottleneck: {result4.primary_bottleneck}")
print(f"Speedup vs 1 GPU:  {result1.total_decode_time/result4.total_decode_time:.2f}x")
print()

print("=" * 70)
print("Analysis:")
print("=" * 70)
print("With pipeline parallelism, each GPU processes a fraction of the layers.")
print("Expected behavior:")
print("  - Compute workload scales down by pipeline_parallel_size")
print("  - Memory per GPU scales down (model weights split)")
print("  - Network communication increases (inter-stage transfers)")
print()
print(f"PP=2 should reduce compute by ~2x compared to PP=1")
print(f"PP=4 should reduce compute by ~4x compared to PP=1")
