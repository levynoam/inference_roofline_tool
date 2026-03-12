"""
Test Llama 4 models in inference calculations
"""

from llm_configs import LLAMA_4_8B, LLAMA_4_70B, LLAMA_4_405B
from inference_performance import InferencePerformance, SystemConstraints

print("="*70)
print("Testing Llama 4 Models")
print("="*70)

# Test Llama 4 8B
print("\n1. Llama 4 8B on A100-80GB")
print("-" * 70)
model = LLAMA_4_8B
perf = InferencePerformance(model)
gpu = SystemConstraints.from_gpu_spec("A100-80GB")

result = perf.calculate_achievable_ttft(
    system_constraints=gpu,
    batch_size=1,
    sequence_length=2048
)

print(f"Model: {model.model_name}")
print(f"Parameters: {model.total_parameters/1e9:.1f}B")
print(f"Max Context: {model.max_sequence_length:,} tokens")
print(f"Achievable TTFT: {result.achievable_ttft*1000:.2f} ms")
print(f"Bottleneck: {result.bottleneck_resource}")
print(f"Compute Util: {result.compute_utilization*100:.1f}%")
print(f"Memory Usage: {result.memory_utilization*100:.1f}%")

# Test Llama 4 70B with Tensor Parallelism
print("\n2. Llama 4 70B on 4x A100-80GB (TP=4)")
print("-" * 70)
from inference_performance import ParallelismConfig, ParallelismType

model = LLAMA_4_70B
perf = InferencePerformance(model)
gpu = SystemConstraints.from_gpu_spec("A100-80GB")
parallel_config = ParallelismConfig(
    parallelism_type=ParallelismType.TENSOR_PARALLEL,
    tensor_parallel_size=4
)

result = perf.calculate_achievable_ttft(
    system_constraints=gpu,
    batch_size=1,
    sequence_length=2048,
    parallelism_config=parallel_config
)

print(f"Model: {model.model_name}")
print(f"Parameters: {model.total_parameters/1e9:.1f}B")
print(f"Max Context: {model.max_sequence_length:,} tokens")
print(f"Achievable TTFT: {result.achievable_ttft*1000:.2f} ms")
print(f"Bottleneck: {result.bottleneck_resource}")
print(f"Memory per GPU: {result.memory_used/1e9:.1f} GB")
print(f"Memory Fit: {'✓ Yes' if result.memory_utilization < 1.0 else '✗ OOM'}")

# Test Llama 4 405B 
print("\n3. Llama 4 405B on MI300X")
print("-" * 70)
model = LLAMA_4_405B
perf = InferencePerformance(model)
gpu = SystemConstraints.from_gpu_spec("MI300X")

result = perf.calculate_achievable_ttft(
    system_constraints=gpu,
    batch_size=1,
    sequence_length=2048
)

print(f"Model: {model.model_name}")
print(f"Parameters: {model.total_parameters/1e9:.1f}B")
print(f"Max Context: {model.max_sequence_length:,} tokens")
print(f"Achievable TTFT: {result.achievable_ttft*1000:.2f} ms")
print(f"Memory Used: {result.memory_used/1e9:.1f} GB")
print(f"Memory Available: {result.memory_available/1e9:.1f} GB")
print(f"Memory Fit: {'✓ Yes' if result.memory_utilization < 1.0 else '✗ OOM'}")

# Test Decode Performance
print("\n4. Llama 4 8B Decode Performance")
print("-" * 70)
model = LLAMA_4_8B
perf = InferencePerformance(model)
gpu = SystemConstraints.from_gpu_spec("A100-80GB")

result = perf.calculate_decode_performance(
    system_constraints=gpu,
    batch_size=4,
    prefill_length=8192,  # Using longer context (Llama 4 supports 128K)
    output_length=256
)

print(f"Configuration: Batch=4, Prefill=8192, Output=256")
print(f"Total Time: {result.total_decode_time*1000:.2f} ms")
print(f"TPS per user: {result.tokens_per_second_per_user:.2f} tokens/sec")
print(f"Total Throughput: {result.total_throughput:.2f} tokens/sec")
print(f"Primary Bottleneck: {result.primary_bottleneck}")
print(f"Avg Memory BW Util: {result.avg_memory_bw_utilization*100:.1f}%")

print("\n" + "="*70)
print("All Llama 4 tests completed successfully!")
print("="*70)
