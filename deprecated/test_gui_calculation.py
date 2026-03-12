"""
Test script to debug GUI calculation issues
"""

from llm_configs import LLAMA_3_8B
from inference_performance import (
    InferencePerformance,
    SystemConstraints,
    ParallelismConfig
)

print("="*60)
print("Testing GUI Calculation Logic")
print("="*60)

# Test 1: Create system constraints manually (simulating GUI inputs)
print("\nTest 1: Create SystemConstraints from parameters")
try:
    memory_capacity = float("80") * 1e9  # GB to bytes
    compute_throughput = float("312") * 1e12  # TFLOPS to FLOPS
    memory_bandwidth = float("2039") * 1e9  # GB/s to bytes/s
    network_bandwidth = float("600") * 1e9  # GB/s to bytes/s
    
    gpu = SystemConstraints(
        memory_capacity=memory_capacity,
        memory_bandwidth=memory_bandwidth,
        compute_throughput=compute_throughput,
        network_bandwidth=network_bandwidth
    )
    
    print(f"✓ SystemConstraints created successfully")
    print(f"  Memory: {gpu.memory_capacity/1e9:.1f} GB")
    print(f"  Compute: {gpu.compute_throughput/1e12:.1f} TFLOPS")
    print(f"  Memory BW: {gpu.memory_bandwidth/1e9:.1f} GB/s")
    print(f"  Network BW: {gpu.network_bandwidth/1e9:.1f} GB/s")
except Exception as e:
    print(f"✗ Failed to create SystemConstraints: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Calculate TTFT
print("\nTest 2: Calculate Achievable TTFT")
try:
    model = LLAMA_3_8B
    perf = InferencePerformance(model)
    
    result = perf.calculate_achievable_ttft(
        system_constraints=gpu,
        batch_size=1,
        sequence_length=2048,
        parallelism_config=ParallelismConfig()
    )
    
    print(f"✓ TTFT calculation successful")
    print(f"  Achievable TTFT: {result.achievable_ttft*1000:.2f} ms")
    print(f"  Bottleneck: {result.bottleneck_resource}")
    print(f"  Compute Util: {result.compute_utilization*100:.1f}%")
    print(f"  Memory Util: {result.memory_utilization*100:.1f}%")
except Exception as e:
    print(f"✗ Failed to calculate TTFT: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Calculate Decode
print("\nTest 3: Calculate Decode Performance")
try:
    result = perf.calculate_decode_performance(
        system_constraints=gpu,
        batch_size=1,
        prefill_length=2048,
        output_length=128,
        parallelism_config=ParallelismConfig()
    )
    
    print(f"✓ Decode calculation successful")
    print(f"  Total Time: {result.total_decode_time*1000:.2f} ms")
    print(f"  TPS: {result.tokens_per_second_per_user:.2f} tokens/sec")
    print(f"  Primary Bottleneck: {result.primary_bottleneck}")
    print(f"  Avg Compute Util: {result.avg_compute_utilization*100:.1f}%")
    print(f"  Avg Memory BW Util: {result.avg_memory_bw_utilization*100:.1f}%")
except Exception as e:
    print(f"✗ Failed to calculate decode: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Test with invalid inputs (what GUI might send)
print("\nTest 4: Test with edge cases")
try:
    # Empty strings
    memory_capacity = float("") * 1e9
except ValueError as e:
    print(f"✓ Correctly catches empty string: {e}")

try:
    # Non-numeric
    memory_capacity = float("abc") * 1e9
except ValueError as e:
    print(f"✓ Correctly catches non-numeric: {e}")

print("\n" + "="*60)
print("All tests completed")
print("="*60)
