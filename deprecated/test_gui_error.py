"""Test script to reproduce GUI calculate button error"""
import traceback
from llm_configs import LLAMA_3_8B
from inference_performance import InferencePerformance, SystemConstraints, ParallelismConfig, ParallelismType

try:
    # Simulate GUI button click
    model = LLAMA_3_8B
    perf = InferencePerformance(model)
    
    # System constraints (B200 defaults)
    gpu = SystemConstraints(
        memory_capacity=192e9,  # 192 GB
        memory_bandwidth=8000e9,  # 8000 GB/s
        compute_throughput=1250e12,  # 1250 TFLOPS
        network_bandwidth=900e9  # 900 GB/s
    )
    
    batch_size = 1
    sequence_length = 2048
    kernel_latency_s = 5.0 * 1e-6
    
    # Test with no parallelism
    parallel_config = None
    
    print("Testing TTFT calculation...")
    result = perf.calculate_achievable_ttft(
        system_constraints=gpu,
        batch_size=batch_size,
        sequence_length=sequence_length,
        parallelism_config=parallel_config,
        kernel_launch_latency=kernel_latency_s
    )
    
    print(f"✓ TTFT calculation successful: {result.achievable_ttft*1000:.2f} ms")
    print(f"  Result type: {type(result)}")
    print(f"  Result attributes: {dir(result)}")
    
    # Test the display logic
    if parallel_config is None:
        parallel_info = "\n  Parallelism:     None (Single GPU)"
        print(f"✓ Parallel info created: {repr(parallel_info)}")
    
    print("\n✓ All tests passed!")
    
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    traceback.print_exc()
