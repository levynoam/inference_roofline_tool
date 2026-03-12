"""Test script to simulate GUI Calculate button click"""
import traceback
from llm_configs import LLAMA_3_8B
from inference_performance import InferencePerformance, SystemConstraints, ParallelismConfig, ParallelismType

try:
    print("=== Simulating GUI Calculate Button Click ===\n")
    
    # Get model
    model = LLAMA_3_8B
    perf = InferencePerformance(model)
    print(f"✓ Model loaded: Llama 3 8B")
    
    # System constraints (B200 defaults from GUI)
    gpu = SystemConstraints(
        memory_capacity=192e9,  # 192 GB
        memory_bandwidth=8000e9,  # 8000 GB/s
        compute_throughput=1250e12,  # 1250 TFLOPS
        network_bandwidth=900e9  # 900 GB/s
    )
    print(f"✓ System constraints created")
    
    # GUI defaults
    batch_size = 1
    sequence_length = 2048
    kernel_latency_s = 5.0 * 1e-6
    
    # Parallelism = None (as selected in GUI)
    parallel_config = ParallelismConfig()  # This is what get_parallelism_config() returns
    print(f"✓ Parallelism config: {parallel_config.parallelism_type}, total_gpus={parallel_config.total_gpus}")
    
    # Calculate TTFT (same as GUI)
    print("\n--- Running TTFT Calculation ---")
    result = perf.calculate_achievable_ttft(
        system_constraints=gpu,
        batch_size=batch_size,
        sequence_length=sequence_length,
        parallelism_config=parallel_config,
        kernel_launch_latency=kernel_latency_s
    )
    print(f"✓ Calculation successful!")
    print(f"  TTFT: {result.achievable_ttft*1000:.2f} ms")
    print(f"  Bottleneck: {result.bottleneck_resource}")
    
    # Test display logic (same as GUI)
    print("\n--- Testing Display Logic ---")
    if parallel_config is None:
        parallel_info = "\n  Parallelism:     None (Single GPU)"
    else:
        num_gpus = parallel_config.total_gpus
        if num_gpus > 1:
            parallel_info = f"\n  Parallelism:     {parallel_config.parallelism_type.name} (N={num_gpus} GPUs)"
        else:
            parallel_info = "\n  Parallelism:     None (Single GPU)"
    
    print(f"✓ Display logic successful!")
    print(f"  Parallel info: {repr(parallel_info)}")
    
    # Build output string (same as GUI)
    memory_gb = 192
    compute_tflops = 1250
    memory_bw_gbs = 8000
    
    output = f"""
{'='*60}
ACHIEVABLE TIME TO FIRST TOKEN (TTFT)
{'='*60}

Configuration:
  Model:           Llama 3 8B{parallel_info}
  GPU Memory:      {memory_gb} GB (per GPU)
  GPU Compute:     {compute_tflops} TFLOPS (per GPU)
  Memory BW:       {memory_bw_gbs} GB/s (per GPU)
  Batch Size:      {batch_size}
  Sequence Length: {sequence_length}

Performance Metrics:
  Achievable TTFT: {result.achievable_ttft*1000:.2f} ms
  Throughput:      {batch_size/result.achievable_ttft:.2f} requests/sec
"""
    
    print("\n--- Output Preview ---")
    print(output[:300] + "...")
    
    print("\n✅ ALL TESTS PASSED - GUI should work!")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
