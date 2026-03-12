"""
Test decode step skipping optimization
Verify that N=1 and N=100 give roughly the same results
"""

from llm_configs import LLAMA_3_8B
from inference_performance import InferencePerformance, SystemConstraints

def test_decode_skip_accuracy():
    """Test that decode_step_skip produces accurate results"""
    
    model = LLAMA_3_8B
    perf = InferencePerformance(model)
    
    # B200 GPU configuration
    gpu = SystemConstraints(
        memory_capacity=192e9,
        memory_bandwidth=8000e9,
        compute_throughput=1250e12,
        network_bandwidth=900e9
    )
    
    batch_size = 1
    prefill_length = 2048
    output_length = 1000
    kernel_latency = 5e-6
    
    print("Testing decode step skip optimization...")
    print(f"Configuration: batch_size={batch_size}, prefill_length={prefill_length}, output_length={output_length}")
    print()
    
    # Calculate with N=1 (no skipping)
    print("Running with decode_step_skip=1 (no optimization)...")
    result_n1 = perf.calculate_decode_performance(
        system_constraints=gpu,
        batch_size=batch_size,
        prefill_length=prefill_length,
        output_length=output_length,
        parallelism_config=None,
        kernel_launch_latency=kernel_latency,
        decode_step_skip=1
    )
    
    # Calculate with N=100 (default optimization)
    print("Running with decode_step_skip=100 (optimized)...")
    result_n100 = perf.calculate_decode_performance(
        system_constraints=gpu,
        batch_size=batch_size,
        prefill_length=prefill_length,
        output_length=output_length,
        parallelism_config=None,
        kernel_launch_latency=kernel_latency,
        decode_step_skip=100
    )
    
    print()
    print("=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)
    print()
    
    # Compare key metrics
    print(f"{'Metric':<40} {'N=1':<15} {'N=100':<15} {'Diff %':<10}")
    print("-" * 70)
    
    metrics = [
        ("Total Decode Time (ms)", result_n1.total_decode_time * 1000, result_n100.total_decode_time * 1000),
        ("Avg Step Time (ms)", result_n1.avg_step_time * 1000, result_n100.avg_step_time * 1000),
        ("Min Step Time (ms)", result_n1.min_step_time * 1000, result_n100.min_step_time * 1000),
        ("Max Step Time (ms)", result_n1.max_step_time * 1000, result_n100.max_step_time * 1000),
        ("Tokens per Second per User", result_n1.tokens_per_second_per_user, result_n100.tokens_per_second_per_user),
        ("Total Throughput (tokens/s)", result_n1.total_throughput, result_n100.total_throughput),
        ("Avg Memory Utilization (%)", result_n1.avg_memory_utilization * 100, result_n100.avg_memory_utilization * 100),
        ("Avg Memory BW Utilization (%)", result_n1.avg_memory_bw_utilization * 100, result_n100.avg_memory_bw_utilization * 100),
        ("Avg Compute Utilization (%)", result_n1.avg_compute_utilization * 100, result_n100.avg_compute_utilization * 100),
        ("Avg Network BW Utilization (%)", result_n1.avg_network_bw_utilization * 100, result_n100.avg_network_bw_utilization * 100),
    ]
    
    max_diff = 0.0
    for name, val1, val100 in metrics:
        diff_pct = abs(val1 - val100) / val1 * 100 if val1 != 0 else 0
        max_diff = max(max_diff, diff_pct)
        print(f"{name:<40} {val1:<15.4f} {val100:<15.4f} {diff_pct:<10.2f}")
    
    print()
    print(f"Primary Bottleneck (N=1):   {result_n1.primary_bottleneck}")
    print(f"Primary Bottleneck (N=100): {result_n100.primary_bottleneck}")
    
    print()
    print("=" * 70)
    print(f"Maximum difference: {max_diff:.2f}%")
    
    # Check if results are close enough (within 1%)
    if max_diff < 1.0:
        print("✅ PASS: Results match within 1% tolerance")
        return True
    else:
        print("❌ FAIL: Results differ by more than 1%")
        return False

if __name__ == "__main__":
    success = test_decode_skip_accuracy()
    exit(0 if success else 1)
