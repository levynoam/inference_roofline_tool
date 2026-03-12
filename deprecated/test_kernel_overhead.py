"""
Test kernel launch overhead impact
"""

from llm_configs import get_model
from inference_performance import InferencePerformance, ParallelismConfig, ParallelismType


def test_kernel_launch_impact():
    """Demonstrate the impact of kernel launch latency"""
    print("=" * 80)
    print("KERNEL LAUNCH OVERHEAD ANALYSIS")
    print("=" * 80)
    
    model = get_model("llama-3-8b")
    perf = InferencePerformance(model)
    
    batch_size = 1
    sequence_length = 2048
    time_to_first_token = 0.100  # 100ms - aggressive target
    parallelism = ParallelismConfig(ParallelismType.NONE, 1, 1, 1)
    
    print(f"\nModel: {model.model_name}")
    print(f"Kernel Launch Latency: {model.kernel_launch_latency * 1e6:.1f} µs")
    print(f"Target TTFT: {time_to_first_token * 1000:.2f} ms")
    print(f"Workload: batch={batch_size}, seq_len={sequence_length}")
    
    try:
        resources = perf.calculate_prefill_resources(
            batch_size, sequence_length, time_to_first_token, 1, parallelism
        )
        
        overhead_pct = (resources.kernel_launch_overhead / time_to_first_token) * 100
        
        print(f"\nKernel Launches: {resources.num_kernel_launches}")
        print(f"Total Overhead: {resources.kernel_launch_overhead * 1000:.3f} ms ({overhead_pct:.1f}% of TTFT)")
        print(f"Effective Compute Time: {resources.effective_compute_time * 1000:.3f} ms")
        print(f"\nRequired Compute: {resources.compute_flops_per_sec / 1e12:.2f} TFLOP/s")
        print(f"Required Memory BW: {resources.memory_bandwidth_per_gpu / (1024**3):.2f} GB/s")
        
    except ValueError as e:
        print(f"\n❌ ERROR: {e}")
    
    print("\n" + "=" * 80)


def compare_kernel_latencies():
    """Compare different kernel launch latencies"""
    print("=" * 80)
    print("COMPARISON: Different Kernel Launch Latencies")
    print("=" * 80)
    
    model = get_model("llama-3-8b")
    
    batch_size = 1
    sequence_length = 2048
    time_to_first_token = 0.200  # 200ms
    parallelism = ParallelismConfig(ParallelismType.NONE, 1, 1, 1)
    
    # Test different kernel latencies
    latencies = [
        (1e-6, "Optimized (1 µs)"),
        (5e-6, "Moderate (5 µs)"),
        (10e-6, "High (10 µs)"),
        (20e-6, "Very High (20 µs)"),
    ]
    
    print(f"\nModel: {model.model_name}")
    print(f"Target TTFT: {time_to_first_token * 1000:.0f} ms")
    print(f"Workload: batch={batch_size}, seq_len={sequence_length}\n")
    print(f"{'Latency':<20} | {'Overhead':>10} | {'Overhead %':>10} | {'Compute':>12}")
    print("-" * 70)
    
    for latency, name in latencies:
        # Temporarily set kernel launch latency
        original_latency = model.kernel_launch_latency
        model.kernel_launch_latency = latency
        
        perf = InferencePerformance(model)
        
        try:
            resources = perf.calculate_prefill_resources(
                batch_size, sequence_length, time_to_first_token, 1, parallelism
            )
            
            overhead_pct = (resources.kernel_launch_overhead / time_to_first_token) * 100
            
            print(f"{name:<20} | "
                  f"{resources.kernel_launch_overhead * 1000:9.2f}ms | "
                  f"{overhead_pct:9.1f}% | "
                  f"{resources.compute_flops_per_sec / 1e12:11.2f}T")
        except ValueError:
            print(f"{name:<20} | {'EXCEEDS TTFT':>10} | {'N/A':>10} | {'N/A':>12}")
        
        # Restore original latency
        model.kernel_launch_latency = original_latency
    
    print("=" * 80)


def show_kernel_count_by_model():
    """Show kernel launch count varies by model architecture"""
    print("=" * 80)
    print("KERNEL LAUNCHES BY MODEL")
    print("=" * 80)
    
    models = ["llama-3-8b", "llama-3-70b", "deepseek-v3", "mixtral-8x7b"]
    parallelism = ParallelismConfig(ParallelismType.NONE, 1, 1, 1)
    
    print(f"\n{'Model':<25} | {'Layers':>8} | {'Type':>8} | {'Kernels':>10} | {'Overhead@5µs':>15}")
    print("-" * 85)
    
    for model_key in models:
        model = get_model(model_key)
        perf = InferencePerformance(model)
        
        num_kernels = perf.calculate_num_kernel_launches(parallelism)
        overhead_ms = num_kernels * model.kernel_launch_latency * 1000
        
        model_type = "MoE" if model.is_moe else "Dense"
        
        print(f"{model.model_name:<25} | "
              f"{model.num_layers:8} | "
              f"{model_type:>8} | "
              f"{num_kernels:10} | "
              f"{overhead_ms:14.2f}ms")
    
    print("=" * 85)


if __name__ == "__main__":
    test_kernel_launch_impact()
    print("\n")
    compare_kernel_latencies()
    print("\n")
    show_kernel_count_by_model()
