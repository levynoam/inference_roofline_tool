"""
Demonstration of dtype_override parameter
Shows how different data types affect memory, bandwidth, and performance
"""

from llm_configs import LLAMA_3_8B
from inference_performance import InferencePerformance, SystemConstraints, ParallelismConfig


def main():
    # Setup
    model = LLAMA_3_8B
    perf = InferencePerformance(model)
    gpu = SystemConstraints(
        memory_capacity=80e9,  # 80GB HBM
        memory_bandwidth=2039e9,  # 2TB/s
        compute_throughput=312e12,  # 312 TFLOPS (FP16)
        network_bandwidth=600e9,  # 600 GB/s
    )
    parallelism = ParallelismConfig()
    
    print("=" * 80)
    print("Data Type Override Demonstration")
    print("=" * 80)
    print(f"Model: {model.model_name}")
    print(f"Default dtype: {model.dtype}")
    print(f"Parameters: {model.active_parameters/1e9:.1f}B")
    print()
    
    # Test TTFT with different dtypes
    print("TTFT Performance (Batch=8, Sequence=2048)")
    print("-" * 80)
    print(f"{'Dtype':<12} {'Memory (GB)':<15} {'Bandwidth (%)':<18} {'TTFT (ms)':<15}")
    print("-" * 80)
    
    dtypes = [
        ("Model Default", None),
        ("INT4 (4-bit)", "int4"),
        ("INT8 (8-bit)", "int8"),
        ("FP16 (16-bit)", "float16"),
        ("FP32 (32-bit)", "float32"),
    ]
    
    for name, dtype in dtypes:
        result = perf.calculate_achievable_ttft(
            gpu, batch_size=8, sequence_length=2048,
            parallelism_config=parallelism,
            dtype_override=dtype
        )
        memory_gb = result.memory_used / 1e9
        bw_pct = result.memory_bandwidth_utilization * 100
        ttft_ms = result.achievable_ttft * 1000
        
        print(f"{name:<12} {memory_gb:>13.2f}  {bw_pct:>16.1f}  {ttft_ms:>13.2f}")
    
    print()
    print("=" * 80)
    
    # Test Decode with different dtypes
    print("Decode Performance (Batch=1, Prefill=2048, Output=1000)")
    print("-" * 80)
    print(f"{'Dtype':<12} {'Weights (GB)':<15} {'KV Cache (MB)':<18} {'TPS':<15}")
    print("-" * 80)
    
    for name, dtype in dtypes:
        result = perf.calculate_decode_performance(
            gpu, batch_size=1, prefill_length=2048, output_length=1000,
            parallelism_config=parallelism,
            dtype_override=dtype
        )
        weights_gb = result.avg_memory_weights / 1e9
        kv_mb = result.avg_memory_kv_cache / 1e6
        tps = result.tokens_per_second_per_user
        
        print(f"{name:<12} {weights_gb:>13.2f}  {kv_mb:>16.1f}  {tps:>13.1f}")
    
    print()
    print("=" * 80)
    
    # Show memory savings comparison
    print("Memory Savings vs FP16")
    print("-" * 80)
    
    result_fp16 = perf.calculate_decode_performance(
        gpu, 1, 2048, 1000, parallelism, dtype_override="float16"
    )
    
    for name, dtype in [("INT4 (4-bit)", "int4"), ("INT8 (8-bit)", "int8")]:
        result = perf.calculate_decode_performance(
            gpu, 1, 2048, 1000, parallelism, dtype_override=dtype
        )
        
        weight_savings = (1 - result.avg_memory_weights / result_fp16.avg_memory_weights) * 100
        kv_savings = (1 - result.avg_memory_kv_cache / result_fp16.avg_memory_kv_cache) * 100
        
        print(f"{name}:")
        print(f"  Weight Memory Savings: {weight_savings:.1f}%")
        print(f"  KV Cache Savings:      {kv_savings:.1f}%")
        print()
    
    print("=" * 80)
    print("Key Insights:")
    print("- INT4 uses 4x less memory than FP16 (weights and KV cache)")
    print("- INT8 uses 2x less memory than FP16")
    print("- Lower precision can improve TTFT when memory bandwidth constrained")
    print("- Lower precision enables higher throughput (TPS) in decode")
    print("=" * 80)


if __name__ == "__main__":
    main()
