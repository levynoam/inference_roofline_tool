"""
Example usage of inference performance library
Demonstrates prefill resource calculation
"""

from llm_configs import get_model
from inference_performance import InferencePerformance, ParallelismConfig, ParallelismType


def example_single_gpu_prefill():
    """Example: Calculate prefill resources for single GPU"""
    print("=" * 80)
    print("EXAMPLE 1: Single GPU Prefill")
    print("=" * 80)
    
    # Load Llama 3 8B
    model = get_model("llama-3-8b")
    perf = InferencePerformance(model)
    
    # Configuration
    batch_size = 1
    sequence_length = 2048
    time_to_first_token = 0.5  # 500ms target
    num_gpus = 1
    
    parallelism = ParallelismConfig(
        parallelism_type=ParallelismType.NONE,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=1
    )
    
    # Calculate resources
    resources = perf.calculate_prefill_resources(
        batch_size=batch_size,
        sequence_length=sequence_length,
        time_to_first_token=time_to_first_token,
        num_gpus=num_gpus,
        parallelism_config=parallelism
    )
    
    print(f"\nModel: {model.model_name}")
    print(f"Config: batch_size={batch_size}, seq_len={sequence_length}, TTFT={time_to_first_token}s")
    print(resources.summary())
    print()


def example_tensor_parallel():
    """Example: Tensor parallelism across 4 GPUs"""
    print("=" * 80)
    print("EXAMPLE 2: Tensor Parallel (4 GPUs)")
    print("=" * 80)
    
    # Load Llama 3 70B
    model = get_model("llama-3-70b")
    perf = InferencePerformance(model)
    
    # Configuration
    batch_size = 4
    sequence_length = 4096
    time_to_first_token = 1.0  # 1 second target
    num_gpus = 4
    
    parallelism = ParallelismConfig(
        parallelism_type=ParallelismType.TENSOR_PARALLEL,
        tensor_parallel_size=4,
        pipeline_parallel_size=1,
        data_parallel_size=1
    )
    
    # Calculate resources
    resources = perf.calculate_prefill_resources(
        batch_size=batch_size,
        sequence_length=sequence_length,
        time_to_first_token=time_to_first_token,
        num_gpus=num_gpus,
        parallelism_config=parallelism
    )
    
    print(f"\nModel: {model.model_name}")
    print(f"Config: batch_size={batch_size}, seq_len={sequence_length}, TTFT={time_to_first_token}s")
    print(f"Parallelism: Tensor Parallel (TP={parallelism.tensor_parallel_size})")
    print(resources.summary())
    print()


def example_pipeline_parallel():
    """Example: Pipeline parallelism"""
    print("=" * 80)
    print("EXAMPLE 3: Pipeline Parallel (8 GPUs)")
    print("=" * 80)
    
    # Load Llama 3 70B
    model = get_model("llama-3-70b")
    perf = InferencePerformance(model)
    
    # Configuration
    batch_size = 1
    sequence_length = 8192
    time_to_first_token = 2.0  # 2 seconds target
    num_gpus = 8
    
    parallelism = ParallelismConfig(
        parallelism_type=ParallelismType.PIPELINE_PARALLEL,
        tensor_parallel_size=1,
        pipeline_parallel_size=8,
        data_parallel_size=1
    )
    
    # Calculate resources
    resources = perf.calculate_prefill_resources(
        batch_size=batch_size,
        sequence_length=sequence_length,
        time_to_first_token=time_to_first_token,
        num_gpus=num_gpus,
        parallelism_config=parallelism
    )
    
    print(f"\nModel: {model.model_name}")
    print(f"Config: batch_size={batch_size}, seq_len={sequence_length}, TTFT={time_to_first_token}s")
    print(f"Parallelism: Pipeline Parallel (PP={parallelism.pipeline_parallel_size})")
    print(resources.summary())
    print()


def example_3d_parallel():
    """Example: 3D parallelism (TP + PP + DP)"""
    print("=" * 80)
    print("EXAMPLE 4: 3D Parallelism (TP=4, PP=2, DP=2 = 16 GPUs)")
    print("=" * 80)
    
    # Load DeepSeek V3 (large MoE model)
    model = get_model("deepseek-v3")
    perf = InferencePerformance(model)
    
    # Configuration
    batch_size = 8  # Split across DP=2, so 4 per GPU group
    sequence_length = 4096
    time_to_first_token = 1.5  # 1.5 seconds target
    num_gpus = 16
    
    parallelism = ParallelismConfig(
        parallelism_type=ParallelismType.FULL_3D,
        tensor_parallel_size=4,
        pipeline_parallel_size=2,
        data_parallel_size=2
    )
    
    # Calculate resources
    resources = perf.calculate_prefill_resources(
        batch_size=batch_size,
        sequence_length=sequence_length,
        time_to_first_token=time_to_first_token,
        num_gpus=num_gpus,
        parallelism_config=parallelism
    )
    
    print(f"\nModel: {model.model_name}")
    print(f"Config: batch_size={batch_size}, seq_len={sequence_length}, TTFT={time_to_first_token}s")
    print(f"Parallelism: 3D (TP={parallelism.tensor_parallel_size}, "
          f"PP={parallelism.pipeline_parallel_size}, DP={parallelism.data_parallel_size})")
    print(resources.summary())
    print()


def compare_parallelism_strategies():
    """Compare different parallelism strategies for the same workload"""
    print("=" * 80)
    print("COMPARISON: Different Parallelism Strategies")
    print("=" * 80)
    
    model = get_model("llama-3-70b")
    perf = InferencePerformance(model)
    
    batch_size = 4
    sequence_length = 4096
    time_to_first_token = 1.0
    
    strategies = [
        ("TP=4", ParallelismConfig(ParallelismType.TENSOR_PARALLEL, 4, 1, 1), 4),
        ("PP=4", ParallelismConfig(ParallelismType.PIPELINE_PARALLEL, 1, 4, 1), 4),
        ("TP=2,PP=2", ParallelismConfig(ParallelismType.TENSOR_PIPELINE, 2, 2, 1), 4),
    ]
    
    print(f"\nModel: {model.model_name}")
    print(f"Workload: batch={batch_size}, seq_len={sequence_length}, TTFT={time_to_first_token}s")
    print()
    print(f"{'Strategy':<15} | {'Memory':>10} | {'Mem BW':>10} | {'Net BW':>10} | {'Compute':>10} | {'AI':>8}")
    print("-" * 90)
    
    for name, config, gpus in strategies:
        resources = perf.calculate_prefill_resources(
            batch_size, sequence_length, time_to_first_token, gpus, config
        )
        
        print(f"{name:<15} | "
              f"{resources.memory_per_gpu / (1024**3):9.2f}G | "
              f"{resources.memory_bandwidth_per_gpu / (1024**3):9.2f}G | "
              f"{resources.network_bandwidth_per_gpu / (1024**3):9.2f}G | "
              f"{resources.compute_flops_per_sec / 1e12:9.2f}T | "
              f"{resources.arithmetic_intensity:8.2f}")
    
    print()


def analyze_scaling():
    """Analyze how resources scale with sequence length"""
    print("=" * 80)
    print("SCALING ANALYSIS: Sequence Length")
    print("=" * 80)
    
    model = get_model("llama-3-8b")
    perf = InferencePerformance(model)
    
    batch_size = 1
    time_to_first_token = 0.5
    num_gpus = 1
    parallelism = ParallelismConfig(ParallelismType.NONE, 1, 1, 1)
    
    print(f"\nModel: {model.model_name}")
    print(f"Config: batch={batch_size}, TTFT={time_to_first_token}s, single GPU")
    print()
    print(f"{'Seq Len':>8} | {'Memory':>10} | {'Mem BW':>10} | {'Compute':>10} | {'AI':>8}")
    print("-" * 65)
    
    for seq_len in [512, 1024, 2048, 4096, 8192]:
        resources = perf.calculate_prefill_resources(
            batch_size, seq_len, time_to_first_token, num_gpus, parallelism
        )
        
        print(f"{seq_len:8} | "
              f"{resources.memory_per_gpu / (1024**3):9.2f}G | "
              f"{resources.memory_bandwidth_per_gpu / (1024**3):9.2f}G | "
              f"{resources.compute_flops_per_sec / 1e12:9.2f}T | "
              f"{resources.arithmetic_intensity:8.2f}")
    
    print()


def hardware_requirements():
    """Determine if hardware can meet requirements"""
    print("=" * 80)
    print("HARDWARE REQUIREMENTS CHECK")
    print("=" * 80)
    
    model = get_model("llama-3-8b")
    perf = InferencePerformance(model)
    
    # Target workload
    batch_size = 4
    sequence_length = 2048
    time_to_first_token = 0.200  # 200ms target (aggressive)
    num_gpus = 1
    parallelism = ParallelismConfig(ParallelismType.NONE, 1, 1, 1)
    
    resources = perf.calculate_prefill_resources(
        batch_size, sequence_length, time_to_first_token, num_gpus, parallelism
    )
    
    # Example GPU specs (e.g., NVIDIA A100 80GB)
    gpu_memory = 80 * (1024**3)  # 80 GB
    gpu_memory_bandwidth = 2000 * (1024**3)  # 2 TB/s
    gpu_compute = 312e12  # 312 TFLOP/s (FP16)
    
    print(f"\nModel: {model.model_name}")
    print(f"Workload: batch={batch_size}, seq_len={sequence_length}, TTFT={time_to_first_token*1000:.0f}ms")
    print()
    print("Required Resources:")
    print(f"  Memory:          {resources.memory_per_gpu / (1024**3):8.2f} GB")
    print(f"  Memory BW:       {resources.memory_bandwidth_per_gpu / (1024**3):8.2f} GB/s")
    print(f"  Compute:         {resources.compute_flops_per_sec / 1e12:8.2f} TFLOP/s")
    print()
    print("GPU Specs (A100 80GB):")
    print(f"  Memory:          {gpu_memory / (1024**3):8.2f} GB")
    print(f"  Memory BW:       {gpu_memory_bandwidth / (1024**3):8.2f} GB/s")
    print(f"  Compute:         {gpu_compute / 1e12:8.2f} TFLOP/s")
    print()
    print("Analysis:")
    
    memory_ok = resources.memory_per_gpu <= gpu_memory
    mem_bw_ok = resources.memory_bandwidth_per_gpu <= gpu_memory_bandwidth
    compute_ok = resources.compute_flops_per_sec <= gpu_compute
    
    print(f"  Memory:          {'✓ OK' if memory_ok else '✗ INSUFFICIENT'} "
          f"({resources.memory_per_gpu / gpu_memory * 100:.1f}% utilized)")
    print(f"  Memory BW:       {'✓ OK' if mem_bw_ok else '✗ INSUFFICIENT'} "
          f"({resources.memory_bandwidth_per_gpu / gpu_memory_bandwidth * 100:.1f}% utilized)")
    print(f"  Compute:         {'✓ OK' if compute_ok else '✗ INSUFFICIENT'} "
          f"({resources.compute_flops_per_sec / gpu_compute * 100:.1f}% utilized)")
    
    if memory_ok and mem_bw_ok and compute_ok:
        print("\n✓ Hardware can meet requirements!")
    else:
        print("\n✗ Hardware cannot meet requirements - need more resources or relaxed constraints")
    
    print()


def main():
    """Run all examples"""
    example_single_gpu_prefill()
    example_tensor_parallel()
    example_pipeline_parallel()
    example_3d_parallel()
    compare_parallelism_strategies()
    analyze_scaling()
    hardware_requirements()


if __name__ == "__main__":
    main()
