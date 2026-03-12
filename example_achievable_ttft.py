"""
Example: Calculate achievable TTFT from system constraints
This demonstrates the INVERSE of prefill resource calculation:
- Input: GPU specifications + workload configuration
- Output: Achievable TTFT + resource utilization

This answers: "Given my hardware, how fast can I run inference?"
"""

from llm_configs import LLAMA_3_8B, LLAMA_3_70B, DEEPSEEK_3_2, MIXTRAL_8X7B
from inference_performance import InferencePerformance, SystemConstraints, ParallelismConfig, ParallelismType

def analyze_single_gpu():
    """Analyze what TTFT is achievable on a single GPU"""
    print("="*80)
    print("SINGLE GPU ANALYSIS")
    print("="*80)
    print()
    
    # Test with Llama 3 8B on A100-80GB
    model = LLAMA_3_8B
    perf = InferencePerformance(model)
    
    # Define system constraints
    gpu = SystemConstraints.from_gpu_spec("A100-80GB")
    
    # Calculate achievable TTFT
    batch_size = 1
    sequence_length = 2048
    
    result = perf.calculate_achievable_ttft(
        system_constraints=gpu,
        batch_size=batch_size,
        sequence_length=sequence_length
    )
    
    print(f"Model: {model.model_name}")
    print(f"GPU: A100-80GB")
    print(f"Workload: batch_size={batch_size}, sequence_length={sequence_length}")
    print()
    print(result.summary())
    print()


def compare_gpus():
    """Compare different GPUs on the same workload"""
    print("="*80)
    print("GPU COMPARISON")
    print("="*80)
    print()
    
    model = LLAMA_3_8B
    perf = InferencePerformance(model)
    
    batch_size = 1
    sequence_length = 2048
    
    gpu_names = ["A100-40GB", "A100-80GB", "H100-80GB", "MI300X"]
    
    print(f"Model: {model.model_name}")
    print(f"Workload: batch_size={batch_size}, sequence_length={sequence_length}")
    print()
    
    for gpu_name in gpu_names:
        gpu = SystemConstraints.from_gpu_spec(gpu_name)
        result = perf.calculate_achievable_ttft(
            system_constraints=gpu,
            batch_size=batch_size,
            sequence_length=sequence_length
        )
        
        print(f"{gpu_name:12s}: TTFT={result.achievable_ttft*1000:6.2f}ms, "
              f"Bottleneck={result.bottleneck_resource:15s}, "
              f"Memory={result.memory_utilization*100:5.1f}%, "
              f"Compute={result.compute_utilization*100:5.1f}%")
    
    print()


def analyze_scaling():
    """Analyze how TTFT scales with batch size and sequence length"""
    print("="*80)
    print("BATCH SIZE SCALING")
    print("="*80)
    print()
    
    model = LLAMA_3_8B
    perf = InferencePerformance(model)
    gpu = SystemConstraints.from_gpu_spec("A100-80GB")
    
    sequence_length = 2048
    batch_sizes = [1, 2, 4, 8, 16, 32]
    
    print(f"Model: {model.model_name}")
    print(f"GPU: A100-80GB")
    print(f"Sequence Length: {sequence_length}")
    print()
    print(f"{'Batch':>8s}  {'TTFT (ms)':>10s}  {'Bottleneck':>15s}  {'Mem%':>6s}  {'Compute%':>10s}  {'MemBW%':>8s}")
    print("-" * 80)
    
    for batch_size in batch_sizes:
        result = perf.calculate_achievable_ttft(
            system_constraints=gpu,
            batch_size=batch_size,
            sequence_length=sequence_length
        )
        
        print(f"{batch_size:8d}  {result.achievable_ttft*1000:10.2f}  "
              f"{result.bottleneck_resource:>15s}  "
              f"{result.memory_utilization*100:6.1f}  "
              f"{result.compute_utilization*100:10.1f}  "
              f"{result.memory_bandwidth_utilization*100:8.1f}")
    
    print()


def analyze_model_comparison():
    """Compare different models on the same GPU"""
    print("="*80)
    print("MODEL COMPARISON")
    print("="*80)
    print()
    
    gpu = SystemConstraints.from_gpu_spec("A100-80GB")
    batch_size = 1
    sequence_length = 2048
    
    models = [
        ("Llama 3 8B", LLAMA_3_8B),
        ("Llama 3 70B", LLAMA_3_70B),
        ("DeepSeek 3.2", DEEPSEEK_3_2),
        ("Mixtral 8x7B", MIXTRAL_8X7B),
    ]
    
    print(f"GPU: A100-80GB")
    print(f"Workload: batch_size={batch_size}, sequence_length={sequence_length}")
    print()
    
    for name, model in models:
        # Check if model fits in memory first
        perf = InferencePerformance(model)
        result = perf.calculate_achievable_ttft(
            system_constraints=gpu,
            batch_size=batch_size,
            sequence_length=sequence_length
        )
        
        if result.memory_utilization > 1.0:
            print(f"{name:15s}: OOM - requires {result.memory_used/(1024**3):.1f}GB")
        else:
            print(f"{name:15s}: TTFT={result.achievable_ttft*1000:7.2f}ms, "
                  f"Memory={result.memory_utilization*100:5.1f}%, "
                  f"Bottleneck={result.bottleneck_resource}")
    
    print()


def analyze_parallelism():
    """Analyze the effect of tensor parallelism"""
    print("="*80)
    print("TENSOR PARALLELISM ANALYSIS")
    print("="*80)
    print()
    
    # Use Llama 3 70B which doesn't fit on single GPU
    model = LLAMA_3_70B
    perf = InferencePerformance(model)
    gpu = SystemConstraints.from_gpu_spec("A100-80GB")
    
    batch_size = 1
    sequence_length = 2048
    
    print(f"Model: {model.model_name}")
    print(f"GPU: A100-80GB")
    print(f"Workload: batch_size={batch_size}, sequence_length={sequence_length}")
    print()
    print(f"{'TP':>4s}  {'TTFT (ms)':>10s}  {'Bottleneck':>15s}  {'Mem%':>6s}  {'Compute%':>10s}  {'Network%':>10s}")
    print("-" * 80)
    
    for tp in [1, 2, 4, 8]:
        if tp == 1:
            parallelism = ParallelismConfig()
        else:
            parallelism = ParallelismConfig(
                parallelism_type=ParallelismType.TENSOR_PARALLEL,
                tensor_parallel_size=tp
            )
        result = perf.calculate_achievable_ttft(
            system_constraints=gpu,
            batch_size=batch_size,
            sequence_length=sequence_length,
            parallelism_config=parallelism
        )
        
        print(f"{tp:4d}  {result.achievable_ttft*1000:10.2f}  "
              f"{result.bottleneck_resource:>15s}  "
              f"{result.memory_utilization*100:6.1f}  "
              f"{result.compute_utilization*100:10.1f}  "
              f"{result.network_bandwidth_utilization*100:10.1f}")
    
    print()


def analyze_mla_benefit():
    """Compare MLA vs standard attention on memory-constrained scenarios"""
    print("="*80)
    print("MLA MEMORY BENEFIT")
    print("="*80)
    print()
    
    gpu = SystemConstraints.from_gpu_spec("A100-40GB")  # Smaller memory
    
    # DeepSeek 3.2 has MLA
    model_mla = DEEPSEEK_3_2
    perf_mla = InferencePerformance(model_mla)
    
    # For comparison, let's test with different batch sizes
    sequence_length = 4096
    batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    
    print(f"Model: {model_mla.model_name} (with MLA)")
    print(f"GPU: A100-40GB")
    print(f"Sequence Length: {sequence_length}")
    print()
    print(f"{'Batch':>8s}  {'TTFT (ms)':>10s}  {'Memory %':>10s}  {'Status':>10s}")
    print("-" * 60)
    
    for batch_size in batch_sizes:
        result = perf_mla.calculate_achievable_ttft(
            system_constraints=gpu,
            batch_size=batch_size,
            sequence_length=sequence_length
        )
        
        status = "OK" if result.memory_utilization <= 1.0 else "OOM"
        print(f"{batch_size:8d}  {result.achievable_ttft*1000:10.2f}  "
              f"{result.memory_utilization*100:10.1f}  "
              f"{status:>10s}")
    
    print()
    print("Note: MLA uses 96.9% less KV cache memory than standard attention!")
    print("This enables much larger batch sizes and longer sequences.")
    print()


if __name__ == "__main__":
    analyze_single_gpu()
    compare_gpus()
    analyze_scaling()
    analyze_model_comparison()
    analyze_parallelism()
    analyze_mla_benefit()
