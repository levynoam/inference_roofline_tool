"""
Example: Decode Phase Performance Analysis

Demonstrates calculating decode performance (autoregressive token generation):
- Token-by-token generation after prefill
- Growing KV cache makes later tokens more expensive
- Calculate TPS (tokens per second) and throughput
"""

from llm_configs import LLAMA_3_8B, LLAMA_3_70B, DEEPSEEK_3_2, MIXTRAL_8X7B
from inference_performance import InferencePerformance, SystemConstraints, ParallelismConfig, ParallelismType


def analyze_basic_decode():
    """Basic decode analysis"""
    print("="*80)
    print("BASIC DECODE ANALYSIS")
    print("="*80)
    print()
    
    model = LLAMA_3_8B
    perf = InferencePerformance(model)
    gpu = SystemConstraints.from_gpu_spec("A100-80GB")
    
    # Configuration
    batch_size = 1
    prefill_length = 2048
    output_length = 512
    
    result = perf.calculate_decode_performance(
        system_constraints=gpu,
        batch_size=batch_size,
        prefill_length=prefill_length,
        output_length=output_length
    )
    
    print(f"Model: {model.model_name}")
    print(f"GPU: A100-80GB")
    print()
    print(result.summary())
    print()


def analyze_batch_size_impact():
    """Show how batch size affects decode throughput"""
    print("="*80)
    print("BATCH SIZE IMPACT ON DECODE")
    print("="*80)
    print()
    
    model = LLAMA_3_8B
    perf = InferencePerformance(model)
    gpu = SystemConstraints.from_gpu_spec("A100-80GB")
    
    prefill_length = 2048
    output_length = 512
    batch_sizes = [1, 2, 4, 8, 16, 32]
    
    print(f"Model: {model.model_name}")
    print(f"GPU: A100-80GB")
    print(f"Prefill: {prefill_length} tokens, Output: {output_length} tokens")
    print()
    print(f"{'Batch':>6s}  {'Total Time':>11s}  {'TPS/User':>10s}  {'Throughput':>11s}  {'Bottleneck':>15s}  {'Compute%':>9s}")
    print("-" * 80)
    
    for batch_size in batch_sizes:
        result = perf.calculate_decode_performance(
            system_constraints=gpu,
            batch_size=batch_size,
            prefill_length=prefill_length,
            output_length=output_length
        )
        
        print(f"{batch_size:6d}  {result.total_decode_time*1000:8.2f} ms  "
              f"{result.tokens_per_second_per_user:9.2f}  "
              f"{result.total_throughput:10.2f}  "
              f"{result.primary_bottleneck:>15s}  "
              f"{result.avg_compute_utilization*100:8.1f}%")
    
    print()


def analyze_sequence_length_cost():
    """Show how generation length affects performance"""
    print("="*80)
    print("SEQUENCE LENGTH IMPACT")
    print("="*80)
    print()
    
    model = LLAMA_3_8B
    perf = InferencePerformance(model)
    gpu = SystemConstraints.from_gpu_spec("A100-80GB")
    
    batch_size = 1
    prefill_length = 2048
    output_lengths = [128, 256, 512, 1024, 2048]
    
    print(f"Model: {model.model_name}")
    print(f"GPU: A100-80GB")
    print(f"Batch Size: {batch_size}, Prefill: {prefill_length} tokens")
    print()
    print(f"{'Output':>8s}  {'Total Time':>11s}  {'Avg Step':>10s}  {'TPS':>10s}  {'Bottleneck':>15s}")
    print("-" * 80)
    
    for output_length in output_lengths:
        result = perf.calculate_decode_performance(
            system_constraints=gpu,
            batch_size=batch_size,
            prefill_length=prefill_length,
            output_length=output_length
        )
        
        print(f"{output_length:8d}  {result.total_decode_time*1000:8.2f} ms  "
              f"{result.avg_step_time*1000:7.4f} ms  "
              f"{result.tokens_per_second_per_user:9.2f}  "
              f"{result.primary_bottleneck:>15s}")
    
    print()


def analyze_step_by_step():
    """Detailed analysis showing cost increase over steps"""
    print("="*80)
    print("STEP-BY-STEP COST ANALYSIS")
    print("="*80)
    print()
    
    model = LLAMA_3_8B
    perf = InferencePerformance(model)
    gpu = SystemConstraints.from_gpu_spec("A100-80GB")
    
    batch_size = 1
    prefill_length = 512
    output_length = 32  # Short output to show individual steps
    
    result = perf.calculate_decode_performance(
        system_constraints=gpu,
        batch_size=batch_size,
        prefill_length=prefill_length,
        output_length=output_length,
        return_step_details=True
    )
    
    print(f"Model: {model.model_name}")
    print(f"GPU: A100-80GB")
    print(f"Batch: {batch_size}, Prefill: {prefill_length}, Output: {output_length}")
    print()
    print(f"{'Step':>5s}  {'Context':>8s}  {'Time (ms)':>10s}  {'FLOPs':>10s}  {'Bottleneck':>15s}")
    print("-" * 65)
    
    # Show every 4th step to keep output manageable
    for i in range(0, len(result.step_details), 4):
        step = result.step_details[i]
        print(f"{step.step:5d}  {step.context_length:8d}  "
              f"{step.step_time*1000:10.4f}  "
              f"{step.compute_flops/1e9:9.2f}G  "
              f"{step.bottleneck:>15s}")
    
    # Show last step
    step = result.step_details[-1]
    print(f"{step.step:5d}  {step.context_length:8d}  "
          f"{step.step_time*1000:10.4f}  "
          f"{step.compute_flops/1e9:9.2f}G  "
          f"{step.bottleneck:>15s}")
    
    print()
    print(f"First step time: {result.step_details[0].step_time*1000:.4f} ms")
    print(f"Last step time:  {result.step_details[-1].step_time*1000:.4f} ms")
    print(f"Increase:        {(result.step_details[-1].step_time / result.step_details[0].step_time - 1)*100:.1f}%")
    print()


def compare_models():
    """Compare decode performance across models"""
    print("="*80)
    print("MODEL COMPARISON - DECODE")
    print("="*80)
    print()
    
    gpu = SystemConstraints.from_gpu_spec("A100-80GB")
    batch_size = 8
    prefill_length = 2048
    output_length = 512
    
    models = [
        ("Llama 3 8B", LLAMA_3_8B),
        ("DeepSeek 3.2", DEEPSEEK_3_2),
        ("Mixtral 8x7B", MIXTRAL_8X7B),
    ]
    
    print(f"GPU: A100-80GB")
    print(f"Batch: {batch_size}, Prefill: {prefill_length}, Output: {output_length}")
    print()
    print(f"{'Model':>15s}  {'Total Time':>11s}  {'TPS/User':>10s}  {'Throughput':>11s}  {'Bottleneck':>15s}")
    print("-" * 80)
    
    for name, model in models:
        perf = InferencePerformance(model)
        result = perf.calculate_decode_performance(
            system_constraints=gpu,
            batch_size=batch_size,
            prefill_length=prefill_length,
            output_length=output_length
        )
        
        print(f"{name:>15s}  {result.total_decode_time*1000:8.2f} ms  "
              f"{result.tokens_per_second_per_user:9.2f}  "
              f"{result.total_throughput:10.2f}  "
              f"{result.primary_bottleneck:>15s}")
    
    print()


def compare_gpus_decode():
    """Compare different GPUs for decode"""
    print("="*80)
    print("GPU COMPARISON - DECODE")
    print("="*80)
    print()
    
    model = LLAMA_3_8B
    perf = InferencePerformance(model)
    
    batch_size = 8
    prefill_length = 2048
    output_length = 512
    
    gpu_names = ["A100-40GB", "A100-80GB", "H100-80GB", "MI300X"]
    
    print(f"Model: {model.model_name}")
    print(f"Batch: {batch_size}, Prefill: {prefill_length}, Output: {output_length}")
    print()
    print(f"{'GPU':>12s}  {'Total Time':>11s}  {'TPS/User':>10s}  {'Throughput':>11s}  {'Speedup':>8s}")
    print("-" * 75)
    
    baseline_time = None
    for gpu_name in gpu_names:
        gpu = SystemConstraints.from_gpu_spec(gpu_name)
        result = perf.calculate_decode_performance(
            system_constraints=gpu,
            batch_size=batch_size,
            prefill_length=prefill_length,
            output_length=output_length
        )
        
        if baseline_time is None:
            baseline_time = result.total_decode_time
            speedup_str = "1.00x"
        else:
            speedup = baseline_time / result.total_decode_time
            speedup_str = f"{speedup:.2f}x"
        
        print(f"{gpu_name:>12s}  {result.total_decode_time*1000:8.2f} ms  "
              f"{result.tokens_per_second_per_user:9.2f}  "
              f"{result.total_throughput:10.2f}  "
              f"{speedup_str:>8s}")
    
    print()


def analyze_mla_decode_benefit():
    """Show MLA memory benefit during decode"""
    print("="*80)
    print("MLA BENEFIT DURING DECODE")
    print("="*80)
    print()
    
    model = DEEPSEEK_3_2
    perf = InferencePerformance(model)
    gpu = SystemConstraints.from_gpu_spec("A100-80GB")
    
    # Large batch to show KV cache benefit
    batch_sizes = [1, 4, 16, 64]
    prefill_length = 4096
    output_length = 512
    
    print(f"Model: {model.model_name} (with MLA)")
    print(f"GPU: A100-80GB")
    print(f"Prefill: {prefill_length}, Output: {output_length}")
    print()
    print(f"{'Batch':>6s}  {'Total Time':>11s}  {'TPS/User':>10s}  {'Throughput':>11s}  {'Mem%':>6s}")
    print("-" * 70)
    
    for batch_size in batch_sizes:
        result = perf.calculate_decode_performance(
            system_constraints=gpu,
            batch_size=batch_size,
            prefill_length=prefill_length,
            output_length=output_length
        )
        
        print(f"{batch_size:6d}  {result.total_decode_time*1000:8.2f} ms  "
              f"{result.tokens_per_second_per_user:9.2f}  "
              f"{result.total_throughput:10.2f}  "
              f"{result.avg_memory_utilization*100:5.1f}%")
    
    print()
    print("Note: MLA enables 32x larger batches due to KV cache compression!")
    print()


if __name__ == "__main__":
    analyze_basic_decode()
    analyze_batch_size_impact()
    analyze_sequence_length_cost()
    analyze_step_by_step()
    compare_models()
    compare_gpus_decode()
    analyze_mla_decode_benefit()
