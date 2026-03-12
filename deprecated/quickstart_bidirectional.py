"""
Quick Start: Bidirectional Performance Modeling

This demonstrates both directions of inference performance calculation:
1. Forward: Target TTFT → Required Resources
2. Backward: Available Resources → Achievable TTFT
"""

from llm_configs import LLAMA_3_8B
from inference_performance import (
    InferencePerformance, 
    SystemConstraints, 
    ParallelismConfig
)

def main():
    # Setup
    model = LLAMA_3_8B
    perf = InferencePerformance(model)
    
    batch_size = 1
    sequence_length = 2048
    parallelism = ParallelismConfig()  # Single GPU
    
    print("="*80)
    print("BIDIRECTIONAL PERFORMANCE MODELING")
    print("="*80)
    print()
    print(f"Model: {model.model_name}")
    print(f"Workload: batch_size={batch_size}, sequence_length={sequence_length}")
    print()
    
    # ========== FORWARD DIRECTION ==========
    print("="*80)
    print("FORWARD: Target TTFT → Required Resources")
    print("="*80)
    print()
    
    # What resources do I need to achieve 100ms TTFT?
    target_ttft = 0.100  # 100ms
    
    forward_result = perf.calculate_prefill_resources(
        batch_size=batch_size,
        sequence_length=sequence_length,
        time_to_first_token=target_ttft,
        num_gpus=1,
        parallelism_config=parallelism
    )
    
    print(f"Target TTFT: {target_ttft * 1000:.2f} ms")
    print()
    print("Required Resources:")
    print(f"  Memory:          {forward_result.memory_per_gpu / (1024**3):.2f} GB")
    print(f"  Memory BW:       {forward_result.memory_bandwidth_per_gpu / (1024**3):.2f} GB/s")
    print(f"  Compute:         {forward_result.compute_flops_per_sec / 1e12:.2f} TFLOP/s")
    print(f"  Network BW:      {forward_result.network_bandwidth_per_gpu / (1024**3):.2f} GB/s")
    print()
    print(f"Analysis: {'Compute' if forward_result.compute_bound else 'Memory'}-bound")
    print(f"  Arithmetic Intensity: {forward_result.arithmetic_intensity:.2f} FLOPs/byte")
    print()
    
    # Can an A100-80GB meet these requirements?
    a100 = SystemConstraints.from_gpu_spec("A100-80GB")
    
    print("A100-80GB Specifications:")
    print(f"  Memory:          {a100.memory_capacity / (1024**3):.2f} GB")
    print(f"  Memory BW:       {a100.memory_bandwidth / (1024**3):.2f} GB/s")
    print(f"  Compute:         {a100.compute_throughput / 1e12:.2f} TFLOP/s")
    print(f"  Network BW:      {a100.network_bandwidth / (1024**3):.2f} GB/s")
    print()
    
    # Check if requirements are met
    memory_ok = forward_result.memory_per_gpu <= a100.memory_capacity
    memory_bw_ok = forward_result.memory_bandwidth_per_gpu <= a100.memory_bandwidth
    compute_ok = forward_result.compute_flops_per_sec <= a100.compute_throughput
    network_ok = forward_result.network_bandwidth_per_gpu <= a100.network_bandwidth
    
    print("Requirements Check:")
    print(f"  Memory:          {'✓' if memory_ok else '✗'} ({forward_result.memory_per_gpu / a100.memory_capacity * 100:.1f}%)")
    print(f"  Memory BW:       {'✓' if memory_bw_ok else '✗'} ({forward_result.memory_bandwidth_per_gpu / a100.memory_bandwidth * 100:.1f}%)")
    print(f"  Compute:         {'✓' if compute_ok else '✗'} ({forward_result.compute_flops_per_sec / a100.compute_throughput * 100:.1f}%)")
    print(f"  Network BW:      {'✓' if network_ok else '✗'} ({forward_result.network_bandwidth_per_gpu / a100.network_bandwidth * 100:.1f}%)")
    print()
    
    if all([memory_ok, memory_bw_ok, compute_ok, network_ok]):
        print("✓ A100-80GB can meet the 100ms TTFT target!")
    else:
        print("✗ A100-80GB cannot meet the 100ms TTFT target")
    
    print()
    
    # ========== BACKWARD DIRECTION ==========
    print("="*80)
    print("BACKWARD: Available Resources → Achievable TTFT")
    print("="*80)
    print()
    
    # What performance can I actually achieve with an A100-80GB?
    backward_result = perf.calculate_achievable_ttft(
        system_constraints=a100,
        batch_size=batch_size,
        sequence_length=sequence_length,
        parallelism_config=parallelism
    )
    
    print(backward_result.summary())
    print()
    
    # Compare
    print("="*80)
    print("COMPARISON")
    print("="*80)
    print()
    print(f"Target TTFT (forward):     {target_ttft * 1000:7.2f} ms")
    print(f"Achievable TTFT (backward): {backward_result.achievable_ttft * 1000:7.2f} ms")
    print()
    
    if backward_result.achievable_ttft <= target_ttft:
        slack = ((target_ttft - backward_result.achievable_ttft) / target_ttft) * 100
        print(f"✓ Can exceed target by {slack:.1f}% ({(target_ttft - backward_result.achievable_ttft)*1000:.2f}ms faster)")
    else:
        shortfall = ((backward_result.achievable_ttft - target_ttft) / target_ttft) * 100
        print(f"✗ Falls short of target by {shortfall:.1f}% ({(backward_result.achievable_ttft - target_ttft)*1000:.2f}ms slower)")
    
    print()
    print("Why the difference?")
    if backward_result.achievable_ttft > target_ttft:
        print(f"  The bottleneck is: {backward_result.bottleneck_resource}")
        if backward_result.bottleneck_resource == "Compute":
            print(f"  A100-80GB has {a100.compute_throughput/1e12:.0f} TFLOP/s, but we need {forward_result.compute_flops_per_sec/1e12:.0f} TFLOP/s")
            print(f"  Shortfall: {(forward_result.compute_flops_per_sec / a100.compute_throughput - 1) * 100:.1f}%")
    
    print()

if __name__ == "__main__":
    main()
