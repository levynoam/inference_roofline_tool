"""
Complete Inference Analysis: Prefill + Decode

This demonstrates end-to-end inference performance modeling:
1. Prefill: Process input prompt
2. Decode: Generate output tokens
3. Total: Complete request latency and throughput
"""

from llm_configs import LLAMA_3_8B
from inference_performance import InferencePerformance, SystemConstraints


def analyze_complete_inference():
    """Analyze complete inference request (prefill + decode)"""
    print("="*80)
    print("COMPLETE INFERENCE ANALYSIS: Prefill + Decode")
    print("="*80)
    print()
    
    model = LLAMA_3_8B
    perf = InferencePerformance(model)
    gpu = SystemConstraints.from_gpu_spec("A100-80GB")
    
    # Configuration
    batch_size = 8
    input_length = 2048   # Prompt length
    output_length = 512    # Generated tokens
    
    print(f"Model: {model.model_name}")
    print(f"GPU: A100-80GB")
    print(f"Batch Size: {batch_size}")
    print(f"Input Length: {input_length} tokens")
    print(f"Output Length: {output_length} tokens")
    print()
    
    # === PREFILL PHASE ===
    print("-" * 80)
    print("PHASE 1: PREFILL (Process Input Prompt)")
    print("-" * 80)
    
    prefill_result = perf.calculate_achievable_ttft(
        system_constraints=gpu,
        batch_size=batch_size,
        sequence_length=input_length
    )
    
    print(f"Time to First Token: {prefill_result.achievable_ttft * 1000:.2f} ms")
    print(f"Bottleneck: {prefill_result.bottleneck_resource}")
    print(f"Compute Utilization: {prefill_result.compute_utilization * 100:.1f}%")
    print(f"Memory BW Utilization: {prefill_result.memory_bandwidth_utilization * 100:.1f}%")
    print()
    
    # === DECODE PHASE ===
    print("-" * 80)
    print("PHASE 2: DECODE (Generate Output Tokens)")
    print("-" * 80)
    
    decode_result = perf.calculate_decode_performance(
        system_constraints=gpu,
        batch_size=batch_size,
        prefill_length=input_length,
        output_length=output_length
    )
    
    print(f"Total Decode Time: {decode_result.total_decode_time * 1000:.2f} ms")
    print(f"Tokens per Second: {decode_result.tokens_per_second_per_user:.2f} tokens/sec")
    print(f"Total Throughput: {decode_result.total_throughput:.2f} tokens/sec")
    print(f"Bottleneck: {decode_result.primary_bottleneck}")
    print(f"Avg Compute Utilization: {decode_result.avg_compute_utilization * 100:.1f}%")
    print(f"Avg Memory BW Utilization: {decode_result.avg_memory_bw_utilization * 100:.1f}%")
    print()
    
    # === TOTAL ===
    print("=" * 80)
    print("TOTAL REQUEST ANALYSIS")
    print("=" * 80)
    
    total_time = prefill_result.achievable_ttft + decode_result.total_decode_time
    total_tokens = input_length + output_length
    
    # Per-request metrics
    time_per_request = total_time  # For one request in the batch
    tokens_per_request = output_length  # Output tokens per request
    
    # System throughput metrics
    requests_per_second = batch_size / total_time
    tokens_per_second_system = (batch_size * output_length) / total_time
    
    # Cost breakdown
    prefill_percentage = (prefill_result.achievable_ttft / total_time) * 100
    decode_percentage = (decode_result.total_decode_time / total_time) * 100
    
    print()
    print("Per-Request Metrics:")
    print(f"  Total Latency:       {time_per_request * 1000:.2f} ms")
    print(f"    Prefill:           {prefill_result.achievable_ttft * 1000:.2f} ms ({prefill_percentage:.1f}%)")
    print(f"    Decode:            {decode_result.total_decode_time * 1000:.2f} ms ({decode_percentage:.1f}%)")
    print(f"  Output Tokens:       {tokens_per_request}")
    print(f"  Time per Token:      {(decode_result.total_decode_time / output_length) * 1000:.4f} ms")
    print()
    
    print("System Throughput:")
    print(f"  Requests/sec:        {requests_per_second:.2f} req/sec")
    print(f"  Tokens/sec:          {tokens_per_second_system:.2f} tokens/sec")
    print(f"  User Throughput:     {decode_result.tokens_per_second_per_user:.2f} tokens/sec/user")
    print()
    
    print("Resource Utilization:")
    print(f"  Prefill Phase:")
    print(f"    Compute:           {prefill_result.compute_utilization * 100:.1f}%")
    print(f"    Memory BW:         {prefill_result.memory_bandwidth_utilization * 100:.1f}%")
    print(f"  Decode Phase:")
    print(f"    Compute:           {decode_result.avg_compute_utilization * 100:.1f}%")
    print(f"    Memory BW:         {decode_result.avg_memory_bw_utilization * 100:.1f}%")
    print()
    
    print("Performance Characteristics:")
    if prefill_result.compute_utilization > 0.8:
        print(f"  ✓ Prefill is compute-bound ({prefill_result.bottleneck_resource})")
    else:
        print(f"  ⚠ Prefill bottleneck: {prefill_result.bottleneck_resource}")
    
    if decode_result.avg_memory_bw_utilization > 0.8:
        print(f"  ✓ Decode is memory bandwidth-bound ({decode_result.primary_bottleneck})")
    else:
        print(f"  ⚠ Decode bottleneck: {decode_result.primary_bottleneck}")
    print()


def compare_workload_patterns():
    """Compare different workload patterns"""
    print("="*80)
    print("WORKLOAD PATTERN COMPARISON")
    print("="*80)
    print()
    
    model = LLAMA_3_8B
    perf = InferencePerformance(model)
    gpu = SystemConstraints.from_gpu_spec("A100-80GB")
    
    batch_size = 8
    
    # Different workload patterns
    workloads = [
        ("Short Q&A", 512, 128),      # Quick responses
        ("Code Gen", 1024, 512),       # Medium code generation
        ("Article", 2048, 1024),       # Long-form content
        ("Summary", 4096, 256),        # Summarization
    ]
    
    print(f"Model: {model.model_name}, GPU: A100-80GB, Batch: {batch_size}")
    print()
    print(f"{'Workload':>12s}  {'Input':>6s}  {'Output':>7s}  {'Total':>10s}  {'Prefill%':>9s}  {'Decode%':>8s}  {'TPS':>8s}")
    print("-" * 85)
    
    for name, input_len, output_len in workloads:
        # Prefill
        prefill = perf.calculate_achievable_ttft(
            system_constraints=gpu,
            batch_size=batch_size,
            sequence_length=input_len
        )
        
        # Decode
        decode = perf.calculate_decode_performance(
            system_constraints=gpu,
            batch_size=batch_size,
            prefill_length=input_len,
            output_length=output_len
        )
        
        total_time = prefill.achievable_ttft + decode.total_decode_time
        prefill_pct = (prefill.achievable_ttft / total_time) * 100
        decode_pct = (decode.total_decode_time / total_time) * 100
        
        print(f"{name:>12s}  {input_len:6d}  {output_len:7d}  "
              f"{total_time*1000:7.2f} ms  {prefill_pct:8.1f}%  {decode_pct:7.1f}%  "
              f"{decode.tokens_per_second_per_user:7.2f}")
    
    print()
    print("Insights:")
    print("  • Longer outputs → Decode phase dominates (90%+ of time)")
    print("  • Short outputs → Prefill takes significant time")
    print("  • TPS is fairly constant (~110 tokens/sec) - decode bound!")
    print()


def analyze_batch_size_tradeoff():
    """Analyze batch size trade-off between latency and throughput"""
    print("="*80)
    print("BATCH SIZE TRADE-OFF: Latency vs Throughput")
    print("="*80)
    print()
    
    model = LLAMA_3_8B
    perf = InferencePerformance(model)
    gpu = SystemConstraints.from_gpu_spec("A100-80GB")
    
    input_length = 2048
    output_length = 512
    batch_sizes = [1, 2, 4, 8, 16, 32]
    
    print(f"Model: {model.model_name}, GPU: A100-80GB")
    print(f"Input: {input_length}, Output: {output_length}")
    print()
    print(f"{'Batch':>6s}  {'Latency':>11s}  {'TPS/User':>10s}  {'Total TPS':>10s}  {'Req/sec':>9s}  {'Efficiency':>11s}")
    print("-" * 80)
    
    baseline_throughput = None
    
    for batch_size in batch_sizes:
        # Prefill
        prefill = perf.calculate_achievable_ttft(
            system_constraints=gpu,
            batch_size=batch_size,
            sequence_length=input_length
        )
        
        # Decode
        decode = perf.calculate_decode_performance(
            system_constraints=gpu,
            batch_size=batch_size,
            prefill_length=input_length,
            output_length=output_length
        )
        
        total_latency = prefill.achievable_ttft + decode.total_decode_time
        total_throughput = decode.total_throughput
        requests_per_sec = batch_size / total_latency
        
        if baseline_throughput is None:
            baseline_throughput = total_throughput
            efficiency = 100.0
        else:
            # Efficiency = actual throughput / ideal linear scaling
            ideal_throughput = baseline_throughput * batch_size
            efficiency = (total_throughput / ideal_throughput) * 100
        
        print(f"{batch_size:6d}  {total_latency*1000:8.2f} ms  "
              f"{decode.tokens_per_second_per_user:9.2f}  "
              f"{total_throughput:9.2f}  "
              f"{requests_per_sec:8.2f}  "
              f"{efficiency:10.1f}%")
    
    print()
    print("Trade-offs:")
    print("  • Batch↑ → Latency↑ (users wait longer)")
    print("  • Batch↑ → Throughput↑ (system serves more tokens)")
    print("  • Batch↑ → Efficiency↓ (memory BW saturation)")
    print()
    print("Recommendation:")
    print("  • Interactive (low latency): Batch 1-4")
    print("  • High throughput (batch): Batch 16-32")
    print()


if __name__ == "__main__":
    analyze_complete_inference()
    print("\n\n")
    compare_workload_patterns()
    print("\n\n")
    analyze_batch_size_tradeoff()
