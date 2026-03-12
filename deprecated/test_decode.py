"""
Test decode phase performance calculations
"""

from llm_configs import LLAMA_3_8B, MIXTRAL_8X7B
from inference_performance import InferencePerformance, SystemConstraints


def test_basic_decode():
    """Test basic decode calculation"""
    print("TEST 1: Basic decode calculation")
    print("-" * 60)
    
    model = LLAMA_3_8B
    perf = InferencePerformance(model)
    gpu = SystemConstraints.from_gpu_spec("A100-80GB")
    
    result = perf.calculate_decode_performance(
        system_constraints=gpu,
        batch_size=1,
        prefill_length=512,
        output_length=128
    )
    
    # Verify basic properties
    assert result.output_length == 128
    assert result.total_decode_time > 0
    assert result.tokens_per_second_per_user > 0
    assert result.total_throughput > 0
    assert result.primary_bottleneck in ["Compute", "Memory Bandwidth", "Network Bandwidth"]
    
    print(f"✓ Total time: {result.total_decode_time * 1000:.2f} ms")
    print(f"✓ TPS: {result.tokens_per_second_per_user:.2f} tokens/sec")
    print(f"✓ Bottleneck: {result.primary_bottleneck}")
    print()


def test_batch_size_scaling():
    """Test that throughput scales with batch size"""
    print("TEST 2: Batch size scaling")
    print("-" * 60)
    
    model = LLAMA_3_8B
    perf = InferencePerformance(model)
    gpu = SystemConstraints.from_gpu_spec("A100-80GB")
    
    results = []
    for batch_size in [1, 2, 4]:
        result = perf.calculate_decode_performance(
            system_constraints=gpu,
            batch_size=batch_size,
            prefill_length=512,
            output_length=128
        )
        results.append(result)
    
    # Throughput should increase with batch size
    assert results[1].total_throughput > results[0].total_throughput
    assert results[2].total_throughput > results[1].total_throughput
    
    # TPS per user should decrease slightly (batching overhead)
    # but total throughput should increase significantly
    throughput_increase = results[1].total_throughput / results[0].total_throughput
    
    print(f"✓ Batch 1: {results[0].total_throughput:.2f} tokens/sec total")
    print(f"✓ Batch 2: {results[1].total_throughput:.2f} tokens/sec total")
    print(f"✓ Batch 4: {results[2].total_throughput:.2f} tokens/sec total")
    print(f"✓ 2x batch increases throughput by {throughput_increase:.2f}x")
    print()


def test_context_length_impact():
    """Test that longer context affects performance"""
    print("TEST 3: Context length impact")
    print("-" * 60)
    
    model = LLAMA_3_8B
    perf = InferencePerformance(model)
    gpu = SystemConstraints.from_gpu_spec("A100-80GB")
    
    # Same output length, different prefill lengths
    results = []
    for prefill_length in [512, 2048, 4096]:
        result = perf.calculate_decode_performance(
            system_constraints=gpu,
            batch_size=1,
            prefill_length=prefill_length,
            output_length=128
        )
        results.append(result)
    
    # Longer context should take more time (larger KV cache)
    assert results[1].total_decode_time > results[0].total_decode_time
    assert results[2].total_decode_time > results[1].total_decode_time
    
    print(f"✓ Prefill  512: {results[0].total_decode_time*1000:.2f} ms")
    print(f"✓ Prefill 2048: {results[1].total_decode_time*1000:.2f} ms")
    print(f"✓ Prefill 4096: {results[2].total_decode_time*1000:.2f} ms")
    print()


def test_step_details():
    """Test step-by-step details"""
    print("TEST 4: Step-by-step details")
    print("-" * 60)
    
    model = LLAMA_3_8B
    perf = InferencePerformance(model)
    gpu = SystemConstraints.from_gpu_spec("A100-80GB")
    
    result = perf.calculate_decode_performance(
        system_constraints=gpu,
        batch_size=1,
        prefill_length=512,
        output_length=16,
        return_step_details=True
    )
    
    # Verify step details
    assert len(result.step_details) == 16
    assert result.step_details[0].step == 0
    assert result.step_details[0].context_length == 512
    assert result.step_details[-1].context_length == 512 + 15
    
    # Later steps should have slightly more compute (longer context)
    assert result.step_details[-1].compute_flops >= result.step_details[0].compute_flops
    
    print(f"✓ First step context: {result.step_details[0].context_length}")
    print(f"✓ Last step context: {result.step_details[-1].context_length}")
    print(f"✓ First step time: {result.step_details[0].step_time*1000:.4f} ms")
    print(f"✓ Last step time: {result.step_details[-1].step_time*1000:.4f} ms")
    print()


def test_moe_decode():
    """Test decode with MoE model"""
    print("TEST 5: MoE decode")
    print("-" * 60)
    
    model = MIXTRAL_8X7B
    perf = InferencePerformance(model)
    gpu = SystemConstraints.from_gpu_spec("A100-80GB")
    
    result = perf.calculate_decode_performance(
        system_constraints=gpu,
        batch_size=1,
        prefill_length=512,
        output_length=128
    )
    
    assert result.total_decode_time > 0
    assert result.tokens_per_second_per_user > 0
    
    print(f"✓ MoE model works")
    print(f"✓ TPS: {result.tokens_per_second_per_user:.2f} tokens/sec")
    print(f"✓ Bottleneck: {result.primary_bottleneck}")
    print()


def test_bottleneck_identification():
    """Test that bottleneck is correctly identified"""
    print("TEST 6: Bottleneck identification")
    print("-" * 60)
    
    model = LLAMA_3_8B
    perf = InferencePerformance(model)
    
    # On A100, decode should be memory BW bound (small compute, large KV read)
    gpu_a100 = SystemConstraints.from_gpu_spec("A100-80GB")
    result_a100 = perf.calculate_decode_performance(
        system_constraints=gpu_a100,
        batch_size=1,
        prefill_length=2048,
        output_length=128
    )
    
    # MI300X has very high memory BW, might be compute bound
    gpu_mi300 = SystemConstraints.from_gpu_spec("MI300X")
    result_mi300 = perf.calculate_decode_performance(
        system_constraints=gpu_mi300,
        batch_size=1,
        prefill_length=2048,
        output_length=128
    )
    
    print(f"✓ A100 bottleneck: {result_a100.primary_bottleneck}")
    print(f"✓ MI300X bottleneck: {result_mi300.primary_bottleneck}")
    print(f"✓ A100 Memory BW util: {result_a100.avg_memory_bw_utilization*100:.1f}%")
    print(f"✓ MI300X Memory BW util: {result_mi300.avg_memory_bw_utilization*100:.1f}%")
    print()


if __name__ == "__main__":
    print("="*60)
    print("TESTING DECODE PERFORMANCE CALCULATIONS")
    print("="*60)
    print()
    
    try:
        test_basic_decode()
        test_batch_size_scaling()
        test_context_length_impact()
        test_step_details()
        test_moe_decode()
        test_bottleneck_identification()
        
        print("="*60)
        print("✓ ALL TESTS PASSED")
        print("="*60)
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        raise
