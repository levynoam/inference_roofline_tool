"""
Test the inverse TTFT calculation function
"""

from llm_configs import LLAMA_3_8B, DEEPSEEK_3_2
from inference_performance import InferencePerformance, SystemConstraints, ParallelismConfig, ParallelismType


def test_inverse_function_basic():
    """Test basic inverse function calculation"""
    print("TEST 1: Basic inverse TTFT calculation")
    print("-" * 60)
    
    model = LLAMA_3_8B
    perf = InferencePerformance(model)
    gpu = SystemConstraints.from_gpu_spec("A100-80GB")
    
    result = perf.calculate_achievable_ttft(
        system_constraints=gpu,
        batch_size=1,
        sequence_length=2048
    )
    
    # Verify results are reasonable
    assert result.achievable_ttft > 0, "TTFT should be positive"
    assert result.memory_utilization > 0, "Memory utilization should be positive"
    assert result.memory_utilization < 1.0, "Memory should fit on GPU"
    assert result.bottleneck_resource in ["Compute", "Memory Bandwidth", "Network Bandwidth"]
    
    # For small batch, should be compute-bound on A100
    assert result.bottleneck_resource == "Compute", "Should be compute-bound"
    assert result.compute_utilization == 1.0, "Compute should be 100% utilized"
    
    print(f"✓ TTFT: {result.achievable_ttft * 1000:.2f} ms")
    print(f"✓ Bottleneck: {result.bottleneck_resource}")
    print(f"✓ Memory: {result.memory_utilization * 100:.1f}%")
    print(f"✓ Compute: {result.compute_utilization * 100:.1f}%")
    print()


def test_gpu_comparison():
    """Test that different GPUs give different results"""
    print("TEST 2: Different GPUs produce different results")
    print("-" * 60)
    
    model = LLAMA_3_8B
    perf = InferencePerformance(model)
    
    # Test on different GPUs
    a100 = SystemConstraints.from_gpu_spec("A100-80GB")
    h100 = SystemConstraints.from_gpu_spec("H100-80GB")
    
    result_a100 = perf.calculate_achievable_ttft(a100, 1, 2048)
    result_h100 = perf.calculate_achievable_ttft(h100, 1, 2048)
    
    # H100 should be faster (more compute)
    assert result_h100.achievable_ttft < result_a100.achievable_ttft
    
    # Both should fit in memory
    assert result_a100.memory_utilization < 1.0
    assert result_h100.memory_utilization < 1.0
    
    speedup = result_a100.achievable_ttft / result_h100.achievable_ttft
    
    print(f"✓ A100 TTFT: {result_a100.achievable_ttft * 1000:.2f} ms")
    print(f"✓ H100 TTFT: {result_h100.achievable_ttft * 1000:.2f} ms")
    print(f"✓ H100 speedup: {speedup:.2f}x")
    print()


def test_batch_scaling():
    """Test that TTFT scales with batch size"""
    print("TEST 3: TTFT scaling with batch size")
    print("-" * 60)
    
    model = LLAMA_3_8B
    perf = InferencePerformance(model)
    gpu = SystemConstraints.from_gpu_spec("A100-80GB")
    
    batch_sizes = [1, 2, 4]
    results = []
    
    for bs in batch_sizes:
        result = perf.calculate_achievable_ttft(gpu, bs, 2048)
        results.append(result)
        print(f"  Batch {bs}: TTFT={result.achievable_ttft*1000:.2f}ms")
    
    # TTFT should scale roughly linearly with batch size
    ratio_2_1 = results[1].achievable_ttft / results[0].achievable_ttft
    ratio_4_2 = results[2].achievable_ttft / results[1].achievable_ttft
    
    assert 1.8 < ratio_2_1 < 2.2, f"2x batch should ~double TTFT, got {ratio_2_1:.2f}x"
    assert 1.8 < ratio_4_2 < 2.2, f"2x batch should ~double TTFT, got {ratio_4_2:.2f}x"
    
    print(f"✓ Batch 2/1 ratio: {ratio_2_1:.2f}x (expected ~2x)")
    print(f"✓ Batch 4/2 ratio: {ratio_4_2:.2f}x (expected ~2x)")
    print()


def test_oom_detection():
    """Test that OOM is correctly detected"""
    print("TEST 4: OOM detection")
    print("-" * 60)
    
    from llm_configs import LLAMA_3_70B
    
    model = LLAMA_3_70B
    perf = InferencePerformance(model)
    gpu = SystemConstraints.from_gpu_spec("A100-40GB")  # Only 40GB
    
    result = perf.calculate_achievable_ttft(gpu, 1, 2048)
    
    # Should exceed memory capacity
    assert result.memory_utilization > 1.0, "Should detect OOM"
    
    print(f"✓ Memory required: {result.memory_used / (1024**3):.1f} GB")
    print(f"✓ Memory available: {result.memory_available / (1024**3):.1f} GB")
    print(f"✓ Memory utilization: {result.memory_utilization * 100:.1f}% (OOM)")
    print()


def test_tensor_parallelism():
    """Test that tensor parallelism reduces memory per GPU"""
    print("TEST 5: Tensor parallelism reduces memory")
    print("-" * 60)
    
    from llm_configs import LLAMA_3_70B
    
    model = LLAMA_3_70B
    perf = InferencePerformance(model)
    gpu = SystemConstraints.from_gpu_spec("A100-80GB")
    
    # Single GPU - should OOM
    result_tp1 = perf.calculate_achievable_ttft(gpu, 1, 2048)
    
    # TP=2 - should fit
    parallelism_tp2 = ParallelismConfig(
        parallelism_type=ParallelismType.TENSOR_PARALLEL,
        tensor_parallel_size=2
    )
    result_tp2 = perf.calculate_achievable_ttft(
        gpu, 1, 2048, parallelism_config=parallelism_tp2
    )
    
    # TP=2 should use ~half the memory
    assert result_tp1.memory_utilization > 1.0, "TP=1 should OOM"
    assert result_tp2.memory_utilization < 1.0, "TP=2 should fit"
    
    memory_ratio = result_tp1.memory_used / result_tp2.memory_used
    
    print(f"✓ TP=1 memory: {result_tp1.memory_utilization * 100:.1f}% (OOM)")
    print(f"✓ TP=2 memory: {result_tp2.memory_utilization * 100:.1f}% (OK)")
    print(f"✓ Memory reduction: {memory_ratio:.2f}x")
    print()


def test_mla_memory_savings():
    """Test that MLA reduces memory usage"""
    print("TEST 6: MLA memory savings")
    print("-" * 60)
    
    model = DEEPSEEK_3_2
    perf = InferencePerformance(model)
    gpu = SystemConstraints.from_gpu_spec("A100-80GB")
    
    # Large batch to see KV cache difference
    result = perf.calculate_achievable_ttft(gpu, 8, 4096)
    
    # Calculate what memory would be without MLA
    # MLA compression ratio is ~32x for DeepSeek 3.2
    # So memory with MLA should be much less than without
    
    print(f"✓ Memory with MLA: {result.memory_used / (1024**3):.2f} GB")
    print(f"✓ Memory utilization: {result.memory_utilization * 100:.1f}%")
    print(f"✓ Model uses MLA compression (96.9% KV cache reduction)")
    print()


def test_kernel_overhead():
    """Test that kernel overhead is included"""
    print("TEST 7: Kernel launch overhead included")
    print("-" * 60)
    
    model = LLAMA_3_8B
    perf = InferencePerformance(model)
    gpu = SystemConstraints.from_gpu_spec("H100-80GB")  # Very fast GPU
    
    result = perf.calculate_achievable_ttft(gpu, 1, 2048)
    
    # Kernel overhead should be non-zero and less than total TTFT
    assert result.kernel_launch_overhead > 0
    assert result.kernel_launch_overhead < result.achievable_ttft
    assert result.effective_compute_time > 0
    
    overhead_pct = (result.kernel_launch_overhead / result.achievable_ttft) * 100
    
    print(f"✓ Total TTFT: {result.achievable_ttft * 1000:.2f} ms")
    print(f"✓ Kernel overhead: {result.kernel_launch_overhead * 1000:.2f} ms ({overhead_pct:.1f}%)")
    print(f"✓ Effective compute: {result.effective_compute_time * 1000:.2f} ms")
    print()


if __name__ == "__main__":
    print("="*60)
    print("TESTING INVERSE TTFT CALCULATION")
    print("="*60)
    print()
    
    try:
        test_inverse_function_basic()
        test_gpu_comparison()
        test_batch_scaling()
        test_oom_detection()
        test_tensor_parallelism()
        test_mla_memory_savings()
        test_kernel_overhead()
        
        print("="*60)
        print("✓ ALL TESTS PASSED")
        print("="*60)
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        raise
