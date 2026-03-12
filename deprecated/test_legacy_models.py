"""
Test legacy models to ensure MLA additions didn't break existing functionality
"""

from llm_configs import get_model, ALL_MODELS
from inference_performance import InferencePerformance, ParallelismConfig, ParallelismType


def test_model_basic_functionality(model_key: str):
    """Test basic functionality for a model"""
    try:
        model = get_model(model_key)
        
        # Test 1: Model loads correctly
        assert model is not None, f"Model {model_key} failed to load"
        assert model.total_parameters > 0, f"Model {model_key} has invalid parameter count"
        
        # Test 2: KV cache calculation works
        kv_cache = model.get_kv_cache_size(batch_size=1, sequence_length=2048)
        assert kv_cache > 0, f"Model {model_key} has invalid KV cache size"
        
        # Test 3: Summary generation works
        summary = model.summary()
        assert len(summary) > 0, f"Model {model_key} failed to generate summary"
        
        # Test 4: Inference performance calculation works
        perf = InferencePerformance(model)
        parallelism = ParallelismConfig(ParallelismType.NONE, 1, 1, 1)
        
        resources = perf.calculate_prefill_resources(
            batch_size=1,
            sequence_length=2048,
            time_to_first_token=0.5,
            num_gpus=1,
            parallelism_config=parallelism
        )
        
        assert resources.memory_per_gpu > 0, f"Model {model_key} has invalid memory calculation"
        assert resources.compute_per_gpu > 0, f"Model {model_key} has invalid compute calculation"
        assert resources.num_kernel_launches > 0, f"Model {model_key} has invalid kernel count"
        
        return True, "PASS"
    
    except Exception as e:
        return False, f"FAIL: {str(e)}"


def test_kv_cache_consistency():
    """Test that KV cache calculations are consistent and reasonable"""
    print("=" * 80)
    print("KV CACHE CONSISTENCY TEST")
    print("=" * 80)
    
    test_cases = [
        ("llama-3-8b", 2048, 0.25, 1.1),  # (model, seq_len, expected_gb, tolerance)
        ("llama-3-70b", 2048, 0.62, 1.1),  # 80 layers, more KV cache
        ("mistral-7b", 2048, 0.25, 1.1),
        ("llama-2-7b", 2048, 1.00, 1.1),  # Full MHA has more KV cache
    ]
    
    all_passed = True
    
    print(f"\n{'Model':<20} | {'Seq Len':>8} | {'Expected':>10} | {'Actual':>10} | {'Status':>10}")
    print("-" * 75)
    
    for model_key, seq_len, expected_gb, tolerance in test_cases:
        model = get_model(model_key)
        kv_cache = model.get_kv_cache_size(batch_size=1, sequence_length=seq_len)
        actual_gb = kv_cache / (1024**3)
        
        # Check if within tolerance
        ratio = actual_gb / expected_gb
        passed = (1/tolerance) <= ratio <= tolerance
        status = "✓ PASS" if passed else "✗ FAIL"
        
        if not passed:
            all_passed = False
        
        print(f"{model.model_name:<20} | {seq_len:8} | {expected_gb:9.2f}G | {actual_gb:9.2f}G | {status:>10}")
    
    print("\n" + "=" * 80)
    return all_passed


def test_compute_scaling():
    """Test that compute scales properly with sequence length"""
    print("=" * 80)
    print("COMPUTE SCALING TEST")
    print("=" * 80)
    
    model = get_model("llama-3-8b")
    perf = InferencePerformance(model)
    parallelism = ParallelismConfig(ParallelismType.NONE, 1, 1, 1)
    
    print(f"\nModel: {model.model_name}")
    print(f"Testing compute scaling with sequence length (should be ~quadratic for attention)\n")
    
    results = []
    base_seq = 1024
    
    for seq_len in [1024, 2048, 4096]:
        resources = perf.calculate_prefill_resources(
            batch_size=1,
            sequence_length=seq_len,
            time_to_first_token=1.0,
            num_gpus=1,
            parallelism_config=parallelism
        )
        results.append((seq_len, resources.compute_per_gpu))
    
    print(f"{'Seq Len':>8} | {'Compute':>12} | {'Ratio to 1024':>15}")
    print("-" * 45)
    
    base_compute = results[0][1]
    all_passed = True
    
    for seq_len, compute in results:
        ratio = compute / base_compute
        seq_multiplier = seq_len / base_seq
        expected_ratio_quadratic = seq_multiplier ** 2
        
        # Ratio should be between linear and quadratic growth
        # Linear: seq_multiplier, Quadratic: seq_multiplier^2
        # Actual is mix: reasonable range is [seq_multiplier, seq_multiplier^2 * 1.5]
        reasonable = seq_multiplier <= ratio <= expected_ratio_quadratic * 1.5
        status = "✓" if reasonable else "✗"
        
        if not reasonable:
            all_passed = False
        
        print(f"{seq_len:8} | {compute/1e12:11.2f}T | {ratio:14.2f}x {status}")
    
    print("\n" + "=" * 80)
    return all_passed


def test_parallelism_splitting():
    """Test that parallelism correctly splits memory"""
    print("=" * 80)
    print("PARALLELISM MEMORY SPLITTING TEST")
    print("=" * 80)
    
    model = get_model("llama-3-8b")
    perf = InferencePerformance(model)
    
    batch_size = 4
    sequence_length = 2048
    time_to_first_token = 0.5
    
    # Test different parallelism strategies
    configs = [
        ("Single GPU", ParallelismConfig(ParallelismType.NONE, 1, 1, 1), 1),
        ("TP=2", ParallelismConfig(ParallelismType.TENSOR_PARALLEL, 2, 1, 1), 2),
        ("TP=4", ParallelismConfig(ParallelismType.TENSOR_PARALLEL, 4, 1, 1), 4),
        ("PP=2", ParallelismConfig(ParallelismType.PIPELINE_PARALLEL, 1, 2, 1), 2),
    ]
    
    print(f"\nModel: {model.model_name}")
    print(f"Workload: batch={batch_size}, seq_len={sequence_length}\n")
    print(f"{'Config':<15} | {'Model Mem':>10} | {'KV Cache':>10} | {'Status':>10}")
    print("-" * 60)
    
    single_gpu_memory = None
    all_passed = True
    
    for name, config, gpus in configs:
        resources = perf.calculate_prefill_resources(
            batch_size, sequence_length, time_to_first_token, gpus, config
        )
        
        model_mem_gb = resources.memory_model_weights / (1024**3)
        kv_cache_gb = resources.memory_kv_cache / (1024**3)
        
        if single_gpu_memory is None:
            single_gpu_memory = resources.memory_model_weights
            status = "✓ PASS"
        else:
            # For TP and PP, model memory should be split
            if config.parallelism_type == ParallelismType.TENSOR_PARALLEL:
                expected = single_gpu_memory / config.tensor_parallel_size
                passed = abs(resources.memory_model_weights - expected) < expected * 0.01
            elif config.parallelism_type == ParallelismType.PIPELINE_PARALLEL:
                expected = single_gpu_memory / config.pipeline_parallel_size
                passed = abs(resources.memory_model_weights - expected) < expected * 0.01
            else:
                passed = True
            
            status = "✓ PASS" if passed else "✗ FAIL"
            if not passed:
                all_passed = False
        
        print(f"{name:<15} | {model_mem_gb:9.2f}G | {kv_cache_gb:9.2f}G | {status:>10}")
    
    print("\n" + "=" * 80)
    return all_passed


def test_moe_vs_dense():
    """Test that MoE models behave differently from dense models"""
    print("=" * 80)
    print("MoE vs DENSE MODEL TEST")
    print("=" * 80)
    
    dense_model = get_model("mistral-7b")
    moe_model = get_model("mixtral-8x7b")
    
    print(f"\nDense Model: {dense_model.model_name}")
    print(f"  Total params: {dense_model.total_parameters / 1e9:.1f}B")
    print(f"  Active params: {dense_model.active_parameters / 1e9:.1f}B")
    print(f"  Is MoE: {dense_model.is_moe}")
    
    print(f"\nMoE Model: {moe_model.model_name}")
    print(f"  Total params: {moe_model.total_parameters / 1e9:.1f}B")
    print(f"  Active params: {moe_model.active_parameters / 1e9:.1f}B")
    print(f"  Is MoE: {moe_model.is_moe}")
    print(f"  Experts: {moe_model.moe_config.num_experts}")
    print(f"  Active experts: {moe_model.moe_config.num_experts_per_token}")
    
    # Test kernel counts
    dense_perf = InferencePerformance(dense_model)
    moe_perf = InferencePerformance(moe_model)
    parallelism = ParallelismConfig(ParallelismType.NONE, 1, 1, 1)
    
    dense_kernels = dense_perf.calculate_num_kernel_launches(parallelism)
    moe_kernels = moe_perf.calculate_num_kernel_launches(parallelism)
    
    print(f"\nKernel Launches:")
    print(f"  Dense: {dense_kernels}")
    print(f"  MoE: {moe_kernels}")
    
    # MoE should have more kernels (12 per layer vs 9 for dense)
    # With 32 layers: dense = 3 + 32*9 = 291, MoE = 3 + 32*12 = 387
    passed = moe_kernels > dense_kernels
    status = "✓ PASS" if passed else "✗ FAIL"
    
    print(f"\nMoE has more kernels than dense: {status}")
    
    # Test that active params < total params for MoE
    passed_active = moe_model.active_parameters < moe_model.total_parameters
    status_active = "✓ PASS" if passed_active else "✗ FAIL"
    print(f"MoE active < total params: {status_active}")
    
    print("\n" + "=" * 80)
    return passed and passed_active


def test_kernel_overhead():
    """Test that kernel launch overhead is calculated"""
    print("=" * 80)
    print("KERNEL LAUNCH OVERHEAD TEST")
    print("=" * 80)
    
    model = get_model("llama-3-8b")
    perf = InferencePerformance(model)
    parallelism = ParallelismConfig(ParallelismType.NONE, 1, 1, 1)
    
    resources = perf.calculate_prefill_resources(
        batch_size=1,
        sequence_length=2048,
        time_to_first_token=0.5,
        num_gpus=1,
        parallelism_config=parallelism
    )
    
    print(f"\nModel: {model.model_name}")
    print(f"Kernel launch latency: {model.kernel_launch_latency * 1e6:.1f} µs")
    print(f"Number of kernels: {resources.num_kernel_launches}")
    print(f"Total overhead: {resources.kernel_launch_overhead * 1000:.2f} ms")
    print(f"Effective compute time: {resources.effective_compute_time * 1000:.2f} ms")
    
    # Validate calculations
    expected_overhead = resources.num_kernel_launches * model.kernel_launch_latency
    overhead_correct = abs(resources.kernel_launch_overhead - expected_overhead) < 1e-9
    
    expected_compute_time = resources.time_to_first_token - resources.kernel_launch_overhead
    compute_time_correct = abs(resources.effective_compute_time - expected_compute_time) < 1e-9
    
    print(f"\nOverhead calculation: {'✓ PASS' if overhead_correct else '✗ FAIL'}")
    print(f"Effective time calculation: {'✓ PASS' if compute_time_correct else '✗ FAIL'}")
    
    print("\n" + "=" * 80)
    return overhead_correct and compute_time_correct


def run_all_tests():
    """Run comprehensive test suite"""
    print("\n" + "=" * 80)
    print("LEGACY MODEL COMPATIBILITY TEST SUITE")
    print("=" * 80)
    print()
    
    # Test all models for basic functionality
    print("=" * 80)
    print("BASIC FUNCTIONALITY TEST - ALL MODELS")
    print("=" * 80)
    print()
    
    all_models_passed = True
    results = []
    
    for model_key in ALL_MODELS.keys():
        # Skip the new MLA model for legacy tests
        if model_key == "deepseek-3.2":
            continue
        
        passed, message = test_model_basic_functionality(model_key)
        results.append((model_key, passed, message))
        
        if not passed:
            all_models_passed = False
    
    # Print results
    print(f"{'Model':<20} | {'Status':<50}")
    print("-" * 75)
    for model_key, passed, message in results:
        status_symbol = "✓" if passed else "✗"
        print(f"{model_key:<20} | {status_symbol} {message}")
    
    print()
    
    # Run specific tests
    test_results = {
        "All Models Basic": all_models_passed,
        "KV Cache Consistency": test_kv_cache_consistency(),
        "Compute Scaling": test_compute_scaling(),
        "Parallelism Splitting": test_parallelism_splitting(),
        "MoE vs Dense": test_moe_vs_dense(),
        "Kernel Overhead": test_kernel_overhead(),
    }
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print()
    
    all_passed = True
    for test_name, passed in test_results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:<30}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✓ ALL TESTS PASSED - Legacy models work correctly!")
    else:
        print("✗ SOME TESTS FAILED - Please review")
    print("=" * 80)
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
