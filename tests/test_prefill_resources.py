"""
Tests for TestPrefillResources
"""

import pytest
from llm_configs import (
    LLAMA_3_8B, LLAMA_3_70B, LLAMA_2_7B,
    DEEPSEEK_V3, DEEPSEEK_3_2,
    MISTRAL_7B, MIXTRAL_8X7B, GPT3_175B
)
from llm_architecture import LLMArchitecture, AttentionConfig, AttentionType, ActivationType
from inference_performance import (
    InferencePerformance,
    SystemConstraints,
    ParallelismConfig,
    ParallelismType,
    PrefillResources,
    ResourceUtilization,
    DecodePerformance
)


class TestPrefillResources:
    """Test calculate_prefill_resources() with various parameters"""
    
    def test_basic_prefill(self):
        """Test basic prefill calculation"""
        model = LLAMA_3_8B
        perf = InferencePerformance(model)
        
        result = perf.calculate_prefill_resources(
            batch_size=1,
            sequence_length=2048,
            time_to_first_token=0.1,
            num_gpus=1,
            parallelism_config=ParallelismConfig()
        )
        
        assert isinstance(result, PrefillResources)
        assert result.time_to_first_token == 0.1
        assert result.memory_per_gpu > 0
        assert result.compute_flops_per_sec > 0
    
    def test_all_models_prefill(self, all_models):
        """Test prefill with all model configurations"""
        for model in all_models:
            perf = InferencePerformance(model)
            
            result = perf.calculate_prefill_resources(
                batch_size=1,
                sequence_length=1024,
                time_to_first_token=0.05,
                num_gpus=1,
                parallelism_config=ParallelismConfig()
            )
            
            # Verify basic properties
            assert result.memory_per_gpu > 0, f"Model {model.model_name} has zero memory"
            assert result.compute_flops_per_sec > 0, f"Model {model.model_name} has zero compute"
            assert result.memory_bandwidth_per_gpu > 0
            
            # Verify memory breakdown
            assert result.memory_model_weights > 0
            assert result.memory_kv_cache > 0
            assert result.memory_activations > 0
            assert (result.memory_model_weights + result.memory_kv_cache + 
                   result.memory_activations) == pytest.approx(result.memory_per_gpu, rel=0.01)
    
    def test_batch_size_scaling_prefill(self, batch_sizes):
        """Test that compute scales with batch size"""
        model = LLAMA_3_8B
        perf = InferencePerformance(model)
        
        results = []
        for batch_size in batch_sizes[:4]:  # Test first 4 batch sizes
            result = perf.calculate_prefill_resources(
                batch_size=batch_size,
                sequence_length=1024,
                time_to_first_token=0.1,
                num_gpus=1,
                parallelism_config=ParallelismConfig()
            )
            results.append(result)
        
        # Compute should scale roughly linearly with batch size
        for i in range(1, len(results)):
            ratio = results[i].compute_per_gpu / results[0].compute_per_gpu
            expected_ratio = batch_sizes[i] / batch_sizes[0]
            assert ratio == pytest.approx(expected_ratio, rel=0.1)
    
    def test_sequence_length_scaling_prefill(self, sequence_lengths):
        """Test that compute scales with sequence length"""
        model = LLAMA_3_8B
        perf = InferencePerformance(model)
        
        results = []
        for seq_len in sequence_lengths[:4]:  # Test first 4 lengths
            result = perf.calculate_prefill_resources(
                batch_size=1,
                sequence_length=seq_len,
                time_to_first_token=0.1,
                num_gpus=1,
                parallelism_config=ParallelismConfig()
            )
            results.append(result)
        
        # Compute should scale between linear and quadratic with sequence length
        for i in range(1, len(results)):
            ratio = results[i].compute_per_gpu / results[0].compute_per_gpu
            seq_ratio = sequence_lengths[i] / sequence_lengths[0]
            # Should be between seq_ratio and seq_ratio^2
            assert seq_ratio <= ratio <= seq_ratio * seq_ratio * 1.5
    
    def test_ttft_scaling_prefill(self):
        """Test that compute scales inversely with TTFT"""
        model = LLAMA_3_8B
        perf = InferencePerformance(model)
        
        ttfts = [0.05, 0.1, 0.2]
        results = []
        
        for ttft in ttfts:
            result = perf.calculate_prefill_resources(
                batch_size=1,
                sequence_length=1024,
                time_to_first_token=ttft,
                num_gpus=1,
                parallelism_config=ParallelismConfig()
            )
            results.append(result)
        
        # Shorter TTFT requires more compute power
        assert results[0].compute_flops_per_sec > results[1].compute_flops_per_sec
        assert results[1].compute_flops_per_sec > results[2].compute_flops_per_sec
        
        # Should scale inversely
        ratio_01 = results[0].compute_flops_per_sec / results[1].compute_flops_per_sec
        ttft_ratio = ttfts[1] / ttfts[0]
        assert ratio_01 == pytest.approx(ttft_ratio, rel=0.05)
    
    def test_parallelism_prefill(self, parallelism_configs):
        """Test prefill with different parallelism configurations"""
        model = LLAMA_3_8B
        perf = InferencePerformance(model)
        
        for parallel_config in parallelism_configs:
            num_gpus = parallel_config.total_gpus
            
            result = perf.calculate_prefill_resources(
                batch_size=1,
                sequence_length=1024,
                time_to_first_token=0.1,
                num_gpus=num_gpus,
                parallelism_config=parallel_config
            )
            
            # Memory per GPU should decrease with more GPUs
            assert result.memory_per_gpu > 0
            
            # Compute per GPU should decrease with tensor parallelism
            if parallel_config.tensor_parallel_size > 1:
                assert result.compute_per_gpu < 1e15  # Reasonable compute range
    
    def test_kernel_overhead_prefill(self):
        """Test kernel launch overhead is calculated"""
        model = LLAMA_3_8B
        perf = InferencePerformance(model)
        
        result = perf.calculate_prefill_resources(
            batch_size=1,
            sequence_length=1024,
            time_to_first_token=0.1,
            num_gpus=1,
            parallelism_config=ParallelismConfig()
        )
        
        assert result.num_kernel_launches > 0
        assert result.kernel_launch_overhead > 0
        assert result.kernel_launch_overhead < result.time_to_first_token
        assert result.effective_compute_time > 0
        assert result.effective_compute_time < result.time_to_first_token
    
    def test_arithmetic_intensity_prefill(self):
        """Test arithmetic intensity calculation"""
        model = LLAMA_3_8B
        perf = InferencePerformance(model)
        
        result = perf.calculate_prefill_resources(
            batch_size=1,
            sequence_length=2048,
            time_to_first_token=0.1,
            num_gpus=1,
            parallelism_config=ParallelismConfig()
        )
        
        assert result.arithmetic_intensity > 0
        assert isinstance(result.compute_bound, bool)
        
        # Prefill should typically be compute-bound
        assert result.compute_bound == True


# =============================================================================
# Test calculate_achievable_ttft() - Backward Direction
# =============================================================================

