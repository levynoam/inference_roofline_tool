"""
Tests for TestPrefillResourcesExtended
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


class TestPrefillResourcesExtended:
    """Extended coverage tests for prefill resources"""
    
    def test_prefill_memory_breakdown_ratios(self):
        """Test that memory breakdown ratios are reasonable"""
        model = LLAMA_3_70B  # Large model
        perf = InferencePerformance(model)
        
        result = perf.calculate_prefill_resources(
            batch_size=32,
            sequence_length=4096,
            time_to_first_token=0.2,
            num_gpus=1,
            parallelism_config=ParallelismConfig()
        )
        
        # Weights should be largest component for large models
        assert result.memory_model_weights > result.memory_kv_cache
        assert result.memory_model_weights > result.memory_activations
        
        # All components should be significant (>1% each)
        total = result.memory_per_gpu
        assert result.memory_model_weights / total > 0.01
        assert result.memory_kv_cache / total > 0.01
        assert result.memory_activations / total > 0.01
    
    def test_prefill_with_extreme_batch_sizes(self):
        """Test prefill with very small and very large batch sizes"""
        model = LLAMA_3_8B
        perf = InferencePerformance(model)
        
        batch_sizes = [1, 128]
        results = []
        
        for bs in batch_sizes:
            result = perf.calculate_prefill_resources(
                batch_size=bs,
                sequence_length=512,
                time_to_first_token=0.1,
                num_gpus=1,
                parallelism_config=ParallelismConfig()
            )
            results.append(result)
        
        # Large batch should require proportionally more compute
        compute_ratio = results[1].compute_per_gpu / results[0].compute_per_gpu
        assert 100 < compute_ratio < 150, f"Compute ratio {compute_ratio} seems off"
    
    def test_prefill_bandwidth_vs_compute_tradeoff(self):
        """Test bandwidth requirements vs compute requirements"""
        model = MISTRAL_7B
        perf = InferencePerformance(model)
        
        # Fast TTFT = high compute, high bandwidth
        fast_result = perf.calculate_prefill_resources(
            batch_size=1,
            sequence_length=2048,
            time_to_first_token=0.01,  # Very fast
            num_gpus=1,
            parallelism_config=ParallelismConfig()
        )
        
        # Slow TTFT = lower compute, lower bandwidth
        slow_result = perf.calculate_prefill_resources(
            batch_size=1,
            sequence_length=2048,
            time_to_first_token=0.5,  # Slower
            num_gpus=1,
            parallelism_config=ParallelismConfig()
        )
        
        assert fast_result.compute_flops_per_sec > slow_result.compute_flops_per_sec
        assert fast_result.memory_bandwidth_per_gpu > slow_result.memory_bandwidth_per_gpu
    
    def test_prefill_moe_model_compute_scaling(self):
        """Test that MoE models have higher compute for expert layers"""
        model = MIXTRAL_8X7B
        perf = InferencePerformance(model)
        
        result = perf.calculate_prefill_resources(
            batch_size=4,
            sequence_length=1024,
            time_to_first_token=0.1,
            num_gpus=1,
            parallelism_config=ParallelismConfig()
        )
        
        # MoE should have significant compute requirements
        assert result.compute_flops_per_sec > 1e15  # At least 1 PFLOPS
        assert result.memory_per_gpu > 20e9  # At least 20GB memory
    
    def test_prefill_dtype_impact_on_memory(self):
        """Test that different dtypes affect memory requirements"""
        model = LLAMA_3_8B
        perf = InferencePerformance(model)
        
        dtypes = ['float16', 'int8', 'int4']
        memory_results = []
        
        for dtype in dtypes:
            result = perf.calculate_prefill_resources(
                batch_size=1,
                sequence_length=1024,
                time_to_first_token=0.1,
                num_gpus=1,
                parallelism_config=ParallelismConfig(),
                dtype_override=dtype
            )
            memory_results.append(result.memory_per_gpu)
        
        # int8 should use less memory than float16
        assert memory_results[1] < memory_results[0]
        # int4 should use less memory than int8
        assert memory_results[2] < memory_results[1]
    
    def test_prefill_parallelism_memory_distribution(self):
        """Test that parallelism properly distributes memory"""
        model = GPT3_175B
        perf = InferencePerformance(model)
        
        configs = [
            (1, ParallelismConfig()),
            (4, ParallelismConfig(parallelism_type=ParallelismType.TENSOR_PARALLEL, tensor_parallel_size=4)),
            (8, ParallelismConfig(parallelism_type=ParallelismType.TENSOR_PARALLEL, tensor_parallel_size=8))
        ]
        
        for num_gpus, parallel_config in configs:
            result = perf.calculate_prefill_resources(
                batch_size=1,
                sequence_length=1024,
                time_to_first_token=0.1,
                num_gpus=num_gpus,
                parallelism_config=parallel_config
            )
            
            # With more GPUs, memory per GPU should decrease
            assert result.memory_per_gpu > 0
            if num_gpus > 1:
                # Memory per GPU should be less than total model size
                assert result.memory_per_gpu < model.total_parameters * 2  # FP16
    
    def test_prefill_kernel_overhead_vs_compute_time(self):
        """Test relationship between kernel overhead and compute time"""
        model = LLAMA_2_7B
        perf = InferencePerformance(model)
        
        result = perf.calculate_prefill_resources(
            batch_size=16,
            sequence_length=2048,
            time_to_first_token=0.15,
            num_gpus=1,
            parallelism_config=ParallelismConfig()
        )
        
        # Kernel overhead should be small compared to total time
        overhead_fraction = result.kernel_launch_overhead / result.time_to_first_token
        assert overhead_fraction < 0.2, f"Kernel overhead is {overhead_fraction*100:.1f}% of total time"
        
        # Effective compute time + kernel overhead should be less than or equal to TTFT
        assert (result.effective_compute_time + result.kernel_launch_overhead) <= result.time_to_first_token
    
    def test_prefill_network_requirements_with_parallelism(self):
        """Test network bandwidth requirements with pipeline parallelism"""
        model = LLAMA_3_70B
        perf = InferencePerformance(model)
        
        parallel_config = ParallelismConfig(
            parallelism_type=ParallelismType.PIPELINE_PARALLEL,
            pipeline_parallel_size=4
        )
        
        result = perf.calculate_prefill_resources(
            batch_size=8,
            sequence_length=2048,
            time_to_first_token=0.2,
            num_gpus=4,
            parallelism_config=parallel_config
        )
        
        # Pipeline parallelism should require network bandwidth
        assert result.network_bandwidth_per_gpu > 0


