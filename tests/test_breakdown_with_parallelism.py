"""
Tests for TestBreakdownWithParallelism
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


class TestBreakdownWithParallelism:
    """Test that breakdowns work correctly with parallelism"""
    
    def test_tensor_parallel_breakdown(self):
        """Test breakdowns with tensor parallelism"""
        model = LLAMA_3_70B  # Use larger model that benefits from TP
        perf = InferencePerformance(model)
        
        gpu = SystemConstraints(
            memory_capacity=80e9,
            memory_bandwidth=2039e9,
            compute_throughput=312e12,
            network_bandwidth=600e9
        )
        
        parallel_config = ParallelismConfig(
            parallelism_type=ParallelismType.TENSOR_PARALLEL,
            tensor_parallel_size=4,
            pipeline_parallel_size=1,
            data_parallel_size=1
        )
        
        result = perf.calculate_achievable_ttft(
            system_constraints=gpu,
            batch_size=1,
            sequence_length=2048,
            parallelism_config=parallel_config
        )
        
        # With TP=4, weights should be divided across GPUs
        # KV cache should also be divided
        assert result.memory_weights > 0
        assert result.memory_kv_cache > 0
        
        # Compute breakdown should still work
        assert result.compute_attention > 0
        assert result.compute_ffn > 0


