"""
Tests for TestComputeBreakdown
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


class TestComputeBreakdown:
    """Test compute breakdown into attention, FFN, and other"""
    
    def test_ttft_compute_breakdown(self):
        """Test that TTFT compute breakdown is correctly calculated"""
        model = LLAMA_3_8B
        perf = InferencePerformance(model)
        
        gpu = SystemConstraints(
            memory_capacity=80e9,
            memory_bandwidth=2039e9,
            compute_throughput=312e12,
            network_bandwidth=600e9
        )
        
        result = perf.calculate_achievable_ttft(
            system_constraints=gpu,
            batch_size=1,
            sequence_length=2048
        )
        
        # Check that breakdown components exist
        assert hasattr(result, 'compute_attention')
        assert hasattr(result, 'compute_ffn')
        assert hasattr(result, 'compute_other')
        
        # Check reasonable values
        assert result.compute_attention > 0
        assert result.compute_ffn > 0
        assert result.compute_other > 0
    
    def test_decode_compute_breakdown(self):
        """Test that decode compute breakdown is correctly calculated"""
        model = LLAMA_3_8B
        perf = InferencePerformance(model)
        
        gpu = SystemConstraints(
            memory_capacity=80e9,
            memory_bandwidth=2039e9,
            compute_throughput=312e12,
            network_bandwidth=600e9
        )
        
        result = perf.calculate_decode_performance(
            system_constraints=gpu,
            batch_size=1,
            prefill_length=2048,
            output_length=100
        )
        
        # Check that breakdown components exist
        assert hasattr(result, 'total_compute_attention')
        assert hasattr(result, 'total_compute_ffn')
        assert hasattr(result, 'total_compute_other')
        
        # Check reasonable values
        assert result.total_compute_attention > 0
        assert result.total_compute_ffn > 0
        assert result.total_compute_other >= 0  # Other might be small
    
    def test_attention_grows_with_sequence(self):
        """Test that attention compute grows faster than FFN with sequence length"""
        model = LLAMA_3_8B
        perf = InferencePerformance(model)
        
        gpu = SystemConstraints(
            memory_capacity=80e9,
            memory_bandwidth=2039e9,
            compute_throughput=312e12,
            network_bandwidth=600e9
        )
        
        result_short = perf.calculate_achievable_ttft(
            system_constraints=gpu,
            batch_size=1,
            sequence_length=512
        )
        
        result_long = perf.calculate_achievable_ttft(
            system_constraints=gpu,
            batch_size=1,
            sequence_length=2048
        )
        
        # Attention should grow quadratically (O(L^2))
        # For 4x sequence, attention should be ~16x
        attention_ratio = result_long.compute_attention / result_short.compute_attention
        
        # FFN should grow linearly (O(L))
        # For 4x sequence, FFN should be ~4x
        ffn_ratio = result_long.compute_ffn / result_short.compute_ffn
        
        # Attention should grow much faster than FFN
        # Note: In prefill, attention includes QKV projections which are O(L),
        # not just the O(L^2) attention matmul. So growth is less than 16x.
        assert attention_ratio > ffn_ratio
        
        # Rough checks (allow tolerance for mixed O(L) and O(L^2) operations)
        assert 4 < attention_ratio < 8  # Should be between 4x and 8x
        assert 3 < ffn_ratio < 5  # Should be around 4x
    
    def test_breakdown_method_directly(self):
        """Test the breakdown methods directly"""
        model = LLAMA_3_8B
        perf = InferencePerformance(model)
        parallelism_config = ParallelismConfig()
        
        # Test prefill compute breakdown
        breakdown = perf.calculate_prefill_compute_breakdown(
            batch_size=1,
            sequence_length=2048,
            parallelism_config=parallelism_config
        )
        
        assert 'attention' in breakdown
        assert 'ffn' in breakdown
        assert 'other' in breakdown
        assert 'total' in breakdown
        
        # Total should equal sum of parts
        assert abs(breakdown['total'] - (breakdown['attention'] + breakdown['ffn'] + breakdown['other'])) < 1e6


