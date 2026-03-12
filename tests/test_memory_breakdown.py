"""
Tests for TestMemoryBreakdown
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


class TestMemoryBreakdown:
    """Test memory breakdown into weights, KV cache, and activations"""
    
    def test_ttft_memory_breakdown(self):
        """Test that TTFT memory breakdown is correctly calculated"""
        model = LLAMA_3_8B
        perf = InferencePerformance(model)
        
        gpu = SystemConstraints(
            memory_capacity=80e9,  # 80 GB
            memory_bandwidth=2039e9,  # 2 TB/s
            compute_throughput=312e12,  # 312 TFLOP/s
            network_bandwidth=600e9  # 600 GB/s
        )
        
        batch_size = 1
        sequence_length = 2048
        
        result = perf.calculate_achievable_ttft(
            system_constraints=gpu,
            batch_size=batch_size,
            sequence_length=sequence_length
        )
        
        # Check that breakdown components exist
        assert hasattr(result, 'memory_weights')
        assert hasattr(result, 'memory_kv_cache')
        assert hasattr(result, 'memory_activations')
        
        # Check that components sum to total (approximately)
        total_breakdown = result.memory_weights + result.memory_kv_cache + result.memory_activations
        assert abs(total_breakdown - result.memory_used) < 1e6  # Within 1 MB
        
        # Check reasonable values
        assert result.memory_weights > 0
        assert result.memory_kv_cache > 0
        assert result.memory_activations > 0
        
        # Weights should be largest for small batch/sequence
        assert result.memory_weights > result.memory_kv_cache
    
    def test_decode_memory_breakdown(self):
        """Test that decode memory breakdown is correctly calculated"""
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
        assert hasattr(result, 'avg_memory_weights')
        assert hasattr(result, 'avg_memory_kv_cache')
        assert hasattr(result, 'avg_memory_activations')
        
        # Check reasonable values
        assert result.avg_memory_weights > 0
        assert result.avg_memory_kv_cache > 0
        assert result.avg_memory_activations > 0
    
    def test_kv_cache_grows_with_sequence(self):
        """Test that KV cache grows with sequence length"""
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
        
        # KV cache should be 4x larger for 4x sequence length
        ratio = result_long.memory_kv_cache / result_short.memory_kv_cache
        assert 3.5 < ratio < 4.5  # Allow some tolerance
        
        # Weights should be the same
        assert abs(result_long.memory_weights - result_short.memory_weights) < 1e6


