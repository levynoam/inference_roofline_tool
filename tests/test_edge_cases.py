"""
Tests for TestEdgeCases
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


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_very_small_batch_size(self):
        """Test with batch size = 1"""
        model = LLAMA_3_8B
        perf = InferencePerformance(model)
        gpu = SystemConstraints.from_gpu_spec("A100-80GB")
        
        result = perf.calculate_achievable_ttft(
            system_constraints=gpu,
            batch_size=1,
            sequence_length=128
        )
        
        assert result.achievable_ttft > 0
    
    def test_very_large_batch_size(self):
        """Test with large batch size"""
        model = LLAMA_3_8B
        perf = InferencePerformance(model)
        gpu = SystemConstraints.from_gpu_spec("A100-80GB")
        
        result = perf.calculate_achievable_ttft(
            system_constraints=gpu,
            batch_size=128,
            sequence_length=512
        )
        
        assert result.achievable_ttft > 0
    
    def test_very_short_sequence(self):
        """Test with short sequence"""
        model = LLAMA_3_8B
        perf = InferencePerformance(model)
        gpu = SystemConstraints.from_gpu_spec("A100-80GB")
        
        result = perf.calculate_achievable_ttft(
            system_constraints=gpu,
            batch_size=1,
            sequence_length=1
        )
        
        assert result.achievable_ttft > 0
    
    def test_very_long_sequence(self):
        """Test with long sequence"""
        model = LLAMA_3_8B
        perf = InferencePerformance(model)
        gpu = SystemConstraints.from_gpu_spec("A100-80GB")
        
        result = perf.calculate_achievable_ttft(
            system_constraints=gpu,
            batch_size=1,
            sequence_length=16384
        )
        
        assert result.achievable_ttft > 0
    
    def test_mla_model(self):
        """Test with MLA-enabled model"""
        model = DEEPSEEK_3_2
        perf = InferencePerformance(model)
        gpu = SystemConstraints.from_gpu_spec("A100-80GB")
        
        # Should work with MLA
        result = perf.calculate_achievable_ttft(
            system_constraints=gpu,
            batch_size=1,
            sequence_length=2048
        )
        
        assert result.achievable_ttft > 0
        
        # Decode should also work
        decode = perf.calculate_decode_performance(
            system_constraints=gpu,
            batch_size=1,
            prefill_length=2048,
            output_length=128
        )
        
        assert decode.total_decode_time > 0
    
    def test_moe_model(self):
        """Test with MoE model"""
        model = MIXTRAL_8X7B
        perf = InferencePerformance(model)
        gpu = SystemConstraints.from_gpu_spec("A100-80GB")
        
        # Should work with MoE
        result = perf.calculate_achievable_ttft(
            system_constraints=gpu,
            batch_size=1,
            sequence_length=2048
        )
        
        assert result.achievable_ttft > 0
        
        # Decode should also work
        decode = perf.calculate_decode_performance(
            system_constraints=gpu,
            batch_size=1,
            prefill_length=2048,
            output_length=128
        )
        
        assert decode.total_decode_time > 0


# =============================================================================
# Breakdown Tests
# =============================================================================

