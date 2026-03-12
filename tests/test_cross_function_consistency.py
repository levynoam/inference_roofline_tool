"""
Tests for TestCrossFunctionConsistency
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


class TestCrossFunctionConsistency:
    """Test consistency between different functions"""
    
    def test_forward_backward_consistency(self):
        """Test forward and backward directions are consistent"""
        model = LLAMA_3_8B
        perf = InferencePerformance(model)
        gpu = SystemConstraints.from_gpu_spec("A100-80GB")
        
        batch_size = 1
        sequence_length = 2048
        
        # Backward: Calculate achievable TTFT
        backward = perf.calculate_achievable_ttft(
            system_constraints=gpu,
            batch_size=batch_size,
            sequence_length=sequence_length
        )
        
        # Forward: Use that TTFT to calculate resources
        forward = perf.calculate_prefill_resources(
            batch_size=batch_size,
            sequence_length=sequence_length,
            time_to_first_token=backward.achievable_ttft,
            num_gpus=1,
            parallelism_config=ParallelismConfig()
        )
        
        # Resources should match GPU capabilities (within tolerance)
        # Compute should be close to GPU compute
        compute_ratio = forward.compute_flops_per_sec / gpu.compute_throughput
        assert 0.9 <= compute_ratio <= 1.1
        
        # Memory should fit
        assert forward.memory_per_gpu <= gpu.memory_capacity
    
    def test_memory_consistency_prefill_decode(self):
        """Test memory usage is consistent between prefill and decode"""
        model = LLAMA_3_8B
        perf = InferencePerformance(model)
        gpu = SystemConstraints.from_gpu_spec("A100-80GB")
        
        batch_size = 1
        sequence_length = 2048
        
        # Prefill
        prefill = perf.calculate_prefill_resources(
            batch_size=batch_size,
            sequence_length=sequence_length,
            time_to_first_token=0.1,
            num_gpus=1,
            parallelism_config=ParallelismConfig()
        )
        
        # Decode
        decode = perf.calculate_decode_performance(
            system_constraints=gpu,
            batch_size=batch_size,
            prefill_length=sequence_length,
            output_length=1  # Just first token
        )
        
        # Memory should be similar (decode has slightly larger KV cache)
        assert decode.avg_memory_utilization * gpu.memory_capacity >= prefill.memory_per_gpu * 0.9
    
    def test_compute_bound_vs_memory_bound(self):
        """Test that prefill is compute-bound and decode is memory-bound"""
        model = LLAMA_3_8B
        perf = InferencePerformance(model)
        gpu = SystemConstraints.from_gpu_spec("A100-80GB")
        
        # Prefill
        prefill = perf.calculate_achievable_ttft(
            system_constraints=gpu,
            batch_size=1,
            sequence_length=2048
        )
        
        # Decode
        decode = perf.calculate_decode_performance(
            system_constraints=gpu,
            batch_size=1,
            prefill_length=2048,
            output_length=128
        )
        
        # Prefill should be compute-bound
        assert prefill.bottleneck_resource == "Compute"
        assert prefill.compute_utilization > 0.9
        
        # Decode should be memory bandwidth-bound
        assert decode.primary_bottleneck == "Memory Bandwidth"
        assert decode.avg_memory_bw_utilization > 0.5


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

