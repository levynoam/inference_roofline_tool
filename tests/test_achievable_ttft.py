"""
Tests for TestAchievableTTFT
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


class TestAchievableTTFT:
    """Test calculate_achievable_ttft() with various parameters"""
    
    def test_basic_achievable_ttft(self):
        """Test basic TTFT calculation"""
        model = LLAMA_3_8B
        perf = InferencePerformance(model)
        gpu = SystemConstraints.from_gpu_spec("A100-80GB")
        
        result = perf.calculate_achievable_ttft(
            system_constraints=gpu,
            batch_size=1,
            sequence_length=2048
        )
        
        assert isinstance(result, ResourceUtilization)
        assert result.achievable_ttft > 0
        assert result.bottleneck_resource in ["Compute", "Memory Bandwidth", "Network Bandwidth"]
        assert 0 <= result.memory_utilization <= 2.0  # Allow slight overflow
        assert 0 <= result.compute_utilization <= 1.0
    
    def test_all_models_achievable_ttft(self, all_models):
        """Test achievable TTFT with all models"""
        gpu = SystemConstraints.from_gpu_spec("A100-80GB")
        
        for model in all_models:
            perf = InferencePerformance(model)
            
            result = perf.calculate_achievable_ttft(
                system_constraints=gpu,
                batch_size=1,
                sequence_length=1024
            )
            
            # Verify basic properties
            assert result.achievable_ttft > 0, f"Model {model.model_name} has zero TTFT"
            assert result.memory_used > 0
            assert result.compute_used > 0
            
            # Utilizations should be in valid range (allow OOM)
            assert result.memory_utilization >= 0
            assert 0 <= result.compute_utilization <= 1.0
    
    def test_all_gpus_achievable_ttft(self, all_gpus):
        """Test achievable TTFT with all GPU types"""
        model = LLAMA_3_8B
        perf = InferencePerformance(model)
        
        results = []
        for gpu_name in all_gpus:
            gpu = SystemConstraints.from_gpu_spec(gpu_name)
            
            result = perf.calculate_achievable_ttft(
                system_constraints=gpu,
                batch_size=1,
                sequence_length=2048
            )
            
            results.append((gpu_name, result))
            
            # Verify basic properties
            assert result.achievable_ttft > 0
            assert result.memory_used > 0
        
        # H100 should be faster than A100
        a100_80_time = next(r.achievable_ttft for name, r in results if name == "A100-80GB")
        h100_time = next(r.achievable_ttft for name, r in results if name == "H100-80GB")
        assert h100_time < a100_80_time
    
    def test_batch_size_scaling_achievable_ttft(self, batch_sizes):
        """Test TTFT scales with batch size"""
        model = LLAMA_3_8B
        perf = InferencePerformance(model)
        gpu = SystemConstraints.from_gpu_spec("A100-80GB")
        
        results = []
        for batch_size in batch_sizes[:4]:
            result = perf.calculate_achievable_ttft(
                system_constraints=gpu,
                batch_size=batch_size,
                sequence_length=1024
            )
            results.append(result)
        
        # TTFT should increase with batch size
        for i in range(1, len(results)):
            assert results[i].achievable_ttft >= results[i-1].achievable_ttft
    
    def test_sequence_length_scaling_achievable_ttft(self, sequence_lengths):
        """Test TTFT scales with sequence length"""
        model = LLAMA_3_8B
        perf = InferencePerformance(model)
        gpu = SystemConstraints.from_gpu_spec("A100-80GB")
        
        results = []
        for seq_len in sequence_lengths[:4]:
            result = perf.calculate_achievable_ttft(
                system_constraints=gpu,
                batch_size=1,
                sequence_length=seq_len
            )
            results.append(result)
        
        # TTFT should increase with sequence length
        for i in range(1, len(results)):
            assert results[i].achievable_ttft > results[i-1].achievable_ttft
    
    def test_parallelism_achievable_ttft(self):
        """Test achievable TTFT with parallelism"""
        model = LLAMA_3_70B  # Large model needs parallelism
        perf = InferencePerformance(model)
        gpu = SystemConstraints.from_gpu_spec("A100-80GB")
        
        # TP=1 should OOM
        result_tp1 = perf.calculate_achievable_ttft(
            system_constraints=gpu,
            batch_size=1,
            sequence_length=2048
        )
        
        # TP=2 should fit
        parallel_tp2 = ParallelismConfig(
            parallelism_type=ParallelismType.TENSOR_PARALLEL,
            tensor_parallel_size=2
        )
        result_tp2 = perf.calculate_achievable_ttft(
            system_constraints=gpu,
            batch_size=1,
            sequence_length=2048,
            parallelism_config=parallel_tp2
        )
        
        # TP=2 should use less memory per GPU
        assert result_tp2.memory_used < result_tp1.memory_used
        assert result_tp1.memory_utilization > 1.0  # OOM
        assert result_tp2.memory_utilization < 1.0  # Fits
    
    def test_bottleneck_identification_achievable_ttft(self):
        """Test bottleneck is correctly identified"""
        model = LLAMA_3_8B
        perf = InferencePerformance(model)
        
        # A100 should be compute-bound for prefill
        gpu_a100 = SystemConstraints.from_gpu_spec("A100-80GB")
        result = perf.calculate_achievable_ttft(
            system_constraints=gpu_a100,
            batch_size=1,
            sequence_length=2048
        )
        
        assert result.bottleneck_resource == "Compute"
        assert result.compute_utilization == 1.0
    
    def test_oom_detection_achievable_ttft(self):
        """Test OOM is detected"""
        model = LLAMA_3_70B
        perf = InferencePerformance(model)
        gpu = SystemConstraints.from_gpu_spec("A100-40GB")  # Small memory
        
        result = perf.calculate_achievable_ttft(
            system_constraints=gpu,
            batch_size=1,
            sequence_length=2048
        )
        
        # Should detect OOM
        assert result.memory_utilization > 1.0
        assert result.memory_used > result.memory_available


# =============================================================================
# Test calculate_decode_performance() - Decode Phase
# =============================================================================

