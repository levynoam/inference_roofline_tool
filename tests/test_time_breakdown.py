"""
Tests for TestTimeBreakdown
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


class TestTimeBreakdown:
    """Test time breakdown into compute busy, kernel launch, and idle"""
    
    def test_ttft_time_breakdown_basic(self):
        """Test that TTFT time breakdown components sum to total time"""
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
        
        # Check that time breakdown fields exist
        assert hasattr(result, 'time_compute_busy')
        assert hasattr(result, 'time_kernel_launch')
        assert hasattr(result, 'time_idle')
        
        # Check that times are non-negative
        assert result.time_compute_busy >= 0
        assert result.time_kernel_launch >= 0
        assert result.time_idle >= 0
        
        # Check that times sum to total TTFT
        total_breakdown = result.time_compute_busy + result.time_kernel_launch + result.time_idle
        assert abs(total_breakdown - result.achievable_ttft) < 1e-9
    
    def test_decode_time_breakdown_basic(self):
        """Test that decode time breakdown components sum to total time"""
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
        
        # Check that time breakdown fields exist
        assert hasattr(result, 'total_time_compute_busy')
        assert hasattr(result, 'total_time_kernel_launch')
        assert hasattr(result, 'total_time_idle')
        
        # Check that times are non-negative
        assert result.total_time_compute_busy >= 0
        assert result.total_time_kernel_launch >= 0
        assert result.total_time_idle >= 0
        
        # Check that times sum to total decode time
        total_breakdown = (result.total_time_compute_busy + 
                          result.total_time_kernel_launch + 
                          result.total_time_idle)
        assert abs(total_breakdown - result.total_decode_time) < 1e-6
    
    def test_idle_time_vs_bottleneck(self):
        """Test that idle time is non-zero when bottleneck is not compute"""
        model = LLAMA_3_8B
        perf = InferencePerformance(model)
        
        # Create a compute-bound scenario (high compute, low memory BW)
        gpu_compute_bound = SystemConstraints(
            memory_capacity=80e9,
            memory_bandwidth=20000e9,  # Very high - won't be bottleneck
            compute_throughput=10e12,   # Low - will be bottleneck
            network_bandwidth=600e9
        )
        
        result_compute_bound = perf.calculate_achievable_ttft(
            system_constraints=gpu_compute_bound,
            batch_size=1,
            sequence_length=2048
        )
        
        # When compute-bound, idle should be close to 0
        assert result_compute_bound.time_idle < 0.001, \
            f"Compute-bound should have minimal idle time, got {result_compute_bound.time_idle}"
        assert result_compute_bound.bottleneck_resource == "Compute"
        
        # Create a memory-bandwidth-bound scenario (low memory BW, high compute)
        gpu_memory_bound = SystemConstraints(
            memory_capacity=80e9,
            memory_bandwidth=200e9,     # Low - will be bottleneck
            compute_throughput=5000e12, # Very high - won't be bottleneck
            network_bandwidth=600e9
        )
        
        result_memory_bound = perf.calculate_achievable_ttft(
            system_constraints=gpu_memory_bound,
            batch_size=1,
            sequence_length=2048
        )
        
        # When memory-bound, idle should be significant (compute finishes before memory)
        assert result_memory_bound.time_idle > 0.01, \
            f"Memory-bound should have significant idle time, got {result_memory_bound.time_idle}"
        assert result_memory_bound.bottleneck_resource == "Memory Bandwidth"
    
    def test_time_values_in_reasonable_range(self):
        """Test that time values are in reasonable ranges (seconds)"""
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
        
        # Time should be in reasonable range (milliseconds to seconds)
        assert 0.001 < result.achievable_ttft < 10.0
        
        # Kernel launch should be microseconds to milliseconds
        assert 1e-6 < result.time_kernel_launch < 0.1
        
        # Compute busy should be most of the time
        assert result.time_compute_busy > result.time_kernel_launch


# =============================================================================
# Test MLA (Multi-head Latent Attention)
# =============================================================================

