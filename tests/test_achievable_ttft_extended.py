"""
Tests for TestAchievableTTFTExtended
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


class TestAchievableTTFTExtended:
    """Extended coverage tests for achievable TTFT"""
    
    def test_ttft_memory_constrained_scenario(self):
        """Test TTFT when severely memory constrained"""
        model = LLAMA_3_70B
        perf = InferencePerformance(model)
        
        # Very limited memory - should cause OOM
        small_gpu = SystemConstraints(
            memory_capacity=10e9,  # Only 10GB
            memory_bandwidth=1e12,
            compute_throughput=300e12,
            network_bandwidth=100e9
        )
        
        result = perf.calculate_achievable_ttft(
            system_constraints=small_gpu,
            batch_size=1,
            sequence_length=1024
        )
        
        # Should detect OOM
        assert result.memory_utilization > 1.0  # Over 100% utilization indicates OOM
    
    def test_ttft_compute_vs_bandwidth_bottleneck(self):
        """Test TTFT under different bottleneck conditions"""
        model = LLAMA_3_8B
        perf = InferencePerformance(model)
        
        # Compute-bound scenario
        compute_bound_gpu = SystemConstraints(
            memory_capacity=80e9,
            memory_bandwidth=2000e9,  # High bandwidth
            compute_throughput=100e12,  # Low compute
            network_bandwidth=400e9
        )
        
        # Bandwidth-bound scenario
        bandwidth_bound_gpu = SystemConstraints(
            memory_capacity=80e9,
            memory_bandwidth=200e9,  # Low bandwidth
            compute_throughput=4500e12,  # High compute
            network_bandwidth=400e9
        )
        
        result_compute = perf.calculate_achievable_ttft(
            system_constraints=compute_bound_gpu,
            batch_size=1,
            sequence_length=2048
        )
        
        result_bandwidth = perf.calculate_achievable_ttft(
            system_constraints=bandwidth_bound_gpu,
            batch_size=1,
            sequence_length=2048
        )
        
        # Compute bound should have high compute utilization
        assert result_compute.compute_utilization > 0.8
        
        # Bandwidth bound should have high memory bandwidth utilization  
        assert result_bandwidth.memory_bandwidth_utilization > 0.8
    
    def test_ttft_scales_with_tensor_parallelism(self):
        """Test that TTFT improves with tensor parallelism"""
        model = LLAMA_3_70B
        perf = InferencePerformance(model)
        gpu = SystemConstraints.from_gpu_spec("H100-80GB")
        
        # No parallelism
        result_1gpu = perf.calculate_achievable_ttft(
            system_constraints=gpu,
            batch_size=1,
            sequence_length=4096,
            parallelism_config=ParallelismConfig()
        )
        
        # With tensor parallelism
        result_4gpu = perf.calculate_achievable_ttft(
            system_constraints=gpu,
            batch_size=1,
            sequence_length=4096,
            parallelism_config=ParallelismConfig(
                parallelism_type=ParallelismType.TENSOR_PARALLEL,
                tensor_parallel_size=4
            )
        )
        
        # TTFT should improve (be lower) with more GPUs
        assert result_4gpu.achievable_ttft < result_1gpu.achievable_ttft
    
    def test_ttft_utilization_percentages(self):
        """Test that resource utilization percentages are reasonable"""
        model = MISTRAL_7B
        perf = InferencePerformance(model)
        gpu = SystemConstraints.from_gpu_spec("A100-80GB")
        
        result = perf.calculate_achievable_ttft(
            system_constraints=gpu,
            batch_size=4,
            sequence_length=2048
        )
        
        # Utilizations should be between 0 and 1
        assert 0 <= result.compute_utilization <= 1
        assert 0 <= result.memory_bandwidth_utilization <= 1
        assert 0 <= result.network_bandwidth_utilization <= 1
        
        # At least one should be close to 1 (the bottleneck)
        max_util = max(result.compute_utilization, result.memory_bandwidth_utilization, result.network_bandwidth_utilization)
        assert max_util > 0.5, "At least one resource should be significantly utilized"
    
    def test_ttft_with_kernel_latency_variations(self):
        """Test TTFT with different kernel launch latencies"""
        model = LLAMA_3_8B
        perf = InferencePerformance(model)
        gpu = SystemConstraints.from_gpu_spec("A100-80GB")
        
        latencies = [1e-6, 5e-6, 20e-6]  # 1µs, 5µs, 20µs
        results = []
        
        for latency in latencies:
            result = perf.calculate_achievable_ttft(
                system_constraints=gpu,
                batch_size=1,
                sequence_length=1024,
                kernel_launch_latency=latency
            )
            results.append(result)
        
        # Higher kernel latency should increase TTFT
        assert results[1].achievable_ttft > results[0].achievable_ttft
        assert results[2].achievable_ttft > results[1].achievable_ttft


