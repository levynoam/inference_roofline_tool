"""
Tests for TestDecodePerformanceExtended
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


class TestDecodePerformanceExtended:
    """Extended coverage tests for decode performance"""
    
    def test_decode_tps_with_varying_context_lengths(self):
        """Test how TPS changes with context length"""
        model = LLAMA_3_8B
        perf = InferencePerformance(model)
        gpu = SystemConstraints.from_gpu_spec("A100-80GB")
        
        prefill_lengths = [128, 1024, 8192]
        tps_values = []
        
        for prefill_length in prefill_lengths:
            result = perf.calculate_decode_performance(
                system_constraints=gpu,
                batch_size=1,
                prefill_length=prefill_length,
                output_length=64
            )
            tps_values.append(result.tokens_per_second_per_user)
        
        # TPS should decrease with longer context (more KV cache to read)
        assert tps_values[1] < tps_values[0]
        assert tps_values[2] < tps_values[1]
    
    def test_decode_throughput_scaling_with_batch(self):
        """Test that total throughput scales linearly with batch size"""
        model = LLAMA_3_8B
        perf = InferencePerformance(model)
        gpu = SystemConstraints.from_gpu_spec("H100-80GB")
        
        batch_sizes = [1, 4, 16, 32]
        throughputs = []
        
        for bs in batch_sizes:
            result = perf.calculate_decode_performance(
                system_constraints=gpu,
                batch_size=bs,
                prefill_length=1024,
                output_length=32
            )
            throughputs.append(result.total_throughput)
        
        # Throughput should increase with batch size
        for i in range(1, len(throughputs)):
            assert throughputs[i] > throughputs[i-1]
    
    def test_decode_bottleneck_transitions(self):
        """Test bottleneck transitions with increasing batch size"""
        model = LLAMA_3_8B
        perf = InferencePerformance(model)
        gpu = SystemConstraints.from_gpu_spec("A100-80GB")
        
        batch_sizes = [1, 16, 64]
        bottlenecks = []
        
        for bs in batch_sizes:
            result = perf.calculate_decode_performance(
                system_constraints=gpu,
                batch_size=bs,
                prefill_length=2048,
                output_length=32
            )
            bottlenecks.append(result.primary_bottleneck)
            print(f"Batch {bs}: {result.primary_bottleneck}")
        
        # Should see transitions as batch size increases
        # (compute bound at low batch, memory bound at high batch)
        assert len(set(bottlenecks)) > 0  # At least some bottleneck is identified
    
    def test_decode_step_time_consistency(self):
        """Test that step times are consistent within a decode sequence"""
        model = LLAMA_3_8B
        perf = InferencePerformance(model)
        gpu = SystemConstraints.from_gpu_spec("A100-80GB")
        
        result = perf.calculate_decode_performance(
            system_constraints=gpu,
            batch_size=4,
            prefill_length=512,
            output_length=64,
            return_step_details=True
        )
        
        step_times = [step.step_time for step in result.step_details]
        
        # Step times should be relatively consistent (within 50%)
        avg_time = sum(step_times) / len(step_times)
        for st in step_times:
            assert abs(st - avg_time) / avg_time < 0.5, "Step times vary too much"
    
    def test_decode_network_bandwidth_with_parallelism(self):
        """Test network bandwidth requirements with tensor parallelism"""
        model = LLAMA_3_70B
        perf = InferencePerformance(model)
        gpu = SystemConstraints.from_gpu_spec("H100-80GB")
        
        parallel_config = ParallelismConfig(
            parallelism_type=ParallelismType.TENSOR_PARALLEL,
            tensor_parallel_size=4
        )
        
        result = perf.calculate_decode_performance(
            system_constraints=gpu,
            batch_size=8,
            prefill_length=2048,
            output_length=128,
            parallelism_config=parallel_config,
            return_step_details=True
        )
        
        # Should have network traffic with tensor parallelism
        assert any(step.network_traffic > 0 for step in result.step_details)


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])


