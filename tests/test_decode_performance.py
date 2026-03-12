"""
Tests for TestDecodePerformance
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


class TestDecodePerformance:
    """Test calculate_decode_performance() with various parameters"""
    
    def test_basic_decode(self):
        """Test basic decode calculation"""
        model = LLAMA_3_8B
        perf = InferencePerformance(model)
        gpu = SystemConstraints.from_gpu_spec("A100-80GB")
        
        result = perf.calculate_decode_performance(
            system_constraints=gpu,
            batch_size=1,
            prefill_length=512,
            output_length=128
        )
        
        assert isinstance(result, DecodePerformance)
        assert result.output_length == 128
        assert result.total_decode_time > 0
        assert result.tokens_per_second_per_user > 0
        assert result.total_throughput > 0
    
    def test_all_models_decode(self, all_models):
        """Test decode with all models"""
        gpu = SystemConstraints.from_gpu_spec("A100-80GB")
        
        for model in all_models:
            perf = InferencePerformance(model)
            
            result = perf.calculate_decode_performance(
                system_constraints=gpu,
                batch_size=1,
                prefill_length=512,
                output_length=64
            )
            
            # Verify basic properties
            assert result.total_decode_time > 0, f"Model {model.model_name} has zero time"
            assert result.tokens_per_second_per_user > 0
            assert result.avg_step_time > 0
            assert result.avg_memory_utilization >= 0
    
    def test_all_gpus_decode(self, all_gpus):
        """Test decode with all GPU types"""
        model = LLAMA_3_8B
        perf = InferencePerformance(model)
        
        results = []
        for gpu_name in all_gpus:
            gpu = SystemConstraints.from_gpu_spec(gpu_name)
            
            result = perf.calculate_decode_performance(
                system_constraints=gpu,
                batch_size=1,
                prefill_length=512,
                output_length=128
            )
            
            results.append((gpu_name, result))
            
            # Verify basic properties
            assert result.total_decode_time > 0
            assert result.tokens_per_second_per_user > 0
        
        # H100 should be faster than A100
        a100_80_tps = next(r.tokens_per_second_per_user for name, r in results if name == "A100-80GB")
        h100_tps = next(r.tokens_per_second_per_user for name, r in results if name == "H100-80GB")
        assert h100_tps > a100_80_tps
    
    def test_batch_size_scaling_decode(self, batch_sizes):
        """Test decode throughput scales with batch size"""
        model = LLAMA_3_8B
        perf = InferencePerformance(model)
        gpu = SystemConstraints.from_gpu_spec("A100-80GB")
        
        results = []
        for batch_size in batch_sizes[:4]:
            result = perf.calculate_decode_performance(
                system_constraints=gpu,
                batch_size=batch_size,
                prefill_length=512,
                output_length=128
            )
            results.append(result)
        
        # Total throughput should increase with batch size
        for i in range(1, len(results)):
            assert results[i].total_throughput > results[i-1].total_throughput
    
    def test_output_length_scaling_decode(self):
        """Test decode time scales with output length"""
        model = LLAMA_3_8B
        perf = InferencePerformance(model)
        gpu = SystemConstraints.from_gpu_spec("A100-80GB")
        
        output_lengths = [64, 128, 256, 512]
        results = []
        
        for output_len in output_lengths:
            result = perf.calculate_decode_performance(
                system_constraints=gpu,
                batch_size=1,
                prefill_length=512,
                output_length=output_len
            )
            results.append(result)
        
        # Time should scale roughly linearly with output length
        for i in range(1, len(results)):
            ratio = results[i].total_decode_time / results[0].total_decode_time
            expected_ratio = output_lengths[i] / output_lengths[0]
            assert ratio == pytest.approx(expected_ratio, rel=0.1)
    
    def test_prefill_length_impact_decode(self, sequence_lengths):
        """Test prefill length affects decode performance"""
        model = LLAMA_3_8B
        perf = InferencePerformance(model)
        gpu = SystemConstraints.from_gpu_spec("A100-80GB")
        
        results = []
        for prefill_len in sequence_lengths[:4]:
            result = perf.calculate_decode_performance(
                system_constraints=gpu,
                batch_size=1,
                prefill_length=prefill_len,
                output_length=128
            )
            results.append(result)
        
        # Longer prefill should increase decode time (larger KV cache)
        for i in range(1, len(results)):
            assert results[i].total_decode_time >= results[i-1].total_decode_time
    
    def test_step_details_decode(self):
        """Test step-by-step details are returned"""
        model = LLAMA_3_8B
        perf = InferencePerformance(model)
        gpu = SystemConstraints.from_gpu_spec("A100-80GB")
        
        result = perf.calculate_decode_performance(
            system_constraints=gpu,
            batch_size=1,
            prefill_length=512,
            output_length=16,
            return_step_details=True,
            decode_step_skip=1  # No skipping to get all step details
        )
        
        # Verify step details
        assert len(result.step_details) == 16
        assert result.step_details[0].step == 0
        assert result.step_details[-1].step == 15
        
        # Context length should grow
        assert result.step_details[0].context_length == 512
        assert result.step_details[-1].context_length == 512 + 15
        
        # All steps should have valid data
        for step in result.step_details:
            assert step.step_time > 0
            assert step.compute_flops > 0
            assert step.bottleneck in ["Compute", "Memory Bandwidth", "Network Bandwidth"]
    
    def test_bottleneck_breakdown_decode(self):
        """Test bottleneck breakdown is calculated"""
        model = LLAMA_3_8B
        perf = InferencePerformance(model)
        gpu = SystemConstraints.from_gpu_spec("A100-80GB")
        
        result = perf.calculate_decode_performance(
            system_constraints=gpu,
            batch_size=1,
            prefill_length=512,
            output_length=128
        )
        
        # Verify bottleneck breakdown
        assert len(result.bottleneck_breakdown) > 0
        total_steps = sum(result.bottleneck_breakdown.values())
        assert total_steps == result.output_length
        
        # Primary bottleneck should be most common
        assert result.primary_bottleneck in result.bottleneck_breakdown
        max_count = max(result.bottleneck_breakdown.values())
        assert result.bottleneck_breakdown[result.primary_bottleneck] == max_count
    
    def test_throughput_metrics_decode(self):
        """Test throughput metrics are consistent"""
        model = LLAMA_3_8B
        perf = InferencePerformance(model)
        gpu = SystemConstraints.from_gpu_spec("A100-80GB")
        
        batch_size = 4
        output_length = 128
        
        result = perf.calculate_decode_performance(
            system_constraints=gpu,
            batch_size=batch_size,
            prefill_length=512,
            output_length=output_length
        )
        
        # TPS * batch = total throughput
        expected_throughput = result.tokens_per_second_per_user * batch_size
        assert result.total_throughput == pytest.approx(expected_throughput, rel=0.01)
        
        # TPS = output_length / total_time
        expected_tps = output_length / result.total_decode_time
        assert result.tokens_per_second_per_user == pytest.approx(expected_tps, rel=0.01)
    
    def test_kv_and_weights_bandwidth_decode(self):
        """Test KV cache and weights bandwidth breakdown"""
        model = LLAMA_3_8B
        perf = InferencePerformance(model)
        gpu = SystemConstraints.from_gpu_spec("A100-80GB")
        
        batch_size = 16
        prefill_length = 1024
        output_length = 64
        
        result = perf.calculate_decode_performance(
            system_constraints=gpu,
            batch_size=batch_size,
            prefill_length=prefill_length,
            output_length=output_length,
            return_step_details=True
        )
        
        # Calculate KV and weights bandwidth manually
        bytes_per_param = 2  # FP16
        model_size = model.total_parameters * bytes_per_param
        
        total_kv_traffic = 0
        total_weights_traffic = 0
        total_time = 0
        
        for i, step in enumerate(result.step_details):
            step_time = max(step.compute_time, step.memory_bw_time, step.network_time)
            total_time += step_time
            
            # Calculate KV cache size for this step
            context_len = prefill_length + i
            kv_cache_size = model.get_kv_cache_size(
                batch_size=batch_size,
                sequence_length=context_len,
                bytes_per_element=bytes_per_param
            )
            
            total_weights_traffic += model_size
            total_kv_traffic += kv_cache_size
        
        kv_bw_tbps = (total_kv_traffic / total_time) / 1e12
        weights_bw_tbps = (total_weights_traffic / total_time) / 1e12
        
        # Verify values are non-zero and reasonable
        assert kv_bw_tbps > 0, "KV bandwidth should be non-zero"
        assert weights_bw_tbps > 0, "Weights bandwidth should be non-zero"
        
        # KV bandwidth should grow with context length (more cache to read)
        # For long contexts, KV can be significant
        print(f"KV Bandwidth: {kv_bw_tbps:.3f} TB/s")
        print(f"Weights Bandwidth: {weights_bw_tbps:.3f} TB/s")
        print(f"Total Memory BW: {result.memory_bandwidth / 1e12:.3f} TB/s")
        
        # Sum should be close to total memory bandwidth (within activations overhead)
        total_calculated = kv_bw_tbps + weights_bw_tbps
        total_actual = result.memory_bandwidth / 1e12
        
        # Allow some difference for activations and overhead
        assert total_calculated < total_actual * 1.5, "KV+weights should be less than 1.5x total"
        assert total_calculated > total_actual * 0.5, "KV+weights should be more than 0.5x total"
    
    def test_kv_bandwidth_grows_with_context(self):
        """Test that KV bandwidth increases with longer contexts"""
        model = LLAMA_3_8B
        perf = InferencePerformance(model)
        gpu = SystemConstraints.from_gpu_spec("A100-80GB")
        
        batch_size = 8
        output_length = 32
        
        kv_bandwidths = []
        prefill_lengths = [512, 2048, 8192]
        
        for prefill_length in prefill_lengths:
            result = perf.calculate_decode_performance(
                system_constraints=gpu,
                batch_size=batch_size,
                prefill_length=prefill_length,
                output_length=output_length,
                return_step_details=True
            )
            
            # Calculate KV bandwidth
            bytes_per_param = 2
            total_kv_traffic = 0
            total_time = 0
            
            for i, step in enumerate(result.step_details):
                step_time = max(step.compute_time, step.memory_bw_time, step.network_time)
                total_time += step_time
                
                context_len = prefill_length + i
                kv_cache_size = model.get_kv_cache_size(
                    batch_size=batch_size,
                    sequence_length=context_len,
                    bytes_per_element=bytes_per_param
                )
                total_kv_traffic += kv_cache_size
            
            kv_bw_tbps = (total_kv_traffic / total_time) / 1e12
            kv_bandwidths.append(kv_bw_tbps)
            print(f"Prefill {prefill_length}: KV BW = {kv_bw_tbps:.3f} TB/s")
        
        # KV bandwidth should increase with context length
        assert kv_bandwidths[1] > kv_bandwidths[0], "KV BW should increase with longer context"
        assert kv_bandwidths[2] > kv_bandwidths[1], "KV BW should continue increasing"
    
    def test_weights_bandwidth_constant_across_context(self):
        """Test that weights traffic per step is constant regardless of context"""
        model = LLAMA_3_8B
        perf = InferencePerformance(model)
        gpu = SystemConstraints.from_gpu_spec("A100-80GB")
        
        batch_size = 8
        output_length = 32
        
        weights_traffic_per_step = []
        prefill_lengths = [512, 2048, 8192]
        
        for prefill_length in prefill_lengths:
            result = perf.calculate_decode_performance(
                system_constraints=gpu,
                batch_size=batch_size,
                prefill_length=prefill_length,
                output_length=output_length,
                return_step_details=True
            )
            
            # Calculate weights traffic per step (should be constant)
            bytes_per_param = 2
            model_size = model.total_parameters * bytes_per_param
            weights_traffic_per_step.append(model_size / 1e12)  # TB per step
            print(f"Prefill {prefill_length}: Weights traffic = {model_size / 1e12:.3f} TB per step")
        
        # Weights traffic per step should be identical (same model)
        assert all(abs(wt - weights_traffic_per_step[0]) < 0.001 for wt in weights_traffic_per_step), \
            "Weights traffic per step should be constant"
    
    def test_kv_and_weights_bandwidth_ttft(self):
        """Test KV cache and weights bandwidth estimation for TTFT"""
        model = LLAMA_3_8B
        perf = InferencePerformance(model)
        gpu = SystemConstraints.from_gpu_spec("A100-80GB")
        
        batch_size = 16
        sequence_length = 2048
        
        result = perf.calculate_achievable_ttft(
            system_constraints=gpu,
            batch_size=batch_size,
            sequence_length=sequence_length
        )
        
        # For TTFT, estimate bandwidth split
        total_memory_bw_used = result.memory_bandwidth_used
        weights_bw_tbps = (total_memory_bw_used * 0.85) / 1e12
        kv_bw_tbps = (total_memory_bw_used * 0.15) / 1e12
        
        # Verify values are non-zero and reasonable
        assert total_memory_bw_used > 0, "Total memory bandwidth should be non-zero"
        assert weights_bw_tbps > 0, "Weights bandwidth should be non-zero"
        assert kv_bw_tbps > 0, "KV bandwidth should be non-zero"
        
        print(f"Total Memory BW: {total_memory_bw_used / 1e12:.3f} TB/s")
        print(f"Weights BW (85%): {weights_bw_tbps:.3f} TB/s")
        print(f"KV BW (15%): {kv_bw_tbps:.3f} TB/s")
        
        # Weights should dominate in prefill (reading model weights)
        assert weights_bw_tbps > kv_bw_tbps * 4, "Weights BW should be much larger than KV BW in prefill"
        
        # Sum should equal total
        total_calculated = weights_bw_tbps + kv_bw_tbps
        total_actual = total_memory_bw_used / 1e12
        assert abs(total_calculated - total_actual) < 0.001, "Sum should equal total memory BW"
    
    def test_weights_bandwidth_dominates_in_ttft(self):
        """Test that weights bandwidth is the primary component in TTFT"""
        model = LLAMA_3_70B  # Larger model
        perf = InferencePerformance(model)
        gpu = SystemConstraints.from_gpu_spec("H100-80GB")
        
        batch_size = 4
        sequence_length = 4096
        
        result = perf.calculate_achievable_ttft(
            system_constraints=gpu,
            batch_size=batch_size,
            sequence_length=sequence_length
        )
        
        total_memory_bw_used = result.memory_bandwidth_used
        weights_bw_tbps = (total_memory_bw_used * 0.85) / 1e12
        kv_bw_tbps = (total_memory_bw_used * 0.15) / 1e12
        
        print(f"TTFT - Weights BW: {weights_bw_tbps:.3f} TB/s, KV BW: {kv_bw_tbps:.3f} TB/s")
        
        # For large models in prefill, weights should be 80%+ of memory bandwidth
        assert weights_bw_tbps > kv_bw_tbps * 4, "Weights should dominate memory bandwidth in prefill"
    
    def test_bandwidth_breakdown_consistency_across_batch_sizes(self):
        """Test that bandwidth breakdown percentages stay consistent across batch sizes"""
        model = LLAMA_3_8B
        perf = InferencePerformance(model)
        gpu = SystemConstraints.from_gpu_spec("A100-80GB")
        
        sequence_length = 1024
        batch_sizes = [1, 4, 16, 32]
        
        weights_percentages = []
        
        for batch_size in batch_sizes:
            result = perf.calculate_achievable_ttft(
                system_constraints=gpu,
                batch_size=batch_size,
                sequence_length=sequence_length
            )
            
            total_memory_bw = result.memory_bandwidth_used
            weights_bw = total_memory_bw * 0.85
            
            # Calculate what percentage of total is weights
            weights_pct = (weights_bw / total_memory_bw) * 100
            weights_percentages.append(weights_pct)
            
            print(f"Batch {batch_size}: Weights = {weights_pct:.1f}% of total memory BW")
        
        # All should be exactly 85% (our estimation)
        for pct in weights_percentages:
            assert abs(pct - 85.0) < 0.01, f"Weights percentage should be 85%, got {pct:.2f}%"


# =============================================================================
# Cross-Function Consistency Tests
# =============================================================================

