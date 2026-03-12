"""
Tests to verify that GPU scaling works correctly for all parallelism types.
All parallelism types (Pipeline, Tensor, Data) should scale performance by 
roughly the number of GPUs in steady-state throughput.
"""
import pytest
from llm_configs import LLAMA_3_70B
from inference_performance import (
    InferencePerformance,
    SystemConstraints, 
    ParallelismConfig,
    ParallelismType
)


@pytest.fixture
def base_system():
    """Standard system configuration."""
    return SystemConstraints.from_gpu_spec("H100-80GB")


@pytest.fixture
def llama3_70b():
    """Get Llama 3 70B configuration."""
    return InferencePerformance(LLAMA_3_70B)


class TestTensorParallelScaling:
    """Test that Tensor Parallelism scales performance by number of GPUs."""
    
    def test_tp_scales_decode_time(self, base_system, llama3_70b):
        """Decode time should be ~1/N with TP=N (compute-bound scenario)."""
        batch_size = 32
        kv_cache_tokens = 2048
        
        # TP=1
        config_tp1 = ParallelismConfig(
            parallelism_type=ParallelismType.TENSOR_PARALLEL,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1
        )
        result_tp1 = llama3_70b.calculate_decode_performance(
            system_constraints=base_system,
            batch_size=batch_size,
            prefill_length=kv_cache_tokens,
            output_length=1,  # Single decode step
            parallelism_config=config_tp1
        )
        
        # TP=4
        config_tp4 = ParallelismConfig(
            parallelism_type=ParallelismType.TENSOR_PARALLEL,
            tensor_parallel_size=4,
            pipeline_parallel_size=1,
            data_parallel_size=1
        )
        result_tp4 = llama3_70b.calculate_decode_performance(
            system_constraints=base_system,
            batch_size=batch_size,
            prefill_length=kv_cache_tokens,
            output_length=1,
            parallelism_config=config_tp4
        )
        
        # Should be roughly 4x faster
        speedup = result_tp1.avg_step_time / result_tp4.avg_step_time
        assert 3.0 < speedup < 5.0, f"TP=4 speedup should be ~4x, got {speedup:.2f}x"
    
    def test_tp_scales_prefill_time(self, base_system, llama3_70b):
        """Prefill time should be ~1/N with TP=N."""
        batch_size = 8
        sequence_length = 1024
        
        # TP=1
        config_tp1 = ParallelismConfig(
            parallelism_type=ParallelismType.TENSOR_PARALLEL,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1
        )
        result_tp1 = llama3_70b.calculate_achievable_ttft(
            system_constraints=base_system,
            batch_size=batch_size,
            sequence_length=sequence_length,
            parallelism_config=config_tp1
        )
        
        # TP=8
        config_tp8 = ParallelismConfig(
            parallelism_type=ParallelismType.TENSOR_PARALLEL,
            tensor_parallel_size=8,
            pipeline_parallel_size=1,
            data_parallel_size=1
        )
        result_tp8 = llama3_70b.calculate_achievable_ttft(
            system_constraints=base_system,
            batch_size=batch_size,
            sequence_length=sequence_length,
            parallelism_config=config_tp8
        )
        
        # Should be roughly 8x faster
        speedup = result_tp1.achievable_ttft / result_tp8.achievable_ttft
        assert 6.0 < speedup < 10.0, f"TP=8 speedup should be ~8x, got {speedup:.2f}x"


class TestPipelineParallelScaling:
    """Test that Pipeline Parallelism scales performance by number of GPUs."""
    
    def test_pp_scales_decode_time(self, base_system, llama3_70b):
        """Decode time should be ~1/N with PP=N in steady state."""
        batch_size = 32
        kv_cache_tokens = 2048
        
        # PP=1
        config_pp1 = ParallelismConfig(
            parallelism_type=ParallelismType.PIPELINE_PARALLEL,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1
        )
        result_pp1 = llama3_70b.calculate_decode_performance(
            system_constraints=base_system,
            batch_size=batch_size,
            prefill_length=kv_cache_tokens,
            output_length=1,
            parallelism_config=config_pp1
        )
        
        # PP=4
        config_pp4 = ParallelismConfig(
            parallelism_type=ParallelismType.PIPELINE_PARALLEL,
            tensor_parallel_size=1,
            pipeline_parallel_size=4,
            data_parallel_size=1
        )
        result_pp4 = llama3_70b.calculate_decode_performance(
            system_constraints=base_system,
            batch_size=batch_size,
            prefill_length=kv_cache_tokens,
            output_length=1,
            parallelism_config=config_pp4
        )
        
        # Should be roughly 4x faster
        speedup = result_pp1.avg_step_time / result_pp4.avg_step_time
        assert 3.0 < speedup < 5.0, f"PP=4 speedup should be ~4x, got {speedup:.2f}x"
    
    def test_pp_scales_prefill_time(self, base_system, llama3_70b):
        """Prefill time should be ~1/N with PP=N in steady state."""
        batch_size = 8
        sequence_length = 1024
        
        # PP=1
        config_pp1 = ParallelismConfig(
            parallelism_type=ParallelismType.PIPELINE_PARALLEL,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1
        )
        result_pp1 = llama3_70b.calculate_achievable_ttft(
            system_constraints=base_system,
            batch_size=batch_size,
            sequence_length=sequence_length,
            parallelism_config=config_pp1
        )
        
        # PP=2
        config_pp2 = ParallelismConfig(
            parallelism_type=ParallelismType.PIPELINE_PARALLEL,
            tensor_parallel_size=1,
            pipeline_parallel_size=2,
            data_parallel_size=1
        )
        result_pp2 = llama3_70b.calculate_achievable_ttft(
            system_constraints=base_system,
            batch_size=batch_size,
            sequence_length=sequence_length,
            parallelism_config=config_pp2
        )
        
        # Should be roughly 2x faster
        speedup = result_pp1.achievable_ttft / result_pp2.achievable_ttft
        assert 1.5 < speedup < 2.5, f"PP=2 speedup should be ~2x, got {speedup:.2f}x"
    
    def test_pp_scales_ttft(self, base_system, llama3_70b):
        """TTFT should scale with PP in steady state."""
        # Memory-constrained scenario
        constrained_system = SystemConstraints(
            memory_capacity=80 * (1024**3),
            memory_bandwidth=1000 * (1024**3),  # Constrained
            compute_throughput=312e12,
            network_bandwidth=450 * (1024**3)
        )
        
        batch_size = 4
        sequence_length = 512
        
        # PP=1
        config_pp1 = ParallelismConfig(
            parallelism_type=ParallelismType.PIPELINE_PARALLEL,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1
        )
        result_pp1 = llama3_70b.calculate_achievable_ttft(
            system_constraints=constrained_system,
            batch_size=batch_size,
            sequence_length=sequence_length,
            parallelism_config=config_pp1
        )
        
        # PP=4
        config_pp4 = ParallelismConfig(
            parallelism_type=ParallelismType.PIPELINE_PARALLEL,
            tensor_parallel_size=1,
            pipeline_parallel_size=4,
            data_parallel_size=1
        )
        result_pp4 = llama3_70b.calculate_achievable_ttft(
            system_constraints=constrained_system,
            batch_size=batch_size,
            sequence_length=sequence_length,
            parallelism_config=config_pp4
        )
        
        # Should be roughly 4x faster
        speedup = result_pp1.achievable_ttft / result_pp4.achievable_ttft
        assert 3.0 < speedup < 5.0, f"PP=4 speedup should be ~4x, got {speedup:.2f}x"


class TestDataParallelScaling:
    """Test that Data Parallelism scales performance by number of GPUs."""
    
    def test_dp_scales_decode_time(self, base_system, llama3_70b):
        """Decode time should scale with DP when not dominated by weight loading."""
        # Use very large batch so KV cache dominates over weights
        batch_size = 256  # Larger batch for more KV cache
        kv_cache_tokens = 8192  # Longer context for larger KV cache
        
        # DP=1
        config_dp1 = ParallelismConfig(
            parallelism_type=ParallelismType.DATA_PARALLEL,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1
        )
        result_dp1 = llama3_70b.calculate_decode_performance(
            system_constraints=base_system,
            batch_size=batch_size,
            prefill_length=kv_cache_tokens,
            output_length=1,
            parallelism_config=config_dp1
        )
        
        # DP=2 - each replica processes half the batch
        config_dp2 = ParallelismConfig(
            parallelism_type=ParallelismType.DATA_PARALLEL,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=2
        )
        result_dp2 = llama3_70b.calculate_decode_performance(
            system_constraints=base_system,
            batch_size=batch_size,
            prefill_length=kv_cache_tokens,
            output_length=1,
            parallelism_config=config_dp2
        )
        
        # Should show speedup (may be less than 2x if weight loading dominates)
        speedup = result_dp1.avg_step_time / result_dp2.avg_step_time
        assert speedup > 1.3, f"DP=2 speedup should be >1.3x, got {speedup:.2f}x"
    
    def test_dp_scales_prefill_time(self, base_system, llama3_70b):
        """Prefill time should be ~1/N with DP=N (per replica)."""
        batch_size = 32
        sequence_length = 1024
        
        # DP=1
        config_dp1 = ParallelismConfig(
            parallelism_type=ParallelismType.DATA_PARALLEL,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1
        )
        result_dp1 = llama3_70b.calculate_achievable_ttft(
            system_constraints=base_system,
            batch_size=batch_size,
            sequence_length=sequence_length,
            parallelism_config=config_dp1
        )
        
        # DP=4 - each replica processes batch/4
        config_dp4 = ParallelismConfig(
            parallelism_type=ParallelismType.DATA_PARALLEL,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=4
        )
        result_dp4 = llama3_70b.calculate_achievable_ttft(
            system_constraints=base_system,
            batch_size=batch_size,
            sequence_length=sequence_length,
            parallelism_config=config_dp4
        )
        
        # Should be roughly 4x faster
        speedup = result_dp1.achievable_ttft / result_dp4.achievable_ttft
        assert 3.0 < speedup < 5.0, f"DP=4 speedup should be ~4x, got {speedup:.2f}x"


class TestCombinedParallelism:
    """Test that combinations of parallelism types scale multiplicatively."""
    
    def test_tp_plus_pp_scales_multiplicatively(self, base_system, llama3_70b):
        """TP=2 + PP=2 should give ~4x speedup."""
        batch_size = 32
        kv_cache_tokens = 2048
        
        # Baseline
        config_base = ParallelismConfig(
            parallelism_type=ParallelismType.NONE,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1
        )
        result_base = llama3_70b.calculate_decode_performance(
            system_constraints=base_system,
            batch_size=batch_size,
            prefill_length=kv_cache_tokens,
            output_length=1,
            parallelism_config=config_base
        )
        
        # TP=2, PP=2
        config_combined = ParallelismConfig(
            parallelism_type=ParallelismType.TENSOR_PIPELINE,
            tensor_parallel_size=2,
            pipeline_parallel_size=2,
            data_parallel_size=1
        )
        result_combined = llama3_70b.calculate_decode_performance(
            system_constraints=base_system,
            batch_size=batch_size,
            prefill_length=kv_cache_tokens,
            output_length=1,
            parallelism_config=config_combined
        )
        
        # Should be roughly 4x faster
        speedup = result_base.avg_step_time / result_combined.avg_step_time
        assert 3.0 < speedup < 5.0, f"TP=2+PP=2 speedup should be ~4x, got {speedup:.2f}x"
    
    def test_all_parallelism_scales_multiplicatively(self, base_system, llama3_70b):
        """TP + PP + DP should show combined scaling benefit."""
        batch_size = 64
        kv_cache_tokens = 2048
        
        # Baseline
        config_base = ParallelismConfig(
            parallelism_type=ParallelismType.NONE,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1
        )
        result_base = llama3_70b.calculate_decode_performance(
            system_constraints=base_system,
            batch_size=batch_size,
            prefill_length=kv_cache_tokens,
            output_length=1,
            parallelism_config=config_base
        )
        
        # TP=2, PP=2, DP=2 (8 GPUs total)
        config_combined = ParallelismConfig(
            parallelism_type=ParallelismType.FULL_3D,
            tensor_parallel_size=2,
            pipeline_parallel_size=2,
            data_parallel_size=2
        )
        result_combined = llama3_70b.calculate_decode_performance(
            system_constraints=base_system,
            batch_size=batch_size,
            prefill_length=kv_cache_tokens,
            output_length=1,
            parallelism_config=config_combined
        )
        
        # TP and PP multiply, DP provides additional benefit (may be less than 2x for decode)
        # Expect at least 3x from TP*PP=4 with some DP benefit
        speedup = result_base.avg_step_time / result_combined.avg_step_time
        assert speedup > 3.0, f"TP=2+PP=2+DP=2 speedup should be >3x, got {speedup:.2f}x"


class TestMemoryBandwidthScaling:
    """Test that memory-bandwidth-bound scenarios scale correctly."""
    
    def test_pp_scales_in_memory_bound_scenario(self, llama3_70b):
        """PP should scale even when memory-bound."""
        # Create memory-constrained system
        mem_constrained = SystemConstraints(
            memory_capacity=80 * (1024**3),
            memory_bandwidth=1000 * (1024**3),  # Low bandwidth
            compute_throughput=312e12,
            network_bandwidth=450 * (1024**3)
        )
        
        batch_size = 1  # Small batch to be memory-bound
        kv_cache_tokens = 4096
        
        # PP=1
        config_pp1 = ParallelismConfig(
            parallelism_type=ParallelismType.PIPELINE_PARALLEL,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1
        )
        result_pp1 = llama3_70b.calculate_decode_performance(
            system_constraints=mem_constrained,
            batch_size=batch_size,
            prefill_length=kv_cache_tokens,
            output_length=1,
            parallelism_config=config_pp1
        )
        
        # PP=2
        config_pp2 = ParallelismConfig(
            parallelism_type=ParallelismType.PIPELINE_PARALLEL,
            tensor_parallel_size=1,
            pipeline_parallel_size=2,
            data_parallel_size=1
        )
        result_pp2 = llama3_70b.calculate_decode_performance(
            system_constraints=mem_constrained,
            batch_size=batch_size,
            prefill_length=kv_cache_tokens,
            output_length=1,
            parallelism_config=config_pp2
        )
        
        # Should scale even when memory-bound
        speedup = result_pp1.avg_step_time / result_pp2.avg_step_time
        assert 1.5 < speedup < 2.5, f"PP=2 speedup should be ~2x even when memory-bound, got {speedup:.2f}x"
    
    def test_tp_scales_in_memory_bound_scenario(self, llama3_70b):
        """TP should scale when memory-bound."""
        mem_constrained = SystemConstraints(
            memory_capacity=80 * (1024**3),
            memory_bandwidth=800 * (1024**3),
            compute_throughput=312e12,
            network_bandwidth=450 * (1024**3)
        )
        
        batch_size = 1
        kv_cache_tokens = 4096
        
        # TP=1
        config_tp1 = ParallelismConfig(
            parallelism_type=ParallelismType.TENSOR_PARALLEL,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1
        )
        result_tp1 = llama3_70b.calculate_decode_performance(
            system_constraints=mem_constrained,
            batch_size=batch_size,
            prefill_length=kv_cache_tokens,
            output_length=1,
            parallelism_config=config_tp1
        )
        
        # TP=4
        config_tp4 = ParallelismConfig(
            parallelism_type=ParallelismType.TENSOR_PARALLEL,
            tensor_parallel_size=4,
            pipeline_parallel_size=1,
            data_parallel_size=1
        )
        result_tp4 = llama3_70b.calculate_decode_performance(
            system_constraints=mem_constrained,
            batch_size=batch_size,
            prefill_length=kv_cache_tokens,
            output_length=1,
            parallelism_config=config_tp4
        )
        
        # Should scale
        speedup = result_tp1.avg_step_time / result_tp4.avg_step_time
        assert 3.0 < speedup < 5.0, f"TP=4 speedup should be ~4x when memory-bound, got {speedup:.2f}x"


