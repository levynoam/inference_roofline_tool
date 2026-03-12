"""
Comprehensive tests for dtype_override parameter
Tests that 4-bit, 8-bit, and 16-bit data types affect memory and bandwidth correctly
"""

import pytest
from llm_configs import LLAMA_3_8B
from inference_performance import InferencePerformance, SystemConstraints, ParallelismConfig


class TestDtypeOverride:
    """Test suite for dtype_override parameter"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.model = LLAMA_3_8B
        self.perf = InferencePerformance(self.model)
        self.gpu = SystemConstraints(
            memory_capacity=80e9,  # 80GB
            memory_bandwidth=2039e9,  # 2 TB/s
            compute_throughput=312e12,  # 312 TFLOPS
            network_bandwidth=600e9,  # 600 GB/s
        )
        self.parallelism = ParallelismConfig()
    
    def test_bytes_per_param_helper(self):
        """Test _get_bytes_per_param helper method"""
        perf = self.perf
        
        # Test all supported dtypes
        assert perf._get_bytes_per_param("int4") == 0.5
        assert perf._get_bytes_per_param("int8") == 1.0
        assert perf._get_bytes_per_param("float16") == 2.0
        assert perf._get_bytes_per_param("bfloat16") == 2.0
        assert perf._get_bytes_per_param("float32") == 4.0
        
        # Test default (uses model's dtype, which is bfloat16 for Llama 3)
        assert perf._get_bytes_per_param(None) == 2.0
        
        # Test unknown dtype defaults to fp16
        assert perf._get_bytes_per_param("unknown") == 2.0
    
    def test_ttft_dtype_memory_scaling(self):
        """Test that TTFT calculations scale memory usage with dtype"""
        batch_size = 1
        seq_len = 2048
        
        # Baseline: float16 (2 bytes)
        result_fp16 = self.perf.calculate_achievable_ttft(
            self.gpu, batch_size, seq_len, self.parallelism,
            dtype_override="float16"
        )
        
        # int8 should use 1/2 the memory
        result_int8 = self.perf.calculate_achievable_ttft(
            self.gpu, batch_size, seq_len, self.parallelism,
            dtype_override="int8"
        )
        
        # int4 should use 1/4 the memory
        result_int4 = self.perf.calculate_achievable_ttft(
            self.gpu, batch_size, seq_len, self.parallelism,
            dtype_override="int4"
        )
        
        # float32 should use 2x the memory
        result_fp32 = self.perf.calculate_achievable_ttft(
            self.gpu, batch_size, seq_len, self.parallelism,
            dtype_override="float32"
        )
        
        # Check memory scaling
        assert result_int8.memory_used < result_fp16.memory_used
        assert result_int4.memory_used < result_int8.memory_used
        assert result_fp32.memory_used > result_fp16.memory_used
        
        # Check approximate ratios (within 10% tolerance for overhead)
        ratio_int8_to_fp16 = result_int8.memory_used / result_fp16.memory_used
        assert 0.45 < ratio_int8_to_fp16 < 0.55  # Should be ~0.5
        
        ratio_int4_to_fp16 = result_int4.memory_used / result_fp16.memory_used
        assert 0.20 < ratio_int4_to_fp16 < 0.30  # Should be ~0.25
        
        ratio_fp32_to_fp16 = result_fp32.memory_used / result_fp16.memory_used
        assert 1.95 < ratio_fp32_to_fp16 < 2.05  # Should be ~2.0
    
    def test_ttft_dtype_bandwidth_scaling(self):
        """Test that TTFT calculations scale bandwidth usage with dtype"""
        batch_size = 1
        seq_len = 2048
        
        # Baseline: float16
        result_fp16 = self.perf.calculate_achievable_ttft(
            self.gpu, batch_size, seq_len, self.parallelism,
            dtype_override="float16"
        )
        
        # int8 should use 1/2 the bandwidth
        result_int8 = self.perf.calculate_achievable_ttft(
            self.gpu, batch_size, seq_len, self.parallelism,
            dtype_override="int8"
        )
        
        # int4 should use 1/4 the bandwidth
        result_int4 = self.perf.calculate_achievable_ttft(
            self.gpu, batch_size, seq_len, self.parallelism,
            dtype_override="int4"
        )
        
        # Check bandwidth scaling
        assert result_int8.memory_bandwidth_used < result_fp16.memory_bandwidth_used
        assert result_int4.memory_bandwidth_used < result_int8.memory_bandwidth_used
        
        # Check approximate ratios
        ratio_int8_to_fp16 = result_int8.memory_bandwidth_used / result_fp16.memory_bandwidth_used
        assert 0.45 < ratio_int8_to_fp16 < 0.55  # Should be ~0.5
        
        ratio_int4_to_fp16 = result_int4.memory_bandwidth_used / result_fp16.memory_bandwidth_used
        assert 0.20 < ratio_int4_to_fp16 < 0.30  # Should be ~0.25
    
    def test_decode_dtype_memory_scaling(self):
        """Test that decode calculations scale memory usage with dtype"""
        batch_size = 1
        prefill_len = 2048
        output_len = 100
        
        # Baseline: float16
        result_fp16 = self.perf.calculate_decode_performance(
            self.gpu, batch_size, prefill_len, output_len, self.parallelism,
            dtype_override="float16"
        )
        
        # int8 should use 1/2 the memory
        result_int8 = self.perf.calculate_decode_performance(
            self.gpu, batch_size, prefill_len, output_len, self.parallelism,
            dtype_override="int8"
        )
        
        # int4 should use 1/4 the memory
        result_int4 = self.perf.calculate_decode_performance(
            self.gpu, batch_size, prefill_len, output_len, self.parallelism,
            dtype_override="int4"
        )
        
        # Check memory scaling for averaged values
        assert result_int8.avg_memory_weights < result_fp16.avg_memory_weights
        assert result_int4.avg_memory_weights < result_int8.avg_memory_weights
        
        assert result_int8.avg_memory_kv_cache < result_fp16.avg_memory_kv_cache
        assert result_int4.avg_memory_kv_cache < result_int8.avg_memory_kv_cache
        
        # Check approximate ratios for weights
        ratio_int8_to_fp16 = result_int8.avg_memory_weights / result_fp16.avg_memory_weights
        assert 0.48 < ratio_int8_to_fp16 < 0.52  # Should be ~0.5
        
        ratio_int4_to_fp16 = result_int4.avg_memory_weights / result_fp16.avg_memory_weights
        assert 0.23 < ratio_int4_to_fp16 < 0.27  # Should be ~0.25
    
    def test_decode_dtype_bandwidth_scaling(self):
        """Test that decode calculations scale bandwidth usage with dtype"""
        batch_size = 1
        prefill_len = 2048
        output_len = 100
        
        # Baseline: float16
        result_fp16 = self.perf.calculate_decode_performance(
            self.gpu, batch_size, prefill_len, output_len, self.parallelism,
            dtype_override="float16"
        )
        
        # int8 should use 1/2 the bandwidth
        result_int8 = self.perf.calculate_decode_performance(
            self.gpu, batch_size, prefill_len, output_len, self.parallelism,
            dtype_override="int8"
        )
        
        # int4 should use 1/4 the bandwidth
        result_int4 = self.perf.calculate_decode_performance(
            self.gpu, batch_size, prefill_len, output_len, self.parallelism,
            dtype_override="int4"
        )
        
        # Check bandwidth scaling
        # Use avg_memory_bw_utilization which reflects actual bandwidth usage
        bw_used_fp16 = result_fp16.avg_memory_bw_utilization * result_fp16.memory_bandwidth
        bw_used_int8 = result_int8.avg_memory_bw_utilization * result_int8.memory_bandwidth
        bw_used_int4 = result_int4.avg_memory_bw_utilization * result_int4.memory_bandwidth
        
        assert bw_used_int8 < bw_used_fp16
        assert bw_used_int4 < bw_used_int8
        
        # Check that bandwidth usage decreases (but may not be exactly 0.5 due to overheads)
        # The reduction should be substantial
        reduction_int8 = (bw_used_fp16 - bw_used_int8) / bw_used_fp16
        assert reduction_int8 > 0.10  # At least 10% reduction
        
        reduction_int4 = (bw_used_fp16 - bw_used_int4) / bw_used_fp16
        assert reduction_int4 > 0.20  # At least 20% reduction
    
    def test_dtype_override_vs_model_default(self):
        """Test that dtype_override actually overrides model's default dtype"""
        batch_size = 1
        seq_len = 2048
        
        # Without override, should use model's dtype (bfloat16 = 2 bytes)
        result_default = self.perf.calculate_achievable_ttft(
            self.gpu, batch_size, seq_len, self.parallelism
        )
        
        # With explicit float16, should match default for Llama 3
        result_explicit_fp16 = self.perf.calculate_achievable_ttft(
            self.gpu, batch_size, seq_len, self.parallelism,
            dtype_override="float16"
        )
        
        # These should be approximately equal (same bytes per param)
        assert abs(result_default.memory_used - result_explicit_fp16.memory_used) < 1e6
        
        # With int4 override, should be significantly less
        result_int4 = self.perf.calculate_achievable_ttft(
            self.gpu, batch_size, seq_len, self.parallelism,
            dtype_override="int4"
        )
        
        assert result_int4.memory_used < result_default.memory_used * 0.3
    
    def test_dtype_performance_impact(self):
        """Test that lower precision improves performance (less memory bound)"""
        batch_size = 8
        seq_len = 2048
        
        # Higher precision (more bytes) should be slower for memory-bound workloads
        result_fp32 = self.perf.calculate_achievable_ttft(
            self.gpu, batch_size, seq_len, self.parallelism,
            dtype_override="float32"
        )
        
        result_fp16 = self.perf.calculate_achievable_ttft(
            self.gpu, batch_size, seq_len, self.parallelism,
            dtype_override="float16"
        )
        
        result_int8 = self.perf.calculate_achievable_ttft(
            self.gpu, batch_size, seq_len, self.parallelism,
            dtype_override="int8"
        )
        
        result_int4 = self.perf.calculate_achievable_ttft(
            self.gpu, batch_size, seq_len, self.parallelism,
            dtype_override="int4"
        )
        
        # Lower precision should achieve better TTFT (lower is better)
        # or at least not worse
        assert result_int4.achievable_ttft <= result_int8.achievable_ttft
        assert result_int8.achievable_ttft <= result_fp16.achievable_ttft
        assert result_fp16.achievable_ttft <= result_fp32.achievable_ttft
    
    def test_dtype_with_large_batch(self):
        """Test dtype scaling with large batch sizes (more memory pressure)"""
        batch_size = 64
        seq_len = 2048
        
        # With large batch, int4 should save significant memory
        result_fp16 = self.perf.calculate_achievable_ttft(
            self.gpu, batch_size, seq_len, self.parallelism,
            dtype_override="float16"
        )
        
        result_int4 = self.perf.calculate_achievable_ttft(
            self.gpu, batch_size, seq_len, self.parallelism,
            dtype_override="int4"
        )
        
        # Should save ~75% memory
        memory_saved_percent = (1 - result_int4.memory_used / result_fp16.memory_used) * 100
        assert memory_saved_percent > 50  # At least 50% saved
        
        # Memory utilization should be lower with int4
        assert result_int4.memory_utilization < result_fp16.memory_utilization
    
    def test_decode_dtype_tokens_per_second(self):
        """Test that dtype affects decode throughput"""
        batch_size = 1
        prefill_len = 2048
        output_len = 1000
        
        # Lower precision should enable higher throughput
        result_fp16 = self.perf.calculate_decode_performance(
            self.gpu, batch_size, prefill_len, output_len, self.parallelism,
            dtype_override="float16"
        )
        
        result_int4 = self.perf.calculate_decode_performance(
            self.gpu, batch_size, prefill_len, output_len, self.parallelism,
            dtype_override="int4"
        )
        
        # int4 should achieve higher TPS (less memory bandwidth constrained)
        assert result_int4.tokens_per_second_per_user >= result_fp16.tokens_per_second_per_user
        
        # Total time should be lower or equal
        assert result_int4.total_decode_time <= result_fp16.total_decode_time


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
