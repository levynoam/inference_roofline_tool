"""
Tests for Qwen3-Coder-Next model configuration.

Qwen3-Coder-Next is Alibaba's next-generation coding MoE model.
Key features:
- 512 experts with top-10 routing (massive expert pool)
- MLA-style compressed attention
- GQA with 8:1 ratio (16 heads, 2 KV heads)
- 262K context
- 48 layers, hidden_dim=2048
- Shared expert
"""

import pytest
from llm_configs import ALL_MODELS, get_model, QWEN3_CODER_NEXT
from llm_architecture import (
    LLMArchitecture,
    AttentionType,
    ActivationType,
    ArchitectureType,
    NormalizationType,
    PositionEncodingType,
)
from inference_performance import (
    InferencePerformance,
    SystemConstraints,
    ParallelismConfig,
    ParallelismType,
    DecodePerformance,
)


class TestQwen3CoderNextModelConfiguration:
    """Test Qwen3-Coder-Next model configuration is correct."""
    
    def test_qwen3_coder_next_in_all_models(self):
        """Qwen3-Coder-Next should be accessible from ALL_MODELS."""
        assert "qwen3-coder-next" in ALL_MODELS
        assert ALL_MODELS["qwen3-coder-next"] is QWEN3_CODER_NEXT
    
    def test_qwen3_coder_next_get_model(self):
        """get_model() should return Qwen3-Coder-Next."""
        model = get_model("qwen3-coder-next")
        assert model is QWEN3_CODER_NEXT
        assert model.model_name == "Qwen3-Coder-Next"
    
    def test_qwen3_coder_next_basic_architecture(self):
        """Verify basic architecture parameters."""
        assert QWEN3_CODER_NEXT.model_name == "Qwen3-Coder-Next"
        assert QWEN3_CODER_NEXT.model_family == "Qwen"
        assert QWEN3_CODER_NEXT.version == "3.0-Next"
        assert QWEN3_CODER_NEXT.architecture_type == ArchitectureType.DECODER_ONLY
    
    def test_qwen3_coder_next_dimensions(self):
        """Verify dimensions match config.json."""
        assert QWEN3_CODER_NEXT.hidden_dim == 2048
        assert QWEN3_CODER_NEXT.num_layers == 48
        assert QWEN3_CODER_NEXT.vocab_size == 151936
        assert QWEN3_CODER_NEXT.max_sequence_length == 262144  # 262K context
    
    def test_qwen3_coder_next_attention_config(self):
        """Verify attention configuration with GQA."""
        attn = QWEN3_CODER_NEXT.attention_config
        assert attn.num_attention_heads == 16
        assert attn.num_key_value_heads == 2  # GQA with 8:1 ratio
        assert attn.attention_type == AttentionType.GROUPED_QUERY
        assert attn.head_dim == 256
    
    def test_qwen3_coder_next_gqa_ratio(self):
        """Verify 8:1 GQA ratio."""
        attn = QWEN3_CODER_NEXT.attention_config
        gqa_ratio = attn.num_attention_heads / attn.num_key_value_heads
        assert gqa_ratio == 8.0
    
    def test_qwen3_coder_next_mla_config(self):
        """Verify MLA configuration."""
        attn = QWEN3_CODER_NEXT.attention_config
        assert attn.use_mla == True
        assert attn.mla_kv_lora_rank == 128  # Compressed KV dimension
    
    def test_qwen3_coder_next_moe_config(self):
        """Verify MoE configuration with massive expert pool."""
        assert QWEN3_CODER_NEXT.is_moe == True
        moe = QWEN3_CODER_NEXT.moe_config
        assert moe is not None
        assert moe.num_experts == 512  # Massive expert pool
        assert moe.num_experts_per_token == 10  # top-10 routing
        assert moe.shared_expert == True  # Has shared expert
        assert moe.router_type == "top_k"
    
    def test_qwen3_coder_next_ffn_config(self):
        """Verify FFN configuration."""
        ffn = QWEN3_CODER_NEXT.ffn_config
        assert ffn.intermediate_size == 512  # moe_intermediate_size per expert
        assert ffn.activation == ActivationType.SILU
    
    def test_qwen3_coder_next_normalization(self):
        """Verify normalization type."""
        assert QWEN3_CODER_NEXT.normalization_type == NormalizationType.RMS_NORM
    
    def test_qwen3_coder_next_position_encoding(self):
        """Verify position encoding uses RoPE with 5M theta."""
        assert QWEN3_CODER_NEXT.position_encoding == PositionEncodingType.ROTARY
        assert QWEN3_CODER_NEXT.rope_theta == 5000000.0  # 5M base
    
    def test_qwen3_coder_next_dtype(self):
        """Verify default dtype is bfloat16."""
        assert QWEN3_CODER_NEXT.dtype == "bfloat16"
    
    def test_qwen3_coder_next_parameter_counts(self):
        """Verify parameter counts are reasonable."""
        assert QWEN3_CODER_NEXT.total_parameters == 53_000_000_000  # 53B total
        assert QWEN3_CODER_NEXT.active_parameters == 1_000_000_000  # ~1B active
        # Active should be much less than total for MoE
        assert QWEN3_CODER_NEXT.active_parameters < QWEN3_CODER_NEXT.total_parameters


class TestQwen3CoderNextMLABehavior:
    """Test that MLA is properly handled in calculations."""
    
    def test_kv_cache_uses_mla_rank(self):
        """KV cache should use compressed MLA rank instead of full dimensions."""
        # With MLA: KV cache uses mla_kv_lora_rank instead of (num_kv_heads * head_dim)
        # Test by checking actual KV cache size from memory footprint
        
        memory = QWEN3_CODER_NEXT.get_memory_footprint(
            batch_size=1,
            sequence_length=1000,
        )
        
        kv_cache_total = memory["kv_cache"]
        
        # Expected: 2 (K+V) * 128 (mla_kv_lora_rank) * 48 (layers) * 1000 (seq_len) * 2 (bytes)
        expected_kv = 2 * 128 * 48 * 1000 * 2
        assert kv_cache_total == expected_kv
        
        # Verify compression: Should be much smaller than without MLA
        # Without MLA would be: 2 * 2 * 256 * 48 * 1000 * 2
        kv_without_mla = 2 * 2 * 256 * 48 * 1000 * 2
        assert kv_cache_total < kv_without_mla
        
        # Compression ratio should be 4x (512 / 128 = 4)
        compression_ratio = kv_without_mla / kv_cache_total
        assert compression_ratio == 4.0


class TestQwen3CoderNextInference:
    """Test inference performance calculations for Qwen3-Coder-Next."""
    
    def test_can_create_inference_performance(self):
        """Should be able to create InferencePerformance calculator."""
        perf = InferencePerformance(QWEN3_CODER_NEXT)
        assert perf.model is QWEN3_CODER_NEXT
    
    def test_prefill_calculation(self):
        """Test prefill phase calculation."""
        perf = InferencePerformance(QWEN3_CODER_NEXT)
        
        # H100 GPU constraints
        gpu = SystemConstraints(
            memory_capacity=80e9,  # 80GB
            memory_bandwidth=3.35e12,  # 3.35 TB/s
            compute_throughput=989.4e12,  # 989.4 TFLOPS BF16
            network_bandwidth=900e9,  # 900 GB/s NVLink
        )
        
        result = perf.calculate_achievable_ttft(
            system_constraints=gpu,
            batch_size=1,
            sequence_length=2048,
        )
        
        assert result.achievable_ttft > 0
        assert result.compute_utilization >= 0
        assert result.memory_bandwidth_utilization >= 0
    
    def test_decode_calculation(self):
        """Test decode phase calculation."""
        perf = InferencePerformance(QWEN3_CODER_NEXT)
        
        # H100 GPU constraints
        gpu = SystemConstraints(
            memory_capacity=80e9,  # 80GB
            memory_bandwidth=3.35e12,  # 3.35 TB/s
            compute_throughput=989.4e12,  # 989.4 TFLOPS BF16
            network_bandwidth=900e9,  # 900 GB/s NVLink
        )
        
        result = perf.calculate_decode_performance(
            system_constraints=gpu,
            batch_size=1,
            prefill_length=2048,
            output_length=100,
        )
        
        assert result.total_throughput > 0
        assert result.avg_step_time > 0
        assert result.avg_compute_utilization >= 0
        assert result.avg_memory_bw_utilization >= 0


class TestQwen3CoderNextMoEScaling:
    """Test MoE-specific behaviors with large expert pool."""
    
    def test_active_parameters_much_smaller_than_total(self):
        """With 512 experts and top-10, active should be ~2% of total."""
        active_ratio = QWEN3_CODER_NEXT.active_parameters / QWEN3_CODER_NEXT.total_parameters
        # 10 experts out of 512 = ~1.95%, plus some overhead for attention/shared
        assert active_ratio < 0.05  # Less than 5%
        assert active_ratio > 0.01  # More than 1%
    
    def test_inference_uses_active_parameters(self):
        """Inference calculations should use active parameters."""
        perf = InferencePerformance(QWEN3_CODER_NEXT)
        
        gpu = SystemConstraints(
            memory_capacity=80e9,
            memory_bandwidth=3.35e12,
            compute_throughput=989.4e12,
            network_bandwidth=900e9,
        )
        
        # Calculate memory for model weights
        model_memory = QWEN3_CODER_NEXT.active_parameters * 2  # BF16 = 2 bytes
        
        # Should be able to fit in 80GB with room for KV cache
        assert model_memory < 80e9  # ~2GB for active params


class TestQwen3CoderNextMemoryFootprint:
    """Test memory footprint calculations."""
    
    def test_memory_footprint_includes_kv_cache(self):
        """Memory footprint should include KV cache."""
        memory = QWEN3_CODER_NEXT.get_memory_footprint(
            batch_size=1,
            sequence_length=2048,
        )
        
        assert "model_parameters" in memory
        assert "kv_cache" in memory
        assert "activations" in memory
        
        # All components should be positive
        assert memory["model_parameters"] > 0
        assert memory["kv_cache"] > 0
        assert memory["activations"] > 0
    
    def test_kv_cache_scales_with_sequence_length(self):
        """KV cache should scale linearly with sequence length."""
        mem_1k = QWEN3_CODER_NEXT.get_memory_footprint(batch_size=1, sequence_length=1024)
        mem_2k = QWEN3_CODER_NEXT.get_memory_footprint(batch_size=1, sequence_length=2048)
        
        kv_1k = mem_1k["kv_cache"]
        kv_2k = mem_2k["kv_cache"]
        
        # Should be approximately 2x
        ratio = kv_2k / kv_1k
        assert 1.9 < ratio < 2.1  # Allow small tolerance
    
    def test_kv_cache_scales_with_batch_size(self):
        """KV cache should scale linearly with batch size."""
        mem_b1 = QWEN3_CODER_NEXT.get_memory_footprint(batch_size=1, sequence_length=2048)
        mem_b4 = QWEN3_CODER_NEXT.get_memory_footprint(batch_size=4, sequence_length=2048)
        
        kv_b1 = mem_b1["kv_cache"]
        kv_b4 = mem_b4["kv_cache"]
        
        # Should be approximately 4x
        ratio = kv_b4 / kv_b1
        assert 3.9 < ratio < 4.1  # Allow small tolerance


class TestQwen3CoderNextLongContext:
    """Test behavior with long context (262K)."""
    
    def test_supports_long_context(self):
        """Model should support up to 262K tokens."""
        assert QWEN3_CODER_NEXT.max_sequence_length == 262144
    
    def test_long_context_memory_requirements(self):
        """Test memory requirements for long context."""
        # At 256K context with batch=1
        memory = QWEN3_CODER_NEXT.get_memory_footprint(
            batch_size=1,
            sequence_length=256000,
        )
        
        # KV cache should be dominant at long context
        assert memory["kv_cache"] > memory["model_parameters"]
        
        # Calculate expected KV cache size
        # 2 (K+V) * 128 (mla_rank) * 48 (layers) * 256000 (seq_len) * 2 (bytes)
        expected_kv = 2 * 128 * 48 * 256000 * 2
        assert abs(memory["kv_cache"] - expected_kv) / expected_kv < 0.01  # Within 1%
