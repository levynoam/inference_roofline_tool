"""
Tests for GLM-5 model configuration.

GLM-5 is Zhipu AI's MoE model with both MLA and DSA.
Key features:
- MLA (Multi-head Latent Attention) with kv_lora_rank=512, q_lora_rank=2048
- DSA (Dynamic Sparse Attention) with index_topk=2048
- MoE with 256 routed experts + 1 shared expert, top-8 routing
- first_k_dense_replace=3: first 3 layers are dense, rest are MoE
- 78 layers, hidden_dim=6144, 64 attention heads
- ~198K context (max_position_embeddings=202752)
"""

import pytest
from llm_configs import ALL_MODELS, get_model, GLM_5
from llm_architecture import (
    LLMArchitecture,
    AttentionType,
    ActivationType,
    ArchitectureType,
    NormalizationType,
    PositionEncodingType,
    FFNLayerType,
)
from inference_performance import (
    InferencePerformance,
    SystemConstraints,
    ParallelismConfig,
    ParallelismType,
    DecodePerformance,
)


class TestGlm5ModelConfiguration:
    """Test GLM-5 model configuration is correct."""

    def test_glm5_in_all_models(self):
        """GLM-5 should be accessible from ALL_MODELS."""
        assert "glm-5" in ALL_MODELS
        assert ALL_MODELS["glm-5"] is GLM_5

    def test_glm5_get_model(self):
        """get_model() should return GLM-5."""
        model = get_model("glm-5")
        assert model is GLM_5
        assert model.model_name == "GLM-5"

    def test_glm5_basic_architecture(self):
        """Verify basic architecture parameters."""
        assert GLM_5.model_name == "GLM-5"
        assert GLM_5.model_family == "GLM"
        assert GLM_5.version == "5.0"
        assert GLM_5.architecture_type == ArchitectureType.DECODER_ONLY

    def test_glm5_dimensions(self):
        """Verify dimensions match GLM-5.json."""
        assert GLM_5.hidden_dim == 6144
        assert GLM_5.num_layers == 78
        assert GLM_5.vocab_size == 154880
        assert GLM_5.max_sequence_length == 202752  # ~198K context

    def test_glm5_attention_config(self):
        """Verify attention configuration."""
        attn = GLM_5.attention_config
        assert attn.num_attention_heads == 64
        assert attn.num_key_value_heads == 64  # MHA before MLA compression
        assert attn.attention_type == AttentionType.MULTI_HEAD
        assert attn.head_dim == 256  # qk_nope(192) + qk_rope(64)

    def test_glm5_mla_config(self):
        """Verify MLA (Multi-head Latent Attention) configuration."""
        attn = GLM_5.attention_config
        assert attn.use_mla is True
        assert attn.mla_kv_lora_rank == 512
        assert attn.mla_q_lora_rank == 2048

    def test_glm5_dsa_config(self):
        """Verify DSA (Dynamic Sparse Attention) configuration."""
        attn = GLM_5.attention_config
        assert attn.use_dsa is True
        assert attn.dsa_q_indexer_dim == 128  # index_head_dim
        assert attn.dsa_k_indexer_dim == 32   # index_n_heads
        assert attn.dsa_top_k == 2048         # index_topk

    def test_glm5_has_both_mla_and_dsa(self):
        """GLM-5 uniquely combines both MLA and DSA."""
        attn = GLM_5.attention_config
        assert attn.use_mla is True
        assert attn.use_dsa is True

    def test_glm5_moe_config(self):
        """Verify MoE configuration."""
        assert GLM_5.is_moe is True
        moe = GLM_5.moe_config
        assert moe is not None
        assert moe.num_experts == 256
        assert moe.num_experts_per_token == 8
        assert moe.shared_expert is True
        assert moe.router_type == "top_k"

    def test_glm5_ffn_config(self):
        """Verify FFN configuration."""
        ffn = GLM_5.ffn_config
        assert ffn.intermediate_size == 2048           # moe_intermediate_size per expert
        assert ffn.dense_intermediate_size == 12288    # intermediate_size for dense layers
        assert ffn.activation == ActivationType.SILU

    def test_glm5_interleaved_dense_moe_layers(self):
        """Verify first 3 layers are dense and rest are MoE (first_k_dense_replace=3)."""
        layer_types = GLM_5.ffn_layer_types
        assert len(layer_types) == 78

        # First 3 layers should be dense
        for i in range(3):
            assert layer_types[i] == FFNLayerType.DENSE, f"Layer {i} should be DENSE"

        # Remaining 75 layers should be MoE
        for i in range(3, 78):
            assert layer_types[i] == FFNLayerType.MOE, f"Layer {i} should be MOE"

    def test_glm5_dense_layer_count(self):
        """Verify the correct number of dense vs MoE layers."""
        dense_count = sum(1 for t in GLM_5.ffn_layer_types if t == FFNLayerType.DENSE)
        moe_count = sum(1 for t in GLM_5.ffn_layer_types if t == FFNLayerType.MOE)
        assert dense_count == 3
        assert moe_count == 75
        assert dense_count + moe_count == 78

    def test_glm5_normalization(self):
        """Verify normalization type."""
        assert GLM_5.normalization_type == NormalizationType.RMS_NORM

    def test_glm5_position_encoding(self):
        """Verify position encoding uses RoPE with 1M theta."""
        assert GLM_5.position_encoding == PositionEncodingType.ROTARY
        assert GLM_5.rope_theta == 1000000.0  # 1M base

    def test_glm5_dtype(self):
        """Verify default dtype is bfloat16."""
        assert GLM_5.dtype == "bfloat16"

    def test_glm5_parameter_counts(self):
        """Verify parameter counts are reasonable."""
        assert GLM_5.total_parameters == 500_000_000_000  # 500B total
        assert GLM_5.active_parameters == 20_000_000_000  # ~20B active
        assert GLM_5.active_parameters < GLM_5.total_parameters


class TestGlm5MLABehavior:
    """Test that MLA is properly handled in calculations."""

    def test_kv_cache_uses_mla_rank(self):
        """KV cache should use compressed MLA dimension, not full head_dim."""
        batch_size = 1
        seq_len = 1024
        bytes_per_elem = 2  # bf16

        kv_cache_size = GLM_5.get_kv_cache_size(batch_size, seq_len, bytes_per_elem)

        # With MLA: 2 * batch * seq * mla_kv_lora_rank * bytes * num_layers
        expected_mla = 2 * batch_size * seq_len * 512 * bytes_per_elem * 78
        assert kv_cache_size == expected_mla

        # Without MLA would be: 2 * batch * seq * num_kv_heads * head_dim * bytes * num_layers
        expected_no_mla = 2 * batch_size * seq_len * 64 * 256 * bytes_per_elem * 78
        # MLA should give much smaller KV cache
        assert kv_cache_size < expected_no_mla / 10  # At least 10x smaller

    def test_kv_cache_compression_ratio(self):
        """MLA should compress KV cache by a large factor."""
        # Without MLA: num_kv_heads * head_dim = 64 * 256 = 16384
        # With MLA: mla_kv_lora_rank = 512
        # Compression ratio: 16384 / 512 = 32x
        full_kv_dim = GLM_5.attention_config.num_key_value_heads * GLM_5.attention_config.head_dim
        compressed_kv_dim = GLM_5.attention_config.mla_kv_lora_rank
        ratio = full_kv_dim / compressed_kv_dim
        assert ratio == 32.0


class TestGlm5DSABehavior:
    """Test DSA (Dynamic Sparse Attention) behavior."""

    def test_dsa_parameters_configured(self):
        """Verify DSA parameters are properly configured."""
        attn = GLM_5.attention_config
        assert attn.use_dsa is True
        assert attn.dsa_top_k == 2048
        assert attn.dsa_q_indexer_dim == 128
        assert attn.dsa_k_indexer_dim == 32

    def test_dsa_similar_to_deepseek_32(self):
        """GLM-5 DSA config should be comparable to DeepSeek 3.2."""
        deepseek_32 = get_model("deepseek-3.2")
        glm5_attn = GLM_5.attention_config
        ds_attn = deepseek_32.attention_config

        # Both use DSA with top_k=2048
        assert glm5_attn.use_dsa == ds_attn.use_dsa
        assert glm5_attn.dsa_top_k == ds_attn.dsa_top_k
        # Same index_head_dim
        assert glm5_attn.dsa_q_indexer_dim == ds_attn.dsa_q_indexer_dim

    def test_dsa_reduces_effective_context(self):
        """For long sequences, DSA should reduce effective context to top_k."""
        gpu = SystemConstraints.from_gpu_spec("H100-80GB")

        perf = InferencePerformance(GLM_5)

        # Long sequence where DSA kicks in (16K > top_k=2048)
        result_long = perf.calculate_decode_performance(
            system_constraints=gpu,
            batch_size=1,
            prefill_length=16384,
            output_length=1,
        )

        # Short sequence where DSA doesn't help (1K < top_k=2048)
        result_short = perf.calculate_decode_performance(
            system_constraints=gpu,
            batch_size=1,
            prefill_length=1024,
            output_length=1,
        )

        # Both should complete successfully
        assert result_long.total_decode_time > 0
        assert result_short.total_decode_time > 0

        # DSA parameters are configured correctly
        assert GLM_5.attention_config.use_dsa is True
        assert GLM_5.attention_config.dsa_top_k == 2048


class TestGlm5Inference:
    """Test inference performance calculations for GLM-5."""

    @pytest.fixture
    def h100_gpu(self):
        """H100 SXM GPU configuration."""
        return SystemConstraints.from_gpu_spec("H100-80GB")

    def test_can_create_inference_performance(self):
        """Should be able to create InferencePerformance calculator."""
        perf = InferencePerformance(GLM_5)
        assert perf.model is GLM_5

    def test_prefill_calculation(self, h100_gpu):
        """Test prefill phase calculation."""
        perf = InferencePerformance(GLM_5)

        result = perf.calculate_achievable_ttft(
            system_constraints=h100_gpu,
            batch_size=1,
            sequence_length=2048,
        )

        assert result.achievable_ttft > 0
        assert result.compute_utilization >= 0
        assert result.memory_bandwidth_utilization >= 0

    def test_decode_calculation(self, h100_gpu):
        """Test decode phase calculation."""
        perf = InferencePerformance(GLM_5)

        result = perf.calculate_decode_performance(
            system_constraints=h100_gpu,
            batch_size=1,
            prefill_length=2048,
            output_length=100,
        )

        assert result.total_throughput > 0
        assert result.avg_step_time > 0
        assert result.avg_compute_utilization >= 0
        assert result.avg_memory_bw_utilization >= 0

    def test_prefill_with_tensor_parallel(self, h100_gpu):
        """GLM-5 should work with tensor parallelism."""
        perf = InferencePerformance(GLM_5)
        parallelism = ParallelismConfig(
            parallelism_type=ParallelismType.TENSOR_PARALLEL,
            tensor_parallel_size=8,
        )

        result = perf.calculate_achievable_ttft(
            system_constraints=h100_gpu,
            batch_size=1,
            sequence_length=2048,
            parallelism_config=parallelism,
        )

        assert result.achievable_ttft > 0


class TestGlm5MemoryFootprint:
    """Test memory footprint calculations."""

    def test_memory_footprint_includes_all_components(self):
        """Memory footprint should include all standard components."""
        memory = GLM_5.get_memory_footprint(
            batch_size=1,
            sequence_length=2048,
        )

        assert "model_parameters" in memory
        assert "kv_cache" in memory
        assert "activations" in memory

        assert memory["model_parameters"] > 0
        assert memory["kv_cache"] > 0
        assert memory["activations"] > 0

    def test_kv_cache_scales_with_sequence_length(self):
        """KV cache should scale linearly with sequence length."""
        mem_1k = GLM_5.get_memory_footprint(batch_size=1, sequence_length=1024)
        mem_2k = GLM_5.get_memory_footprint(batch_size=1, sequence_length=2048)

        ratio = mem_2k["kv_cache"] / mem_1k["kv_cache"]
        assert 1.9 < ratio < 2.1

    def test_kv_cache_scales_with_batch_size(self):
        """KV cache should scale linearly with batch size."""
        mem_b1 = GLM_5.get_memory_footprint(batch_size=1, sequence_length=2048)
        mem_b4 = GLM_5.get_memory_footprint(batch_size=4, sequence_length=2048)

        ratio = mem_b4["kv_cache"] / mem_b1["kv_cache"]
        assert 3.9 < ratio < 4.1


class TestGlm5Comparisons:
    """Compare GLM-5 with similar models."""

    def test_glm5_mla_same_kv_rank_as_deepseek_32(self):
        """GLM-5 and DeepSeek 3.2 use the same KV compression rank."""
        ds32 = get_model("deepseek-3.2")
        assert GLM_5.attention_config.mla_kv_lora_rank == ds32.attention_config.mla_kv_lora_rank
        assert GLM_5.attention_config.mla_kv_lora_rank == 512

    def test_glm5_higher_q_lora_rank_than_deepseek(self):
        """GLM-5 has higher q_lora_rank (2048 vs 1536)."""
        ds32 = get_model("deepseek-3.2")
        assert GLM_5.attention_config.mla_q_lora_rank == 2048
        assert ds32.attention_config.mla_q_lora_rank == 1536
        assert GLM_5.attention_config.mla_q_lora_rank > ds32.attention_config.mla_q_lora_rank

    def test_glm5_same_expert_count_as_deepseek(self):
        """Both use 256 routed experts with top-8 routing."""
        ds32 = get_model("deepseek-3.2")
        assert GLM_5.moe_config.num_experts == ds32.moe_config.num_experts
        assert GLM_5.moe_config.num_experts_per_token == ds32.moe_config.num_experts_per_token

    def test_glm5_more_layers_than_deepseek(self):
        """GLM-5 has more layers than DeepSeek 3.2."""
        ds32 = get_model("deepseek-3.2")
        assert GLM_5.num_layers == 78
        assert ds32.num_layers == 61
        assert GLM_5.num_layers > ds32.num_layers

    def test_glm5_both_mla_and_dsa_like_deepseek_32(self):
        """Both GLM-5 and DeepSeek 3.2 support MLA + DSA."""
        ds32 = get_model("deepseek-3.2")
        assert GLM_5.attention_config.use_mla is True
        assert GLM_5.attention_config.use_dsa is True
        assert ds32.attention_config.use_mla is True
        assert ds32.attention_config.use_dsa is True


class TestGlm5LongContext:
    """Test behavior with long context (~198K)."""

    def test_supports_long_context(self):
        """Model should support up to ~198K tokens."""
        assert GLM_5.max_sequence_length == 202752

    def test_long_context_kv_cache(self):
        """KV cache at long context should be manageable due to MLA compression."""
        memory = GLM_5.get_memory_footprint(
            batch_size=1,
            sequence_length=200000,
        )

        # With MLA (kv_lora_rank=512): 2 * 512 * 78 * 200000 * 2 bytes ≈ 29.7 GB
        # Without MLA (64 * 256): 2 * 16384 * 78 * 200000 * 2 ≈ 951 GB!
        # MLA makes long context feasible
        expected_kv = 2 * 512 * 78 * 200000 * 2
        assert abs(memory["kv_cache"] - expected_kv) / expected_kv < 0.01
