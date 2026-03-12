"""
Tests for Kimi K2.5 model configuration.

Kimi K2.5 is Moonshot AI's multimodal model with a DeepSeek V3-based text backbone.
Key features:
- MLA (Multi-head Latent Attention) with kv_lora_rank=512, q_lora_rank=1536
- MoE with 384 routed experts + 1 shared expert, top-8 routing
- 256K context with YaRN RoPE scaling
- 61 layers, hidden_dim=7168
"""

import pytest
from llm_configs import ALL_MODELS, get_model, KIMI_K25
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


class TestKimiK25ModelConfiguration:
    """Test Kimi K2.5 model configuration is correct."""
    
    def test_kimi_k25_in_all_models(self):
        """Kimi K2.5 should be accessible from ALL_MODELS."""
        assert "kimi-k2.5" in ALL_MODELS
        assert ALL_MODELS["kimi-k2.5"] is KIMI_K25
    
    def test_kimi_k25_get_model(self):
        """get_model() should return Kimi K2.5."""
        model = get_model("kimi-k2.5")
        assert model is KIMI_K25
        assert model.model_name == "Kimi-K2.5"
    
    def test_kimi_k25_basic_architecture(self):
        """Verify basic architecture parameters."""
        assert KIMI_K25.model_name == "Kimi-K2.5"
        assert KIMI_K25.model_family == "Kimi"
        assert KIMI_K25.version == "2.5"
        assert KIMI_K25.architecture_type == ArchitectureType.DECODER_ONLY
    
    def test_kimi_k25_dimensions(self):
        """Verify dimensions match config.json."""
        assert KIMI_K25.hidden_dim == 7168
        assert KIMI_K25.num_layers == 61
        assert KIMI_K25.vocab_size == 163840
        assert KIMI_K25.max_sequence_length == 262144  # 256K context
    
    def test_kimi_k25_attention_config(self):
        """Verify attention configuration with MLA."""
        attn = KIMI_K25.attention_config
        assert attn.num_attention_heads == 64
        assert attn.num_key_value_heads == 64  # MHA before MLA compression
        assert attn.attention_type == AttentionType.MULTI_HEAD
        assert attn.head_dim == 192  # qk_nope(128) + qk_rope(64)
    
    def test_kimi_k25_mla_config(self):
        """Verify MLA (Multi-head Latent Attention) configuration."""
        attn = KIMI_K25.attention_config
        assert attn.use_mla == True
        assert attn.mla_kv_lora_rank == 512
        assert attn.mla_q_lora_rank == 1536
    
    def test_kimi_k25_moe_config(self):
        """Verify MoE configuration."""
        assert KIMI_K25.is_moe == True
        moe = KIMI_K25.moe_config
        assert moe is not None
        assert moe.num_experts == 384
        assert moe.num_experts_per_token == 8
        assert moe.shared_expert == True
        assert moe.router_type == "top_k"
    
    def test_kimi_k25_ffn_config(self):
        """Verify FFN configuration."""
        ffn = KIMI_K25.ffn_config
        assert ffn.intermediate_size == 18432
        assert ffn.activation == ActivationType.SILU
    
    def test_kimi_k25_normalization(self):
        """Verify normalization type."""
        assert KIMI_K25.normalization_type == NormalizationType.RMS_NORM
    
    def test_kimi_k25_position_encoding(self):
        """Verify position encoding uses RoPE."""
        assert KIMI_K25.position_encoding == PositionEncodingType.ROTARY
        assert KIMI_K25.rope_theta == 50000.0
    
    def test_kimi_k25_dtype(self):
        """Verify default dtype is bfloat16."""
        assert KIMI_K25.dtype == "bfloat16"
    
    def test_kimi_k25_parameters(self):
        """Verify parameter counts."""
        # Total parameters ~1T
        assert KIMI_K25.total_parameters == 1_000_000_000_000
        # Active parameters ~32B (8 experts + shared + attention)
        assert KIMI_K25.active_parameters == 32_000_000_000


class TestKimiK25MLABehavior:
    """Test that MLA is properly handled in calculations."""
    
    def test_kv_cache_uses_mla_rank(self):
        """KV cache should use compressed MLA dimension, not full head_dim."""
        # With MLA, KV cache uses mla_kv_lora_rank instead of num_kv_heads * head_dim
        batch_size = 1
        seq_len = 1024
        bytes_per_elem = 2  # bf16
        
        kv_cache_size = KIMI_K25.get_kv_cache_size(batch_size, seq_len, bytes_per_elem)
        
        # With MLA: 2 * batch * seq * mla_kv_lora_rank * bytes * num_layers
        # = 2 * 1 * 1024 * 512 * 2 * 61 = 127.9 MB
        expected_mla = 2 * batch_size * seq_len * 512 * bytes_per_elem * 61
        
        # Without MLA would be: 2 * batch * seq * num_kv_heads * head_dim * bytes * num_layers
        # = 2 * 1 * 1024 * 64 * 192 * 2 * 61 = 3.06 GB
        expected_no_mla = 2 * batch_size * seq_len * 64 * 192 * bytes_per_elem * 61
        
        # MLA should give much smaller KV cache
        assert kv_cache_size == expected_mla
        assert kv_cache_size < expected_no_mla / 10  # At least 10x smaller


class TestKimiK25InferenceCalculations:
    """Test Kimi K2.5 model works with inference performance calculations."""
    
    @pytest.fixture
    def h100_gpu(self):
        """H100 SXM GPU configuration."""
        return SystemConstraints.from_gpu_spec("H100-80GB")
    
    def test_kimi_k25_prefill(self):
        """Kimi K2.5 should complete prefill calculation without errors."""
        perf = InferencePerformance(KIMI_K25)
        result = perf.calculate_prefill_resources(
            batch_size=1,
            sequence_length=2048,
            time_to_first_token=0.5,
            num_gpus=8,  # Large model needs multiple GPUs
            parallelism_config=ParallelismConfig(
                parallelism_type=ParallelismType.TENSOR_PARALLEL,
                tensor_parallel_size=8
            )
        )
        
        assert result is not None
        assert result.time_to_first_token > 0
        assert result.compute_per_gpu > 0
    
    def test_kimi_k25_decode(self, h100_gpu):
        """Kimi K2.5 should complete decode calculation without errors."""
        perf = InferencePerformance(KIMI_K25)
        parallelism = ParallelismConfig(
            parallelism_type=ParallelismType.TENSOR_PARALLEL,
            tensor_parallel_size=8
        )
        result = perf.calculate_decode_performance(
            system_constraints=h100_gpu,
            batch_size=1,
            prefill_length=2048,
            output_length=256,
            parallelism_config=parallelism
        )
        
        assert isinstance(result, DecodePerformance)
        assert result.total_decode_time > 0
        assert result.tokens_per_second_per_user > 0
        assert result.total_throughput > 0
    
    def test_kimi_k25_with_tensor_parallel(self):
        """Kimi K2.5 should work with tensor parallelism."""
        parallelism = ParallelismConfig(
            parallelism_type=ParallelismType.TENSOR_PARALLEL,
            tensor_parallel_size=8
        )
        perf = InferencePerformance(KIMI_K25)
        result = perf.calculate_prefill_resources(
            batch_size=1,
            sequence_length=2048,
            time_to_first_token=0.5,
            num_gpus=8,
            parallelism_config=parallelism
        )
        
        assert result is not None
        assert result.time_to_first_token > 0
    
    def test_kimi_k25_with_pipeline_parallel(self):
        """Kimi K2.5 should work with pipeline parallelism."""
        parallelism = ParallelismConfig(
            parallelism_type=ParallelismType.PIPELINE_PARALLEL,
            pipeline_parallel_size=4
        )
        perf = InferencePerformance(KIMI_K25)
        result = perf.calculate_prefill_resources(
            batch_size=1,
            sequence_length=2048,
            time_to_first_token=0.5,
            num_gpus=4,
            parallelism_config=parallelism
        )
        
        assert result is not None
        assert result.time_to_first_token > 0
    
    def test_kimi_k25_long_context(self):
        """Kimi K2.5 supports 256K context - test with longer sequences."""
        perf = InferencePerformance(KIMI_K25)
        parallelism = ParallelismConfig(
            parallelism_type=ParallelismType.TENSOR_PARALLEL,
            tensor_parallel_size=8
        )
        result = perf.calculate_prefill_resources(
            batch_size=1,
            sequence_length=32768,  # 32K input
            time_to_first_token=5.0,
            num_gpus=8,
            parallelism_config=parallelism
        )
        
        assert result is not None
        assert result.time_to_first_token > 0


class TestKimiK25ComparisonWithDeepSeek:
    """Compare Kimi K2.5 with DeepSeek models."""
    
    def test_kimi_k25_larger_than_deepseek_v3(self):
        """Kimi K2.5 has more experts than DeepSeek V3."""
        deepseek = get_model("deepseek-v3")
        
        assert KIMI_K25.moe_config.num_experts > deepseek.moe_config.num_experts
        # 384 > 256
    
    def test_kimi_k25_same_active_experts(self):
        """Both use top-8 expert routing."""
        deepseek = get_model("deepseek-v3")
        
        assert KIMI_K25.moe_config.num_experts_per_token == deepseek.moe_config.num_experts_per_token
        # Both use 8
    
    def test_kimi_k25_uses_mla(self):
        """Kimi K2.5 uses MLA like DeepSeek 3.2."""
        deepseek_32 = get_model("deepseek-3.2")
        
        assert KIMI_K25.attention_config.use_mla == True
        assert deepseek_32.attention_config.use_mla == True
    
    def test_kimi_k25_longer_context(self):
        """Kimi K2.5 has longer context than DeepSeek V3."""
        deepseek = get_model("deepseek-v3")
        
        # 256K > 160K
        assert KIMI_K25.max_sequence_length > deepseek.max_sequence_length
    
    def test_kimi_k25_mla_same_kv_rank_as_deepseek_32(self):
        """Kimi K2.5 and DeepSeek 3.2 use the same KV compression rank."""
        deepseek_32 = get_model("deepseek-3.2")
        
        # Both Kimi K2.5 and DeepSeek 3.2 use kv_lora_rank=512
        assert KIMI_K25.attention_config.mla_kv_lora_rank == deepseek_32.attention_config.mla_kv_lora_rank
        assert KIMI_K25.attention_config.mla_kv_lora_rank == 512
