"""
Tests for Hunyuan A13B model configuration.

Hunyuan A13B is Tencent's MoE model with:
- 64 routed experts + 1 shared expert per layer, top-8 routing
- GQA with 4:1 ratio (32 heads, 8 KV heads)
- 32K context
- 32 layers, hidden_dim=4096
- ~80B total parameters, ~13B active (hence "A13B" name)
"""

import pytest
from llm_configs import ALL_MODELS, get_model, HUNYUAN_A13B
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


class TestHunyuanA13BModelConfiguration:
    """Test Hunyuan A13B model configuration is correct."""
    
    def test_hunyuan_a13b_in_all_models(self):
        """Hunyuan A13B should be accessible from ALL_MODELS."""
        assert "hunyuan-a13b" in ALL_MODELS
        assert ALL_MODELS["hunyuan-a13b"] is HUNYUAN_A13B
    
    def test_hunyuan_a13b_get_model(self):
        """get_model() should return Hunyuan A13B."""
        model = get_model("hunyuan-a13b")
        assert model is HUNYUAN_A13B
        assert model.model_name == "Hunyuan-A13B"
    
    def test_hunyuan_a13b_basic_architecture(self):
        """Verify basic architecture parameters."""
        assert HUNYUAN_A13B.model_name == "Hunyuan-A13B"
        assert HUNYUAN_A13B.model_family == "Hunyuan"
        assert HUNYUAN_A13B.version == "1.0"
        assert HUNYUAN_A13B.architecture_type == ArchitectureType.DECODER_ONLY
    
    def test_hunyuan_a13b_dimensions(self):
        """Verify dimensions match config.json."""
        assert HUNYUAN_A13B.hidden_dim == 4096
        assert HUNYUAN_A13B.num_layers == 32
        assert HUNYUAN_A13B.vocab_size == 128167
        assert HUNYUAN_A13B.max_sequence_length == 32768  # 32K context
    
    def test_hunyuan_a13b_attention_config(self):
        """Verify attention configuration with GQA."""
        attn = HUNYUAN_A13B.attention_config
        assert attn.num_attention_heads == 32
        assert attn.num_key_value_heads == 8  # GQA 4:1 ratio
        assert attn.attention_type == AttentionType.GROUPED_QUERY
        assert attn.head_dim == 128
    
    def test_hunyuan_a13b_gqa_ratio(self):
        """Verify GQA ratio is 4:1."""
        attn = HUNYUAN_A13B.attention_config
        ratio = attn.num_attention_heads // attn.num_key_value_heads
        assert ratio == 4
    
    def test_hunyuan_a13b_no_mla(self):
        """Verify MLA is not used."""
        attn = HUNYUAN_A13B.attention_config
        assert attn.use_mla == False
    
    def test_hunyuan_a13b_moe_config(self):
        """Verify MoE configuration."""
        assert HUNYUAN_A13B.is_moe == True
        moe = HUNYUAN_A13B.moe_config
        assert moe is not None
        assert moe.num_experts == 64
        assert moe.num_experts_per_token == 8  # moe_topk
        assert moe.shared_expert == True  # num_shared_expert=1
        assert moe.router_type == "top_k"
    
    def test_hunyuan_a13b_ffn_config(self):
        """Verify FFN configuration."""
        ffn = HUNYUAN_A13B.ffn_config
        assert ffn.intermediate_size == 3072  # moe_intermediate_size
        assert ffn.activation == ActivationType.SILU
        assert ffn.use_gating == False
    
    def test_hunyuan_a13b_normalization(self):
        """Verify normalization type is RMSNorm."""
        assert HUNYUAN_A13B.normalization_type == NormalizationType.RMS_NORM
    
    def test_hunyuan_a13b_position_encoding(self):
        """Verify position encoding uses RoPE."""
        assert HUNYUAN_A13B.position_encoding == PositionEncodingType.ROTARY
        assert HUNYUAN_A13B.rope_theta == 10000.0
    
    def test_hunyuan_a13b_dtype(self):
        """Verify default dtype is bfloat16."""
        assert HUNYUAN_A13B.dtype == "bfloat16"
    
    def test_hunyuan_a13b_tie_embeddings(self):
        """Verify embeddings are tied."""
        assert HUNYUAN_A13B.tie_word_embeddings == True
    
    def test_hunyuan_a13b_parameters(self):
        """Verify parameter counts."""
        # Total parameters ~80B
        assert HUNYUAN_A13B.total_parameters == 80_000_000_000
        # Active parameters ~13B (8 experts + shared + attention)
        assert HUNYUAN_A13B.active_parameters == 13_000_000_000


class TestHunyuanA13BInferenceCalculations:
    """Test Hunyuan A13B model works with inference performance calculations."""
    
    @pytest.fixture
    def h100_gpu(self):
        """H100 SXM GPU configuration."""
        return SystemConstraints.from_gpu_spec("H100-80GB")
    
    def test_hunyuan_a13b_prefill(self):
        """Hunyuan A13B should complete prefill calculation without errors."""
        perf = InferencePerformance(HUNYUAN_A13B)
        result = perf.calculate_prefill_resources(
            batch_size=1,
            sequence_length=2048,
            time_to_first_token=0.2,
            num_gpus=2,
            parallelism_config=ParallelismConfig(
                parallelism_type=ParallelismType.TENSOR_PARALLEL,
                tensor_parallel_size=2
            )
        )
        
        assert result is not None
        assert result.time_to_first_token > 0
        assert result.compute_per_gpu > 0
    
    def test_hunyuan_a13b_decode(self, h100_gpu):
        """Hunyuan A13B should complete decode calculation without errors."""
        perf = InferencePerformance(HUNYUAN_A13B)
        parallelism = ParallelismConfig(
            parallelism_type=ParallelismType.TENSOR_PARALLEL,
            tensor_parallel_size=2
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
    
    def test_hunyuan_a13b_with_tensor_parallel(self):
        """Hunyuan A13B should work with tensor parallelism."""
        parallelism = ParallelismConfig(
            parallelism_type=ParallelismType.TENSOR_PARALLEL,
            tensor_parallel_size=4
        )
        perf = InferencePerformance(HUNYUAN_A13B)
        result = perf.calculate_prefill_resources(
            batch_size=1,
            sequence_length=2048,
            time_to_first_token=0.2,
            num_gpus=4,
            parallelism_config=parallelism
        )
        
        assert result is not None
        assert result.time_to_first_token > 0
    
    def test_hunyuan_a13b_single_gpu(self, h100_gpu):
        """Hunyuan A13B can run on single GPU due to smaller active params."""
        perf = InferencePerformance(HUNYUAN_A13B)
        result = perf.calculate_decode_performance(
            system_constraints=h100_gpu,
            batch_size=1,
            prefill_length=1024,
            output_length=128
        )
        
        assert isinstance(result, DecodePerformance)
        assert result.total_decode_time > 0


class TestHunyuanA13BComparisonWithOtherModels:
    """Compare Hunyuan A13B with other MoE models."""
    
    def test_hunyuan_a13b_fewer_experts_than_deepseek(self):
        """Hunyuan A13B has fewer experts than DeepSeek V3."""
        deepseek = get_model("deepseek-v3")
        
        # 64 < 256
        assert HUNYUAN_A13B.moe_config.num_experts < deepseek.moe_config.num_experts
    
    def test_hunyuan_a13b_same_active_experts_as_deepseek(self):
        """Both use top-8 expert routing."""
        deepseek = get_model("deepseek-v3")
        
        assert HUNYUAN_A13B.moe_config.num_experts_per_token == deepseek.moe_config.num_experts_per_token
        # Both use 8
    
    def test_hunyuan_a13b_more_experts_than_gpt_oss_20b(self):
        """Hunyuan A13B has more experts than GPT-OSS-20B."""
        gpt_oss = get_model("gpt-oss-20b")
        
        # 64 > 32
        assert HUNYUAN_A13B.moe_config.num_experts > gpt_oss.moe_config.num_experts
    
    def test_hunyuan_a13b_uses_gqa(self):
        """Hunyuan A13B uses GQA, unlike DeepSeek V3."""
        deepseek = get_model("deepseek-v3")
        
        assert HUNYUAN_A13B.attention_config.attention_type == AttentionType.GROUPED_QUERY
        assert deepseek.attention_config.attention_type == AttentionType.MULTI_HEAD
    
    def test_hunyuan_a13b_smaller_than_qwen3(self):
        """Hunyuan A13B is smaller than Qwen3-480B."""
        qwen3 = get_model("qwen3-480b")
        
        assert HUNYUAN_A13B.total_parameters < qwen3.total_parameters
        assert HUNYUAN_A13B.active_parameters < qwen3.active_parameters
    
    def test_hunyuan_a13b_similar_active_to_gpt_oss_120b(self):
        """Hunyuan A13B has similar active params to GPT-OSS-120B."""
        gpt_oss = get_model("gpt-oss-120b")
        
        # Hunyuan A13B: 13B active, GPT-OSS-120B: 12B active
        ratio = HUNYUAN_A13B.active_parameters / gpt_oss.active_parameters
        assert 0.8 < ratio < 1.5  # Within 50% of each other
