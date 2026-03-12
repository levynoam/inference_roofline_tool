"""
Tests for Qwen3-480B model configuration.

Qwen3-480B is Alibaba's large MoE model.
Key features:
- 160 experts with top-8 routing
- GQA with 12:1 ratio (96 heads, 8 KV heads)
- 256K context
- 62 layers, hidden_dim=6144
- No shared expert
"""

import pytest
from llm_configs import ALL_MODELS, get_model, QWEN3_480B
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


class TestQwen3480BModelConfiguration:
    """Test Qwen3-480B model configuration is correct."""
    
    def test_qwen3_480b_in_all_models(self):
        """Qwen3-480B should be accessible from ALL_MODELS."""
        assert "qwen3-480b" in ALL_MODELS
        assert ALL_MODELS["qwen3-480b"] is QWEN3_480B
    
    def test_qwen3_480b_get_model(self):
        """get_model() should return Qwen3-480B."""
        model = get_model("qwen3-480b")
        assert model is QWEN3_480B
        assert model.model_name == "Qwen3-480B"
    
    def test_qwen3_480b_basic_architecture(self):
        """Verify basic architecture parameters."""
        assert QWEN3_480B.model_name == "Qwen3-480B"
        assert QWEN3_480B.model_family == "Qwen"
        assert QWEN3_480B.version == "3.0"
        assert QWEN3_480B.architecture_type == ArchitectureType.DECODER_ONLY
    
    def test_qwen3_480b_dimensions(self):
        """Verify dimensions match config.json."""
        assert QWEN3_480B.hidden_dim == 6144
        assert QWEN3_480B.num_layers == 62
        assert QWEN3_480B.vocab_size == 151936
        assert QWEN3_480B.max_sequence_length == 262144  # 256K context
    
    def test_qwen3_480b_attention_config(self):
        """Verify attention configuration with GQA."""
        attn = QWEN3_480B.attention_config
        assert attn.num_attention_heads == 96
        assert attn.num_key_value_heads == 8  # GQA with 12:1 ratio
        assert attn.attention_type == AttentionType.GROUPED_QUERY
        assert attn.head_dim == 128
    
    def test_qwen3_480b_gqa_ratio(self):
        """Verify 12:1 GQA ratio."""
        attn = QWEN3_480B.attention_config
        gqa_ratio = attn.num_attention_heads / attn.num_key_value_heads
        assert gqa_ratio == 12.0
    
    def test_qwen3_480b_moe_config(self):
        """Verify MoE configuration."""
        assert QWEN3_480B.is_moe == True
        moe = QWEN3_480B.moe_config
        assert moe is not None
        assert moe.num_experts == 160
        assert moe.num_experts_per_token == 8
        assert moe.shared_expert == False  # No shared expert
        assert moe.router_type == "top_k"
    
    def test_qwen3_480b_ffn_config(self):
        """Verify FFN configuration."""
        ffn = QWEN3_480B.ffn_config
        assert ffn.intermediate_size == 2560  # moe_intermediate_size per expert
        assert ffn.activation == ActivationType.SILU
    
    def test_qwen3_480b_normalization(self):
        """Verify normalization type."""
        assert QWEN3_480B.normalization_type == NormalizationType.RMS_NORM
    
    def test_qwen3_480b_position_encoding(self):
        """Verify position encoding uses RoPE with high theta."""
        assert QWEN3_480B.position_encoding == PositionEncodingType.ROTARY
        assert QWEN3_480B.rope_theta == 10000000.0  # 10M base
    
    def test_qwen3_480b_dtype(self):
        """Verify default dtype is bfloat16."""
        assert QWEN3_480B.dtype == "bfloat16"
    
    def test_qwen3_480b_parameters(self):
        """Verify parameter counts."""
        assert QWEN3_480B.total_parameters == 480_000_000_000  # 480B
        assert QWEN3_480B.active_parameters == 35_000_000_000  # ~35B active


class TestQwen3480BInferenceCalculations:
    """Test Qwen3-480B model works with inference performance calculations."""
    
    @pytest.fixture
    def h100_gpu(self):
        """H100 SXM GPU configuration."""
        return SystemConstraints.from_gpu_spec("H100-80GB")
    
    def test_qwen3_480b_prefill(self):
        """Qwen3-480B should complete prefill calculation without errors."""
        perf = InferencePerformance(QWEN3_480B)
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
    
    def test_qwen3_480b_decode(self, h100_gpu):
        """Qwen3-480B should complete decode calculation without errors."""
        perf = InferencePerformance(QWEN3_480B)
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
    
    def test_qwen3_480b_with_tensor_parallel(self):
        """Qwen3-480B should work with tensor parallelism."""
        parallelism = ParallelismConfig(
            parallelism_type=ParallelismType.TENSOR_PARALLEL,
            tensor_parallel_size=8
        )
        perf = InferencePerformance(QWEN3_480B)
        result = perf.calculate_prefill_resources(
            batch_size=1,
            sequence_length=2048,
            time_to_first_token=0.5,
            num_gpus=8,
            parallelism_config=parallelism
        )
        
        assert result is not None
        assert result.time_to_first_token > 0
    
    def test_qwen3_480b_with_pipeline_parallel(self):
        """Qwen3-480B should work with pipeline parallelism."""
        parallelism = ParallelismConfig(
            parallelism_type=ParallelismType.PIPELINE_PARALLEL,
            pipeline_parallel_size=4
        )
        perf = InferencePerformance(QWEN3_480B)
        result = perf.calculate_prefill_resources(
            batch_size=1,
            sequence_length=2048,
            time_to_first_token=0.5,
            num_gpus=4,
            parallelism_config=parallelism
        )
        
        assert result is not None
        assert result.time_to_first_token > 0


class TestQwen3480BComparisonWithOtherModels:
    """Compare Qwen3-480B with other MoE models."""
    
    def test_qwen3_480b_more_experts_than_gpt_oss_120b(self):
        """Qwen3-480B has more experts than GPT-OSS-120B."""
        gpt_oss = get_model("gpt-oss-120b")
        
        assert QWEN3_480B.moe_config.num_experts > gpt_oss.moe_config.num_experts
        # 160 > 128
    
    def test_qwen3_480b_same_active_experts_as_deepseek(self):
        """Qwen3-480B uses top-8 like DeepSeek V3."""
        deepseek = get_model("deepseek-v3")
        
        assert QWEN3_480B.moe_config.num_experts_per_token == deepseek.moe_config.num_experts_per_token
        # Both use 8
    
    def test_qwen3_480b_uses_gqa(self):
        """Qwen3-480B uses GQA like Llama models."""
        llama3 = get_model("llama-3-8b")
        
        assert QWEN3_480B.attention_config.attention_type == AttentionType.GROUPED_QUERY
        assert llama3.attention_config.attention_type == AttentionType.GROUPED_QUERY
    
    def test_qwen3_480b_aggressive_gqa(self):
        """Qwen3-480B has 12:1 GQA ratio (more aggressive than Llama's 8:1)."""
        llama3_70b = get_model("llama-3-70b")
        
        qwen_ratio = QWEN3_480B.attention_config.num_attention_heads / QWEN3_480B.attention_config.num_key_value_heads
        llama_ratio = llama3_70b.attention_config.num_attention_heads / llama3_70b.attention_config.num_key_value_heads
        
        # Qwen3 uses 12:1, Llama-3-70B uses 8:1
        assert qwen_ratio > llama_ratio
    
    def test_qwen3_480b_high_rope_theta(self):
        """Qwen3-480B uses very high rope_theta for long context."""
        llama4 = get_model("llama-4-scout")
        
        # Qwen3: 10M, Llama4: 500K
        assert QWEN3_480B.rope_theta > llama4.rope_theta
