"""
Tests for LFM2-3B (Liquid Foundation Model 2) model configuration.

Tests verify:
1. Model loads correctly from ALL_MODELS
2. Model has correct architecture parameters
3. Model works with inference performance calculations
4. Model integrates with various parallelism configurations
"""

import pytest
from llm_configs import ALL_MODELS, get_model, LFM2_3B
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


class TestLFM2ModelConfiguration:
    """Test LFM2 model configuration is correct."""
    
    def test_lfm2_in_all_models(self):
        """LFM2-3B should be accessible from ALL_MODELS."""
        assert "lfm2-3b" in ALL_MODELS
        assert ALL_MODELS["lfm2-3b"] is LFM2_3B
    
    def test_lfm2_get_model(self):
        """get_model() should return LFM2-3B."""
        model = get_model("lfm2-3b")
        assert model is LFM2_3B
        assert model.model_name == "LFM2-3B"
    
    def test_lfm2_basic_architecture(self):
        """Verify basic architecture parameters match config.json."""
        assert LFM2_3B.model_name == "LFM2-3B"
        assert LFM2_3B.model_family == "Liquid"
        assert LFM2_3B.version == "2.0"
        assert LFM2_3B.architecture_type == ArchitectureType.DECODER_ONLY
    
    def test_lfm2_dimensions(self):
        """Verify dimensions match config.json."""
        # From config.json
        assert LFM2_3B.hidden_dim == 2048
        assert LFM2_3B.num_layers == 16  # 10 conv + 6 attention
        assert LFM2_3B.vocab_size == 65536
        assert LFM2_3B.max_sequence_length == 128000  # 128K context
    
    def test_lfm2_attention_config(self):
        """Verify attention configuration matches config.json."""
        attn = LFM2_3B.attention_config
        assert attn.num_attention_heads == 32
        assert attn.num_key_value_heads == 8  # GQA
        assert attn.attention_type == AttentionType.GROUPED_QUERY
        assert attn.head_dim == 64  # 2048 / 32 = 64
    
    def test_lfm2_ffn_config(self):
        """Verify FFN configuration matches config.json."""
        ffn = LFM2_3B.ffn_config
        assert ffn.intermediate_size == 12288
        assert ffn.activation == ActivationType.SWIGLU
        assert ffn.use_gating == True
    
    def test_lfm2_normalization(self):
        """Verify normalization type."""
        assert LFM2_3B.normalization_type == NormalizationType.RMS_NORM
    
    def test_lfm2_position_encoding(self):
        """Verify position encoding uses RoPE with correct theta."""
        assert LFM2_3B.position_encoding == PositionEncodingType.ROTARY
        assert LFM2_3B.rope_theta == 1000000.0  # 1M base
    
    def test_lfm2_dtype(self):
        """Verify default dtype is bfloat16."""
        assert LFM2_3B.dtype == "bfloat16"
    
    def test_lfm2_parameters(self):
        """Verify parameter count is approximately 3B."""
        assert LFM2_3B.total_parameters == 3_000_000_000


class TestLFM2InferenceCalculations:
    """Test LFM2 model works with inference performance calculations."""
    
    @pytest.fixture
    def h100_gpu(self):
        """H100 SXM GPU configuration."""
        return SystemConstraints.from_gpu_spec("H100-80GB")
    
    def test_lfm2_prefill(self, h100_gpu):
        """LFM2 should complete prefill calculation without errors."""
        perf = InferencePerformance(LFM2_3B)
        result = perf.calculate_prefill_resources(
            batch_size=1,
            sequence_length=1024,
            time_to_first_token=0.1,
            num_gpus=1,
            parallelism_config=ParallelismConfig()
        )
        
        assert result is not None
        assert result.time_to_first_token > 0
        assert result.compute_per_gpu > 0
    
    def test_lfm2_decode(self, h100_gpu):
        """LFM2 should complete decode calculation without errors."""
        perf = InferencePerformance(LFM2_3B)
        result = perf.calculate_decode_performance(
            system_constraints=h100_gpu,
            batch_size=1,
            prefill_length=1024,
            output_length=256
        )
        
        assert isinstance(result, DecodePerformance)
        assert result.total_decode_time > 0
        assert result.tokens_per_second_per_user > 0
        assert result.total_throughput > 0
    
    def test_lfm2_with_tensor_parallel(self):
        """LFM2 should work with tensor parallelism."""
        parallelism = ParallelismConfig(
            parallelism_type=ParallelismType.TENSOR_PARALLEL,
            tensor_parallel_size=2
        )
        perf = InferencePerformance(LFM2_3B)
        result = perf.calculate_prefill_resources(
            batch_size=1,
            sequence_length=1024,
            time_to_first_token=0.1,
            num_gpus=2,
            parallelism_config=parallelism
        )
        
        assert result is not None
        assert result.time_to_first_token > 0
    
    def test_lfm2_with_pipeline_parallel(self):
        """LFM2 should work with pipeline parallelism."""
        parallelism = ParallelismConfig(
            parallelism_type=ParallelismType.PIPELINE_PARALLEL,
            pipeline_parallel_size=2
        )
        perf = InferencePerformance(LFM2_3B)
        result = perf.calculate_prefill_resources(
            batch_size=1,
            sequence_length=1024,
            time_to_first_token=0.1,
            num_gpus=2,
            parallelism_config=parallelism
        )
        
        assert result is not None
        assert result.time_to_first_token > 0
    
    def test_lfm2_batch_throughput(self, h100_gpu):
        """Verify batching improves throughput."""
        perf = InferencePerformance(LFM2_3B)
        
        result_batch1 = perf.calculate_decode_performance(
            system_constraints=h100_gpu,
            batch_size=1,
            prefill_length=512,
            output_length=128
        )
        
        result_batch8 = perf.calculate_decode_performance(
            system_constraints=h100_gpu,
            batch_size=8,
            prefill_length=512,
            output_length=128
        )
        
        # Larger batch should have higher total throughput (tokens/s)
        assert result_batch8.total_throughput > result_batch1.total_throughput
    
    def test_lfm2_long_context(self):
        """LFM2 supports 128K context - test with longer sequences."""
        perf = InferencePerformance(LFM2_3B)
        result = perf.calculate_prefill_resources(
            batch_size=1,
            sequence_length=8192,  # 8K input
            time_to_first_token=0.5,
            num_gpus=1,
            parallelism_config=ParallelismConfig()
        )
        
        assert result is not None
        assert result.time_to_first_token > 0


class TestLFM2ComparisonWithOtherModels:
    """Compare LFM2 with similar-sized models."""
    
    @pytest.fixture
    def h100_gpu(self):
        """H100 SXM GPU configuration."""
        return SystemConstraints.from_gpu_spec("H100-80GB")
    
    def test_lfm2_smaller_than_llama3_8b(self):
        """LFM2-3B should have fewer parameters than Llama-3-8B."""
        llama3 = get_model("llama-3-8b")
        assert LFM2_3B.total_parameters < llama3.total_parameters
    
    def test_lfm2_uses_gqa_like_llama3(self):
        """LFM2 uses GQA similar to Llama-3 models."""
        assert LFM2_3B.attention_config.attention_type == AttentionType.GROUPED_QUERY
        
        llama3 = get_model("llama-3-8b")
        assert llama3.attention_config.attention_type == AttentionType.GROUPED_QUERY
    
    def test_lfm2_gqa_ratio(self):
        """LFM2 has 4:1 GQA ratio (32 heads / 8 KV heads)."""
        attn = LFM2_3B.attention_config
        gqa_ratio = attn.num_attention_heads / attn.num_key_value_heads
        assert gqa_ratio == 4.0
