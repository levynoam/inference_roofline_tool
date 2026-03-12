"""
Tests for performance modeling of sliding window attention.

These tests verify that:
1. Prefill FLOPs correctly account for sliding window attention
2. Decode FLOPs correctly account for sliding window attention  
3. Full attention layers use L*L complexity, sliding uses L*min(L, window)
4. Hybrid sliding/full models have correct mixed calculations
5. GPT-OSS models with alternating patterns are handled correctly
"""

import pytest
from llm_architecture import (
    LLMArchitecture, AttentionConfig, FFNConfig, MoEConfig,
    ArchitectureType, AttentionType, ActivationType,
    NormalizationType, PositionEncodingType, LayerAttentionType
)
from llm_configs import GPT_OSS_120B, GPT_OSS_20B
from inference_performance import (
    InferencePerformance, SystemConstraints, ParallelismConfig, ParallelismType
)


class TestSlidingWindowPrefillCompute:
    """Tests for prefill compute with sliding window attention"""
    
    @pytest.fixture
    def parallel_config(self):
        """Single GPU parallelism config"""
        return ParallelismConfig(
            parallelism_type=ParallelismType.TENSOR_PARALLEL,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
        )
    
    @pytest.fixture
    def full_attention_model(self):
        """Model with all full attention layers"""
        return LLMArchitecture(
            model_name="test-full-attn",
            model_family="Test",
            version="1.0",
            architecture_type=ArchitectureType.DECODER_ONLY,
            num_layers=4,
            hidden_dim=512,
            vocab_size=32000,
            max_sequence_length=4096,
            attention_config=AttentionConfig(
                num_attention_heads=8,
                num_key_value_heads=8,
                attention_type=AttentionType.MULTI_HEAD,
                head_dim=64,
            ),
            ffn_config=FFNConfig(
                intermediate_size=2048,
                activation=ActivationType.SILU,
            ),
            total_parameters=100_000_000,
            active_parameters=100_000_000,
        )
    
    @pytest.fixture
    def sliding_attention_model(self):
        """Model with all sliding window attention layers"""
        return LLMArchitecture(
            model_name="test-sliding-attn",
            model_family="Test",
            version="1.0",
            architecture_type=ArchitectureType.DECODER_ONLY,
            num_layers=4,
            hidden_dim=512,
            vocab_size=32000,
            max_sequence_length=4096,
            attention_config=AttentionConfig(
                num_attention_heads=8,
                num_key_value_heads=8,
                attention_type=AttentionType.SLIDING_WINDOW,
                head_dim=64,
                sliding_window_size=256,
            ),
            ffn_config=FFNConfig(
                intermediate_size=2048,
                activation=ActivationType.SILU,
            ),
            total_parameters=100_000_000,
            active_parameters=100_000_000,
        )
    
    @pytest.fixture
    def hybrid_attention_model(self):
        """Model with alternating sliding/full attention"""
        return LLMArchitecture(
            model_name="test-hybrid-attn",
            model_family="Test",
            version="1.0",
            architecture_type=ArchitectureType.DECODER_ONLY,
            num_layers=4,
            hidden_dim=512,
            vocab_size=32000,
            max_sequence_length=4096,
            attention_config=AttentionConfig(
                num_attention_heads=8,
                num_key_value_heads=8,
                attention_type=AttentionType.GROUPED_QUERY,
                head_dim=64,
                sliding_window_size=256,
            ),
            ffn_config=FFNConfig(
                intermediate_size=2048,
                activation=ActivationType.SILU,
            ),
            layer_types=[
                LayerAttentionType.SLIDING_ATTENTION,
                LayerAttentionType.FULL_ATTENTION,
                LayerAttentionType.SLIDING_ATTENTION,
                LayerAttentionType.FULL_ATTENTION,
            ],
            total_parameters=100_000_000,
            active_parameters=100_000_000,
        )
    
    def test_full_attention_uses_quadratic(self, full_attention_model, parallel_config):
        """Test full attention uses L*L for attention computation"""
        perf = InferencePerformance(full_attention_model)
        
        L = 1024
        breakdown = perf.calculate_prefill_compute_breakdown(
            batch_size=1, sequence_length=L, parallelism_config=parallel_config
        )
        
        # Attention FLOPs should include L*L terms for Q@K^T and attn@V
        # Each layer: 2 * B * heads * L * L * head_dim (for Q@K^T)
        #           + 2 * B * heads * L * L * head_dim (for attn@V)
        # Plus linear projections
        
        assert breakdown['attention'] > 0
        
        # Run again with 2x sequence length
        breakdown_2x = perf.calculate_prefill_compute_breakdown(
            batch_size=1, sequence_length=2*L, parallelism_config=parallel_config
        )
        
        # For full attention, attention FLOPs scale quadratically with L
        # (ignoring linear projection terms for this comparison)
        # At 2x seq len, should be ~4x more attention FLOPs
        ratio = breakdown_2x['attention'] / breakdown['attention']
        # Should be close to 4 for pure quadratic scaling
        # Allow some tolerance for linear terms
        assert ratio >= 3.0  # At least 3x (accounting for linear terms)
    
    def test_sliding_attention_uses_linear(self, sliding_attention_model, parallel_config):
        """Test sliding attention uses L*window for attention (effectively linear for L > window)"""
        perf = InferencePerformance(sliding_attention_model)
        window_size = 256
        
        L = 1024  # 4x window size
        breakdown = perf.calculate_prefill_compute_breakdown(
            batch_size=1, sequence_length=L, parallelism_config=parallel_config
        )
        
        assert breakdown['attention'] > 0
        
        # Run with 2x sequence length (still > window)
        breakdown_2x = perf.calculate_prefill_compute_breakdown(
            batch_size=1, sequence_length=2*L, parallelism_config=parallel_config
        )
        
        # For sliding attention with L >> window, attention FLOPs scale ~linearly
        # Because effective attention length is capped at window_size
        ratio = breakdown_2x['attention'] / breakdown['attention']
        # Should be closer to 2 (linear scaling) than 4 (quadratic)
        assert ratio < 3.0  # Less than 3x = not quadratic
        assert ratio > 1.5  # At least 1.5x = grows with L
    
    def test_sliding_less_flops_than_full(self, full_attention_model, sliding_attention_model, parallel_config):
        """Test sliding attention has fewer FLOPs than full attention for long sequences"""
        perf_full = InferencePerformance(full_attention_model)
        perf_sliding = InferencePerformance(sliding_attention_model)
        
        # For sequence >> window_size, sliding should have fewer attention FLOPs
        L = 2048  # Much larger than window_size=256
        
        full_breakdown = perf_full.calculate_prefill_compute_breakdown(
            batch_size=1, sequence_length=L, parallelism_config=parallel_config
        )
        sliding_breakdown = perf_sliding.calculate_prefill_compute_breakdown(
            batch_size=1, sequence_length=L, parallelism_config=parallel_config
        )
        
        # Sliding attention should have significantly fewer FLOPs
        assert sliding_breakdown['attention'] < full_breakdown['attention']
        
        # Ratio should be approximately window_size / L for attention matrix ops
        # (but linear projections are same, so ratio won't be exactly that)
        assert sliding_breakdown['attention'] < 0.5 * full_breakdown['attention']
    
    def test_sliding_equals_full_for_short_sequences(self, full_attention_model, sliding_attention_model, parallel_config):
        """Test sliding attention equals full attention when L <= window_size"""
        perf_full = InferencePerformance(full_attention_model)
        perf_sliding = InferencePerformance(sliding_attention_model)
        
        # For sequence <= window_size, sliding = full
        L = 128  # Less than window_size=256
        
        full_breakdown = perf_full.calculate_prefill_compute_breakdown(
            batch_size=1, sequence_length=L, parallelism_config=parallel_config
        )
        sliding_breakdown = perf_sliding.calculate_prefill_compute_breakdown(
            batch_size=1, sequence_length=L, parallelism_config=parallel_config
        )
        
        # Should be equal when L <= window_size
        assert sliding_breakdown['attention'] == full_breakdown['attention']
    
    def test_hybrid_attention_between(self, full_attention_model, sliding_attention_model, hybrid_attention_model, parallel_config):
        """Test hybrid model FLOPs is between full and sliding"""
        perf_full = InferencePerformance(full_attention_model)
        perf_sliding = InferencePerformance(sliding_attention_model)
        perf_hybrid = InferencePerformance(hybrid_attention_model)
        
        L = 2048
        
        full_breakdown = perf_full.calculate_prefill_compute_breakdown(
            batch_size=1, sequence_length=L, parallelism_config=parallel_config
        )
        sliding_breakdown = perf_sliding.calculate_prefill_compute_breakdown(
            batch_size=1, sequence_length=L, parallelism_config=parallel_config
        )
        hybrid_breakdown = perf_hybrid.calculate_prefill_compute_breakdown(
            batch_size=1, sequence_length=L, parallelism_config=parallel_config
        )
        
        # Hybrid (2 sliding + 2 full) should be between all-sliding and all-full
        assert sliding_breakdown['attention'] < hybrid_breakdown['attention']
        assert hybrid_breakdown['attention'] < full_breakdown['attention']


class TestSlidingWindowDecodeCompute:
    """Tests for decode compute with sliding window attention"""
    
    @pytest.fixture
    def parallel_config(self):
        return ParallelismConfig(
            parallelism_type=ParallelismType.TENSOR_PARALLEL,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
        )
    
    @pytest.fixture
    def sliding_model(self):
        """Model with sliding window attention"""
        return LLMArchitecture(
            model_name="test-sliding",
            model_family="Test",
            version="1.0",
            architecture_type=ArchitectureType.DECODER_ONLY,
            num_layers=4,
            hidden_dim=512,
            vocab_size=32000,
            max_sequence_length=4096,
            attention_config=AttentionConfig(
                num_attention_heads=8,
                num_key_value_heads=8,
                attention_type=AttentionType.SLIDING_WINDOW,
                head_dim=64,
                sliding_window_size=256,
            ),
            ffn_config=FFNConfig(intermediate_size=2048),
            total_parameters=100_000_000,
        )
    
    @pytest.fixture
    def full_model(self):
        """Model with full attention"""
        return LLMArchitecture(
            model_name="test-full",
            model_family="Test",
            version="1.0",
            architecture_type=ArchitectureType.DECODER_ONLY,
            num_layers=4,
            hidden_dim=512,
            vocab_size=32000,
            max_sequence_length=4096,
            attention_config=AttentionConfig(
                num_attention_heads=8,
                num_key_value_heads=8,
                attention_type=AttentionType.MULTI_HEAD,
                head_dim=64,
            ),
            ffn_config=FFNConfig(intermediate_size=2048),
            total_parameters=100_000_000,
        )
    
    def test_decode_sliding_capped_at_window(self, sliding_model, parallel_config):
        """Test decode attention is capped at window size"""
        perf = InferencePerformance(sliding_model)
        window_size = 256
        
        # Context larger than window
        context_1024 = 1024
        breakdown_1024 = perf._calculate_decode_step_compute_breakdown(
            batch_size=1, context_length=context_1024, parallelism_config=parallel_config
        )
        
        # Even larger context
        context_4096 = 4096
        breakdown_4096 = perf._calculate_decode_step_compute_breakdown(
            batch_size=1, context_length=context_4096, parallelism_config=parallel_config
        )
        
        # Attention FLOPs should be similar when both contexts > window
        # (only linear projection terms scale with context, not attention matrix)
        # Allow some tolerance for those linear terms
        assert breakdown_1024['attention'] == pytest.approx(breakdown_4096['attention'], rel=0.1)
    
    def test_decode_sliding_vs_full(self, sliding_model, full_model, parallel_config):
        """Test sliding decode has fewer attention FLOPs than full"""
        perf_sliding = InferencePerformance(sliding_model)
        perf_full = InferencePerformance(full_model)
        
        context = 2048
        
        sliding_breakdown = perf_sliding._calculate_decode_step_compute_breakdown(
            batch_size=1, context_length=context, parallelism_config=parallel_config
        )
        full_breakdown = perf_full._calculate_decode_step_compute_breakdown(
            batch_size=1, context_length=context, parallelism_config=parallel_config
        )
        
        # Sliding should have fewer attention FLOPs
        assert sliding_breakdown['attention'] < full_breakdown['attention']


class TestGPTOSSPerformance:
    """Tests for GPT-OSS models with hybrid sliding/full attention"""
    
    @pytest.fixture
    def parallel_config(self):
        return ParallelismConfig(
            parallelism_type=ParallelismType.TENSOR_PARALLEL,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
        )
    
    def test_gpt_oss_120b_has_mixed_layers(self):
        """Verify GPT-OSS-120B has alternating layer types"""
        assert GPT_OSS_120B.layer_types is not None
        assert len(GPT_OSS_120B.layer_types) == 36
        assert GPT_OSS_120B.get_num_full_attention_layers() == 18
        assert GPT_OSS_120B.get_num_sliding_attention_layers() == 18
    
    def test_gpt_oss_120b_prefill_computes(self, parallel_config):
        """Test GPT-OSS-120B prefill compute breakdown works"""
        perf = InferencePerformance(GPT_OSS_120B)
        
        breakdown = perf.calculate_prefill_compute_breakdown(
            batch_size=1, sequence_length=1024, parallelism_config=parallel_config
        )
        
        assert breakdown['attention'] > 0
        assert breakdown['ffn'] > 0
        assert breakdown['total'] > 0
        assert 'mamba' in breakdown  # Should have mamba key (even if 0)
        assert breakdown['mamba'] == 0  # GPT-OSS has no Mamba layers
    
    def test_gpt_oss_120b_decode_computes(self, parallel_config):
        """Test GPT-OSS-120B decode compute breakdown works"""
        perf = InferencePerformance(GPT_OSS_120B)
        
        breakdown = perf._calculate_decode_step_compute_breakdown(
            batch_size=1, context_length=4096, parallelism_config=parallel_config
        )
        
        assert breakdown['attention'] > 0
        assert breakdown['ffn'] > 0
        assert breakdown['total'] > 0
    
    def test_gpt_oss_20b_computes(self, parallel_config):
        """Test GPT-OSS-20B model works with hybrid attention"""
        perf = InferencePerformance(GPT_OSS_20B)
        
        breakdown = perf.calculate_prefill_compute_breakdown(
            batch_size=1, sequence_length=1024, parallelism_config=parallel_config
        )
        
        assert breakdown['total'] > 0


class TestNoRegressionFullAttentionModels:
    """Ensure models without sliding window still work correctly"""
    
    @pytest.fixture
    def parallel_config(self):
        return ParallelismConfig(
            parallelism_type=ParallelismType.TENSOR_PARALLEL,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
        )
    
    def test_llama3_still_works(self, parallel_config):
        """Test Llama-3-8B (no sliding window) still computes correctly"""
        from llm_configs import LLAMA_3_8B
        
        assert LLAMA_3_8B.layer_types is None  # No per-layer types
        
        perf = InferencePerformance(LLAMA_3_8B)
        breakdown = perf.calculate_prefill_compute_breakdown(
            batch_size=1, sequence_length=1024, parallelism_config=parallel_config
        )
        
        assert breakdown['attention'] > 0
        assert breakdown['ffn'] > 0
        assert breakdown['mamba'] == 0  # No Mamba layers
        assert breakdown['total'] > breakdown['attention']  # Total includes FFN
    
    def test_deepseek_v3_still_works(self, parallel_config):
        """Test DeepSeek-V3 (MoE, no sliding) still computes correctly"""
        from llm_configs import DEEPSEEK_V3
        
        assert DEEPSEEK_V3.is_moe is True
        
        perf = InferencePerformance(DEEPSEEK_V3)
        breakdown = perf.calculate_prefill_compute_breakdown(
            batch_size=1, sequence_length=1024, parallelism_config=parallel_config
        )
        
        assert breakdown['attention'] > 0
        assert breakdown['ffn'] > 0
        assert breakdown['total'] > 0
