"""
Tests for performance modeling of Mamba-2 hybrid architectures.

These tests verify that:
1. Mamba layers use Mamba-specific FLOPs instead of attention
2. Hybrid models correctly mix Mamba and attention FLOPs
3. Nemotron-3-30B model has correct compute breakdown
4. Mamba FLOPs scale linearly with sequence length (not quadratic)
5. FFN is still computed for all layers regardless of Mamba/Attention
"""

import pytest
from llm_architecture import (
    LLMArchitecture, AttentionConfig, FFNConfig, MoEConfig,
    ArchitectureType, AttentionType, ActivationType,
    NormalizationType, PositionEncodingType, 
    HybridLayerType, Mamba2Config
)
from llm_configs import NEMOTRON_3_30B
from inference_performance import (
    InferencePerformance, SystemConstraints, ParallelismConfig, ParallelismType
)


class TestMambaHybridPrefillCompute:
    """Tests for prefill compute with Mamba-2 hybrid architecture"""
    
    @pytest.fixture
    def parallel_config(self):
        """Single GPU parallelism config"""
        return ParallelismConfig(
            parallelism_type=ParallelismType.TENSOR_PARALLEL,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
        )
    
    @pytest.fixture
    def pure_attention_model(self):
        """Model with all attention layers (no Mamba)"""
        return LLMArchitecture(
            model_name="test-pure-attention",
            model_family="Test",
            version="1.0",
            architecture_type=ArchitectureType.DECODER_ONLY,
            num_layers=4,
            hidden_dim=1024,
            vocab_size=32000,
            max_sequence_length=4096,
            attention_config=AttentionConfig(
                num_attention_heads=16,
                num_key_value_heads=4,
                attention_type=AttentionType.GROUPED_QUERY,
                head_dim=64,
            ),
            ffn_config=FFNConfig(
                intermediate_size=2048,
                activation=ActivationType.SILU,
            ),
            total_parameters=500_000_000,
            active_parameters=500_000_000,
        )
    
    @pytest.fixture
    def pure_mamba_model(self):
        """Model with all Mamba layers (no attention)"""
        return LLMArchitecture(
            model_name="test-pure-mamba",
            model_family="Test",
            version="1.0",
            architecture_type=ArchitectureType.DECODER_ONLY,
            num_layers=4,
            hidden_dim=1024,
            vocab_size=32000,
            max_sequence_length=4096,
            attention_config=AttentionConfig(
                num_attention_heads=16,
                num_key_value_heads=4,
                attention_type=AttentionType.GROUPED_QUERY,
                head_dim=64,
            ),
            ffn_config=FFNConfig(
                intermediate_size=2048,
                activation=ActivationType.SILU,
            ),
            hybrid_layer_types=[HybridLayerType.MAMBA] * 4,
            mamba_config=Mamba2Config(
                num_heads=16,
                head_dim=64,
                state_size=128,
                chunk_size=128,
            ),
            total_parameters=500_000_000,
            active_parameters=500_000_000,
        )
    
    @pytest.fixture
    def hybrid_model(self):
        """Model with alternating Mamba and Attention layers"""
        return LLMArchitecture(
            model_name="test-hybrid",
            model_family="Test",
            version="1.0",
            architecture_type=ArchitectureType.DECODER_ONLY,
            num_layers=4,
            hidden_dim=1024,
            vocab_size=32000,
            max_sequence_length=4096,
            attention_config=AttentionConfig(
                num_attention_heads=16,
                num_key_value_heads=4,
                attention_type=AttentionType.GROUPED_QUERY,
                head_dim=64,
            ),
            ffn_config=FFNConfig(
                intermediate_size=2048,
                activation=ActivationType.SILU,
            ),
            hybrid_layer_types=[
                HybridLayerType.MAMBA,
                HybridLayerType.ATTENTION,
                HybridLayerType.MAMBA,
                HybridLayerType.ATTENTION,
            ],
            mamba_config=Mamba2Config(
                num_heads=16,
                head_dim=64,
                state_size=128,
                chunk_size=128,
            ),
            total_parameters=500_000_000,
            active_parameters=500_000_000,
        )
    
    def test_pure_attention_no_mamba_flops(self, pure_attention_model, parallel_config):
        """Test pure attention model has no Mamba FLOPs"""
        perf = InferencePerformance(pure_attention_model)
        
        breakdown = perf.calculate_prefill_compute_breakdown(
            batch_size=1, sequence_length=1024, parallelism_config=parallel_config
        )
        
        assert breakdown['attention'] > 0
        assert breakdown['mamba'] == 0
        assert breakdown['ffn'] > 0
    
    def test_pure_mamba_no_attention_flops(self, pure_mamba_model, parallel_config):
        """Test pure Mamba model has no attention FLOPs"""
        perf = InferencePerformance(pure_mamba_model)
        
        breakdown = perf.calculate_prefill_compute_breakdown(
            batch_size=1, sequence_length=1024, parallelism_config=parallel_config
        )
        
        assert breakdown['attention'] == 0
        assert breakdown['mamba'] > 0
        assert breakdown['ffn'] > 0  # FFN still computed for all layers
    
    def test_hybrid_has_both(self, hybrid_model, parallel_config):
        """Test hybrid model has both attention and Mamba FLOPs"""
        perf = InferencePerformance(hybrid_model)
        
        breakdown = perf.calculate_prefill_compute_breakdown(
            batch_size=1, sequence_length=1024, parallelism_config=parallel_config
        )
        
        assert breakdown['attention'] > 0
        assert breakdown['mamba'] > 0
        assert breakdown['ffn'] > 0
        
        # Total should include all components
        expected_total = breakdown['attention'] + breakdown['mamba'] + breakdown['ffn'] + breakdown['other']
        assert breakdown['total'] == pytest.approx(expected_total, rel=0.001)
    
    def test_mamba_scales_linearly(self, pure_mamba_model, parallel_config):
        """Test Mamba FLOPs scale linearly with sequence length"""
        perf = InferencePerformance(pure_mamba_model)
        
        L = 512
        breakdown_1x = perf.calculate_prefill_compute_breakdown(
            batch_size=1, sequence_length=L, parallelism_config=parallel_config
        )
        
        breakdown_2x = perf.calculate_prefill_compute_breakdown(
            batch_size=1, sequence_length=2*L, parallelism_config=parallel_config
        )
        
        # Mamba should scale linearly (ratio ~= 2)
        mamba_ratio = breakdown_2x['mamba'] / breakdown_1x['mamba']
        assert mamba_ratio == pytest.approx(2.0, rel=0.1)
    
    def test_attention_scales_quadratically(self, pure_attention_model, parallel_config):
        """Test attention FLOPs scale quadratically for comparison"""
        perf = InferencePerformance(pure_attention_model)
        
        # Use longer sequences to make quadratic term more dominant
        L = 1024
        breakdown_1x = perf.calculate_prefill_compute_breakdown(
            batch_size=1, sequence_length=L, parallelism_config=parallel_config
        )
        
        breakdown_2x = perf.calculate_prefill_compute_breakdown(
            batch_size=1, sequence_length=2*L, parallelism_config=parallel_config
        )
        
        # Attention should scale quadratically (ratio > 2.5 due to L*L terms)
        attn_ratio = breakdown_2x['attention'] / breakdown_1x['attention']
        # Should be closer to 4 than to 2, showing super-linear scaling
        assert attn_ratio > 2.5  # More than linear (2x)
    
    def test_mamba_more_efficient_at_long_sequences(self, pure_attention_model, pure_mamba_model, parallel_config):
        """Test Mamba has fewer FLOPs than attention at long sequences"""
        perf_attn = InferencePerformance(pure_attention_model)
        perf_mamba = InferencePerformance(pure_mamba_model)
        
        # At long sequence, Mamba (linear) should be more efficient than attention (quadratic)
        L = 4096
        
        attn_breakdown = perf_attn.calculate_prefill_compute_breakdown(
            batch_size=1, sequence_length=L, parallelism_config=parallel_config
        )
        mamba_breakdown = perf_mamba.calculate_prefill_compute_breakdown(
            batch_size=1, sequence_length=L, parallelism_config=parallel_config
        )
        
        # At 4096 sequence length, Mamba should be significantly more efficient
        # (in terms of the sequence-modeling computation: attention vs mamba)
        assert mamba_breakdown['mamba'] < attn_breakdown['attention']


class TestMambaHybridDecodeCompute:
    """Tests for decode compute with Mamba-2 hybrid architecture"""
    
    @pytest.fixture
    def parallel_config(self):
        return ParallelismConfig(
            parallelism_type=ParallelismType.TENSOR_PARALLEL,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
        )
    
    @pytest.fixture
    def hybrid_model(self):
        """Model with alternating Mamba and Attention layers"""
        return LLMArchitecture(
            model_name="test-hybrid",
            model_family="Test",
            version="1.0",
            architecture_type=ArchitectureType.DECODER_ONLY,
            num_layers=4,
            hidden_dim=1024,
            vocab_size=32000,
            max_sequence_length=4096,
            attention_config=AttentionConfig(
                num_attention_heads=16,
                num_key_value_heads=4,
                attention_type=AttentionType.GROUPED_QUERY,
                head_dim=64,
            ),
            ffn_config=FFNConfig(
                intermediate_size=2048,
                activation=ActivationType.SILU,
            ),
            hybrid_layer_types=[
                HybridLayerType.MAMBA,
                HybridLayerType.ATTENTION,
                HybridLayerType.MAMBA,
                HybridLayerType.ATTENTION,
            ],
            mamba_config=Mamba2Config(
                num_heads=16,
                head_dim=64,
                state_size=128,
            ),
            total_parameters=500_000_000,
            active_parameters=500_000_000,
        )
    
    def test_decode_has_mamba_flops(self, hybrid_model, parallel_config):
        """Test decode breakdown includes Mamba FLOPs"""
        perf = InferencePerformance(hybrid_model)
        
        breakdown = perf._calculate_decode_step_compute_breakdown(
            batch_size=1, context_length=1024, parallelism_config=parallel_config
        )
        
        assert breakdown['attention'] > 0
        assert breakdown['mamba'] > 0
        assert breakdown['ffn'] > 0
        assert breakdown['total'] == breakdown['attention'] + breakdown['mamba'] + breakdown['ffn'] + breakdown['other']
    
    def test_decode_mamba_constant_per_token(self, hybrid_model, parallel_config):
        """Test Mamba decode FLOPs are constant regardless of context length"""
        perf = InferencePerformance(hybrid_model)
        
        # Different context lengths
        breakdown_1k = perf._calculate_decode_step_compute_breakdown(
            batch_size=1, context_length=1024, parallelism_config=parallel_config
        )
        breakdown_4k = perf._calculate_decode_step_compute_breakdown(
            batch_size=1, context_length=4096, parallelism_config=parallel_config
        )
        
        # Mamba FLOPs should be the same (O(1) per token)
        assert breakdown_1k['mamba'] == breakdown_4k['mamba']
        
        # Attention FLOPs should scale with context (attending to more KV)
        assert breakdown_4k['attention'] > breakdown_1k['attention']


class TestNemotronPerformance:
    """Tests for Nemotron-3-30B hybrid Mamba/Attention model"""
    
    @pytest.fixture
    def parallel_config(self):
        return ParallelismConfig(
            parallelism_type=ParallelismType.TENSOR_PARALLEL,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
        )
    
    def test_nemotron_layer_counts(self):
        """Verify Nemotron has correct layer type counts"""
        assert NEMOTRON_3_30B.num_layers == 52
        assert NEMOTRON_3_30B.hybrid_layer_types is not None
        
        mamba_count = NEMOTRON_3_30B.get_num_mamba_layers()
        attention_count = NEMOTRON_3_30B.get_num_attention_layers_hybrid()
        
        # 52 layers total
        assert mamba_count + attention_count == 52
        assert mamba_count > 0  # Has Mamba layers
        assert attention_count > 0  # Has Attention layers
    
    def test_nemotron_has_mamba_config(self):
        """Verify Nemotron has Mamba configuration"""
        assert NEMOTRON_3_30B.mamba_config is not None
        assert NEMOTRON_3_30B.mamba_config.num_heads == 64
        assert NEMOTRON_3_30B.mamba_config.head_dim == 64
        assert NEMOTRON_3_30B.mamba_config.state_size == 128
    
    def test_nemotron_prefill_breakdown(self, parallel_config):
        """Test Nemotron prefill compute breakdown"""
        perf = InferencePerformance(NEMOTRON_3_30B)
        
        breakdown = perf.calculate_prefill_compute_breakdown(
            batch_size=1, sequence_length=1024, parallelism_config=parallel_config
        )
        
        # Should have both attention and Mamba FLOPs
        assert breakdown['attention'] > 0
        assert breakdown['mamba'] > 0
        assert breakdown['ffn'] > 0
        
        # Total should be sum of all components
        expected_total = breakdown['attention'] + breakdown['mamba'] + breakdown['ffn'] + breakdown['other']
        assert breakdown['total'] == pytest.approx(expected_total, rel=0.001)
    
    def test_nemotron_decode_breakdown(self, parallel_config):
        """Test Nemotron decode compute breakdown"""
        perf = InferencePerformance(NEMOTRON_3_30B)
        
        breakdown = perf._calculate_decode_step_compute_breakdown(
            batch_size=1, context_length=4096, parallelism_config=parallel_config
        )
        
        # Should have both attention and Mamba FLOPs
        assert breakdown['attention'] > 0
        assert breakdown['mamba'] > 0
        assert breakdown['ffn'] > 0
    
    def test_nemotron_is_moe(self):
        """Verify Nemotron is also an MoE model"""
        assert NEMOTRON_3_30B.is_moe is True
        assert NEMOTRON_3_30B.moe_config is not None
        assert NEMOTRON_3_30B.moe_config.num_experts == 128
        assert NEMOTRON_3_30B.moe_config.num_experts_per_token == 6


class TestMambaFlopsCalculation:
    """Tests for Mamba FLOPs calculation correctness"""
    
    @pytest.fixture
    def mamba_config(self):
        """Standard Mamba-2 config for testing"""
        return Mamba2Config(
            num_heads=64,
            head_dim=64,
            state_size=128,
            chunk_size=128,
            expand=2,
        )
    
    def test_prefill_flops_formula(self, mamba_config):
        """Test prefill FLOPs match expected formula"""
        T = 1024
        d_model = 2688
        
        flops = mamba_config.get_prefill_flops(T, d_model)
        
        # Verify it's positive and reasonable
        assert flops > 0
        
        # Verify formula components:
        # in_proj: 2 * T * d_model * d_proj
        # out_proj: 2 * T * d_inner * d_model
        # ssm: T * (6 * H * d_head * N + 2 * H * d_head)
        
        H = mamba_config.num_heads
        d_head = mamba_config.head_dim
        N = mamba_config.state_size
        d_inner = mamba_config.d_inner
        d_proj = 2 * d_inner + 2 * H * N + H
        
        expected_in = 2 * T * d_model * d_proj
        expected_out = 2 * T * d_inner * d_model
        expected_ssm = T * (6 * H * d_head * N + 2 * H * d_head)
        expected_total = expected_in + expected_ssm + expected_out
        
        assert flops == expected_total
    
    def test_decode_flops_formula(self, mamba_config):
        """Test decode FLOPs match expected formula"""
        d_model = 2688
        
        flops = mamba_config.get_decode_flops(d_model)
        
        # Verify it's positive
        assert flops > 0
        
        # Decode should be less than prefill for same d_model (no sequence dimension)
        prefill_flops = mamba_config.get_prefill_flops(1, d_model)
        # For T=1, should be approximately equal
        assert flops == pytest.approx(prefill_flops, rel=0.1)


class TestNoRegressionNonHybridModels:
    """Ensure non-hybrid models still work correctly"""
    
    @pytest.fixture
    def parallel_config(self):
        return ParallelismConfig(
            parallelism_type=ParallelismType.TENSOR_PARALLEL,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
        )
    
    def test_llama3_no_mamba(self, parallel_config):
        """Test Llama-3-8B (no Mamba) still works"""
        from llm_configs import LLAMA_3_8B
        
        assert LLAMA_3_8B.hybrid_layer_types is None
        assert LLAMA_3_8B.mamba_config is None
        
        perf = InferencePerformance(LLAMA_3_8B)
        breakdown = perf.calculate_prefill_compute_breakdown(
            batch_size=1, sequence_length=1024, parallelism_config=parallel_config
        )
        
        assert breakdown['attention'] > 0
        assert breakdown['mamba'] == 0
        assert breakdown['ffn'] > 0
    
    def test_llama4_no_mamba(self, parallel_config):
        """Test Llama-4 Scout (interleaved MoE, no Mamba) still works"""
        from llm_configs import LLAMA_4_SCOUT
        
        assert LLAMA_4_SCOUT.hybrid_layer_types is None
        assert LLAMA_4_SCOUT.mamba_config is None
        
        perf = InferencePerformance(LLAMA_4_SCOUT)
        breakdown = perf.calculate_prefill_compute_breakdown(
            batch_size=1, sequence_length=1024, parallelism_config=parallel_config
        )
        
        assert breakdown['mamba'] == 0
        assert breakdown['ffn'] > 0  # MoE FFN
