"""
Tests for sliding window attention and per-layer attention type selection.

This tests the hybrid attention architecture feature where models can have:
- Per-layer attention type specification (sliding vs full attention)
- Sliding window attention that caps KV cache at window_size tokens
- Proper KV cache calculation that accounts for different layer types

Key scenarios tested:
1. Homogeneous full attention (all layers full) - baseline
2. Homogeneous sliding attention (all layers sliding) - uniform sliding
3. Hybrid alternating (sliding/full interleaved) - like GPT-OSS-120B
4. Custom patterns (first N layers sliding, rest full, etc.)
5. KV cache savings calculations
6. Per-layer attention type query methods
"""

import pytest
from llm_configs import ALL_MODELS, get_model, GPT_OSS_120B
from llm_architecture import (
    LLMArchitecture,
    AttentionConfig,
    FFNConfig,
    AttentionType,
    ActivationType,
    ArchitectureType,
    NormalizationType,
    PositionEncodingType,
    LayerAttentionType,
)
from inference_performance import (
    InferencePerformance,
    SystemConstraints,
    ParallelismConfig,
    ParallelismType,
    DecodePerformance,
)


class TestLayerAttentionTypeEnum:
    """Test the LayerAttentionType enum."""
    
    def test_layer_attention_types_exist(self):
        """Verify both attention types are defined."""
        assert LayerAttentionType.FULL_ATTENTION is not None
        assert LayerAttentionType.SLIDING_ATTENTION is not None
    
    def test_layer_attention_type_values(self):
        """Verify enum values are correct strings."""
        assert LayerAttentionType.FULL_ATTENTION.value == "full_attention"
        assert LayerAttentionType.SLIDING_ATTENTION.value == "sliding_attention"


class TestGPTOSS120BModelConfiguration:
    """Test GPT-OSS-120B model configuration with hybrid attention."""
    
    def test_gpt_oss_120b_in_all_models(self):
        """GPT-OSS-120B should be accessible from ALL_MODELS."""
        assert "gpt-oss-120b" in ALL_MODELS
        assert ALL_MODELS["gpt-oss-120b"] is GPT_OSS_120B
    
    def test_gpt_oss_120b_get_model(self):
        """get_model() should return GPT-OSS-120B."""
        model = get_model("gpt-oss-120b")
        assert model is GPT_OSS_120B
        assert model.model_name == "GPT-OSS-120B"
    
    def test_gpt_oss_120b_basic_architecture(self):
        """Verify basic architecture parameters."""
        assert GPT_OSS_120B.model_name == "GPT-OSS-120B"
        assert GPT_OSS_120B.model_family == "GPT-OSS"
        assert GPT_OSS_120B.architecture_type == ArchitectureType.DECODER_ONLY
    
    def test_gpt_oss_120b_dimensions(self):
        """Verify dimensions match config.json."""
        assert GPT_OSS_120B.hidden_dim == 2880
        assert GPT_OSS_120B.num_layers == 36
        assert GPT_OSS_120B.vocab_size == 201088
        assert GPT_OSS_120B.max_sequence_length == 131072  # 128K context
    
    def test_gpt_oss_120b_attention_config(self):
        """Verify attention configuration."""
        attn = GPT_OSS_120B.attention_config
        assert attn.num_attention_heads == 64
        assert attn.num_key_value_heads == 8  # GQA 8:1 ratio
        assert attn.attention_type == AttentionType.GROUPED_QUERY
        assert attn.head_dim == 64
        assert attn.sliding_window_size == 128
        assert attn.attention_bias == True
    
    def test_gpt_oss_120b_has_layer_types(self):
        """Verify layer_types is specified."""
        assert GPT_OSS_120B.layer_types is not None
        assert len(GPT_OSS_120B.layer_types) == 36  # Same as num_layers
    
    def test_gpt_oss_120b_alternating_pattern(self):
        """Verify alternating sliding/full attention pattern."""
        layer_types = GPT_OSS_120B.layer_types
        
        for i, layer_type in enumerate(layer_types):
            if i % 2 == 0:
                assert layer_type == LayerAttentionType.SLIDING_ATTENTION, \
                    f"Layer {i} should be sliding"
            else:
                assert layer_type == LayerAttentionType.FULL_ATTENTION, \
                    f"Layer {i} should be full"
    
    def test_gpt_oss_120b_layer_counts(self):
        """Verify correct counts of each layer type."""
        # 36 layers, alternating: 18 sliding + 18 full
        assert GPT_OSS_120B.get_num_full_attention_layers() == 18
        assert GPT_OSS_120B.get_num_sliding_attention_layers() == 18
    
    def test_gpt_oss_120b_moe_config(self):
        """Verify MoE configuration."""
        assert GPT_OSS_120B.is_moe == True
        moe = GPT_OSS_120B.moe_config
        assert moe is not None
        assert moe.num_experts == 128
        assert moe.num_experts_per_token == 4


class TestPerLayerAttentionTypeQuery:
    """Test methods to query per-layer attention types."""
    
    @pytest.fixture
    def hybrid_model(self):
        """Create a simple hybrid attention model for testing."""
        # 8 layers: alternating sliding/full
        return LLMArchitecture(
            model_name="Test-Hybrid",
            model_family="Test",
            version="1.0",
            num_layers=8,
            hidden_dim=512,
            vocab_size=32000,
            attention_config=AttentionConfig(
                num_attention_heads=8,
                num_key_value_heads=8,
                head_dim=64,
                sliding_window_size=256,
            ),
            ffn_config=FFNConfig(intermediate_size=2048),
            layer_types=[
                LayerAttentionType.SLIDING_ATTENTION,
                LayerAttentionType.FULL_ATTENTION,
                LayerAttentionType.SLIDING_ATTENTION,
                LayerAttentionType.FULL_ATTENTION,
                LayerAttentionType.SLIDING_ATTENTION,
                LayerAttentionType.FULL_ATTENTION,
                LayerAttentionType.SLIDING_ATTENTION,
                LayerAttentionType.FULL_ATTENTION,
            ],
            total_parameters=100_000_000,
        )
    
    @pytest.fixture
    def full_attention_model(self):
        """Create a model with all full attention layers."""
        return LLMArchitecture(
            model_name="Test-Full",
            model_family="Test",
            version="1.0",
            num_layers=8,
            hidden_dim=512,
            vocab_size=32000,
            attention_config=AttentionConfig(
                num_attention_heads=8,
                num_key_value_heads=8,
                head_dim=64,
            ),
            ffn_config=FFNConfig(intermediate_size=2048),
            # No layer_types specified - defaults to full attention
            total_parameters=100_000_000,
        )
    
    @pytest.fixture
    def sliding_only_model(self):
        """Create a model with all sliding window attention."""
        return LLMArchitecture(
            model_name="Test-Sliding",
            model_family="Test",
            version="1.0",
            num_layers=8,
            hidden_dim=512,
            vocab_size=32000,
            attention_config=AttentionConfig(
                num_attention_heads=8,
                num_key_value_heads=8,
                head_dim=64,
                attention_type=AttentionType.SLIDING_WINDOW,
                sliding_window_size=256,
            ),
            ffn_config=FFNConfig(intermediate_size=2048),
            # No layer_types - uses attention_config.attention_type
            total_parameters=100_000_000,
        )
    
    def test_get_layer_attention_type_hybrid(self, hybrid_model):
        """Test getting attention type for each layer in hybrid model."""
        for i in range(8):
            layer_type = hybrid_model.get_layer_attention_type(i)
            if i % 2 == 0:
                assert layer_type == LayerAttentionType.SLIDING_ATTENTION
            else:
                assert layer_type == LayerAttentionType.FULL_ATTENTION
    
    def test_get_layer_attention_type_full_model(self, full_attention_model):
        """Test that full attention model returns FULL_ATTENTION for all layers."""
        for i in range(8):
            assert full_attention_model.get_layer_attention_type(i) == LayerAttentionType.FULL_ATTENTION
    
    def test_get_layer_attention_type_sliding_model(self, sliding_only_model):
        """Test that sliding model returns SLIDING_ATTENTION for all layers."""
        for i in range(8):
            assert sliding_only_model.get_layer_attention_type(i) == LayerAttentionType.SLIDING_ATTENTION
    
    def test_get_layer_attention_type_out_of_range(self, hybrid_model):
        """Test that out of range layer index raises ValueError."""
        with pytest.raises(ValueError):
            hybrid_model.get_layer_attention_type(-1)
        
        with pytest.raises(ValueError):
            hybrid_model.get_layer_attention_type(8)
        
        with pytest.raises(ValueError):
            hybrid_model.get_layer_attention_type(100)
    
    def test_get_num_full_layers_hybrid(self, hybrid_model):
        """Test counting full attention layers in hybrid model."""
        assert hybrid_model.get_num_full_attention_layers() == 4
    
    def test_get_num_sliding_layers_hybrid(self, hybrid_model):
        """Test counting sliding attention layers in hybrid model."""
        assert hybrid_model.get_num_sliding_attention_layers() == 4
    
    def test_get_num_full_layers_full_model(self, full_attention_model):
        """Test that all layers are full attention in full model."""
        assert full_attention_model.get_num_full_attention_layers() == 8
        assert full_attention_model.get_num_sliding_attention_layers() == 0
    
    def test_get_num_layers_sliding_model(self, sliding_only_model):
        """Test that all layers are sliding in sliding model."""
        assert sliding_only_model.get_num_full_attention_layers() == 0
        assert sliding_only_model.get_num_sliding_attention_layers() == 8


class TestKVCacheSlidingWindow:
    """Test KV cache calculations with sliding window attention."""
    
    @pytest.fixture
    def full_attention_model(self):
        """8 layers, 8 KV heads, head_dim=64, no sliding window."""
        return LLMArchitecture(
            model_name="Test-Full",
            model_family="Test",
            version="1.0",
            num_layers=8,
            hidden_dim=512,
            vocab_size=32000,
            attention_config=AttentionConfig(
                num_attention_heads=8,
                num_key_value_heads=8,
                head_dim=64,
            ),
            ffn_config=FFNConfig(intermediate_size=2048),
            total_parameters=100_000_000,
        )
    
    @pytest.fixture
    def sliding_only_model(self):
        """8 layers, all sliding window with window_size=128."""
        return LLMArchitecture(
            model_name="Test-Sliding",
            model_family="Test",
            version="1.0",
            num_layers=8,
            hidden_dim=512,
            vocab_size=32000,
            attention_config=AttentionConfig(
                num_attention_heads=8,
                num_key_value_heads=8,
                head_dim=64,
                attention_type=AttentionType.SLIDING_WINDOW,
                sliding_window_size=128,
            ),
            ffn_config=FFNConfig(intermediate_size=2048),
            total_parameters=100_000_000,
        )
    
    @pytest.fixture
    def hybrid_model(self):
        """8 layers, alternating sliding(128)/full, window_size=128."""
        return LLMArchitecture(
            model_name="Test-Hybrid",
            model_family="Test",
            version="1.0",
            num_layers=8,
            hidden_dim=512,
            vocab_size=32000,
            attention_config=AttentionConfig(
                num_attention_heads=8,
                num_key_value_heads=8,
                head_dim=64,
                sliding_window_size=128,
            ),
            ffn_config=FFNConfig(intermediate_size=2048),
            layer_types=[
                LayerAttentionType.SLIDING_ATTENTION,
                LayerAttentionType.FULL_ATTENTION,
                LayerAttentionType.SLIDING_ATTENTION,
                LayerAttentionType.FULL_ATTENTION,
                LayerAttentionType.SLIDING_ATTENTION,
                LayerAttentionType.FULL_ATTENTION,
                LayerAttentionType.SLIDING_ATTENTION,
                LayerAttentionType.FULL_ATTENTION,
            ],
            total_parameters=100_000_000,
        )
    
    def test_full_attention_kv_cache_basic(self, full_attention_model):
        """Test KV cache for full attention model."""
        batch_size = 1
        seq_len = 1024
        bytes_per_elem = 2
        
        kv_cache = full_attention_model.get_kv_cache_size(batch_size, seq_len, bytes_per_elem)
        
        # Expected: 2 (K+V) * batch * num_kv_heads * seq_len * head_dim * bytes * num_layers
        # = 2 * 1 * 8 * 1024 * 64 * 2 * 8 = 16,777,216 bytes = 16 MB
        expected = 2 * 1 * 8 * 1024 * 64 * 2 * 8
        assert kv_cache == expected
    
    def test_sliding_only_kv_cache_capped(self, sliding_only_model):
        """Test that sliding window caps KV cache at window_size."""
        batch_size = 1
        seq_len = 1024  # Much larger than window_size=128
        bytes_per_elem = 2
        
        kv_cache = sliding_only_model.get_kv_cache_size(batch_size, seq_len, bytes_per_elem)
        
        # Expected: KV cache should be based on window_size (128), not seq_len (1024)
        # = 2 * 1 * 8 * 128 * 64 * 2 * 8 = 2,097,152 bytes = 2 MB
        expected = 2 * 1 * 8 * 128 * 64 * 2 * 8
        assert kv_cache == expected
    
    def test_sliding_window_savings(self, full_attention_model, sliding_only_model):
        """Test that sliding window provides expected KV cache savings."""
        batch_size = 1
        seq_len = 1024
        bytes_per_elem = 2
        window_size = 128
        
        full_kv = full_attention_model.get_kv_cache_size(batch_size, seq_len, bytes_per_elem)
        sliding_kv = sliding_only_model.get_kv_cache_size(batch_size, seq_len, bytes_per_elem)
        
        # Sliding should be window_size/seq_len = 128/1024 = 1/8 of full
        expected_ratio = window_size / seq_len
        actual_ratio = sliding_kv / full_kv
        
        assert abs(actual_ratio - expected_ratio) < 0.001
    
    def test_hybrid_kv_cache_intermediate(self, hybrid_model, full_attention_model, sliding_only_model):
        """Test that hybrid model KV cache is between full and sliding-only."""
        batch_size = 1
        seq_len = 1024
        bytes_per_elem = 2
        
        full_kv = full_attention_model.get_kv_cache_size(batch_size, seq_len, bytes_per_elem)
        sliding_kv = sliding_only_model.get_kv_cache_size(batch_size, seq_len, bytes_per_elem)
        hybrid_kv = hybrid_model.get_kv_cache_size(batch_size, seq_len, bytes_per_elem)
        
        # Hybrid should be between full and sliding
        assert sliding_kv < hybrid_kv < full_kv
    
    def test_hybrid_kv_cache_exact_calculation(self, hybrid_model):
        """Test exact KV cache calculation for hybrid model."""
        batch_size = 1
        seq_len = 1024
        bytes_per_elem = 2
        window_size = 128
        
        kv_cache = hybrid_model.get_kv_cache_size(batch_size, seq_len, bytes_per_elem)
        
        # 4 full layers: 2 * 1 * 8 * 1024 * 64 * 2 = 2,097,152 per layer * 4 = 8,388,608
        # 4 sliding layers: 2 * 1 * 8 * 128 * 64 * 2 = 262,144 per layer * 4 = 1,048,576
        # Total = 9,437,184 bytes
        kv_per_token = 2 * 1 * 8 * 64 * 2  # 2048 bytes per token
        expected_full = kv_per_token * seq_len * 4
        expected_sliding = kv_per_token * window_size * 4
        expected_total = expected_full + expected_sliding
        
        assert kv_cache == expected_total
    
    def test_sliding_short_sequence(self, sliding_only_model):
        """Test sliding window when sequence is shorter than window."""
        batch_size = 1
        seq_len = 64  # Shorter than window_size=128
        bytes_per_elem = 2
        
        kv_cache = sliding_only_model.get_kv_cache_size(batch_size, seq_len, bytes_per_elem)
        
        # When seq_len < window_size, we store the full sequence
        expected = 2 * 1 * 8 * 64 * 64 * 2 * 8  # 64 tokens, not 128
        assert kv_cache == expected


class TestKVCacheBreakdown:
    """Test KV cache breakdown by layer type."""
    
    @pytest.fixture
    def hybrid_model(self):
        """8 layers, 4 sliding + 4 full."""
        return LLMArchitecture(
            model_name="Test-Hybrid",
            model_family="Test",
            version="1.0",
            num_layers=8,
            hidden_dim=512,
            vocab_size=32000,
            attention_config=AttentionConfig(
                num_attention_heads=8,
                num_key_value_heads=8,
                head_dim=64,
                sliding_window_size=128,
            ),
            ffn_config=FFNConfig(intermediate_size=2048),
            layer_types=[
                LayerAttentionType.SLIDING_ATTENTION,
                LayerAttentionType.FULL_ATTENTION,
                LayerAttentionType.SLIDING_ATTENTION,
                LayerAttentionType.FULL_ATTENTION,
                LayerAttentionType.SLIDING_ATTENTION,
                LayerAttentionType.FULL_ATTENTION,
                LayerAttentionType.SLIDING_ATTENTION,
                LayerAttentionType.FULL_ATTENTION,
            ],
            total_parameters=100_000_000,
        )
    
    def test_kv_cache_breakdown_structure(self, hybrid_model):
        """Test that breakdown returns correct structure."""
        breakdown = hybrid_model.get_kv_cache_size_breakdown(
            batch_size=1, sequence_length=1024, bytes_per_element=2
        )
        
        assert 'full_attention_layers' in breakdown
        assert 'sliding_attention_layers' in breakdown
        assert 'total' in breakdown
        assert 'num_full_layers' in breakdown
        assert 'num_sliding_layers' in breakdown
    
    def test_kv_cache_breakdown_layer_counts(self, hybrid_model):
        """Test that breakdown reports correct layer counts."""
        breakdown = hybrid_model.get_kv_cache_size_breakdown(
            batch_size=1, sequence_length=1024, bytes_per_element=2
        )
        
        assert breakdown['num_full_layers'] == 4
        assert breakdown['num_sliding_layers'] == 4
    
    def test_kv_cache_breakdown_total_matches(self, hybrid_model):
        """Test that breakdown total matches get_kv_cache_size."""
        batch_size = 1
        seq_len = 1024
        bytes_per_elem = 2
        
        breakdown = hybrid_model.get_kv_cache_size_breakdown(batch_size, seq_len, bytes_per_elem)
        total_kv = hybrid_model.get_kv_cache_size(batch_size, seq_len, bytes_per_elem)
        
        assert breakdown['total'] == total_kv
        assert breakdown['full_attention_layers'] + breakdown['sliding_attention_layers'] == total_kv
    
    def test_kv_cache_breakdown_proportions(self, hybrid_model):
        """Test that breakdown shows expected proportions."""
        breakdown = hybrid_model.get_kv_cache_size_breakdown(
            batch_size=1, sequence_length=1024, bytes_per_element=2
        )
        
        # Full attention (4 layers) stores 1024 tokens each
        # Sliding attention (4 layers) stores 128 tokens each
        # Full should be 1024/128 = 8x larger per layer
        # With equal layer counts: full_total / sliding_total = 8
        
        ratio = breakdown['full_attention_layers'] / breakdown['sliding_attention_layers']
        expected_ratio = 1024 / 128  # seq_len / window_size
        
        assert abs(ratio - expected_ratio) < 0.001


class TestCustomLayerPatterns:
    """Test various custom per-layer attention patterns."""
    
    def test_first_half_sliding_second_half_full(self):
        """Test pattern where first half is sliding, second half is full."""
        layer_types = (
            [LayerAttentionType.SLIDING_ATTENTION] * 4 +
            [LayerAttentionType.FULL_ATTENTION] * 4
        )
        
        model = LLMArchitecture(
            model_name="Test-HalfHalf",
            model_family="Test",
            version="1.0",
            num_layers=8,
            hidden_dim=512,
            vocab_size=32000,
            attention_config=AttentionConfig(
                num_attention_heads=8,
                num_key_value_heads=8,
                head_dim=64,
                sliding_window_size=128,
            ),
            ffn_config=FFNConfig(intermediate_size=2048),
            layer_types=layer_types,
            total_parameters=100_000_000,
        )
        
        # First 4 layers should be sliding
        for i in range(4):
            assert model.get_layer_attention_type(i) == LayerAttentionType.SLIDING_ATTENTION
        
        # Last 4 layers should be full
        for i in range(4, 8):
            assert model.get_layer_attention_type(i) == LayerAttentionType.FULL_ATTENTION
    
    def test_single_full_layer_rest_sliding(self):
        """Test pattern with only one full attention layer."""
        layer_types = [LayerAttentionType.SLIDING_ATTENTION] * 7 + [LayerAttentionType.FULL_ATTENTION]
        
        model = LLMArchitecture(
            model_name="Test-SingleFull",
            model_family="Test",
            version="1.0",
            num_layers=8,
            hidden_dim=512,
            vocab_size=32000,
            attention_config=AttentionConfig(
                num_attention_heads=8,
                num_key_value_heads=8,
                head_dim=64,
                sliding_window_size=128,
            ),
            ffn_config=FFNConfig(intermediate_size=2048),
            layer_types=layer_types,
            total_parameters=100_000_000,
        )
        
        assert model.get_num_full_attention_layers() == 1
        assert model.get_num_sliding_attention_layers() == 7
        
        # KV cache should be mostly sliding
        kv_cache = model.get_kv_cache_size(batch_size=1, sequence_length=1024, bytes_per_element=2)
        
        # Compare to all-full model
        all_full_layers = [LayerAttentionType.FULL_ATTENTION] * 8
        full_model = LLMArchitecture(
            model_name="Test-AllFull",
            model_family="Test",
            version="1.0",
            num_layers=8,
            hidden_dim=512,
            vocab_size=32000,
            attention_config=AttentionConfig(
                num_attention_heads=8,
                num_key_value_heads=8,
                head_dim=64,
            ),
            ffn_config=FFNConfig(intermediate_size=2048),
            layer_types=all_full_layers,
            total_parameters=100_000_000,
        )
        
        full_kv_cache = full_model.get_kv_cache_size(batch_size=1, sequence_length=1024, bytes_per_element=2)
        
        # Should be much smaller than all-full
        assert kv_cache < full_kv_cache * 0.3  # Less than 30% of full
    
    def test_every_third_layer_full(self):
        """Test pattern with every 3rd layer being full attention."""
        # 12 layers: full at 0, 3, 6, 9
        layer_types = [
            LayerAttentionType.FULL_ATTENTION if i % 3 == 0 else LayerAttentionType.SLIDING_ATTENTION
            for i in range(12)
        ]
        
        model = LLMArchitecture(
            model_name="Test-EveryThird",
            model_family="Test",
            version="1.0",
            num_layers=12,
            hidden_dim=512,
            vocab_size=32000,
            attention_config=AttentionConfig(
                num_attention_heads=8,
                num_key_value_heads=8,
                head_dim=64,
                sliding_window_size=256,
            ),
            ffn_config=FFNConfig(intermediate_size=2048),
            layer_types=layer_types,
            total_parameters=100_000_000,
        )
        
        assert model.get_num_full_attention_layers() == 4
        assert model.get_num_sliding_attention_layers() == 8
        
        # Verify specific layers
        for i in range(12):
            if i % 3 == 0:
                assert model.get_layer_attention_type(i) == LayerAttentionType.FULL_ATTENTION
            else:
                assert model.get_layer_attention_type(i) == LayerAttentionType.SLIDING_ATTENTION


class TestSlidingWindowWithMLA:
    """Test sliding window combined with MLA compression."""
    
    def test_sliding_window_with_mla(self):
        """Test that MLA compression works with sliding window."""
        model = LLMArchitecture(
            model_name="Test-MLA-Sliding",
            model_family="Test",
            version="1.0",
            num_layers=8,
            hidden_dim=512,
            vocab_size=32000,
            attention_config=AttentionConfig(
                num_attention_heads=8,
                num_key_value_heads=8,
                head_dim=64,
                attention_type=AttentionType.SLIDING_WINDOW,
                sliding_window_size=128,
                use_mla=True,
                mla_kv_lora_rank=32,  # Compress to 32 dims
            ),
            ffn_config=FFNConfig(intermediate_size=2048),
            total_parameters=100_000_000,
        )
        
        batch_size = 1
        seq_len = 1024
        bytes_per_elem = 2
        
        kv_cache = model.get_kv_cache_size(batch_size, seq_len, bytes_per_elem)
        
        # MLA: KV dimension is mla_kv_lora_rank (32) not num_kv_heads * head_dim (8*64=512)
        # Sliding: effective seq_len is min(1024, 128) = 128
        # Expected: 2 * batch * 128 * 32 * bytes * 8 layers = 131,072 bytes
        expected = 2 * 1 * 128 * 32 * 2 * 8
        
        assert kv_cache == expected


class TestGPTOSS120BInference:
    """Test GPT-OSS-120B with inference performance calculations."""
    
    @pytest.fixture
    def h100_gpu(self):
        """H100 SXM GPU configuration."""
        return SystemConstraints.from_gpu_spec("H100-80GB")
    
    def test_gpt_oss_120b_kv_cache_savings(self):
        """Test that hybrid attention provides KV cache savings."""
        # Create equivalent full-attention model
        full_model = LLMArchitecture(
            model_name="GPT-OSS-120B-FullAttn",
            model_family="GPT-OSS",
            version="1.0",
            num_layers=36,
            hidden_dim=2880,
            vocab_size=201088,
            attention_config=AttentionConfig(
                num_attention_heads=64,
                num_key_value_heads=8,
                head_dim=64,
                # No sliding window
            ),
            ffn_config=FFNConfig(intermediate_size=2880),
            # No layer_types - all full attention
            total_parameters=120_000_000_000,
        )
        
        batch_size = 1
        seq_len = 4096  # Long context to see savings
        bytes_per_elem = 2
        
        hybrid_kv = GPT_OSS_120B.get_kv_cache_size(batch_size, seq_len, bytes_per_elem)
        full_kv = full_model.get_kv_cache_size(batch_size, seq_len, bytes_per_elem)
        
        # Hybrid should use less KV cache
        assert hybrid_kv < full_kv
        
        # Calculate expected savings
        # 18 full layers: 4096 tokens each
        # 18 sliding layers: 128 tokens each
        # Savings = (18 * (4096 - 128)) / (36 * 4096) ≈ 48.4%
        savings = (full_kv - hybrid_kv) / full_kv
        
        # Should save roughly 48% of KV cache
        assert 0.45 < savings < 0.52
    
    def test_gpt_oss_120b_prefill(self):
        """GPT-OSS-120B should complete prefill calculation."""
        perf = InferencePerformance(GPT_OSS_120B)
        result = perf.calculate_prefill_resources(
            batch_size=1,
            sequence_length=2048,
            time_to_first_token=0.3,
            num_gpus=4,
            parallelism_config=ParallelismConfig(
                parallelism_type=ParallelismType.TENSOR_PARALLEL,
                tensor_parallel_size=4
            )
        )
        
        assert result is not None
        assert result.time_to_first_token > 0
    
    def test_gpt_oss_120b_decode(self, h100_gpu):
        """GPT-OSS-120B should complete decode calculation."""
        perf = InferencePerformance(GPT_OSS_120B)
        parallelism = ParallelismConfig(
            parallelism_type=ParallelismType.TENSOR_PARALLEL,
            tensor_parallel_size=4
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
    
    def test_gpt_oss_120b_kv_cache_breakdown(self):
        """Test KV cache breakdown for GPT-OSS-120B."""
        breakdown = GPT_OSS_120B.get_kv_cache_size_breakdown(
            batch_size=1, sequence_length=8192, bytes_per_element=2
        )
        
        # 18 full + 18 sliding layers
        assert breakdown['num_full_layers'] == 18
        assert breakdown['num_sliding_layers'] == 18
        
        # Full layers should dominate the cache
        assert breakdown['full_attention_layers'] > breakdown['sliding_attention_layers']
        
        # With seq_len=8192 and window=128:
        # full_attention_layers should be 8192/128 = 64x larger per layer
        ratio = breakdown['full_attention_layers'] / breakdown['sliding_attention_layers']
        expected_ratio = 8192 / 128
        
        assert abs(ratio - expected_ratio) < 0.001


class TestEdgeCases:
    """Test edge cases for sliding window attention."""
    
    def test_no_sliding_window_size_specified(self):
        """Test behavior when sliding window size is not specified."""
        model = LLMArchitecture(
            model_name="Test-NoWindow",
            model_family="Test",
            version="1.0",
            num_layers=4,
            hidden_dim=256,
            vocab_size=32000,
            attention_config=AttentionConfig(
                num_attention_heads=4,
                num_key_value_heads=4,
                head_dim=64,
                # No sliding_window_size
            ),
            ffn_config=FFNConfig(intermediate_size=1024),
            layer_types=[
                LayerAttentionType.SLIDING_ATTENTION,
                LayerAttentionType.FULL_ATTENTION,
                LayerAttentionType.SLIDING_ATTENTION,
                LayerAttentionType.FULL_ATTENTION,
            ],
            total_parameters=50_000_000,
        )
        
        # When no sliding_window_size, sliding layers should fall back to full seq_len
        kv_cache = model.get_kv_cache_size(batch_size=1, sequence_length=1024, bytes_per_element=2)
        
        # Should be same as all-full since window_size defaults to seq_len
        expected_full = 2 * 1 * 4 * 1024 * 64 * 2 * 4
        assert kv_cache == expected_full
    
    def test_window_larger_than_sequence(self):
        """Test when sliding window is larger than sequence length."""
        model = LLMArchitecture(
            model_name="Test-LargeWindow",
            model_family="Test",
            version="1.0",
            num_layers=4,
            hidden_dim=256,
            vocab_size=32000,
            attention_config=AttentionConfig(
                num_attention_heads=4,
                num_key_value_heads=4,
                head_dim=64,
                attention_type=AttentionType.SLIDING_WINDOW,
                sliding_window_size=2048,  # Larger than seq_len we'll use
            ),
            ffn_config=FFNConfig(intermediate_size=1024),
            total_parameters=50_000_000,
        )
        
        # Use seq_len=512, smaller than window_size=2048
        kv_cache = model.get_kv_cache_size(batch_size=1, sequence_length=512, bytes_per_element=2)
        
        # Should use seq_len since it's smaller than window
        expected = 2 * 1 * 4 * 512 * 64 * 2 * 4
        assert kv_cache == expected
    
    def test_batch_size_scaling(self):
        """Test that KV cache scales linearly with batch size."""
        model = LLMArchitecture(
            model_name="Test-BatchScale",
            model_family="Test",
            version="1.0",
            num_layers=4,
            hidden_dim=256,
            vocab_size=32000,
            attention_config=AttentionConfig(
                num_attention_heads=4,
                num_key_value_heads=4,
                head_dim=64,
                sliding_window_size=128,
            ),
            ffn_config=FFNConfig(intermediate_size=1024),
            layer_types=[LayerAttentionType.SLIDING_ATTENTION] * 2 + [LayerAttentionType.FULL_ATTENTION] * 2,
            total_parameters=50_000_000,
        )
        
        kv_batch_1 = model.get_kv_cache_size(batch_size=1, sequence_length=1024, bytes_per_element=2)
        kv_batch_4 = model.get_kv_cache_size(batch_size=4, sequence_length=1024, bytes_per_element=2)
        
        assert kv_batch_4 == kv_batch_1 * 4
    
    def test_empty_layer_types_list(self):
        """Test that empty layer_types list is treated as no layer_types."""
        model = LLMArchitecture(
            model_name="Test-EmptyLayerTypes",
            model_family="Test",
            version="1.0",
            num_layers=4,
            hidden_dim=256,
            vocab_size=32000,
            attention_config=AttentionConfig(
                num_attention_heads=4,
                num_key_value_heads=4,
                head_dim=64,
            ),
            ffn_config=FFNConfig(intermediate_size=1024),
            layer_types=[],  # Empty list
            total_parameters=50_000_000,
        )
        
        # With empty layer_types, iteration should return 0 for both layer counts
        # since we iterate over an empty list
        kv_cache = model.get_kv_cache_size(batch_size=1, sequence_length=1024, bytes_per_element=2)
        
        # Empty layer_types = 0 layers in hybrid calculation
        assert kv_cache == 0
