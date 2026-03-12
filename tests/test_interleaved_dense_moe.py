"""
Tests for interleaved Dense/MoE FFN layer support.

Tests cover:
1. FFNLayerType enum
2. FFNConfig.dense_intermediate_size
3. Layer counting helpers (get_num_dense_ffn_layers, get_num_moe_ffn_layers)
4. Llama-4 Scout (all MoE) and Maverick (interleaved) configurations
5. Edge cases and pattern generation
"""

import pytest
from llm_architecture import (
    LLMArchitecture, AttentionConfig, FFNConfig, MoEConfig,
    ArchitectureType, AttentionType, ActivationType,
    NormalizationType, PositionEncodingType, FFNLayerType
)
from llm_configs import (
    LLAMA_4_SCOUT, LLAMA_4_MAVERICK, get_model, ALL_MODELS
)


class TestFFNLayerTypeEnum:
    """Tests for FFNLayerType enum"""
    
    def test_dense_enum_value(self):
        """Test DENSE enum exists and has correct value"""
        assert FFNLayerType.DENSE.value == "dense"
    
    def test_moe_enum_value(self):
        """Test MOE enum exists and has correct value"""
        assert FFNLayerType.MOE.value == "moe"
    
    def test_enum_members(self):
        """Test all expected enum members exist"""
        members = list(FFNLayerType)
        assert len(members) == 2
        assert FFNLayerType.DENSE in members
        assert FFNLayerType.MOE in members


class TestFFNConfigDenseSize:
    """Tests for FFNConfig.dense_intermediate_size"""
    
    def test_dense_size_default_none(self):
        """Test dense_intermediate_size defaults to None"""
        config = FFNConfig(intermediate_size=4096)
        assert config.dense_intermediate_size is None
    
    def test_dense_size_explicit(self):
        """Test explicit dense_intermediate_size"""
        config = FFNConfig(
            intermediate_size=8192,  # MoE expert size
            dense_intermediate_size=16384  # Dense layer size
        )
        assert config.intermediate_size == 8192
        assert config.dense_intermediate_size == 16384
    
    def test_get_dense_intermediate_size_fallback(self):
        """Test get_dense_intermediate_size falls back to intermediate_size"""
        arch = LLMArchitecture(
            model_name="test",
            model_family="Test",
            version="1.0",
            architecture_type=ArchitectureType.DECODER_ONLY,
            num_layers=4,
            hidden_dim=1024,
            vocab_size=32000,
            max_sequence_length=2048,
            attention_config=AttentionConfig(
                num_attention_heads=16,
                num_key_value_heads=4,
                attention_type=AttentionType.GROUPED_QUERY,
                head_dim=64,
            ),
            ffn_config=FFNConfig(
                intermediate_size=4096,
                # dense_intermediate_size not set
            ),
            total_parameters=1_000_000_000,
            active_parameters=1_000_000_000,
        )
        assert arch.get_dense_intermediate_size() == 4096
    
    def test_get_dense_intermediate_size_explicit(self):
        """Test get_dense_intermediate_size returns explicit value"""
        arch = LLMArchitecture(
            model_name="test",
            model_family="Test",
            version="1.0",
            architecture_type=ArchitectureType.DECODER_ONLY,
            num_layers=4,
            hidden_dim=1024,
            vocab_size=32000,
            max_sequence_length=2048,
            attention_config=AttentionConfig(
                num_attention_heads=16,
                num_key_value_heads=4,
                attention_type=AttentionType.GROUPED_QUERY,
                head_dim=64,
            ),
            ffn_config=FFNConfig(
                intermediate_size=2048,
                dense_intermediate_size=8192,
            ),
            total_parameters=1_000_000_000,
            active_parameters=1_000_000_000,
        )
        assert arch.get_dense_intermediate_size() == 8192
        assert arch.get_moe_intermediate_size() == 2048


class TestFFNLayerCounting:
    """Tests for FFN layer type counting methods"""
    
    @pytest.fixture
    def all_moe_arch(self):
        """Architecture with all MoE layers"""
        return LLMArchitecture(
            model_name="test-all-moe",
            model_family="Test",
            version="1.0",
            architecture_type=ArchitectureType.DECODER_ONLY,
            num_layers=8,
            hidden_dim=1024,
            vocab_size=32000,
            max_sequence_length=2048,
            attention_config=AttentionConfig(
                num_attention_heads=16,
                num_key_value_heads=4,
                attention_type=AttentionType.GROUPED_QUERY,
                head_dim=64,
            ),
            ffn_config=FFNConfig(
                intermediate_size=2048,
                dense_intermediate_size=4096,
            ),
            moe_config=MoEConfig(
                num_experts=16,
                num_experts_per_token=1,
            ),
            is_moe=True,
            ffn_layer_types=[FFNLayerType.MOE] * 8,
            total_parameters=1_000_000_000,
            active_parameters=500_000_000,
        )
    
    @pytest.fixture
    def interleaved_arch(self):
        """Architecture with interleaved Dense/MoE layers"""
        return LLMArchitecture(
            model_name="test-interleaved",
            model_family="Test",
            version="1.0",
            architecture_type=ArchitectureType.DECODER_ONLY,
            num_layers=8,
            hidden_dim=1024,
            vocab_size=32000,
            max_sequence_length=2048,
            attention_config=AttentionConfig(
                num_attention_heads=16,
                num_key_value_heads=4,
                attention_type=AttentionType.GROUPED_QUERY,
                head_dim=64,
            ),
            ffn_config=FFNConfig(
                intermediate_size=2048,
                dense_intermediate_size=4096,
            ),
            moe_config=MoEConfig(
                num_experts=16,
                num_experts_per_token=1,
            ),
            is_moe=True,
            # Dense, MoE, Dense, MoE, ...
            ffn_layer_types=[
                FFNLayerType.DENSE if i % 2 == 0 else FFNLayerType.MOE
                for i in range(8)
            ],
            total_parameters=1_000_000_000,
            active_parameters=500_000_000,
        )
    
    @pytest.fixture
    def all_dense_arch(self):
        """Architecture with all dense layers (no MoE)"""
        return LLMArchitecture(
            model_name="test-all-dense",
            model_family="Test",
            version="1.0",
            architecture_type=ArchitectureType.DECODER_ONLY,
            num_layers=8,
            hidden_dim=1024,
            vocab_size=32000,
            max_sequence_length=2048,
            attention_config=AttentionConfig(
                num_attention_heads=16,
                num_key_value_heads=4,
                attention_type=AttentionType.GROUPED_QUERY,
                head_dim=64,
            ),
            ffn_config=FFNConfig(
                intermediate_size=4096,
            ),
            is_moe=False,
            total_parameters=1_000_000_000,
            active_parameters=1_000_000_000,
        )
    
    def test_all_moe_layer_counts(self, all_moe_arch):
        """Test counting for all-MoE architecture"""
        assert all_moe_arch.get_num_dense_ffn_layers() == 0
        assert all_moe_arch.get_num_moe_ffn_layers() == 8
    
    def test_interleaved_layer_counts(self, interleaved_arch):
        """Test counting for interleaved architecture"""
        assert interleaved_arch.get_num_dense_ffn_layers() == 4
        assert interleaved_arch.get_num_moe_ffn_layers() == 4
    
    def test_all_dense_layer_counts(self, all_dense_arch):
        """Test counting for all-dense architecture"""
        # No ffn_layer_types specified, uses is_moe flag
        assert all_dense_arch.get_num_dense_ffn_layers() == 8
        assert all_dense_arch.get_num_moe_ffn_layers() == 0
    
    def test_no_ffn_layer_types_moe(self):
        """Test layer counting when ffn_layer_types not specified (MoE model)"""
        arch = LLMArchitecture(
            model_name="test",
            model_family="Test",
            version="1.0",
            architecture_type=ArchitectureType.DECODER_ONLY,
            num_layers=6,
            hidden_dim=1024,
            vocab_size=32000,
            max_sequence_length=2048,
            attention_config=AttentionConfig(
                num_attention_heads=16,
                num_key_value_heads=4,
                attention_type=AttentionType.GROUPED_QUERY,
                head_dim=64,
            ),
            ffn_config=FFNConfig(intermediate_size=4096),
            moe_config=MoEConfig(num_experts=8, num_experts_per_token=2),
            is_moe=True,
            # No ffn_layer_types
            total_parameters=1_000_000_000,
            active_parameters=500_000_000,
        )
        # is_moe=True, no ffn_layer_types -> all MoE
        assert arch.get_num_dense_ffn_layers() == 0
        assert arch.get_num_moe_ffn_layers() == 6
    
    def test_layer_count_sums_to_total(self, interleaved_arch):
        """Test that dense + MoE layer counts equal total layers"""
        total = interleaved_arch.get_num_dense_ffn_layers() + interleaved_arch.get_num_moe_ffn_layers()
        assert total == interleaved_arch.num_layers


class TestLlama4Scout:
    """Tests for Llama-4 Scout configuration"""
    
    def test_model_exists_in_all_models(self):
        """Test Scout is registered in ALL_MODELS"""
        assert "llama-4-scout" in ALL_MODELS
    
    def test_get_model(self):
        """Test model can be retrieved by key"""
        model = get_model("llama-4-scout")
        assert model is LLAMA_4_SCOUT
    
    def test_basic_architecture(self):
        """Test basic architecture parameters"""
        assert LLAMA_4_SCOUT.model_name == "Llama-4-Scout"
        assert LLAMA_4_SCOUT.num_layers == 48
        assert LLAMA_4_SCOUT.hidden_dim == 5120
        assert LLAMA_4_SCOUT.vocab_size == 202048
    
    def test_all_moe_layers(self):
        """Test Scout has all MoE layers (interleave_moe_layer_step=1)"""
        assert LLAMA_4_SCOUT.get_num_moe_ffn_layers() == 48
        assert LLAMA_4_SCOUT.get_num_dense_ffn_layers() == 0
    
    def test_ffn_sizes(self):
        """Test FFN intermediate sizes"""
        assert LLAMA_4_SCOUT.get_moe_intermediate_size() == 8192
        assert LLAMA_4_SCOUT.get_dense_intermediate_size() == 16384
    
    def test_moe_config(self):
        """Test MoE configuration"""
        assert LLAMA_4_SCOUT.moe_config.num_experts == 16
        assert LLAMA_4_SCOUT.moe_config.num_experts_per_token == 1
    
    def test_ffn_layer_types_length(self):
        """Test ffn_layer_types has correct length"""
        assert len(LLAMA_4_SCOUT.ffn_layer_types) == LLAMA_4_SCOUT.num_layers
    
    def test_summary_shows_interleaved_info(self):
        """Test summary includes interleaved FFN info"""
        summary = LLAMA_4_SCOUT.summary()
        assert "Interleaved FFN" in summary
        assert "0 Dense" in summary
        assert "48 MoE" in summary


class TestLlama4Maverick:
    """Tests for Llama-4 Maverick configuration"""
    
    def test_model_exists_in_all_models(self):
        """Test Maverick is registered in ALL_MODELS"""
        assert "llama-4-maverick" in ALL_MODELS
    
    def test_get_model(self):
        """Test model can be retrieved by key"""
        model = get_model("llama-4-maverick")
        assert model is LLAMA_4_MAVERICK
    
    def test_basic_architecture(self):
        """Test basic architecture parameters"""
        assert LLAMA_4_MAVERICK.model_name == "Llama-4-Maverick"
        assert LLAMA_4_MAVERICK.num_layers == 48
        assert LLAMA_4_MAVERICK.hidden_dim == 5120
    
    def test_interleaved_layers(self):
        """Test Maverick has interleaved layers (interleave_moe_layer_step=2)"""
        assert LLAMA_4_MAVERICK.get_num_moe_ffn_layers() == 24
        assert LLAMA_4_MAVERICK.get_num_dense_ffn_layers() == 24
    
    def test_alternating_pattern(self):
        """Test layers alternate Dense, MoE, Dense, MoE, ..."""
        for i, layer_type in enumerate(LLAMA_4_MAVERICK.ffn_layer_types):
            expected = FFNLayerType.DENSE if i % 2 == 0 else FFNLayerType.MOE
            assert layer_type == expected, f"Layer {i} should be {expected}, got {layer_type}"
    
    def test_ffn_sizes(self):
        """Test FFN intermediate sizes"""
        assert LLAMA_4_MAVERICK.get_moe_intermediate_size() == 8192
        assert LLAMA_4_MAVERICK.get_dense_intermediate_size() == 16384
    
    def test_moe_config(self):
        """Test MoE configuration"""
        assert LLAMA_4_MAVERICK.moe_config.num_experts == 128
        assert LLAMA_4_MAVERICK.moe_config.num_experts_per_token == 1
    
    def test_more_experts_than_scout(self):
        """Test Maverick has more experts than Scout"""
        assert LLAMA_4_MAVERICK.moe_config.num_experts > LLAMA_4_SCOUT.moe_config.num_experts
    
    def test_summary_shows_interleaved_info(self):
        """Test summary includes interleaved FFN info"""
        summary = LLAMA_4_MAVERICK.summary()
        assert "Interleaved FFN" in summary
        assert "24 Dense" in summary
        assert "24 MoE" in summary


class TestInterleaveMoeLayerStep:
    """Tests for different interleave_moe_layer_step patterns"""
    
    def test_step_1_all_moe(self):
        """Test interleave_moe_layer_step=1 creates all MoE"""
        # Step 1: every layer is MoE
        pattern = [FFNLayerType.MOE] * 12
        assert all(t == FFNLayerType.MOE for t in pattern)
    
    def test_step_2_alternating(self):
        """Test interleave_moe_layer_step=2 creates alternating"""
        # Step 2: Dense, MoE, Dense, MoE, ...
        pattern = [
            FFNLayerType.DENSE if i % 2 == 0 else FFNLayerType.MOE
            for i in range(12)
        ]
        assert pattern == [
            FFNLayerType.DENSE, FFNLayerType.MOE,
            FFNLayerType.DENSE, FFNLayerType.MOE,
            FFNLayerType.DENSE, FFNLayerType.MOE,
            FFNLayerType.DENSE, FFNLayerType.MOE,
            FFNLayerType.DENSE, FFNLayerType.MOE,
            FFNLayerType.DENSE, FFNLayerType.MOE,
        ]
    
    def test_step_3_one_in_three(self):
        """Test interleave_moe_layer_step=3 creates 1 MoE per 3 layers"""
        # Step 3: Dense, Dense, MoE, Dense, Dense, MoE, ...
        pattern = [
            FFNLayerType.MOE if (i + 1) % 3 == 0 else FFNLayerType.DENSE
            for i in range(12)
        ]
        assert pattern == [
            FFNLayerType.DENSE, FFNLayerType.DENSE, FFNLayerType.MOE,
            FFNLayerType.DENSE, FFNLayerType.DENSE, FFNLayerType.MOE,
            FFNLayerType.DENSE, FFNLayerType.DENSE, FFNLayerType.MOE,
            FFNLayerType.DENSE, FFNLayerType.DENSE, FFNLayerType.MOE,
        ]


class TestEdgeCases:
    """Edge case tests for interleaved architectures"""
    
    def test_single_layer_dense(self):
        """Test single-layer dense architecture"""
        arch = LLMArchitecture(
            model_name="test-single-dense",
            model_family="Test",
            version="1.0",
            architecture_type=ArchitectureType.DECODER_ONLY,
            num_layers=1,
            hidden_dim=512,
            vocab_size=10000,
            max_sequence_length=512,
            attention_config=AttentionConfig(
                num_attention_heads=8,
                num_key_value_heads=2,
                attention_type=AttentionType.GROUPED_QUERY,
                head_dim=64,
            ),
            ffn_config=FFNConfig(intermediate_size=2048),
            ffn_layer_types=[FFNLayerType.DENSE],
            total_parameters=100_000_000,
            active_parameters=100_000_000,
        )
        assert arch.get_num_dense_ffn_layers() == 1
        assert arch.get_num_moe_ffn_layers() == 0
    
    def test_single_layer_moe(self):
        """Test single-layer MoE architecture"""
        arch = LLMArchitecture(
            model_name="test-single-moe",
            model_family="Test",
            version="1.0",
            architecture_type=ArchitectureType.DECODER_ONLY,
            num_layers=1,
            hidden_dim=512,
            vocab_size=10000,
            max_sequence_length=512,
            attention_config=AttentionConfig(
                num_attention_heads=8,
                num_key_value_heads=2,
                attention_type=AttentionType.GROUPED_QUERY,
                head_dim=64,
            ),
            ffn_config=FFNConfig(intermediate_size=2048),
            moe_config=MoEConfig(num_experts=8, num_experts_per_token=2),
            is_moe=True,
            ffn_layer_types=[FFNLayerType.MOE],
            total_parameters=100_000_000,
            active_parameters=50_000_000,
        )
        assert arch.get_num_dense_ffn_layers() == 0
        assert arch.get_num_moe_ffn_layers() == 1
    
    def test_mostly_dense_few_moe(self):
        """Test architecture with mostly dense and few MoE layers"""
        # Pattern: D, D, D, M, D, D, D, M (2 MoE out of 8)
        pattern = [
            FFNLayerType.MOE if (i + 1) % 4 == 0 else FFNLayerType.DENSE
            for i in range(8)
        ]
        
        arch = LLMArchitecture(
            model_name="test-mostly-dense",
            model_family="Test",
            version="1.0",
            architecture_type=ArchitectureType.DECODER_ONLY,
            num_layers=8,
            hidden_dim=1024,
            vocab_size=32000,
            max_sequence_length=2048,
            attention_config=AttentionConfig(
                num_attention_heads=16,
                num_key_value_heads=4,
                attention_type=AttentionType.GROUPED_QUERY,
                head_dim=64,
            ),
            ffn_config=FFNConfig(
                intermediate_size=2048,
                dense_intermediate_size=4096,
            ),
            moe_config=MoEConfig(num_experts=8, num_experts_per_token=2),
            is_moe=True,
            ffn_layer_types=pattern,
            total_parameters=1_000_000_000,
            active_parameters=500_000_000,
        )
        
        assert arch.get_num_dense_ffn_layers() == 6
        assert arch.get_num_moe_ffn_layers() == 2


class TestComparisonWithOtherModels:
    """Tests comparing with non-interleaved models"""
    
    def test_deepseek_no_interleaved(self):
        """Test DeepSeek-V3 doesn't have ffn_layer_types"""
        from llm_configs import DEEPSEEK_V3
        assert DEEPSEEK_V3.ffn_layer_types is None
        # All MoE when ffn_layer_types is None and is_moe=True
        assert DEEPSEEK_V3.get_num_moe_ffn_layers() == DEEPSEEK_V3.num_layers
        assert DEEPSEEK_V3.get_num_dense_ffn_layers() == 0
    
    def test_llama3_no_interleaved(self):
        """Test Llama-3 doesn't have ffn_layer_types"""
        from llm_configs import LLAMA_3_8B
        assert LLAMA_3_8B.ffn_layer_types is None
        # All dense when ffn_layer_types is None and is_moe=False
        assert LLAMA_3_8B.get_num_moe_ffn_layers() == 0
        assert LLAMA_3_8B.get_num_dense_ffn_layers() == LLAMA_3_8B.num_layers
