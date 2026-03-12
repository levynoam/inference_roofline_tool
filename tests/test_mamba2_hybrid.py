"""
Tests for Mamba-2 hybrid architecture support.

Tests cover:
1. Mamba2Config - state size, FLOPs calculations, kernel launches
2. HybridLayerType - layer type counting and iteration
3. Nemotron-3-30B - hybrid Mamba/Attention architecture validation
4. State size calculations - KV cache vs Mamba state
5. Memory footprint with Mamba state
"""

import pytest
from llm_architecture import (
    LLMArchitecture, AttentionConfig, FFNConfig, MoEConfig,
    ArchitectureType, AttentionType, ActivationType,
    NormalizationType, PositionEncodingType,
    HybridLayerType, Mamba2Config
)
from llm_configs import NEMOTRON_3_30B, get_model, ALL_MODELS


class TestMamba2Config:
    """Tests for Mamba2Config dataclass"""
    
    @pytest.fixture
    def mamba_config(self):
        """Standard Mamba-2 config for testing"""
        return Mamba2Config(
            num_heads=64,
            head_dim=64,
            state_size=128,
            chunk_size=128,
            expand=2,
            conv_kernel=4,
        )
    
    def test_d_inner_property(self, mamba_config):
        """Test d_inner = num_heads * head_dim"""
        assert mamba_config.d_inner == 64 * 64 == 4096
    
    def test_state_size_bytes(self, mamba_config):
        """Test Mamba state size calculation"""
        # State shape: (batch, H, d_head, N)
        batch_size = 1
        bytes_per_element = 2  # bf16
        
        expected = batch_size * 64 * 64 * 128 * 2  # B * H * d_head * N * bytes
        assert mamba_config.get_state_size_bytes(batch_size, bytes_per_element) == expected
    
    def test_state_size_scales_with_batch(self, mamba_config):
        """Test state size scales linearly with batch size"""
        state_b1 = mamba_config.get_state_size_bytes(1, 2)
        state_b4 = mamba_config.get_state_size_bytes(4, 2)
        assert state_b4 == 4 * state_b1
    
    def test_prefill_flops(self, mamba_config):
        """Test prefill FLOPs calculation"""
        T = 1024  # sequence length
        d_model = 2688
        
        flops = mamba_config.get_prefill_flops(T, d_model)
        # Should be positive and scale with sequence length
        assert flops > 0
        
        # Compare with longer sequence
        flops_2x = mamba_config.get_prefill_flops(T * 2, d_model)
        # Prefill should scale roughly linearly with T (due to chunking)
        assert flops_2x > flops
    
    def test_decode_flops(self, mamba_config):
        """Test decode FLOPs per token"""
        d_model = 2688
        
        flops = mamba_config.get_decode_flops(d_model)
        # Should be positive
        assert flops > 0
    
    def test_prefill_kernel_launches(self, mamba_config):
        """Test prefill kernel launch count"""
        T = 1024
        
        launches = mamba_config.get_prefill_kernel_launches(T)
        # At minimum: chunk_sums, scan, and final output
        assert launches >= 3
        
        # More chunks = more launches
        T_long = 4096
        launches_long = mamba_config.get_prefill_kernel_launches(T_long)
        assert launches_long >= launches
    
    def test_decode_kernel_launches(self, mamba_config):
        """Test decode kernel launches (constant per token)"""
        assert mamba_config.get_decode_kernel_launches() == 4


class TestHybridLayerTypes:
    """Tests for hybrid layer type handling"""
    
    @pytest.fixture
    def hybrid_arch(self):
        """Create a simple hybrid architecture for testing"""
        return LLMArchitecture(
            model_name="test-hybrid",
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
            ffn_config=FFNConfig(
                intermediate_size=2048,
                activation=ActivationType.SILU,
            ),
            # Pattern: M A M A M A (alternating Mamba and Attention)
            hybrid_layer_types=[
                HybridLayerType.MAMBA,
                HybridLayerType.ATTENTION,
                HybridLayerType.MAMBA,
                HybridLayerType.ATTENTION,
                HybridLayerType.MAMBA,
                HybridLayerType.ATTENTION,
            ],
            mamba_config=Mamba2Config(
                num_heads=16,
                head_dim=64,
                state_size=64,
            ),
            total_parameters=1_000_000_000,
            active_parameters=1_000_000_000,
        )
    
    def test_count_mamba_layers(self, hybrid_arch):
        """Test counting Mamba layers"""
        assert hybrid_arch.get_num_mamba_layers() == 3
    
    def test_count_attention_layers_hybrid(self, hybrid_arch):
        """Test counting attention layers in hybrid model"""
        assert hybrid_arch.get_num_attention_layers_hybrid() == 3
    
    def test_count_mlp_only_layers(self, hybrid_arch):
        """Test counting MLP-only layers (none in this model)"""
        assert hybrid_arch.get_num_mlp_only_layers() == 0
    
    def test_non_hybrid_mamba_count(self):
        """Test Mamba layer count for non-hybrid models returns 0"""
        from llm_configs import LLAMA_3_8B
        assert LLAMA_3_8B.get_num_mamba_layers() == 0
    
    def test_non_hybrid_attention_count(self):
        """Test attention layer count for non-hybrid models returns all layers"""
        from llm_configs import LLAMA_3_8B
        assert LLAMA_3_8B.get_num_attention_layers_hybrid() == LLAMA_3_8B.num_layers


class TestMambaStateSize:
    """Tests for Mamba state size calculations"""
    
    @pytest.fixture
    def hybrid_arch(self):
        """Hybrid architecture with known Mamba config"""
        return LLMArchitecture(
            model_name="test-mamba-state",
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
                activation=ActivationType.SILU,
            ),
            hybrid_layer_types=[
                HybridLayerType.MAMBA,
                HybridLayerType.MAMBA,
                HybridLayerType.ATTENTION,
                HybridLayerType.ATTENTION,
            ],
            mamba_config=Mamba2Config(
                num_heads=16,
                head_dim=64,
                state_size=128,
            ),
            total_parameters=1_000_000_000,
            active_parameters=1_000_000_000,
        )
    
    def test_mamba_state_size(self, hybrid_arch):
        """Test total Mamba state size calculation"""
        batch_size = 1
        bytes_per_element = 2
        
        # 2 Mamba layers * (B * H * d_head * N * bytes)
        expected_per_layer = 1 * 16 * 64 * 128 * 2
        expected_total = 2 * expected_per_layer
        
        assert hybrid_arch.get_mamba_state_size(batch_size, bytes_per_element) == expected_total
    
    def test_mamba_state_size_no_mamba_layers(self):
        """Test Mamba state size is 0 for pure attention models"""
        from llm_configs import LLAMA_3_8B
        assert LLAMA_3_8B.get_mamba_state_size(batch_size=1) == 0
    
    def test_total_inference_state_size(self, hybrid_arch):
        """Test combined KV cache + Mamba state calculation"""
        state = hybrid_arch.get_total_inference_state_size(
            batch_size=1,
            sequence_length=1024,
            bytes_per_element=2
        )
        
        assert 'kv_cache' in state
        assert 'mamba_state' in state
        assert 'total' in state
        assert state['total'] == state['kv_cache'] + state['mamba_state']
        assert state['num_attention_layers'] == 2
        assert state['num_mamba_layers'] == 2


class TestNemotron30B:
    """Tests for Nemotron-3-30B model configuration"""
    
    def test_model_exists_in_all_models(self):
        """Test Nemotron is registered in ALL_MODELS"""
        assert "nemotron-3-30b" in ALL_MODELS
    
    def test_get_model(self):
        """Test model can be retrieved by key"""
        model = get_model("nemotron-3-30b")
        assert model is NEMOTRON_3_30B
    
    def test_basic_architecture(self):
        """Test basic architecture parameters"""
        assert NEMOTRON_3_30B.model_name == "Nemotron-3-30B"
        assert NEMOTRON_3_30B.num_layers == 52
        assert NEMOTRON_3_30B.hidden_dim == 2688
        assert NEMOTRON_3_30B.vocab_size == 131072
        assert NEMOTRON_3_30B.max_sequence_length == 262144
    
    def test_attention_config(self):
        """Test attention configuration"""
        attn = NEMOTRON_3_30B.attention_config
        assert attn.num_attention_heads == 32
        assert attn.num_key_value_heads == 2
        assert attn.head_dim == 128
        assert attn.attention_type == AttentionType.GROUPED_QUERY
    
    def test_moe_config(self):
        """Test MoE configuration"""
        moe = NEMOTRON_3_30B.moe_config
        assert moe.num_experts == 128
        assert moe.num_experts_per_token == 6
        assert moe.shared_expert is True
    
    def test_mamba_config(self):
        """Test Mamba-2 configuration"""
        mamba = NEMOTRON_3_30B.mamba_config
        assert mamba is not None
        assert mamba.num_heads == 64
        assert mamba.head_dim == 64
        assert mamba.state_size == 128
        assert mamba.chunk_size == 128
    
    def test_hybrid_layer_pattern(self):
        """Test hybrid layer pattern parsing"""
        # Pattern: MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME
        # 23 M, 23 E, 6 * = 52 total layers
        assert len(NEMOTRON_3_30B.hybrid_layer_types) == 52
        assert NEMOTRON_3_30B.get_num_mamba_layers() == 23
        # 23 E + 6 * = 29 attention layers
        assert NEMOTRON_3_30B.get_num_attention_layers_hybrid() == 29
    
    def test_layer_count_consistency(self):
        """Test layer counts sum to total"""
        mamba = NEMOTRON_3_30B.get_num_mamba_layers()
        attention = NEMOTRON_3_30B.get_num_attention_layers_hybrid()
        mlp = NEMOTRON_3_30B.get_num_mlp_only_layers()
        
        assert mamba + attention + mlp == NEMOTRON_3_30B.num_layers
    
    def test_mamba_state_size(self):
        """Test Mamba state size for Nemotron"""
        state_size = NEMOTRON_3_30B.get_mamba_state_size(batch_size=1, bytes_per_element=2)
        
        # 23 Mamba layers * (1 * 64 * 64 * 128 * 2) bytes
        expected = 23 * (1 * 64 * 64 * 128 * 2)
        assert state_size == expected
    
    def test_total_inference_state(self):
        """Test combined inference state calculation"""
        state = NEMOTRON_3_30B.get_total_inference_state_size(
            batch_size=1,
            sequence_length=4096,
            bytes_per_element=2
        )
        
        assert state['kv_cache'] > 0  # 29 attention layers
        assert state['mamba_state'] > 0  # 23 Mamba layers
        assert state['total'] > state['kv_cache']  # Mamba adds to total
    
    def test_memory_footprint_includes_mamba(self):
        """Test memory footprint includes Mamba state"""
        memory = NEMOTRON_3_30B.get_memory_footprint(batch_size=1, sequence_length=4096)
        
        assert 'mamba_state' in memory
        assert memory['mamba_state'] > 0
        assert memory['kv_cache'] > 0
    
    def test_summary_shows_hybrid_info(self):
        """Test summary includes hybrid architecture info"""
        summary = NEMOTRON_3_30B.summary()
        
        assert "Hybrid Architecture" in summary
        assert "29 Attention" in summary
        assert "23 Mamba" in summary
        assert "Mamba-2:" in summary


class TestMambaFlopsFormulas:
    """Tests for Mamba-2 FLOPs calculation formulas"""
    
    @pytest.fixture
    def mamba_config(self):
        """Nemotron-like Mamba config"""
        return Mamba2Config(
            num_heads=64,  # H
            head_dim=64,   # d_head
            state_size=128,  # N
            chunk_size=128,  # C
            expand=2,
        )
    
    def test_prefill_flops_components(self, mamba_config):
        """Test prefill FLOPs formula components"""
        T = 1024
        d_model = 2688
        H = mamba_config.num_heads
        d_head = mamba_config.head_dim
        N = mamba_config.state_size
        C = mamba_config.chunk_size
        D = mamba_config.d_inner
        
        flops = mamba_config.get_prefill_flops(T, d_model)
        
        # Verify components contribute positively
        # 1. Input projections: 2 * T * d_model * (expand * D)
        # 2. Conv1d: T * D * conv_kernel
        # 3. SSM operations: ~T * H * d_head * N * (chunk ops)
        # 4. Output projection: T * D * d_model
        
        assert flops > T * d_model  # At minimum, input/output projections
    
    def test_decode_flops_components(self, mamba_config):
        """Test decode FLOPs formula"""
        d_model = 2688
        
        flops = mamba_config.get_decode_flops(d_model)
        
        # Decode should be cheaper than prefill with T=chunk_size
        prefill_one_chunk = mamba_config.get_prefill_flops(mamba_config.chunk_size, d_model)
        
        # Decode FLOPs should be positive
        assert flops > 0
    
    def test_flops_scale_with_model_size(self):
        """Test FLOPs scale appropriately with model dimensions"""
        small = Mamba2Config(num_heads=32, head_dim=32, state_size=64)
        large = Mamba2Config(num_heads=64, head_dim=64, state_size=128)
        
        d_model = 1024
        T = 1024
        
        small_flops = small.get_prefill_flops(T, d_model)
        large_flops = large.get_prefill_flops(T, d_model)
        
        # Larger config should have more FLOPs
        assert large_flops > small_flops


class TestHybridArchitectureEdgeCases:
    """Edge case tests for hybrid architectures"""
    
    def test_all_mamba_layers(self):
        """Test architecture with all Mamba layers"""
        arch = LLMArchitecture(
            model_name="test-all-mamba",
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
                activation=ActivationType.SILU,
            ),
            hybrid_layer_types=[HybridLayerType.MAMBA] * 4,
            mamba_config=Mamba2Config(num_heads=16, head_dim=64, state_size=64),
            total_parameters=1_000_000_000,
            active_parameters=1_000_000_000,
        )
        
        assert arch.get_num_mamba_layers() == 4
        assert arch.get_num_attention_layers_hybrid() == 0
        # KV cache should be 0 for all-Mamba
        assert arch.get_kv_cache_size(1, 1024, 2) == 0
    
    def test_mixed_mlp_layers(self):
        """Test architecture with MLP-only layers"""
        arch = LLMArchitecture(
            model_name="test-mixed-mlp",
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
            ffn_config=FFNConfig(
                intermediate_size=2048,
                activation=ActivationType.SILU,
            ),
            hybrid_layer_types=[
                HybridLayerType.ATTENTION,
                HybridLayerType.MAMBA,
                HybridLayerType.MLP,
                HybridLayerType.ATTENTION,
                HybridLayerType.MAMBA,
                HybridLayerType.MLP,
            ],
            mamba_config=Mamba2Config(num_heads=16, head_dim=64, state_size=64),
            total_parameters=1_000_000_000,
            active_parameters=1_000_000_000,
        )
        
        assert arch.get_num_attention_layers_hybrid() == 2
        assert arch.get_num_mamba_layers() == 2
        assert arch.get_num_mlp_only_layers() == 2
    
    def test_mamba_config_without_hybrid_layers(self):
        """Test Mamba config is ignored without hybrid_layer_types"""
        arch = LLMArchitecture(
            model_name="test-mamba-no-hybrid",
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
                activation=ActivationType.SILU,
            ),
            # Has Mamba config but no hybrid_layer_types
            mamba_config=Mamba2Config(num_heads=16, head_dim=64, state_size=64),
            total_parameters=1_000_000_000,
            active_parameters=1_000_000_000,
        )
        
        # Without hybrid_layer_types, Mamba state should be 0
        assert arch.get_num_mamba_layers() == 0
        assert arch.get_mamba_state_size(1, 2) == 0


class TestActivationTypeRelu2:
    """Tests for RELU2 activation type"""
    
    def test_relu2_enum_exists(self):
        """Test RELU2 is a valid activation type"""
        assert ActivationType.RELU2.value == "relu2"
    
    def test_nemotron_uses_relu2(self):
        """Test Nemotron uses RELU2 activation"""
        assert NEMOTRON_3_30B.ffn_config.activation == ActivationType.RELU2
