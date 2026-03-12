"""
Tests for Linear Attention support.

Linear attention replaces softmax(Q @ K^T) @ V with Q @ (phi(K)^T @ phi(V)),
changing compute complexity from O(L^2) to O(L) and using a fixed-size state
instead of growing KV cache.
"""

import pytest
from llm_architecture import (
    LLMArchitecture, AttentionConfig, FFNConfig, MoEConfig,
    ArchitectureType, AttentionType, ActivationType,
    NormalizationType, PositionEncodingType, LayerAttentionType,
    FFNLayerType, LinearAttentionConfig,
)
from llm_configs import QWEN3_5_397B, get_model
from inference_performance import (
    InferencePerformance,
    SystemConstraints,
    ParallelismConfig,
    ParallelismType,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def qwen35():
    """Qwen3.5-397B model config."""
    return QWEN3_5_397B


@pytest.fixture
def linear_attn_config():
    """Standalone linear attention config for unit testing."""
    return LinearAttentionConfig(
        num_key_heads=16,
        key_head_dim=128,
        num_value_heads=64,
        value_head_dim=128,
        conv_kernel_dim=4,
    )


@pytest.fixture
def simple_hybrid_model():
    """
    Minimal model with 4 layers: 3 linear + 1 full attention.
    Useful for verifying per-layer logic without MoE complexity.
    """
    return LLMArchitecture(
        model_name="test-hybrid-linear",
        model_family="test",
        version="1.0",
        num_layers=4,
        hidden_dim=512,
        vocab_size=1000,
        max_sequence_length=8192,
        attention_config=AttentionConfig(
            num_attention_heads=8,
            num_key_value_heads=2,
            attention_type=AttentionType.GROUPED_QUERY,
            head_dim=64,
        ),
        ffn_config=FFNConfig(
            intermediate_size=2048,
            activation=ActivationType.SILU,
            use_gating=True,
        ),
        layer_types=[
            LayerAttentionType.LINEAR_ATTENTION,
            LayerAttentionType.LINEAR_ATTENTION,
            LayerAttentionType.LINEAR_ATTENTION,
            LayerAttentionType.FULL_ATTENTION,
        ],
        linear_attention_config=LinearAttentionConfig(
            num_key_heads=4,
            key_head_dim=32,
            num_value_heads=8,
            value_head_dim=32,
            conv_kernel_dim=4,
        ),
        normalization_type=NormalizationType.RMS_NORM,
        position_encoding=PositionEncodingType.ROTARY,
        dtype="bfloat16",
    )


@pytest.fixture
def gpu():
    """H100-80GB system constraints."""
    return SystemConstraints.from_gpu_spec("H100-80GB")


# ============================================================================
# LinearAttentionConfig unit tests
# ============================================================================


class TestLinearAttentionConfig:
    """Test the LinearAttentionConfig dataclass methods."""

    def test_total_dims(self, linear_attn_config):
        """Test total key/value dimension properties."""
        lac = linear_attn_config
        assert lac.total_key_dim == 16 * 128  # 2048
        assert lac.total_value_dim == 64 * 128  # 8192

    def test_state_size_constant(self, linear_attn_config):
        """State size must NOT depend on sequence length."""
        lac = linear_attn_config
        state_b1 = lac.get_state_size_bytes(batch_size=1)
        # State shape: batch * key_head_dim * num_value_heads * value_head_dim * bytes
        expected = 1 * 128 * 64 * 128 * 2  # bf16
        assert state_b1 == expected

        # Scaling with batch size
        state_b4 = lac.get_state_size_bytes(batch_size=4)
        assert state_b4 == 4 * state_b1

    def test_prefill_flops_linear_in_L(self, linear_attn_config):
        """Prefill FLOPs must scale linearly with sequence length."""
        lac = linear_attn_config
        flops_1k = lac.get_prefill_flops(
            sequence_length=1024, batch_size=1, num_query_heads=32, hidden_dim=4096
        )
        flops_2k = lac.get_prefill_flops(
            sequence_length=2048, batch_size=1, num_query_heads=32, hidden_dim=4096
        )
        # Linear: doubling L should double FLOPs (within floating point)
        ratio = flops_2k / flops_1k
        assert abs(ratio - 2.0) < 0.01, f"Expected ratio ~2.0, got {ratio}"

    def test_decode_flops_constant(self, linear_attn_config):
        """Decode FLOPs must NOT depend on context length (constant cost)."""
        lac = linear_attn_config
        # get_decode_flops signature has no context_length parameter
        flops = lac.get_decode_flops(batch_size=1, num_query_heads=32, hidden_dim=4096)
        assert flops > 0
        # Calling again gives same value — there's no context param
        assert lac.get_decode_flops(batch_size=1, num_query_heads=32, hidden_dim=4096) == flops

    def test_decode_flops_scales_with_batch(self, linear_attn_config):
        """Decode FLOPs must scale linearly with batch size."""
        lac = linear_attn_config
        flops_b1 = lac.get_decode_flops(batch_size=1, num_query_heads=32, hidden_dim=4096)
        flops_b4 = lac.get_decode_flops(batch_size=4, num_query_heads=32, hidden_dim=4096)
        ratio = flops_b4 / flops_b1
        assert abs(ratio - 4.0) < 0.01

    def test_decode_state_traffic(self, linear_attn_config):
        """State traffic = 2 * state_size (read + write)."""
        lac = linear_attn_config
        traffic = lac.get_decode_state_traffic(batch_size=1)
        state = lac.get_state_size_bytes(batch_size=1)
        assert traffic == 2 * state


# ============================================================================
# LLMArchitecture layer counting and KV cache tests
# ============================================================================


class TestLinearAttentionArchitecture:
    """Test LLMArchitecture methods with linear attention layers."""

    def test_qwen35_layer_counts(self, qwen35):
        """Verify layer type counts for Qwen3.5."""
        assert qwen35.get_num_linear_attention_layers() == 45
        assert qwen35.get_num_full_attention_layers() == 15
        assert qwen35.get_num_sliding_attention_layers() == 0
        assert qwen35.num_layers == 60

    def test_simple_layer_counts(self, simple_hybrid_model):
        """Verify layer type counts for simple hybrid model."""
        m = simple_hybrid_model
        assert m.get_num_linear_attention_layers() == 3
        assert m.get_num_full_attention_layers() == 1
        assert m.get_num_sliding_attention_layers() == 0

    def test_get_layer_attention_type(self, simple_hybrid_model):
        """get_layer_attention_type returns correct type per layer."""
        m = simple_hybrid_model
        assert m.get_layer_attention_type(0) == LayerAttentionType.LINEAR_ATTENTION
        assert m.get_layer_attention_type(1) == LayerAttentionType.LINEAR_ATTENTION
        assert m.get_layer_attention_type(2) == LayerAttentionType.LINEAR_ATTENTION
        assert m.get_layer_attention_type(3) == LayerAttentionType.FULL_ATTENTION

    def test_kv_cache_excludes_linear_layers(self, simple_hybrid_model):
        """KV cache should only cover full attention layers, not linear."""
        m = simple_hybrid_model
        # 1 full attention layer
        kv = m.get_kv_cache_size(batch_size=1, sequence_length=1024, bytes_per_element=2)

        # Expected: 1 full attention layer * 2 * 1 * (num_kv_heads * head_dim) * seq_len * bytes
        kv_dim = m.attention_config.num_key_value_heads * m.attention_config.head_dim  # 2 * 64 = 128
        expected = 1 * 2 * 1 * kv_dim * 1024 * 2  # 1 layer
        assert kv == expected, f"Expected {expected}, got {kv}"

    def test_kv_cache_no_scaling_with_linear_layers(self, simple_hybrid_model):
        """Adding more linear layers should not increase KV cache."""
        m = simple_hybrid_model
        kv_short = m.get_kv_cache_size(batch_size=1, sequence_length=1024, bytes_per_element=2)
        kv_long = m.get_kv_cache_size(batch_size=1, sequence_length=8192, bytes_per_element=2)

        # KV should scale with seq_len but only for 1 full attention layer
        ratio = kv_long / kv_short
        assert abs(ratio - 8.0) < 0.01  # 8192 / 1024 = 8x

    def test_linear_attention_state_size(self, simple_hybrid_model):
        """Linear attention state should be constant (no seq_len dependency)."""
        m = simple_hybrid_model
        state = m.get_linear_attention_state_size(batch_size=1, bytes_per_element=2)
        assert state > 0

        # State per layer: batch * key_head_dim * num_value_heads * value_head_dim * bytes
        lac = m.linear_attention_config
        state_per_layer = 1 * lac.key_head_dim * lac.num_value_heads * lac.value_head_dim * 2
        expected = 3 * state_per_layer  # 3 linear layers
        assert state == expected

    def test_qwen35_state_much_smaller_than_kv(self, qwen35):
        """For long contexts, linear attention state is much smaller than KV cache."""
        seq_len = 131072  # 128K
        kv_cache = qwen35.get_kv_cache_size(batch_size=1, sequence_length=seq_len, bytes_per_element=2)
        linear_state = qwen35.get_linear_attention_state_size(batch_size=1, bytes_per_element=2)

        # Linear state is constant, KV grows with seq_len
        assert linear_state > 0
        assert kv_cache > 0
        # At 128K context, the ratio should be very large
        ratio = kv_cache / linear_state
        assert ratio > 10, f"Expected KV/state ratio >> 10 at 128K, got {ratio:.1f}"

    def test_kv_cache_breakdown_includes_linear(self, qwen35):
        """KV cache breakdown should report linear attention state."""
        breakdown = qwen35.get_kv_cache_size_breakdown(
            batch_size=1, sequence_length=4096, bytes_per_element=2
        )
        assert 'linear_attention_state' in breakdown
        assert 'num_linear_layers' in breakdown
        assert breakdown['num_linear_layers'] == 45
        assert breakdown['num_full_layers'] == 15
        assert breakdown['linear_attention_state'] > 0
        assert breakdown['full_attention_layers'] > 0

        # Total = full + sliding + linear_state
        assert breakdown['total'] == (
            breakdown['full_attention_layers']
            + breakdown['sliding_attention_layers']
            + breakdown['linear_attention_state']
        )

    def test_total_inference_state_includes_linear(self, qwen35):
        """Total inference state should include linear attention state."""
        state = qwen35.get_total_inference_state_size(
            batch_size=1, sequence_length=4096, bytes_per_element=2
        )
        assert 'linear_attention_state' in state
        assert 'num_linear_attention_layers' in state
        assert state['linear_attention_state'] > 0
        assert state['num_linear_attention_layers'] == 45
        assert state['total'] == (
            state['kv_cache'] + state['mamba_state'] + state['linear_attention_state']
        )

    def test_memory_footprint_includes_linear_state(self, simple_hybrid_model):
        """Memory footprint should include linear attention state."""
        mem = simple_hybrid_model.get_memory_footprint(batch_size=1, sequence_length=1024)
        assert 'linear_attention_state' in mem
        assert mem['linear_attention_state'] > 0

    def test_model_available_in_all_models(self):
        """Qwen3.5-397B should be accessible via get_model."""
        model = get_model("qwen3.5-397b")
        assert model.model_name == "Qwen3.5-397B"
        assert model.linear_attention_config is not None


# ============================================================================
# Inference performance tests
# ============================================================================


class TestLinearAttentionPerformance:
    """Test prefill and decode performance with linear attention."""

    def test_prefill_flops_less_than_all_full(self, simple_hybrid_model, gpu):
        """
        Hybrid model with 3 linear + 1 full should use fewer prefill FLOPs
        than an equivalent model with all 4 full attention layers.
        """
        # Create an equivalent all-full model
        all_full = LLMArchitecture(
            model_name="test-all-full",
            model_family="test",
            version="1.0",
            num_layers=4,
            hidden_dim=simple_hybrid_model.hidden_dim,
            vocab_size=simple_hybrid_model.vocab_size,
            max_sequence_length=simple_hybrid_model.max_sequence_length,
            attention_config=simple_hybrid_model.attention_config,
            ffn_config=simple_hybrid_model.ffn_config,
            normalization_type=NormalizationType.RMS_NORM,
            position_encoding=PositionEncodingType.ROTARY,
            dtype="bfloat16",
        )

        perf_hybrid = InferencePerformance(simple_hybrid_model)
        perf_full = InferencePerformance(all_full)

        pc = ParallelismConfig()

        # Use a long sequence to amplify the O(L^2) vs O(L) difference
        seq_len = 4096
        hybrid_bd = perf_hybrid.calculate_prefill_compute_breakdown(1, seq_len, pc)
        full_bd = perf_full.calculate_prefill_compute_breakdown(1, seq_len, pc)

        # Hybrid should have fewer attention FLOPs (linear O(L) vs full O(L^2))
        assert hybrid_bd['attention'] < full_bd['attention'], (
            f"Hybrid attention FLOPs ({hybrid_bd['attention']:.2e}) should be < "
            f"full attention FLOPs ({full_bd['attention']:.2e})"
        )

    def test_prefill_flops_scale_linearly_for_linear_layers(self, gpu):
        """
        For a model that is ALL linear attention, prefill FLOPs should scale
        linearly with sequence length (no L^2 term).
        """
        model = LLMArchitecture(
            model_name="test-all-linear",
            model_family="test",
            version="1.0",
            num_layers=4,
            hidden_dim=512,
            vocab_size=1000,
            max_sequence_length=8192,
            attention_config=AttentionConfig(
                num_attention_heads=8,
                num_key_value_heads=2,
                attention_type=AttentionType.GROUPED_QUERY,
                head_dim=64,
            ),
            ffn_config=FFNConfig(
                intermediate_size=2048,
                activation=ActivationType.SILU,
            ),
            layer_types=[LayerAttentionType.LINEAR_ATTENTION] * 4,
            linear_attention_config=LinearAttentionConfig(
                num_key_heads=4, key_head_dim=32,
                num_value_heads=8, value_head_dim=32,
                conv_kernel_dim=4,
            ),
            dtype="bfloat16",
        )

        perf = InferencePerformance(model)
        pc = ParallelismConfig()

        bd_1k = perf.calculate_prefill_compute_breakdown(1, 1024, pc)
        bd_4k = perf.calculate_prefill_compute_breakdown(1, 4096, pc)

        # Total FLOPs should scale roughly linearly (FFN is O(L), linear attn is O(L))
        ratio = bd_4k['total'] / bd_1k['total']
        # With all O(L) terms, expect ratio ~4.0
        assert 3.5 < ratio < 4.5, f"Expected ~4x scaling, got {ratio:.2f}x"

        # Specifically attention FLOPs should scale linearly
        attn_ratio = bd_4k['attention'] / bd_1k['attention']
        assert 3.5 < attn_ratio < 4.5, f"Expected ~4x attention scaling, got {attn_ratio:.2f}x"

    def test_decode_flops_constant_for_linear_layers(self, gpu):
        """
        For all-linear-attention model, decode FLOPs should NOT grow with context.
        """
        model = LLMArchitecture(
            model_name="test-all-linear",
            model_family="test",
            version="1.0",
            num_layers=4,
            hidden_dim=512,
            vocab_size=1000,
            max_sequence_length=65536,
            attention_config=AttentionConfig(
                num_attention_heads=8,
                num_key_value_heads=2,
                attention_type=AttentionType.GROUPED_QUERY,
                head_dim=64,
            ),
            ffn_config=FFNConfig(
                intermediate_size=2048,
                activation=ActivationType.SILU,
            ),
            layer_types=[LayerAttentionType.LINEAR_ATTENTION] * 4,
            linear_attention_config=LinearAttentionConfig(
                num_key_heads=4, key_head_dim=32,
                num_value_heads=8, value_head_dim=32,
                conv_kernel_dim=4,
            ),
            dtype="bfloat16",
        )

        perf = InferencePerformance(model)
        pc = ParallelismConfig()

        bd_1k = perf._calculate_decode_step_compute_breakdown(1, 1024, pc)
        bd_32k = perf._calculate_decode_step_compute_breakdown(1, 32768, pc)

        # Attention FLOPs should be constant (no context dependency)
        assert bd_1k['attention'] == bd_32k['attention'], (
            f"Linear attention decode FLOPs should be constant: "
            f"{bd_1k['attention']:.0f} vs {bd_32k['attention']:.0f}"
        )
        # Total also constant (FFN is O(1) per token already)
        assert bd_1k['total'] == bd_32k['total']

    def test_hybrid_decode_grows_slower(self, simple_hybrid_model, gpu):
        """
        Hybrid model (3 linear + 1 full) decode FLOPs should grow more slowly
        with context than all-full model.
        """
        all_full = LLMArchitecture(
            model_name="test-all-full",
            model_family="test",
            version="1.0",
            num_layers=4,
            hidden_dim=simple_hybrid_model.hidden_dim,
            vocab_size=simple_hybrid_model.vocab_size,
            max_sequence_length=simple_hybrid_model.max_sequence_length,
            attention_config=simple_hybrid_model.attention_config,
            ffn_config=simple_hybrid_model.ffn_config,
            dtype="bfloat16",
        )

        perf_hybrid = InferencePerformance(simple_hybrid_model)
        perf_full = InferencePerformance(all_full)
        pc = ParallelismConfig()

        ctx_short = 1024
        ctx_long = 32768

        hybrid_short = perf_hybrid._calculate_decode_step_compute_breakdown(1, ctx_short, pc)
        hybrid_long = perf_hybrid._calculate_decode_step_compute_breakdown(1, ctx_long, pc)
        full_short = perf_full._calculate_decode_step_compute_breakdown(1, ctx_short, pc)
        full_long = perf_full._calculate_decode_step_compute_breakdown(1, ctx_long, pc)

        hybrid_growth = hybrid_long['attention'] / hybrid_short['attention']
        full_growth = full_long['attention'] / full_short['attention']

        # Full attention attention FLOPs scale ~32x (32768/1024)
        # Hybrid: only 1/4 of layers scale, 3/4 are constant → growth should be much less
        assert hybrid_growth < full_growth, (
            f"Hybrid growth ({hybrid_growth:.1f}x) should be < full growth ({full_growth:.1f}x)"
        )

    def test_qwen35_prefill_resources(self, qwen35, gpu):
        """Test that full prefill pipeline runs for Qwen3.5."""
        perf = InferencePerformance(qwen35)
        pc = ParallelismConfig(
            parallelism_type=ParallelismType.TENSOR_PARALLEL,
            tensor_parallel_size=8,
        )
        result = perf.calculate_prefill_resources(
            batch_size=1,
            sequence_length=2048,
            time_to_first_token=1.0,
            num_gpus=8,
            parallelism_config=pc,
        )
        assert result.compute_per_gpu > 0
        assert result.memory_model_weights > 0

    def test_qwen35_decode_performance(self, qwen35, gpu):
        """Test that full decode pipeline runs for Qwen3.5."""
        perf = InferencePerformance(qwen35)
        pc = ParallelismConfig(
            parallelism_type=ParallelismType.TENSOR_PARALLEL,
            tensor_parallel_size=8,
        )
        result = perf.calculate_decode_performance(
            system_constraints=gpu,
            batch_size=1,
            prefill_length=2048,
            output_length=50,
            parallelism_config=pc,
        )
        assert result.tokens_per_second_per_user > 0
        assert result.avg_step_time > 0

    def test_decode_memory_traffic_linear_constant(self, gpu):
        """
        For an all-linear model, decode memory traffic for attention state
        should NOT grow with context.
        """
        model = LLMArchitecture(
            model_name="test-all-linear",
            model_family="test",
            version="1.0",
            num_layers=4,
            hidden_dim=512,
            vocab_size=1000,
            max_sequence_length=65536,
            attention_config=AttentionConfig(
                num_attention_heads=8,
                num_key_value_heads=2,
                attention_type=AttentionType.GROUPED_QUERY,
                head_dim=64,
            ),
            ffn_config=FFNConfig(
                intermediate_size=2048,
                activation=ActivationType.SILU,
            ),
            layer_types=[LayerAttentionType.LINEAR_ATTENTION] * 4,
            linear_attention_config=LinearAttentionConfig(
                num_key_heads=4, key_head_dim=32,
                num_value_heads=8, value_head_dim=32,
                conv_kernel_dim=4,
            ),
            dtype="bfloat16",
        )

        perf = InferencePerformance(model)
        pc = ParallelismConfig()

        traffic_1k = perf._calculate_decode_step_memory_traffic(1, 1024, pc)
        traffic_32k = perf._calculate_decode_step_memory_traffic(1, 32768, pc)

        # Traffic should be identical (no KV cache, only fixed state + weights)
        assert traffic_1k == traffic_32k, (
            f"All-linear model decode traffic should be constant: "
            f"{traffic_1k:.0f} vs {traffic_32k:.0f}"
        )

    def test_kernel_count_linear_vs_full(self, simple_hybrid_model):
        """Linear attention layers should have fewer kernel launches than full."""
        # Hybrid: 3 linear + 1 full
        perf_hybrid = InferencePerformance(simple_hybrid_model)
        pc = ParallelismConfig()
        kernels_hybrid = perf_hybrid.calculate_num_kernel_launches(pc)

        # All-full equivalent
        all_full = LLMArchitecture(
            model_name="test-all-full",
            model_family="test",
            version="1.0",
            num_layers=4,
            hidden_dim=simple_hybrid_model.hidden_dim,
            vocab_size=simple_hybrid_model.vocab_size,
            max_sequence_length=simple_hybrid_model.max_sequence_length,
            attention_config=simple_hybrid_model.attention_config,
            ffn_config=simple_hybrid_model.ffn_config,
            dtype="bfloat16",
        )
        perf_full = InferencePerformance(all_full)
        kernels_full = perf_full.calculate_num_kernel_launches(pc)

        # Linear layers have fewer kernels (no softmax kernel)
        assert kernels_hybrid <= kernels_full, (
            f"Hybrid kernels ({kernels_hybrid}) should be <= full kernels ({kernels_full})"
        )


# ============================================================================
# Edge cases
# ============================================================================


class TestLinearAttentionEdgeCases:
    """Edge cases and boundary conditions."""

    def test_no_linear_config_returns_zero_state(self):
        """Model without linear_attention_config returns 0 state size."""
        model = LLMArchitecture(
            model_name="test-no-linear",
            model_family="test",
            version="1.0",
            num_layers=4,
            hidden_dim=512,
            vocab_size=1000,
            attention_config=AttentionConfig(num_attention_heads=8, head_dim=64),
            ffn_config=FFNConfig(intermediate_size=2048),
            dtype="bfloat16",
        )
        assert model.get_num_linear_attention_layers() == 0
        assert model.get_linear_attention_state_size(batch_size=1) == 0

    def test_layer_types_none_returns_zero_linear(self):
        """Model with layer_types=None has 0 linear attention layers."""
        model = LLMArchitecture(
            model_name="test",
            model_family="test",
            version="1.0",
            num_layers=4,
            hidden_dim=512,
            vocab_size=1000,
            attention_config=AttentionConfig(num_attention_heads=8, head_dim=64),
            ffn_config=FFNConfig(intermediate_size=2048),
            dtype="bfloat16",
        )
        assert model.layer_types is None
        assert model.get_num_linear_attention_layers() == 0

    def test_batch_size_1_sequence_length_1(self, simple_hybrid_model):
        """Minimum batch and sequence should work."""
        perf = InferencePerformance(simple_hybrid_model)
        pc = ParallelismConfig()
        bd = perf.calculate_prefill_compute_breakdown(1, 1, pc)
        assert bd['total'] > 0
        assert bd['attention'] > 0

    def test_qwen35_summary_contains_linear_info(self, qwen35):
        """Summary should mention linear attention config."""
        summary = qwen35.summary()
        assert "Linear" in summary
        assert "45" in summary  # 45 linear layers
        assert "15" in summary  # 15 full layers

    def test_qwen35_layer_pattern(self, qwen35):
        """Verify the 3-linear + 1-full repeating pattern."""
        for i in range(60):
            expected = (
                LayerAttentionType.FULL_ATTENTION
                if (i + 1) % 4 == 0
                else LayerAttentionType.LINEAR_ATTENTION
            )
            actual = qwen35.get_layer_attention_type(i)
            assert actual == expected, f"Layer {i}: expected {expected}, got {actual}"
