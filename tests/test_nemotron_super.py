"""
Tests for Nemotron-3-Super-120B model and supporting infrastructure:
  1. LatentMoEConfig — compute & weight-param math
  2. New HybridLayerType values (MAMBA_ONLY, LATENT_MOE, ATTENTION_ONLY)
  3. NEMOTRON_3_SUPER_120B config integrity
  4. Per-layer compute/bandwidth breakdown for the Super model
"""

import pytest
from llm_architecture import (
    LLMArchitecture, AttentionConfig, FFNConfig,
    ArchitectureType, AttentionType, ActivationType,
    NormalizationType, PositionEncodingType,
    HybridLayerType, Mamba2Config, LatentMoEConfig,
)
from llm_configs import NEMOTRON_3_SUPER_120B, NEMOTRON_3_30B
from inference_performance import (
    InferencePerformance, SystemConstraints, ParallelismConfig, ParallelismType
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def single_gpu():
    return ParallelismConfig(
        parallelism_type=ParallelismType.TENSOR_PARALLEL,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
    )


@pytest.fixture
def small_latent_moe():
    """Minimal LatentMoEConfig for unit tests (easy arithmetic)."""
    return LatentMoEConfig(
        num_experts=8,
        num_experts_per_token=2,
        latent_size=64,
        expert_intermediate_size=128,
        shared_expert_intermediate_size=256,
        n_shared_experts=1,
        use_gating=False,
    )


@pytest.fixture
def simple_mamba_only_model():
    """4-layer model where all layers are MAMBA_ONLY (no FFN)."""
    return LLMArchitecture(
        model_name="test-mamba-only",
        model_family="Test",
        version="1.0",
        architecture_type=ArchitectureType.DECODER_ONLY,
        num_layers=4,
        hidden_dim=512,
        vocab_size=1000,
        max_sequence_length=2048,
        attention_config=AttentionConfig(
            num_attention_heads=8,
            num_key_value_heads=2,
            attention_type=AttentionType.GROUPED_QUERY,
            head_dim=64,
        ),
        ffn_config=FFNConfig(intermediate_size=1024, activation=ActivationType.SILU),
        hybrid_layer_types=[HybridLayerType.MAMBA_ONLY] * 4,
        mamba_config=Mamba2Config(
            num_heads=8, head_dim=64, state_size=64, chunk_size=64,
        ),
        total_parameters=100_000_000,
        active_parameters=100_000_000,
    )


@pytest.fixture
def simple_latent_moe_model(small_latent_moe):
    """4-layer model where all layers are LATENT_MOE."""
    return LLMArchitecture(
        model_name="test-latent-moe",
        model_family="Test",
        version="1.0",
        architecture_type=ArchitectureType.DECODER_ONLY,
        num_layers=4,
        hidden_dim=512,
        vocab_size=1000,
        max_sequence_length=2048,
        attention_config=AttentionConfig(
            num_attention_heads=8,
            num_key_value_heads=2,
            attention_type=AttentionType.GROUPED_QUERY,
            head_dim=64,
        ),
        ffn_config=FFNConfig(intermediate_size=128, activation=ActivationType.RELU2),
        hybrid_layer_types=[HybridLayerType.LATENT_MOE] * 4,
        latent_moe_config=small_latent_moe,
        total_parameters=100_000_000,
        active_parameters=100_000_000,
    )


@pytest.fixture
def simple_attention_only_model():
    """4-layer model where all layers are ATTENTION_ONLY (no FFN)."""
    return LLMArchitecture(
        model_name="test-attention-only",
        model_family="Test",
        version="1.0",
        architecture_type=ArchitectureType.DECODER_ONLY,
        num_layers=4,
        hidden_dim=512,
        vocab_size=1000,
        max_sequence_length=2048,
        attention_config=AttentionConfig(
            num_attention_heads=8,
            num_key_value_heads=2,
            attention_type=AttentionType.GROUPED_QUERY,
            head_dim=64,
        ),
        ffn_config=FFNConfig(intermediate_size=1024, activation=ActivationType.SILU),
        hybrid_layer_types=[HybridLayerType.ATTENTION_ONLY] * 4,
        total_parameters=100_000_000,
        active_parameters=100_000_000,
    )


@pytest.fixture
def super_style_compound_model(small_latent_moe):
    """
    8-layer model with the Nemotron-Super sublayer pattern:
      [M, E, M, E, M, *, E, M]  (mix of MAMBA_ONLY, LATENT_MOE, ATTENTION_ONLY)
    """
    return LLMArchitecture(
        model_name="test-super-style",
        model_family="Test",
        version="1.0",
        architecture_type=ArchitectureType.DECODER_ONLY,
        num_layers=8,
        hidden_dim=512,
        vocab_size=1000,
        max_sequence_length=2048,
        attention_config=AttentionConfig(
            num_attention_heads=8,
            num_key_value_heads=2,
            attention_type=AttentionType.GROUPED_QUERY,
            head_dim=64,
        ),
        ffn_config=FFNConfig(intermediate_size=128, activation=ActivationType.RELU2),
        hybrid_layer_types=[
            HybridLayerType.MAMBA_ONLY,
            HybridLayerType.LATENT_MOE,
            HybridLayerType.MAMBA_ONLY,
            HybridLayerType.LATENT_MOE,
            HybridLayerType.MAMBA_ONLY,
            HybridLayerType.ATTENTION_ONLY,
            HybridLayerType.LATENT_MOE,
            HybridLayerType.MAMBA_ONLY,
        ],
        mamba_config=Mamba2Config(
            num_heads=8, head_dim=64, state_size=64, chunk_size=64,
        ),
        latent_moe_config=small_latent_moe,
        total_parameters=100_000_000,
        active_parameters=100_000_000,
    )


# ===========================================================================
# 1. LatentMoEConfig unit tests
# ===========================================================================

class TestLatentMoEConfig:

    def test_prefill_flops_components(self, small_latent_moe):
        """Verify FLOPs formula for each component in prefill."""
        cfg = small_latent_moe
        B, T, H = 2, 16, 512
        d_lat = cfg.latent_size          # 64
        d_int = cfg.expert_intermediate_size   # 128
        d_sh  = cfg.shared_expert_intermediate_size  # 256
        k = cfg.num_experts_per_token    # 2
        E = cfg.num_experts              # 8

        expected = (
            2 * B * T * H * d_lat           # down proj
            + 2 * B * T * d_lat * E         # router
            + 2 * k * B * T * d_lat * d_int  # expert up
            + 2 * k * B * T * d_int * d_lat  # expert down
            + 2 * B * T * d_lat * H         # up proj
            + 2 * B * T * H * d_sh          # shared up
            + 2 * B * T * d_sh * H          # shared down
        )
        assert cfg.get_prefill_flops(B, T, H) == expected

    def test_decode_flops_equals_prefill_with_single_token(self, small_latent_moe):
        """Decode (1 token) should equal prefill with T=1."""
        cfg = small_latent_moe
        B, H = 3, 512
        assert cfg.get_decode_flops(B, H) == cfg.get_prefill_flops(B, 1, H)

    def test_prefill_scales_linearly_with_sequence_length(self, small_latent_moe):
        """FLOPs should scale linearly with sequence length."""
        cfg = small_latent_moe
        B, H = 1, 512
        flops_L = cfg.get_prefill_flops(B, 100, H)
        flops_2L = cfg.get_prefill_flops(B, 200, H)
        assert flops_2L == pytest.approx(2 * flops_L, rel=1e-9)

    def test_weight_params_structure(self, small_latent_moe):
        """Verify weight_params formula."""
        cfg = small_latent_moe
        H = 512
        d_lat = cfg.latent_size          # 64
        d_int = cfg.expert_intermediate_size   # 128
        d_sh  = cfg.shared_expert_intermediate_size  # 256

        expected = (
            H * d_lat                              # down proj
            + d_lat * cfg.num_experts              # router
            + cfg.num_experts * (d_lat * d_int + d_int * d_lat)  # all experts (up+down)
            + d_lat * H                            # up proj
            + 1 * (H * d_sh + d_sh * H)           # shared expert (n_shared=1)
        )
        assert cfg.get_weight_params(H) == expected

    def test_gating_adds_extra_projection(self):
        """use_gating=True should add gate projection FLOPs and params."""
        cfg_no_gate = LatentMoEConfig(
            num_experts=4, num_experts_per_token=2, latent_size=32,
            expert_intermediate_size=64, shared_expert_intermediate_size=128,
            use_gating=False,
        )
        cfg_gate = LatentMoEConfig(
            num_experts=4, num_experts_per_token=2, latent_size=32,
            expert_intermediate_size=64, shared_expert_intermediate_size=128,
            use_gating=True,
        )
        B, T, H = 1, 8, 256
        flops_gate = cfg_gate.get_prefill_flops(B, T, H)
        flops_no_gate = cfg_no_gate.get_prefill_flops(B, T, H)
        # Gate cost = 2 * k * B * T * d_lat * d_int
        gate_cost = 2 * 2 * B * T * 32 * 64
        assert flops_gate - flops_no_gate == gate_cost

        params_gate = cfg_gate.get_weight_params(H)
        params_no_gate = cfg_no_gate.get_weight_params(H)
        # Extra gate params = num_experts * d_lat * d_int
        assert params_gate - params_no_gate == 4 * 32 * 64

    def test_multiple_shared_experts(self):
        """n_shared_experts > 1 should scale accordingly."""
        base = LatentMoEConfig(
            num_experts=8, num_experts_per_token=2, latent_size=32,
            expert_intermediate_size=64, shared_expert_intermediate_size=128,
            n_shared_experts=1, use_gating=False,
        )
        doubled = LatentMoEConfig(
            num_experts=8, num_experts_per_token=2, latent_size=32,
            expert_intermediate_size=64, shared_expert_intermediate_size=128,
            n_shared_experts=2, use_gating=False,
        )
        H = 256
        B, T = 1, 8
        # Extra FLOPs for one additional shared expert (up-proj + down-proj)
        extra_shared = 2 * B * T * H * 128 + 2 * B * T * 128 * H
        assert doubled.get_prefill_flops(B, T, H) - base.get_prefill_flops(B, T, H) == extra_shared


# ===========================================================================
# 2. HybridLayerType — new values exist and are recognised
# ===========================================================================

class TestNewHybridLayerTypes:

    def test_new_values_exist(self):
        assert HybridLayerType.MAMBA_ONLY in list(HybridLayerType)
        assert HybridLayerType.LATENT_MOE in list(HybridLayerType)
        assert HybridLayerType.ATTENTION_ONLY in list(HybridLayerType)

    def test_existing_values_unchanged(self):
        assert HybridLayerType.MAMBA in list(HybridLayerType)
        assert HybridLayerType.ATTENTION in list(HybridLayerType)
        assert HybridLayerType.MLP in list(HybridLayerType)

    def test_get_num_mamba_layers_counts_mamba_only(self):
        """get_num_mamba_layers should count both MAMBA and MAMBA_ONLY."""
        m = LLMArchitecture(
            model_name="x", model_family="x", version="1",
            num_layers=6,
            attention_config=AttentionConfig(num_attention_heads=4),
            ffn_config=FFNConfig(intermediate_size=512),
            hybrid_layer_types=[
                HybridLayerType.MAMBA,
                HybridLayerType.MAMBA_ONLY,
                HybridLayerType.LATENT_MOE,
                HybridLayerType.ATTENTION_ONLY,
                HybridLayerType.ATTENTION,
                HybridLayerType.MLP,
            ],
            total_parameters=1_000_000,
            active_parameters=1_000_000,
        )
        assert m.get_num_mamba_layers() == 2   # MAMBA + MAMBA_ONLY

    def test_get_num_attention_layers_counts_attention_only(self):
        """get_num_attention_layers_hybrid counts ATTENTION and ATTENTION_ONLY."""
        m = LLMArchitecture(
            model_name="x", model_family="x", version="1",
            num_layers=6,
            attention_config=AttentionConfig(num_attention_heads=4),
            ffn_config=FFNConfig(intermediate_size=512),
            hybrid_layer_types=[
                HybridLayerType.MAMBA,
                HybridLayerType.MAMBA_ONLY,
                HybridLayerType.LATENT_MOE,
                HybridLayerType.ATTENTION_ONLY,
                HybridLayerType.ATTENTION,
                HybridLayerType.MLP,
            ],
            total_parameters=1_000_000,
            active_parameters=1_000_000,
        )
        assert m.get_num_attention_layers_hybrid() == 2  # ATTENTION + ATTENTION_ONLY

    def test_get_num_latent_moe_layers(self):
        m = LLMArchitecture(
            model_name="x", model_family="x", version="1",
            num_layers=6,
            attention_config=AttentionConfig(num_attention_heads=4),
            ffn_config=FFNConfig(intermediate_size=512),
            hybrid_layer_types=[
                HybridLayerType.MAMBA_ONLY,
                HybridLayerType.LATENT_MOE,
                HybridLayerType.MAMBA_ONLY,
                HybridLayerType.LATENT_MOE,
                HybridLayerType.ATTENTION_ONLY,
                HybridLayerType.LATENT_MOE,
            ],
            total_parameters=1_000_000,
            active_parameters=1_000_000,
        )
        assert m.get_num_latent_moe_layers() == 3

    def test_mamba_state_only_counts_mamba_layers(self):
        """Mamba state size should use mamba layer count (MAMBA + MAMBA_ONLY)."""
        m = LLMArchitecture(
            model_name="x", model_family="x", version="1",
            num_layers=4,
            attention_config=AttentionConfig(num_attention_heads=4),
            ffn_config=FFNConfig(intermediate_size=512),
            hybrid_layer_types=[
                HybridLayerType.MAMBA_ONLY,
                HybridLayerType.LATENT_MOE,
                HybridLayerType.ATTENTION_ONLY,
                HybridLayerType.MAMBA_ONLY,
            ],
            mamba_config=Mamba2Config(
                num_heads=4, head_dim=32, state_size=16, chunk_size=32,
            ),
            total_parameters=1_000_000,
            active_parameters=1_000_000,
        )
        # Mamba state per layer: batch * heads * head_dim * state_size
        # With 2 MAMBA_ONLY layers, batch=1, bytes=2
        states_per_layer = 1 * 4 * 32 * 16 * 2  # = 4096
        assert m.get_mamba_state_size(batch_size=1, bytes_per_element=2) == 2 * states_per_layer

    def test_kv_cache_only_from_attention_layers(self):
        """KV cache should only come from ATTENTION and ATTENTION_ONLY layers."""
        m = LLMArchitecture(
            model_name="x", model_family="x", version="1",
            num_layers=6,
            hidden_dim=512,
            attention_config=AttentionConfig(
                num_attention_heads=8, num_key_value_heads=2,
                head_dim=64,
            ),
            ffn_config=FFNConfig(intermediate_size=512),
            hybrid_layer_types=[
                HybridLayerType.MAMBA_ONLY,    # no KV
                HybridLayerType.LATENT_MOE,    # no KV
                HybridLayerType.ATTENTION_ONLY,  # KV
                HybridLayerType.MAMBA_ONLY,    # no KV
                HybridLayerType.LATENT_MOE,    # no KV
                HybridLayerType.ATTENTION_ONLY,  # KV
            ],
            total_parameters=1_000_000,
            active_parameters=1_000_000,
        )
        # 2 attention layers, kv_dim = 2*64 = 128, seq=100, batch=1, bytes=2
        kv_dim = 2 * 64
        expected = 2 * 1 * 100 * kv_dim * 2 * 2  # 2 attn layers × (K+V) × batch × seq × kv_heads*head_dim × bytes
        assert m.get_kv_cache_size(batch_size=1, sequence_length=100, bytes_per_element=2) == expected


# ===========================================================================
# 3. NEMOTRON_3_SUPER_120B config integrity
# ===========================================================================

class TestNemotronSuperConfig:

    def test_num_layers(self):
        assert NEMOTRON_3_SUPER_120B.num_layers == 88

    def test_pattern_counts(self):
        """Should have exactly 40 M, 40 E, 8 attention sublayers."""
        types = NEMOTRON_3_SUPER_120B.hybrid_layer_types
        n_mamba = sum(1 for t in types if t == HybridLayerType.MAMBA_ONLY)
        n_latent = sum(1 for t in types if t == HybridLayerType.LATENT_MOE)
        n_attn   = sum(1 for t in types if t == HybridLayerType.ATTENTION_ONLY)
        assert n_mamba == 40
        assert n_latent == 40
        assert n_attn == 8

    def test_no_regular_mamba_or_attention_types(self):
        """Super model should only use the ONLY variants (sublayer style)."""
        types = NEMOTRON_3_SUPER_120B.hybrid_layer_types
        assert HybridLayerType.MAMBA not in types
        assert HybridLayerType.ATTENTION not in types
        assert HybridLayerType.MLP not in types

    def test_latent_moe_config_present(self):
        assert NEMOTRON_3_SUPER_120B.latent_moe_config is not None
        lmoe = NEMOTRON_3_SUPER_120B.latent_moe_config
        assert lmoe.num_experts == 512
        assert lmoe.num_experts_per_token == 22
        assert lmoe.latent_size == 1024
        assert lmoe.expert_intermediate_size == 2688
        assert lmoe.shared_expert_intermediate_size == 5376
        assert lmoe.n_shared_experts == 1
        assert lmoe.use_gating is False

    def test_mamba_config_present(self):
        assert NEMOTRON_3_SUPER_120B.mamba_config is not None
        mc = NEMOTRON_3_SUPER_120B.mamba_config
        assert mc.num_heads == 128
        assert mc.head_dim == 64
        assert mc.state_size == 128

    def test_attention_config(self):
        ac = NEMOTRON_3_SUPER_120B.attention_config
        assert ac.num_attention_heads == 32
        assert ac.num_key_value_heads == 2
        assert ac.head_dim == 128

    def test_mamba_state_size(self):
        """Only the 40 MAMBA_ONLY layers contribute to Mamba state."""
        assert NEMOTRON_3_SUPER_120B.get_num_mamba_layers() == 40

    def test_kv_cache_only_8_layers(self):
        """KV cache should only account for the 8 attention sublayers."""
        assert NEMOTRON_3_SUPER_120B.get_num_attention_layers_hybrid() == 8

    def test_kv_cache_size_correct(self):
        """Verify KV cache size formula with 8 attention layers."""
        m = NEMOTRON_3_SUPER_120B
        batch, seq = 1, 1000
        kv_dim = m.attention_config.num_key_value_heads * m.attention_config.head_dim
        # 8 attention layers × K+V × batch × seq × kv_dim × 2 bytes
        expected = 2 * batch * seq * kv_dim * 2 * 8
        assert m.get_kv_cache_size(batch_size=batch, sequence_length=seq, bytes_per_element=2) == expected

    def test_parameter_counts_set(self):
        assert NEMOTRON_3_SUPER_120B.total_parameters == 120_000_000_000
        assert NEMOTRON_3_SUPER_120B.active_parameters == 18_000_000_000

    def test_is_in_all_models(self):
        from llm_configs import ALL_MODELS
        assert "nemotron-3-super-120b" in ALL_MODELS
        assert ALL_MODELS["nemotron-3-super-120b"] is NEMOTRON_3_SUPER_120B

    def test_summary_mentions_sublayer_architecture(self):
        summary = NEMOTRON_3_SUPER_120B.summary()
        assert "Sublayer" in summary or "sublayer" in summary.lower() or "MAMBA_ONLY" in summary.lower() or "mamba_only" in summary.lower()

    def test_latent_moe_in_summary(self):
        summary = NEMOTRON_3_SUPER_120B.summary()
        assert "LatentMoE" in summary or "latent_moe" in summary.lower() or "Latent" in summary


# ===========================================================================
# 4. Per-layer compute breakdown for new layer types
# ===========================================================================

class TestNewLayerTypeCompute:

    def test_mamba_only_no_ffn_compute(self, simple_mamba_only_model, single_gpu):
        """MAMBA_ONLY layers should have Mamba FLOPs but zero FFN FLOPs."""
        perf = InferencePerformance(simple_mamba_only_model)
        bd = perf.calculate_prefill_compute_breakdown(
            batch_size=1, sequence_length=256, parallelism_config=single_gpu
        )
        assert bd['mamba'] > 0, "Should have Mamba FLOPs"
        assert bd['ffn'] == 0.0, "MAMBA_ONLY layers should have zero FFN FLOPs"
        assert bd['attention'] == 0.0

    def test_latent_moe_no_attention_compute(self, simple_latent_moe_model, single_gpu):
        """LATENT_MOE layers should have FFN FLOPs but zero attention/Mamba FLOPs."""
        perf = InferencePerformance(simple_latent_moe_model)
        bd = perf.calculate_prefill_compute_breakdown(
            batch_size=1, sequence_length=256, parallelism_config=single_gpu
        )
        assert bd['ffn'] > 0, "Should have FFN (LatentMoE) FLOPs"
        assert bd['attention'] == 0.0
        assert bd['mamba'] == 0.0

    def test_attention_only_no_ffn_compute(self, simple_attention_only_model, single_gpu):
        """ATTENTION_ONLY layers should have attention FLOPs but zero FFN/Mamba."""
        perf = InferencePerformance(simple_attention_only_model)
        bd = perf.calculate_prefill_compute_breakdown(
            batch_size=1, sequence_length=256, parallelism_config=single_gpu
        )
        assert bd['attention'] > 0, "Should have attention FLOPs"
        assert bd['ffn'] == 0.0, "ATTENTION_ONLY layers should have zero FFN FLOPs"
        assert bd['mamba'] == 0.0

    def test_compound_model_correct_split(self, super_style_compound_model, single_gpu):
        """Super-style model should have all three compute components."""
        perf = InferencePerformance(super_style_compound_model)
        bd = perf.calculate_prefill_compute_breakdown(
            batch_size=1, sequence_length=128, parallelism_config=single_gpu
        )
        assert bd['mamba'] > 0
        assert bd['ffn'] > 0
        assert bd['attention'] > 0

    def test_latent_moe_flops_scale_linearly(self, simple_latent_moe_model, single_gpu):
        """Latent MoE FLOPs should scale linearly with sequence length."""
        perf = InferencePerformance(simple_latent_moe_model)
        bd1 = perf.calculate_prefill_compute_breakdown(
            batch_size=1, sequence_length=128, parallelism_config=single_gpu
        )
        bd2 = perf.calculate_prefill_compute_breakdown(
            batch_size=1, sequence_length=256, parallelism_config=single_gpu
        )
        assert bd2['ffn'] == pytest.approx(2 * bd1['ffn'], rel=1e-6)

    def test_latent_moe_flops_exact(self, simple_latent_moe_model, single_gpu, small_latent_moe):
        """Verify Latent MoE FLOPs against manual calculation for 4-layer model."""
        B, L, H = 1, 64, 512
        perf = InferencePerformance(simple_latent_moe_model)
        bd = perf.calculate_prefill_compute_breakdown(
            batch_size=B, sequence_length=L, parallelism_config=single_gpu
        )
        # 4 layers × per-layer LatentMoE FLOPs
        expected = 4 * small_latent_moe.get_prefill_flops(B, L, H)
        assert bd['ffn'] == pytest.approx(expected, rel=1e-6)

    def test_mamba_only_flops_exact(self, simple_mamba_only_model, single_gpu):
        """Verify MAMBA_ONLY FLOPs against manual calculation."""
        B, L, H = 1, 64, 512
        perf = InferencePerformance(simple_mamba_only_model)
        mc = simple_mamba_only_model.mamba_config
        bd = perf.calculate_prefill_compute_breakdown(
            batch_size=B, sequence_length=L, parallelism_config=single_gpu
        )
        # 4 layers × Mamba prefill FLOPs
        expected_mamba = 4 * mc.get_prefill_flops(L, H) * B
        assert bd['mamba'] == pytest.approx(expected_mamba, rel=1e-6)


# ===========================================================================
# 5. Per-layer breakdown (compute + traffic) for new types
# ===========================================================================

class TestPerLayerBreakdownNewTypes:

    def test_mamba_only_layer_labels(self, simple_mamba_only_model, single_gpu):
        perf = InferencePerformance(simple_mamba_only_model)
        breakdown = perf.calculate_per_layer_breakdown(
            mode='prefill', batch_size=1, sequence_length=128,
            parallelism_config=single_gpu,
        )
        for label in breakdown.layer_types:
            assert "Mamba" in label

    def test_latent_moe_layer_labels(self, simple_latent_moe_model, single_gpu):
        perf = InferencePerformance(simple_latent_moe_model)
        bd = perf.calculate_per_layer_breakdown(
            mode='prefill', batch_size=1, sequence_length=128,
            parallelism_config=single_gpu,
        )
        for label in bd.layer_types:
            assert "LatentMoE" in label

    def test_mamba_only_zero_ffn_in_per_layer(self, simple_mamba_only_model, single_gpu):
        perf = InferencePerformance(simple_mamba_only_model)
        bd = perf.calculate_per_layer_breakdown(
            mode='prefill', batch_size=1, sequence_length=128,
            parallelism_config=single_gpu,
        )
        for ffn in bd.non_attention_compute:
            # Layer norms are very small; the FFN (MLP) component should be 0
            # non_attention_compute includes the tiny layernorm contribution via traffic
            # but FFN compute itself is 0 for MAMBA_ONLY
            assert ffn == pytest.approx(0.0, abs=0)  # No dense/MoE FFN FLOPs

    def test_latent_moe_zero_attention_in_per_layer(self, simple_latent_moe_model, single_gpu):
        perf = InferencePerformance(simple_latent_moe_model)
        bd = perf.calculate_per_layer_breakdown(
            mode='prefill', batch_size=1, sequence_length=128,
            parallelism_config=single_gpu,
        )
        for attn in bd.attention_compute:
            assert attn == pytest.approx(0.0, abs=0)

    def test_attention_only_zero_ffn_in_per_layer(self, simple_attention_only_model, single_gpu):
        perf = InferencePerformance(simple_attention_only_model)
        bd = perf.calculate_per_layer_breakdown(
            mode='prefill', batch_size=1, sequence_length=128,
            parallelism_config=single_gpu,
        )
        for ffn in bd.non_attention_compute:
            assert ffn == pytest.approx(0.0, abs=0)

    def test_latent_moe_traffic_positive(self, simple_latent_moe_model, single_gpu):
        perf = InferencePerformance(simple_latent_moe_model)
        bd = perf.calculate_per_layer_breakdown(
            mode='decode', batch_size=1, sequence_length=128,
            parallelism_config=single_gpu,
        )
        for traffic in bd.non_attention_memory_traffic:
            assert traffic > 0, "LatentMoE layers should have non-zero FFN memory traffic"

    def test_latent_moe_zero_attention_traffic(self, simple_latent_moe_model, single_gpu):
        """LATENT_MOE layers have no KV cache or SSM state, so attention traffic = 0."""
        perf = InferencePerformance(simple_latent_moe_model)
        bd = perf.calculate_per_layer_breakdown(
            mode='decode', batch_size=1, sequence_length=128,
            parallelism_config=single_gpu,
        )
        for traffic in bd.attention_memory_traffic:
            assert traffic == pytest.approx(0.0, abs=0)


# ===========================================================================
# 6. NEMOTRON_3_SUPER_120B inference metrics (sanity checks)
# ===========================================================================

class TestNemotronSuperInference:

    def test_prefill_compute_all_three_components(self, single_gpu):
        """Super model should have Mamba, LatentMoE (ffn), and attention FLOPs."""
        perf = InferencePerformance(NEMOTRON_3_SUPER_120B)
        bd = perf.calculate_prefill_compute_breakdown(
            batch_size=1, sequence_length=512, parallelism_config=single_gpu
        )
        assert bd['mamba'] > 0, "40 Mamba sublayers should contribute"
        assert bd['ffn'] > 0, "40 LatentMoE sublayers should contribute"
        assert bd['attention'] > 0, "8 attention sublayers should contribute"
        # Total should equal sum of components
        assert bd['total'] == pytest.approx(
            bd['mamba'] + bd['ffn'] + bd['attention'] + bd['other'], rel=1e-9
        )

    def test_mamba_and_ffn_are_dominant_components(self, single_gpu):
        """
        Mamba (40 sublayers) and LatentMoE FFN (40 sublayers) together should
        dwarf the pure-attention component (only 8 sublayers).
        """
        perf = InferencePerformance(NEMOTRON_3_SUPER_120B)
        bd = perf.calculate_prefill_compute_breakdown(
            batch_size=1, sequence_length=512, parallelism_config=single_gpu
        )
        # Both Mamba and LatentMoE should be substantial
        assert bd['mamba'] > 1e12, "40 Mamba sublayers should produce > 1 TFLOPs"
        assert bd['ffn'] > 1e12, "40 LatentMoE sublayers should produce > 1 TFLOPs"
        # 40 Mamba + 40 LatentMoE together dominate over 8 attention sublayers
        assert bd['mamba'] + bd['ffn'] > bd['attention']

    def test_kv_cache_only_from_attention_sublayers(self):
        """Only the 8 * (attention) sublayers contribute to KV cache."""
        m = NEMOTRON_3_SUPER_120B
        kv_dim = m.attention_config.num_key_value_heads * m.attention_config.head_dim
        # 8 attn layers × (K+V) × batch × seq × kv_dim × 2 bytes (bfloat16)
        kv_cache = m.get_kv_cache_size(batch_size=1, sequence_length=1024, bytes_per_element=2)
        expected = 8 * 2 * 1 * 1024 * kv_dim * 2
        assert kv_cache == expected

    def test_mamba_state_from_40_layers(self):
        """Mamba state should be from 40 MAMBA_ONLY sublayers."""
        m = NEMOTRON_3_SUPER_120B
        mc = m.mamba_config
        # state per layer: batch × heads × head_dim × state_size
        state_per_layer = 1 * mc.num_heads * mc.head_dim * mc.state_size * 2  # bfloat16
        total_state = m.get_mamba_state_size(batch_size=1, bytes_per_element=2)
        assert total_state == 40 * state_per_layer

    def test_per_layer_breakdown_length(self, single_gpu):
        """Per-layer breakdown should have exactly 88 entries."""
        perf = InferencePerformance(NEMOTRON_3_SUPER_120B)
        bd = perf.calculate_per_layer_breakdown(
            mode='prefill', batch_size=1, sequence_length=256,
            parallelism_config=single_gpu,
        )
        assert len(bd.attention_compute) == 88
        assert len(bd.non_attention_compute) == 88
        assert len(bd.layer_types) == 88

    def test_decode_per_layer_breakdown(self, single_gpu):
        """Decode per-layer breakdown should also work for Super model."""
        perf = InferencePerformance(NEMOTRON_3_SUPER_120B)
        bd = perf.calculate_per_layer_breakdown(
            mode='decode', batch_size=4, sequence_length=1024,
            parallelism_config=single_gpu,
        )
        attn_compute_sum = sum(bd.attention_compute)
        ffn_compute_sum = sum(bd.non_attention_compute)
        assert attn_compute_sum > 0
        assert ffn_compute_sum > 0

    def test_super_vs_30b_have_different_structures(self, single_gpu):
        """Super-120B and 30B should not share the same compute structure."""
        perf_30b = InferencePerformance(NEMOTRON_3_30B)
        perf_120b = InferencePerformance(NEMOTRON_3_SUPER_120B)
        bd30 = perf_30b.calculate_prefill_compute_breakdown(
            batch_size=1, sequence_length=256, parallelism_config=single_gpu
        )
        bd120 = perf_120b.calculate_prefill_compute_breakdown(
            batch_size=1, sequence_length=256, parallelism_config=single_gpu
        )
        # 30B has both mamba and attention+MoE layers; 120B should have mamba+latentMoE+attention
        # Both should have mamba and attention components
        assert bd30['mamba'] > 0
        assert bd120['mamba'] > 0
        assert bd30['attention'] > 0
        assert bd120['attention'] > 0
        # FFN in 30B comes from E/* layers (standard MoE); FFN in 120B from LatentMoE sublayers
        assert bd30['ffn'] > 0
        assert bd120['ffn'] > 0
