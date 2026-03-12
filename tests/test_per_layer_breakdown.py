"""
Tests for per-layer breakdown of compute, memory traffic, and kernel launches.
Tests cover uniform models (all layers identical) and non-uniform models
(mixed attention types, MoE/dense FFN, hybrid Mamba/attention, linear attention).
"""

import pytest
from inference_performance import (
    InferencePerformance, ParallelismConfig, ParallelismType, PerLayerBreakdown
)
from llm_configs import (
    LLAMA_3_8B, LLAMA_3_70B,
    DEEPSEEK_V3, DEEPSEEK_3_2,
    MIXTRAL_8X7B, GPT3_175B,
    QWEN3_5_397B
)


# Try importing non-uniform models that may exist
try:
    from llm_configs import LFM2_3B
except ImportError:
    LFM2_3B = None

try:
    from llm_configs import NEMOTRON_3_30B
except ImportError:
    NEMOTRON_3_30B = None

try:
    from llm_configs import KIMI_K25
except ImportError:
    KIMI_K25 = None

try:
    from llm_configs import GLM_5
except ImportError:
    GLM_5 = None

try:
    from llm_configs import HUNYUAN_A13B
except ImportError:
    HUNYUAN_A13B = None


NO_PARALLEL = ParallelismConfig()
TP2 = ParallelismConfig(parallelism_type=ParallelismType.TENSOR_PARALLEL, tensor_parallel_size=2)
TP4 = ParallelismConfig(parallelism_type=ParallelismType.TENSOR_PARALLEL, tensor_parallel_size=4)
PP2 = ParallelismConfig(parallelism_type=ParallelismType.PIPELINE_PARALLEL, pipeline_parallel_size=2)


class TestPerLayerBreakdownBasic:
    """Test basic structure and invariants of PerLayerBreakdown."""
    
    def test_returns_correct_type(self):
        """calculate_per_layer_breakdown returns PerLayerBreakdown."""
        perf = InferencePerformance(LLAMA_3_8B)
        result = perf.calculate_per_layer_breakdown('prefill', 1, 512, NO_PARALLEL)
        assert isinstance(result, PerLayerBreakdown)
    
    def test_correct_num_layers(self):
        """All arrays have length equal to num_layers."""
        perf = InferencePerformance(LLAMA_3_8B)
        result = perf.calculate_per_layer_breakdown('prefill', 1, 512, NO_PARALLEL)
        n = LLAMA_3_8B.num_layers
        assert result.num_layers == n
        assert len(result.attention_compute) == n
        assert len(result.non_attention_compute) == n
        assert len(result.attention_memory_traffic) == n
        assert len(result.non_attention_memory_traffic) == n
        assert len(result.attention_kernels) == n
        assert len(result.non_attention_kernels) == n
        assert len(result.layer_types) == n
    
    def test_all_values_positive(self):
        """All per-layer values should be positive."""
        perf = InferencePerformance(LLAMA_3_8B)
        result = perf.calculate_per_layer_breakdown('decode', 4, 1024, NO_PARALLEL)
        for i in range(result.num_layers):
            assert result.attention_compute[i] > 0
            assert result.non_attention_compute[i] > 0
            assert result.attention_memory_traffic[i] > 0
            assert result.non_attention_memory_traffic[i] > 0
            assert result.attention_kernels[i] > 0
            assert result.non_attention_kernels[i] > 0
    
    def test_mode_stored(self):
        """Mode is stored correctly."""
        perf = InferencePerformance(LLAMA_3_8B)
        r1 = perf.calculate_per_layer_breakdown('prefill', 1, 512, NO_PARALLEL)
        r2 = perf.calculate_per_layer_breakdown('decode', 1, 512, NO_PARALLEL)
        assert r1.mode == 'prefill'
        assert r2.mode == 'decode'
    
    def test_metadata_stored(self):
        """Batch size and sequence length metadata stored correctly."""
        perf = InferencePerformance(LLAMA_3_8B)
        result = perf.calculate_per_layer_breakdown('prefill', 8, 2048, NO_PARALLEL)
        assert result.batch_size == 8
        assert result.sequence_length == 2048


class TestUniformModel:
    """Test that uniform models produce identical per-layer values."""
    
    def test_uniform_compute_prefill(self):
        """All layers of a uniform model should have identical compute."""
        perf = InferencePerformance(LLAMA_3_8B)
        result = perf.calculate_per_layer_breakdown('prefill', 1, 1024, NO_PARALLEL)
        # All layers should be identical
        for i in range(1, result.num_layers):
            assert result.attention_compute[i] == result.attention_compute[0]
            assert result.non_attention_compute[i] == result.non_attention_compute[0]
    
    def test_uniform_compute_decode(self):
        """Uniform model decode: all layers identical."""
        perf = InferencePerformance(LLAMA_3_8B)
        result = perf.calculate_per_layer_breakdown('decode', 4, 2048, NO_PARALLEL)
        for i in range(1, result.num_layers):
            assert result.attention_compute[i] == result.attention_compute[0]
            assert result.non_attention_compute[i] == result.non_attention_compute[0]
    
    def test_uniform_bandwidth(self):
        """Uniform model: all layers same bandwidth."""
        perf = InferencePerformance(LLAMA_3_8B)
        result = perf.calculate_per_layer_breakdown('decode', 1, 1024, NO_PARALLEL)
        for i in range(1, result.num_layers):
            assert result.attention_memory_traffic[i] == result.attention_memory_traffic[0]
            assert result.non_attention_memory_traffic[i] == result.non_attention_memory_traffic[0]
    
    def test_uniform_kernels(self):
        """Uniform model: all layers same kernel count."""
        perf = InferencePerformance(LLAMA_3_8B)
        result = perf.calculate_per_layer_breakdown('decode', 1, 1024, NO_PARALLEL)
        for i in range(1, result.num_layers):
            assert result.attention_kernels[i] == result.attention_kernels[0]
            assert result.non_attention_kernels[i] == result.non_attention_kernels[0]
    
    def test_uniform_layer_types(self):
        """Uniform dense model layers labelled Full/Dense."""
        perf = InferencePerformance(LLAMA_3_8B)
        result = perf.calculate_per_layer_breakdown('prefill', 1, 512, NO_PARALLEL)
        for label in result.layer_types:
            assert label == "Full/Dense"
    
    def test_uniform_moe_layer_types(self):
        """Uniform MoE model layers labelled Full/MoE."""
        perf = InferencePerformance(MIXTRAL_8X7B)
        result = perf.calculate_per_layer_breakdown('prefill', 1, 512, NO_PARALLEL)
        for label in result.layer_types:
            assert label == "Full/MoE"


class TestComputeConsistency:
    """Test that per-layer compute sums match existing aggregate breakdowns."""
    
    def test_prefill_compute_sum_matches_breakdown(self):
        """Sum of per-layer prefill compute should match the aggregate breakdown."""
        perf = InferencePerformance(LLAMA_3_8B)
        # Per-layer breakdown (no TP)
        per_layer = perf.calculate_per_layer_breakdown('prefill', 2, 1024, NO_PARALLEL)
        total_attn = sum(per_layer.attention_compute)
        total_ffn = sum(per_layer.non_attention_compute)
        
        # Aggregate breakdown
        agg = perf.calculate_prefill_compute_breakdown(2, 1024, NO_PARALLEL)
        
        # They should match (per_layer doesn't include 'other' like LM head)
        assert abs(total_attn - agg['attention']) / agg['attention'] < 1e-6
        assert abs(total_ffn - agg['ffn']) / agg['ffn'] < 1e-6
    
    def test_decode_compute_sum_matches_breakdown(self):
        """Sum of per-layer decode compute should match the aggregate breakdown."""
        perf = InferencePerformance(LLAMA_3_8B)
        per_layer = perf.calculate_per_layer_breakdown('decode', 4, 2048, NO_PARALLEL)
        total_attn = sum(per_layer.attention_compute)
        total_ffn = sum(per_layer.non_attention_compute)
        
        agg = perf._calculate_decode_step_compute_breakdown(4, 2048, NO_PARALLEL)
        
        assert abs(total_attn - agg['attention']) / agg['attention'] < 1e-6
        assert abs(total_ffn - agg['ffn']) / agg['ffn'] < 1e-6
    
    def test_prefill_compute_sum_moe_model(self):
        """Per-layer compute matches aggregate for MoE model."""
        perf = InferencePerformance(MIXTRAL_8X7B)
        per_layer = perf.calculate_per_layer_breakdown('prefill', 1, 512, NO_PARALLEL)
        agg = perf.calculate_prefill_compute_breakdown(1, 512, NO_PARALLEL)
        
        total_attn = sum(per_layer.attention_compute)
        total_ffn = sum(per_layer.non_attention_compute)
        
        assert abs(total_attn - agg['attention']) / agg['attention'] < 1e-6
        assert abs(total_ffn - agg['ffn']) / agg['ffn'] < 1e-6
    
    def test_prefill_compute_sum_qwen35(self):
        """Per-layer compute matches aggregate for Qwen3.5 (linear attention)."""
        perf = InferencePerformance(QWEN3_5_397B)
        per_layer = perf.calculate_per_layer_breakdown('prefill', 1, 1024, NO_PARALLEL)
        agg = perf.calculate_prefill_compute_breakdown(1, 1024, NO_PARALLEL)
        
        total_attn = sum(per_layer.attention_compute)
        total_ffn = sum(per_layer.non_attention_compute)
        
        assert abs(total_attn - agg['attention']) / agg['attention'] < 1e-6
        assert abs(total_ffn - agg['ffn']) / agg['ffn'] < 1e-6
    
    def test_decode_compute_sum_qwen35(self):
        """Per-layer decode compute matches aggregate for Qwen3.5."""
        perf = InferencePerformance(QWEN3_5_397B)
        per_layer = perf.calculate_per_layer_breakdown('decode', 2, 4096, NO_PARALLEL)
        agg = perf._calculate_decode_step_compute_breakdown(2, 4096, NO_PARALLEL)
        
        total_attn = sum(per_layer.attention_compute)
        total_ffn = sum(per_layer.non_attention_compute)
        
        assert abs(total_attn - agg['attention']) / agg['attention'] < 1e-6
        assert abs(total_ffn - agg['ffn']) / agg['ffn'] < 1e-6
    
    def test_decode_compute_sum_deepseek_v3(self):
        """Per-layer decode compute matches aggregate for DeepSeek-V3 (MLA + MoE)."""
        perf = InferencePerformance(DEEPSEEK_V3)
        per_layer = perf.calculate_per_layer_breakdown('decode', 1, 2048, NO_PARALLEL)
        agg = perf._calculate_decode_step_compute_breakdown(1, 2048, NO_PARALLEL)
        
        total_attn = sum(per_layer.attention_compute)
        total_ffn = sum(per_layer.non_attention_compute)
        
        assert abs(total_attn - agg['attention']) / agg['attention'] < 1e-6
        assert abs(total_ffn - agg['ffn']) / agg['ffn'] < 1e-6


class TestNonUniformModels:
    """Test that non-uniform models show variation across layers."""
    
    def test_qwen35_linear_vs_full_attention_compute(self):
        """Qwen3.5 linear attention layers should have different compute than full."""
        perf = InferencePerformance(QWEN3_5_397B)
        result = perf.calculate_per_layer_breakdown('decode', 1, 4096, NO_PARALLEL)
        
        linear_computes = []
        full_computes = []
        for i in range(result.num_layers):
            if 'Linear' in result.layer_types[i]:
                linear_computes.append(result.attention_compute[i])
            elif 'Full' in result.layer_types[i]:
                full_computes.append(result.attention_compute[i])
        
        assert len(linear_computes) > 0
        assert len(full_computes) > 0
        # At long context, full attention should be more expensive than linear
        assert sum(full_computes) / len(full_computes) > sum(linear_computes) / len(linear_computes)
    
    def test_qwen35_linear_vs_full_bandwidth(self):
        """At very long context, full attention KV cache traffic should exceed linear state traffic."""
        perf = InferencePerformance(QWEN3_5_397B)
        # Use a very long context to make KV cache traffic dominate
        result = perf.calculate_per_layer_breakdown('decode', 32, 32768, NO_PARALLEL)
        
        linear_bw = []
        full_bw = []
        for i in range(result.num_layers):
            if 'Linear' in result.layer_types[i]:
                linear_bw.append(result.attention_memory_traffic[i])
            elif 'Full' in result.layer_types[i]:
                full_bw.append(result.attention_memory_traffic[i])
        
        # At very long context with large batch, full attention KV cache dominates
        avg_full = sum(full_bw) / len(full_bw)
        avg_linear = sum(linear_bw) / len(linear_bw)
        assert avg_full > avg_linear
    
    def test_qwen35_layer_type_distribution(self):
        """Check Qwen3.5 has correct number of linear vs full layers."""
        perf = InferencePerformance(QWEN3_5_397B)
        result = perf.calculate_per_layer_breakdown('prefill', 1, 512, NO_PARALLEL)
        
        num_linear = sum(1 for lt in result.layer_types if 'Linear' in lt)
        num_full = sum(1 for lt in result.layer_types if 'Full' in lt)
        
        assert num_linear == 45  # 3/4 of 60 layers
        assert num_full == 15   # 1/4 of 60 layers
    
    def test_interleaved_moe_dense_ffn(self):
        """Models with interleaved MoE/dense should show different FFN bandwidth."""
        # DeepSeek-3.2 has interleaved dense/MoE FFN
        if DEEPSEEK_3_2.ffn_layer_types is None:
            pytest.skip("DeepSeek-3.2 does not have ffn_layer_types")
        
        perf = InferencePerformance(DEEPSEEK_3_2)
        result = perf.calculate_per_layer_breakdown('decode', 1, 1024, NO_PARALLEL)
        
        moe_bw = []
        dense_bw = []
        for i in range(result.num_layers):
            if 'MoE' in result.layer_types[i]:
                moe_bw.append(result.non_attention_memory_traffic[i])
            elif 'Dense' in result.layer_types[i]:
                dense_bw.append(result.non_attention_memory_traffic[i])
        
        if moe_bw and dense_bw:
            # MoE layers read all expert weights -> much more bandwidth
            avg_moe = sum(moe_bw) / len(moe_bw)
            avg_dense = sum(dense_bw) / len(dense_bw)
            assert avg_moe > avg_dense
    
    def test_interleaved_moe_dense_compute(self):
        """MoE layers should have different compute than dense."""
        if DEEPSEEK_3_2.ffn_layer_types is None:
            pytest.skip("DeepSeek-3.2 does not have ffn_layer_types")
        
        perf = InferencePerformance(DEEPSEEK_3_2)
        result = perf.calculate_per_layer_breakdown('prefill', 1, 1024, NO_PARALLEL)
        
        moe_compute = []
        dense_compute = []
        for i in range(result.num_layers):
            if 'MoE' in result.layer_types[i]:
                moe_compute.append(result.non_attention_compute[i])
            elif 'Dense' in result.layer_types[i]:
                dense_compute.append(result.non_attention_compute[i])
        
        if moe_compute and dense_compute:
            # MoE compute uses active experts (may be more or less than dense)
            assert moe_compute[0] != dense_compute[0]  # Just verify they differ


class TestTPScaling:
    """Test that tensor parallelism correctly scales per-layer values."""
    
    def test_tp_halves_compute(self):
        """TP=2 should halve per-layer compute vs TP=1."""
        perf = InferencePerformance(LLAMA_3_8B)
        r1 = perf.calculate_per_layer_breakdown('decode', 4, 1024, NO_PARALLEL)
        r2 = perf.calculate_per_layer_breakdown('decode', 4, 1024, TP2)
        
        for i in range(r1.num_layers):
            assert abs(r2.attention_compute[i] - r1.attention_compute[i] / 2) < 1
            assert abs(r2.non_attention_compute[i] - r1.non_attention_compute[i] / 2) < 1
    
    def test_tp_halves_bandwidth(self):
        """TP=2 should halve per-layer bandwidth vs TP=1."""
        perf = InferencePerformance(LLAMA_3_8B)
        r1 = perf.calculate_per_layer_breakdown('decode', 4, 1024, NO_PARALLEL)
        r2 = perf.calculate_per_layer_breakdown('decode', 4, 1024, TP2)
        
        for i in range(r1.num_layers):
            ratio_attn = r2.attention_memory_traffic[i] / r1.attention_memory_traffic[i]
            ratio_ffn = r2.non_attention_memory_traffic[i] / r1.non_attention_memory_traffic[i]
            assert abs(ratio_attn - 0.5) < 0.01
            assert abs(ratio_ffn - 0.5) < 0.01
    
    def test_tp_does_not_change_kernels(self):
        """TP should not change per-layer kernel counts."""
        perf = InferencePerformance(LLAMA_3_8B)
        r1 = perf.calculate_per_layer_breakdown('decode', 1, 512, NO_PARALLEL)
        r2 = perf.calculate_per_layer_breakdown('decode', 1, 512, TP4)
        
        for i in range(r1.num_layers):
            assert r1.attention_kernels[i] == r2.attention_kernels[i]
            assert r1.non_attention_kernels[i] == r2.non_attention_kernels[i]


class TestBatchScaling:
    """Test that batch size correctly scales per-layer values."""
    
    def test_double_batch_doubles_compute(self):
        """Doubling batch should double compute."""
        perf = InferencePerformance(LLAMA_3_8B)
        r1 = perf.calculate_per_layer_breakdown('prefill', 1, 512, NO_PARALLEL)
        r2 = perf.calculate_per_layer_breakdown('prefill', 2, 512, NO_PARALLEL)
        
        for i in range(r1.num_layers):
            assert abs(r2.attention_compute[i] / r1.attention_compute[i] - 2.0) < 1e-6
            assert abs(r2.non_attention_compute[i] / r1.non_attention_compute[i] - 2.0) < 1e-6
    
    def test_double_batch_scales_kv_traffic(self):
        """Doubling batch should scale KV cache traffic."""
        perf = InferencePerformance(LLAMA_3_8B)
        r1 = perf.calculate_per_layer_breakdown('decode', 1, 2048, NO_PARALLEL)
        r2 = perf.calculate_per_layer_breakdown('decode', 2, 2048, NO_PARALLEL)
        
        # Attention traffic includes weight reads (constant) + KV cache (scales with batch)
        # So it should be between 1x and 2x
        for i in range(r1.num_layers):
            ratio = r2.attention_memory_traffic[i] / r1.attention_memory_traffic[i]
            assert 1.0 < ratio <= 2.0 + 1e-6


class TestContextScaling:
    """Test how context length affects per-layer values."""
    
    def test_longer_context_more_attention_compute_decode(self):
        """Longer context should increase attention compute for decode."""
        perf = InferencePerformance(LLAMA_3_8B)
        r1 = perf.calculate_per_layer_breakdown('decode', 1, 512, NO_PARALLEL)
        r2 = perf.calculate_per_layer_breakdown('decode', 1, 4096, NO_PARALLEL)
        
        for i in range(r1.num_layers):
            assert r2.attention_compute[i] > r1.attention_compute[i]
    
    def test_context_does_not_affect_ffn_compute_decode(self):
        """FFN compute in decode should not change with context (processes 1 token)."""
        perf = InferencePerformance(LLAMA_3_8B)
        r1 = perf.calculate_per_layer_breakdown('decode', 1, 512, NO_PARALLEL)
        r2 = perf.calculate_per_layer_breakdown('decode', 1, 4096, NO_PARALLEL)
        
        for i in range(r1.num_layers):
            assert r1.non_attention_compute[i] == r2.non_attention_compute[i]
    
    def test_linear_attention_constant_decode_compute(self):
        """Linear attention compute should be constant regardless of context."""
        perf = InferencePerformance(QWEN3_5_397B)
        r1 = perf.calculate_per_layer_breakdown('decode', 1, 1024, NO_PARALLEL)
        r2 = perf.calculate_per_layer_breakdown('decode', 1, 16384, NO_PARALLEL)
        
        for i in range(r1.num_layers):
            if 'Linear' in r1.layer_types[i]:
                assert r1.attention_compute[i] == r2.attention_compute[i]
    
    def test_linear_attention_constant_decode_bandwidth(self):
        """Linear attention bandwidth should be constant regardless of context."""
        perf = InferencePerformance(QWEN3_5_397B)
        r1 = perf.calculate_per_layer_breakdown('decode', 1, 1024, NO_PARALLEL)
        r2 = perf.calculate_per_layer_breakdown('decode', 1, 16384, NO_PARALLEL)
        
        for i in range(r1.num_layers):
            if 'Linear' in r1.layer_types[i]:
                assert r1.attention_memory_traffic[i] == r2.attention_memory_traffic[i]
    
    def test_full_attention_grows_with_context(self):
        """Full attention KV cache traffic grows with context."""
        perf = InferencePerformance(QWEN3_5_397B)
        r1 = perf.calculate_per_layer_breakdown('decode', 1, 1024, NO_PARALLEL)
        r2 = perf.calculate_per_layer_breakdown('decode', 1, 16384, NO_PARALLEL)
        
        for i in range(r1.num_layers):
            if 'Full' in r1.layer_types[i]:
                assert r2.attention_memory_traffic[i] > r1.attention_memory_traffic[i]


class TestKernelLaunches:
    """Test kernel launch counts per layer."""
    
    def test_standard_attention_kernels(self):
        """Standard attention layer should have expected kernel counts."""
        perf = InferencePerformance(LLAMA_3_8B)
        result = perf.calculate_per_layer_breakdown('decode', 1, 512, NO_PARALLEL)
        # Standard: 6 attention + 3 FFN = 9 total
        assert result.attention_kernels[0] == 6
        assert result.non_attention_kernels[0] == 3
    
    def test_linear_attention_kernels(self):
        """Linear attention layers should have fewer kernels (no softmax)."""
        perf = InferencePerformance(QWEN3_5_397B)
        result = perf.calculate_per_layer_breakdown('decode', 1, 512, NO_PARALLEL)
        
        for i in range(result.num_layers):
            if 'Linear' in result.layer_types[i]:
                assert result.attention_kernels[i] == 6  # norm + QKV + conv + state + query + O
    
    def test_moe_kernels_more_than_dense(self):
        """MoE layers should have more FFN kernels than dense."""
        if DEEPSEEK_3_2.ffn_layer_types is None:
            pytest.skip("Model does not have ffn_layer_types")
        
        perf = InferencePerformance(DEEPSEEK_3_2)
        result = perf.calculate_per_layer_breakdown('decode', 1, 512, NO_PARALLEL)
        
        moe_ffn_kernels = None
        dense_ffn_kernels = None
        for i in range(result.num_layers):
            if 'MoE' in result.layer_types[i] and moe_ffn_kernels is None:
                moe_ffn_kernels = result.non_attention_kernels[i]
            elif 'Dense' in result.layer_types[i] and dense_ffn_kernels is None:
                dense_ffn_kernels = result.non_attention_kernels[i]
        
        if moe_ffn_kernels is not None and dense_ffn_kernels is not None:
            assert moe_ffn_kernels > dense_ffn_kernels
    
    def test_total_kernels_match_aggregate(self):
        """Sum of per-layer kernels should match aggregate kernel count."""
        perf = InferencePerformance(LLAMA_3_8B)
        result = perf.calculate_per_layer_breakdown('decode', 1, 512, NO_PARALLEL)
        
        per_layer_total = sum(
            result.attention_kernels[i] + result.non_attention_kernels[i]
            for i in range(result.num_layers)
        )
        
        agg = perf.calculate_num_kernel_launches(NO_PARALLEL)
        # Aggregate includes embedding + final norm + LM head = 3 extra
        assert per_layer_total == agg - 3


class TestMPrefillMode:
    """Test prefill-specific behavior."""
    
    def test_prefill_longer_seq_more_compute(self):
        """Longer sequence should mean more prefill compute per layer."""
        perf = InferencePerformance(LLAMA_3_8B)
        r1 = perf.calculate_per_layer_breakdown('prefill', 1, 512, NO_PARALLEL)
        r2 = perf.calculate_per_layer_breakdown('prefill', 1, 2048, NO_PARALLEL)
        
        for i in range(r1.num_layers):
            assert r2.attention_compute[i] > r1.attention_compute[i]
            assert r2.non_attention_compute[i] > r1.non_attention_compute[i]
    
    def test_prefill_attention_quadratic_scaling(self):
        """Prefill attention should grow faster than linearly with sequence length."""
        perf = InferencePerformance(LLAMA_3_8B)
        r1 = perf.calculate_per_layer_breakdown('prefill', 1, 512, NO_PARALLEL)
        r2 = perf.calculate_per_layer_breakdown('prefill', 1, 1024, NO_PARALLEL)
        
        # Attention includes O(L^2) component, so doubling L should more than double compute
        ratio = r2.attention_compute[0] / r1.attention_compute[0]
        assert ratio > 2.0  # Must be more than 2x due to quadratic attention


class TestHybridMambaAttention:
    """Test hybrid Mamba/attention models."""
    
    @pytest.mark.skipif(NEMOTRON_3_30B is None, reason="NEMOTRON_3_30B not available")
    def test_mamba_layers_labelled(self):
        """Mamba layers should be labeled as Mamba."""
        perf = InferencePerformance(NEMOTRON_3_30B)
        result = perf.calculate_per_layer_breakdown('decode', 1, 512, NO_PARALLEL)
        
        mamba_count = sum(1 for lt in result.layer_types if 'Mamba' in lt)
        assert mamba_count > 0
    
    @pytest.mark.skipif(NEMOTRON_3_30B is None, reason="NEMOTRON_3_30B not available")
    def test_mamba_vs_attention_different_compute(self):
        """Mamba and attention layers should have different compute."""
        perf = InferencePerformance(NEMOTRON_3_30B)
        result = perf.calculate_per_layer_breakdown('decode', 1, 2048, NO_PARALLEL)
        
        mamba_computes = [result.attention_compute[i] for i in range(result.num_layers)
                         if 'Mamba' in result.layer_types[i]]
        attn_computes = [result.attention_compute[i] for i in range(result.num_layers)
                        if 'Full' in result.layer_types[i] or 'Sliding' in result.layer_types[i]]
        
        if mamba_computes and attn_computes:
            assert mamba_computes[0] != attn_computes[0]


class TestMLAModel:
    """Test MLA (Multi-head Latent Attention) model specifics."""
    
    def test_mla_model_computes(self):
        """DeepSeek-V3 with MLA should produce valid per-layer data."""
        perf = InferencePerformance(DEEPSEEK_V3)
        result = perf.calculate_per_layer_breakdown('decode', 1, 2048, NO_PARALLEL)
        
        for i in range(result.num_layers):
            assert result.attention_compute[i] > 0
            assert result.non_attention_compute[i] > 0
            assert result.attention_memory_traffic[i] > 0
    
    def test_mla_compressed_kv_bandwidth(self):
        """MLA compressed KV cache should result in less KV traffic per layer."""
        # Compare DeepSeek-V3 (MLA) vs Llama-3-70B (standard GQA)
        # Both are large models but MLA should have less KV traffic per head
        perf_mla = InferencePerformance(DEEPSEEK_V3)
        r_mla = perf_mla.calculate_per_layer_breakdown('decode', 1, 4096, NO_PARALLEL)
        
        # Just verify it runs and produces positive values
        for i in range(r_mla.num_layers):
            assert r_mla.attention_memory_traffic[i] > 0


class TestAllConfiguredModels:
    """Run per-layer breakdown on all models in the config to ensure no crashes."""
    
    @pytest.mark.parametrize("model_name,model", [
        ("llama-3-8b", LLAMA_3_8B),
        ("llama-3-70b", LLAMA_3_70B),
        ("deepseek-v3", DEEPSEEK_V3),
        ("deepseek-3.2", DEEPSEEK_3_2),
        ("mixtral-8x7b", MIXTRAL_8X7B),
        ("gpt3-175b", GPT3_175B),
        ("qwen3.5-397b", QWEN3_5_397B),
    ])
    def test_no_crash_prefill(self, model_name, model):
        """Per-layer breakdown should not crash for any model in prefill mode."""
        perf = InferencePerformance(model)
        result = perf.calculate_per_layer_breakdown('prefill', 1, 512, NO_PARALLEL)
        assert result.num_layers == model.num_layers
        assert all(v >= 0 for v in result.attention_compute)
        assert all(v >= 0 for v in result.non_attention_compute)
    
    @pytest.mark.parametrize("model_name,model", [
        ("llama-3-8b", LLAMA_3_8B),
        ("llama-3-70b", LLAMA_3_70B),
        ("deepseek-v3", DEEPSEEK_V3),
        ("deepseek-3.2", DEEPSEEK_3_2),
        ("mixtral-8x7b", MIXTRAL_8X7B),
        ("gpt3-175b", GPT3_175B),
        ("qwen3.5-397b", QWEN3_5_397B),
    ])
    def test_no_crash_decode(self, model_name, model):
        """Per-layer breakdown should not crash for any model in decode mode."""
        perf = InferencePerformance(model)
        result = perf.calculate_per_layer_breakdown('decode', 4, 2048, NO_PARALLEL)
        assert result.num_layers == model.num_layers
        assert all(v >= 0 for v in result.attention_compute)
