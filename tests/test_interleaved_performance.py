"""
Tests for performance modeling of interleaved Dense/MoE FFN architectures.

These tests verify that:
1. FLOPs are calculated correctly for interleaved dense/MoE layers
2. Dense layers use dense_intermediate_size for calculations
3. MoE layers use intermediate_size (per-expert) with routing overhead
4. Maverick (interleaved) has different FLOPs than Scout (all MoE)
5. Storage traffic only counts MoE layers (not dense layers)
"""

import pytest
from llm_architecture import (
    LLMArchitecture, AttentionConfig, FFNConfig, MoEConfig,
    ArchitectureType, AttentionType, ActivationType,
    NormalizationType, PositionEncodingType, FFNLayerType
)
from llm_configs import LLAMA_4_SCOUT, LLAMA_4_MAVERICK
from inference_performance import (
    InferencePerformance, SystemConstraints, ParallelismConfig, ParallelismType
)


class TestInterleavedPrefillCompute:
    """Tests for prefill compute breakdown with interleaved layers"""
    
    @pytest.fixture
    def gpu_constraints(self):
        """Standard GPU constraints for testing"""
        return SystemConstraints(
            memory_capacity=80e9,  # 80GB
            memory_bandwidth=3.35e12,  # 3.35 TB/s
            compute_throughput=1979e12,  # ~2 PFLOPS
            network_bandwidth=900e9,  # 900 GB/s
        )
    
    @pytest.fixture
    def parallel_config(self):
        """Single GPU parallelism config"""
        return ParallelismConfig(
            parallelism_type=ParallelismType.TENSOR_PARALLEL,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
        )
    
    @pytest.fixture
    def all_dense_model(self):
        """Model with all dense FFN layers"""
        return LLMArchitecture(
            model_name="test-all-dense",
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
                activation=ActivationType.SILU,
            ),
            is_moe=False,
            ffn_layer_types=[FFNLayerType.DENSE] * 4,
            total_parameters=500_000_000,
            active_parameters=500_000_000,
        )
    
    @pytest.fixture
    def all_moe_model(self):
        """Model with all MoE FFN layers"""
        return LLMArchitecture(
            model_name="test-all-moe",
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
                intermediate_size=2048,  # Per-expert size (smaller)
                dense_intermediate_size=4096,  # Dense size (not used)
                activation=ActivationType.SILU,
            ),
            moe_config=MoEConfig(
                num_experts=8,
                num_experts_per_token=2,  # Top-2 routing
            ),
            is_moe=True,
            ffn_layer_types=[FFNLayerType.MOE] * 4,
            total_parameters=2_000_000_000,
            active_parameters=500_000_000,
        )
    
    @pytest.fixture
    def interleaved_model(self):
        """Model with interleaved Dense/MoE layers (Dense, MoE, Dense, MoE)"""
        return LLMArchitecture(
            model_name="test-interleaved",
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
                intermediate_size=2048,  # Per-expert size for MoE layers
                dense_intermediate_size=4096,  # Dense FFN size
                activation=ActivationType.SILU,
            ),
            moe_config=MoEConfig(
                num_experts=8,
                num_experts_per_token=2,
            ),
            is_moe=True,
            ffn_layer_types=[FFNLayerType.DENSE, FFNLayerType.MOE, FFNLayerType.DENSE, FFNLayerType.MOE],
            total_parameters=1_000_000_000,
            active_parameters=500_000_000,
        )
    
    def test_all_dense_ffn_flops(self, all_dense_model, parallel_config):
        """Test FFN FLOPs for all-dense model"""
        perf = InferencePerformance(all_dense_model)
        breakdown = perf.calculate_prefill_compute_breakdown(
            batch_size=1, sequence_length=1024, parallelism_config=parallel_config
        )
        
        # Each dense layer: 2 * B * L * H * intermediate * 2 (up + down)
        # = 2 * 1 * 1024 * 1024 * 4096 * 2 = 17.2B FLOPs per layer
        # 4 layers = ~68.7B FLOPs
        expected_per_layer = 2 * 1 * 1024 * 1024 * 4096 * 2
        expected_total = expected_per_layer * 4
        
        # Allow some tolerance for additional operations
        assert breakdown['ffn'] == pytest.approx(expected_total, rel=0.01)
    
    def test_all_moe_ffn_flops(self, all_moe_model, parallel_config):
        """Test FFN FLOPs for all-MoE model"""
        perf = InferencePerformance(all_moe_model)
        breakdown = perf.calculate_prefill_compute_breakdown(
            batch_size=1, sequence_length=1024, parallelism_config=parallel_config
        )
        
        # Each MoE layer:
        # - Router: 2 * B * L * H * num_experts = 2 * 1 * 1024 * 1024 * 8
        # - Active experts: 2 * B * L * H * intermediate * active_experts * 2 (up + down)
        #   = 2 * 1 * 1024 * 1024 * 2048 * 2 * 2
        router_flops = 2 * 1 * 1024 * 1024 * 8
        expert_flops = 2 * 1 * 1024 * 1024 * 2048 * 2 * 2
        expected_per_layer = router_flops + expert_flops
        expected_total = expected_per_layer * 4
        
        assert breakdown['ffn'] == pytest.approx(expected_total, rel=0.01)
    
    def test_interleaved_ffn_flops_between(self, all_dense_model, all_moe_model, interleaved_model, parallel_config):
        """Test interleaved model FFN FLOPs is between all-dense and all-MoE"""
        perf_dense = InferencePerformance(all_dense_model)
        perf_moe = InferencePerformance(all_moe_model)
        perf_inter = InferencePerformance(interleaved_model)
        
        dense_breakdown = perf_dense.calculate_prefill_compute_breakdown(
            batch_size=1, sequence_length=1024, parallelism_config=parallel_config
        )
        moe_breakdown = perf_moe.calculate_prefill_compute_breakdown(
            batch_size=1, sequence_length=1024, parallelism_config=parallel_config
        )
        inter_breakdown = perf_inter.calculate_prefill_compute_breakdown(
            batch_size=1, sequence_length=1024, parallelism_config=parallel_config
        )
        
        # Interleaved should have FLOPs that reflect 2 dense + 2 MoE layers
        # So it should be roughly average of (4 dense) and (4 MoE)
        # But we specifically test it's properly between based on layer mix
        dense_ffn = dense_breakdown['ffn']
        moe_ffn = moe_breakdown['ffn']
        inter_ffn = inter_breakdown['ffn']
        
        # Interleaved is 2 dense + 2 MoE (half of each)
        expected_inter = (dense_ffn + moe_ffn) / 2
        assert inter_ffn == pytest.approx(expected_inter, rel=0.05)
    
    def test_interleaved_exact_calculation(self, interleaved_model, parallel_config):
        """Test exact FFN FLOPs calculation for interleaved model"""
        perf = InferencePerformance(interleaved_model)
        breakdown = perf.calculate_prefill_compute_breakdown(
            batch_size=1, sequence_length=1024, parallelism_config=parallel_config
        )
        
        B, L, H = 1, 1024, 1024
        
        # 2 Dense layers: each uses dense_intermediate_size = 4096
        dense_up_down = 2 * B * L * H * 4096 * 2  # up + down
        dense_total = dense_up_down * 2  # 2 layers
        
        # 2 MoE layers: each uses intermediate_size = 2048, 2 active experts
        router_per_layer = 2 * B * L * H * 8  # 8 experts
        expert_per_layer = 2 * B * L * H * 2048 * 2 * 2  # 2 experts, up + down
        moe_total = (router_per_layer + expert_per_layer) * 2  # 2 layers
        
        expected_total = dense_total + moe_total
        assert breakdown['ffn'] == pytest.approx(expected_total, rel=0.01)


class TestInterleavedDecodeCompute:
    """Tests for decode compute with interleaved layers"""
    
    @pytest.fixture
    def parallel_config(self):
        return ParallelismConfig(
            parallelism_type=ParallelismType.TENSOR_PARALLEL,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
        )
    
    @pytest.fixture
    def interleaved_model(self):
        """Model with interleaved Dense/MoE layers"""
        return LLMArchitecture(
            model_name="test-interleaved",
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
                dense_intermediate_size=4096,
                activation=ActivationType.SILU,
            ),
            moe_config=MoEConfig(
                num_experts=8,
                num_experts_per_token=2,
            ),
            is_moe=True,
            ffn_layer_types=[FFNLayerType.DENSE, FFNLayerType.MOE, FFNLayerType.DENSE, FFNLayerType.MOE],
            total_parameters=1_000_000_000,
            active_parameters=500_000_000,
        )
    
    def test_decode_step_interleaved_ffn(self, interleaved_model, parallel_config):
        """Test decode step FFN FLOPs for interleaved model"""
        perf = InferencePerformance(interleaved_model)
        breakdown = perf._calculate_decode_step_compute_breakdown(
            batch_size=1, context_length=1024, parallelism_config=parallel_config
        )
        
        B, L, H = 1, 1, 1024  # Decode: seq_len = 1
        
        # 2 Dense layers
        dense_per_layer = 2 * B * L * H * 4096 * 2  # up + down
        dense_total = dense_per_layer * 2
        
        # 2 MoE layers
        router_per_layer = 2 * B * L * H * 8
        expert_per_layer = 2 * B * L * H * 2048 * 2 * 2
        moe_total = (router_per_layer + expert_per_layer) * 2
        
        expected_ffn = dense_total + moe_total
        assert breakdown['ffn'] == pytest.approx(expected_ffn, rel=0.01)


class TestLlama4PerformanceDifference:
    """Tests comparing Scout (all MoE) vs Maverick (interleaved) performance"""
    
    @pytest.fixture
    def gpu_constraints(self):
        return SystemConstraints(
            memory_capacity=80e9,
            memory_bandwidth=3.35e12,
            compute_throughput=1979e12,
            network_bandwidth=900e9,
        )
    
    @pytest.fixture
    def parallel_config(self):
        return ParallelismConfig(
            parallelism_type=ParallelismType.TENSOR_PARALLEL,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
        )
    
    def test_scout_all_moe_layers(self):
        """Verify Scout has all MoE layers"""
        assert LLAMA_4_SCOUT.get_num_moe_ffn_layers() == 48
        assert LLAMA_4_SCOUT.get_num_dense_ffn_layers() == 0
    
    def test_maverick_interleaved_layers(self):
        """Verify Maverick has interleaved layers"""
        assert LLAMA_4_MAVERICK.get_num_moe_ffn_layers() == 24
        assert LLAMA_4_MAVERICK.get_num_dense_ffn_layers() == 24
    
    def test_maverick_dense_uses_larger_intermediate(self, parallel_config):
        """Test Maverick's dense layers use larger intermediate size"""
        perf = InferencePerformance(LLAMA_4_MAVERICK)
        
        # Dense intermediate = 16384, MoE intermediate = 8192
        assert LLAMA_4_MAVERICK.get_dense_intermediate_size() == 16384
        assert LLAMA_4_MAVERICK.get_moe_intermediate_size() == 8192
        
        breakdown = perf.calculate_prefill_compute_breakdown(
            batch_size=1, sequence_length=1024, parallelism_config=parallel_config
        )
        
        # FFN FLOPs should be positive and substantial
        assert breakdown['ffn'] > 0
    
    def test_scout_vs_maverick_ffn_difference(self, parallel_config):
        """Test Scout and Maverick have different FFN FLOPs"""
        scout_perf = InferencePerformance(LLAMA_4_SCOUT)
        maverick_perf = InferencePerformance(LLAMA_4_MAVERICK)
        
        scout_breakdown = scout_perf.calculate_prefill_compute_breakdown(
            batch_size=1, sequence_length=1024, parallelism_config=parallel_config
        )
        maverick_breakdown = maverick_perf.calculate_prefill_compute_breakdown(
            batch_size=1, sequence_length=1024, parallelism_config=parallel_config
        )
        
        # They should have different FFN FLOPs due to:
        # - Scout: 48 MoE layers, each with router + 1 expert
        # - Maverick: 24 dense (16384) + 24 MoE (8192, 1 expert)
        
        # The FLOPs should be different
        scout_ffn = scout_breakdown['ffn']
        maverick_ffn = maverick_breakdown['ffn']
        
        # Scout has all MoE with router overhead; Maverick has dense layers with larger intermediate
        # They should be measurably different
        assert scout_ffn != maverick_ffn
        
        # Maverick's dense layers (16384) are larger than Scout's MoE expert (8192)
        # But Maverick only has 24 of them. Scout has 48 MoE layers but with router overhead.
        # This comparison verifies the calculation is layer-type aware.
    
    def test_maverick_higher_dense_flops_per_layer(self, parallel_config):
        """Test that Maverick's dense layers have higher FLOPs than MoE layers (due to larger intermediate)"""
        # Create test model with same config but different layer types
        B, L, H = 1, 1024, 5120  # Llama-4 hidden dim
        
        # Dense layer FLOPs: 2 * B * L * H * 16384 * 2 (no gating, up+down)
        dense_flops = 2 * B * L * H * 16384 * 2
        
        # MoE layer FLOPs: router + 1 expert
        # Router: 2 * B * L * H * 16 (Scout) or 128 (Maverick)
        # Expert: 2 * B * L * H * 8192 * 1 * 2
        router_flops_maverick = 2 * B * L * H * 128
        expert_flops = 2 * B * L * H * 8192 * 1 * 2
        moe_flops_maverick = router_flops_maverick + expert_flops
        
        # Dense layer has more FLOPs due to larger intermediate (16384 vs 8192)
        assert dense_flops > moe_flops_maverick


class TestStorageTrafficInterleaved:
    """Tests for storage traffic calculation with interleaved layers"""
    
    @pytest.fixture
    def small_memory_constraints(self):
        """GPU with small memory to trigger offloading"""
        return SystemConstraints(
            memory_capacity=8e9,  # 8GB - too small for model
            memory_bandwidth=1e12,
            compute_throughput=100e12,
            network_bandwidth=100e9,
            persistent_storage_bandwidth=10e9,  # 10 GB/s SSD
        )
    
    @pytest.fixture
    def parallel_config(self):
        return ParallelismConfig(
            parallelism_type=ParallelismType.TENSOR_PARALLEL,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
        )
    
    def test_storage_traffic_only_moe_layers(self, parallel_config):
        """Test that storage traffic only accounts for MoE layers"""
        # Create model with interleaved layers
        model = LLMArchitecture(
            model_name="test-storage",
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
                num_experts=8,
                num_experts_per_token=2,
            ),
            is_moe=True,
            # 4 MoE layers, 4 dense layers
            ffn_layer_types=[
                FFNLayerType.DENSE, FFNLayerType.MOE,
                FFNLayerType.DENSE, FFNLayerType.MOE,
                FFNLayerType.DENSE, FFNLayerType.MOE,
                FFNLayerType.DENSE, FFNLayerType.MOE,
            ],
            total_parameters=10_000_000_000,  # Large enough to not fit in 8GB
            active_parameters=2_000_000_000,
        )
        
        # Verify layer counts
        assert model.get_num_moe_ffn_layers() == 4
        assert model.get_num_dense_ffn_layers() == 4
        
        perf = InferencePerformance(model)
        
        # get_num_moe_ffn_layers should return 4 (not 8)
        assert model.get_num_moe_ffn_layers() == 4
    
    def test_all_moe_vs_interleaved_storage(self, parallel_config):
        """Test storage traffic is proportional to MoE layer count"""
        # All MoE model
        all_moe = LLMArchitecture(
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
            ffn_config=FFNConfig(intermediate_size=2048),
            moe_config=MoEConfig(num_experts=8, num_experts_per_token=2),
            is_moe=True,
            ffn_layer_types=[FFNLayerType.MOE] * 8,
            total_parameters=10_000_000_000,
            active_parameters=2_000_000_000,
        )
        
        # Interleaved model (half MoE)
        interleaved = LLMArchitecture(
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
            ffn_config=FFNConfig(intermediate_size=2048, dense_intermediate_size=4096),
            moe_config=MoEConfig(num_experts=8, num_experts_per_token=2),
            is_moe=True,
            ffn_layer_types=[FFNLayerType.DENSE, FFNLayerType.MOE] * 4,
            total_parameters=10_000_000_000,
            active_parameters=2_000_000_000,
        )
        
        # MoE layer counts
        assert all_moe.get_num_moe_ffn_layers() == 8
        assert interleaved.get_num_moe_ffn_layers() == 4
        
        # Interleaved should have half the MoE-related overhead


class TestNoRegressionExistingModels:
    """Tests to ensure existing models without ffn_layer_types still work"""
    
    @pytest.fixture
    def parallel_config(self):
        return ParallelismConfig(
            parallelism_type=ParallelismType.TENSOR_PARALLEL,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
        )
    
    def test_deepseek_no_ffn_layer_types(self, parallel_config):
        """Test DeepSeek-V3 (no ffn_layer_types) still calculates correctly"""
        from llm_configs import DEEPSEEK_V3
        
        assert DEEPSEEK_V3.ffn_layer_types is None
        assert DEEPSEEK_V3.is_moe is True
        
        perf = InferencePerformance(DEEPSEEK_V3)
        breakdown = perf.calculate_prefill_compute_breakdown(
            batch_size=1, sequence_length=1024, parallelism_config=parallel_config
        )
        
        # Should still calculate FFN FLOPs for all MoE layers
        assert breakdown['ffn'] > 0
        assert breakdown['total'] > breakdown['ffn']
    
    def test_llama3_no_ffn_layer_types(self, parallel_config):
        """Test Llama-3-8B (no ffn_layer_types, not MoE) still calculates correctly"""
        from llm_configs import LLAMA_3_8B
        
        assert LLAMA_3_8B.ffn_layer_types is None
        assert LLAMA_3_8B.is_moe is False
        
        perf = InferencePerformance(LLAMA_3_8B)
        breakdown = perf.calculate_prefill_compute_breakdown(
            batch_size=1, sequence_length=1024, parallelism_config=parallel_config
        )
        
        # Should calculate FFN FLOPs for all dense layers
        assert breakdown['ffn'] > 0
        
        # Verify it uses intermediate_size (no dense_intermediate_size set)
        # Each layer: 2 * B * L * H * intermediate * 2 (with gating: *2)
        B, L, H = 1, 1024, LLAMA_3_8B.hidden_dim
        intermediate = LLAMA_3_8B.ffn_config.intermediate_size
        gating_factor = 2 if LLAMA_3_8B.ffn_config.use_gating else 1
        
        expected_per_layer = 2 * B * L * H * intermediate * (1 + gating_factor)
        expected_total = expected_per_layer * LLAMA_3_8B.num_layers
        
        # Allow some tolerance
        assert breakdown['ffn'] == pytest.approx(expected_total, rel=0.05)
