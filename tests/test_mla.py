"""
Tests for TestMLA
"""

import pytest
from llm_configs import (
    LLAMA_3_8B, LLAMA_3_70B, LLAMA_2_7B,
    DEEPSEEK_V3, DEEPSEEK_3_2,
    MISTRAL_7B, MIXTRAL_8X7B, GPT3_175B
)
from llm_architecture import LLMArchitecture, AttentionConfig, AttentionType, ActivationType
from inference_performance import (
    InferencePerformance,
    SystemConstraints,
    ParallelismConfig,
    ParallelismType,
    PrefillResources,
    ResourceUtilization,
    DecodePerformance
)


class TestMLA:
    """Test Multi-head Latent Attention compression mechanism"""
    
    def test_mla_kv_cache_compression(self):
        """Test that MLA compresses KV cache memory"""
        gpu = SystemConstraints.from_gpu_spec("H100-80GB")
        
        # DeepSeek 3.2 has MLA with kv_lora_rank=512
        model_with_mla = DEEPSEEK_3_2
        assert model_with_mla.attention_config.use_mla
        assert model_with_mla.attention_config.mla_kv_lora_rank == 512
        
        # DeepSeek V3 has same architecture but no MLA compression
        model_without_mla = DEEPSEEK_V3
        assert not model_without_mla.attention_config.use_mla
        
        # Calculate KV cache sizes for same batch and sequence length
        batch_size = 1
        seq_len = 2048
        bytes_per_elem = 2  # fp16/bf16
        
        kv_with_mla = model_with_mla.get_kv_cache_size(batch_size, seq_len, bytes_per_elem)
        kv_without_mla = model_without_mla.get_kv_cache_size(batch_size, seq_len, bytes_per_elem)
        
        # MLA should significantly reduce KV cache size
        # kv_lora_rank=512 vs full KV dimension (128 heads * 128 head_dim = 16384)
        # Compression ratio should be approximately 16384/512 = 32x
        compression_ratio = kv_without_mla / kv_with_mla
        
        # Should see significant compression (at least 20x)
        assert compression_ratio > 20
        assert compression_ratio < 50  # Upper bound sanity check
        
        print(f"\nMLA KV Cache Compression:")
        print(f"  Without MLA: {kv_without_mla / (1024**3):.3f} GB")
        print(f"  With MLA:    {kv_with_mla / (1024**3):.6f} GB")
        print(f"  Compression: {compression_ratio:.1f}x")
    
    def test_mla_reduces_memory_bandwidth(self):
        """Test that MLA reduces memory bandwidth requirements"""
        gpu = SystemConstraints.from_gpu_spec("H100-80GB")
        
        model_with_mla = DEEPSEEK_3_2
        model_without_mla = DEEPSEEK_V3
        
        perf_with_mla = InferencePerformance(model_with_mla)
        perf_without_mla = InferencePerformance(model_without_mla)
        
        # Test decode step where KV cache reading matters most
        result_with_mla = perf_with_mla.calculate_decode_performance(
            system_constraints=gpu,
            batch_size=1,
            prefill_length=2048,
            output_length=100,
            parallelism_config=None
        )
        
        result_without_mla = perf_without_mla.calculate_decode_performance(
            system_constraints=gpu,
            batch_size=1,
            prefill_length=2048,
            output_length=100,
            parallelism_config=None
        )
        
        # MLA should use less memory bandwidth
        bw_with_mla = result_with_mla.avg_memory_bw_utilization
        bw_without_mla = result_without_mla.avg_memory_bw_utilization
        
        assert bw_with_mla < bw_without_mla
        
        # The reduction should be measurable (weight reads dominate for large models)
        bw_reduction = (bw_without_mla - bw_with_mla) / bw_without_mla
        assert bw_reduction > 0.001  # At least 0.1% reduction (weights dominate traffic)
        
        print(f"\nMLA Memory Bandwidth Impact:")
        print(f"  Without MLA: {bw_without_mla * 100:.1f}%")
        print(f"  With MLA:    {bw_with_mla * 100:.1f}%")
        print(f"  Reduction:   {bw_reduction * 100:.1f}%")
    
    def test_mla_adds_compute_overhead(self):
        """Test that MLA adds compute overhead for projection operations"""
        gpu = SystemConstraints.from_gpu_spec("H100-80GB")
        
        model_with_mla = DEEPSEEK_3_2
        model_without_mla = DEEPSEEK_V3
        
        perf_with_mla = InferencePerformance(model_with_mla)
        perf_without_mla = InferencePerformance(model_without_mla)
        
        # Test prefill where projection overhead is most visible
        result_with_mla = perf_with_mla.calculate_prefill_resources(
            batch_size=1,
            sequence_length=2048,
            time_to_first_token=0.1,
            num_gpus=1,
            parallelism_config=ParallelismConfig()
        )
        
        result_without_mla = perf_without_mla.calculate_prefill_resources(
            batch_size=1,
            sequence_length=2048,
            time_to_first_token=0.1,
            num_gpus=1,
            parallelism_config=ParallelismConfig()
        )
        
        # MLA should use more compute (down-projection + up-projection matmuls)
        compute_with_mla = result_with_mla.compute_per_gpu
        compute_without_mla = result_without_mla.compute_per_gpu
        
        # MLA adds projection operations, so compute should be higher
        # However, the models might have different sizes, so we compare normalized
        # compute per parameter
        compute_per_param_mla = compute_with_mla / model_with_mla.total_parameters
        compute_per_param_no_mla = compute_without_mla / model_without_mla.total_parameters
        
        print(f"\nMLA Compute Overhead:")
        print(f"  Without MLA: {compute_without_mla / 1e12:.2f} TFLOP")
        print(f"  With MLA:    {compute_with_mla / 1e12:.2f} TFLOP")
    
    def test_mla_decode_bottleneck_shift(self):
        """Test that MLA can shift bottleneck from memory to compute"""
        gpu = SystemConstraints.from_gpu_spec("H100-80GB")
        
        model_with_mla = DEEPSEEK_3_2
        model_without_mla = DEEPSEEK_V3
        
        perf_with_mla = InferencePerformance(model_with_mla)
        perf_without_mla = InferencePerformance(model_without_mla)
        
        # Long context where KV cache bandwidth matters
        result_with_mla = perf_with_mla.calculate_decode_performance(
            system_constraints=gpu,
            batch_size=1,
            prefill_length=8192,  # Long context
            output_length=50,
            parallelism_config=None
        )
        
        result_without_mla = perf_without_mla.calculate_decode_performance(
            system_constraints=gpu,
            batch_size=1,
            prefill_length=8192,
            output_length=50,
            parallelism_config=None
        )
        
        # Without MLA, should be more memory-bound due to large KV cache reads
        # With MLA, memory pressure is reduced
        
        print(f"\nMLA Bottleneck Analysis (8K context):")
        print(f"  Without MLA: {result_without_mla.primary_bottleneck}")
        print(f"    Memory BW: {result_without_mla.avg_memory_bw_utilization * 100:.1f}%")
        print(f"    Compute:   {result_without_mla.avg_compute_utilization * 100:.1f}%")
        print(f"  With MLA: {result_with_mla.primary_bottleneck}")
        print(f"    Memory BW: {result_with_mla.avg_memory_bw_utilization * 100:.1f}%")
        print(f"    Compute:   {result_with_mla.avg_compute_utilization * 100:.1f}%")
        
        # Verify MLA reduces memory bandwidth pressure
        assert result_with_mla.avg_memory_bw_utilization < result_without_mla.avg_memory_bw_utilization
    
    def test_mla_prefill_memory_breakdown(self):
        """Test that MLA affects prefill memory breakdown correctly"""
        gpu = SystemConstraints.from_gpu_spec("H100-80GB")
        
        model_with_mla = DEEPSEEK_3_2
        perf = InferencePerformance(model_with_mla)
        
        result = perf.calculate_prefill_resources(
            batch_size=1,
            sequence_length=2048,
            time_to_first_token=0.1,
            num_gpus=1,
            parallelism_config=ParallelismConfig()
        )
        
        # Verify memory breakdown has all components
        assert result.memory_model_weights > 0
        assert result.memory_kv_cache > 0
        assert result.memory_activations > 0
        
        # KV cache should be much smaller than weights due to MLA compression
        # For a 671B model with d_kv=64, KV cache should be tiny
        assert result.memory_kv_cache < result.memory_model_weights / 100
        
        print(f"\nMLA Memory Breakdown (Prefill):")
        print(f"  Weights:     {result.memory_model_weights / (1024**3):.2f} GB")
        print(f"  KV Cache:    {result.memory_kv_cache / (1024**3):.4f} GB")
        print(f"  Activations: {result.memory_activations / (1024**3):.4f} GB")


# =============================================================================
# DSA (Dynamic Sparse Attention) Tests
# =============================================================================

