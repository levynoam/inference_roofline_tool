"""
Test DeepSeek 3.2 with MLA and sparse attention
Compare with standard models to show memory savings
"""

from llm_configs import get_model, list_models
from inference_performance import InferencePerformance, ParallelismConfig, ParallelismType


def test_deepseek_32_mla():
    """Test DeepSeek 3.2 with MLA"""
    print("=" * 80)
    print("DeepSeek 3.2 with MLA (Multi-head Latent Attention)")
    print("=" * 80)
    
    model = get_model("deepseek-3.2")
    print(f"\n{model.summary()}\n")
    
    perf = InferencePerformance(model)
    parallelism = ParallelismConfig(ParallelismType.NONE, 1, 1, 1)
    
    # Test with long context
    batch_size = 1
    sequence_length = 8192
    time_to_first_token = 1.0
    
    resources = perf.calculate_prefill_resources(
        batch_size, sequence_length, time_to_first_token, 1, parallelism
    )
    
    print(resources.summary())
    print("\n" + "=" * 80)


def compare_kv_cache_memory():
    """Compare KV cache memory between standard and MLA models"""
    print("=" * 80)
    print("KV CACHE MEMORY COMPARISON: Standard vs MLA")
    print("=" * 80)
    
    # Create a comparison table
    print(f"\n{'Model':<20} | {'Seq Len':>8} | {'KV Cache':>12} | {'Notes':<30}")
    print("-" * 85)
    
    sequence_lengths = [4096, 8192, 16384, 32768]
    
    # Llama 3 8B (standard)
    llama = get_model("llama-3-8b")
    print(f"\n{llama.model_name} (Standard GQA)")
    for seq_len in sequence_lengths:
        kv_size = llama.get_kv_cache_size(batch_size=1, sequence_length=seq_len)
        print(f"{'':20} | {seq_len:8} | {kv_size / (1024**3):11.2f}G | 32 heads, GQA (8 KV heads)")
    
    # DeepSeek 3.2 (MLA)
    deepseek = get_model("deepseek-3.2")
    print(f"\n{deepseek.model_name} (MLA Compressed)")
    for seq_len in sequence_lengths:
        kv_size = deepseek.get_kv_cache_size(batch_size=1, sequence_length=seq_len)
        compression_ratio = (128 * 128) / 512  # From full to compressed
        print(f"{'':20} | {seq_len:8} | {kv_size / (1024**3):11.2f}G | "
              f"MLA rank=512 ({compression_ratio:.0f}x compression)")
    
    # Calculate savings
    print(f"\n{'Memory Savings with MLA:'}")
    for seq_len in sequence_lengths:
        standard_size = llama.get_kv_cache_size(batch_size=1, sequence_length=seq_len)
        mla_size = deepseek.get_kv_cache_size(batch_size=1, sequence_length=seq_len)
        savings_pct = (1 - mla_size / standard_size) * 100
        print(f"  Seq {seq_len:5d}: {savings_pct:5.1f}% reduction "
              f"({standard_size / (1024**3):.2f}G -> {mla_size / (1024**3):.2f}G)")
    
    print("\n" + "=" * 80)


def compare_inference_performance():
    """Compare inference performance with and without MLA"""
    print("=" * 80)
    print("INFERENCE PERFORMANCE: Impact of MLA")
    print("=" * 80)
    
    batch_size = 1
    sequence_length = 8192
    time_to_first_token = 1.0
    parallelism = ParallelismConfig(ParallelismType.NONE, 1, 1, 1)
    
    print(f"\nWorkload: batch={batch_size}, seq_len={sequence_length}, TTFT={time_to_first_token}s\n")
    print(f"{'Model':<20} | {'Memory':>10} | {'KV Cache':>10} | {'Compute':>10} | {'Mem BW':>10} | {'Kernels':>8}")
    print("-" * 95)
    
    models = ["llama-3-8b", "deepseek-3.2"]
    
    for model_key in models:
        model = get_model(model_key)
        perf = InferencePerformance(model)
        
        resources = perf.calculate_prefill_resources(
            batch_size, sequence_length, time_to_first_token, 1, parallelism
        )
        
        mla_note = " (MLA)" if model.attention_config.use_mla else ""
        
        print(f"{model.model_name:<20} | "
              f"{resources.memory_per_gpu / (1024**3):9.2f}G | "
              f"{resources.memory_kv_cache / (1024**3):9.2f}G | "
              f"{resources.compute_flops_per_sec / 1e12:9.2f}T | "
              f"{resources.memory_bandwidth_per_gpu / (1024**3):9.2f}G | "
              f"{resources.num_kernel_launches:8}{mla_note}")
    
    print("\n" + "=" * 80)


def test_long_context_scaling():
    """Test how MLA helps with very long contexts"""
    print("=" * 80)
    print("LONG CONTEXT SCALING with MLA")
    print("=" * 80)
    
    model = get_model("deepseek-3.2")
    perf = InferencePerformance(model)
    parallelism = ParallelismConfig(ParallelismType.NONE, 1, 1, 1)
    
    batch_size = 1
    time_to_first_token = 2.0
    
    print(f"\nModel: {model.model_name} (MLA rank={model.attention_config.mla_kv_lora_rank})")
    print(f"Max context: {model.max_sequence_length:,} tokens")
    print(f"Target TTFT: {time_to_first_token}s\n")
    
    print(f"{'Seq Len':>10} | {'Total Mem':>10} | {'KV Cache':>10} | {'Compute':>10} | {'Feasible':>10}")
    print("-" * 70)
    
    # Test various context lengths up to max
    for seq_len in [8192, 16384, 32768, 65536, 131072]:
        if seq_len > model.max_sequence_length:
            break
        
        resources = perf.calculate_prefill_resources(
            batch_size, seq_len, time_to_first_token, 1, parallelism
        )
        
        # Check if feasible on typical GPU (80GB)
        gpu_memory = 80 * (1024**3)
        feasible = "✓ Yes" if resources.memory_per_gpu <= gpu_memory else "✗ No"
        
        print(f"{seq_len:10,} | "
              f"{resources.memory_per_gpu / (1024**3):9.2f}G | "
              f"{resources.memory_kv_cache / (1024**3):9.2f}G | "
              f"{resources.compute_flops_per_sec / 1e12:9.2f}T | "
              f"{feasible:>10}")
    
    print("\n" + "=" * 80)


def show_mla_details():
    """Show detailed MLA architecture information"""
    print("=" * 80)
    print("MLA ARCHITECTURE DETAILS")
    print("=" * 80)
    
    model = get_model("deepseek-3.2")
    
    print(f"\nModel: {model.model_name}")
    print(f"\nAttention Configuration:")
    print(f"  Number of attention heads: {model.attention_config.num_attention_heads}")
    print(f"  Head dimension: {model.attention_config.head_dim}")
    print(f"  Standard KV dimension: {model.attention_config.num_key_value_heads} heads × "
          f"{model.attention_config.head_dim} dim = "
          f"{model.attention_config.num_key_value_heads * model.attention_config.head_dim} total")
    
    print(f"\nMLA Compression:")
    print(f"  KV LoRA rank: {model.attention_config.mla_kv_lora_rank}")
    print(f"  Q LoRA rank: {model.attention_config.mla_q_lora_rank}")
    
    full_kv_dim = model.attention_config.num_key_value_heads * model.attention_config.head_dim
    compression_ratio = full_kv_dim / model.attention_config.mla_kv_lora_rank
    print(f"  Compression ratio: {compression_ratio:.1f}x "
          f"({full_kv_dim} -> {model.attention_config.mla_kv_lora_rank})")
    
    if model.attention_config.use_sparse_attention:
        print(f"\nSparse Attention (DSA):")
        print(f"  Block size: {model.attention_config.sparse_block_size}")
        print(f"  Local blocks: {model.attention_config.sparse_local_blocks} "
              f"({model.attention_config.sparse_local_blocks * model.attention_config.sparse_block_size} tokens)")
        print(f"  Global blocks: {model.attention_config.sparse_global_blocks} "
              f"({model.attention_config.sparse_global_blocks * model.attention_config.sparse_block_size} tokens)")
        
        total_attended = (model.attention_config.sparse_local_blocks + 
                         model.attention_config.sparse_global_blocks) * model.attention_config.sparse_block_size
        print(f"  Total attention span: {total_attended} tokens per query")
    
    print(f"\nMemory Impact (batch=1, seq=8192):")
    kv_cache = model.get_kv_cache_size(batch_size=1, sequence_length=8192)
    print(f"  KV cache size: {kv_cache / (1024**3):.2f} GB")
    
    # Compare to standard
    standard_kv = (2 * 1 * model.attention_config.num_key_value_heads * 
                   8192 * model.attention_config.head_dim * model.num_layers * 2)
    print(f"  Standard KV (no compression): {standard_kv / (1024**3):.2f} GB")
    print(f"  Memory saved: {(standard_kv - kv_cache) / (1024**3):.2f} GB "
          f"({(1 - kv_cache/standard_kv)*100:.1f}% reduction)")
    
    print("\n" + "=" * 80)


def main():
    """Run all MLA tests"""
    # Show all available models including new one
    list_models()
    print("\n")
    
    # Test DeepSeek 3.2
    test_deepseek_32_mla()
    print("\n")
    
    # Show MLA architecture details
    show_mla_details()
    print("\n")
    
    # Compare KV cache memory
    compare_kv_cache_memory()
    print("\n")
    
    # Compare inference performance
    compare_inference_performance()
    print("\n")
    
    # Test long context scaling
    test_long_context_scaling()


if __name__ == "__main__":
    main()
