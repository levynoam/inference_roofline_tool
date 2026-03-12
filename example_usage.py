"""
Example usage of LLM architecture data structures
Demonstrates how to use the flexible architecture modeling
"""

from llm_configs import get_model, list_models, ALL_MODELS
from llm_architecture import (
    LLMArchitecture, AttentionConfig, FFNConfig, MoEConfig,
    ArchitectureType, AttentionType, ActivationType,
    NormalizationType, PositionEncodingType
)


def compare_models(model_keys: list[str]):
    """Compare multiple models side by side"""
    print("\n" + "=" * 100)
    print("MODEL COMPARISON")
    print("=" * 100)
    
    models = [get_model(key) for key in model_keys]
    
    # Compare key metrics
    metrics = [
        ("Model", lambda m: m.model_name),
        ("Total Params", lambda m: f"{m.total_parameters / 1e9:.1f}B"),
        ("Active Params", lambda m: f"{m.active_parameters / 1e9:.1f}B" if m.is_moe else "N/A"),
        ("Layers", lambda m: m.num_layers),
        ("Hidden Dim", lambda m: m.hidden_dim),
        ("Attn Heads", lambda m: m.attention_config.num_attention_heads),
        ("KV Heads", lambda m: m.attention_config.num_key_value_heads),
        ("FFN Size", lambda m: m.ffn_config.intermediate_size),
        ("Vocab", lambda m: m.vocab_size),
        ("Max Seq", lambda m: m.max_sequence_length),
        ("MoE", lambda m: f"{m.moe_config.num_experts}x{m.moe_config.num_experts_per_token}" if m.is_moe else "No"),
    ]
    
    # Print comparison table
    for metric_name, metric_fn in metrics:
        row = f"{metric_name:15s} |"
        for model in models:
            value = str(metric_fn(model))
            row += f" {value:15s} |"
        print(row)
    
    print("=" * 100)


def analyze_memory_scaling(model_key: str, batch_sizes: list[int], seq_lengths: list[int]):
    """Analyze how memory scales with batch size and sequence length"""
    model = get_model(model_key)
    
    print(f"\n{'=' * 80}")
    print(f"MEMORY SCALING ANALYSIS: {model.model_name}")
    print(f"{'=' * 80}")
    
    print(f"\n{'Batch':>8s} {'SeqLen':>8s} | {'Model':>10s} {'KV Cache':>10s} {'Acts':>10s} {'Total':>10s}")
    print("-" * 80)
    
    for bs in batch_sizes:
        for seq_len in seq_lengths:
            memory = model.get_memory_footprint(batch_size=bs, sequence_length=seq_len)
            print(
                f"{bs:8d} {seq_len:8d} | "
                f"{memory['model_parameters'] / (1024**3):9.2f}G "
                f"{memory['kv_cache'] / (1024**3):9.2f}G "
                f"{memory['activations'] / (1024**3):9.2f}G "
                f"{memory['total'] / (1024**3):9.2f}G"
            )
    
    print("=" * 80)


def create_custom_model_example():
    """Example: Create a custom model configuration"""
    print("\n" + "=" * 80)
    print("CREATING CUSTOM MODEL: Hypothetical Llama 4 Maverick")
    print("=" * 80)
    
    # Hypothetical Llama 4 Maverick with advanced features
    llama4_maverick = LLMArchitecture(
        model_name="Llama-4-Maverick-16B",
        model_family="Llama",
        version="4.0",
        architecture_type=ArchitectureType.DECODER_ONLY,
        
        # Larger than 8B, smaller than 70B
        num_layers=40,
        hidden_dim=5120,
        vocab_size=128256,
        max_sequence_length=131072,  # 128K context
        
        # Advanced attention with more aggressive GQA
        attention_config=AttentionConfig(
            num_attention_heads=40,
            num_key_value_heads=4,  # Very aggressive GQA ratio
            attention_type=AttentionType.GROUPED_QUERY,
            head_dim=128,
        ),
        
        # Larger FFN
        ffn_config=FFNConfig(
            intermediate_size=18432,
            activation=ActivationType.SWIGLU,
            use_gating=True,
        ),
        
        # Advanced features
        normalization_type=NormalizationType.RMS_NORM,
        position_encoding=PositionEncodingType.ROTARY,
        rope_theta=1000000.0,  # Higher for longer context
        tie_word_embeddings=False,
        dtype="bfloat16",
        
        # Add custom metadata
        metadata={
            "training_tokens": "15T",
            "release_date": "2025-Q4",
            "optimizations": ["flash_attention_3", "paged_attention"],
            "license": "Llama 4 License"
        }
    )
    
    print(llama4_maverick.summary())
    print("\nEstimated Parameters:", f"{llama4_maverick.total_parameters / 1e9:.2f}B")
    
    # Memory analysis
    print("\nMemory Requirements:")
    memory = llama4_maverick.get_memory_footprint(batch_size=1, sequence_length=8192)
    for component, size in memory.items():
        print(f"  {component:20s}: {size / (1024**3):8.2f} GB")
    
    print("=" * 80)
    
    return llama4_maverick


def analyze_moe_efficiency(dense_model_key: str, moe_model_key: str):
    """Compare dense vs MoE model efficiency"""
    dense = get_model(dense_model_key)
    moe = get_model(moe_model_key)
    
    print(f"\n{'=' * 80}")
    print(f"DENSE vs MoE COMPARISON")
    print(f"{'=' * 80}")
    
    print(f"\n{'Metric':<30s} | {'Dense':>20s} | {'MoE':>20s}")
    print("-" * 80)
    
    print(f"{'Model':<30s} | {dense.model_name:>20s} | {moe.model_name:>20s}")
    print(f"{'Total Parameters':<30s} | {dense.total_parameters / 1e9:19.1f}B | {moe.total_parameters / 1e9:19.1f}B")
    print(f"{'Active Parameters':<30s} | {dense.active_parameters / 1e9:19.1f}B | {moe.active_parameters / 1e9:19.1f}B")
    
    if moe.is_moe:
        utilization = (moe.active_parameters / moe.total_parameters) * 100
        print(f"{'Parameter Utilization':<30s} | {'100.0%':>20s} | {utilization:19.1f}%")
        print(f"{'Experts (Total/Active)':<30s} | {'1/1':>20s} | {f'{moe.moe_config.num_experts}/{moe.moe_config.num_experts_per_token}':>20s}")
    
    # Memory comparison
    dense_mem = dense.get_memory_footprint(batch_size=1, sequence_length=2048)
    moe_mem = moe.get_memory_footprint(batch_size=1, sequence_length=2048)
    
    print(f"\n{'Memory (bs=1, seq=2048)':<30s} | {'Dense':>20s} | {'MoE':>20s}")
    print("-" * 80)
    print(f"{'Model Weights':<30s} | {dense_mem['model_parameters'] / (1024**3):19.2f}G | {moe_mem['model_parameters'] / (1024**3):19.2f}G")
    print(f"{'KV Cache':<30s} | {dense_mem['kv_cache'] / (1024**3):19.2f}G | {moe_mem['kv_cache'] / (1024**3):19.2f}G")
    print(f"{'Total':<30s} | {dense_mem['total'] / (1024**3):19.2f}G | {moe_mem['total'] / (1024**3):19.2f}G")
    
    print("=" * 80)


def main():
    """Run all examples"""
    # List all available models
    list_models()
    
    # Compare multiple dense models
    print("\n\n=== Comparing Llama Family ===")
    compare_models(["llama-2-7b", "llama-3-8b", "llama-3-70b"])
    
    # Compare MoE variants
    print("\n\n=== Comparing MoE Models ===")
    compare_models(["mistral-7b", "mixtral-8x7b", "deepseek-v3"])
    
    # Memory scaling analysis
    analyze_memory_scaling(
        "llama-3-8b",
        batch_sizes=[1, 4, 8],
        seq_lengths=[1024, 2048, 4096]
    )
    
    # Create custom model
    custom_model = create_custom_model_example()
    
    # MoE efficiency analysis
    analyze_moe_efficiency("mistral-7b", "mixtral-8x7b")
    
    # Detailed model analysis
    print("\n\n=== Detailed Analysis: DeepSeek V3 ===")
    deepseek = get_model("deepseek-v3")
    print(deepseek.summary())
    
    print("\nDeepSeek V3 KV Cache Scaling:")
    for seq_len in [4096, 8192, 16384, 32768, 65536, 131072]:
        kv_size = deepseek.get_kv_cache_size(batch_size=1, sequence_length=seq_len)
        print(f"  Seq Length {seq_len:6d}: {kv_size / (1024**3):6.2f} GB")


if __name__ == "__main__":
    main()
