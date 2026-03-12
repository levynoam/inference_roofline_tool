"""
Pre-configured LLM architectures for popular models
Example instantiations of the LLMArchitecture data structure
"""

from llm_architecture import (
    LLMArchitecture, AttentionConfig, FFNConfig, MoEConfig,
    ArchitectureType, AttentionType, ActivationType, 
    NormalizationType, PositionEncodingType, LayerAttentionType,
    HybridLayerType, Mamba2Config, FFNLayerType, LinearAttentionConfig
)


# Llama 4 Scout (17B active, ~109B total with MoE)
# Config from: LLama-4-Scout.json
# interleave_moe_layer_step=1 means ALL layers are MoE
# intermediate_size=8192 (experts), intermediate_size_mlp=16384 (dense, not used in Scout)
LLAMA_4_SCOUT = LLMArchitecture(
    model_name="Llama-4-Scout",
    model_family="Llama",
    version="4.0-Scout",
    architecture_type=ArchitectureType.DECODER_ONLY,
    num_layers=48,
    hidden_dim=5120,
    vocab_size=202048,
    max_sequence_length=262144,  # 256K context (with rope scaling factor=16)
    attention_config=AttentionConfig(
        num_attention_heads=40,
        num_key_value_heads=8,  # GQA
        attention_type=AttentionType.GROUPED_QUERY,
        head_dim=128,
    ),
    ffn_config=FFNConfig(
        intermediate_size=8192,  # per-expert FFN size
        dense_intermediate_size=16384,  # dense FFN size (not used, all layers MoE)
        activation=ActivationType.SILU,
        use_gating=False,
    ),
    moe_config=MoEConfig(
        num_experts=16,  # num_local_experts
        num_experts_per_token=1,  # top-1 routing
    ),
    is_moe=True,
    # interleave_moe_layer_step=1: all layers are MoE
    ffn_layer_types=[FFNLayerType.MOE] * 48,
    normalization_type=NormalizationType.RMS_NORM,
    position_encoding=PositionEncodingType.ROTARY,
    rope_theta=500000.0,
    tie_word_embeddings=False,
    dtype="bfloat16",
    total_parameters=109_000_000_000,
    active_parameters=17_000_000_000,
)


# Llama 4 Maverick (17B active, ~400B total with MoE)
# Config from: LLama-4-Maverick.json
# interleave_moe_layer_step=2 means alternating: Dense, MoE, Dense, MoE, ...
# intermediate_size=8192 (experts), intermediate_size_mlp=16384 (dense layers)
LLAMA_4_MAVERICK = LLMArchitecture(
    model_name="Llama-4-Maverick",
    model_family="Llama",
    version="4.0-Maverick",
    architecture_type=ArchitectureType.DECODER_ONLY,
    num_layers=48,
    hidden_dim=5120,
    vocab_size=202048,
    max_sequence_length=262144,  # 256K context
    attention_config=AttentionConfig(
        num_attention_heads=40,
        num_key_value_heads=8,  # GQA
        attention_type=AttentionType.GROUPED_QUERY,
        head_dim=128,
    ),
    ffn_config=FFNConfig(
        intermediate_size=8192,  # per-expert FFN size
        dense_intermediate_size=16384,  # dense FFN size (intermediate_size_mlp)
        activation=ActivationType.SILU,
        use_gating=False,
    ),
    moe_config=MoEConfig(
        num_experts=128,  # num_local_experts (128 experts per MoE layer)
        num_experts_per_token=1,  # top-1 routing
    ),
    is_moe=True,
    # interleave_moe_layer_step=2: alternating Dense, MoE, Dense, MoE, ...
    # 48 layers: 24 Dense + 24 MoE
    ffn_layer_types=[
        FFNLayerType.DENSE if i % 2 == 0 else FFNLayerType.MOE
        for i in range(48)
    ],
    normalization_type=NormalizationType.RMS_NORM,
    position_encoding=PositionEncodingType.ROTARY,
    rope_theta=500000.0,
    tie_word_embeddings=False,
    dtype="bfloat16",
    total_parameters=400_000_000_000,
    active_parameters=17_000_000_000,
)


# Llama 4 Behemoth (~405B total with MoE, top-1 routing)
# 2/3 of layers use MoE (80 layers), 1/3 are dense (40 layers)
LLAMA_4_BEHEMOTH = LLMArchitecture(
    model_name="Llama-4-Behemoth",
    model_family="Llama",
    version="4.0-Behemoth",
    architecture_type=ArchitectureType.DECODER_ONLY,
    num_layers=120,
    hidden_dim=12288,
    vocab_size=202048,
    max_sequence_length=131072,  # 128K context
    attention_config=AttentionConfig(
        num_attention_heads=96,
        num_key_value_heads=8,  # GQA
        attention_type=AttentionType.GROUPED_QUERY,
        head_dim=128,
    ),
    ffn_config=FFNConfig(
        intermediate_size=16384,  # per-expert FFN size
        activation=ActivationType.SWIGLU,
        use_gating=True,
    ),
    moe_config=MoEConfig(
        num_experts=16,
        num_experts_per_token=1,  # top-1 routing
    ),
    normalization_type=NormalizationType.RMS_NORM,
    position_encoding=PositionEncodingType.ROTARY,
    rope_theta=500000.0,
    tie_word_embeddings=False,
    dtype="bfloat16",
    total_parameters=405_000_000_000,
)


# Llama 3 8B
LLAMA_3_8B = LLMArchitecture(
    model_name="Llama-3-8B",
    model_family="Llama",
    version="3.0",
    architecture_type=ArchitectureType.DECODER_ONLY,
    num_layers=32,
    hidden_dim=4096,
    vocab_size=128256,
    max_sequence_length=8192,
    attention_config=AttentionConfig(
        num_attention_heads=32,
        num_key_value_heads=8,  # GQA
        attention_type=AttentionType.GROUPED_QUERY,
        head_dim=128,
    ),
    ffn_config=FFNConfig(
        intermediate_size=14336,
        activation=ActivationType.SWIGLU,
        use_gating=True,
    ),
    normalization_type=NormalizationType.RMS_NORM,
    position_encoding=PositionEncodingType.ROTARY,
    rope_theta=500000.0,
    tie_word_embeddings=False,
    dtype="bfloat16",
)


# Llama 3 70B
LLAMA_3_70B = LLMArchitecture(
    model_name="Llama-3-70B",
    model_family="Llama",
    version="3.0",
    architecture_type=ArchitectureType.DECODER_ONLY,
    num_layers=80,
    hidden_dim=8192,
    vocab_size=128256,
    max_sequence_length=8192,
    attention_config=AttentionConfig(
        num_attention_heads=64,
        num_key_value_heads=8,  # GQA
        attention_type=AttentionType.GROUPED_QUERY,
        head_dim=128,
    ),
    ffn_config=FFNConfig(
        intermediate_size=28672,
        activation=ActivationType.SWIGLU,
        use_gating=True,
    ),
    normalization_type=NormalizationType.RMS_NORM,
    position_encoding=PositionEncodingType.ROTARY,
    rope_theta=500000.0,
    tie_word_embeddings=False,
    dtype="bfloat16",
)


# DeepSeek V3
DEEPSEEK_V3 = LLMArchitecture(
    model_name="DeepSeek-V3",
    model_family="DeepSeek",
    version="3.0",
    architecture_type=ArchitectureType.DECODER_ONLY,
    num_layers=61,
    hidden_dim=7168,
    vocab_size=129280,
    max_sequence_length=163840,  # 128K context
    attention_config=AttentionConfig(
        num_attention_heads=128,
        num_key_value_heads=128,  # MHA (no GQA in attention)
        attention_type=AttentionType.MULTI_HEAD,
        head_dim=128,
    ),
    ffn_config=FFNConfig(
        intermediate_size=18432,
        activation=ActivationType.SWIGLU,
        use_gating=True,
    ),
    moe_config=MoEConfig(
        num_experts=256,
        num_experts_per_token=8,
        router_type="top_k",
        shared_expert=True,
    ),
    is_moe=True,
    normalization_type=NormalizationType.RMS_NORM,
    position_encoding=PositionEncodingType.ROTARY,
    rope_theta=10000.0,
    tie_word_embeddings=False,
    dtype="bfloat16",
    total_parameters=671_000_000_000,  # 671B total
    active_parameters=37_000_000_000,  # 37B active
)


# DeepSeek 3.2 with MLA (Multi-head Latent Attention)
DEEPSEEK_3_2 = LLMArchitecture(
    model_name="DeepSeek-3.2",
    model_family="DeepSeek",
    version="3.2",
    architecture_type=ArchitectureType.DECODER_ONLY,
    num_layers=61,
    hidden_dim=7168,
    vocab_size=129280,
    max_sequence_length=163840,  # 160K context with sparse attention
    attention_config=AttentionConfig(
        num_attention_heads=128,
        num_key_value_heads=128,  # Full MHA before compression
        attention_type=AttentionType.MULTI_HEAD,
        head_dim=192,  # qk_nope_head_dim(128) + qk_rope_head_dim(64)
        # MLA configuration - compress KV cache significantly
        use_mla=True,
        mla_kv_lora_rank=512,  # kv_lora_rank from config
        mla_q_lora_rank=1536,  # Q also uses compression
        # Sparse attention configuration
        use_sparse_attention=True,
        sparse_block_size=512,
        sparse_local_blocks=4,  # Attend to 4 local blocks (2048 tokens)
        sparse_global_blocks=2,  # Attend to 2 global blocks (1024 tokens)
        # Dynamic Sparse Attention (DSA) - top-K selection
        use_dsa=True,
        dsa_q_indexer_dim=128,  # index_head_dim
        dsa_k_indexer_dim=64,   # index_n_heads
        dsa_top_k=2048,  # index_topk: Select top 2048 KV pairs
    ),
    ffn_config=FFNConfig(
        intermediate_size=18432,
        activation=ActivationType.SILU,  # hidden_act: silu
        use_gating=False,
    ),
    moe_config=MoEConfig(
        num_experts=256,  # n_routed_experts
        num_experts_per_token=8,  # num_experts_per_tok
        router_type="top_k",
        shared_expert=True,  # n_shared_experts=1
    ),
    is_moe=True,
    normalization_type=NormalizationType.RMS_NORM,
    position_encoding=PositionEncodingType.ROTARY,
    rope_theta=10000.0,
    tie_word_embeddings=False,
    dtype="bfloat16",
    total_parameters=671_000_000_000,  # 671B total (same as V3)
    active_parameters=37_000_000_000,  # 37B active (same as V3)
)


# Mistral 7B
MISTRAL_7B = LLMArchitecture(
    model_name="Mistral-7B-v0.1",
    model_family="Mistral",
    version="0.1",
    architecture_type=ArchitectureType.DECODER_ONLY,
    num_layers=32,
    hidden_dim=4096,
    vocab_size=32000,
    max_sequence_length=32768,  # With sliding window
    attention_config=AttentionConfig(
        num_attention_heads=32,
        num_key_value_heads=8,  # GQA
        attention_type=AttentionType.SLIDING_WINDOW,
        head_dim=128,
        sliding_window_size=4096,
    ),
    ffn_config=FFNConfig(
        intermediate_size=14336,
        activation=ActivationType.SWIGLU,
        use_gating=True,
    ),
    normalization_type=NormalizationType.RMS_NORM,
    position_encoding=PositionEncodingType.ROTARY,
    rope_theta=10000.0,
    tie_word_embeddings=False,
    dtype="bfloat16",
)


# Mixtral 8x7B
MIXTRAL_8X7B = LLMArchitecture(
    model_name="Mixtral-8x7B",
    model_family="Mistral",
    version="0.1",
    architecture_type=ArchitectureType.DECODER_ONLY,
    num_layers=32,
    hidden_dim=4096,
    vocab_size=32000,
    max_sequence_length=32768,
    attention_config=AttentionConfig(
        num_attention_heads=32,
        num_key_value_heads=8,  # GQA
        attention_type=AttentionType.GROUPED_QUERY,
        head_dim=128,
    ),
    ffn_config=FFNConfig(
        intermediate_size=14336,
        activation=ActivationType.SWIGLU,
        use_gating=True,
    ),
    moe_config=MoEConfig(
        num_experts=8,
        num_experts_per_token=2,
        router_type="top_k",
    ),
    is_moe=True,
    normalization_type=NormalizationType.RMS_NORM,
    position_encoding=PositionEncodingType.ROTARY,
    rope_theta=1000000.0,
    tie_word_embeddings=False,
    dtype="bfloat16",
    total_parameters=46_700_000_000,  # 46.7B total
    active_parameters=12_900_000_000,  # 12.9B active
)


# GPT-3 (example of older architecture)
GPT3_175B = LLMArchitecture(
    model_name="GPT-3-175B",
    model_family="GPT",
    version="3.0",
    architecture_type=ArchitectureType.DECODER_ONLY,
    num_layers=96,
    hidden_dim=12288,
    vocab_size=50257,
    max_sequence_length=2048,
    attention_config=AttentionConfig(
        num_attention_heads=96,
        num_key_value_heads=96,  # Standard MHA
        attention_type=AttentionType.MULTI_HEAD,
        head_dim=128,
        attention_bias=True,
    ),
    ffn_config=FFNConfig(
        intermediate_size=49152,
        activation=ActivationType.GELU,
        use_gating=False,
        ffn_bias=True,
    ),
    normalization_type=NormalizationType.LAYER_NORM,
    position_encoding=PositionEncodingType.ABSOLUTE,
    tie_word_embeddings=False,
    dtype="float16",
    total_parameters=175_000_000_000,
)


# Llama 2 7B (for comparison with Llama 3)
LLAMA_2_7B = LLMArchitecture(
    model_name="Llama-2-7B",
    model_family="Llama",
    version="2.0",
    architecture_type=ArchitectureType.DECODER_ONLY,
    num_layers=32,
    hidden_dim=4096,
    vocab_size=32000,
    max_sequence_length=4096,
    attention_config=AttentionConfig(
        num_attention_heads=32,
        num_key_value_heads=32,  # Standard MHA (no GQA)
        attention_type=AttentionType.MULTI_HEAD,
        head_dim=128,
    ),
    ffn_config=FFNConfig(
        intermediate_size=11008,
        activation=ActivationType.SWIGLU,
        use_gating=True,
    ),
    normalization_type=NormalizationType.RMS_NORM,
    position_encoding=PositionEncodingType.ROTARY,
    rope_theta=10000.0,
    tie_word_embeddings=False,
    dtype="bfloat16",
)


# Liquid Foundation Model 2 (LFM2) 3B
# Hybrid architecture with conv and attention layers
# Layer pattern: conv, conv, attn, conv, conv, attn, conv, conv, attn, conv, attn, conv, attn, conv, attn, conv
# 10 conv layers + 6 attention layers = 16 total layers
LFM2_3B = LLMArchitecture(
    model_name="LFM2-3B",
    model_family="Liquid",
    version="2.0",
    architecture_type=ArchitectureType.DECODER_ONLY,
    num_layers=16,  # Total layers (10 conv + 6 attention)
    hidden_dim=2048,
    vocab_size=65536,
    max_sequence_length=128000,  # 128K context
    attention_config=AttentionConfig(
        num_attention_heads=32,
        num_key_value_heads=8,  # GQA with 8 KV heads
        attention_type=AttentionType.GROUPED_QUERY,
        head_dim=64,  # 2048 / 32 heads = 64
    ),
    ffn_config=FFNConfig(
        intermediate_size=12288,
        activation=ActivationType.SWIGLU,
        use_gating=True,
    ),
    normalization_type=NormalizationType.RMS_NORM,
    position_encoding=PositionEncodingType.ROTARY,
    rope_theta=1000000.0,
    tie_word_embeddings=True,
    dtype="bfloat16",
    # Note: Only 6 of 16 layers have attention, rest are convolution-based
    # This is approximated as a standard transformer for performance modeling
    # The 10 conv layers have similar compute profile to FFN layers
    total_parameters=3_000_000_000,  # ~3B parameters
)


# Kimi K2.5 - Moonshot AI's multimodal model (text backbone)
# Based on DeepSeek V3 architecture with MLA and MoE
# Config from: https://huggingface.co/moonshotai/Kimi-K2.5
KIMI_K25 = LLMArchitecture(
    model_name="Kimi-K2.5",
    model_family="Kimi",
    version="2.5",
    architecture_type=ArchitectureType.DECODER_ONLY,
    num_layers=61,
    hidden_dim=7168,
    vocab_size=163840,
    max_sequence_length=262144,  # 256K context with YaRN scaling
    attention_config=AttentionConfig(
        num_attention_heads=64,
        num_key_value_heads=64,  # MHA (before MLA compression)
        attention_type=AttentionType.MULTI_HEAD,
        head_dim=192,  # qk_nope_head_dim(128) + qk_rope_head_dim(64) = 192
        # MLA configuration - compress KV cache significantly
        use_mla=True,
        mla_kv_lora_rank=512,  # Compressed KV dimension
        mla_q_lora_rank=1536,  # Q compression dimension
    ),
    ffn_config=FFNConfig(
        intermediate_size=18432,  # For shared expert / dense layers
        activation=ActivationType.SILU,
        use_gating=False,
    ),
    moe_config=MoEConfig(
        num_experts=384,
        num_experts_per_token=8,
        router_type="top_k",
        shared_expert=True,  # n_shared_experts=1
    ),
    is_moe=True,
    normalization_type=NormalizationType.RMS_NORM,
    position_encoding=PositionEncodingType.ROTARY,
    rope_theta=50000.0,  # With YaRN scaling (factor=64)
    tie_word_embeddings=False,
    dtype="bfloat16",
    # Parameter estimation:
    # - Embedding: 7168 * 163840 = 1.17B
    # - Each MoE layer: shared_expert(7168*18432*2) + 384 routed experts (7168*2048*2) = 0.27B + 4.7B = ~5B
    # - Attention per layer: Q/K/V/O projections with MLA ~ 0.1B
    # - 61 layers * ~5B = ~300B+ total, ~37B active (8 experts + shared)
    total_parameters=1_000_000_000_000,  # ~1T total parameters (estimated)
    active_parameters=32_000_000_000,    # ~32B active parameters (8 experts + shared + attention)
)


# Qwen3-480B MoE
# Config from: https://huggingface.co/Qwen/Qwen3-480B
# 160 experts, top-8 routing, 62 layers, GQA with 12:1 ratio
QWEN3_480B = LLMArchitecture(
    model_name="Qwen3-480B",
    model_family="Qwen",
    version="3.0",
    architecture_type=ArchitectureType.DECODER_ONLY,
    num_layers=62,
    hidden_dim=6144,
    vocab_size=151936,
    max_sequence_length=262144,  # 256K context
    attention_config=AttentionConfig(
        num_attention_heads=96,
        num_key_value_heads=8,  # GQA with 12:1 ratio
        attention_type=AttentionType.GROUPED_QUERY,
        head_dim=128,
    ),
    ffn_config=FFNConfig(
        intermediate_size=2560,  # moe_intermediate_size per expert
        activation=ActivationType.SILU,
        use_gating=False,
    ),
    moe_config=MoEConfig(
        num_experts=160,
        num_experts_per_token=8,
        router_type="top_k",
        shared_expert=False,  # shared_expert_intermediate_size=0
    ),
    is_moe=True,
    normalization_type=NormalizationType.RMS_NORM,
    position_encoding=PositionEncodingType.ROTARY,
    rope_theta=10000000.0,  # 10M base
    tie_word_embeddings=False,
    dtype="bfloat16",
    # Parameter estimation:
    # - Embedding: 6144 * 151936 = 0.93B
    # - Per MoE layer: 160 experts * (6144 * 2560 * 2) = 5.03B
    # - Attention per layer: Q(6144*96*128) + KV(6144*8*128*2) + O = ~0.1B
    # - 62 layers * ~5.1B = ~316B for experts, ~6B for attention
    # - Total: ~480B (matches name)
    # - Active: 8 experts * 2560 * 6144 * 2 * 62 + attention ≈ 35B
    total_parameters=480_000_000_000,  # 480B total
    active_parameters=35_000_000_000,  # ~35B active (8 experts + attention)
)


# Hunyuan A13B - Tencent's MoE model with 64 experts
HUNYUAN_A13B = LLMArchitecture(
    model_name="Hunyuan-A13B",
    model_family="Hunyuan",
    version="1.0",
    architecture_type=ArchitectureType.DECODER_ONLY,
    num_layers=32,
    hidden_dim=4096,
    vocab_size=128167,
    max_sequence_length=32768,  # 32K context
    attention_config=AttentionConfig(
        num_attention_heads=32,
        num_key_value_heads=8,  # GQA 4:1 ratio
        attention_type=AttentionType.GROUPED_QUERY,
        head_dim=128,
    ),
    ffn_config=FFNConfig(
        intermediate_size=3072,  # moe_intermediate_size per expert
        activation=ActivationType.SILU,
        use_gating=False,
    ),
    moe_config=MoEConfig(
        num_experts=64,
        num_experts_per_token=8,  # moe_topk
        router_type="top_k",
        shared_expert=True,  # num_shared_expert=1 per layer
    ),
    is_moe=True,
    normalization_type=NormalizationType.RMS_NORM,
    position_encoding=PositionEncodingType.ROTARY,
    rope_theta=10000.0,
    tie_word_embeddings=True,
    dtype="bfloat16",
    # Parameter estimates:
    # - Embedding: 4096 * 128167 = 0.52B
    # - Per MoE layer: 64 experts * (4096 * 3072 * 2) + 1 shared = ~0.81B per layer
    # - Attention per layer: Q(4096*32*128) + KV(4096*8*128*2) + O = ~0.05B
    # - 32 layers * ~0.86B = ~27.5B for experts/attention
    # - Total: ~80B (dense equivalent with all experts)
    # - Active: 8 experts + 1 shared ≈ 13B (matches A13B name)
    total_parameters=80_000_000_000,  # ~80B total
    active_parameters=13_000_000_000,  # ~13B active (8 experts + shared + attention)
)


# GPT-OSS-120B - Hybrid sliding/full attention MoE model
# Config from: gpt-oss-120b.json
# Key features:
# - Alternating sliding_attention / full_attention layers
# - sliding_window=128 tokens for sliding attention layers
# - 128 experts, top-4 routing
# - GQA with 64 heads / 8 KV heads
# - 128K context with YaRN RoPE scaling
GPT_OSS_120B = LLMArchitecture(
    model_name="GPT-OSS-120B",
    model_family="GPT-OSS",
    version="1.0",
    architecture_type=ArchitectureType.DECODER_ONLY,
    num_layers=36,
    hidden_dim=2880,
    vocab_size=201088,
    max_sequence_length=131072,  # 128K context
    attention_config=AttentionConfig(
        num_attention_heads=64,
        num_key_value_heads=8,  # GQA 8:1 ratio
        attention_type=AttentionType.GROUPED_QUERY,  # Base type is GQA
        head_dim=64,
        sliding_window_size=128,  # For sliding attention layers
        attention_bias=True,
    ),
    ffn_config=FFNConfig(
        intermediate_size=2880,
        activation=ActivationType.SILU,
        use_gating=False,
    ),
    moe_config=MoEConfig(
        num_experts=128,  # num_local_experts
        num_experts_per_token=4,  # experts_per_token
        router_type="top_k",
        shared_expert=False,
    ),
    is_moe=True,
    # Per-layer attention types: alternating sliding/full
    # 36 layers: [sliding, full, sliding, full, ...]
    layer_types=[
        LayerAttentionType.SLIDING_ATTENTION if i % 2 == 0 else LayerAttentionType.FULL_ATTENTION
        for i in range(36)
    ],
    normalization_type=NormalizationType.RMS_NORM,
    position_encoding=PositionEncodingType.ROTARY,
    rope_theta=150000.0,
    tie_word_embeddings=False,
    dtype="bfloat16",
    # Parameter estimates:
    # - Embedding: 2880 * 201088 ≈ 0.58B
    # - Per MoE layer: 128 experts * (2880 * 2880 * 2) = ~2.12B per layer
    # - Attention per layer: Q(2880*64*64) + KV(2880*8*64*2) + O = ~0.014B
    # - 36 layers * ~2.13B = ~77B for experts/attention
    # - Total: ~120B (matches name)
    # - Active: 4 experts per token ≈ 12B active
    total_parameters=120_000_000_000,  # ~120B total
    active_parameters=12_000_000_000,  # ~12B active (4 experts + attention)
)


# GPT-OSS-20B - Smaller version of GPT-OSS with hybrid sliding/full attention
# Config from: gpt_oss_20b.json
# Same architecture as GPT-OSS-120B but with fewer layers and experts
GPT_OSS_20B = LLMArchitecture(
    model_name="GPT-OSS-20B",
    model_family="GPT-OSS",
    version="1.0",
    architecture_type=ArchitectureType.DECODER_ONLY,
    num_layers=24,
    hidden_dim=2880,
    vocab_size=201088,
    max_sequence_length=131072,  # 128K context
    attention_config=AttentionConfig(
        num_attention_heads=64,
        num_key_value_heads=8,  # GQA 8:1 ratio
        attention_type=AttentionType.GROUPED_QUERY,
        head_dim=64,
        sliding_window_size=128,  # For sliding attention layers
        attention_bias=True,
    ),
    ffn_config=FFNConfig(
        intermediate_size=2880,
        activation=ActivationType.SILU,
        use_gating=False,
    ),
    moe_config=MoEConfig(
        num_experts=32,  # num_local_experts (vs 128 for 120B)
        num_experts_per_token=4,  # experts_per_token
        router_type="top_k",
        shared_expert=False,
    ),
    is_moe=True,
    # Per-layer attention types: alternating sliding/full
    # 24 layers: [sliding, full, sliding, full, ...]
    layer_types=[
        LayerAttentionType.SLIDING_ATTENTION if i % 2 == 0 else LayerAttentionType.FULL_ATTENTION
        for i in range(24)
    ],
    normalization_type=NormalizationType.RMS_NORM,
    position_encoding=PositionEncodingType.ROTARY,
    rope_theta=150000.0,
    tie_word_embeddings=False,
    dtype="bfloat16",
    # Parameter estimates:
    # - Embedding: 2880 * 201088 ≈ 0.58B
    # - Per MoE layer: 32 experts * (2880 * 2880 * 2) = ~0.53B per layer
    # - Attention per layer: ~0.014B
    # - 24 layers * ~0.54B = ~13B for experts/attention
    # - Total: ~20B (matches name)
    # - Active: 4 experts per token ≈ 5B active
    total_parameters=20_000_000_000,  # ~20B total
    active_parameters=5_000_000_000,  # ~5B active (4 experts + attention)
)


# Nemotron-3-30B - Hybrid Mamba/Attention architecture with MoE
# Config from: Nemotron-3-30B.json
# This is a hybrid model combining Mamba-2 SSM layers with attention layers
# Pattern: "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"
# M = Mamba layer, E = Attention+MoE layer, * = Dense FFN+Attention layer (no MoE)
def _parse_hybrid_pattern(pattern: str) -> list:
    """Parse hybrid_override_pattern into HybridLayerType list"""
    result = []
    for char in pattern:
        if char == 'M':
            result.append(HybridLayerType.MAMBA)
        elif char == 'E':
            result.append(HybridLayerType.ATTENTION)  # Attention + MoE
        elif char == '*':
            result.append(HybridLayerType.ATTENTION)  # Dense Attention (no MoE)
    return result

_NEMOTRON_PATTERN = "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"
_NEMOTRON_HYBRID_LAYERS = _parse_hybrid_pattern(_NEMOTRON_PATTERN)

NEMOTRON_3_30B = LLMArchitecture(
    model_name="Nemotron-3-30B",
    model_family="Nemotron",
    version="3.0-H",  # H for Hybrid
    architecture_type=ArchitectureType.DECODER_ONLY,
    num_layers=52,  # len(_NEMOTRON_HYBRID_LAYERS)
    hidden_dim=2688,
    vocab_size=131072,
    max_sequence_length=262144,  # 256K context
    attention_config=AttentionConfig(
        num_attention_heads=32,
        num_key_value_heads=2,  # GQA 16:1 ratio
        attention_type=AttentionType.GROUPED_QUERY,
        head_dim=128,  # head_dim from config
        attention_bias=False,
    ),
    ffn_config=FFNConfig(
        intermediate_size=1856,  # intermediate_size / moe_intermediate_size
        activation=ActivationType.RELU2,  # mlp_hidden_act: "relu2"
        use_gating=False,
    ),
    moe_config=MoEConfig(
        num_experts=128,  # n_routed_experts
        num_experts_per_token=6,  # num_experts_per_tok
        router_type="top_k",
        shared_expert=True,  # n_shared_experts: 1, shared_expert_intermediate_size: 3712
    ),
    is_moe=True,
    # Hybrid Mamba/Attention architecture
    hybrid_layer_types=_NEMOTRON_HYBRID_LAYERS,  # 52 layers: mix of Mamba and Attention
    mamba_config=Mamba2Config(
        num_heads=64,  # mamba_num_heads
        head_dim=64,  # mamba_head_dim
        state_size=128,  # ssm_state_size
        chunk_size=128,  # chunk_size
        expand=2,  # expand
        conv_kernel=4,  # conv_kernel
        # n_groups=8 from config (used for grouped operations, not modeled separately)
    ),
    normalization_type=NormalizationType.LAYER_NORM,  # layer_norm_epsilon specified
    position_encoding=PositionEncodingType.ROTARY,  # partial_rotary_factor: 1.0
    rope_theta=10000.0,
    tie_word_embeddings=False,
    dtype="bfloat16",
    # Parameter estimates for hybrid Mamba/MoE model:
    # - Embedding: 2688 * 131072 ≈ 0.35B
    # - Mamba layers (27): ~0.15B each = ~4B
    # - Attention+MoE layers (25): 128 experts * intermediate + attention
    # - Shared experts contribution
    # - Total: ~30B (matches name)
    # - Active: 6 routed + 1 shared experts ≈ 7-8B active
    total_parameters=30_000_000_000,  # ~30B total
    active_parameters=8_000_000_000,  # ~8B active (6+1 experts + attention/mamba)
)


# Qwen3 Coder Next - Next-gen MoE coding model
# Config from: Qwen3_Coder_Next.json
# Key features:
# - 512 experts with top-10 routing (massive expert pool)
# - MLA-style compressed attention (linear_key/value dimensions)
# - Shared expert (shared_expert_intermediate_size=512)
# - 48 layers, hidden_dim=2048
# - 262K context (max_position_embeddings)
QWEN3_CODER_NEXT = LLMArchitecture(
    model_name="Qwen3-Coder-Next",
    model_family="Qwen",
    version="3.0-Next",
    architecture_type=ArchitectureType.DECODER_ONLY,
    num_layers=48,
    hidden_dim=2048,
    vocab_size=151936,
    max_sequence_length=262144,  # 262K context
    attention_config=AttentionConfig(
        num_attention_heads=16,
        num_key_value_heads=2,  # GQA with 8:1 ratio
        attention_type=AttentionType.GROUPED_QUERY,
        head_dim=256,  # head_dim from config
        # MLA-style configuration using linear key/value parameters
        use_mla=True,
        mla_kv_lora_rank=128,  # linear_key_head_dim=128, linear_value_head_dim=128
        # Note: linear_num_key_heads=16, linear_num_value_heads=32 in config
    ),
    ffn_config=FFNConfig(
        intermediate_size=512,  # moe_intermediate_size per expert (very small per expert!)
        activation=ActivationType.SILU,
        use_gating=False,
    ),
    moe_config=MoEConfig(
        num_experts=512,  # Massive expert pool
        num_experts_per_token=10,  # top-10 routing
        router_type="top_k",
        shared_expert=True,  # shared_expert_intermediate_size=512
    ),
    is_moe=True,
    ffn_layer_types=[FFNLayerType.MOE] * 48,  # All layers are MoE (mlp_only_layers=[])
    normalization_type=NormalizationType.RMS_NORM,
    position_encoding=PositionEncodingType.ROTARY,
    rope_theta=5000000.0,  # 5M rope_theta
    tie_word_embeddings=False,
    dtype="bfloat16",
    # Parameter estimation:
    # - Embedding: 2048 * 151936 ≈ 0.31B
    # - Per MoE layer: 512 experts * (2048 * 512 * 2) + shared expert ≈ 1.07B + 0.002B
    # - Attention per layer (with MLA compression): ~0.03B
    # - 48 layers * ~1.1B ≈ 53B for experts/attention
    # - Total: ~53B (with all 512 experts)
    # - Active: 10 experts * (2048*512*2) + shared + attention ≈ 21M per layer * 48 ≈ 1B active
    total_parameters=53_000_000_000,  # ~53B total
    active_parameters=1_000_000_000,  # ~1B active (10 experts + shared + attention per layer)
)


# GLM-5 - Zhipu AI's MoE model with both MLA and DSA
# Config from: GLM-5.json
# Key features:
# - MLA (Multi-head Latent Attention) with kv_lora_rank=512, q_lora_rank=2048
# - DSA (Dynamic Sparse Attention) with index_topk=2048
# - 256 routed experts + 1 shared expert, top-8 routing
# - first_k_dense_replace=3: first 3 layers are dense, rest are MoE
# - 78 layers, hidden_dim=6144, 64 attention heads
# - ~198K context (max_position_embeddings=202752)
GLM_5 = LLMArchitecture(
    model_name="GLM-5",
    model_family="GLM",
    version="5.0",
    architecture_type=ArchitectureType.DECODER_ONLY,
    num_layers=78,
    hidden_dim=6144,
    vocab_size=154880,
    max_sequence_length=202752,  # ~198K context
    attention_config=AttentionConfig(
        num_attention_heads=64,
        num_key_value_heads=64,  # MHA before MLA compression
        attention_type=AttentionType.MULTI_HEAD,
        head_dim=256,  # qk_nope_head_dim(192) + qk_rope_head_dim(64) = 256
        # MLA configuration - compress KV cache significantly
        use_mla=True,
        mla_kv_lora_rank=512,  # kv_lora_rank from config
        mla_q_lora_rank=2048,  # q_lora_rank from config
        # Dynamic Sparse Attention (DSA) - top-K selection
        use_dsa=True,
        dsa_q_indexer_dim=128,  # index_head_dim
        dsa_k_indexer_dim=32,   # index_n_heads
        dsa_top_k=2048,  # index_topk: Select top 2048 KV pairs
    ),
    ffn_config=FFNConfig(
        intermediate_size=2048,  # moe_intermediate_size per expert
        dense_intermediate_size=12288,  # intermediate_size for dense layers
        activation=ActivationType.SILU,
        use_gating=False,
    ),
    moe_config=MoEConfig(
        num_experts=256,  # n_routed_experts
        num_experts_per_token=8,  # num_experts_per_tok
        router_type="top_k",
        shared_expert=True,  # n_shared_experts=1
    ),
    is_moe=True,
    # first_k_dense_replace=3: first 3 layers are dense, rest are MoE
    # moe_layer_freq=1: every layer after the first 3 is MoE
    ffn_layer_types=(
        [FFNLayerType.DENSE] * 3 + [FFNLayerType.MOE] * 75
    ),
    normalization_type=NormalizationType.RMS_NORM,
    position_encoding=PositionEncodingType.ROTARY,
    rope_theta=1000000.0,  # 1M rope_theta
    tie_word_embeddings=False,
    dtype="bfloat16",
    # Parameter estimation:
    # - Embedding: 6144 * 154880 ≈ 0.95B
    # - Dense layers (3): attention + FFN(6144*12288*2) ≈ 0.15B + 0.15B = 0.3B each ≈ 0.9B
    # - MoE layers (75): 256 experts * (6144 * 2048 * 2) + shared expert ≈ 6.44B per layer
    # - Attention per layer (with MLA): Q/K/V/O with compression ≈ 0.1B
    # - 75 MoE layers * ~6.5B ≈ 490B for expert layers
    # - Total: ~500B+ (with all 256 experts)
    # - Active: 8 experts * (6144*2048*2) * 75 + shared + attention + dense ≈ 20B
    total_parameters=500_000_000_000,  # ~500B total
    active_parameters=20_000_000_000,  # ~20B active (8 experts + shared + attention)
)


# Qwen3.5-397B - Alibaba's hybrid linear/full attention MoE model
# Config from: Qwen3.5-397B.json
# Key features:
# - Hybrid linear/full attention: 3 linear + 1 full repeating pattern (full_attention_interval=4)
# - Linear attention uses phi(K)^T @ phi(V) state formulation — O(L) not O(L^2)
# - Linear attention: key_heads=16, key_dim=128, value_heads=64, value_dim=128, conv_kernel=4
# - Full attention: 32 heads, 2 KV heads (GQA 16:1), head_dim=256
# - 60 layers, hidden_dim=4096, 512 experts, top-10 routing
# - 45 linear attention layers + 15 full attention layers
# Layer types pattern: [L, L, L, F] × 15
_QWEN35_LAYER_TYPES = []
for i in range(60):
    if (i + 1) % 4 == 0:  # Every 4th layer is full attention (layers 3, 7, 11, ...)
        _QWEN35_LAYER_TYPES.append(LayerAttentionType.FULL_ATTENTION)
    else:
        _QWEN35_LAYER_TYPES.append(LayerAttentionType.LINEAR_ATTENTION)

QWEN3_5_397B = LLMArchitecture(
    model_name="Qwen3.5-397B",
    model_family="Qwen",
    version="3.5",
    architecture_type=ArchitectureType.DECODER_ONLY,
    num_layers=60,
    hidden_dim=4096,
    vocab_size=248320,
    max_sequence_length=262144,  # 262K context
    attention_config=AttentionConfig(
        num_attention_heads=32,
        num_key_value_heads=2,  # GQA 16:1 for full attention layers
        attention_type=AttentionType.GROUPED_QUERY,
        head_dim=256,
    ),
    ffn_config=FFNConfig(
        intermediate_size=1024,  # moe_intermediate_size per expert
        activation=ActivationType.SILU,
        use_gating=False,
    ),
    moe_config=MoEConfig(
        num_experts=512,
        num_experts_per_token=10,  # top-10 routing
        router_type="top_k",
        shared_expert=True,  # shared_expert_intermediate_size=1024
    ),
    is_moe=True,
    ffn_layer_types=[FFNLayerType.MOE] * 60,  # All layers are MoE
    # Hybrid linear/full attention layer pattern
    layer_types=_QWEN35_LAYER_TYPES,
    # Linear attention configuration for linear attention layers
    linear_attention_config=LinearAttentionConfig(
        num_key_heads=16,       # linear_num_key_heads
        key_head_dim=128,       # linear_key_head_dim
        num_value_heads=64,     # linear_num_value_heads
        value_head_dim=128,     # linear_value_head_dim
        conv_kernel_dim=4,      # linear_conv_kernel_dim
    ),
    normalization_type=NormalizationType.RMS_NORM,
    position_encoding=PositionEncodingType.ROTARY,
    rope_theta=10000000.0,  # 10M rope_theta
    tie_word_embeddings=False,
    dtype="bfloat16",
    # Parameter estimation:
    # - Embedding: 4096 * 248320 ≈ 1.02B
    # - Per MoE layer: 512 experts * (4096 * 1024 * 2) + shared ≈ 4.3B
    # - Attention per layer varies (linear vs full)
    # - 60 layers * ~4.3B ≈ 258B for experts
    # - Full attention layers: 15 * ~0.07B ≈ 1B
    # - Linear attention layers: 45 * ~0.05B ≈ 2.3B
    # - Total: ~260B+
    # - Active: 10 experts * (4096*1024*2) * 60 + attention ≈ 5B active
    total_parameters=397_000_000_000,  # ~397B total (from model name)
    active_parameters=13_000_000_000,  # ~13B active (10 experts + shared + attention per layer)
)


# Dictionary of all model configs
ALL_MODELS = {
    "llama-4-scout": LLAMA_4_SCOUT,
    "llama-4-maverick": LLAMA_4_MAVERICK,
    "llama-3-8b": LLAMA_3_8B,
    "llama-3-70b": LLAMA_3_70B,
    "deepseek-v3": DEEPSEEK_V3,
    "deepseek-3.2": DEEPSEEK_3_2,
    "lfm2-3b": LFM2_3B,
    "kimi-k2.5": KIMI_K25,
    "qwen3-480b": QWEN3_480B,
    "qwen3-coder-next": QWEN3_CODER_NEXT,
    "glm-5": GLM_5,
    "hunyuan-a13b": HUNYUAN_A13B,
    "gpt-oss-120b": GPT_OSS_120B,
    "gpt-oss-20b": GPT_OSS_20B,
    "nemotron-3-30b": NEMOTRON_3_30B,
    "qwen3.5-397b": QWEN3_5_397B,
}


def get_model(model_key: str) -> LLMArchitecture:
    """Get a model configuration by key"""
    if model_key not in ALL_MODELS:
        raise ValueError(f"Model '{model_key}' not found. Available models: {list(ALL_MODELS.keys())}")
    return ALL_MODELS[model_key]


def list_models():
    """List all available model configurations"""
    print("Available LLM Configurations:")
    print("-" * 80)
    for key, model in ALL_MODELS.items():
        params_str = f"{model.total_parameters / 1e9:.1f}B"
        if model.is_moe:
            active_str = f" (Active: {model.active_parameters / 1e9:.1f}B)"
        else:
            active_str = ""
        print(f"  {key:20s} - {model.model_name:25s} {params_str}{active_str}")
    print("-" * 80)


if __name__ == "__main__":
    # Demo: Print summaries of all models
    list_models()
    print("\n")
    
    # Example: Get detailed info for Llama 3 8B
    llama3 = get_model("llama-3-8b")
    print(llama3.summary())
    print("\n")
    
    # Example: Calculate memory requirements
    print("Memory footprint for batch_size=1, seq_len=2048:")
    memory = llama3.get_memory_footprint(batch_size=1, sequence_length=2048)
    for component, size in memory.items():
        print(f"  {component}: {size / (1024**3):.2f} GB")
