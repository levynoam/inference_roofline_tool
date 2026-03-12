# LLM Architecture Structures Guide

A guide to defining and using LLM architectures with the flexible data structures in `llm_architecture.py`.

## Overview

The architecture system provides Python dataclasses to represent any modern LLM architecture. These structures are designed for:
- **Flexibility**: Support diverse architectures (dense, MoE, MLA, DSA)
- **Completeness**: Capture all relevant architectural details
- **Type Safety**: Use enums and type hints for validation
- **Ease of Use**: Simple, intuitive API with sensible defaults

## Core Structures

### `LLMArchitecture`

The main dataclass representing a complete LLM.

**Required Fields:**
```python
@dataclass
class LLMArchitecture:
    model_name: str              # e.g., "Llama-3-8B"
    model_family: str            # e.g., "Llama"
    version: str                 # e.g., "3.0"
    num_layers: int              # Number of transformer layers
    hidden_dim: int              # Hidden dimension size
    vocab_size: int              # Vocabulary size
    max_sequence_length: int     # Maximum context length
    attention_config: AttentionConfig  # Attention configuration
    ffn_config: FFNConfig        # Feed-forward configuration
```

**Optional Fields:**
- `architecture_type`: DECODER_ONLY (default), ENCODER_DECODER, ENCODER_ONLY
- `moe_config`: MoEConfig for Mixture of Experts models
- `is_moe`: Boolean flag for MoE
- `normalization_type`: LAYER_NORM (default) or RMS_NORM
- `position_encoding`: ROTARY (default), ABSOLUTE, ALIBI, etc.
- `dtype`: "float16" (default), "float32", "bfloat16", "int8", "int4"
- `kernel_launch_latency`: Kernel launch overhead in seconds (default: 5e-6)
- `tie_word_embeddings`: Share input/output embeddings (default: False)

**Example:**
```python
from llm_architecture import LLMArchitecture, AttentionConfig, FFNConfig, ActivationType

my_model = LLMArchitecture(
    model_name="MyModel-7B",
    model_family="Custom",
    version="1.0",
    num_layers=32,
    hidden_dim=4096,
    vocab_size=32000,
    max_sequence_length=4096,
    attention_config=AttentionConfig(num_attention_heads=32),
    ffn_config=FFNConfig(
        intermediate_size=11008,
        activation=ActivationType.SILU
    )
)
```

### `AttentionConfig`

Configures the attention mechanism.

**Basic Configuration:**
```python
@dataclass
class AttentionConfig:
    num_attention_heads: int                    # Number of attention heads
    num_key_value_heads: Optional[int] = None   # For GQA/MQA (None = same as attention heads)
    attention_type: AttentionType = MULTI_HEAD  # MHA, GQA, MQA, SLIDING_WINDOW
    head_dim: Optional[int] = None              # Auto-calculated if None
```

**Attention Types:**
- `AttentionType.MULTI_HEAD`: Standard multi-head attention (MHA)
- `AttentionType.GROUPED_QUERY`: Grouped query attention (GQA)
- `AttentionType.MULTI_QUERY`: Multi-query attention (MQA, num_kv_heads=1)
- `AttentionType.SLIDING_WINDOW`: Local sliding window attention

**Advanced Features:**

1. **Multi-head Latent Attention (MLA)** - Compresses KV cache:
```python
attention_config = AttentionConfig(
    num_attention_heads=32,
    use_mla=True,
    mla_kv_lora_rank=512,  # Compressed KV dimension
    mla_q_lora_rank=1536   # Optional Q compression
)
```

2. **Dynamic Sparse Attention (DSA)** - Top-K KV selection:
```python
attention_config = AttentionConfig(
    num_attention_heads=32,
    use_dsa=True,
    dsa_q_indexer_dim=128,  # Query indexer dimension
    dsa_k_indexer_dim=64,   # Key indexer dimension
    dsa_top_k=2048          # Number of KV pairs to select
)
```

**Examples:**

*Standard MHA:*
```python
mha = AttentionConfig(num_attention_heads=32)
```

*Grouped Query Attention (8 KV heads):*
```python
gqa = AttentionConfig(
    num_attention_heads=32,
    num_key_value_heads=8,
    attention_type=AttentionType.GROUPED_QUERY
)
```

*Multi-Query Attention:*
```python
mqa = AttentionConfig(
    num_attention_heads=32,
    num_key_value_heads=1,
    attention_type=AttentionType.MULTI_QUERY
)
```

### `FFNConfig`

Configures the feed-forward network.

**Configuration:**
```python
@dataclass
class FFNConfig:
    intermediate_size: int                      # FFN hidden dimension
    activation: ActivationType = GELU           # Activation function
    use_gating: bool = False                    # True for SwiGLU, GeGLU
    ffn_dropout: float = 0.0
    ffn_bias: bool = False
```

**Activation Types:**
- `ActivationType.GELU`: Gaussian Error Linear Unit
- `ActivationType.SILU`: Sigmoid Linear Unit (used in SwiGLU)
- `ActivationType.RELU`: Rectified Linear Unit
- `ActivationType.SWIGLU`: SwiGLU (auto-sets use_gating=True)
- `ActivationType.GEGLU`: GeGLU (auto-sets use_gating=True)

**Examples:**

*Standard FFN with GELU:*
```python
ffn = FFNConfig(
    intermediate_size=11008,
    activation=ActivationType.GELU
)
```

*SwiGLU (Llama-style):*
```python
swiglu_ffn = FFNConfig(
    intermediate_size=11008,
    activation=ActivationType.SWIGLU,
    use_gating=True  # SwiGLU uses gating
)
```

### `MoEConfig`

Configures Mixture of Experts architecture.

**Configuration:**
```python
@dataclass
class MoEConfig:
    num_experts: int                            # Total number of experts
    num_experts_per_token: int                  # Top-K experts activated
    expert_capacity_factor: float = 1.0
    router_type: Literal["top_k", "switch", "expert_choice"] = "top_k"
    shared_expert: bool = False                 # Shared expert (e.g., DeepSeek)
```

**Example:**
```python
moe = MoEConfig(
    num_experts=8,
    num_experts_per_token=2,  # Activate top-2 experts
    router_type="top_k",
    shared_expert=True        # DeepSeek-style shared expert
)

moe_model = LLMArchitecture(
    model_name="MyMoE-8x7B",
    model_family="Custom",
    version="1.0",
    num_layers=32,
    hidden_dim=4096,
    vocab_size=32000,
    max_sequence_length=4096,
    attention_config=AttentionConfig(num_attention_heads=32),
    ffn_config=FFNConfig(intermediate_size=14336),
    moe_config=moe,
    is_moe=True
)
```

## Enumerations

### `ArchitectureType`
- `DECODER_ONLY`: GPT-style (most common)
- `ENCODER_DECODER`: T5-style
- `ENCODER_ONLY`: BERT-style

### `NormalizationType`
- `LAYER_NORM`: Standard layer normalization
- `RMS_NORM`: Root Mean Square normalization (Llama, modern models)
- `GROUP_NORM`: Group normalization

### `PositionEncodingType`
- `ROTARY`: Rotary Position Embedding (RoPE, most modern models)
- `ABSOLUTE`: Learned absolute positions
- `ALIBI`: Attention with Linear Biases
- `RELATIVE`: Relative position encoding
- `NONE`: No position encoding

## Methods

### `estimate_parameters()`
Calculates total parameter count.

```python
model = LLMArchitecture(...)
param_count = model.estimate_parameters()
print(f"Model has {param_count / 1e9:.2f}B parameters")
```

Called automatically in `__post_init__()` if `total_parameters` not provided.

### `get_kv_cache_size()`
Calculates KV cache size for given batch and sequence length.

```python
batch_size = 16
seq_length = 2048
dtype_bytes = 2  # float16

kv_cache_bytes = model.get_kv_cache_size(
    batch_size=batch_size,
    sequence_length=seq_length,
    bytes_per_element=dtype_bytes
)

print(f"KV cache: {kv_cache_bytes / 1e9:.2f} GB")
```

Automatically handles:
- GQA/MQA (reduced KV heads)
- MLA compression (reduced dimensions)
- DSA (only top-K stored)

### `get_activation_memory()`
Estimates activation memory for a single sequence.

```python
seq_length = 2048
dtype_bytes = 2

activation_bytes = model.get_activation_memory(
    sequence_length=seq_length,
    bytes_per_element=dtype_bytes
)

print(f"Activations per sequence: {activation_bytes / 1e9:.2f} GB")
```

## Using Pre-configured Models

The `llm_configs.py` file provides ready-to-use configurations:

```python
from llm_configs import (
    LLAMA_2_7B,
    LLAMA_3_8B,
    LLAMA_3_70B,
    DEEPSEEK_V3,
    MISTRAL_7B,
    MIXTRAL_8X7B,
    GPT3_175B
)

# Use directly
from inference_performance import InferencePerformance
perf = InferencePerformance(LLAMA_3_8B)

# Inspect configuration
print(f"Model: {LLAMA_3_8B.model_name}")
print(f"Layers: {LLAMA_3_8B.num_layers}")
print(f"Hidden dim: {LLAMA_3_8B.hidden_dim}")
print(f"Parameters: {LLAMA_3_8B.total_parameters / 1e9:.2f}B")

# Access all available models
from llm_configs import ALL_MODELS
for name, model in ALL_MODELS.items():
    print(f"{name}: {model.total_parameters / 1e9:.1f}B params")
```

## Creating Custom Architectures

### Example 1: Dense Transformer (GPT-style)
```python
from llm_architecture import (
    LLMArchitecture, AttentionConfig, FFNConfig,
    AttentionType, ActivationType, NormalizationType, PositionEncodingType
)

custom_model = LLMArchitecture(
    model_name="CustomGPT-13B",
    model_family="GPT",
    version="1.0",
    
    # Core dimensions
    num_layers=40,
    hidden_dim=5120,
    vocab_size=50257,
    max_sequence_length=8192,
    
    # Standard MHA
    attention_config=AttentionConfig(
        num_attention_heads=40,
        attention_type=AttentionType.MULTI_HEAD
    ),
    
    # FFN with GELU
    ffn_config=FFNConfig(
        intermediate_size=20480,
        activation=ActivationType.GELU
    ),
    
    # Modern settings
    normalization_type=NormalizationType.LAYER_NORM,
    position_encoding=PositionEncodingType.ROTARY,
    dtype="bfloat16"
)
```

### Example 2: Llama-style Model with GQA
```python
llama_style = LLMArchitecture(
    model_name="LlamaStyle-7B",
    model_family="Llama",
    version="1.0",
    
    num_layers=32,
    hidden_dim=4096,
    vocab_size=32000,
    max_sequence_length=4096,
    
    # Grouped Query Attention (8 KV heads)
    attention_config=AttentionConfig(
        num_attention_heads=32,
        num_key_value_heads=8,
        attention_type=AttentionType.GROUPED_QUERY
    ),
    
    # SwiGLU FFN
    ffn_config=FFNConfig(
        intermediate_size=11008,
        activation=ActivationType.SWIGLU,
        use_gating=True
    ),
    
    # Llama-style settings
    normalization_type=NormalizationType.RMS_NORM,
    position_encoding=PositionEncodingType.ROTARY,
    rope_theta=10000.0,
    dtype="bfloat16"
)
```

### Example 3: MoE Model (Mixtral-style)
```python
moe_model = LLMArchitecture(
    model_name="CustomMoE-8x7B",
    model_family="Mixtral",
    version="1.0",
    
    num_layers=32,
    hidden_dim=4096,
    vocab_size=32000,
    max_sequence_length=32768,
    
    # GQA
    attention_config=AttentionConfig(
        num_attention_heads=32,
        num_key_value_heads=8,
        attention_type=AttentionType.GROUPED_QUERY,
        sliding_window_size=4096  # Optional sliding window
    ),
    
    # FFN (will be MoE)
    ffn_config=FFNConfig(
        intermediate_size=14336,
        activation=ActivationType.SWIGLU,
        use_gating=True
    ),
    
    # MoE configuration
    moe_config=MoEConfig(
        num_experts=8,
        num_experts_per_token=2
    ),
    is_moe=True,
    
    normalization_type=NormalizationType.RMS_NORM,
    position_encoding=PositionEncodingType.ROTARY,
    dtype="bfloat16"
)
```

### Example 4: Advanced Model with MLA and DSA
```python
advanced_model = LLMArchitecture(
    model_name="Advanced-671B",
    model_family="DeepSeek",
    version="3.0",
    
    num_layers=61,
    hidden_dim=7168,
    vocab_size=129280,
    max_sequence_length=65536,
    
    # MLA + DSA for efficient long context
    attention_config=AttentionConfig(
        num_attention_heads=128,
        num_key_value_heads=128,
        attention_type=AttentionType.MULTI_HEAD,
        
        # Multi-head Latent Attention
        use_mla=True,
        mla_kv_lora_rank=512,
        mla_q_lora_rank=1536,
        
        # Dynamic Sparse Attention
        use_dsa=True,
        dsa_q_indexer_dim=128,
        dsa_k_indexer_dim=64,
        dsa_top_k=2048
    ),
    
    # MoE FFN
    ffn_config=FFNConfig(
        intermediate_size=18432,
        activation=ActivationType.SWIGLU,
        use_gating=True
    ),
    
    moe_config=MoEConfig(
        num_experts=256,
        num_experts_per_token=8,
        shared_expert=True  # DeepSeek-style
    ),
    is_moe=True,
    
    normalization_type=NormalizationType.RMS_NORM,
    position_encoding=PositionEncodingType.ROTARY,
    rope_theta=10000.0,
    dtype="bfloat16"
)
```

## Best Practices

1. **Start from Examples**: Use pre-configured models as templates
2. **Set Sensible Defaults**: The library provides reasonable defaults for most fields
3. **Validate Dimensions**: Ensure `hidden_dim` is divisible by `num_attention_heads`
4. **Match Real Models**: When modeling existing models, verify against published specs
5. **Use Enums**: Prefer enums over strings for type safety
6. **Test**: Use `estimate_parameters()` to validate your configuration matches expectations

## Common Patterns

### Standard Dense Transformer
```python
LLMArchitecture(
    ...,
    attention_config=AttentionConfig(num_attention_heads=N),
    ffn_config=FFNConfig(intermediate_size=4*hidden_dim),
    normalization_type=NormalizationType.LAYER_NORM,
    position_encoding=PositionEncodingType.ABSOLUTE
)
```

### Modern Llama-style
```python
LLMArchitecture(
    ...,
    attention_config=AttentionConfig(
        num_attention_heads=N,
        num_key_value_heads=N//4,  # GQA
        attention_type=AttentionType.GROUPED_QUERY
    ),
    ffn_config=FFNConfig(
        intermediate_size=int(hidden_dim * 2.67),
        activation=ActivationType.SWIGLU,
        use_gating=True
    ),
    normalization_type=NormalizationType.RMS_NORM,
    position_encoding=PositionEncodingType.ROTARY
)
```

### MoE with Shared Expert
```python
LLMArchitecture(
    ...,
    ffn_config=FFNConfig(...),
    moe_config=MoEConfig(
        num_experts=N,
        num_experts_per_token=K,
        shared_expert=True
    ),
    is_moe=True
)
```

## Integration with InferencePerformance

Once you have an `LLMArchitecture`, use it with `InferencePerformance`:

```python
from inference_performance import InferencePerformance, SystemConstraints

# Create your model
my_model = LLMArchitecture(...)

# Create performance calculator
perf = InferencePerformance(my_model)

# Run analyses
gpu = SystemConstraints.from_gpu_spec("H100")

ttft_result = perf.calculate_achievable_ttft(
    system_constraints=gpu,
    batch_size=16,
    sequence_length=2048
)

decode_result = perf.calculate_decode_performance(
    system_constraints=gpu,
    batch_size=32,
    prefill_length=2048,
    decode_steps=512
)
```

The architecture structures seamlessly integrate with all performance analysis tools.
