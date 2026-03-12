# MLA and Sparse Attention Support

## Overview

The framework now supports **Multi-head Latent Attention (MLA)** and **DeepSeek Sparse Attention (DSA)**, advanced architectural features that dramatically reduce memory requirements while maintaining model quality.

## Multi-head Latent Attention (MLA)

### What is MLA?

MLA compresses the Key-Value (KV) cache using low-rank decomposition, storing a compressed latent representation instead of full KV tensors. This provides massive memory savings for long-context inference.

### Key Benefits

- **32x KV cache compression** (configurable via rank)
- **96.9% KV cache memory reduction** compared to standard full attention
- Enables **longer context windows** on the same hardware
- Minimal compute overhead for compression/decompression

### Configuration

```python
from llm_architecture import AttentionConfig

attention_config = AttentionConfig(
    num_attention_heads=128,
    num_key_value_heads=128,
    head_dim=128,
    # Enable MLA
    use_mla=True,
    mla_kv_lora_rank=512,  # Compressed dimension (from 16384 to 512 = 32x)
    mla_q_lora_rank=1536,  # Optional Q compression
)
```

### How It Works

**Standard Attention KV Cache:**
- Stores: `num_kv_heads × head_dim` values per token
- Example: 128 heads × 128 dim = 16,384 dimensions per token

**MLA Compressed KV Cache:**
- Stores: `mla_kv_lora_rank` values per token
- Example: 512 dimensions per token (32x compression)

**Compression Process:**
1. Down-project K and V to latent space: `hidden_dim → kv_lora_rank`
2. Store compressed representation
3. Up-project during attention: `kv_lora_rank → num_kv_heads × head_dim`

### Memory Impact

For DeepSeek 3.2 (batch=1, seq=8192):
- **Standard KV cache**: 30.00 GB
- **MLA KV cache**: 0.94 GB
- **Savings**: 29.06 GB (96.9% reduction)

### Compute Impact

MLA adds compression/decompression operations:
- **Additional FLOPs per layer**:
  - KV compression: `2 × batch × seq × hidden × kv_lora_rank`
  - KV decompression: `2 × batch × seq × kv_lora_rank × (kv_heads × head_dim)`
  - Q compression/decompression (if enabled)

- **Additional kernels**: +2 per layer (compression + decompression)

Trade-off: Slightly more compute for massive memory savings

## DeepSeek Sparse Attention (DSA)

### What is DSA?

Sparse attention patterns that reduce the quadratic attention complexity by only attending to a subset of tokens using local and global blocks.

### Configuration

```python
attention_config = AttentionConfig(
    # ... other params ...
    use_sparse_attention=True,
    sparse_block_size=512,  # Size of each attention block
    sparse_local_blocks=4,  # Number of local blocks (nearby tokens)
    sparse_global_blocks=2,  # Number of global blocks (distant tokens)
)
```

### Pattern

Each query token attends to:
- **Local blocks**: 4 blocks × 512 tokens = 2,048 nearby tokens
- **Global blocks**: 2 blocks × 512 tokens = 1,024 global tokens
- **Total attention span**: 3,072 tokens per query (instead of full sequence)

This reduces attention complexity from O(n²) to O(n × attention_span).

## DeepSeek 3.2 Model

Pre-configured model combining MLA and sparse attention:

```python
from llm_configs import get_model

model = get_model("deepseek-3.2")

# Architecture:
# - 60 layers
# - 5120 hidden dim
# - 128 attention heads
# - MLA with rank 512 (32x compression)
# - Sparse attention (4 local + 2 global blocks)
# - 163K max context length
# - 21B parameters
```

## Example Usage

### Basic Usage

```python
from llm_configs import get_model
from inference_performance import InferencePerformance, ParallelismConfig, ParallelismType

# Load DeepSeek 3.2 with MLA
model = get_model("deepseek-3.2")
perf = InferencePerformance(model)

# Calculate resources for long context
resources = perf.calculate_prefill_resources(
    batch_size=1,
    sequence_length=32768,  # 32K context
    time_to_first_token=2.0,
    num_gpus=1,
    parallelism_config=ParallelismConfig(ParallelismType.NONE, 1, 1, 1)
)

print(f"KV cache: {resources.memory_kv_cache / (1024**3):.2f} GB")
print(f"Total memory: {resources.memory_per_gpu / (1024**3):.2f} GB")
```

### Compare Standard vs MLA

```python
# Standard model
llama = get_model("llama-3-8b")
llama_kv = llama.get_kv_cache_size(batch_size=1, sequence_length=8192)

# MLA model
deepseek = get_model("deepseek-3.2")
deepseek_kv = deepseek.get_kv_cache_size(batch_size=1, sequence_length=8192)

print(f"Standard KV cache: {llama_kv / (1024**3):.2f} GB")
print(f"MLA KV cache: {deepseek_kv / (1024**3):.2f} GB")
print(f"Savings: {(1 - deepseek_kv/llama_kv)*100:.1f}%")
```

## Long Context Performance

DeepSeek 3.2 with MLA enables very long contexts on standard hardware:

| Sequence Length | Total Memory | KV Cache | Fits in A100 80GB? |
|-----------------|--------------|----------|--------------------|
| 8,192           | 41.43 GB     | 0.94 GB  | ✓ Yes              |
| 16,384          | 43.74 GB     | 1.88 GB  | ✓ Yes              |
| 32,768          | 48.37 GB     | 3.75 GB  | ✓ Yes              |
| 65,536          | 57.62 GB     | 7.50 GB  | ✓ Yes              |
| 131,072         | 76.12 GB     | 15.00 GB | ✓ Yes*             |

*Requires careful memory management

## Implementation Details

### KV Cache Calculation

```python
if use_mla:
    # Compressed cache
    kv_cache = 2 * batch * seq * mla_kv_lora_rank * layers * bytes
else:
    # Standard cache
    kv_cache = 2 * batch * seq * (kv_heads * head_dim) * layers * bytes
```

### Compute Calculation

MLA adds:
1. Down-projection: `hidden_dim → kv_lora_rank`
2. Up-projection: `kv_lora_rank → kv_heads × head_dim`
3. Standard attention computation on decompressed KV

### Kernel Launches

- **Dense layer**: 9 kernels
- **Dense layer + MLA**: 11 kernels (+2 for compression/decompression)
- **MoE layer + MLA**: 14 kernels

## Comparison Results

### KV Cache Memory (batch=1, seq=8192)

| Model           | KV Cache | Notes                    |
|-----------------|----------|--------------------------|
| Llama 3 8B      | 1.00 GB  | GQA with 8 KV heads      |
| DeepSeek 3.2    | 0.94 GB  | MLA with rank 512        |
| Savings         | 6.2%     | Despite more heads (128) |

The savings become more dramatic with more KV heads:
- **Standard MHA (128 heads)**: ~30 GB
- **MLA compressed**: 0.94 GB
- **Savings**: 96.9%

### Inference Performance (batch=1, seq=8192, TTFT=1s)

| Model        | Total Memory | KV Cache | Compute    | Kernels |
|--------------|--------------|----------|------------|---------|
| Llama 3 8B   | 17.33 GB     | 1.00 GB  | 158.37 T   | 291     |
| DeepSeek 3.2 | 41.43 GB     | 0.94 GB  | 596.58 T   | 663     |

DeepSeek 3.2 has more total memory due to larger model (21B vs 8B), but similar KV cache thanks to MLA.

## When to Use MLA

**Use MLA when:**
- Long context (>8K tokens) is required
- KV cache memory is a bottleneck
- Multiple sequences in batch with varying lengths
- Memory-constrained deployment

**Standard attention may be better when:**
- Very short contexts (<2K tokens)
- Compute is the bottleneck (MLA adds compute)
- Simplicity is preferred over memory savings

## References

MLA is used in DeepSeek models to achieve efficient long-context inference. The technique provides dramatic memory savings with minimal quality loss through learned low-rank compression.

## Testing

Run the test suite:
```bash
python test_mla.py
```

This demonstrates:
- DeepSeek 3.2 architecture with MLA
- KV cache memory comparison
- Inference performance comparison
- Long context scaling analysis
- Detailed MLA architecture breakdown
