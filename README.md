# LLM Inference Performance Analysis System

A comprehensive Python framework for modeling and analyzing Large Language Model inference performance, including resource requirements, bottleneck analysis, and interactive visualization.

## Overview

This system provides:
- **Performance Calculation** - TTFT (Time-To-First-Token) and decode phase analysis
- **Resource Modeling** - Compute, memory, bandwidth, network, and persistent storage requirements
- **Bottleneck Analysis** - Identify limiting resources (compute, memory bandwidth, network, storage)
- **Architecture Support** - Dense models, MoE (with storage offloading), MLA, DSA, Mamba-2 hybrid, Latent MoE, sublayer-style hybrid architectures, and various attention mechanisms
- **Interactive Web UI** - Batch analysis with 9 interactive charts
- **Parallelism Strategies** - Single GPU, TP, PP, and hybrid configurations

## Quick Start

### Web Application
```bash
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Run web app
python web_app.py

# Open browser to http://localhost:5000
```

The web interface provides:
- Model selection from pre-configured options
- GPU configuration (memory, bandwidth, compute)
- Batch analysis with interactive charts
- Export results to Excel

### Command Line Usage

```python
from llm_configs import LLAMA_3_8B
from inference_performance import InferencePerformance, SystemConstraints

# Create performance analyzer
perf = InferencePerformance(LLAMA_3_8B)
gpu = SystemConstraints.from_gpu_spec("H100")

# Calculate TTFT
result = perf.calculate_achievable_ttft(
    system_constraints=gpu,
    batch_size=16,
    sequence_length=2048
)

print(f"TTFT: {result.achievable_ttft * 1000:.2f} ms")
print(f"Bottleneck: {result.bottleneck}")
print(f"Compute Utilization: {result.compute_utilization * 100:.1f}%")

# Calculate decode performance
decode = perf.calculate_decode_performance(
    system_constraints=gpu,
    batch_size=32,
    prefill_length=2048,
    decode_steps=512
)

print(f"TPS: {decode.tps:.2f} tokens/sec per user")
print(f"Total Throughput: {decode.total_throughput:.2f} tokens/sec")
```

## Documentation

**Core Guides:**
- **[INFERENCE_PERFORMANCE_GUIDE.md](INFERENCE_PERFORMANCE_GUIDE.md)** - Complete guide to using the inference performance calculator
- **[LLM_STRUCTS.md](LLM_STRUCTS.md)** - How to define and use LLM architecture structures
- **[WEB_APP.md](WEB_APP.md)** - Web application architecture and API documentation

**Additional Documentation:**
- Legacy documentation is available in the `deprecated/` folder

## Key Components

### 1. `inference_performance.py` - Performance Calculator

**Main Classes:**
- `InferencePerformance` - Calculate TTFT and decode performance
- `SystemConstraints` - GPU specifications (memory, bandwidth, compute)
- `ParallelismConfig` - Distributed execution strategies
- `ResourceUtilization` - TTFT results with utilization metrics
- `DecodePerformance` - Decode results with TPS and throughput

**Key Methods:**
- `calculate_achievable_ttft()` - Time to first token analysis
- `calculate_decode_performance()` - Decode phase performance
- `SystemConstraints.from_gpu_spec()` - Pre-configured GPU specs (H100, A100, etc.)

### 2. `llm_architecture.py` - Architecture Structures

**Main Classes:**
- `LLMArchitecture` - Complete model specification
- `AttentionConfig` - Attention mechanism (MHA, GQA, MQA, MLA, DSA)
- `FFNConfig` - Feed-forward network configuration
- `MoEConfig` - Mixture of Experts configuration
- `LatentMoEConfig` - Latent-space MoE (Nemotron-Super style)
- `Mamba2Config` - Mamba-2 SSM configuration
- `HybridLayerType` - Sublayer types for hybrid architectures

**Supported Features:**
- ✅ Multiple attention types (Multi-Head, Grouped Query, Multi-Query)
- ✅ MLA (Multi-head Latent Attention) for compressed KV cache
- ✅ DSA (Dynamic Sparse Attention) for long context efficiency
- ✅ MoE architectures with configurable experts
- ✅ Latent MoE (experts operate in projected latent space)
- ✅ Mamba-2 SSM layers and Mamba-2/Attention hybrid models
- ✅ Sublayer-style hybrid architectures (e.g., Nemotron-3-Super)
- ✅ Various activations (GELU, SiLU, SwiGLU, GeGLU, ReLU²)
- ✅ Different normalizations (LayerNorm, RMSNorm)
- ✅ Position encodings (RoPE, ALiBi, Absolute)

### 3. `llm_configs.py` - Pre-configured Models

Ready-to-use configurations:
- **Llama 4 Scout / Maverick**
- **Llama 3** (8B, 70B)
- **DeepSeek V3 / 3.2** (671B MoE)
- **Nemotron-3-30B** (hybrid Mamba-2/Attention MoE)
- **Nemotron-3-Super-120B** (sublayer-style hybrid: Mamba-2 + Latent MoE + Attention)
- **Qwen3** (480B MoE, Coder Next)
- **Qwen3.5-397B** (interleaved linear/full attention)
- **GLM-5**, **Hunyuan-A13B**, **GPT-OSS 20B / 120B**
- **Kimi-K2.5**, **LFM2-3B**

### 4. `web_app.py` - Interactive Web Application

Flask-based web UI with:
- Model and GPU configuration
- Single calculation and batch analysis
- 9 interactive Plotly charts
- Excel export functionality
- Real-time performance feedback

## Features

### Performance Analysis
- **TTFT Calculation**: Achievable time-to-first-token with bottleneck identification
- **Decode Performance**: TPS, throughput, and step-by-step resource analysis
- **Utilization Metrics**: Compute, memory bandwidth, network bandwidth, storage bandwidth, memory capacity
- **Kernel Overhead**: Models GPU kernel launch latency impact
- **Bandwidth Breakdown**: Separates KV cache, weights, and activation traffic
- **Persistent Storage Offloading**: Models MoE expert loading from NVMe when model doesn't fit in DRAM

### Parallelism Support
- **None**: Single GPU execution
- **Tensor Parallel (TP)**: Split model across GPUs
- **Pipeline Parallel (PP)**: Layer-wise distribution
- **Hybrid**: Combine TP and PP strategies

### Model Features
- **Dense Transformers**: Standard attention and FFN
- **MoE (Mixture of Experts)**: Sparse expert activation
- **MLA (Multi-head Latent Attention)**: Compressed KV cache
- **DSA (Dynamic Sparse Attention)**: Top-K KV selection for long context
- **GQA/MQA**: Grouped and multi-query attention
- **Mamba-2 Hybrid**: SSM layers interspersed with attention/FFN
- **Latent MoE**: Experts that operate in a projected latent space (Nemotron-Super)
- **Sublayer-style Hybrid**: Each sublayer is a single operation (Mamba-only, LatentMoE-only, Attention-only)

### Web Interface
- **Batch Analysis**: Test multiple batch sizes at once
- **9 Interactive Charts**:
  1. Performance (TTFT or TPS)
  2. Total Throughput
  3. Latency
  4. Compute Utilization
  5. Memory Bandwidth Utilization
  6. Kernel Overhead
  7. Network Bandwidth
  8. Throughput vs TPS (decode only)
  9. Compute/Bandwidth vs Performance
  10. KV Cache Bandwidth
  11. Weights Bandwidth
- **Export**: Save configurations and results to Excel

## Testing

Comprehensive test suite with 500+ tests across multiple files:
```bash
pytest tests/ -v
```

Test coverage includes:
- Prefill resource calculations
- TTFT with various configurations
- Decode performance analysis
- Edge cases (OOM, zero batch, extreme values)
- MLA and DSA support
- Parallelism strategies
- MoE models
- Bandwidth breakdowns
- Mamba-2 hybrid architectures
- Latent MoE (LatentMoEConfig FLOPs, weight params)
- Sublayer-style hybrid (Nemotron-3-Super-120B pattern)

## Requirements

- Python 3.10+
- Flask (for web app)
- NumPy
- Pandas (for Excel export)
- Plotly.js (CDN, for charts)

## Project Structure

```
Sim/
├── inference_performance.py    # Core performance calculator
├── llm_architecture.py        # Model architecture definitions
├── llm_configs.py            # Pre-configured models
├── web_app.py                # Flask web application
├── test_comprehensive.py     # Test suite (81 tests)
├── static/
│   └── app.js               # Frontend JavaScript
├── templates/
│   └── index.html          # Web UI template
├── deprecated/             # Legacy documentation
├── INFERENCE_PERFORMANCE_GUIDE.md  # Performance calculator guide
├── LLM_STRUCTS.md                 # Architecture structures guide
├── WEB_APP.md                     # Web app architecture guide
└── README.md                      # This file
```

## Examples

### Custom Model Definition
```python
from llm_architecture import LLMArchitecture, AttentionConfig, FFNConfig

custom_model = LLMArchitecture(
    model_name="CustomModel-7B",
    model_family="Custom",
    version="1.0",
    num_layers=32,
    hidden_dim=4096,
    vocab_size=32000,
    max_sequence_length=4096,
    attention_config=AttentionConfig(
        num_attention_heads=32,
        num_key_value_heads=8  # GQA
    ),
    ffn_config=FFNConfig(
        intermediate_size=11008,
        activation=ActivationType.SWIGLU
    )
)
```

### Parallelism Configuration
```python
from inference_performance import ParallelismConfig, ParallelismType

# 4-way tensor parallelism
tp_config = ParallelismConfig(
    parallelism_type=ParallelismType.TENSOR_PARALLEL,
    tensor_parallel_size=4
)

result = perf.calculate_achievable_ttft(
    system_constraints=gpu,
    batch_size=16,
    sequence_length=2048,
    parallelism_config=tp_config
)
```

### Bandwidth Analysis
```python
result = perf.calculate_decode_performance(
    system_constraints=gpu,
    batch_size=32,
    prefill_length=2048,
    decode_steps=100,
    return_step_details=True
)

# Examine per-step bandwidth breakdown
for step in result.step_details:
    print(f"Context {step.context_length}: "
          f"Weights={step.weights_memory_traffic/1e12:.2f}TB, "
          f"KV={step.kv_cache_memory_traffic/1e12:.2f}TB")
```

## Use Cases

1. **Hardware Selection**: Determine which GPU meets performance requirements
2. **Batch Size Optimization**: Find optimal batch size for throughput/latency tradeoff
3. **Parallelism Strategy**: Compare TP, PP, and hybrid configurations
4. **Architecture Exploration**: Evaluate impact of MLA, DSA, GQA on performance
5. **Cost Analysis**: Estimate serving costs based on throughput requirements
6. **Capacity Planning**: Plan deployment for target workload

## Best Practices

1. **Start with Pre-configured GPUs**: Use `SystemConstraints.from_gpu_spec()`
2. **Validate with Known Models**: Test against published specifications
3. **Check Memory First**: Ensure `memory_utilization < 1.0` before analyzing performance
4. **Profile Incrementally**: Test single GPU before multi-GPU configurations
5. **Use Batch Analysis**: Visualize trends across batch sizes in web UI
6. **Enable Debug Mode**: Set `DEBUG_VERBOSE = True` for troubleshooting
7. **Leverage Tests**: Run test suite to verify calculations

## Contributing

When extending the system:
1. Add tests for new features in `tests/`
2. Update relevant documentation guide
3. Follow existing code patterns
4. Verify all tests still pass (`pytest tests/ -q`)

## License

This project is licensed under the [MIT License](LICENSE).

## Support

For questions or issues:
1. Check the relevant documentation guide
2. Review test suite for usage examples
3. Enable DEBUG_VERBOSE for detailed logging
4. Examine web app implementation for integration patterns


