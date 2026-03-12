# LLM Inference Performance Analyzer - GUI Guide

## Overview
A comprehensive GUI application for analyzing LLM inference performance. Calculate achievable Time-To-First-Token (TTFT) or decode phase throughput with visual resource utilization analysis.

## Launch the Application
```bash
.\venv\Scripts\python.exe gui_app.py
```

## GUI Layout

### Left Panel: Configuration Inputs

#### 1. Model Selection
- **Model Dropdown**: Choose from 8 pre-configured models:
  - Llama 3 8B / 70B
  - Llama 2 7B
  - DeepSeek V3 / 3.2 (with MLA)
  - Mistral 7B
  - Mixtral 8x7B (MoE)
  - GPT-3 175B
- **Model Info**: Displays parameter count, layer count, and attention heads

#### 2. System Configuration
- **GPU Type**: Select hardware (A100-40GB, A100-80GB, H100-80GB, MI300X)
- **Parallelism**: Choose parallelism strategy
  - None (single GPU)
  - Data Parallel
  - Tensor Parallel
  - Pipeline Parallel
  - 3D Parallel
- **Parallel Size**: Number of GPUs (appears when parallelism is enabled)

#### 3. Analysis Type
- **Calculate Achievable TTFT**: Backward analysis - given GPU constraints, what TTFT can be achieved?
- **Calculate Decode Performance**: Analyze token generation performance (TPS and throughput)

#### 4. Parameters (Dynamic)
**For TTFT Analysis:**
- Batch Size
- Sequence Length

**For Decode Analysis:**
- Batch Size
- Prefill Length (context already processed)
- Output Length (tokens to generate)

### Right Panel: Results & Visualization

#### Performance Metrics Display
Text-based detailed results including:
- **TTFT Mode**: Achievable TTFT, throughput, resource utilization, bottleneck analysis
- **Decode Mode**: Total time, TPS per user, total throughput, time per token, efficiency metrics

#### Resource Utilization Chart
Bar chart visualization showing:
- **Compute**: GPU compute utilization
- **Memory Bandwidth**: Memory bandwidth utilization
- **Network Bandwidth**: Inter-GPU communication (for parallelism)
- **Memory Capacity**: Memory usage vs available

**Color Coding:**
- 🟢 Green: < 80% utilization (healthy)
- 🟠 Orange: 80-100% utilization (near capacity)
- 🔴 Red: > 100% utilization (bottleneck or OOM)

**Bottleneck Highlighting:**
- Primary bottleneck bar has red border (3px width)
- 100% capacity reference line shown

## Example Use Cases

### Use Case 1: Find Achievable TTFT
**Goal**: Determine what latency you can achieve with Llama 3 8B on A100-80GB

1. Select Model: "Llama 3 8B"
2. GPU Type: "A100-80GB"
3. Parallelism: "None"
4. Analysis Type: "Calculate Achievable TTFT"
5. Batch Size: 1
6. Sequence Length: 2048
7. Click "Calculate Performance"

**Expected Result**: ~107 ms TTFT, compute-bound (100% compute utilization)

### Use Case 2: Analyze Decode Throughput
**Goal**: Evaluate token generation performance at different batch sizes

1. Select Model: "Llama 3 8B"
2. GPU Type: "A100-80GB"
3. Analysis Type: "Calculate Decode Performance"
4. Batch Size: 8
5. Prefill Length: 2048
6. Output Length: 512
7. Click "Calculate Performance"

**Expected Result**: ~100 TPS per user, 795 total throughput, memory bandwidth-bound

### Use Case 3: Large Model with Tensor Parallelism
**Goal**: Check if Llama 3 70B fits on multiple GPUs

1. Select Model: "Llama 3 70B"
2. GPU Type: "A100-80GB"
3. Parallelism: "Tensor Parallel"
4. Parallel Size: 4 (shows when Tensor Parallel selected)
5. Analysis Type: "Calculate Achievable TTFT"
6. Batch Size: 1
7. Sequence Length: 2048
8. Click "Calculate Performance"

**Result**: Check memory utilization - should be < 100% with TP=4

### Use Case 4: MoE Model Analysis
**Goal**: Analyze Mixtral 8x7B performance

1. Select Model: "Mixtral 8x7B"
2. GPU Type: "A100-80GB"
3. Analysis Type: "Calculate Decode Performance"
4. Parameters: Batch=1, Prefill=1024, Output=256
5. Click "Calculate Performance"

**Result**: Lower TPS due to MoE routing overhead, but fewer active parameters

## Understanding the Results

### TTFT Analysis Results

```
Achievable TTFT: 107.23 ms
Throughput: 9.33 requests/sec

Resource Utilization:
  Compute:         100.0%  ← Bottleneck
  Memory BW:       15.2%
  Network BW:      0.0%
  Memory Usage:    34.5%

Bottleneck: Compute  ← System is compute-limited
```

**Interpretation:**
- TTFT is limited by GPU compute power (100% utilized)
- Memory bandwidth is underutilized (15%)
- You have memory headroom (can increase batch size)
- To improve TTFT: Need faster GPU (H100) or reduce workload

### Decode Analysis Results

```
Total Time: 5150.12 ms
TPS (per user): 99.42 tokens/sec
Throughput: 795.32 tokens/sec

Resource Utilization:
  Compute:         0.6%
  Memory BW:       85.5%  ← Bottleneck
  Network BW:      0.0%
  Memory Usage:    38.2%

Primary Bottleneck: Memory Bandwidth  ← Decode is memory-bound
```

**Interpretation:**
- Decode is memory bandwidth-limited (typical for autoregressive generation)
- Compute is barely used (0.6%)
- Can increase batch size to better utilize memory bandwidth
- To improve TPS: Need higher memory BW GPU (MI300X has 5.3 TB/s)

### Common Patterns

**Prefill Phase:**
- Usually compute-bound (processing entire sequence at once)
- High compute utilization (80-100%)
- Lower memory bandwidth utilization (10-30%)
- Good for GPU with high TFLOPS

**Decode Phase:**
- Usually memory bandwidth-bound (fetching KV cache each step)
- Low compute utilization (< 5%)
- High memory bandwidth utilization (70-90%)
- Good for GPU with high memory bandwidth

**Out of Memory:**
- Memory Usage > 100%
- Shows "⚠️ OUT OF MEMORY" warning
- Solutions:
  - Enable tensor parallelism (splits weights across GPUs)
  - Reduce batch size
  - Use smaller model
  - Use model with MLA (KV cache compression)

## Modular Design for Future Extensions

The GUI is designed with extensibility in mind:

### Easy to Add Features

**1. Batch Parameter Sweeps**
Add in `InferenceAnalyzerGUI` class:
```python
def run_batch_sweep(self):
    """Run analysis across range of batch sizes"""
    batch_sizes = [1, 2, 4, 8, 16, 32]
    results = []
    for batch in batch_sizes:
        result = self.calculate_single(batch)
        results.append(result)
    self.display_sweep_results(results)
```

**2. Export to Excel**
Add import: `import pandas as pd`
Add method:
```python
def export_to_excel(self, filename):
    """Export results to Excel"""
    df = pd.DataFrame(self.results_history)
    df.to_excel(filename, index=False)
```

**3. Compare Multiple Configurations**
Track multiple runs:
```python
self.comparison_results = []  # Add to __init__

def add_to_comparison(self):
    """Add current result to comparison set"""
    self.comparison_results.append(self.last_result)
    self.update_comparison_chart()
```

**4. Save/Load Configuration Presets**
```python
def save_preset(self, name):
    """Save current configuration"""
    config = {
        'model': self.model_var.get(),
        'gpu': self.gpu_var.get(),
        'batch_size': self.batch_size_var.get(),
        # ... other params
    }
    with open(f'presets/{name}.json', 'w') as f:
        json.dump(config, f)
```

### Architecture Overview

```
gui_app.py
├── MODEL_REGISTRY         # Easy to add new models
├── GPU_SPECS             # Easy to add new GPUs
├── InferenceAnalyzerGUI
│   ├── Input Sections    # Modular UI components
│   │   ├── Model Selection
│   │   ├── System Config
│   │   ├── Analysis Type
│   │   └── Parameters
│   ├── Calculation Logic # Separated from UI
│   │   ├── calculate_performance()
│   │   ├── get_parallelism_config()
│   │   └── (Future: batch_sweep, comparison)
│   ├── Display Logic    # Separated output
│   │   ├── display_ttft_results()
│   │   ├── display_decode_results()
│   │   └── (Future: display_comparison)
│   └── Visualization    # Chart generation
│       ├── visualize_ttft_utilization()
│       ├── visualize_decode_utilization()
│       └── (Future: multi-config comparison chart)
```

## Tips & Best Practices

### Performance Tuning
1. **Start Simple**: Begin with single GPU, small batch size
2. **Identify Bottleneck**: Check visualization to see limiting factor
3. **Scale Up**: Increase batch size if memory BW-bound
4. **Use Parallelism**: Enable TP for large models that don't fit

### Interpreting Bottlenecks
- **Compute-bound**: Increase batch size or use faster GPU
- **Memory BW-bound**: Normal for decode, increase batch size
- **Memory capacity**: Enable parallelism or reduce batch size
- **Network BW-bound**: Reduce parallelism or use faster interconnect

### Model Selection
- **Small inference (latency-critical)**: Use smaller models (7B-8B)
- **High throughput**: Use larger batch sizes with medium models
- **Memory constrained**: Use MLA models (DeepSeek) for KV cache compression
- **Very large models**: Use tensor parallelism (TP=4 or TP=8)

## Troubleshooting

### GUI Won't Launch
```bash
# Install dependencies
.\venv\Scripts\python.exe -m pip install matplotlib

# Run GUI
.\venv\Scripts\python.exe gui_app.py
```

### Calculation Errors
- Check all parameters are valid integers
- Ensure sequence lengths don't exceed model's max_sequence_length
- For large models, enable tensor parallelism to avoid OOM

### Visualization Not Updating
- Click "Calculate Performance" button again
- Check terminal for error messages

## Next Steps

The modular design allows easy extension with:
- ✅ Parameter sweep analysis (batch size, sequence length ranges)
- ✅ Multi-configuration comparison
- ✅ Excel export for results
- ✅ Configuration presets save/load
- ✅ Historical results tracking
- ✅ Advanced visualizations (line charts for sweeps, comparison bars)

All can be added without restructuring existing code!
