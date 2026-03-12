# Web Application Architecture

Guide to the Flask-based web application for LLM inference performance analysis.

## Overview

The web application provides an interactive interface for analyzing LLM inference performance with:
- **Batch Analysis**: Test multiple configurations and visualize results
- **Interactive Charts**: 9 charts showing performance, utilization, and bandwidth metrics
- **Export Functionality**: Save configurations and results to Excel
- **Real-time Calculation**: Instant feedback on performance metrics

## Architecture Components

### Backend: `web_app.py`

Flask application providing REST API endpoints for calculations.

**Key Routes:**

#### 1. `GET /`
Serves the main HTML interface.

**Returns:** Rendered `index.html` template with list of available models

#### 2. `POST /api/calculate`
Performs single performance calculation.

**Request Body:**
```json
{
    "model": "Llama-3-8B",
    "calculation_type": "TTFT" | "DECODE",
    "batch_size": 16,
    "sequence_length": 2048,      // For TTFT
    "prefill_length": 2048,       // For DECODE
    "decode_steps": 512,          // For DECODE
    "memory": 80,                 // GB
    "memory_bw": 3.35,           // TB/s
    "compute": 1979,             // TFLOPS
    "network_bw": 900,           // GB/s
    "storage_bw": 20,            // GB/s (persistent storage, for MoE offloading)
    "kernel_latency": 5,         // microseconds
    "parallelism": "TENSOR_PARALLEL" | "None",
    "tp_size": 1,
    "pp_size": 1,
    "dtype_override": null | "float16" | "bfloat16" | "int8" | "int4"
}
```

**Response:**
```json
{
    "success": true,
    "calculation_type": "TTFT",
    "metrics": {
        "ttft_ms": 125.5,
        "throughput": 127.4
    },
    "utilization": {
        "compute": 85.2,
        "memory_bw": 92.1,
        "network_bw": 0.0,
        "storage_bw": 0.0,        // Storage bandwidth (MoE only when model doesn't fit)
        "memory": 67.3
    },
    "resources": {
        "memory_per_gpu_gb": 53.8,
        "memory_bw_used_tbps": 3.09,
        "compute_tflops": 1685.7,
        "network_bw_gbps": 0.0,
        "storage_bw_used_gbps": 0.0,  // Storage BW used (MoE only)
        "storage_bw_available_gbps": 20.0
    },
    "bottleneck": "memory_bandwidth"  // Can be "storage_bandwidth" for MoE
}
```

#### 3. `POST /api/batch_analyze`
Runs analysis across multiple batch sizes.

**Request Body:**
```json
{
    "model": "Llama-3-8B",
    "calculation_type": "TTFT" | "DECODE",
    "batch_sizes": [1, 2, 4, 8, 16, 32],
    "sequence_length": 2048,      // For TTFT
    "prefill_length": 2048,       // For DECODE
    "decode_steps": 512,          // For DECODE
    "memory": 80,
    "memory_bw": 3.35,
    "compute": 1979,
    "network_bw": 900,
    "storage_bw": 20,            // GB/s
    "kernel_latency": 5,
    "parallelism": "TENSOR_PARALLEL",
    "tp_size": 1,
    "pp_size": 1,
    "dtype_override": null
}
```

**Response:**
```json
{
    "success": true,
    "calculation_type": "TTFT",
    "batch_sizes": [1, 2, 4, 8, 16, 32],
    "performance": [12.5, 18.3, 28.1, 45.6, 78.2, 125.4],
    "throughput": [80.0, 109.3, 142.3, 175.4, 204.6, 255.1],
    "latency": [12.5, 18.3, 28.1, 45.6, 78.2, 125.4],
    "compute": [850.1, 1200.5, 1654.2, 1850.3, 1950.8, 1979.0],
    "memory_bw": [1.2, 1.8, 2.3, 2.8, 3.1, 3.35],
    "network_bw": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "kernel_overhead": [8.5, 6.2, 4.1, 2.8, 1.9, 1.3],
    "bandwidth": [1.2, 1.8, 2.3, 2.8, 3.1, 3.35],
    "tps": [8.5, 8.7, 8.9, 9.1, 9.2, 9.3],           // DECODE only
    "kv_bandwidth": [0.15, 0.22, 0.35, 0.42, 0.47, 0.50],
    "weights_bandwidth": [1.05, 1.58, 1.95, 2.38, 2.63, 2.85]
}
```

#### 4. `POST /api/export`
Exports configuration and results to Excel file.

**Request Body:**
```json
{
    "config": {
        "model": "Llama-3-8B",
        "calculation_type": "TTFT",
        "batch_sizes": [1, 2, 4, 8, 16, 32],
        "sequence_length": 2048,
        "memory": 80,
        "memory_bw": 3.35,
        "compute": 1979,
        "network_bw": 900,
        "kernel_latency": 5,
        "parallelism": "TENSOR_PARALLEL",
        "tp_size": 2,
        "pp_size": 1,
        "dtype_override": null
    },
    "results": {
        "batch_sizes": [1, 2, 4, 8, 16, 32],
        "performance": [...],
        "throughput": [...],
        // ... all batch analysis results
    }
}
```

**Response:** Excel file download (.xlsx)

**Excel Structure:**
- **Configuration Sheet**: All input parameters
- **Results Sheet**: Complete results table with all metrics per batch size

### Frontend: `templates/index.html`

Single-page HTML interface with Bootstrap styling.

**Key Sections:**

1. **Model Selection** (top left):
   - Dropdown for model selection
   - All available models from `llm_configs.py`

2. **GPU Configuration** (top center):
   - Memory capacity (GB)
   - Memory bandwidth (TB/s)
   - Compute throughput (TFLOPS)
   - Network bandwidth (GB/s)
   - Kernel latency (microseconds)

3. **Parallelism Configuration** (top right):
   - Strategy: None, TP, PP, TP+PP
   - Tensor parallel size
   - Pipeline parallel size

4. **Calculation Inputs** (middle):
   - Calculation type: TTFT / DECODE
   - Single batch size (for single calculation)
   - Batch sizes list (for batch analysis, comma-separated)
   - Sequence length (TTFT) or Prefill length (DECODE)
   - Decode steps (DECODE only)
   - Dtype override: None, float32, float16, bfloat16, int8, int4

5. **Action Buttons**:
   - **Calculate**: Single calculation
   - **Batch Analysis**: Multiple batch sizes
   - **Export Configuration**: Save current settings
   - **Export Results**: Save results to Excel (appears after batch analysis)

6. **Results Display** (bottom):
   - Single calculation: Metrics cards with utilization percentages
   - Batch analysis: 9 interactive Plotly charts in 3×3 grid

### Frontend Logic: `static/app.js`

JavaScript handling UI interactions and chart rendering.

**Key Functions:**

#### `calculate()`
Handles single calculation button click:
1. Gathers form data
2. POSTs to `/api/calculate`
3. Displays results in metrics cards
4. Shows color-coded utilization bars

#### `batchAnalyze()`
Handles batch analysis button click:
1. Gathers form data and parses batch sizes list
2. POSTs to `/api/batch_analyze`
3. Calls `displayBatchChart()` with results

#### `displayBatchChart(result)`
Renders all 9 charts using Plotly.js:

**Chart 1: Performance**
- X-axis: Batch size
- Y-axis: Performance metric (TTFT in ms or TPS)
- Color: Purple gradient

**Chart 2: Throughput**
- X-axis: Batch size
- Y-axis: Total throughput (tokens/sec)
- Color: Purple gradient

**Chart 3: Latency**
- X-axis: Batch size
- Y-axis: Latency (ms)
- Color: Purple gradient

**Chart 4: Compute Utilization**
- X-axis: Batch size
- Y-axis: Compute utilization (%)
- Color: Pink/red gradient

**Chart 5: Memory Bandwidth Utilization**
- X-axis: Batch size
- Y-axis: Memory bandwidth utilization (%)
- Color: Blue gradient

**Chart 6: Kernel Overhead**
- X-axis: Batch size
- Y-axis: Kernel overhead (% of execution time)
- Color: Red gradient

**Chart 7: Network Bandwidth**
- X-axis: Batch size
- Y-axis: Network bandwidth (GB/s)
- Color: Cyan/turquoise gradient

**Chart 8: Throughput vs TPS** (DECODE only)
- X-axis: TPS (tokens/sec per user)
- Y-axis: Total throughput (tokens/sec)
- Color: Dark green
- Shows "Only available for decode calculations" message for TTFT

**Chart 9: Compute/Bandwidth vs Performance**
- X-axis: Compute (TFLOPS) for DECODE or Bandwidth (TB/s) for TTFT
- Y-axis: Performance (TPS for DECODE or Throughput for TTFT)
- Color: Purple

**Chart 10: KV Bandwidth**
- X-axis: Batch size
- Y-axis: KV cache bandwidth (TB/s)
- Color: Red gradient

**Chart 11: Weights Bandwidth**
- X-axis: Batch size
- Y-axis: Model weights bandwidth (TB/s)
- Color: Teal/green gradient

**Chart Behavior:**
- All charts use `autorange: true` for automatic axis scaling
- Hover tooltips show exact values
- Charts are responsive and resize with window
- Uses `Plotly.newPlot()` for rendering
- Uses `Plotly.purge()` before clearing to prevent DOM conflicts

#### `exportConfig()`
Exports current configuration:
1. Gathers all form data
2. Creates JSON blob
3. Downloads as `config.json`

#### `exportResults()`
Exports batch analysis results:
1. Gathers configuration and results
2. POSTs to `/api/export`
3. Triggers Excel file download

## Data Flow

### Single Calculation Flow
```
User clicks "Calculate"
  ↓
calculate() gathers form data
  ↓
POST /api/calculate
  ↓
web_app.py:
  - Parses request
  - Creates model and SystemConstraints
  - Calls InferencePerformance methods
  - Returns JSON response
  ↓
JavaScript displays metrics cards
  ↓
User sees results
```

### Batch Analysis Flow
```
User clicks "Batch Analysis"
  ↓
batchAnalyze() gathers form data
  ↓
POST /api/batch_analyze
  ↓
web_app.py:
  - Loops through batch sizes
  - Calculates each batch size
  - Aggregates results
  - Calculates bandwidth breakdowns (KV, weights)
  - Returns arrays of results
  ↓
displayBatchChart() renders 9 Plotly charts
  ↓
User sees interactive visualizations
```

### Export Flow
```
User clicks "Export Results"
  ↓
exportResults() gathers config + results
  ↓
POST /api/export
  ↓
web_app.py:
  - Creates pandas DataFrames
  - Writes to Excel with 2 sheets
  - Returns file as download
  ↓
Browser downloads results.xlsx
  ↓
User opens in Excel/LibreOffice
```

## Backend Implementation Details

### Bandwidth Calculation

The backend calculates two types of memory bandwidth breakdowns:

**For TTFT (Prefill):**
```python
# Estimation approach (no step details available)
total_memory_bw_used = result.memory_bandwidth_used
weights_bw = total_memory_bw_used * 0.85  # 85% weights
kv_bw = total_memory_bw_used * 0.15       # 15% KV + activations
```

**For DECODE:**
```python
# Precise calculation from step details
for each decode step:
    # Weights traffic
    model_size = model.total_parameters * bytes_per_param
    weights_traffic = model_size / (tp_size * pp_size)
    
    # KV cache traffic
    kv_cache_size = model.get_kv_cache_size(
        batch_size, context_length, bytes_per_element
    )
    # Adjust for DSA if enabled
    if model uses DSA:
        kv_cache_size *= (dsa_top_k / total_kv_pairs)
    
    kv_traffic = kv_cache_size / (tp_size * pp_size)
    
    # Aggregate over all steps
    total_weights_bandwidth += weights_traffic / step_time
    total_kv_bandwidth += kv_traffic / step_time

# Average across steps
avg_weights_bw = total_weights_bandwidth / num_steps
avg_kv_bw = total_kv_bandwidth / num_steps
```

**Key Handling:**
- Handles `parallel_config = None` (defaults to tp_size=1, pp_size=1)
- Accounts for MLA compression in KV cache size
- Accounts for DSA top-K selection
- Converts to TB/s for display

### Debug Output

Set `DEBUG_VERBOSE = True` in `web_app.py` for detailed logging:
- Prints all request parameters
- Shows calculation intermediate results
- Logs bandwidth breakdown calculations
- Useful for troubleshooting

## Frontend Implementation Details

### Chart Persistence Bug Fix

**Problem:** Throughput vs TPS chart sometimes shows "decode only" message even for DECODE calculations.

**Root Cause:** Race condition when `Plotly.newPlot()` is called and then `innerHTML` is set, causing DOM conflicts.

**Solution:** Use `Plotly.purge()` before setting `innerHTML`:
```javascript
if (result.calculation_type === 'DECODE' && has_data) {
    Plotly.newPlot('batch-throughput-chart', ...);
} else {
    Plotly.purge('batch-throughput-chart');  // Clean up Plotly state
    document.getElementById('batch-throughput-chart').innerHTML = '<message>';
}
```

This prevents partial rendering artifacts by ensuring Plotly's internal state is cleared before DOM manipulation.

### Dynamic Form Fields

JavaScript handles showing/hiding fields based on calculation type:
```javascript
// Show sequence_length for TTFT
// Show prefill_length and decode_steps for DECODE
if (calculation_type === 'TTFT') {
    document.getElementById('sequence-length-group').style.display = 'block';
    document.getElementById('prefill-length-group').style.display = 'none';
    document.getElementById('decode-steps-group').style.display = 'none';
} else {
    document.getElementById('sequence-length-group').style.display = 'none';
    document.getElementById('prefill-length-group').style.display = 'block';
    document.getElementById('decode-steps-group').style.display = 'block';
}
```

### Color Schemes

Charts use carefully selected gradients:
- **Purple** (compute/performance): Professional, tech-focused
- **Blue** (memory): Cooling, data-related
- **Red/Pink** (overhead/utilization): Warning, attention
- **Green/Teal** (bandwidth): Growth, throughput
- **Cyan** (network): Communication, connectivity

## Deployment

### Local Development
```bash
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Run Flask app
python web_app.py

# Access at http://localhost:5000
```

**Debug Mode:**
```python
if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

### Production Deployment

**Using Gunicorn (Linux):**
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 web_app:app
```

**Using Waitress (Windows):**
```bash
pip install waitress
waitress-serve --listen=0.0.0.0:5000 web_app:app
```

**Docker:**
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "web_app.py"]
```

### Configuration

**Environment Variables:**
- `FLASK_ENV`: Set to 'production' for deployment
- `FLASK_SECRET_KEY`: Set for session security (if using sessions)
- `PORT`: Override default port 5000

**Static Files:**
- Served from `/static/` directory
- Contains `app.js` (frontend logic)
- CSS in `<style>` tag in `index.html`

**Templates:**
- Served from `/templates/` directory
- Only one template: `index.html`

## Extension Points

### Adding New Charts
1. Add chart container in `index.html`:
```html
<div class="col-md-4">
    <div id="new-chart" class="chart-container"></div>
</div>
```

2. Add rendering logic in `app.js`:
```javascript
const newTrace = {
    x: result.batch_sizes,
    y: result.new_metric,
    mode: 'lines+markers',
    name: 'New Metric',
    line: { color: '#hexcolor', width: 3 },
    marker: { size: 8, color: '#hexcolor' }
};

const newLayout = {
    xaxis: { title: 'Batch Size', autorange: true },
    yaxis: { title: 'New Metric', autorange: true },
    hovermode: 'closest',
    margin: { t: 20 }
};

Plotly.newPlot('new-chart', [newTrace], newLayout, { responsive: true });
```

3. Add data calculation in `web_app.py`:
```python
results_new_metric = []
for batch_size in batch_sizes:
    # Calculate metric
    results_new_metric.append(value)

return jsonify({
    ...,
    'new_metric': results_new_metric
})
```

### Adding New Models
1. Define model in `llm_configs.py`:
```python
NEW_MODEL = LLMArchitecture(
    model_name="NewModel-13B",
    ...
)
```

2. Add to `ALL_MODELS` dict:
```python
ALL_MODELS = {
    ...,
    "NewModel-13B": NEW_MODEL
}
```

Model automatically appears in dropdown on next page load.

### Adding GPU Specs
Add to `SystemConstraints.from_gpu_spec()`:
```python
@classmethod
def from_gpu_spec(cls, gpu_name: str):
    specs = {
        ...,
        "NEW_GPU": {
            "memory": 128e9,        # 128 GB
            "memory_bw": 4.0e12,   # 4 TB/s
            "compute": 2500e12,     # 2500 TFLOPS
            "network": 1200e9       # 1200 GB/s
        }
    }
```

## Testing

### Manual Testing Checklist
- [ ] Single TTFT calculation works
- [ ] Single DECODE calculation works
- [ ] Batch TTFT analysis produces 9 charts
- [ ] Batch DECODE analysis produces all charts including TPS
- [ ] Export config downloads JSON
- [ ] Export results downloads Excel
- [ ] All charts render without errors
- [ ] Throughput vs TPS chart shows message for TTFT
- [ ] KV and weights bandwidth charts show non-zero values
- [ ] Parallelism settings affect results correctly
- [ ] Dtype override changes memory usage

### Automated Testing
Run comprehensive test suite:
```bash
pytest test_comprehensive.py -v
```

81 tests cover all backend calculation logic.

## Troubleshooting

### Charts Not Rendering
- Check browser console for JavaScript errors
- Verify Plotly.js CDN is accessible
- Ensure `result` object has expected fields
- Check that arrays are same length

### Wrong Values in Charts
- Enable `DEBUG_VERBOSE = True` in `web_app.py`
- Check terminal output for calculation details
- Verify input parameters are correct
- Test with known configurations

### Export Not Working
- Check pandas is installed: `pip install pandas`
- Verify openpyxl is installed: `pip install openpyxl`
- Check browser console for errors
- Ensure results object is populated

### Memory Errors
- Model too large for single GPU: use tensor parallelism
- Batch size too large: reduce batch size
- Check `memory_utilization` in results - should be < 1.0

## Best Practices

1. **Always validate inputs** - Check reasonable ranges for batch sizes, sequence lengths
2. **Use DEBUG mode for development** - Set `DEBUG_VERBOSE = True`
3. **Test with known configs** - Verify against paper specifications
4. **Handle edge cases** - Zero batch sizes, None parallelism, empty results
5. **Provide user feedback** - Loading indicators, error messages
6. **Document changes** - Update this guide when adding features
7. **Version control** - Commit working states before major changes
8. **Test cross-browser** - Verify in Chrome, Firefox, Edge

## Future Enhancements

Potential improvements:
- [ ] Real-time updates during batch analysis (WebSocket/SSE)
- [ ] Save/load configurations from database
- [ ] Compare multiple models side-by-side
- [ ] Interactive what-if analysis (sliders)
- [ ] Cost estimation based on cloud pricing
- [ ] Historical analysis (track results over time)
- [ ] Multi-user support with authentication
- [ ] API rate limiting and caching
- [ ] Progressive Web App (PWA) for offline use
- [ ] Mobile-responsive design improvements
