// LLM Inference Performance Analyzer - Web UI JavaScript

// Batch parameter defaults
const BATCH_DEFAULTS = {
    'memory': { min: 64, max: 2048 },
    'compute': { min: 1024, max: 16384 },
    'memory_bw': { min: 1000, max: 50000 },
    'network_bw': { min: 400, max: 1600 },
    'storage_bw': { min: 5, max: 1000 },
    'kernel_latency': { min: 1, max: 50 },
    'batch_size': { min: 1, max: 128 },
    'sequence_length': { min: 128, max: 8192 },
    'prefill_length': { min: 128, max: 8192 },
    'output_length': { min: 1024, max: 30000 }
};

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    // Set up event listeners
    document.getElementById('calculate-btn').addEventListener('click', calculatePerformance);
    document.getElementById('batch-btn').addEventListener('click', runBatchAnalysis);
    document.getElementById('export-btn').addEventListener('click', exportConfiguration);
    document.getElementById('export-results-btn').addEventListener('click', exportResults);
    
    // Calculation type change
    document.querySelectorAll('input[name="calc_type"]').forEach(radio => {
        radio.addEventListener('change', updateCalculationType);
    });
    
    // Parallelism change
    document.getElementById('parallelism').addEventListener('change', updateParallelism);
    
    // Batch parameter change
    document.getElementById('batch_param').addEventListener('change', updateBatchDefaults);
    
    // Initialize UI state
    updateCalculationType();
    updateParallelism();
    updateBatchDefaults();
});

function updateCalculationType() {
    const calcType = document.querySelector('input[name="calc_type"]:checked').value;
    const ttftParams = document.getElementById('ttft-params');
    const decodeParams = document.getElementById('decode-params');
    
    if (calcType === 'TTFT') {
        ttftParams.style.display = 'block';
        decodeParams.style.display = 'none';
    } else {
        ttftParams.style.display = 'none';
        decodeParams.style.display = 'block';
    }
}

function updateParallelism() {
    const parallelism = document.getElementById('parallelism').value;
    const parallelParams = document.getElementById('parallel-params');
    
    if (parallelism === 'None') {
        parallelParams.style.display = 'none';
    } else {
        parallelParams.style.display = 'block';
    }
}

function updateBatchDefaults() {
    const param = document.getElementById('batch_param').value;
    const defaults = BATCH_DEFAULTS[param];
    
    if (defaults) {
        document.getElementById('batch_min').value = defaults.min;
        document.getElementById('batch_max').value = defaults.max;
    }
}

function getFormData() {
    const calcType = document.querySelector('input[name="calc_type"]:checked').value;
    const parallelism = document.getElementById('parallelism').value;
    const numGpus = parseInt(document.getElementById('num_gpus').value) || 1;
    const dtype = document.getElementById('dtype').value;
    
    const data = {
        model: document.getElementById('model').value,
        memory: parseFloat(document.getElementById('memory').value),
        compute: parseFloat(document.getElementById('compute').value),
        memory_bw: parseFloat(document.getElementById('memory_bw').value),
        network_bw: parseFloat(document.getElementById('network_bw').value),
        storage_bw: parseFloat(document.getElementById('storage_bw').value),
        kernel_latency: parseFloat(document.getElementById('kernel_latency').value),
        batch_size: parseInt(document.getElementById('batch_size').value),
        calculation_type: calcType,
        parallelism: parallelism,
        tp_size: parallelism === 'TENSOR_PARALLEL' ? numGpus : 1,
        pp_size: parallelism === 'PIPELINE_PARALLEL' ? numGpus : 1
    };
    
    // Add dtype_override only if not using model default
    if (dtype) {
        data.dtype_override = dtype;
    }
    
    if (calcType === 'TTFT') {
        data.sequence_length = parseInt(document.getElementById('sequence_length').value);
    } else {
        data.prefill_length = parseInt(document.getElementById('prefill_length').value);
        data.output_length = parseInt(document.getElementById('output_length').value);
    }
    
    return data;
}

async function calculatePerformance() {
    const btn = document.getElementById('calculate-btn');
    btn.disabled = true;
    btn.textContent = 'Calculating...';
    
    try {
        const data = getFormData();
        
        const response = await fetch('/api/calculate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        
        const result = await response.json();
        
        if (result.success) {
            displayResults(result);
        } else {
            showError(result.error);
        }
    } catch (error) {
        showError(`Request failed: ${error.message}`);
    } finally {
        btn.disabled = false;
        btn.textContent = 'Calculate Performance';
    }
}

function displayResults(result) {
    const resultsDiv = document.getElementById('results-text');
    
    let html = '<h2>Performance Results</h2>';
    
    // Metrics
    html += '<div class="metrics-grid">';
    
    if (result.calculation_type === 'TTFT') {
        html += `
            <div class="metric-card">
                <div class="metric-label">TTFT</div>
                <div class="metric-value">${result.metrics.ttft_ms.toFixed(2)}</div>
                <div class="metric-unit">ms</div>
            </div>
        `;
    } else {
        html += `
            <div class="metric-card">
                <div class="metric-label">TPS</div>
                <div class="metric-value">${result.metrics.tps.toFixed(2)}</div>
                <div class="metric-unit">tok/s</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">System Throughput</div>
                <div class="metric-value">${result.metrics.system_throughput.toFixed(2)}</div>
                <div class="metric-unit">tok/s</div>
            </div>
        `;
    }
    
    html += '</div>';
    
    // Detailed information
    html += '<div class="results-details">';
    html += '<h3>Resource Utilization (per GPU)</h3>';
    html += `
        <div class="detail-row">
            <span class="detail-label">Compute:</span>
            <span class="detail-value">${result.resources.compute_used.toFixed(2)} PFLOPs/s (${result.utilization.compute.toFixed(1)}%)</span>
        </div>
        <div class="detail-row">
            <span class="detail-label">Memory Bandwidth:</span>
            <span class="detail-value">${result.resources.memory_bw_used.toFixed(2)} TB/s (${result.utilization.memory_bw.toFixed(1)}%)</span>
        </div>
        <div class="detail-row">
            <span class="detail-label">Network Bandwidth:</span>
            <span class="detail-value">${result.resources.network_bw_used.toFixed(2)} TB/s (${result.utilization.network_bw.toFixed(1)}%)</span>
        </div>
        <div class="detail-row">
            <span class="detail-label">Storage Bandwidth:</span>
            <span class="detail-value">${result.resources.storage_bw_used ? result.resources.storage_bw_used.toFixed(2) : '0.00'} GB/s (${result.utilization.storage_bw.toFixed(1)}%)</span>
        </div>
        <div class="detail-row">
            <span class="detail-label">Memory Capacity:</span>
            <span class="detail-value">${result.resources.memory_used.toFixed(2)} GB (${result.utilization.memory.toFixed(1)}%)</span>
        </div>
    `;
    
    if (result.bottleneck) {
        html += `<div class="bottleneck">⚠️ Bottleneck: ${result.bottleneck}</div>`;
    }
    
    html += '</div>';
    
    resultsDiv.innerHTML = html;
    
    // Display model architecture
    displayModelArchitecture(result.model_architecture);
    
    // Display per-layer breakdown graphs
    if (result.per_layer_breakdown) {
        displayPerLayerBreakdown(result.per_layer_breakdown);
    }
    
    // Display charts
    displayResourceChart(result);
    displayMemoryChart(result.memory_breakdown);
    displayComputeChart(result.compute_breakdown);
    displayTimeChart(result.time_breakdown);
}

function displayModelArchitecture(arch) {
    const archSection = document.getElementById('model-architecture');
    const archDetails = document.getElementById('architecture-details');
    
    // Show the section
    archSection.style.display = 'block';
    
    // Format parameter counts
    const formatParams = (params) => {
        if (params >= 1e9) return `${(params / 1e9).toFixed(1)}B`;
        if (params >= 1e6) return `${(params / 1e6).toFixed(1)}M`;
        return `${params}`;
    };
    
    // Format attention type for display
    const formatAttentionType = (type) => {
        const typeMap = {
            'grouped_query_attention': 'GQA',
            'multi_head_attention': 'MHA',
            'multi_query_attention': 'MQA',
            'sliding_window': 'Sliding Window'
        };
        return typeMap[type] || type;
    };
    
    let html = '';
    
    // === Basic Architecture ===
    html += '<div class="arch-section-title">Basic</div>';
    
    // Total parameters
    html += `
        <div class="architecture-item">
            <div class="arch-label">Parameters</div>
            <div class="arch-value">${formatParams(arch.parameters)}</div>
        </div>
    `;
    
    // Active parameters (for MoE)
    if (arch.is_moe) {
        html += `
            <div class="architecture-item">
                <div class="arch-label">Active Parameters</div>
                <div class="arch-value">${formatParams(arch.active_parameters)}</div>
            </div>
        `;
    }
    
    // Layers
    html += `
        <div class="architecture-item">
            <div class="arch-label">Layers</div>
            <div class="arch-value">${arch.layers}</div>
        </div>
    `;
    
    // Hidden dimension
    html += `
        <div class="architecture-item">
            <div class="arch-label">Hidden Dimension</div>
            <div class="arch-value">${arch.hidden_dim}</div>
        </div>
    `;
    
    // === Attention Configuration ===
    html += '<div class="arch-section-title">Attention</div>';
    
    // Attention type
    html += `
        <div class="architecture-item">
            <div class="arch-label">Type</div>
            <div class="arch-value">${formatAttentionType(arch.attention_type)}</div>
        </div>
    `;
    
    // Heads
    html += `
        <div class="architecture-item">
            <div class="arch-label">Q Heads / KV Heads</div>
            <div class="arch-value">${arch.num_attention_heads} / ${arch.num_kv_heads}</div>
        </div>
    `;
    
    // Head dimension
    html += `
        <div class="architecture-item">
            <div class="arch-label">Head Dim</div>
            <div class="arch-value">${arch.head_dim}</div>
        </div>
    `;
    
    // MLA (Multi-head Latent Attention)
    if (arch.use_mla) {
        html += `
            <div class="architecture-item highlight">
                <div class="arch-label">MLA KV Rank</div>
                <div class="arch-value">${arch.mla_kv_lora_rank}</div>
            </div>
        `;
    }
    
    // Sliding window attention
    if (arch.sliding_window_size) {
        html += `
            <div class="architecture-item highlight">
                <div class="arch-label">Sliding Window</div>
                <div class="arch-value">${arch.sliding_window_size.toLocaleString()} tokens</div>
            </div>
        `;
    }
    
    // Per-layer attention types (hybrid sliding/full)
    if (arch.has_layer_types) {
        html += `
            <div class="architecture-item highlight">
                <div class="arch-label">Full Attn Layers</div>
                <div class="arch-value">${arch.num_full_attention_layers}</div>
            </div>
            <div class="architecture-item highlight">
                <div class="arch-label">Sliding Attn Layers</div>
                <div class="arch-value">${arch.num_sliding_attention_layers}</div>
            </div>
        `;
    }
    
    // === MoE Configuration ===
    if (arch.is_moe) {
        html += '<div class="arch-section-title">Mixture of Experts</div>';
        html += `
            <div class="architecture-item">
                <div class="arch-label">Total Experts</div>
                <div class="arch-value">${arch.moe_experts}</div>
            </div>
            <div class="architecture-item">
                <div class="arch-label">Active Experts</div>
                <div class="arch-value">${arch.moe_active_experts}</div>
            </div>
        `;
    }
    
    // === Hybrid Mamba/Attention ===
    if (arch.is_hybrid) {
        html += '<div class="arch-section-title">Hybrid Architecture</div>';
        html += `
            <div class="architecture-item highlight">
                <div class="arch-label">Attention Sublayers</div>
                <div class="arch-value">${arch.num_attention_layers}</div>
            </div>
            <div class="architecture-item highlight">
                <div class="arch-label">Mamba Sublayers</div>
                <div class="arch-value">${arch.num_mamba_layers}</div>
            </div>
        `;

        if (arch.num_latent_moe_layers != null && arch.num_latent_moe_layers > 0) {
            html += `
                <div class="architecture-item highlight">
                    <div class="arch-label">LatentMoE Sublayers</div>
                    <div class="arch-value">${arch.num_latent_moe_layers}</div>
                </div>
            `;
        }

        if (arch.has_mamba_config) {
            html += `
                <div class="architecture-item">
                    <div class="arch-label">Mamba State Size</div>
                    <div class="arch-value">${arch.mamba_state_size}</div>
                </div>
                <div class="architecture-item">
                    <div class="arch-label">Mamba Heads</div>
                    <div class="arch-value">${arch.mamba_num_heads} × ${arch.mamba_head_dim}</div>
                </div>
            `;
        }
    }

    // === Latent MoE ===
    if (arch.has_latent_moe) {
        html += '<div class="arch-section-title">Latent MoE</div>';
        html += `
            <div class="architecture-item highlight">
                <div class="arch-label">Experts (total / active)</div>
                <div class="arch-value">${arch.latent_moe_num_experts} / ${arch.latent_moe_num_active}</div>
            </div>
            <div class="architecture-item">
                <div class="arch-label">Latent Size</div>
                <div class="arch-value">${arch.latent_moe_latent_size}</div>
            </div>
            <div class="architecture-item">
                <div class="arch-label">Expert Intermediate Size</div>
                <div class="arch-value">${arch.latent_moe_expert_size}</div>
            </div>
            <div class="architecture-item">
                <div class="arch-label">Shared Expert Size</div>
                <div class="arch-value">${arch.latent_moe_shared_size} (×${arch.latent_moe_n_shared})</div>
            </div>
        `;
    }

    // === Interleaved Dense/MoE FFN ===
    if (arch.has_ffn_layer_types) {
        html += '<div class="arch-section-title">Interleaved FFN</div>';
        html += `
            <div class="architecture-item highlight">
                <div class="arch-label">Dense Layers</div>
                <div class="arch-value">${arch.num_dense_ffn_layers} (size=${arch.dense_intermediate_size})</div>
            </div>
            <div class="architecture-item highlight">
                <div class="arch-label">MoE Layers</div>
                <div class="arch-value">${arch.num_moe_ffn_layers} (size=${arch.moe_intermediate_size})</div>
            </div>
        `;
    }
    
    // === Linear Attention ===
    if (arch.has_linear_attention) {
        html += '<div class="arch-section-title">Linear Attention</div>';
        html += `
            <div class="architecture-item highlight">
                <div class="arch-label">Linear Attn Layers</div>
                <div class="arch-value">${arch.num_linear_attention_layers}</div>
            </div>
            <div class="architecture-item highlight">
                <div class="arch-label">Full Attn Layers</div>
                <div class="arch-value">${arch.num_full_attn_layers_linear}</div>
            </div>
            <div class="architecture-item">
                <div class="arch-label">Key Heads × Dim</div>
                <div class="arch-value">${arch.linear_key_heads} × ${arch.linear_key_dim}</div>
            </div>
            <div class="architecture-item">
                <div class="arch-label">Value Heads × Dim</div>
                <div class="arch-value">${arch.linear_value_heads} × ${arch.linear_value_dim}</div>
            </div>
            <div class="architecture-item">
                <div class="arch-label">Conv Kernel</div>
                <div class="arch-value">${arch.linear_conv_kernel}</div>
            </div>
        `;
    }
    
    archDetails.innerHTML = html;
}

function displayPerLayerBreakdown(plb) {
    const container = document.getElementById('per-layer-graphs');
    container.style.display = 'block';
    
    const n = plb.num_layers;
    const layerIndices = Array.from({length: n}, (_, i) => i);
    
    // Build hover text with layer type info
    const hoverLabels = layerIndices.map(i => `Layer ${i} (${plb.layer_types[i]})`);
    
    // Format helpers
    const formatFlops = (v) => {
        if (v >= 1e15) return (v / 1e15).toFixed(2) + ' PF';
        if (v >= 1e12) return (v / 1e12).toFixed(2) + ' TF';
        if (v >= 1e9) return (v / 1e9).toFixed(2) + ' GF';
        if (v >= 1e6) return (v / 1e6).toFixed(2) + ' MF';
        return v.toFixed(0) + ' F';
    };
    const formatBytes = (v) => {
        if (v >= 1e12) return (v / 1e12).toFixed(2) + ' TB';
        if (v >= 1e9) return (v / 1e9).toFixed(2) + ' GB';
        if (v >= 1e6) return (v / 1e6).toFixed(2) + ' MB';
        if (v >= 1e3) return (v / 1e3).toFixed(2) + ' KB';
        return v.toFixed(0) + ' B';
    };
    
    // Common layout settings
    const modeLabel = plb.mode === 'prefill' ? 'Prefill' : 'Decode';
    const seqLabel = plb.mode === 'prefill' ? 'seq_len' : 'ctx_len';
    const baseLayout = {
        margin: {l: 65, r: 20, t: 35, b: 40},
        legend: {orientation: 'h', y: 1.02, x: 1, xanchor: 'right', yanchor: 'bottom', font: {size: 11}},
        xaxis: {title: 'Layer Index', dtick: n <= 20 ? 1 : (n <= 60 ? 5 : 10)},
    };
    const plotConfig = {responsive: true, displayModeBar: false};
    
    // Compute sum arrays
    const totalCompute = layerIndices.map(i => plb.attention_compute[i] + plb.non_attention_compute[i]);
    const totalBW = layerIndices.map(i => plb.attention_bandwidth[i] + plb.non_attention_bandwidth[i]);
    const totalKernels = layerIndices.map(i => plb.attention_kernels[i] + plb.non_attention_kernels[i]);
    
    const lineAttn = {color: '#667eea', width: 2};
    const lineFFN = {color: '#48bb78', width: 2};
    const lineSum = {color: '#8B4513', width: 2};
    
    const markerAttn = {color: '#4a5bc0', symbol: 'square', size: 6};
    const markerFFN = {color: '#2d8a56', symbol: 'square', size: 6};
    const markerSum = {color: '#5C2E0A', symbol: 'square', size: 6};
    
    // ======= 1. Compute per layer =======
    Plotly.newPlot('per-layer-compute-chart', [
        {
            x: layerIndices, y: plb.attention_compute, name: 'Attention',
            type: 'scatter', mode: 'lines+markers', line: lineAttn, marker: markerAttn,
            text: hoverLabels,
            hovertemplate: '%{text}<br>Attention: %{y:.3s} FLOPs<extra></extra>'
        },
        {
            x: layerIndices, y: plb.non_attention_compute, name: 'FFN',
            type: 'scatter', mode: 'lines+markers', line: lineFFN, marker: markerFFN,
            text: hoverLabels,
            hovertemplate: '%{text}<br>FFN: %{y:.3s} FLOPs<extra></extra>'
        },
        {
            x: layerIndices, y: totalCompute, name: 'Total',
            type: 'scatter', mode: 'lines+markers', line: lineSum, marker: markerSum,
            text: hoverLabels,
            hovertemplate: '%{text}<br>Total: %{y:.3s} FLOPs<extra></extra>'
        }
    ], {
        ...baseLayout,
        title: `Compute per Layer (${modeLabel}, ${seqLabel}=${plb.sequence_length})`,
        yaxis: {title: 'FLOPs per GPU'},
    }, plotConfig);
    
    // ======= 2. Bandwidth per layer (GB/s) =======
    Plotly.newPlot('per-layer-bandwidth-chart', [
        {
            x: layerIndices, y: plb.attention_bandwidth, name: 'Attention',
            type: 'scatter', mode: 'lines+markers', line: lineAttn, marker: markerAttn,
            text: hoverLabels,
            hovertemplate: '%{text}<br>Attention: %{y:,.0f} GB/s<extra></extra>'
        },
        {
            x: layerIndices, y: plb.non_attention_bandwidth, name: 'FFN',
            type: 'scatter', mode: 'lines+markers', line: lineFFN, marker: markerFFN,
            text: hoverLabels,
            hovertemplate: '%{text}<br>FFN: %{y:,.0f} GB/s<extra></extra>'
        },
        {
            x: layerIndices, y: totalBW, name: 'Total',
            type: 'scatter', mode: 'lines+markers', line: lineSum, marker: markerSum,
            text: hoverLabels,
            hovertemplate: '%{text}<br>Total: %{y:,.0f} GB/s<extra></extra>'
        }
    ], {
        ...baseLayout,
        title: `Bandwidth per Layer (${modeLabel}, ${seqLabel}=${plb.sequence_length})`,
        yaxis: {title: 'GB/s per GPU'},
    }, plotConfig);
    
    // ======= 3. Kernel launches per layer =======
    Plotly.newPlot('per-layer-kernels-chart', [
        {
            x: layerIndices, y: plb.attention_kernels, name: 'Attention',
            type: 'scatter', mode: 'lines+markers', line: lineAttn, marker: markerAttn,
            text: hoverLabels,
            hovertemplate: '%{text}<br>Attention: %{y} kernels<extra></extra>'
        },
        {
            x: layerIndices, y: plb.non_attention_kernels, name: 'FFN',
            type: 'scatter', mode: 'lines+markers', line: lineFFN, marker: markerFFN,
            text: hoverLabels,
            hovertemplate: '%{text}<br>FFN: %{y} kernels<extra></extra>'
        },
        {
            x: layerIndices, y: totalKernels, name: 'Total',
            type: 'scatter', mode: 'lines+markers', line: lineSum, marker: markerSum,
            text: hoverLabels,
            hovertemplate: '%{text}<br>Total: %{y} kernels<extra></extra>'
        }
    ], {
        ...baseLayout,
        title: `Kernel Launches per Layer (${modeLabel})`,
        yaxis: {title: 'Kernel Launches'},
    }, plotConfig);
}

function displayResourceChart(result) {
    const utilization = result.utilization;
    
    // Determine colors based on utilization
    const colors = ['compute', 'memory_bw', 'network_bw', 'storage_bw', 'memory'].map(key => {
        const val = utilization[key];
        if (val > 100) return '#e74c3c';  // Red
        if (val > 80) return '#f39c12';   // Orange
        return '#27ae60';                  // Green
    });
    
    const data = [{
        x: ['Compute', 'Memory BW', 'Network BW', 'Storage BW', 'Memory'],
        y: [utilization.compute, utilization.memory_bw, utilization.network_bw, utilization.storage_bw, utilization.memory],
        type: 'bar',
        marker: {
            color: colors,
            line: { width: 2, color: 'black' }
        },
        text: [
            `${utilization.compute.toFixed(1)}%`,
            `${utilization.memory_bw.toFixed(1)}%`,
            `${utilization.network_bw.toFixed(1)}%`,
            `${utilization.storage_bw.toFixed(1)}%`,
            `${utilization.memory.toFixed(1)}%`
        ],
        textposition: 'outside'
    }];
    
    const layout = {
        title: 'Resource Utilization',
        yaxis: {
            title: 'Utilization (%)',
            range: [0, Math.max(...Object.values(utilization)) * 1.2]
        },
        shapes: [{
            type: 'line',
            x0: -0.5,
            x1: 4.5,
            y0: 100,
            y1: 100,
            line: { color: 'red', width: 2, dash: 'dash' }
        }],
        annotations: [{
            x: 3.5,
            y: 102,
            text: '100% Capacity',
            showarrow: false,
            font: { color: 'red', size: 10 }
        }]
    };
    
    Plotly.newPlot('resource-chart', data, layout, { responsive: true });
}

function displayMemoryChart(memory_breakdown) {
    const data = [{
        x: ['Weights', 'KV Cache', 'Activations'],
        y: [memory_breakdown.weights, memory_breakdown.kv_cache, memory_breakdown.activations],
        type: 'bar',
        marker: {
            color: ['#3498db', '#9b59b6', '#1abc9c'],
            line: { width: 2, color: 'black' }
        },
        text: [
            `${memory_breakdown.weights.toFixed(2)} GB`,
            `${memory_breakdown.kv_cache.toFixed(2)} GB`,
            `${memory_breakdown.activations.toFixed(2)} GB`
        ],
        textposition: 'outside'
    }];
    
    const total = memory_breakdown.weights + memory_breakdown.kv_cache + memory_breakdown.activations;
    const layout = {
        title: 'Memory Breakdown (per GPU)',
        yaxis: {
            title: 'Memory (GB)',
            range: [0, Math.max(...[memory_breakdown.weights, memory_breakdown.kv_cache, memory_breakdown.activations]) * 1.2]
        },
        annotations: [{
            x: 1,
            y: Math.max(...[memory_breakdown.weights, memory_breakdown.kv_cache, memory_breakdown.activations]) * 1.15,
            text: `Total: ${total.toFixed(2)} GB`,
            showarrow: false,
            font: { size: 12, color: '#2c3e50' }
        }]
    };
    
    Plotly.newPlot('memory-chart', data, layout, { responsive: true });
}

function displayComputeChart(compute_breakdown) {
    const data = [{
        x: ['Attention', 'FFN', 'Other'],
        y: [compute_breakdown.attention, compute_breakdown.ffn, compute_breakdown.other],
        type: 'bar',
        marker: {
            color: ['#e74c3c', '#f39c12', '#95a5a6'],
            line: { width: 2, color: 'black' }
        },
        text: [
            `${compute_breakdown.attention.toFixed(2)} TFLOPs`,
            `${compute_breakdown.ffn.toFixed(2)} TFLOPs`,
            `${compute_breakdown.other.toFixed(2)} TFLOPs`
        ],
        textposition: 'outside'
    }];
    
    const total = compute_breakdown.attention + compute_breakdown.ffn + compute_breakdown.other;
    const layout = {
        title: 'Compute Breakdown',
        yaxis: {
            title: 'Compute (TFLOPs)',
            range: [0, Math.max(...[compute_breakdown.attention, compute_breakdown.ffn, compute_breakdown.other]) * 1.2]
        },
        annotations: [{
            x: 1,
            y: Math.max(...[compute_breakdown.attention, compute_breakdown.ffn, compute_breakdown.other]) * 1.15,
            text: `Total: ${total.toFixed(2)} TFLOPs`,
            showarrow: false,
            font: { size: 12, color: '#2c3e50' }
        }]
    };
    
    Plotly.newPlot('compute-chart', data, layout, { responsive: true });
}

function displayTimeChart(time_breakdown) {
    const data = [{
        x: ['Compute Busy', 'Kernel Launch', 'Idle'],
        y: [time_breakdown.compute_busy, time_breakdown.kernel_launch, time_breakdown.idle],
        type: 'bar',
        marker: {
            color: ['#3498db', '#e67e22', '#95a5a6'],
            line: { width: 2, color: 'black' }
        },
        text: [
            `${time_breakdown.compute_busy.toFixed(2)} ms`,
            `${time_breakdown.kernel_launch.toFixed(2)} ms`,
            `${time_breakdown.idle.toFixed(2)} ms`
        ],
        textposition: 'outside'
    }];
    
    const total = time_breakdown.compute_busy + time_breakdown.kernel_launch + time_breakdown.idle;
    const maxVal = Math.max(time_breakdown.compute_busy, time_breakdown.kernel_launch, time_breakdown.idle);
    
    const layout = {
        title: 'Time Breakdown',
        yaxis: {
            title: 'Time (ms)',
            range: [0, maxVal * 1.2]
        },
        annotations: [{
            x: 1,
            y: maxVal * 1.15,
            text: `Total: ${total.toFixed(2)} ms`,
            showarrow: false,
            font: { size: 12, color: '#2c3e50' }
        }]
    };
    
    Plotly.newPlot('time-chart', data, layout, { responsive: true });
}

async function runBatchAnalysis() {
    const btn = document.getElementById('batch-btn');
    btn.disabled = true;
    btn.textContent = 'Running...';
    
    try {
        const data = getFormData();
        data.param_name = document.getElementById('batch_param').value;
        data.min_val = parseFloat(document.getElementById('batch_min').value);
        data.max_val = parseFloat(document.getElementById('batch_max').value);
        data.num_points = parseInt(document.getElementById('batch_points').value);
        
        const response = await fetch('/api/batch', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        
        const result = await response.json();
        
        if (result.success) {
            // Store batch results for export
            window.lastBatchResults = result;
            // Show export button
            document.getElementById('export-results-btn').style.display = 'block';
            displayBatchChart(result);
        } else {
            showError(result.error);
        }
    } catch (error) {
        showError(`Batch analysis failed: ${error.message}`);
    } finally {
        btn.disabled = false;
        btn.textContent = 'Run Batch Analysis';
    }
}

function displayBatchChart(result) {
    console.log('Display batch chart called with:', result);
    console.log('Calculation type:', result.calculation_type);
    console.log('TPS array:', result.tps);
    console.log('Throughput array:', result.throughput);
    
    const yLabel = result.calculation_type === 'TTFT' ? 'TTFT (ms)' : 'TPS (tokens/sec)';
    
    // Main performance chart
    const trace = {
        x: result.x_values,
        y: result.y_values,
        mode: 'lines+markers',
        name: yLabel,
        line: { color: '#667eea', width: 3 },
        marker: { size: 8, color: '#09040e' },
        hovertemplate: '%{y:.1f}<extra></extra>'
    };
    
    const layout = {
        xaxis: {
            title: result.param_name,
            type: 'log'
        },
        yaxis: {
            title: yLabel,
            rangemode: 'tozero'
        },
        hovermode: 'closest',
        margin: { t: 20 }
    };
    
    Plotly.newPlot('batch-chart', [trace], layout, { responsive: true });
    
    // Compute chart
    const computeTrace = {
        x: result.x_values,
        y: result.compute,
        mode: 'lines+markers',
        name: 'Compute',
        line: { color: '#667eea', width: 3 },
        marker: { size: 8, color: '#0f0716' },
        hovertemplate: '%{y:.1f}<extra></extra>'
    };
    
    const computeLayout = {
        xaxis: {
            title: result.param_name,
            type: 'log'
        },
        yaxis: {
            title: 'Compute (PF/sec)',
            rangemode: 'tozero'
        },
        hovermode: 'closest',
        margin: { t: 20 }
    };
    
    Plotly.newPlot('batch-compute-chart', [computeTrace], computeLayout, { responsive: true });
    
    // Bandwidth chart
    const bandwidthTrace = {
        x: result.x_values,
        y: result.bandwidth,
        mode: 'lines+markers',
        name: 'Bandwidth',
        line: { color: '#580461', width: 3 },
        marker: { size: 8, color: '#120513' },
        hovertemplate: '%{y:.1f}<extra></extra>'
    };
    
    const bandwidthLayout = {
        xaxis: {
            title: result.param_name,
            type: 'log'
        },
        yaxis: {
            title: 'Memory Bandwidth (TB/sec)',
            rangemode: 'tozero'
        },
        hovermode: 'closest',
        margin: { t: 20 }
    };
    
    Plotly.newPlot('batch-bandwidth-chart', [bandwidthTrace], bandwidthLayout, { responsive: true });
    
    // Network bandwidth chart
    const networkTrace = {
        x: result.x_values,
        y: result.network,
        mode: 'lines+markers',
        name: 'Network Bandwidth',
        line: { color: '#1b1bb1', width: 3 },
        marker: { size: 8, color: '#03050a' },
        hovertemplate: '%{y:.1f}<extra></extra>'
    };
    
    const networkLayout = {
        xaxis: {
            title: result.param_name,
            type: 'log'
        },
        yaxis: {
            title: 'Network Bandwidth (GB/sec)',
            rangemode: 'tozero'
        },
        hovermode: 'closest',
        margin: { t: 20 }
    };
    
    Plotly.newPlot('batch-network-chart', [networkTrace], networkLayout, { responsive: true });
    
    // Kernel launch overhead chart
    const kernelOverheadTrace = {
        x: result.x_values,
        y: result.kernel_overhead,
        mode: 'lines+markers',
        name: 'Kernel Overhead',
        line: { color: '#ff6b6b', width: 3 },
        marker: { size: 8, color: '#180d0e' },
        hovertemplate: '%{y:.1f}<extra></extra>'
    };
    
    const kernelOverheadLayout = {
        xaxis: {
            title: result.param_name,
            type: 'log'
        },
        yaxis: {
            title: 'Kernel Launch Overhead (%)',
            range: [0, 100]
        },
        hovermode: 'closest',
        margin: { t: 20 }
    };
    
    Plotly.newPlot('batch-kernel-overhead-chart', [kernelOverheadTrace], kernelOverheadLayout, { responsive: true });
    
    // Throughput vs TPS chart (only for decode)
    console.log('Checking throughput chart conditions:', {
        calc_type: result.calculation_type,
        has_tps: !!result.tps,
        tps_length: result.tps ? result.tps.length : 0,
        first_tps: result.tps ? result.tps[0] : null
    });
    
    if (result.calculation_type === 'DECODE' && result.tps && result.tps.length > 0) {
        // Filter out null values
        const validIndices = [];
        for (let i = 0; i < result.tps.length; i++) {
            if (result.tps[i] !== null && result.throughput[i] !== null) {
                validIndices.push(i);
            }
        }
        
        if (validIndices.length > 0) {
            const tpsValues = validIndices.map(i => result.tps[i]);
            const throughputValues = validIndices.map(i => result.throughput[i]);
            
            const throughputTrace = {
                x: tpsValues,
                y: throughputValues,
                mode: 'lines+markers',
                name: 'Throughput vs TPS',
                line: { color: '#0b471b', width: 3 },
                marker: { size: 8, color: '#000000' },
                hovertemplate: 'TPS: %{x:.1f}<br>Throughput: %{y:.1f}<extra></extra>'
            };
            
            const throughputLayout = {
                xaxis: {
                    title: 'TPS (tokens/sec per user)',
                    autorange: true
                },
                yaxis: {
                    title: 'Total Throughput (tokens/sec)',
                    autorange: true
                },
                hovermode: 'closest',
                margin: { t: 20 }
            };
            
            Plotly.newPlot('batch-throughput-chart', [throughputTrace], throughputLayout, { responsive: true });
        } else {
            Plotly.purge('batch-throughput-chart');
            document.getElementById('batch-throughput-chart').innerHTML = '<div style="text-align: center; padding: 40px; color: #666;">No valid data points</div>';
        }
    } else {
        // Clear the chart if not decode or no data
        Plotly.purge('batch-throughput-chart');
        document.getElementById('batch-throughput-chart').innerHTML = '<div style="text-align: center; padding: 40px; color: #666;">Only available for decode calculations</div>';
    }
    
    // Chart 7: Compute vs Performance (for DECODE) or Bandwidth vs Performance (for TTFT)
    let xAxisData, xAxisTitle, chartTitle;
    
    if (result.calculation_type === 'TTFT') {
        xAxisData = result.bandwidth;
        xAxisTitle = 'Memory Bandwidth (TB/sec)';
        chartTitle = 'Bandwidth vs Performance';
    } else {
        // Convert from PFLOPS to TFLOPS for decode (multiply by 1000)
        xAxisData = result.compute.map(val => val * 1000);
        xAxisTitle = 'Compute (TF/sec)';
        chartTitle = 'Compute vs Performance';
    }
    
    // Update the chart title
    document.getElementById('batch-compute-performance-title').textContent = chartTitle;
    
    const computePerformanceTrace = {
        x: xAxisData,
        y: result.y_values,
        mode: 'lines+markers',
        name: chartTitle,
        line: { color: '#9b59b6', width: 3 },
        marker: { size: 8, color: '#8e44ad' },
        hovertemplate: '%{x:.1f}<br>Performance: %{y:.1f}<extra></extra>'
    };
    
    const performanceLabel = result.calculation_type === 'TTFT' ? 'TTFT (ms)' : 'TPS (tokens/sec per user)';
    
    const computePerformanceLayout = {
        xaxis: {
            title: xAxisTitle,
            autorange: true
        },
        yaxis: {
            title: performanceLabel,
            autorange: true
        },
        hovermode: 'closest',
        margin: { t: 20 }
    };
    
    Plotly.newPlot('batch-compute-performance-chart', [computePerformanceTrace], computePerformanceLayout, { responsive: true });
    
    // Chart 8: KV Cache Bandwidth
    const kvBandwidthTrace = {
        x: result.x_values,
        y: result.kv_bandwidth,
        mode: 'lines+markers',
        name: 'KV Cache Bandwidth',
        line: { color: '#e74c3c', width: 3 },
        marker: { size: 8, color: '#c0392b' },
        hovertemplate: '%{y:.1f}<extra></extra>'
    };
    
    const kvBandwidthLayout = {
        xaxis: {
            title: result.param_name,
            autorange: true
        },
        yaxis: {
            title: 'KV Bandwidth (TB/sec)',
            autorange: true
        },
        hovermode: 'closest',
        margin: { t: 20 }
    };
    
    Plotly.newPlot('batch-kv-bandwidth-chart', [kvBandwidthTrace], kvBandwidthLayout, { responsive: true });
    
    // Chart 9: Weights Bandwidth
    const weightsBandwidthTrace = {
        x: result.x_values,
        y: result.weights_bandwidth,
        mode: 'lines+markers',
        name: 'Weights Bandwidth',
        line: { color: '#16a085', width: 3 },
        marker: { size: 8, color: '#138d75' },
        hovertemplate: '%{y:.1f}<extra></extra>'
    };
    
    const weightsBandwidthLayout = {
        xaxis: {
            title: result.param_name,
            autorange: true
        },
        yaxis: {
            title: 'Weights Bandwidth (TB/sec)',
            autorange: true
        },
        hovermode: 'closest',
        margin: { t: 20 }
    };
    
    Plotly.newPlot('batch-weights-bandwidth-chart', [weightsBandwidthTrace], weightsBandwidthLayout, { responsive: true });
}

function exportConfiguration() {
    try {
        console.log('Export button clicked');
        
        // Helper function to safely get element value
        const getElementValue = (id) => {
            const element = document.getElementById(id);
            if (!element) {
                throw new Error(`Element with id '${id}' not found`);
            }
            return element.value;
        };
        
        // Get all current form data
        const config = {
            timestamp: new Date().toISOString(),
            model: getElementValue('model'),
            calculation_type: document.querySelector('input[name="calc_type"]:checked').value,
            system: {
                memory: parseFloat(getElementValue('memory')),
                compute: parseFloat(getElementValue('compute')),
                memory_bw: parseFloat(getElementValue('memory_bw')),
                network_bw: parseFloat(getElementValue('network_bw')),
                kernel_latency: parseFloat(getElementValue('kernel_latency')),
                dtype: getElementValue('dtype')
            },
            parallelism: {
                type: getElementValue('parallelism'),
                num_gpus: parseInt(getElementValue('num_gpus'))
            },
            workload: {
                batch_size: parseFloat(getElementValue('batch_size'))
            },
            batch_analysis: {
                parameter: getElementValue('batch_param'),
                min: parseFloat(getElementValue('batch_min')),
                max: parseFloat(getElementValue('batch_max')),
                points: parseInt(getElementValue('batch_points'))
            }
        };
        
        // Add calculation-type specific parameters
        if (config.calculation_type === 'TTFT') {
            config.workload.sequence_length = parseFloat(getElementValue('sequence_length'));
        } else {
            config.workload.prefill_length = parseFloat(getElementValue('prefill_length'));
            config.workload.output_length = parseFloat(getElementValue('output_length'));
        }
        
        console.log('Configuration gathered:', config);
        
        // Create filename with timestamp
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
        const calcType = config.calculation_type.toLowerCase();
        const filename = `llm-config-${calcType}-${timestamp}.json`;
        
        console.log('Creating file:', filename);
        
        // Create blob and download
        const blob = new Blob([JSON.stringify(config, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        console.log('Configuration exported successfully:', filename);
        alert(`Configuration exported: ${filename}`);
    } catch (error) {
        console.error('Error exporting configuration:', error);
        alert('Error exporting configuration: ' + error.message);
    }
}

async function exportResults() {
    try {
        console.log('Export results called');
        if (!window.lastBatchResults) {
            alert('No batch results to export. Run a batch analysis first.');
            return;
        }

        console.log('Gathering configuration...');
        
        // Gather all current configuration with error checking
        const getElementValue = (id, defaultValue = null) => {
            const elem = document.getElementById(id);
            if (!elem) {
                console.error(`Element with id '${id}' not found`);
                return defaultValue;
            }
            console.log(`Got ${id}: ${elem.value}`);
            return elem.value;
        };
        
        const config = {
            model: getElementValue('model'),
            memory: parseFloat(getElementValue('memory', '0')),
            memory_bandwidth: parseFloat(getElementValue('memory_bw', '0')),
            compute: parseFloat(getElementValue('compute', '0')),
            network_bandwidth: parseFloat(getElementValue('network_bw', '0')),
            kernel_latency: parseFloat(getElementValue('kernel_latency', '0')),
            calculation_type: document.querySelector('input[name="calc_type"]:checked')?.value || 'TTFT',
            batch_size: parseInt(getElementValue('batch_size', '1')),
            prefill_length: parseInt(getElementValue('prefill_length', '0')),
            output_length: parseInt(getElementValue('output_length', '0')),
            sequence_length: parseInt(getElementValue('sequence_length', '0')),
            parallelism: getElementValue('parallelism', 'None'),
            num_gpus: parseInt(getElementValue('num_gpus', '1')),
            dtype: getElementValue('dtype', 'fp16'),
            batch_results: window.lastBatchResults
        };
        
        console.log('Configuration gathered:', config);

        // Send to backend for XLSX generation
        const response = await fetch('/api/export_results', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(config)
        });

        if (!response.ok) {
            throw new Error('Failed to generate Excel file');
        }

        // Download the file
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
        a.download = `batch_results_${timestamp}.xlsx`;
        
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

    } catch (error) {
        console.error('Error exporting results:', error);
        alert('Error exporting results: ' + error.message);
    }
}

function showError(message) {
    const resultsDiv = document.getElementById('results-text');
    resultsDiv.innerHTML = `
        <h2>Error</h2>
        <div class="error">${message}</div>
    `;
}
