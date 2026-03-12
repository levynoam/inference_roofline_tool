"""
Web-based UI for LLM Inference Performance Analyzer
"""

from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import pandas as pd
from io import BytesIO
from llm_configs import ALL_MODELS
from inference_performance import InferencePerformance, SystemConstraints, ParallelismConfig, ParallelismType

# Debug flag for verbose output
DEBUG_VERBOSE = False

app = Flask(__name__)

@app.route('/')
def index():
    """Serve the main page"""
    # Get list of models for dropdown
    models = list(ALL_MODELS.keys())
    return render_template('index.html', models=models)

@app.route('/api/calculate', methods=['POST'])
def calculate():
    """Calculate performance based on inputs"""
    print("\n" + "="*60)
    print("CALCULATE REQUEST RECEIVED")
    print("="*60)
    try:
        data = request.json
        print(f"Model: {data.get('model', 'N/A')}")
        print(f"Calculation Type: {data.get('calculation_type', 'N/A')}")
        print(f"Batch Size: {data.get('batch_size', 'N/A')}")
        
        # Get model
        model_name = data['model']
        model = ALL_MODELS[model_name]
        
        # Update kernel launch latency
        kernel_latency_us = float(data['kernel_latency'])
        model.kernel_launch_latency = kernel_latency_us * 1e-6
        
        # Create performance calculator
        perf = InferencePerformance(model)
        
        # Build system constraints
        gpu = SystemConstraints(
            memory_capacity=float(data['memory']) * 1e9,  # GB to bytes
            memory_bandwidth=float(data['memory_bw']) * 1e9,  # GB/s to bytes/s
            compute_throughput=float(data['compute']) * 1e12,  # TFLOPS to FLOPS
            network_bandwidth=float(data['network_bw']) * 1e9,  # GB/s to bytes/s
            persistent_storage_bandwidth=float(data.get('storage_bw', 20)) * 1e9  # GB/s to bytes/s
        )
        
        # Build parallelism config
        parallel_config = None
        if data['parallelism'] != 'None':
            parallel_config = ParallelismConfig(
                parallelism_type=ParallelismType[data['parallelism']],
                tensor_parallel_size=int(data['tp_size']),
                pipeline_parallel_size=int(data['pp_size'])
            )
        
        batch_size = int(data['batch_size'])
        calc_type = data['calculation_type']
        
        # Get dtype_override if provided
        dtype_override = data.get('dtype_override', None)
        
        # Run calculation
        if calc_type == 'TTFT':
            sequence_length = int(data['sequence_length'])
            result = perf.calculate_achievable_ttft(
                system_constraints=gpu,
                batch_size=batch_size,
                sequence_length=sequence_length,
                parallelism_config=parallel_config,
                kernel_launch_latency=kernel_latency_us * 1e-6,
                dtype_override=dtype_override
            )
            
            # Calculate total GPUs for normalization
            total_gpus = 1
            if parallel_config:
                total_gpus = parallel_config.tensor_parallel_size * parallel_config.pipeline_parallel_size
            
            response = {
                'success': True,
                'calculation_type': 'TTFT',
                'metrics': {
                    'ttft_ms': result.achievable_ttft * 1000,
                    'throughput': batch_size / result.achievable_ttft
                },
                'utilization': {
                    'compute': result.compute_utilization * 100,
                    'memory_bw': result.memory_bandwidth_utilization * 100,
                    'network_bw': result.network_bandwidth_utilization * 100,
                    'memory': result.memory_utilization * 100,
                    'storage_bw': result.persistent_storage_bandwidth_utilization * 100
                },
                'resources': {
                    'compute_used': (result.compute_used / total_gpus) / 1e15,  # PFLOPs/s per GPU
                    'compute_available': result.compute_available / 1e15,
                    'memory_bw_used': (result.memory_bandwidth_used / total_gpus) / 1e12,  # TB/s per GPU
                    'memory_bw_available': result.memory_bandwidth_available / 1e12,
                    'network_bw_used': result.network_bandwidth_used / 1e12,  # TB/s per GPU (already per-GPU, don't divide)
                    'network_bw_available': result.network_bandwidth_available / 1e12,
                    'storage_bw_used': (result.persistent_storage_bandwidth_used / total_gpus) / 1e9,  # GB/s per GPU
                    'storage_bw_available': result.persistent_storage_bandwidth_available / 1e9,
                    'memory_used': (result.memory_used / total_gpus) / 1e9,  # GB per GPU
                    'memory_available': result.memory_available / 1e9
                },
                'memory_breakdown': {
                    'weights': (result.memory_weights / total_gpus) / 1e9,  # GB per GPU
                    'kv_cache': (result.memory_kv_cache / total_gpus) / 1e9,
                    'activations': (result.memory_activations / total_gpus) / 1e9
                },
                'compute_breakdown': {
                    'attention': (result.compute_attention / total_gpus) / 1e12,  # TFLOPs per GPU
                    'ffn': (result.compute_ffn / total_gpus) / 1e12,
                    'other': (result.compute_other / total_gpus) / 1e12
                },
                'time_breakdown': {
                    'compute_busy': result.time_compute_busy * 1000,  # ms
                    'kernel_launch': result.time_kernel_launch * 1000,  # ms
                    'idle': result.time_idle * 1000  # ms
                },
                'bottleneck': result.bottleneck_resource,
                'model_architecture': {
                    'name': model.model_name,
                    'parameters': model.total_parameters,
                    'layers': model.num_layers,
                    'hidden_dim': model.hidden_dim,
                    'is_moe': model.is_moe,
                    'moe_experts': model.moe_config.num_experts if model.moe_config else None,
                    'moe_active_experts': model.moe_config.num_experts_per_token if model.moe_config else None,
                    'active_parameters': model.active_parameters if model.is_moe else model.total_parameters,
                    # Attention details
                    'attention_type': model.attention_config.attention_type.value,
                    'num_attention_heads': model.attention_config.num_attention_heads,
                    'num_kv_heads': model.attention_config.num_key_value_heads,
                    'head_dim': model.attention_config.head_dim,
                    'use_mla': model.attention_config.use_mla,
                    'mla_kv_lora_rank': model.attention_config.mla_kv_lora_rank,
                    # Sliding window attention
                    'sliding_window_size': model.attention_config.sliding_window_size,
                    'has_layer_types': model.layer_types is not None,
                    'num_full_attention_layers': model.get_num_full_attention_layers() if model.layer_types else None,
                    'num_sliding_attention_layers': model.get_num_sliding_attention_layers() if model.layer_types else None,
                    # Hybrid Mamba/Attention
                    'is_hybrid': model.hybrid_layer_types is not None,
                    'num_mamba_layers': model.get_num_mamba_layers() if model.hybrid_layer_types else None,
                    'num_attention_layers': model.get_num_attention_layers_hybrid() if model.hybrid_layer_types else None,
                    'has_mamba_config': model.mamba_config is not None,
                    'mamba_state_size': model.mamba_config.state_size if model.mamba_config else None,
                    'mamba_num_heads': model.mamba_config.num_heads if model.mamba_config else None,
                    'mamba_head_dim': model.mamba_config.head_dim if model.mamba_config else None,
                    # Interleaved Dense/MoE FFN
                    'has_ffn_layer_types': model.ffn_layer_types is not None,
                    'num_dense_ffn_layers': model.get_num_dense_ffn_layers() if model.ffn_layer_types else None,
                    'num_moe_ffn_layers': model.get_num_moe_ffn_layers() if model.ffn_layer_types else None,
                    'dense_intermediate_size': model.get_dense_intermediate_size() if model.ffn_layer_types else None,
                    'moe_intermediate_size': model.get_moe_intermediate_size() if model.ffn_layer_types else None,
                    # Linear Attention
                    'has_linear_attention': model.linear_attention_config is not None,
                    'num_linear_attention_layers': model.get_num_linear_attention_layers() if model.linear_attention_config else None,
                    'num_full_attn_layers_linear': (model.num_layers - model.get_num_linear_attention_layers()) if model.linear_attention_config else None,
                    'linear_key_heads': model.linear_attention_config.num_key_heads if model.linear_attention_config else None,
                    'linear_key_dim': model.linear_attention_config.key_head_dim if model.linear_attention_config else None,
                    'linear_value_heads': model.linear_attention_config.num_value_heads if model.linear_attention_config else None,
                    'linear_value_dim': model.linear_attention_config.value_head_dim if model.linear_attention_config else None,
                    'linear_conv_kernel': model.linear_attention_config.conv_kernel_dim if model.linear_attention_config else None,
                }
            }
            
            # Per-layer breakdown for graphs
            p_config = parallel_config or ParallelismConfig()
            plb = perf.calculate_per_layer_breakdown('prefill', batch_size, sequence_length, p_config, dtype_override)

            # Compute per-layer bandwidth (GB/s) from traffic and execution time
            peak_flops = gpu.compute_throughput
            peak_bw = gpu.memory_bandwidth
            k_latency = model.kernel_launch_latency
            attn_bw = []
            non_attn_bw = []
            for i in range(plb.num_layers):
                a_ct = plb.attention_compute[i] / peak_flops if peak_flops > 0 else 0
                a_mt = plb.attention_memory_traffic[i] / peak_bw if peak_bw > 0 else 0
                a_kt = plb.attention_kernels[i] * k_latency
                a_time = max(a_ct, a_mt) + a_kt
                f_ct = plb.non_attention_compute[i] / peak_flops if peak_flops > 0 else 0
                f_mt = plb.non_attention_memory_traffic[i] / peak_bw if peak_bw > 0 else 0
                f_kt = plb.non_attention_kernels[i] * k_latency
                f_time = max(f_ct, f_mt) + f_kt
                attn_bw.append(plb.attention_memory_traffic[i] / a_time / 1e9 if a_time > 0 else 0)
                non_attn_bw.append(plb.non_attention_memory_traffic[i] / f_time / 1e9 if f_time > 0 else 0)

            response['per_layer_breakdown'] = {
                'mode': plb.mode,
                'num_layers': plb.num_layers,
                'sequence_length': plb.sequence_length,
                'layer_types': plb.layer_types,
                'attention_compute': plb.attention_compute,
                'non_attention_compute': plb.non_attention_compute,
                'attention_bandwidth': attn_bw,
                'non_attention_bandwidth': non_attn_bw,
                'attention_kernels': plb.attention_kernels,
                'non_attention_kernels': plb.non_attention_kernels,
            }
            
        else:  # DECODE
            prefill_length = int(data['prefill_length'])
            output_length = int(data['output_length'])
            result = perf.calculate_decode_performance(
                system_constraints=gpu,
                batch_size=batch_size,
                prefill_length=prefill_length,
                output_length=output_length,
                parallelism_config=parallel_config,
                kernel_launch_latency=kernel_latency_us * 1e-6,
                dtype_override=dtype_override,
                return_step_details=True
            )
            
            # Calculate total GPUs for normalization
            total_gpus = 1
            if parallel_config:
                total_gpus = parallel_config.tensor_parallel_size * parallel_config.pipeline_parallel_size
            
            # Calculate actual network bandwidth required (from traffic, not utilization)
            # Use the same approach as compute/memory: divide traffic by bottleneck time (not total time with kernel overhead)
            total_network_traffic = sum(step.network_traffic for step in result.step_details)
            # Calculate bottleneck time (max of compute/memory/network time across all steps, without kernel overhead)
            total_bottleneck_time = sum(max(step.compute_time, step.memory_bw_time, step.network_time) for step in result.step_details)
            avg_network_bw_required = (total_network_traffic / total_bottleneck_time) / 1e12 if total_bottleneck_time > 0 else 0.0  # TB/s
            
            response = {
                'success': True,
                'calculation_type': 'DECODE',
                'metrics': {
                    'total_time_ms': result.total_decode_time * 1000,
                    'avg_step_time_ms': result.avg_step_time * 1000,
                    'tps': result.tokens_per_second_per_user,
                    'system_throughput': result.total_throughput
                },
                'utilization': {
                    'compute': result.avg_compute_utilization * 100,
                    'memory_bw': result.avg_memory_bw_utilization * 100,
                    'network_bw': result.avg_network_bw_utilization * 100,
                    'memory': result.avg_memory_utilization * 100,
                    'storage_bw': result.avg_storage_bw_utilization * 100
                },
                'resources': {
                    'compute_used': (result.avg_compute_utilization * result.compute_throughput) / 1e15,  # PFLOPs/s per GPU
                    'memory_bw_used': (result.avg_memory_bw_utilization * result.memory_bandwidth) / 1e12,  # TB/s per GPU
                    'network_bw_used': avg_network_bw_required,  # TB/s - actual required bandwidth
                    'storage_bw_used': (result.avg_storage_bw_utilization * result.persistent_storage_bandwidth) / 1e9,  # GB/s per GPU
                    'memory_used': (result.avg_memory_utilization * result.memory_capacity) / 1e9  # GB per GPU
                },
                'memory_breakdown': {
                    'weights': result.avg_memory_weights / 1e9,  # GB per GPU (average)
                    'kv_cache': result.avg_memory_kv_cache / 1e9,
                    'activations': result.avg_memory_activations / 1e9
                },
                'compute_breakdown': {
                    'attention': result.total_compute_attention / 1e12,  # TFLOPs (total across all steps)
                    'ffn': result.total_compute_ffn / 1e12,
                    'other': result.total_compute_other / 1e12
                },
                'time_breakdown': {
                    'compute_busy': result.total_time_compute_busy * 1000,  # ms
                    'kernel_launch': result.total_time_kernel_launch * 1000,  # ms
                    'idle': result.total_time_idle * 1000  # ms
                },
                'bottleneck': result.primary_bottleneck,
                'model_architecture': {
                    'name': model.model_name,
                    'parameters': model.total_parameters,
                    'layers': model.num_layers,
                    'hidden_dim': model.hidden_dim,
                    'is_moe': model.is_moe,
                    'moe_experts': model.moe_config.num_experts if model.moe_config else None,
                    'moe_active_experts': model.moe_config.num_experts_per_token if model.moe_config else None,
                    'active_parameters': model.active_parameters if model.is_moe else model.total_parameters,
                    # Attention details
                    'attention_type': model.attention_config.attention_type.value,
                    'num_attention_heads': model.attention_config.num_attention_heads,
                    'num_kv_heads': model.attention_config.num_key_value_heads,
                    'head_dim': model.attention_config.head_dim,
                    'use_mla': model.attention_config.use_mla,
                    'mla_kv_lora_rank': model.attention_config.mla_kv_lora_rank,
                    # Sliding window attention
                    'sliding_window_size': model.attention_config.sliding_window_size,
                    'has_layer_types': model.layer_types is not None,
                    'num_full_attention_layers': model.get_num_full_attention_layers() if model.layer_types else None,
                    'num_sliding_attention_layers': model.get_num_sliding_attention_layers() if model.layer_types else None,
                    # Hybrid Mamba/Attention
                    'is_hybrid': model.hybrid_layer_types is not None,
                    'num_mamba_layers': model.get_num_mamba_layers() if model.hybrid_layer_types else None,
                    'num_attention_layers': model.get_num_attention_layers_hybrid() if model.hybrid_layer_types else None,
                    'has_mamba_config': model.mamba_config is not None,
                    'mamba_state_size': model.mamba_config.state_size if model.mamba_config else None,
                    'mamba_num_heads': model.mamba_config.num_heads if model.mamba_config else None,
                    'mamba_head_dim': model.mamba_config.head_dim if model.mamba_config else None,
                    # Interleaved Dense/MoE FFN
                    'has_ffn_layer_types': model.ffn_layer_types is not None,
                    'num_dense_ffn_layers': model.get_num_dense_ffn_layers() if model.ffn_layer_types else None,
                    'num_moe_ffn_layers': model.get_num_moe_ffn_layers() if model.ffn_layer_types else None,
                    'dense_intermediate_size': model.get_dense_intermediate_size() if model.ffn_layer_types else None,
                    'moe_intermediate_size': model.get_moe_intermediate_size() if model.ffn_layer_types else None,
                    # Linear Attention
                    'has_linear_attention': model.linear_attention_config is not None,
                    'num_linear_attention_layers': model.get_num_linear_attention_layers() if model.linear_attention_config else None,
                    'num_full_attn_layers_linear': (model.num_layers - model.get_num_linear_attention_layers()) if model.linear_attention_config else None,
                    'linear_key_heads': model.linear_attention_config.num_key_heads if model.linear_attention_config else None,
                    'linear_key_dim': model.linear_attention_config.key_head_dim if model.linear_attention_config else None,
                    'linear_value_heads': model.linear_attention_config.num_value_heads if model.linear_attention_config else None,
                    'linear_value_dim': model.linear_attention_config.value_head_dim if model.linear_attention_config else None,
                    'linear_conv_kernel': model.linear_attention_config.conv_kernel_dim if model.linear_attention_config else None,
                }
            }
            
            # Per-layer breakdown for graphs (use mid-point context for decode)
            p_config = parallel_config or ParallelismConfig()
            mid_context = prefill_length + output_length // 2
            plb = perf.calculate_per_layer_breakdown('decode', batch_size, mid_context, p_config, dtype_override)

            # Compute per-layer bandwidth (GB/s) from traffic and execution time
            peak_flops = gpu.compute_throughput
            peak_bw = gpu.memory_bandwidth
            k_latency = model.kernel_launch_latency
            attn_bw = []
            non_attn_bw = []
            for i in range(plb.num_layers):
                a_ct = plb.attention_compute[i] / peak_flops if peak_flops > 0 else 0
                a_mt = plb.attention_memory_traffic[i] / peak_bw if peak_bw > 0 else 0
                a_kt = plb.attention_kernels[i] * k_latency
                a_time = max(a_ct, a_mt) + a_kt
                f_ct = plb.non_attention_compute[i] / peak_flops if peak_flops > 0 else 0
                f_mt = plb.non_attention_memory_traffic[i] / peak_bw if peak_bw > 0 else 0
                f_kt = plb.non_attention_kernels[i] * k_latency
                f_time = max(f_ct, f_mt) + f_kt
                attn_bw.append(plb.attention_memory_traffic[i] / a_time / 1e9 if a_time > 0 else 0)
                non_attn_bw.append(plb.non_attention_memory_traffic[i] / f_time / 1e9 if f_time > 0 else 0)

            response['per_layer_breakdown'] = {
                'mode': plb.mode,
                'num_layers': plb.num_layers,
                'sequence_length': plb.sequence_length,
                'layer_types': plb.layer_types,
                'attention_compute': plb.attention_compute,
                'non_attention_compute': plb.non_attention_compute,
                'attention_bandwidth': attn_bw,
                'non_attention_bandwidth': non_attn_bw,
                'attention_kernels': plb.attention_kernels,
                'non_attention_kernels': plb.non_attention_kernels,
            }
        
        return jsonify(response)
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        tb = traceback.format_exc()
        print(f"\n❌ CALCULATION ERROR:")
        print(error_msg)
        print("\nFull traceback:")
        print(tb)
        return jsonify({
            'success': False,
            'error': error_msg,
            'traceback': tb
        }), 400

@app.route('/api/batch', methods=['POST'])
def batch_analysis():
    """Run batch analysis with parameter sweep"""
    if DEBUG_VERBOSE:
        print("\n" + "="*60)
        print("BATCH ANALYSIS REQUEST RECEIVED")
        print("="*60)
    try:
        data = request.json
        if DEBUG_VERBOSE:
            print(f"Parameter: {data.get('param_name', 'N/A')}")
            print(f"Range: {data.get('min_val', 'N/A')} to {data.get('max_val', 'N/A')}")
            print(f"Points: {data.get('num_points', 'N/A')}")
        
        param_name = data['param_name']
        min_val = float(data['min_val'])
        max_val = float(data['max_val'])
        num_points = int(data['num_points'])
        
        if min_val >= max_val:
            return jsonify({'success': False, 'error': 'Min value must be less than max value'}), 400
        
        if num_points < 2:
            return jsonify({'success': False, 'error': 'Number of points must be at least 2'}), 400
        
        # Generate sweep values
        sweep_values = np.logspace(np.log10(min_val), np.log10(max_val), num_points)
        
        results_x = []
        results_y = []
        results_bandwidth = []
        results_compute = []
        results_network = []
        results_kernel_overhead = []
        results_tps = []
        results_throughput = []
        results_kv_bandwidth = []
        results_weights_bandwidth = []
        
        # Get model
        model_name = data['model']
        model = ALL_MODELS[model_name]
        kernel_latency_us = float(data['kernel_latency'])
        model.kernel_launch_latency = kernel_latency_us * 1e-6
        
        perf = InferencePerformance(model)
        calc_type = data['calculation_type']
        
        for value in sweep_values:
            try:
                # Update the parameter
                current_data = data.copy()
                current_data[param_name] = value
                
                if DEBUG_VERBOSE:
                    print(f"\nDEBUG: Processing {param_name} = {value}")
                
                # Update kernel latency if that's what we're sweeping
                if param_name == 'kernel_latency':
                    model.kernel_launch_latency = float(value) * 1e-6
                
                # Build system constraints
                gpu = SystemConstraints(
                    memory_capacity=float(current_data['memory']) * 1e9,
                    memory_bandwidth=float(current_data['memory_bw']) * 1e9,
                    compute_throughput=float(current_data['compute']) * 1e12,
                    network_bandwidth=float(current_data['network_bw']) * 1e9,
                    persistent_storage_bandwidth=float(current_data.get('storage_bw', 20)) * 1e9
                )
                
                if DEBUG_VERBOSE:
                    print(f"  Storage BW: {gpu.persistent_storage_bandwidth / 1e9:.2f} GB/s")
                
                # Build parallelism config
                parallel_config = None
                if current_data['parallelism'] != 'None':
                    parallel_config = ParallelismConfig(
                        parallelism_type=ParallelismType[current_data['parallelism']],
                        tensor_parallel_size=int(current_data['tp_size']),
                        pipeline_parallel_size=int(current_data['pp_size'])
                    )
                
                batch_size = int(current_data['batch_size'])
                dtype_override = current_data.get('dtype_override', None)
                
                if calc_type == 'TTFT':
                    sequence_length = int(current_data['sequence_length'])
                    result = perf.calculate_achievable_ttft(
                        system_constraints=gpu,
                        batch_size=batch_size,
                        sequence_length=sequence_length,
                        parallelism_config=parallel_config,
                        kernel_launch_latency=float(current_data['kernel_latency']) * 1e-6,
                        dtype_override=dtype_override
                    )
                    results_x.append(value)
                    results_y.append(result.achievable_ttft * 1000)
                    results_bandwidth.append(result.memory_bandwidth_used / 1e12)
                    results_compute.append(result.compute_used / 1e15)
                    results_network.append(result.network_bandwidth_used / 1e9)
                    
                    # Calculate kernel overhead: kernel_time / total_time * 100
                    kernel_overhead_pct = (result.kernel_launch_overhead / result.achievable_ttft) * 100 if result.achievable_ttft > 0 else 0.0
                    results_kernel_overhead.append(kernel_overhead_pct)
                    
                    # Calculate KV and weights bandwidth for TTFT
                    # For prefill, we don't have step-by-step details, so we estimate
                    total_memory_bw_used = result.memory_bandwidth_used  # bytes/sec
                    
                    # Approximate split: ~85% weights, ~15% KV+activations
                    weights_bw_tbps = (total_memory_bw_used * 0.85) / 1e12
                    kv_bw_tbps = (total_memory_bw_used * 0.15) / 1e12
                    
                    if DEBUG_VERBOSE:
                        print(f"DEBUG: TTFT - Calculating KV/weights BW for {param_name}={value}")
                        print(f"  Total Memory BW: {total_memory_bw_used / 1e12:.3f} TB/s")
                        print(f"  KV BW (15%): {kv_bw_tbps:.3f} TB/s")
                        print(f"  Weights BW (85%): {weights_bw_tbps:.3f} TB/s")
                    
                    results_kv_bandwidth.append(kv_bw_tbps)
                    results_weights_bandwidth.append(weights_bw_tbps)
                    
                    # TPS and throughput not available for TTFT
                    results_tps.append(None)
                    results_throughput.append(None)
                else:
                    prefill_length = int(current_data['prefill_length'])
                    output_length = int(current_data['output_length'])
                    result = perf.calculate_decode_performance(
                        system_constraints=gpu,
                        batch_size=batch_size,
                        prefill_length=prefill_length,
                        output_length=output_length,
                        parallelism_config=parallel_config,
                        kernel_launch_latency=float(current_data['kernel_latency']) * 1e-6,
                        dtype_override=dtype_override,
                        return_step_details=True
                    )
                    # Calculate actual compute used per step
                    total_compute = result.total_compute_attention + result.total_compute_ffn + result.total_compute_other
                    compute_per_step = total_compute / output_length
                    compute_pflops = compute_per_step / result.avg_step_time / 1e15
                    
                    # Calculate actual network bandwidth required (from traffic, not utilization)
                    # Use bottleneck time (not total time with kernel overhead)
                    total_network_traffic = sum(step.network_traffic for step in result.step_details)
                    total_bottleneck_time = sum(max(step.compute_time, step.memory_bw_time, step.network_time) for step in result.step_details)
                    avg_network_bw_required = (total_network_traffic / total_bottleneck_time) / 1e9 if total_bottleneck_time > 0 else 0.0  # GB/s
                    
                    results_x.append(value)
                    results_y.append(result.tokens_per_second_per_user)
                    results_bandwidth.append(result.memory_bandwidth / 1e12)
                    results_compute.append(compute_pflops)
                    results_network.append(avg_network_bw_required)
                    
                    # Calculate kernel overhead: kernel_launch_time / total_time * 100
                    kernel_overhead_pct = (result.total_time_kernel_launch / result.total_decode_time) * 100 if result.total_decode_time > 0 else 0.0
                    results_kernel_overhead.append(kernel_overhead_pct)
                    
                    # Track TPS and throughput for decode
                    results_tps.append(result.tokens_per_second_per_user)
                    results_throughput.append(result.total_throughput)
                    
                    # Calculate KV and weights bandwidth separately
                    # We need to recalculate the breakdown from step details
                    if DEBUG_VERBOSE:
                        print(f"DEBUG: Calculating KV/weights BW for {param_name}={value}")
                        print(f"  Has step_details: {hasattr(result, 'step_details')}")
                        if hasattr(result, 'step_details'):
                            print(f"  Number of steps: {len(result.step_details)}")
                    
                    bytes_per_param = perf._get_bytes_per_param(dtype_override)
                    model_size = model.total_parameters * bytes_per_param
                    
                    # Handle parallelism config (may be None)
                    tp_size = parallel_config.tensor_parallel_size if parallel_config else 1
                    pp_size = parallel_config.pipeline_parallel_size if parallel_config else 1
                    
                    model_size /= tp_size
                    
                    total_kv_traffic = 0
                    total_weights_traffic = 0
                    total_time = 0
                    
                    # Calculate per-step breakdown
                    for i, step in enumerate(result.step_details):
                        step_time = max(step.compute_time, step.memory_bw_time, step.network_time)
                        total_time += step_time
                        
                        # Calculate KV cache size for this step
                        context_len = prefill_length + i
                        effective_seq_len = context_len
                        if model.attention_config.use_dsa and model.attention_config.dsa_top_k:
                            effective_seq_len = min(context_len, model.attention_config.dsa_top_k)
                        
                        kv_cache_size = model.get_kv_cache_size(
                            batch_size=batch_size,
                            sequence_length=effective_seq_len,
                            bytes_per_element=bytes_per_param
                        )
                        kv_cache_size /= (tp_size * pp_size)
                        
                        total_weights_traffic += model_size
                        total_kv_traffic += kv_cache_size
                    
                    kv_bw_tbps = (total_kv_traffic / total_time) / 1e12 if total_time > 0 else 0.0
                    weights_bw_tbps = (total_weights_traffic / total_time) / 1e12 if total_time > 0 else 0.0
                    
                    if DEBUG_VERBOSE:
                        print(f"  KV BW: {kv_bw_tbps:.3f} TB/s")
                        print(f"  Weights BW: {weights_bw_tbps:.3f} TB/s")
                    
                    results_kv_bandwidth.append(kv_bw_tbps)
                    results_weights_bandwidth.append(weights_bw_tbps)
                    
                    if DEBUG_VERBOSE:
                        print(f"DEBUG: Decode at {param_name}={value}: TPS={result.tokens_per_second_per_user}, Throughput={result.total_throughput}")
                    
            except Exception as e:
                print(f"Warning: Calculation failed at {param_name}={value}: {e}")
                continue
        
        if len(results_x) == 0:
            return jsonify({'success': False, 'error': 'No successful calculations in batch run'}), 400
        
        if DEBUG_VERBOSE:
            print(f"DEBUG: Final arrays before JSON response:")
            print(f"  results_tps length: {len(results_tps)}, values: {results_tps}")
            print(f"  results_throughput length: {len(results_throughput)}, values: {results_throughput}")
            print(f"  calculation_type: {calc_type}")
        
        return jsonify({
            'success': True,
            'param_name': param_name,
            'x_values': results_x,
            'y_values': results_y,
            'bandwidth': results_bandwidth,
            'compute': results_compute,
            'network': results_network,
            'kernel_overhead': results_kernel_overhead,
            'tps': results_tps,
            'throughput': results_throughput,
            'kv_bandwidth': results_kv_bandwidth,
            'weights_bandwidth': results_weights_bandwidth,
            'calculation_type': calc_type
        })
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        tb = traceback.format_exc()
        print(f"\n❌ BATCH ANALYSIS ERROR:")
        print(error_msg)
        print("\nFull traceback:")
        print(tb)
        return jsonify({
            'success': False,
            'error': error_msg,
            'traceback': tb
        }), 400

@app.route('/api/export_results', methods=['POST'])
def export_results():
    """Export batch analysis results to Excel"""
    try:
        data = request.json
        batch_results = data['batch_results']
        
        # Create DataFrame with all the data
        rows = []
        for i in range(len(batch_results['x_values'])):
            row = {
                # Configuration parameters
                'Model': data['model'],
                'Memory (GB)': data['memory'],
                'Memory BW (GB/s)': data['memory_bandwidth'],
                'Compute (TFLOPS)': data['compute'],
                'Network BW (GB/s)': data['network_bandwidth'],
                'Kernel Latency (us)': data['kernel_latency'],
                'Calculation Type': data['calculation_type'],
                'Batch Size': data['batch_size'],
                'Prefill Length': data['prefill_length'],
                'Output Length': data['output_length'],
                'Sequence Length': data['sequence_length'],
                'Parallelism': data['parallelism'],
                'Num GPUs': data.get('num_gpus', 1),
                'Data Type': data['dtype'],
                # Sweep parameter
                batch_results['param_name']: batch_results['x_values'][i],
                # Results
            }
            
            # Add performance metric (TTFT or TPS)
            if batch_results['calculation_type'] == 'TTFT':
                row['TTFT (ms)'] = batch_results['y_values'][i]
            else:
                row['TPS (tokens/sec)'] = batch_results['y_values'][i]
            
            # Add other metrics
            row['Memory BW Used (TB/sec)'] = batch_results['bandwidth'][i]
            row['Compute Used (PF/sec)'] = batch_results['compute'][i]
            row['Network BW Used (GB/sec)'] = batch_results['network'][i]
            row['Kernel Overhead (%)'] = batch_results['kernel_overhead'][i]
            
            # Add TPS and Throughput if available (decode only)
            if batch_results.get('tps') and i < len(batch_results['tps']):
                tps_val = batch_results['tps'][i]
                throughput_val = batch_results['throughput'][i]
                if tps_val is not None:
                    row['TPS per User'] = tps_val
                if throughput_val is not None:
                    row['Total Throughput'] = throughput_val
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Write to Excel in memory
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Batch Results')
            
            # Auto-adjust column widths
            worksheet = writer.sheets['Batch Results']
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width
        
        output.seek(0)
        
        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name='batch_results.xlsx'
        )
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        tb = traceback.format_exc()
        print(f"Error exporting results: {error_msg}")
        print(tb)
        return jsonify({
            'success': False,
            'error': error_msg
        }), 400

if __name__ == '__main__':
    print("="*60)
    print("LLM Inference Performance Analyzer - Web UI")
    print("="*60)
    print(f"Available models: {len(ALL_MODELS)}")
    for model_name in ALL_MODELS.keys():
        print(f"  - {model_name}")
    print("="*60)
    print("Starting Flask server...")
    print("Open http://127.0.0.1:5000 in your browser")
    print("="*60)
    app.run(debug=True, port=5000)
