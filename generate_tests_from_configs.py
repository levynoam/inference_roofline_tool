"""
Generate unit tests from exported configuration files.

This script reads exported configuration JSON files from the logs/ directory
and generates pytest test cases that can be added to test_comprehensive.py.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any
from llm_configs import ALL_MODELS
from inference_performance import (
    InferencePerformance, SystemConstraints, ParallelismConfig, ParallelismType
)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def run_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run the configuration and capture results."""
    # Load model
    model_name = config['model']
    model = ALL_MODELS.get(model_name)
    if not model:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Build system constraints
    gpu = SystemConstraints(
        memory_capacity=config['system']['memory'] * 1e9,
        memory_bandwidth=config['system']['memory_bw'] * 1e9,
        compute_throughput=config['system']['compute'] * 1e12,
        network_bandwidth=config['system']['network_bw'] * 1e9
    )
    
    # Build parallelism config
    parallel_config = None
    if config['parallelism']['type'] != 'None':
        num_gpus = config['parallelism']['num_gpus']
        if config['parallelism']['type'] == 'TENSOR_PARALLEL':
            tp_size = num_gpus
            pp_size = 1
        else:  # PIPELINE_PARALLEL
            tp_size = 1
            pp_size = num_gpus
        
        parallel_config = ParallelismConfig(
            parallelism_type=ParallelismType[config['parallelism']['type']],
            tensor_parallel_size=tp_size,
            pipeline_parallel_size=pp_size
        )
    
    # Get workload parameters
    batch_size = config['workload']['batch_size']
    kernel_latency = config['system']['kernel_latency'] * 1e-6
    dtype_override = config['system']['dtype'] if config['system']['dtype'] else None
    
    # Create performance analyzer
    perf = InferencePerformance(model)
    
    # Run calculation based on type
    if config['calculation_type'] == 'TTFT':
        sequence_length = config['workload']['sequence_length']
        result = perf.calculate_achievable_ttft(
            system_constraints=gpu,
            batch_size=batch_size,
            sequence_length=sequence_length,
            parallelism_config=parallel_config,
            kernel_launch_latency=kernel_latency,
            dtype_override=dtype_override
        )
        
        return {
            'type': 'TTFT',
            'ttft_ms': result.achievable_ttft * 1000,
            'bottleneck': result.bottleneck_resource,
            'compute_util': result.compute_utilization,
            'memory_bw_util': result.memory_bandwidth_utilization,
            'network_bw_util': result.network_bandwidth_utilization,
            'memory_util': result.memory_utilization,
            'compute_used_pflops': result.compute_used / 1e15,
            'memory_bw_used_tbps': result.memory_bandwidth_used / 1e12,
            'network_bw_used_gbps': result.network_bandwidth_used / 1e9,
            'memory_used_gb': result.memory_used / 1e9
        }
    else:  # DECODE
        prefill_length = config['workload']['prefill_length']
        output_length = config['workload']['output_length']
        result = perf.calculate_decode_performance(
            system_constraints=gpu,
            batch_size=batch_size,
            prefill_length=prefill_length,
            output_length=output_length,
            parallelism_config=parallel_config,
            kernel_launch_latency=kernel_latency,
            dtype_override=dtype_override,
            return_step_details=True
        )
        
        # Calculate network bandwidth from step details
        total_network_traffic = sum(step.network_traffic for step in result.step_details)
        total_bottleneck_time = sum(max(step.compute_time, step.memory_bw_time, step.network_time) 
                                     for step in result.step_details)
        network_bw_gbps = (total_network_traffic / total_bottleneck_time) / 1e9 if total_bottleneck_time > 0 else 0.0
        
        return {
            'type': 'DECODE',
            'total_time_ms': result.total_decode_time * 1000,
            'avg_step_time_ms': result.avg_step_time * 1000,
            'tps': result.tokens_per_second_per_user,
            'primary_bottleneck': result.primary_bottleneck,
            'compute_util': result.avg_compute_utilization,
            'memory_bw_util': result.avg_memory_bw_utilization,
            'network_bw_util': result.avg_network_bw_utilization,
            'memory_util': result.avg_memory_utilization,
            'compute_used_pflops': (result.total_compute_attention + result.total_compute_ffn + result.total_compute_other) / result.total_decode_time / 1e15,
            'memory_bw_used_tbps': sum(step.memory_traffic / max(step.compute_time, step.memory_bw_time, step.network_time) 
                                       for step in result.step_details) / 1e12,
            'network_bw_used_gbps': network_bw_gbps,
            'memory_used_gb': (result.avg_memory_weights + result.avg_memory_kv_cache + result.avg_memory_activations) / 1e9
        }


def generate_test_function(config_name: str, config: Dict[str, Any], results: Dict[str, Any]) -> str:
    """Generate a pytest test function from config and results."""
    # Create test function name from config
    test_name = config_name.replace('.json', '').replace('-', '_').replace('llm_config_', 'test_config_')
    
    # Extract key parameters for test name
    model = config['model'].replace('-', '_')
    calc_type = config['calculation_type'].lower()
    parallelism = config['parallelism']['type'].lower()
    if parallelism != 'none':
        num_gpus = config['parallelism']['num_gpus']
        parallelism = f"{parallelism}_{num_gpus}gpu"
    else:
        parallelism = "no_parallel"
    
    test_name = f"test_{model}_{calc_type}_{parallelism}"
    
    # Build test docstring
    if results['type'] == 'TTFT':
        docstring = f'''    """
    Test {config['model']} TTFT calculation with {config['parallelism']['type']} parallelism.
    
    Configuration:
    - Batch Size: {config['workload']['batch_size']}
    - Sequence Length: {config['workload']['sequence_length']}
    - Memory: {config['system']['memory']} GB
    - Compute: {config['system']['compute']} TFLOPS
    - Memory BW: {config['system']['memory_bw']} GB/s
    - Network BW: {config['system']['network_bw']} GB/s
    
    Expected Results:
    - TTFT: {results['ttft_ms']:.2f} ms
    - Bottleneck: {results['bottleneck']}
    - Network BW Used: {results['network_bw_used_gbps']:.2f} GB/s
    """'''
    else:  # DECODE
        docstring = f'''    """
    Test {config['model']} Decode calculation with {config['parallelism']['type']} parallelism.
    
    Configuration:
    - Batch Size: {config['workload']['batch_size']}
    - Prefill Length: {config['workload']['prefill_length']}
    - Output Length: {config['workload']['output_length']}
    - Memory: {config['system']['memory']} GB
    - Compute: {config['system']['compute']} TFLOPS
    - Memory BW: {config['system']['memory_bw']} GB/s
    - Network BW: {config['system']['network_bw']} GB/s
    
    Expected Results:
    - Total Time: {results['total_time_ms']:.2f} ms
    - TPS: {results['tps']:.2f}
    - Primary Bottleneck: {results['primary_bottleneck']}
    - Network BW Used: {results['network_bw_used_gbps']:.2f} GB/s
    """'''
    
    # Build test function code
    lines = [
        f"def {test_name}():",
        docstring,
        f"    # Load model",
        f"    model = ALL_MODELS['{config['model']}']",
        f"    perf = InferencePerformance(model)",
        f"",
        f"    # System constraints",
        f"    gpu = SystemConstraints(",
        f"        memory_capacity={config['system']['memory']}e9,",
        f"        memory_bandwidth={config['system']['memory_bw']}e9,",
        f"        compute_throughput={config['system']['compute']}e12,",
        f"        network_bandwidth={config['system']['network_bw']}e9",
        f"    )",
        f"",
    ]
    
    # Add parallelism config if needed
    if config['parallelism']['type'] != 'None':
        num_gpus = config['parallelism']['num_gpus']
        if config['parallelism']['type'] == 'TENSOR_PARALLEL':
            tp_size, pp_size = num_gpus, 1
        else:
            tp_size, pp_size = 1, num_gpus
        
        lines.extend([
            f"    # Parallelism config",
            f"    parallel_config = ParallelismConfig(",
            f"        parallelism_type=ParallelismType.{config['parallelism']['type']},",
            f"        tensor_parallel_size={tp_size},",
            f"        pipeline_parallel_size={pp_size}",
            f"    )",
            f"",
        ])
    else:
        lines.append(f"    parallel_config = None\n")
    
    # Add calculation
    if results['type'] == 'TTFT':
        lines.extend([
            f"    # Run TTFT calculation",
            f"    result = perf.calculate_achievable_ttft(",
            f"        system_constraints=gpu,",
            f"        batch_size={config['workload']['batch_size']},",
            f"        sequence_length={config['workload']['sequence_length']},",
            f"        parallelism_config=parallel_config,",
            f"        kernel_launch_latency={config['system']['kernel_latency']}e-6",
        ])
        if config['system']['dtype']:
            lines.append(f"        dtype_override='{config['system']['dtype']}'")
        lines.extend([
            f"    )",
            f"",
            f"    # Assertions",
            f"    assert abs(result.achievable_ttft * 1000 - {results['ttft_ms']:.4f}) < 0.1, 'TTFT mismatch'",
            f"    assert result.bottleneck_resource == '{results['bottleneck']}', 'Bottleneck mismatch'",
            f"    assert abs(result.compute_utilization - {results['compute_util']:.6f}) < 0.01, 'Compute utilization mismatch'",
            f"    assert abs(result.network_bandwidth_utilization - {results['network_bw_util']:.6f}) < 0.01, 'Network utilization mismatch'",
            f"    assert abs(result.network_bandwidth_used / 1e9 - {results['network_bw_used_gbps']:.4f}) < 1.0, 'Network BW used mismatch'",
        ])
    else:  # DECODE
        lines.extend([
            f"    # Run Decode calculation",
            f"    result = perf.calculate_decode_performance(",
            f"        system_constraints=gpu,",
            f"        batch_size={config['workload']['batch_size']},",
            f"        prefill_length={config['workload']['prefill_length']},",
            f"        output_length={config['workload']['output_length']},",
            f"        parallelism_config=parallel_config,",
            f"        kernel_launch_latency={config['system']['kernel_latency']}e-6,",
            f"        return_step_details=True",
        ])
        if config['system']['dtype']:
            lines.append(f"        dtype_override='{config['system']['dtype']}'")
        lines.extend([
            f"    )",
            f"",
            f"    # Calculate network bandwidth from step details",
            f"    total_network_traffic = sum(step.network_traffic for step in result.step_details)",
            f"    total_bottleneck_time = sum(max(step.compute_time, step.memory_bw_time, step.network_time)",
            f"                                 for step in result.step_details)",
            f"    network_bw_gbps = (total_network_traffic / total_bottleneck_time) / 1e9 if total_bottleneck_time > 0 else 0.0",
            f"",
            f"    # Assertions",
            f"    assert abs(result.total_decode_time * 1000 - {results['total_time_ms']:.4f}) < 1.0, 'Total time mismatch'",
            f"    assert abs(result.tokens_per_second_per_user - {results['tps']:.2f}) < 10.0, 'TPS mismatch'",
            f"    assert result.primary_bottleneck == '{results['primary_bottleneck']}', 'Primary bottleneck mismatch'",
            f"    assert abs(result.avg_compute_utilization - {results['compute_util']:.6f}) < 0.01, 'Compute utilization mismatch'",
            f"    assert abs(result.avg_network_bw_utilization - {results['network_bw_util']:.6f}) < 0.01, 'Network utilization mismatch'",
            f"    assert abs(network_bw_gbps - {results['network_bw_used_gbps']:.4f}) < 1.0, 'Network BW used mismatch'",
        ])
    
    return '\n'.join(lines)


def generate_tests_from_logs(logs_dir: str = 'logs', output_file: str = 'test_from_configs.py'):
    """Generate test file from all configs in logs directory."""
    logs_path = Path(logs_dir)
    
    if not logs_path.exists():
        print(f"Logs directory '{logs_dir}' not found")
        return
    
    # Find all JSON config files
    config_files = list(logs_path.glob('llm-config-*.json'))
    
    if not config_files:
        print(f"No config files found in '{logs_dir}'")
        return
    
    print(f"Found {len(config_files)} config file(s)")
    
    # Generate test file header
    test_code = [
        '"""',
        'Unit tests generated from exported configuration files.',
        'Auto-generated by generate_tests_from_configs.py',
        '"""',
        '',
        'import pytest',
        'from llm_configs import ALL_MODELS',
        'from inference_performance import (',
        '    InferencePerformance, SystemConstraints, ParallelismConfig, ParallelismType',
        ')',
        '',
        ''
    ]
    
    # Process each config file
    for config_file in sorted(config_files):
        print(f"Processing: {config_file.name}")
        
        try:
            # Load config and run calculation
            config = load_config(str(config_file))
            results = run_config(config)
            
            # Generate test function
            test_func = generate_test_function(config_file.name, config, results)
            test_code.append(test_func)
            test_code.append('')
            test_code.append('')
            
            print(f"  ✓ Generated test for {config['model']} {config['calculation_type']}")
            
        except Exception as e:
            print(f"  ✗ Error processing {config_file.name}: {e}")
            continue
    
    # Write test file
    with open(output_file, 'w') as f:
        f.write('\n'.join(test_code))
    
    print(f"\nGenerated {output_file} with {len(config_files)} test(s)")
    print(f"Run with: pytest {output_file} -v")


if __name__ == '__main__':
    import sys
    
    logs_dir = sys.argv[1] if len(sys.argv) > 1 else 'logs'
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'test_from_configs.py'
    
    generate_tests_from_logs(logs_dir, output_file)
