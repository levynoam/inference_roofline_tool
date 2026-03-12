"""
Offline Analysis Script for Exported Configurations
Load a configuration JSON exported from the web app and run detailed analysis.
"""

import json
import sys
from llm_configs import ALL_MODELS
from inference_performance import (
    InferencePerformance,
    SystemConstraints,
    ParallelismConfig,
    ParallelismType
)


def load_config(filename):
    """Load configuration from JSON file"""
    with open(filename, 'r') as f:
        return json.load(f)


def analyze_config(config_file):
    """Run detailed analysis on exported configuration"""
    print(f"\n{'='*80}")
    print(f"ANALYZING CONFIGURATION: {config_file}")
    print(f"{'='*80}\n")
    
    # Load configuration
    config = load_config(config_file)
    
    print(f"Timestamp: {config['timestamp']}")
    print(f"Model: {config['model']}")
    print(f"Calculation Type: {config['calculation_type']}")
    
    # Get model
    model_name = config['model']
    model = ALL_MODELS[model_name]
    
    # Build system constraints
    gpu = SystemConstraints(
        memory_capacity=config['system']['memory'] * 1e9,
        memory_bandwidth=config['system']['memory_bw'] * 1e9,
        compute_throughput=config['system']['compute'] * 1e12,
        network_bandwidth=config['system']['network_bw'] * 1e9
    )
    
    print(f"\nSystem Constraints:")
    print(f"  Memory: {config['system']['memory']} GB")
    print(f"  Memory BW: {config['system']['memory_bw']} GB/s")
    print(f"  Compute: {config['system']['compute']} TFLOPS")
    print(f"  Network BW: {config['system']['network_bw']} GB/s")
    print(f"  Kernel Latency: {config['system']['kernel_latency']} µs")
    print(f"  Data Type Override: {config['system']['dtype'] or 'Model Default'}")
    
    # Build parallelism config
    parallel_config = None
    if config['parallelism']['type'] != 'None':
        num_gpus = config['parallelism']['num_gpus']
        # For tensor parallel, pp_size=1; for pipeline parallel, tp_size=1
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
        print(f"\nParallelism:")
        print(f"  Type: {config['parallelism']['type']}")
        print(f"  TP Size: {tp_size}")
        print(f"  PP Size: {pp_size}")
        print(f"  Total GPUs: {parallel_config.total_gpus}")
    
    # Get workload parameters
    batch_size = config['workload']['batch_size']
    kernel_latency = config['system']['kernel_latency'] * 1e-6
    dtype_override = config['system']['dtype'] if config['system']['dtype'] else None
    
    print(f"\nWorkload:")
    print(f"  Batch Size: {batch_size}")
    
    # Create performance analyzer
    perf = InferencePerformance(model)
    
    # Run calculation based on type
    if config['calculation_type'] == 'TTFT':
        sequence_length = config['workload']['sequence_length']
        print(f"  Sequence Length: {sequence_length}")
        
        result = perf.calculate_achievable_ttft(
            system_constraints=gpu,
            batch_size=batch_size,
            sequence_length=sequence_length,
            parallelism_config=parallel_config,
            kernel_launch_latency=kernel_latency,
            dtype_override=dtype_override
        )
        
        print(f"\n{'-'*80}")
        print("TTFT RESULTS:")
        print(f"{'-'*80}")
        print(f"TTFT: {result.achievable_ttft * 1000:.2f} ms")
        print(f"Bottleneck: {result.bottleneck_resource}")
        
        print(f"\nResource Utilization:")
        print(f"  Compute: {result.compute_utilization * 100:.1f}%")
        print(f"  Memory BW: {result.memory_bandwidth_utilization * 100:.1f}%")
        print(f"  Network BW: {result.network_bandwidth_utilization * 100:.1f}%")
        print(f"  Memory Capacity: {result.memory_utilization * 100:.1f}%")
        
        print(f"\nActual Resource Usage:")
        print(f"  Compute: {result.compute_used / 1e15:.2f} PFLOPS")
        print(f"  Memory BW: {result.memory_bandwidth_used / 1e12:.2f} TB/s")
        print(f"  Network BW: {result.network_bandwidth_used / 1e12:.4f} TB/s ({result.network_bandwidth_used / 1e9:.2f} GB/s)")
        print(f"  Memory Used: {result.memory_used / 1e9:.2f} GB")
        
        print(f"\nMemory Breakdown:")
        print(f"  Weights: {result.memory_weights / 1e9:.2f} GB")
        print(f"  KV Cache: {result.memory_kv_cache / 1e9:.2f} GB")
        print(f"  Activations: {result.memory_activations / 1e9:.2f} GB")
        
    else:  # DECODE
        prefill_length = config['workload']['prefill_length']
        output_length = config['workload']['output_length']
        print(f"  Prefill Length: {prefill_length}")
        print(f"  Output Length: {output_length}")
        
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
        
        # Debug: Check first step details
        if result.step_details:
            first_step = result.step_details[0]
            print(f"\n{'-'*80}")
            print("DEBUG: First Step Details")
            print(f"{'-'*80}")
            print(f"  Context Length: {first_step.context_length}")
            print(f"  Network Traffic: {first_step.network_traffic / 1e6:.4f} MB")
            print(f"  Network Time: {first_step.network_time * 1000:.6f} ms")
            print(f"  Compute Time: {first_step.compute_time * 1000:.6f} ms")
            print(f"  Memory BW Time: {first_step.memory_bw_time * 1000:.6f} ms")
            print(f"  Step Time: {first_step.step_time * 1000:.6f} ms")
            print(f"  Parallelism Config: {parallel_config}")
            if parallel_config:
                print(f"    TP Size: {parallel_config.tensor_parallel_size}")
                print(f"    PP Size: {parallel_config.pipeline_parallel_size}")
        
        print(f"\n{'-'*80}")
        print("DECODE RESULTS:")
        print(f"{'-'*80}")
        print(f"Total Decode Time: {result.total_decode_time * 1000:.2f} ms")
        print(f"Avg Step Time: {result.avg_step_time * 1000:.4f} ms")
        print(f"Tokens per Second: {result.tokens_per_second_per_user:.2f}")
        print(f"Primary Bottleneck: {result.primary_bottleneck}")
        
        print(f"\nAverage Resource Utilization:")
        print(f"  Compute: {result.avg_compute_utilization * 100:.1f}%")
        print(f"  Memory BW: {result.avg_memory_bw_utilization * 100:.1f}%")
        print(f"  Network BW: {result.avg_network_bw_utilization * 100:.1f}%")
        print(f"  Memory Capacity: {result.avg_memory_utilization * 100:.1f}%")
        
        # Calculate network bandwidth the same way as web app
        total_network_traffic = sum(step.network_traffic for step in result.step_details)
        total_bottleneck_time = sum(max(step.compute_time, step.memory_bw_time, step.network_time) 
                                     for step in result.step_details)
        avg_network_bw_required = (total_network_traffic / total_bottleneck_time) if total_bottleneck_time > 0 else 0.0
        
        print(f"\nActual Resource Usage:")
        print(f"  Compute: {(result.avg_compute_utilization * result.compute_throughput) / 1e15:.2f} PFLOPS")
        print(f"  Memory BW: {(result.avg_memory_bw_utilization * result.memory_bandwidth) / 1e12:.2f} TB/s")
        print(f"  Network BW (calculated): {avg_network_bw_required / 1e12:.4f} TB/s ({avg_network_bw_required / 1e9:.2f} GB/s)")
        print(f"  Memory Used: {(result.avg_memory_utilization * result.memory_capacity) / 1e9:.2f} GB")
        
        print(f"\nNetwork Traffic Debug:")
        print(f"  Total Network Traffic: {total_network_traffic / 1e9:.4f} GB")
        print(f"  Total Decode Time: {result.total_decode_time * 1000:.4f} ms")
        print(f"  Total Bottleneck Time: {total_bottleneck_time * 1000:.4f} ms")
        print(f"  Network BW (traffic/total_time): {(total_network_traffic / result.total_decode_time) / 1e9:.2f} GB/s")
        print(f"  Network BW (traffic/bottleneck_time): {avg_network_bw_required / 1e9:.2f} GB/s")
        print(f"  System Network BW Limit: {result.network_bandwidth / 1e9:.2f} GB/s")
        
        print(f"\nMemory Breakdown (Average):")
        print(f"  Weights: {result.avg_memory_weights / 1e9:.2f} GB")
        print(f"  KV Cache: {result.avg_memory_kv_cache / 1e9:.2f} GB")
        print(f"  Activations: {result.avg_memory_activations / 1e9:.2f} GB")
        
        print(f"\nBottleneck Breakdown:")
        for resource, count in result.bottleneck_breakdown.items():
            pct = (count / output_length) * 100
            print(f"  {resource}: {count}/{output_length} steps ({pct:.1f}%)")
        
        # Show first few steps in detail
        print(f"\nFirst 5 Steps Detail:")
        for i, step in enumerate(result.step_details[:5]):
            print(f"\n  Step {step.step}:")
            print(f"    Context Length: {step.context_length}")
            print(f"    Step Time: {step.step_time * 1000:.4f} ms")
            print(f"    Compute Time: {step.compute_time * 1000:.4f} ms")
            print(f"    Memory BW Time: {step.memory_bw_time * 1000:.4f} ms")
            print(f"    Network Time: {step.network_time * 1000:.4f} ms")
            print(f"    Network Traffic: {step.network_traffic / 1e6:.4f} MB")
            print(f"    Bottleneck: {step.bottleneck}")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_config.py <config_file.json>")
        print("\nExample: python analyze_config.py llm-config-decode-2026-01-28T12-30-00.json")
        sys.exit(1)
    
    config_file = sys.argv[1]
    analyze_config(config_file)
