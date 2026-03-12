"""
LLM Inference Performance Library
Calculate resource requirements for inference operations
"""

from dataclasses import dataclass, field
from typing import Literal, Optional, Dict, Tuple, List
from enum import Enum
from llm_architecture import (
    LLMArchitecture, HybridLayerType, LayerAttentionType, FFNLayerType
)


class ParallelismType(Enum):
    """Types of model parallelism"""
    NONE = "none"  # Single GPU
    DATA_PARALLEL = "data_parallel"  # DP - replicate model across GPUs
    TENSOR_PARALLEL = "tensor_parallel"  # TP - split layers horizontally
    PIPELINE_PARALLEL = "pipeline_parallel"  # PP - split layers vertically
    TENSOR_PIPELINE = "tensor_pipeline"  # TP + PP
    TENSOR_DATA = "tensor_data"  # TP + DP
    FULL_3D = "full_3d"  # TP + PP + DP


@dataclass
class ParallelismConfig:
    """Configuration for parallelism strategy"""
    parallelism_type: ParallelismType = ParallelismType.NONE
    tensor_parallel_size: int = 1  # Number of GPUs for tensor parallelism
    pipeline_parallel_size: int = 1  # Number of pipeline stages
    data_parallel_size: int = 1  # Number of data parallel replicas
    
    def __post_init__(self):
        """Validate configuration"""
        total_gpus = self.tensor_parallel_size * self.pipeline_parallel_size * self.data_parallel_size
        
        # Validate parallelism type matches configuration
        if self.parallelism_type == ParallelismType.NONE:
            assert total_gpus == 1, "NONE parallelism requires exactly 1 GPU"
        elif self.parallelism_type == ParallelismType.DATA_PARALLEL:
            assert self.tensor_parallel_size == 1 and self.pipeline_parallel_size == 1
        elif self.parallelism_type == ParallelismType.TENSOR_PARALLEL:
            assert self.pipeline_parallel_size == 1 and self.data_parallel_size == 1
        elif self.parallelism_type == ParallelismType.PIPELINE_PARALLEL:
            assert self.tensor_parallel_size == 1 and self.data_parallel_size == 1
        elif self.parallelism_type == ParallelismType.TENSOR_PIPELINE:
            assert self.data_parallel_size == 1
        elif self.parallelism_type == ParallelismType.TENSOR_DATA:
            assert self.pipeline_parallel_size == 1
    
    @property
    def total_gpus(self) -> int:
        """Total number of GPUs"""
        return self.tensor_parallel_size * self.pipeline_parallel_size * self.data_parallel_size


@dataclass
class PrefillResources:
    """Resource requirements for prefill phase"""
    # Memory (bytes)
    memory_per_gpu: float  # Total memory required per GPU
    memory_model_weights: float  # Memory for model weights per GPU
    memory_kv_cache: float  # Memory for KV cache per GPU
    memory_activations: float  # Memory for activations per GPU
    
    # Bandwidth (bytes/sec)
    memory_bandwidth_per_gpu: float  # Memory bandwidth required per GPU
    network_bandwidth_per_gpu: float  # Network bandwidth required per GPU
    
    # Compute (FLOPs)
    compute_per_gpu: float  # Total compute operations per GPU
    compute_flops_per_sec: float  # FLOPs/sec required per GPU
    
    # Time (seconds)
    time_to_first_token: float  # Target TTFT
    
    # Kernel launch overhead
    num_kernel_launches: int  # Total number of kernel launches
    kernel_launch_overhead: float  # Total kernel launch overhead in seconds
    effective_compute_time: float  # Time available for actual compute (TTFT - overhead)
    
    # Additional metrics
    arithmetic_intensity: float  # FLOPs per byte (compute/memory ratio)
    compute_bound: bool  # True if compute-bound, False if memory-bound
    
    def summary(self) -> str:
        """Human-readable summary"""
        lines = [
            "=== Prefill Resource Requirements ===",
            f"Time to First Token: {self.time_to_first_token * 1000:.2f} ms",
            f"  Kernel Launch Overhead: {self.kernel_launch_overhead * 1000:.2f} ms ({self.num_kernel_launches} launches)",
            f"  Effective Compute Time: {self.effective_compute_time * 1000:.2f} ms",
            "",
            "Memory per GPU:",
            f"  Model Weights:  {self.memory_model_weights / (1024**3):8.2f} GB",
            f"  KV Cache:       {self.memory_kv_cache / (1024**3):8.2f} GB",
            f"  Activations:    {self.memory_activations / (1024**3):8.2f} GB",
            f"  Total:          {self.memory_per_gpu / (1024**3):8.2f} GB",
            "",
            "Bandwidth per GPU:",
            f"  Memory BW:      {self.memory_bandwidth_per_gpu / (1024**3):8.2f} GB/s",
            f"  Network BW:     {self.network_bandwidth_per_gpu / (1024**3):8.2f} GB/s",
            "",
            "Compute per GPU:",
            f"  Total FLOPs:    {self.compute_per_gpu / 1e12:8.2f} TFLOPs",
            f"  FLOPs/sec:      {self.compute_flops_per_sec / 1e12:8.2f} TFLOP/s",
            "",
            "Performance Characteristics:",
            f"  Arithmetic Intensity: {self.arithmetic_intensity:.2f} FLOPs/byte",
            f"  Bottleneck: {'Compute' if self.compute_bound else 'Memory'}",
        ]
        return "\n".join(lines)


@dataclass
class SystemConstraints:
    """Hardware system constraints per GPU"""
    memory_capacity: float  # Total memory capacity in bytes
    memory_bandwidth: float  # Memory bandwidth in bytes/sec
    compute_throughput: float  # Compute throughput in FLOPs/sec
    network_bandwidth: float  # Network bandwidth in bytes/sec (for multi-GPU)
    persistent_storage_bandwidth: float = 20 * (1024**3)  # Persistent storage BW in bytes/sec (default 20 GB/s)
    
    @classmethod
    def from_gpu_spec(cls, gpu_name: str):
        """Create SystemConstraints from common GPU specifications"""
        specs = {
            "A100-40GB": cls(
                memory_capacity=40 * (1024**3),
                memory_bandwidth=1555 * (1024**3),  # 1.5 TB/s
                compute_throughput=312e12,  # 312 TFLOP/s (FP16 with sparsity)
                network_bandwidth=600 * (1024**3),  # 600 GB/s (NVLink)
            ),
            "A100-80GB": cls(
                memory_capacity=80 * (1024**3),
                memory_bandwidth=2000 * (1024**3),  # 2 TB/s
                compute_throughput=312e12,  # 312 TFLOP/s (FP16)
                network_bandwidth=600 * (1024**3),  # 600 GB/s (NVLink)
            ),
            "H100-80GB": cls(
                memory_capacity=80 * (1024**3),
                memory_bandwidth=3350 * (1024**3),  # 3.35 TB/s (HBM3)
                compute_throughput=1979e12,  # 1979 TFLOP/s (FP16)
                network_bandwidth=900 * (1024**3),  # 900 GB/s (NVLink 4.0)
            ),
            "MI300X": cls(
                memory_capacity=192 * (1024**3),
                memory_bandwidth=5300 * (1024**3),  # 5.3 TB/s
                compute_throughput=1300e12,  # 1.3 PFLOP/s (FP16)
                network_bandwidth=896 * (1024**3),  # 896 GB/s (Infinity Fabric)
            ),
        }
        
        if gpu_name not in specs:
            available = ", ".join(specs.keys())
            raise ValueError(f"Unknown GPU '{gpu_name}'. Available: {available}")
        
        return specs[gpu_name]


@dataclass
class ResourceUtilization:
    """Resource utilization metrics for a workload"""
    # TTFT results
    achievable_ttft: float  # Achievable time to first token in seconds
    bottleneck_resource: str  # Which resource limits performance
    
    # Resource utilization (0-1, where 1.0 = 100%)
    memory_utilization: float  # Memory capacity used
    memory_bandwidth_utilization: float  # Memory bandwidth used
    compute_utilization: float  # Compute throughput used
    network_bandwidth_utilization: float  # Network bandwidth used
    
    # Actual resource consumption
    memory_used: float  # Memory used in bytes
    memory_bandwidth_used: float  # Memory BW used in bytes/sec
    compute_used: float  # Compute used in FLOPs/sec
    network_bandwidth_used: float  # Network BW used in bytes/sec
    
    # System constraints (for reference)
    memory_available: float  # Available memory in bytes
    memory_bandwidth_available: float  # Available memory BW in bytes/sec
    compute_available: float  # Available compute in FLOPs/sec
    network_bandwidth_available: float  # Available network BW in bytes/sec
    
    # Memory breakdown
    memory_weights: float  # Memory used for model weights in bytes
    memory_kv_cache: float  # Memory used for KV cache in bytes
    memory_activations: float  # Memory used for activations in bytes
    
    # Compute breakdown  
    compute_attention: float  # Compute used for attention in FLOPs
    compute_ffn: float  # Compute used for FFN in FLOPs
    compute_other: float  # Compute used for other ops (layernorm, lm_head) in FLOPs
    
    # Additional details
    kernel_launch_overhead: float  # Kernel launch overhead in seconds
    effective_compute_time: float  # Time available for actual compute
    
    # Time breakdown (in seconds)
    time_compute_busy: float  # Time spent on actual computation (bottleneck time)
    time_kernel_launch: float  # Time spent waiting on kernel launches
    time_idle: float  # Time spent idle (should be 0 or very small)
    
    # Persistent storage (with defaults, must be at end for dataclass)
    persistent_storage_bandwidth_utilization: float = 0.0  # Persistent storage bandwidth used (for MoE offloading)
    persistent_storage_bandwidth_used: float = 0.0  # Persistent storage BW used in bytes/sec
    persistent_storage_bandwidth_available: float = 20 * (1024**3)  # Available persistent storage BW in bytes/sec
    
    def summary(self) -> str:
        """Human-readable summary"""
        lines = [
            "=== System Performance Analysis ===",
            f"Achievable TTFT: {self.achievable_ttft * 1000:.2f} ms",
            f"  Kernel Overhead: {self.kernel_launch_overhead * 1000:.2f} ms",
            f"  Effective Compute: {self.effective_compute_time * 1000:.2f} ms",
            f"Bottleneck: {self.bottleneck_resource}",
            "",
            "Resource Utilization:",
            f"  Memory:          {self.memory_utilization * 100:6.1f}% "
            f"({self.memory_used / (1024**3):6.2f} GB / {self.memory_available / (1024**3):6.2f} GB)",
            f"  Memory BW:       {self.memory_bandwidth_utilization * 100:6.1f}% "
            f"({self.memory_bandwidth_used / (1024**3):6.2f} / {self.memory_bandwidth_available / (1024**3):6.2f} GB/s)",
            f"  Compute:         {self.compute_utilization * 100:6.1f}% "
            f"({self.compute_used / 1e12:6.2f} / {self.compute_available / 1e12:6.2f} TFLOP/s)",
            f"  Network BW:      {self.network_bandwidth_utilization * 100:6.1f}% "
            f"({self.network_bandwidth_used / (1024**3):6.2f} / {self.network_bandwidth_available / (1024**3):6.2f} GB/s)",
        ]
        # Add persistent storage line if it's being used
        if self.persistent_storage_bandwidth_utilization > 0:
            lines.append(
                f"  Storage BW:      {self.persistent_storage_bandwidth_utilization * 100:6.1f}% "
                f"({self.persistent_storage_bandwidth_used / (1024**3):6.2f} / {self.persistent_storage_bandwidth_available / (1024**3):6.2f} GB/s)"
            )
        return "\n".join(lines)


@dataclass
class DecodeStepResources:
    """Resources for a single decode step"""
    step: int  # Which step in the sequence (0-indexed)
    context_length: int  # Total context length at this step (prefill + generated so far)
    
    # Time
    step_time: float  # Time for this step in seconds
    compute_time: float  # Time constrained by compute
    memory_bw_time: float  # Time constrained by memory bandwidth
    network_time: float  # Time constrained by network
    kernel_overhead: float  # Kernel launch overhead
    
    # Resources
    compute_flops: float  # FLOPs for this step
    memory_traffic: float  # Memory traffic in bytes
    network_traffic: float  # Network traffic in bytes
    
    # Bottleneck
    bottleneck: str  # Which resource limits this step
    
    # Persistent storage (with defaults, must be at end for dataclass)
    storage_bw_time: float = 0.0  # Time constrained by persistent storage bandwidth (for MoE offloading)
    storage_traffic: float = 0.0  # Persistent storage traffic in bytes (for MoE offloading)


@dataclass
class DecodePerformance:
    """Performance metrics for decode phase"""
    # Configuration
    batch_size: int
    prefill_length: int  # Input prompt length
    output_length: int  # Number of tokens generated
    total_sequence_length: int  # prefill_length + output_length
    
    # Timing
    total_decode_time: float  # Total time for all decode steps (seconds)
    avg_step_time: float  # Average time per step
    min_step_time: float  # Fastest step
    max_step_time: float  # Slowest step
    
    # Throughput metrics
    tokens_per_second_per_user: float  # TPS per user (output_length / total_time)
    total_throughput: float  # Total tokens/sec (batch_size * TPS)
    
    # Resource utilization (averaged across all steps)
    avg_memory_utilization: float  # Average memory utilization (0-1)
    avg_memory_bw_utilization: float  # Average memory BW utilization (0-1)
    avg_compute_utilization: float  # Average compute utilization (0-1)
    avg_network_bw_utilization: float  # Average network BW utilization (0-1)
    
    # Memory breakdown (averaged across all steps)
    avg_memory_weights: float  # Average memory used for weights in bytes
    avg_memory_kv_cache: float  # Average memory used for KV cache in bytes
    avg_memory_activations: float  # Average memory used for activations in bytes
    
    # Compute breakdown (total across all steps)
    total_compute_attention: float  # Total compute for attention in FLOPs
    total_compute_ffn: float  # Total compute for FFN in FLOPs
    total_compute_other: float  # Total compute for other ops in FLOPs
    
    # Time breakdown (total across all steps, in seconds)
    total_time_compute_busy: float  # Total time spent on actual computation
    total_time_kernel_launch: float  # Total time spent waiting on kernel launches
    total_time_idle: float  # Total idle time (should be 0 or very small)
    
    # Bottleneck analysis
    bottleneck_breakdown: Dict[str, int]  # Count of steps bottlenecked by each resource
    primary_bottleneck: str  # Most common bottleneck
    
    # Per-step details (optional, for analysis)
    step_details: list[DecodeStepResources]  # List of per-step resources
    
    # System constraints (for reference)
    memory_capacity: float
    memory_bandwidth: float
    compute_throughput: float
    network_bandwidth: float
    
    # Persistent storage (with defaults, must be at end for dataclass)
    avg_storage_bw_utilization: float = 0.0  # Average persistent storage BW utilization (0-1)
    persistent_storage_bandwidth: float = 20 * (1024**3)  # Persistent storage BW in bytes/sec
    
    def summary(self) -> str:
        """Human-readable summary"""
        lines = [
            "=== Decode Performance Analysis ===",
            f"Workload:",
            f"  Batch Size:        {self.batch_size}",
            f"  Prefill Length:    {self.prefill_length} tokens",
            f"  Output Length:     {self.output_length} tokens",
            f"  Total Length:      {self.total_sequence_length} tokens",
            "",
            f"Timing:",
            f"  Total Decode Time: {self.total_decode_time * 1000:.2f} ms",
            f"  Avg Step Time:     {self.avg_step_time * 1000:.4f} ms",
            f"  Min Step Time:     {self.min_step_time * 1000:.4f} ms",
            f"  Max Step Time:     {self.max_step_time * 1000:.4f} ms",
            "",
            f"Throughput:",
            f"  TPS per User:      {self.tokens_per_second_per_user:.2f} tokens/sec",
            f"  Total Throughput:  {self.total_throughput:.2f} tokens/sec",
            "",
            f"Average Resource Utilization:",
            f"  Memory:            {self.avg_memory_utilization * 100:6.1f}%",
            f"  Memory BW:         {self.avg_memory_bw_utilization * 100:6.1f}%",
            f"  Compute:           {self.avg_compute_utilization * 100:6.1f}%",
            f"  Network BW:        {self.avg_network_bw_utilization * 100:6.1f}%",
            "",
            f"Bottleneck Analysis:",
        ]
        
        for resource, count in sorted(self.bottleneck_breakdown.items(), key=lambda x: x[1], reverse=True):
            pct = (count / self.output_length) * 100
            lines.append(f"  {resource:15s}: {count:4d} steps ({pct:5.1f}%)")
        
        lines.append(f"  Primary:        {self.primary_bottleneck}")
        
        return "\n".join(lines)


@dataclass
class PerLayerBreakdown:
    """Per-layer breakdown of compute, memory traffic (bandwidth), and kernel launches.
    
    Each list has length num_layers, indexed by global layer index.
    Values represent per-GPU quantities (divided by TP; PP is implicit since
    different layers reside on different GPUs in a pipeline).
    
    Three lines per graph:
    - attention: sequence-mixing component (attention projections, attention scores,
      KV cache / linear state / Mamba state traffic)
    - non_attention: FFN component (dense or MoE weights, router)
    - sum: attention + non_attention (computed by consumer)
    """
    mode: str  # 'prefill' or 'decode'
    num_layers: int
    sequence_length: int  # The sequence length (prefill) or context length (decode) used
    batch_size: int
    
    # Per-layer compute (FLOPs per GPU)
    attention_compute: List[float] = field(default_factory=list)
    non_attention_compute: List[float] = field(default_factory=list)
    
    # Per-layer memory traffic / bandwidth (bytes per GPU)
    attention_memory_traffic: List[float] = field(default_factory=list)
    non_attention_memory_traffic: List[float] = field(default_factory=list)
    
    # Per-layer kernel launches (count per GPU) 
    attention_kernels: List[int] = field(default_factory=list)
    non_attention_kernels: List[int] = field(default_factory=list)
    
    # Layer descriptions for display
    layer_types: List[str] = field(default_factory=list)


class InferencePerformance:
    """Calculate inference performance metrics"""
    
    def __init__(self, model: LLMArchitecture):
        """
        Initialize with LLM architecture
        
        Args:
            model: LLMArchitecture instance defining the model
        """
        self.model = model
    
    def _get_bytes_per_param(self, dtype_override: Optional[str] = None) -> float:
        """Get bytes per parameter based on dtype.
        
        Args:
            dtype_override: Optional dtype to override model's dtype.
                          Supported: 'int4', 'int8', 'float16', 'bfloat16', 'float32'
        
        Returns:
            Bytes per parameter (0.5 for int4, 1 for int8, 2 for fp16, 4 for fp32)
        """
        dtype = dtype_override if dtype_override else self.model.dtype
        
        if dtype in ["int4"]:
            return 0.5
        elif dtype in ["int8"]:
            return 1.0
        elif dtype in ["float16", "bfloat16"]:
            return 2.0
        elif dtype in ["float32"]:
            return 4.0
        else:
            return 2.0  # Default to fp16
    
    def calculate_per_layer_breakdown(
        self,
        mode: str,
        batch_size: int,
        sequence_length: int,
        parallelism_config: ParallelismConfig,
        dtype_override: Optional[str] = None
    ) -> PerLayerBreakdown:
        """
        Calculate per-layer breakdown of compute, memory traffic, and kernel launches.
        
        Args:
            mode: 'prefill' or 'decode'
            batch_size: Number of sequences in batch
            sequence_length: For prefill: input sequence length.
                           For decode: context length at which to compute.
            parallelism_config: Parallelism configuration
            dtype_override: Optional dtype override
            
        Returns:
            PerLayerBreakdown with per-layer arrays for attention and non-attention
        """
        model = self.model
        bytes_per_param = self._get_bytes_per_param(dtype_override)
        effective_batch = batch_size // parallelism_config.data_parallel_size
        tp = parallelism_config.tensor_parallel_size
        H = model.hidden_dim
        L = sequence_length
        is_prefill = (mode == 'prefill')
        seq_len_compute = L if is_prefill else 1  # prefill processes L tokens, decode processes 1
        
        result = PerLayerBreakdown(
            mode=mode,
            num_layers=model.num_layers,
            sequence_length=sequence_length,
            batch_size=batch_size,
        )
        
        for layer_idx in range(model.num_layers):
            # ====== Determine layer types ======
            is_mamba_layer = False
            if model.hybrid_layer_types is not None:
                is_mamba_layer = model.hybrid_layer_types[layer_idx] == HybridLayerType.MAMBA
            
            layer_attn_type = model.get_layer_attention_type(layer_idx)
            
            is_moe_layer = False
            if model.ffn_layer_types is not None:
                is_moe_layer = model.ffn_layer_types[layer_idx] == FFNLayerType.MOE
            elif model.is_moe:
                is_moe_layer = True
            
            # Build layer label
            if is_mamba_layer:
                attn_label = "Mamba"
            elif layer_attn_type == LayerAttentionType.LINEAR_ATTENTION:
                attn_label = "Linear"
            elif layer_attn_type == LayerAttentionType.SLIDING_ATTENTION:
                attn_label = "Sliding"
            else:
                attn_label = "Full"
            ffn_label = "MoE" if is_moe_layer else "Dense"
            result.layer_types.append(f"{attn_label}/{ffn_label}")
            
            # ====== COMPUTE ======
            layer_attn_flops = 0.0
            layer_ffn_flops = 0.0
            
            # --- Attention / sequence-mixing compute ---
            if is_mamba_layer and model.mamba_config is not None:
                if is_prefill:
                    layer_attn_flops = model.mamba_config.get_prefill_flops(L, H) * effective_batch
                else:
                    layer_attn_flops = model.mamba_config.get_decode_flops(H) * effective_batch
            elif layer_attn_type == LayerAttentionType.LINEAR_ATTENTION and model.linear_attention_config is not None:
                lac = model.linear_attention_config
                num_attn_heads = model.attention_config.num_attention_heads
                if is_prefill:
                    layer_attn_flops = lac.get_prefill_flops(L, effective_batch, num_attn_heads, H)
                else:
                    layer_attn_flops = lac.get_decode_flops(effective_batch, num_attn_heads, H)
            else:
                # Standard / sliding attention
                num_attn_heads = model.attention_config.num_attention_heads
                num_kv_heads = model.attention_config.num_key_value_heads
                head_dim = model.attention_config.head_dim
                kv_dim = num_kv_heads * head_dim
                
                effective_attn_len = L
                if layer_attn_type == LayerAttentionType.SLIDING_ATTENTION:
                    window = model.attention_config.sliding_window_size or L
                    effective_attn_len = min(L, window)
                
                if is_prefill:
                    # Prefill attention
                    if model.attention_config.use_mla and model.attention_config.mla_kv_lora_rank:
                        kv_lora_rank = model.attention_config.mla_kv_lora_rank
                        layer_attn_flops += 2 * effective_batch * L * H * kv_lora_rank  # K
                        layer_attn_flops += 2 * effective_batch * L * H * kv_lora_rank  # V
                        if model.attention_config.mla_q_lora_rank:
                            q_lora_rank = model.attention_config.mla_q_lora_rank
                            layer_attn_flops += 2 * effective_batch * L * H * q_lora_rank
                            layer_attn_flops += 2 * effective_batch * L * q_lora_rank * (num_attn_heads * head_dim)
                        else:
                            layer_attn_flops += 2 * effective_batch * L * H * (num_attn_heads * head_dim)
                        layer_attn_flops += 2 * effective_batch * L * kv_lora_rank * (num_kv_heads * head_dim) * 2
                        layer_attn_flops += 2 * effective_batch * num_attn_heads * L * effective_attn_len * head_dim
                        layer_attn_flops += 2 * effective_batch * num_attn_heads * L * effective_attn_len * head_dim
                        layer_attn_flops += 2 * effective_batch * L * (num_attn_heads * head_dim) * H
                    else:
                        layer_attn_flops += 2 * effective_batch * L * H * (num_attn_heads * head_dim)
                        layer_attn_flops += 2 * 2 * effective_batch * L * H * (num_kv_heads * head_dim)
                        layer_attn_flops += 2 * effective_batch * num_attn_heads * L * effective_attn_len * head_dim
                        layer_attn_flops += 2 * effective_batch * num_attn_heads * L * effective_attn_len * head_dim
                        layer_attn_flops += 2 * effective_batch * L * (num_attn_heads * head_dim) * H
                else:
                    # Decode attention
                    q_flops = 2 * effective_batch * H * H
                    if not model.attention_config.use_mla:
                        k_flops = 2 * effective_batch * H * kv_dim
                        v_flops = 2 * effective_batch * H * kv_dim
                    else:
                        k_flops = 2 * effective_batch * H * model.attention_config.mla_kv_lora_rank
                        v_flops = 2 * effective_batch * H * model.attention_config.mla_kv_lora_rank
                        decompress_k = 2 * effective_batch * effective_attn_len * model.attention_config.mla_kv_lora_rank * kv_dim
                        decompress_v = 2 * effective_batch * effective_attn_len * model.attention_config.mla_kv_lora_rank * kv_dim
                        k_flops += decompress_k
                        v_flops += decompress_v
                    
                    if model.attention_config.use_dsa:
                        pseudo_attn = 2 * effective_batch * effective_attn_len * model.attention_config.dsa_q_indexer_dim * model.attention_config.dsa_k_indexer_dim
                        dsa_ctx = min(effective_attn_len, model.attention_config.dsa_top_k)
                        attn_scores = 2 * effective_batch * num_attn_heads * dsa_ctx * head_dim + pseudo_attn
                        attn_output = 2 * effective_batch * num_attn_heads * dsa_ctx * head_dim
                    else:
                        attn_scores = 2 * effective_batch * num_attn_heads * effective_attn_len * head_dim
                        attn_output = 2 * effective_batch * num_attn_heads * effective_attn_len * head_dim
                    
                    out_proj = 2 * effective_batch * H * H
                    layer_attn_flops = q_flops + k_flops + v_flops + attn_scores + attn_output + out_proj
            
            # --- FFN compute ---
            if is_moe_layer and model.moe_config is not None:
                num_active = model.moe_config.num_experts_per_token
                intermediate = model.ffn_config.intermediate_size
                router = 2 * effective_batch * seq_len_compute * H * model.moe_config.num_experts
                up = 2 * effective_batch * seq_len_compute * num_active * H * intermediate
                down = 2 * effective_batch * seq_len_compute * num_active * intermediate * H
                layer_ffn_flops = router + up + down
                if is_prefill and model.ffn_config.use_gating:
                    # Note: existing decode breakdown does not include gating for MoE;
                    # prefill breakdown does. We match both for consistency.
                    layer_ffn_flops += 2 * effective_batch * seq_len_compute * num_active * H * intermediate
            else:
                intermediate = model.get_dense_intermediate_size()
                up = 2 * effective_batch * seq_len_compute * H * intermediate
                down = 2 * effective_batch * seq_len_compute * intermediate * H
                layer_ffn_flops = up + down
                if model.ffn_config.use_gating:
                    layer_ffn_flops += 2 * effective_batch * seq_len_compute * H * intermediate
            
            # Divide by TP (computation is split across TP GPUs)
            result.attention_compute.append(layer_attn_flops / tp)
            result.non_attention_compute.append(layer_ffn_flops / tp)
            
            # ====== MEMORY TRAFFIC (bandwidth) ======
            layer_attn_traffic = 0.0
            layer_ffn_traffic = 0.0
            
            # --- Attention weight parameters ---
            if is_mamba_layer and model.mamba_config is not None:
                mc = model.mamba_config
                d_inner = mc.d_inner
                d_proj = 2 * d_inner + 2 * mc.num_heads * mc.state_size + mc.num_heads
                attn_weight_params = H * d_proj + d_inner * H  # in_proj + out_proj
                attn_weight_params += d_inner * mc.conv_kernel  # conv1d
            elif layer_attn_type == LayerAttentionType.LINEAR_ATTENTION and model.linear_attention_config is not None:
                lac = model.linear_attention_config
                num_attn_heads = model.attention_config.num_attention_heads
                v_per_group = lac.num_value_heads // lac.num_key_heads if lac.num_key_heads > 0 else lac.num_value_heads
                q_proj = H * (num_attn_heads * lac.key_head_dim)
                k_proj = H * (lac.num_key_heads * lac.key_head_dim)
                v_proj = H * (lac.num_value_heads * lac.value_head_dim)
                o_proj = (num_attn_heads * v_per_group * lac.value_head_dim) * H
                conv_params = (lac.num_key_heads * lac.key_head_dim + lac.num_value_heads * lac.value_head_dim) * lac.conv_kernel_dim
                attn_weight_params = q_proj + k_proj + v_proj + o_proj + conv_params
            else:
                # Standard / sliding attention weights
                num_attn_heads = model.attention_config.num_attention_heads
                num_kv_heads = model.attention_config.num_key_value_heads
                head_dim = model.attention_config.head_dim
                if model.attention_config.use_mla and model.attention_config.mla_kv_lora_rank:
                    kv_rank = model.attention_config.mla_kv_lora_rank
                    kv_proj = H * kv_rank
                    kv_decompress = kv_rank * (num_kv_heads * head_dim) * 2
                    if model.attention_config.mla_q_lora_rank:
                        q_rank = model.attention_config.mla_q_lora_rank
                        q_proj = H * q_rank + q_rank * (num_attn_heads * head_dim)
                    else:
                        q_proj = H * (num_attn_heads * head_dim)
                    o_proj = (num_attn_heads * head_dim) * H
                    attn_weight_params = q_proj + kv_proj + kv_decompress + o_proj
                else:
                    q_proj = H * (num_attn_heads * head_dim)
                    k_proj = H * (num_kv_heads * head_dim)
                    v_proj = H * (num_kv_heads * head_dim)
                    o_proj = (num_attn_heads * head_dim) * H
                    attn_weight_params = q_proj + k_proj + v_proj + o_proj
                
                # DSA indexer weights (if applicable)
                if model.attention_config.use_dsa:
                    attn_weight_params += H * model.attention_config.dsa_q_indexer_dim
                    attn_weight_params += H * model.attention_config.dsa_k_indexer_dim
            
            layer_attn_traffic += attn_weight_params * bytes_per_param / tp
            
            # --- KV cache / state traffic ---
            if is_mamba_layer and model.mamba_config is not None:
                # Mamba state: read + write
                mamba_state = model.mamba_config.get_state_size_bytes(effective_batch, int(bytes_per_param))
                layer_attn_traffic += 2 * mamba_state / tp
            elif layer_attn_type == LayerAttentionType.LINEAR_ATTENTION and model.linear_attention_config is not None:
                # Linear attention: fixed-size state read + write
                state_traffic = model.linear_attention_config.get_decode_state_traffic(
                    effective_batch, int(bytes_per_param))
                layer_attn_traffic += state_traffic / tp
            else:
                # KV cache traffic
                if model.attention_config.use_mla and model.attention_config.mla_kv_lora_rank:
                    kv_dim = model.attention_config.mla_kv_lora_rank
                else:
                    kv_dim = model.attention_config.num_key_value_heads * model.attention_config.head_dim
                
                effective_ctx = L
                if layer_attn_type == LayerAttentionType.SLIDING_ATTENTION:
                    window = model.attention_config.sliding_window_size or L
                    effective_ctx = min(L, window)
                
                if model.attention_config.use_dsa and model.attention_config.dsa_top_k:
                    effective_ctx = min(effective_ctx, model.attention_config.dsa_top_k)
                
                # KV cache: 2 tensors (K, V) × batch × seq × kv_dim × bytes
                if is_prefill:
                    # Prefill: write KV cache for all tokens
                    kv_traffic = 2 * effective_batch * effective_ctx * kv_dim * bytes_per_param
                else:
                    # Decode: read existing KV + write 1 new token
                    kv_read = 2 * effective_batch * effective_ctx * kv_dim * bytes_per_param
                    kv_write = 2 * effective_batch * 1 * kv_dim * bytes_per_param
                    kv_traffic = kv_read + kv_write
                
                layer_attn_traffic += kv_traffic / tp
            
            # --- FFN weight parameters ---
            if is_moe_layer and model.moe_config is not None:
                intermediate = model.ffn_config.intermediate_size
                E = model.moe_config.num_experts
                router_params = H * E
                per_expert = H * intermediate + intermediate * H  # up + down
                if model.ffn_config.use_gating:
                    per_expert += H * intermediate  # gate
                ffn_weight_params = router_params + E * per_expert
            else:
                intermediate = model.get_dense_intermediate_size()
                ffn_weight_params = H * intermediate + intermediate * H  # up + down
                if model.ffn_config.use_gating:
                    ffn_weight_params += H * intermediate  # gate
            
            layer_ffn_traffic += ffn_weight_params * bytes_per_param / tp
            
            # Layer norm params (small, added to non-attention)
            layer_ffn_traffic += 2 * 2 * H * bytes_per_param / tp  # 2 layer norms, each 2*H params (scale+bias)
            
            result.attention_memory_traffic.append(layer_attn_traffic)
            result.non_attention_memory_traffic.append(layer_ffn_traffic)
            
            # ====== KERNEL LAUNCHES ======
            layer_attn_kernels = 0
            layer_ffn_kernels = 0
            
            if is_mamba_layer and model.mamba_config is not None:
                if is_prefill:
                    layer_attn_kernels = model.mamba_config.get_prefill_kernel_launches(L)
                else:
                    layer_attn_kernels = model.mamba_config.get_decode_kernel_launches()
                # Mamba layers typically don't have separate FFN in the standard sense,
                # but the MLP part after Mamba still exists
                layer_ffn_kernels = 2  # up/down or gated FFN
            elif layer_attn_type == LayerAttentionType.LINEAR_ATTENTION:
                # Linear attention: norm + Q/K/V proj + conv + state_update + query + O_proj = 8
                # Split: 6 attention kernels (norm, QKV, conv, state, query, O_proj)
                #        2 FFN kernels (norm counted here, up+down)
                layer_attn_kernels = 6
                layer_ffn_kernels = 2  # post-attn norm + FFN fused = actually 3 with norm
            else:
                # Standard/sliding attention:
                # Attention: pre-norm(1) + QKV(1) + scores(1) + softmax(1) + attn_out(1) + O_proj(1) = 6
                layer_attn_kernels = 6
                if model.attention_config.use_mla:
                    layer_attn_kernels += 2  # compression + decompression
                # FFN: post-attn-norm(1) + up+gate(1) + down(1) = 3
                layer_ffn_kernels = 3
            
            # MoE overhead on FFN side
            if is_moe_layer and model.moe_config:
                layer_ffn_kernels += 3  # router + dispatch + combine
            
            result.attention_kernels.append(layer_attn_kernels)
            result.non_attention_kernels.append(layer_ffn_kernels)
        
        return result

    def calculate_achievable_ttft(
        self,
        system_constraints: SystemConstraints,
        batch_size: int,
        sequence_length: int,
        parallelism_config: Optional[ParallelismConfig] = None,
        kernel_launch_latency: float = 5e-6,
        dtype_override: Optional[str] = None
    ) -> ResourceUtilization:
        """
        Calculate achievable TTFT given system constraints (inverse of calculate_prefill_resources).
        
        Instead of: "Given TTFT target, what resources do I need?"
        This answers: "Given system resources, what TTFT can I achieve?"
        
        Args:
            system_constraints: Hardware constraints (memory, bandwidth, compute, network)
            batch_size: Number of sequences processed in parallel
            sequence_length: Length of input sequences
            parallelism_config: Optional parallelism configuration
            kernel_launch_latency: Time per kernel launch (default 5µs)
            dtype_override: Optional dtype override ('int4', 'int8', 'float16', 'bfloat16', 'float32')
            kernel_launch_latency: Time per kernel launch (default 5µs)
        
        Returns:
            ResourceUtilization with achievable TTFT and resource utilization metrics
        """
        if parallelism_config is None:
            parallelism_config = ParallelismConfig(
                tensor_parallel_size=1,
                pipeline_parallel_size=1,
                data_parallel_size=1
            )
        
        # Calculate resource requirements
        total_flops = self.calculate_prefill_compute(batch_size, sequence_length, parallelism_config)
        compute_breakdown = self.calculate_prefill_compute_breakdown(batch_size, sequence_length, parallelism_config)
        
        memory = self.calculate_prefill_memory(batch_size, sequence_length, parallelism_config, dtype_override)
        memory_required = memory["total"]
        
        # Calculate kernel launch overhead
        num_kernel_launches = self.calculate_num_kernel_launches(parallelism_config)
        kernel_launch_overhead = num_kernel_launches * kernel_launch_latency
        
        # Calculate time constrained by each resource (per-GPU model)
        # FLOPs and memory traffic are calculated per-GPU
        # With TP: each GPU processes 1/TP of attention heads
        # With PP: each GPU processes 1/PP of layers (already divided in FLOP calc)
        # With DP: each GPU processes 1/DP of batch (already divided in FLOP calc)
        effective_compute = system_constraints.compute_throughput
        effective_memory_bw = system_constraints.memory_bandwidth
        
        # Divide FLOPs by TP (each GPU does 1/TP of the layer-wise work)
        total_flops = total_flops / parallelism_config.tensor_parallel_size
        
        # 1. Compute time: How long to execute the FLOPs?
        compute_time = total_flops / effective_compute
        
        # 2. Memory bandwidth time: How long to move the data?
        # For prefill, we need to:
        # - Read model weights (~model_size)
        # - Read/write activations
        # - Write KV cache
        # Rough estimate: 2x memory footprint for read+write
        memory_traffic = 2 * memory_required
        memory_bw_time = memory_traffic / effective_memory_bw
        
        # 3. Network time: For tensor/pipeline parallelism
        network_time = 0.0
        if parallelism_config.tensor_parallel_size > 1 or parallelism_config.pipeline_parallel_size > 1:
            # Estimate network traffic
            # Tensor parallel: all-reduce of activations (~batch * seq * hidden)
            # Pipeline parallel: send activations between stages
            activation_size = batch_size * sequence_length * self.model.hidden_dim * 2  # bytes (FP16)
            
            # Each layer needs communication
            num_comms = self.model.num_layers
            if parallelism_config.tensor_parallel_size > 1:
                # All-reduce: 2x data transfer (send + receive)
                num_comms *= 2
            
            network_traffic = activation_size * num_comms
            network_time = network_traffic / system_constraints.network_bandwidth
        
        # The actual time is determined by the slowest resource (bottleneck)
        effective_compute_time = max(compute_time, memory_bw_time, network_time)
        
        # Total TTFT = effective compute time + kernel launch overhead
        achievable_ttft = effective_compute_time + kernel_launch_overhead
        
        # Determine bottleneck
        if effective_compute_time == compute_time:
            bottleneck = "Compute"
        elif effective_compute_time == memory_bw_time:
            bottleneck = "Memory Bandwidth"
        else:
            bottleneck = "Network Bandwidth"
        
        # Calculate resource utilization
        memory_util = memory_required / system_constraints.memory_capacity
        
        # For bandwidth/compute, utilization is how much of the resource is actually used
        # when the bottleneck determines the time
        memory_bw_util = memory_bw_time / effective_compute_time
        compute_util = compute_time / effective_compute_time
        network_bw_util = network_time / effective_compute_time if network_time > 0 else 0.0
        
        # Calculate actual usage rates
        memory_bw_used = memory_traffic / effective_compute_time
        compute_used = total_flops / effective_compute_time
        network_bw_used = (network_traffic / effective_compute_time) if network_time > 0 else 0.0
        
        # Calculate time breakdown
        # Compute busy = time compute is actually working (at 100%)
        time_compute_busy = compute_time
        # Kernel launch = overhead from kernel launches
        time_kernel_launch = kernel_launch_overhead
        # Idle = time waiting on memory bandwidth or network (compute is idle)
        time_idle = effective_compute_time - compute_time
        
        return ResourceUtilization(
            achievable_ttft=achievable_ttft,
            bottleneck_resource=bottleneck,
            memory_utilization=memory_util,
            memory_bandwidth_utilization=memory_bw_util,
            compute_utilization=compute_util,
            network_bandwidth_utilization=network_bw_util,
            memory_used=memory_required,
            memory_bandwidth_used=memory_bw_used,
            compute_used=compute_used,
            network_bandwidth_used=network_bw_used,
            memory_available=system_constraints.memory_capacity,
            memory_bandwidth_available=system_constraints.memory_bandwidth,
            compute_available=system_constraints.compute_throughput,
            network_bandwidth_available=system_constraints.network_bandwidth,
            memory_weights=memory["model_weights"],
            memory_kv_cache=memory["kv_cache"],
            memory_activations=memory["activations"],
            compute_attention=compute_breakdown["attention"],
            compute_ffn=compute_breakdown["ffn"],
            compute_other=compute_breakdown["other"],
            kernel_launch_overhead=kernel_launch_overhead,
            effective_compute_time=effective_compute_time,
            time_compute_busy=time_compute_busy,
            time_kernel_launch=time_kernel_launch,
            time_idle=time_idle
        )
    
    def calculate_decode_performance(
        self,
        system_constraints: SystemConstraints,
        batch_size: int,
        prefill_length: int,
        output_length: int,
        parallelism_config: Optional[ParallelismConfig] = None,
        kernel_launch_latency: float = 5e-6,
        return_step_details: bool = False,
        decode_step_skip: int = 100,
        dtype_override: Optional[str] = None
    ) -> DecodePerformance:
        """
        Calculate decode phase performance (autoregressive token generation).
        
        For each decode step:
        - Process 1 new token per sequence (batch_size tokens total)
        - Attend to all previous tokens (context grows with each step)
        - KV cache grows, making later steps more expensive
        
        Args:
            system_constraints: Hardware constraints (memory, bandwidth, compute, network)
            batch_size: Number of sequences being generated
            prefill_length: Length of input prompt (already processed)
            output_length: Number of tokens to generate
            parallelism_config: Optional parallelism configuration
            kernel_launch_latency: Time per kernel launch (default 5µs)
            return_step_details: If True, include per-step details in result
            decode_step_skip: Sample every Nth step and accumulate (default 100 for performance)
            dtype_override: Optional dtype override ('int4', 'int8', 'float16', 'bfloat16', 'float32')
        
        Returns:
            DecodePerformance with timing, throughput, and resource utilization
        """
        if parallelism_config is None:
            parallelism_config = ParallelismConfig(
                tensor_parallel_size=1,
                pipeline_parallel_size=1,
                data_parallel_size=1
            )
        
        step_details = []
        total_time = 0.0
        
        # Track resource utilization across all steps
        total_memory_util = 0.0
        total_memory_bw_util = 0.0
        total_compute_util = 0.0
        total_network_bw_util = 0.0
        total_storage_bw_util = 0.0  # Track persistent storage utilization
        
        # Track memory breakdown across all steps
        total_memory_weights = 0.0
        total_memory_kv_cache = 0.0
        total_memory_activations = 0.0
        
        # Track compute breakdown (total across all steps)
        total_attention_flops = 0.0
        total_ffn_flops = 0.0
        total_other_flops = 0.0
        
        # Track time breakdown (total across all steps)
        total_time_compute_busy = 0.0
        total_time_kernel_launch = 0.0
        total_time_idle = 0.0
        
        # Track min/max step times
        min_step_time = float('inf')
        max_step_time = 0.0
        
        bottleneck_counts = {}
        
        # Loop over decode steps with skipping for performance
        # We sample every Nth step and accumulate results
        for step in range(0, output_length, decode_step_skip):
            # Current context length = prefill + tokens generated so far
            context_length = prefill_length + step
            
            # Calculate resources for this step
            step_resources = self._calculate_decode_step(
                system_constraints=system_constraints,
                batch_size=batch_size,
                context_length=context_length,
                parallelism_config=parallelism_config,
                kernel_launch_latency=kernel_launch_latency,
                step=step,
                dtype_override=dtype_override
            )
            
            # Determine how many steps this sample represents
            steps_represented = min(decode_step_skip, output_length - step)
            
            # Store step details if requested
            if return_step_details:
                step_details.append(step_resources)
            
            # Track min/max step times
            min_step_time = min(min_step_time, step_resources.step_time)
            max_step_time = max(max_step_time, step_resources.step_time)
            
            # Accumulate results for all steps represented by this sample
            total_time += step_resources.step_time * steps_represented
            
            # Accumulate utilization
            # Calculate what fraction of each resource was used
            memory_breakdown = self.calculate_prefill_memory(
                batch_size, context_length, parallelism_config, dtype_override
            )
            memory_required = memory_breakdown["total"]
            memory_util = memory_required / system_constraints.memory_capacity
            
            # Accumulate memory breakdown
            total_memory_weights += memory_breakdown["model_weights"] * steps_represented
            total_memory_kv_cache += memory_breakdown["kv_cache"] * steps_represented
            total_memory_activations += memory_breakdown["activations"] * steps_represented
            
            # Accumulate compute breakdown
            compute_breakdown = self._calculate_decode_step_compute_breakdown(
                batch_size, context_length, parallelism_config
            )
            total_attention_flops += compute_breakdown["attention"] * steps_represented
            total_ffn_flops += compute_breakdown["ffn"] * steps_represented
            total_other_flops += compute_breakdown["other"] * steps_represented
            
            # Accumulate time breakdown
            # For each step: 
            # - compute_busy = time compute is actually working (at 100%)
            # - kernel_launch = overhead from kernel launches
            # - idle = time waiting on memory/network/storage (bottleneck_time - compute_time)
            step_bottleneck_time = max(step_resources.compute_time, step_resources.memory_bw_time, 
                                      step_resources.network_time, step_resources.storage_bw_time)
            step_compute_busy = step_resources.compute_time
            step_kernel_launch = step_resources.kernel_overhead
            step_idle = step_bottleneck_time - step_resources.compute_time
            
            total_time_compute_busy += step_compute_busy * steps_represented
            total_time_kernel_launch += step_kernel_launch * steps_represented
            total_time_idle += step_idle * steps_represented
            
            compute_util = step_resources.compute_time / step_resources.step_time
            memory_bw_util = step_resources.memory_bw_time / step_resources.step_time
            network_bw_util = step_resources.network_time / step_resources.step_time if step_resources.network_time > 0 else 0.0
            storage_bw_util = step_resources.storage_bw_time / step_resources.step_time if step_resources.storage_bw_time > 0 else 0.0
            
            total_memory_util += memory_util * steps_represented
            total_memory_bw_util += memory_bw_util * steps_represented
            total_compute_util += compute_util * steps_represented
            total_network_bw_util += network_bw_util * steps_represented
            total_storage_bw_util += storage_bw_util * steps_represented
            
            # Track bottlenecks
            bottleneck_counts[step_resources.bottleneck] = bottleneck_counts.get(step_resources.bottleneck, 0) + steps_represented
        
        # Calculate averages
        avg_memory_util = total_memory_util / output_length
        avg_memory_bw_util = total_memory_bw_util / output_length
        avg_compute_util = total_compute_util / output_length
        avg_network_bw_util = total_network_bw_util / output_length
        avg_storage_bw_util = total_storage_bw_util / output_length
        
        # Calculate average memory breakdown
        avg_memory_weights = total_memory_weights / output_length
        avg_memory_kv_cache = total_memory_kv_cache / output_length
        avg_memory_activations = total_memory_activations / output_length
        
        # Determine primary bottleneck
        primary_bottleneck = max(bottleneck_counts.items(), key=lambda x: x[1])[0]
        
        # Calculate throughput metrics
        tokens_per_second_per_user = output_length / total_time
        total_throughput = batch_size * tokens_per_second_per_user
        
        # Calculate average step time
        avg_step_time = total_time / output_length
        
        return DecodePerformance(
            batch_size=batch_size,
            prefill_length=prefill_length,
            output_length=output_length,
            total_sequence_length=prefill_length + output_length,
            total_decode_time=total_time,
            avg_step_time=avg_step_time,
            min_step_time=min_step_time,
            max_step_time=max_step_time,
            tokens_per_second_per_user=tokens_per_second_per_user,
            total_throughput=total_throughput,
            avg_memory_utilization=avg_memory_util,
            avg_memory_bw_utilization=avg_memory_bw_util,
            avg_compute_utilization=avg_compute_util,
            avg_network_bw_utilization=avg_network_bw_util,
            avg_storage_bw_utilization=avg_storage_bw_util,
            avg_memory_weights=avg_memory_weights,
            avg_memory_kv_cache=avg_memory_kv_cache,
            avg_memory_activations=avg_memory_activations,
            total_compute_attention=total_attention_flops,
            total_compute_ffn=total_ffn_flops,
            total_compute_other=total_other_flops,
            total_time_compute_busy=total_time_compute_busy,
            total_time_kernel_launch=total_time_kernel_launch,
            total_time_idle=total_time_idle,
            bottleneck_breakdown=bottleneck_counts,
            primary_bottleneck=primary_bottleneck,
            step_details=step_details if return_step_details else [],
            memory_capacity=system_constraints.memory_capacity,
            memory_bandwidth=system_constraints.memory_bandwidth,
            compute_throughput=system_constraints.compute_throughput,
            network_bandwidth=system_constraints.network_bandwidth,
            persistent_storage_bandwidth=system_constraints.persistent_storage_bandwidth
        )
    
    def _calculate_decode_step(
        self,
        system_constraints: SystemConstraints,
        batch_size: int,
        context_length: int,
        parallelism_config: ParallelismConfig,
        kernel_launch_latency: float,
        step: int,
        dtype_override: Optional[str] = None
    ) -> DecodeStepResources:
        """
        Calculate resources for a single decode step.
        
        Each decode step:
        - Processes batch_size tokens (1 new token per sequence)
        - Attends to context_length tokens (prefill + generated so far)
        """
        # Calculate compute (FLOPs)
        compute_flops = self._calculate_decode_step_compute(
            batch_size, context_length, parallelism_config
        )
        
        # Calculate memory traffic
        memory_traffic = self._calculate_decode_step_memory_traffic(
            batch_size, context_length, parallelism_config, dtype_override
        )
        
        # Calculate persistent storage traffic (for MoE offloading when model doesn't fit in DRAM)
        storage_traffic = 0.0
        storage_bw_time = 0.0
        
        # Check if model fits in DRAM (account for both TP and PP)
        bytes_per_param = self._get_bytes_per_param(dtype_override)
        total_parallelism = parallelism_config.tensor_parallel_size * parallelism_config.pipeline_parallel_size
        model_memory_required = self.model.total_parameters * bytes_per_param / total_parallelism
        model_fits_in_dram = model_memory_required <= system_constraints.memory_capacity
        
        # If MoE model doesn't fit in DRAM, active experts must be loaded from persistent storage
        if self.model.is_moe and not model_fits_in_dram:
            # Calculate size of active experts per token
            num_active_experts = self.model.moe_config.num_experts_per_token
            intermediate_size = self.model.ffn_config.intermediate_size
            
            # Each active expert has: up projection + down projection
            # up: (hidden_dim, intermediate_size), down: (intermediate_size, hidden_dim)
            params_per_expert = 2 * self.model.hidden_dim * intermediate_size
            active_expert_size = num_active_experts * params_per_expert * bytes_per_param
            
            # Scale by parallelism
            active_expert_size /= parallelism_config.tensor_parallel_size
            
            # Must read active experts for each MoE layer (not dense layers in interleaved models)
            num_moe_layers = self.model.get_num_moe_ffn_layers()
            storage_traffic = active_expert_size * num_moe_layers
            
            # Calculate time to read from persistent storage
            storage_bw_time = storage_traffic / system_constraints.persistent_storage_bandwidth
        
        # Calculate network traffic
        network_traffic = self._calculate_decode_step_network_traffic(
            batch_size, context_length, parallelism_config
        )
        
        # Calculate kernel launch overhead
        # Decode has similar kernel structure to prefill (one pass through model)
        num_kernel_launches = self.calculate_num_kernel_launches(parallelism_config)
        kernel_overhead = num_kernel_launches * kernel_launch_latency
        
        # Calculate time constrained by each resource (per-GPU model)
        # FLOPs and memory traffic are calculated per-GPU
        # With TP: each GPU processes 1/TP of attention heads
        # With PP: each GPU processes 1/PP of layers (already divided in FLOP calc)
        # With DP: each GPU processes 1/DP of batch (already divided in FLOP calc)
        effective_compute = system_constraints.compute_throughput
        effective_memory_bw = system_constraints.memory_bandwidth
        
        # Divide FLOPs by TP (each GPU does 1/TP of the layer-wise work)
        compute_flops = compute_flops / parallelism_config.tensor_parallel_size
        
        # Calculate time constraints from each resource
        compute_time = compute_flops / effective_compute
        memory_bw_time = memory_traffic / effective_memory_bw
        network_time = network_traffic / system_constraints.network_bandwidth if network_traffic > 0 else 0.0
        
        # Determine bottleneck (including persistent storage)
        effective_time = max(compute_time, memory_bw_time, network_time, storage_bw_time)
        step_time = effective_time + kernel_overhead
        
        if effective_time == compute_time:
            bottleneck = "Compute"
        elif effective_time == memory_bw_time:
            bottleneck = "Memory Bandwidth"
        elif effective_time == storage_bw_time:
            bottleneck = "Persistent Storage Bandwidth"
        else:
            bottleneck = "Network Bandwidth"
        
        return DecodeStepResources(
            step=step,
            context_length=context_length,
            step_time=step_time,
            compute_time=compute_time,
            memory_bw_time=memory_bw_time,
            network_time=network_time,
            kernel_overhead=kernel_overhead,
            storage_bw_time=storage_bw_time,
            compute_flops=compute_flops,
            memory_traffic=memory_traffic,
            network_traffic=network_traffic,
            storage_traffic=storage_traffic,
            bottleneck=bottleneck
        )
    
    def _calculate_decode_step_compute(
        self,
        batch_size: int,
        context_length: int,
        parallelism_config: ParallelismConfig
    ) -> float:
        """
        Calculate FLOPs for a single decode step.
        
        Key insight: We process 1 new token but attend to context_length tokens.
        """
        model = self.model
        bytes_per_param = 2  # FP16
        
        # Effective batch after data parallelism
        effective_batch = batch_size // parallelism_config.data_parallel_size
        
        # Sequence length for this step = 1 (generating 1 token)
        seq_len = 1
        
        total_flops = 0.0
        
        # Per-layer computation
        # Each pipeline stage processes num_layers / pipeline_parallel_size layers
        layers_per_stage = model.num_layers // parallelism_config.pipeline_parallel_size
        for layer_idx in range(layers_per_stage):
            
            # === Attention ===
            # Calculate KV dimension
            num_kv_heads = model.attention_config.num_key_value_heads
            head_dim = model.attention_config.head_dim
            kv_dim = num_kv_heads * head_dim
            
            # Q projection: (batch, 1, hidden) @ (hidden, hidden) = (batch, 1, hidden)
            q_flops = 2 * effective_batch * seq_len * model.hidden_dim * model.hidden_dim
            
            # K, V projections (if not using MLA)
            if not model.attention_config.use_mla:
                # K projection
                k_flops = 2 * effective_batch * seq_len * model.hidden_dim * kv_dim
                # V projection
                v_flops = 2 * effective_batch * seq_len * model.hidden_dim * kv_dim
            else:
                # MLA: compressed representations
                # K projection to compressed space
                k_flops = 2 * effective_batch * seq_len * model.hidden_dim * model.attention_config.mla_kv_lora_rank
                # V projection to compressed space
                v_flops = 2 * effective_batch * seq_len * model.hidden_dim * model.attention_config.mla_kv_lora_rank
                
                # Decompression for attention (project back to full KV dim)
                # This is done for ALL cached keys/values (context_length)
                decompress_k_flops = 2 * effective_batch * context_length * model.attention_config.mla_kv_lora_rank * kv_dim
                decompress_v_flops = 2 * effective_batch * context_length * model.attention_config.mla_kv_lora_rank * kv_dim
                k_flops += decompress_k_flops
                v_flops += decompress_v_flops
            
            # Attention scores: Q @ K^T
            # (batch, heads, 1, head_dim) @ (batch, heads, head_dim, context_len)
            # = (batch, heads, 1, context_len)
            
            # DSA: Dynamic Sparse Attention with top-K selection
            if model.attention_config.use_dsa:
                # Pseudo-attention for top-K selection
                # [N, d_Q_indexer] @ [d_Q_indexer, d_k_indexer] = [N, d_k_indexer]
                pseudo_attn_flops = 2 * effective_batch * context_length * model.attention_config.dsa_q_indexer_dim * model.attention_config.dsa_k_indexer_dim
                
                # Actual attention only on top-K selected KV pairs
                effective_context = min(context_length, model.attention_config.dsa_top_k)
                attn_scores_flops = 2 * effective_batch * model.attention_config.num_attention_heads * seq_len * effective_context * model.attention_config.head_dim
                attn_output_flops = 2 * effective_batch * model.attention_config.num_attention_heads * seq_len * effective_context * model.attention_config.head_dim
                
                # Add pseudo-attention overhead
                attn_scores_flops += pseudo_attn_flops
            else:
                # Standard full attention
                attn_scores_flops = 2 * effective_batch * model.attention_config.num_attention_heads * seq_len * context_length * model.attention_config.head_dim
                attn_output_flops = 2 * effective_batch * model.attention_config.num_attention_heads * seq_len * context_length * model.attention_config.head_dim
            
            # Softmax (not counted as negligible compared to matmuls)
            
            # Output projection
            out_proj_flops = 2 * effective_batch * seq_len * model.hidden_dim * model.hidden_dim
            
            attention_flops = q_flops + k_flops + v_flops + attn_scores_flops + attn_output_flops + out_proj_flops
            
            # === FFN or MoE ===
            if model.is_moe and model.moe_config is not None:
                # MoE layer
                num_active_experts = model.moe_config.num_experts_per_token
                intermediate_size = model.ffn_config.intermediate_size
                
                # Router
                router_flops = 2 * effective_batch * seq_len * model.hidden_dim * model.moe_config.num_experts
                
                # Expert computation (only active experts)
                # Up projection
                up_flops = 2 * effective_batch * seq_len * num_active_experts * model.hidden_dim * intermediate_size
                
                # Activation (negligible)
                
                # Down projection
                down_flops = 2 * effective_batch * seq_len * num_active_experts * intermediate_size * model.hidden_dim
                
                ffn_flops = router_flops + up_flops + down_flops
            else:
                # Dense FFN
                intermediate_size = model.ffn_config.intermediate_size
                
                # Up projection (and gate if gated)
                if model.ffn_config.use_gating:
                    # Two projections: gate and up
                    up_flops = 2 * 2 * effective_batch * seq_len * model.hidden_dim * intermediate_size
                else:
                    up_flops = 2 * effective_batch * seq_len * model.hidden_dim * intermediate_size
                
                # Activation (negligible)
                
                # Down projection
                down_flops = 2 * effective_batch * seq_len * intermediate_size * model.hidden_dim
                
                ffn_flops = up_flops + down_flops
            
            # Layer norms (negligible compared to matmuls)
            
            total_flops += attention_flops + ffn_flops
        
        # Final layer norm and LM head
        lm_head_flops = 2 * effective_batch * seq_len * model.hidden_dim * model.vocab_size
        total_flops += lm_head_flops
        
        # Note: FLOPs represent total work across all GPUs
        # The parallelism is accounted for by scaling system resources (compute throughput)
        # in the calling function
        
        return total_flops
    
    def _calculate_decode_step_compute_breakdown(
        self,
        batch_size: int,
        context_length: int,
        parallelism_config: ParallelismConfig
    ) -> Dict[str, float]:
        """
        Calculate FLOPs breakdown for a single decode step.
        
        Returns:
            Dictionary with keys: 'attention', 'mamba', 'ffn', 'other', 'total'
        """
        model = self.model
        effective_batch = batch_size // parallelism_config.data_parallel_size
        seq_len = 1
        
        total_attention_flops = 0.0
        total_mamba_flops = 0.0  # Track Mamba FLOPs separately
        total_ffn_flops = 0.0
        other_flops = 0.0
        
        # Per-layer computation
        # Each pipeline stage processes num_layers / pipeline_parallel_size layers
        layers_per_stage = model.num_layers // parallelism_config.pipeline_parallel_size
        for layer_idx in range(layers_per_stage):
            # Check if this is a Mamba layer in hybrid architecture
            is_mamba_layer = False
            if model.hybrid_layer_types is not None:
                is_mamba_layer = model.hybrid_layer_types[layer_idx] == HybridLayerType.MAMBA
            
            if is_mamba_layer and model.mamba_config is not None:
                # Mamba-2 layer: use Mamba-specific decode FLOPs
                layer_mamba_flops = model.mamba_config.get_decode_flops(model.hidden_dim)
                total_mamba_flops += layer_mamba_flops * effective_batch
            else:
                # Determine layer attention type
                layer_attn_type = model.get_layer_attention_type(layer_idx)
                
                if layer_attn_type == LayerAttentionType.LINEAR_ATTENTION and model.linear_attention_config is not None:
                    # Linear attention layer: CONSTANT cost regardless of context length
                    lac = model.linear_attention_config
                    num_attn_heads = model.attention_config.num_attention_heads
                    layer_linear_flops = lac.get_decode_flops(effective_batch, num_attn_heads, model.hidden_dim)
                    total_attention_flops += layer_linear_flops
                else:
                    # === Standard/Sliding Attention ===
                    num_kv_heads = model.attention_config.num_key_value_heads
                    head_dim = model.attention_config.head_dim
                    kv_dim = num_kv_heads * head_dim
                    
                    # Determine effective context for attention computation
                    # For sliding window, attention is limited to window size
                    effective_context = context_length  # Default: full context
                    if layer_attn_type == LayerAttentionType.SLIDING_ATTENTION:
                        window_size = model.attention_config.sliding_window_size or context_length
                        effective_context = min(context_length, window_size)
                    
                    q_flops = 2 * effective_batch * seq_len * model.hidden_dim * model.hidden_dim
                    
                    if not model.attention_config.use_mla:
                        k_flops = 2 * effective_batch * seq_len * model.hidden_dim * kv_dim
                        v_flops = 2 * effective_batch * seq_len * model.hidden_dim * kv_dim
                    else:
                        k_flops = 2 * effective_batch * seq_len * model.hidden_dim * model.attention_config.mla_kv_lora_rank
                        v_flops = 2 * effective_batch * seq_len * model.hidden_dim * model.attention_config.mla_kv_lora_rank
                        # MLA decompression uses effective context (reduced for sliding window)
                        decompress_k_flops = 2 * effective_batch * effective_context * model.attention_config.mla_kv_lora_rank * kv_dim
                        decompress_v_flops = 2 * effective_batch * effective_context * model.attention_config.mla_kv_lora_rank * kv_dim
                        k_flops += decompress_k_flops
                        v_flops += decompress_v_flops
                    
                    # DSA: Dynamic Sparse Attention with top-K selection
                    if model.attention_config.use_dsa:
                        # Pseudo-attention for top-K selection
                        pseudo_attn_flops = 2 * effective_batch * effective_context * model.attention_config.dsa_q_indexer_dim * model.attention_config.dsa_k_indexer_dim
                        
                        # Actual attention only on top-K selected KV pairs
                        dsa_effective_context = min(effective_context, model.attention_config.dsa_top_k)
                        attn_scores_flops = 2 * effective_batch * model.attention_config.num_attention_heads * seq_len * dsa_effective_context * model.attention_config.head_dim
                        attn_output_flops = 2 * effective_batch * model.attention_config.num_attention_heads * seq_len * dsa_effective_context * model.attention_config.head_dim
                        
                        attn_scores_flops += pseudo_attn_flops
                    else:
                        # Standard attention (uses effective context for sliding window)
                        attn_scores_flops = 2 * effective_batch * model.attention_config.num_attention_heads * seq_len * effective_context * model.attention_config.head_dim
                        attn_output_flops = 2 * effective_batch * model.attention_config.num_attention_heads * seq_len * effective_context * model.attention_config.head_dim
                    
                    out_proj_flops = 2 * effective_batch * seq_len * model.hidden_dim * model.hidden_dim
                    
                    attention_flops = q_flops + k_flops + v_flops + attn_scores_flops + attn_output_flops + out_proj_flops
                    total_attention_flops += attention_flops
            
            # === FFN or MoE - check if this specific layer is MoE or Dense ===
            is_moe_layer = False
            if model.ffn_layer_types is not None:
                is_moe_layer = model.ffn_layer_types[layer_idx] == FFNLayerType.MOE
            elif model.is_moe:
                # All layers are MoE if no per-layer specification
                is_moe_layer = True
            
            if is_moe_layer and model.moe_config is not None:
                # MoE layer: router + active experts
                num_active_experts = model.moe_config.num_experts_per_token
                intermediate_size = model.ffn_config.intermediate_size  # Per-expert size
                router_flops = 2 * effective_batch * seq_len * model.hidden_dim * model.moe_config.num_experts
                up_flops = 2 * effective_batch * seq_len * num_active_experts * model.hidden_dim * intermediate_size
                down_flops = 2 * effective_batch * seq_len * num_active_experts * intermediate_size * model.hidden_dim
                ffn_flops = router_flops + up_flops + down_flops
            else:
                # Dense layer: use dense_intermediate_size
                intermediate_size = model.get_dense_intermediate_size()
                if model.ffn_config.use_gating:
                    up_flops = 2 * 2 * effective_batch * seq_len * model.hidden_dim * intermediate_size
                else:
                    up_flops = 2 * effective_batch * seq_len * model.hidden_dim * intermediate_size
                down_flops = 2 * effective_batch * seq_len * intermediate_size * model.hidden_dim
                ffn_flops = up_flops + down_flops
            
            total_ffn_flops += ffn_flops
        
        # Final layer norm and LM head
        lm_head_flops = 2 * effective_batch * seq_len * model.hidden_dim * model.vocab_size
        other_flops += lm_head_flops
        
        return {
            'attention': total_attention_flops,
            'mamba': total_mamba_flops,  # Mamba SSM FLOPs (0 for non-hybrid models)
            'ffn': total_ffn_flops,
            'other': other_flops,
            'total': total_attention_flops + total_mamba_flops + total_ffn_flops + other_flops
        }
    
    def _calculate_decode_step_memory(
        self,
        batch_size: int,
        context_length: int,
        parallelism_config: ParallelismConfig,
        dtype_override: Optional[str] = None
    ) -> float:
        """Calculate memory requirements for decode step"""
        model = self.model
        bytes_per_param = self._get_bytes_per_param(dtype_override)
        
        # Model weights (same as prefill)
        model_size = model.total_parameters * bytes_per_param
        model_size /= parallelism_config.tensor_parallel_size
        
        # KV cache (stores all context)
        kv_cache_size = model.get_kv_cache_size(
            batch_size=batch_size,
            sequence_length=context_length,
            bytes_per_element=bytes_per_param
        )
        kv_cache_size /= (parallelism_config.tensor_parallel_size * 
                         parallelism_config.pipeline_parallel_size)
        
        # Activations (just for 1 token)
        # Much smaller than prefill
        effective_batch = batch_size // parallelism_config.data_parallel_size
        activation_size = effective_batch * 1 * model.hidden_dim * bytes_per_param
        # Multiple activations in pipeline
        activation_size *= 4  # Rough multiplier for intermediate activations
        
        total_memory = model_size + kv_cache_size + activation_size
        
        return total_memory
    
    def _calculate_decode_step_memory_traffic(
        self,
        batch_size: int,
        context_length: int,
        parallelism_config: ParallelismConfig,
        dtype_override: Optional[str] = None
    ) -> float:
        """
        Calculate memory traffic for decode step.
        
        Key insight: Must read the entire KV cache (grows with context)
        With DSA: Only read top-K entries instead of full context
        """
        model = self.model
        bytes_per_param = self._get_bytes_per_param(dtype_override)
        
        # Read model weights (divided by both TP and PP, but NOT DP - weights are duplicated)
        model_size = model.total_parameters * bytes_per_param
        model_size /= (parallelism_config.tensor_parallel_size * 
                      parallelism_config.pipeline_parallel_size)
        
        # Read KV cache (divided by TP, PP, AND DP - each replica has its own KV cache subset)
        # DSA reduces the effective sequence length we need to read
        effective_sequence_length = context_length
        if model.attention_config.use_dsa and model.attention_config.dsa_top_k:
            effective_sequence_length = min(context_length, model.attention_config.dsa_top_k)
        
        # Each DP replica processes a subset of the batch
        effective_batch = batch_size // parallelism_config.data_parallel_size
        
        kv_cache_size = model.get_kv_cache_size(
            batch_size=effective_batch,  # Each replica only loads KV for its batch subset
            sequence_length=effective_sequence_length,  # DSA: read only top-K entries
            bytes_per_element=bytes_per_param
        )
        kv_cache_size /= (parallelism_config.tensor_parallel_size * 
                         parallelism_config.pipeline_parallel_size)
        
        # Linear attention state traffic (read + write fixed-size state per linear layer)
        # This is CONSTANT regardless of context_length — a key advantage
        linear_attn_traffic = 0.0
        if model.linear_attention_config is not None:
            num_linear_layers = model.get_num_linear_attention_layers()
            if num_linear_layers > 0:
                linear_attn_traffic = model.linear_attention_config.get_decode_state_traffic(
                    effective_batch, bytes_per_param
                ) * num_linear_layers
                linear_attn_traffic /= (parallelism_config.tensor_parallel_size * 
                                       parallelism_config.pipeline_parallel_size)
        
        # Read/write activations (small)
        activation_traffic = effective_batch * 1 * model.hidden_dim * bytes_per_param * 2  # read + write
        
        # Total traffic = weights + KV cache read + linear attention state + activation traffic
        total_traffic = model_size + kv_cache_size + linear_attn_traffic + activation_traffic
        
        return total_traffic
    
    def _calculate_decode_step_network_traffic(
        self,
        batch_size: int,
        context_length: int,
        parallelism_config: ParallelismConfig
    ) -> float:
        """Calculate network traffic for decode step"""
        model = self.model
        bytes_per_param = 2  # FP16
        
        network_traffic = 0.0
        effective_batch = batch_size // parallelism_config.data_parallel_size
        
        # Tensor Parallelism: All-reduce activations
        if parallelism_config.tensor_parallel_size > 1:
            # Each layer does all-reduce of (batch, 1, hidden)
            data_per_allreduce = effective_batch * 1 * model.hidden_dim * bytes_per_param
            
            # Two all-reduces per layer (after attention and FFN)
            num_layers = model.num_layers // parallelism_config.pipeline_parallel_size
            
            tp_size = parallelism_config.tensor_parallel_size
            all_reduce_factor = 2 * (tp_size - 1) / tp_size
            
            network_traffic += data_per_allreduce * num_layers * 2 * all_reduce_factor
        
        # Pipeline Parallelism: Send activations between stages
        if parallelism_config.pipeline_parallel_size > 1:
            activation_size = effective_batch * 1 * model.hidden_dim * bytes_per_param
            num_transfers = parallelism_config.pipeline_parallel_size - 1
            network_traffic += activation_size * num_transfers
        
        return network_traffic
    
    def calculate_num_kernel_launches(
        self,
        parallelism_config: ParallelismConfig
    ) -> int:
        """
        Calculate number of kernel launches for moderately optimized implementation
        
        Assumes:
        - Fused operations where easy (QKV projection, activation+gating)
        - Separate kernels for layer norms, attention ops, and FFN stages
        
        Per layer in moderately optimized implementation:
        - Pre-attention layer norm: 1 kernel
        - QKV projection (fused): 1 kernel
        - MLA compression/decompression (if used): +2 kernels
        - Attention score computation: 1 kernel
        - Softmax: 1 kernel (not easily fused with matmul)
        - Attention output (matmul with V): 1 kernel
        - Attention output projection: 1 kernel
        - Post-attention layer norm: 1 kernel
        - FFN up projection + activation (fused): 1 kernel
        - FFN gating (if used, fused with up): 0 additional
        - FFN down projection: 1 kernel
        
        Total per dense layer: ~9 kernels (11 with MLA)
        
        For MoE layers:
        - Router: 1 kernel
        - Expert selection/dispatch: 1 kernel
        - Per-expert computation: similar to FFN but may need more kernels for gathering
        - Expert combine: 1 kernel
        Total per MoE layer: ~12 kernels (14 with MLA)
        
        Args:
            parallelism_config: Parallelism configuration
            
        Returns:
            Total number of kernel launches
        """
        # Initial operations
        num_kernels = 0
        
        # Embedding lookup: 1 kernel
        num_kernels += 1
        
        # Per-layer kernels
        layers_per_gpu = self.model.num_layers // parallelism_config.pipeline_parallel_size
        
        # Determine per-layer kernel count based on layer types
        if self.model.layer_types is not None and self.model.linear_attention_config is not None:
            # Hybrid linear/full attention: count per-layer
            for layer_idx in range(layers_per_gpu):
                layer_type = self.model.get_layer_attention_type(layer_idx)
                if layer_type == LayerAttentionType.LINEAR_ATTENTION:
                    # Linear attention layer:
                    # Pre-attention norm: 1, Q/K/V projections: 1, Convolution: 1,
                    # State update: 1, Q @ state: 1, Output projection: 1,
                    # Post-attention norm: 1, FFN: 2 = ~8 kernels
                    layer_kernels = 8
                else:
                    # Standard/sliding attention layer
                    if self.model.is_moe and self.model.moe_config:
                        layer_kernels = 12
                    else:
                        layer_kernels = 9
                    if self.model.attention_config.use_mla:
                        layer_kernels += 2
                
                # MoE overhead for FFN (applies to all layer types)
                if self.model.ffn_layer_types is not None:
                    if self.model.ffn_layer_types[layer_idx] == FFNLayerType.MOE:
                        layer_kernels += 3  # Router + dispatch + combine
                
                num_kernels += layer_kernels
        else:
            if self.model.is_moe and self.model.moe_config:
                kernels_per_layer = 12
            else:
                kernels_per_layer = 9
            
            # Add MLA overhead if used
            if self.model.attention_config.use_mla:
                kernels_per_layer += 2  # Compression and decompression kernels
            
            num_kernels += layers_per_gpu * kernels_per_layer
        
        # Final layer norm: 1 kernel
        num_kernels += 1
        
        # LM head projection: 1 kernel
        num_kernels += 1
        
        return num_kernels
    
    def calculate_prefill_compute(
        self,
        batch_size: int,
        sequence_length: int,
        parallelism_config: ParallelismConfig
    ) -> float:
        """
        Calculate total FLOPs for prefill phase
        
        Prefill processes the entire prompt in one forward pass:
        - Attention: O(batch * seq^2 * hidden_dim)
        - FFN: O(batch * seq * hidden_dim * intermediate_size)
        
        Args:
            batch_size: Number of sequences in batch
            sequence_length: Input sequence length
            parallelism_config: Parallelism configuration
            
        Returns:
            Total FLOPs for prefill
        """
        B = batch_size
        L = sequence_length
        H = self.model.hidden_dim
        N = self.model.num_layers
        
        # Account for data parallelism - each DP rank processes subset of batch
        effective_batch = B // parallelism_config.data_parallel_size
        
        # Per-layer compute
        flops_per_layer = 0
        
        # 1. Attention compute
        num_attn_heads = self.model.attention_config.num_attention_heads
        num_kv_heads = self.model.attention_config.num_key_value_heads
        head_dim = self.model.attention_config.head_dim
        
        if self.model.attention_config.use_mla and self.model.attention_config.mla_kv_lora_rank:
            # MLA: Multi-head Latent Attention with compressed KV
            kv_lora_rank = self.model.attention_config.mla_kv_lora_rank
            
            # Down-project to latent space for KV
            # K: batch * seq * hidden -> batch * seq * kv_lora_rank
            flops_per_layer += 2 * effective_batch * L * H * kv_lora_rank
            # V: batch * seq * hidden -> batch * seq * kv_lora_rank
            flops_per_layer += 2 * effective_batch * L * H * kv_lora_rank
            
            # Q projection (standard or with lora)
            if self.model.attention_config.mla_q_lora_rank:
                # Q also uses low-rank projection
                q_lora_rank = self.model.attention_config.mla_q_lora_rank
                flops_per_layer += 2 * effective_batch * L * H * q_lora_rank
                # Up-project Q for attention
                flops_per_layer += 2 * effective_batch * L * q_lora_rank * (num_attn_heads * head_dim)
            else:
                # Standard Q projection
                flops_per_layer += 2 * effective_batch * L * H * (num_attn_heads * head_dim)
            
            # Up-project latent KV to full dimension for attention computation
            # This happens during attention computation
            flops_per_layer += 2 * effective_batch * L * kv_lora_rank * (num_kv_heads * head_dim) * 2  # K and V
            
            # Attention scores and output (same as standard)
            flops_per_layer += 2 * effective_batch * num_attn_heads * L * L * head_dim  # Q @ K^T
            flops_per_layer += 2 * effective_batch * num_attn_heads * L * L * head_dim  # Attn @ V
            
            # Output projection
            flops_per_layer += 2 * effective_batch * L * (num_attn_heads * head_dim) * H
        else:
            # Standard attention
            # Q projection
            flops_per_layer += 2 * effective_batch * L * H * (num_attn_heads * head_dim)
            # K, V projections (may be smaller for GQA/MQA)
            flops_per_layer += 2 * 2 * effective_batch * L * H * (num_kv_heads * head_dim)
            
            # Attention scores: Q @ K^T -> (batch * heads * seq * seq)
            flops_per_layer += 2 * effective_batch * num_attn_heads * L * L * head_dim
            
            # Attention @ V -> (batch * heads * seq * head_dim)
            flops_per_layer += 2 * effective_batch * num_attn_heads * L * L * head_dim
            
            # Output projection
            flops_per_layer += 2 * effective_batch * L * (num_attn_heads * head_dim) * H
        
        # 2. FFN compute
        if self.model.is_moe and self.model.moe_config:
            # MoE: only active experts contribute
            # Gate: batch * seq * hidden * num_experts (routing)
            flops_per_layer += 2 * effective_batch * L * H * self.model.moe_config.num_experts
            
            # Active expert computation
            num_active_experts = self.model.moe_config.num_experts_per_token
            intermediate = self.model.ffn_config.intermediate_size
            
            # Up projection (per active expert)
            flops_per_layer += 2 * effective_batch * L * H * intermediate * num_active_experts
            if self.model.ffn_config.use_gating:
                flops_per_layer += 2 * effective_batch * L * H * intermediate * num_active_experts
            
            # Down projection (per active expert)
            flops_per_layer += 2 * effective_batch * L * intermediate * H * num_active_experts
        else:
            # Dense FFN
            intermediate = self.model.ffn_config.intermediate_size
            
            # Up projection
            flops_per_layer += 2 * effective_batch * L * H * intermediate
            if self.model.ffn_config.use_gating:
                # Gated activation (e.g., SwiGLU) has two up projections
                flops_per_layer += 2 * effective_batch * L * H * intermediate
            
            # Down projection
            flops_per_layer += 2 * effective_batch * L * intermediate * H
        
        # 3. Layer norm (negligible compared to matmuls, but include for completeness)
        flops_per_layer += 2 * effective_batch * L * H  # Two layer norms per layer
        
        # Total across layers per pipeline stage
        # Each pipeline stage processes num_layers / pipeline_parallel_size layers
        layers_per_stage = N // parallelism_config.pipeline_parallel_size
        total_flops = flops_per_layer * layers_per_stage
        
        # Add embedding and final layer norm
        total_flops += 2 * effective_batch * L * H  # Embedding lookup (approximate)
        total_flops += 2 * effective_batch * L * H  # Final layer norm
        
        # LM head (output projection)
        total_flops += 2 * effective_batch * L * H * self.model.vocab_size
        
        return total_flops
    
    def calculate_prefill_compute_breakdown(
        self,
        batch_size: int,
        sequence_length: int,
        parallelism_config: ParallelismConfig
    ) -> Dict[str, float]:
        """
        Calculate FLOPs breakdown for prefill phase
        
        Returns:
            Dictionary with keys: 'attention', 'ffn', 'other', 'total'
        """
        B = batch_size
        L = sequence_length
        H = self.model.hidden_dim
        N = self.model.num_layers
        
        effective_batch = B // parallelism_config.data_parallel_size
        
        attention_flops = 0.0
        ffn_flops = 0.0
        mamba_flops = 0.0  # Track Mamba FLOPs separately
        other_flops = 0.0
        
        # Per-layer compute
        for layer_idx in range(N):
            # Check if this is a Mamba layer in hybrid architecture
            is_mamba_layer = False
            if self.model.hybrid_layer_types is not None:
                is_mamba_layer = self.model.hybrid_layer_types[layer_idx] == HybridLayerType.MAMBA
            
            if is_mamba_layer and self.model.mamba_config is not None:
                # Mamba-2 layer: use Mamba-specific FLOPs calculation
                layer_mamba_flops = self.model.mamba_config.get_prefill_flops(L, H)
                mamba_flops += layer_mamba_flops * effective_batch
            else:
                # Determine layer attention type
                layer_attn_type = self.model.get_layer_attention_type(layer_idx)
                
                if layer_attn_type == LayerAttentionType.LINEAR_ATTENTION and self.model.linear_attention_config is not None:
                    # Linear attention layer: O(L) instead of O(L^2)
                    # Uses phi(K)^T @ phi(V) state formulation
                    lac = self.model.linear_attention_config
                    num_attn_heads = self.model.attention_config.num_attention_heads
                    layer_attn_flops = lac.get_prefill_flops(L, effective_batch, num_attn_heads, H)
                    attention_flops += layer_attn_flops
                else:
                    # Standard or sliding window attention layer
                    num_attn_heads = self.model.attention_config.num_attention_heads
                    num_kv_heads = self.model.attention_config.num_key_value_heads
                    head_dim = self.model.attention_config.head_dim
                    
                    # Determine effective sequence length for attention computation
                    effective_attn_len = L  # Default: full sequence
                    if layer_attn_type == LayerAttentionType.SLIDING_ATTENTION:
                        window_size = self.model.attention_config.sliding_window_size or L
                        effective_attn_len = min(L, window_size)
                
                    layer_attn_flops = 0.0
                    if self.model.attention_config.use_mla and self.model.attention_config.mla_kv_lora_rank:
                        kv_lora_rank = self.model.attention_config.mla_kv_lora_rank
                        layer_attn_flops += 2 * effective_batch * L * H * kv_lora_rank  # K
                        layer_attn_flops += 2 * effective_batch * L * H * kv_lora_rank  # V
                        
                        if self.model.attention_config.mla_q_lora_rank:
                            q_lora_rank = self.model.attention_config.mla_q_lora_rank
                            layer_attn_flops += 2 * effective_batch * L * H * q_lora_rank
                            layer_attn_flops += 2 * effective_batch * L * q_lora_rank * (num_attn_heads * head_dim)
                        else:
                            layer_attn_flops += 2 * effective_batch * L * H * (num_attn_heads * head_dim)
                        
                        layer_attn_flops += 2 * effective_batch * L * kv_lora_rank * (num_kv_heads * head_dim) * 2
                        # Use effective_attn_len for attention matrix operations
                        layer_attn_flops += 2 * effective_batch * num_attn_heads * L * effective_attn_len * head_dim
                        layer_attn_flops += 2 * effective_batch * num_attn_heads * L * effective_attn_len * head_dim
                        layer_attn_flops += 2 * effective_batch * L * (num_attn_heads * head_dim) * H
                    else:
                        layer_attn_flops += 2 * effective_batch * L * H * (num_attn_heads * head_dim)
                        layer_attn_flops += 2 * 2 * effective_batch * L * H * (num_kv_heads * head_dim)
                        # Use effective_attn_len for attention matrix operations
                        layer_attn_flops += 2 * effective_batch * num_attn_heads * L * effective_attn_len * head_dim
                        layer_attn_flops += 2 * effective_batch * num_attn_heads * L * effective_attn_len * head_dim
                        layer_attn_flops += 2 * effective_batch * L * (num_attn_heads * head_dim) * H
                    
                    attention_flops += layer_attn_flops
            
            # 2. FFN compute - check if this layer is Dense or MoE
            # Note: In hybrid Mamba/Attention architectures, MLP layers might be separate
            # For now, all non-Mamba layers have FFN (standard transformer behavior)
            # Mamba layers may also have FFN depending on architecture
            layer_ffn_flops = 0.0
            
            # Determine if this specific layer is MoE or Dense
            is_moe_layer = False
            if self.model.ffn_layer_types is not None:
                # Use per-layer FFN type specification
                from llm_architecture import FFNLayerType
                is_moe_layer = self.model.ffn_layer_types[layer_idx] == FFNLayerType.MOE
            elif self.model.is_moe:
                # All layers are MoE if no per-layer specification
                is_moe_layer = True
            
            if is_moe_layer and self.model.moe_config:
                # MoE layer: router + active experts
                layer_ffn_flops += 2 * effective_batch * L * H * self.model.moe_config.num_experts  # Router
                num_active_experts = self.model.moe_config.num_experts_per_token
                intermediate = self.model.ffn_config.intermediate_size  # Per-expert size
                layer_ffn_flops += 2 * effective_batch * L * H * intermediate * num_active_experts
                if self.model.ffn_config.use_gating:
                    layer_ffn_flops += 2 * effective_batch * L * H * intermediate * num_active_experts
                layer_ffn_flops += 2 * effective_batch * L * intermediate * H * num_active_experts
            else:
                # Dense layer: use dense_intermediate_size
                intermediate = self.model.get_dense_intermediate_size()
                layer_ffn_flops += 2 * effective_batch * L * H * intermediate
                if self.model.ffn_config.use_gating:
                    layer_ffn_flops += 2 * effective_batch * L * H * intermediate
                layer_ffn_flops += 2 * effective_batch * L * intermediate * H
            
            ffn_flops += layer_ffn_flops
            
            # 3. Layer norm
            other_flops += 2 * effective_batch * L * H  # Two layer norms per layer
        
        # Add embedding, final layer norm, and LM head
        other_flops += 2 * effective_batch * L * H  # Embedding
        other_flops += 2 * effective_batch * L * H  # Final layer norm
        other_flops += 2 * effective_batch * L * H * self.model.vocab_size  # LM head
        
        return {
            'attention': attention_flops,
            'mamba': mamba_flops,  # Mamba SSM FLOPs (0 for non-hybrid models)
            'ffn': ffn_flops,
            'other': other_flops,
            'total': attention_flops + mamba_flops + ffn_flops + other_flops
        }
    
    def calculate_prefill_memory(
        self,
        batch_size: int,
        sequence_length: int,
        parallelism_config: ParallelismConfig,
        dtype_override: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Calculate memory requirements for prefill
        
        Args:
            batch_size: Number of sequences
            sequence_length: Length of input sequences
            parallelism_config: Parallelism configuration
            dtype_override: Optional dtype override ('int4', 'int8', 'float16', 'bfloat16', 'float32')
        
        Returns:
            Dictionary with memory breakdown in bytes
        """
        # Determine bytes per parameter based on dtype
        bytes_per_param = self._get_bytes_per_param(dtype_override)
        
        # Model weights per GPU
        # NOTE: Use total_parameters (not active_parameters) because we need to store ALL experts in memory,
        # even though only some are active per token. active_parameters is only for compute/bandwidth.
        if parallelism_config.parallelism_type == ParallelismType.DATA_PARALLEL:
            # Each GPU has full model
            model_memory = self.model.total_parameters * bytes_per_param
        elif parallelism_config.parallelism_type in [
            ParallelismType.TENSOR_PARALLEL,
            ParallelismType.TENSOR_PIPELINE,
            ParallelismType.TENSOR_DATA
        ]:
            # Model split across tensor parallel GPUs
            model_memory = (self.model.total_parameters * bytes_per_param) / parallelism_config.tensor_parallel_size
        elif parallelism_config.parallelism_type == ParallelismType.PIPELINE_PARALLEL:
            # Model split across pipeline stages
            model_memory = (self.model.total_parameters * bytes_per_param) / parallelism_config.pipeline_parallel_size
        elif parallelism_config.parallelism_type == ParallelismType.FULL_3D:
            # Split across both tensor and pipeline
            model_memory = (self.model.total_parameters * bytes_per_param) / (
                parallelism_config.tensor_parallel_size * parallelism_config.pipeline_parallel_size
            )
        else:  # NONE
            model_memory = self.model.total_parameters * bytes_per_param
        
        # KV cache per GPU
        # Use the model's KV cache calculation which handles MLA compression
        effective_batch = batch_size // parallelism_config.data_parallel_size
        layers_per_gpu = self.model.num_layers // parallelism_config.pipeline_parallel_size
        
        if self.model.attention_config.use_mla and self.model.attention_config.mla_kv_lora_rank:
            # MLA: Compressed KV cache
            kv_cache_memory = (
                2 *  # K and V
                effective_batch *
                sequence_length *
                self.model.attention_config.mla_kv_lora_rank *  # Compressed dimension
                layers_per_gpu *
                bytes_per_param
            )
            # Note: Tensor parallelism doesn't split MLA latent dimension
        else:
            # Standard KV cache with potential tensor parallelism splitting
            kv_heads_per_gpu = self.model.attention_config.num_key_value_heads // parallelism_config.tensor_parallel_size
            
            kv_cache_memory = (
                2 *  # K and V
                effective_batch *
                kv_heads_per_gpu *
                sequence_length *
                self.model.attention_config.head_dim *
                layers_per_gpu *
                bytes_per_param
            )
        
        # Activations (rough estimate)
        # Activations are temporary and depend on implementation
        # Main activations: attention outputs, FFN intermediates
        # Estimate: batch * seq * hidden * num_layers * multiplier
        effective_batch = batch_size // parallelism_config.data_parallel_size
        layers_per_gpu = self.model.num_layers // parallelism_config.pipeline_parallel_size
        
        # Attention activations
        attn_activations = effective_batch * sequence_length * self.model.hidden_dim * bytes_per_param * 4
        
        # FFN activations (intermediate size is larger)
        ffn_activations = effective_batch * sequence_length * self.model.ffn_config.intermediate_size * bytes_per_param * 2
        
        # Peak activation memory (not all layers at once, but needs some buffer)
        activation_memory = (attn_activations + ffn_activations) * 2  # Buffer for 2 layers
        
        return {
            "model_weights": model_memory,
            "kv_cache": kv_cache_memory,
            "activations": activation_memory,
            "total": model_memory + kv_cache_memory + activation_memory
        }
    
    def calculate_prefill_memory_bandwidth(
        self,
        batch_size: int,
        sequence_length: int,
        time_to_first_token: float,
        parallelism_config: ParallelismConfig,
        dtype_override: Optional[str] = None
    ) -> float:
        """
        Calculate required memory bandwidth for prefill
        
        Memory bandwidth = data moved / time
        
        For prefill:
        - Read all model weights at least once
        - Read/write activations multiple times
        - Write KV cache once
        
        Args:
            batch_size: Batch size
            sequence_length: Sequence length
            time_to_first_token: Target TTFT in seconds
            parallelism_config: Parallelism config
            dtype_override: Optional dtype override ('int4', 'int8', 'float16', 'bfloat16', 'float32')
            
        Returns:
            Required memory bandwidth in bytes/sec
        """
        memory = self.calculate_prefill_memory(batch_size, sequence_length, parallelism_config, dtype_override)
        
        bytes_per_param = self._get_bytes_per_param(dtype_override)
        
        # Data movement:
        # 1. Read model weights once per forward pass
        weight_reads = memory["model_weights"]
        
        # 2. Write KV cache
        kv_writes = memory["kv_cache"]
        
        # 3. Activations - read and write multiple times (rough estimate: 4x)
        # This is a simplification; actual implementation matters
        activation_traffic = memory["activations"] * 4
        
        total_data_movement = weight_reads + kv_writes + activation_traffic
        
        # Required bandwidth
        required_bandwidth = total_data_movement / time_to_first_token
        
        return required_bandwidth
    
    def calculate_prefill_network_bandwidth(
        self,
        batch_size: int,
        sequence_length: int,
        time_to_first_token: float,
        parallelism_config: ParallelismConfig
    ) -> float:
        """
        Calculate required network bandwidth for communication
        
        Network communication happens for:
        - Tensor parallelism: All-reduce after each layer
        - Pipeline parallelism: Send activations between stages
        - Data parallelism: Usually only for training (gradients)
        
        Args:
            batch_size: Batch size
            sequence_length: Sequence length
            time_to_first_token: Target TTFT in seconds
            parallelism_config: Parallelism config
            
        Returns:
            Required network bandwidth in bytes/sec
        """
        if parallelism_config.parallelism_type == ParallelismType.NONE:
            return 0.0
        
        bytes_per_param = 2 if self.model.dtype in ["float16", "bfloat16"] else 4
        effective_batch = batch_size // parallelism_config.data_parallel_size
        
        network_traffic = 0.0
        
        # Tensor Parallelism: All-reduce activations after each layer
        if parallelism_config.tensor_parallel_size > 1:
            # Each layer does an all-reduce of batch * seq * hidden
            data_per_allreduce = effective_batch * sequence_length * self.model.hidden_dim * bytes_per_param
            
            # Two all-reduces per layer (after attention and FFN)
            num_layers = self.model.num_layers // parallelism_config.pipeline_parallel_size
            
            # All-reduce requires 2(N-1)/N * data where N is TP size
            # For ring all-reduce
            tp_size = parallelism_config.tensor_parallel_size
            all_reduce_factor = 2 * (tp_size - 1) / tp_size
            
            network_traffic += data_per_allreduce * num_layers * 2 * all_reduce_factor
        
        # Pipeline Parallelism: Send activations between stages
        if parallelism_config.pipeline_parallel_size > 1:
            # Each pipeline stage sends activations to next stage
            # Activation size: batch * seq * hidden
            activation_size = effective_batch * sequence_length * self.model.hidden_dim * bytes_per_param
            
            # Number of pipeline stages - 1 (each sends to next)
            num_transfers = parallelism_config.pipeline_parallel_size - 1
            
            # For prefill, simplified: one forward pass through pipeline
            network_traffic += activation_size * num_transfers
        
        # Data Parallelism: No communication during inference (only in training)
        
        # Calculate bandwidth
        required_bandwidth = network_traffic / time_to_first_token
        
        return required_bandwidth
    
    def calculate_prefill_resources(
        self,
        batch_size: int,
        sequence_length: int,
        time_to_first_token: float,
        num_gpus: int,
        parallelism_config: ParallelismConfig,
        dtype_override: Optional[str] = None
    ) -> PrefillResources:
        """
        Calculate all resource requirements for prefill phase
        
        Args:
            batch_size: Number of sequences in batch
            sequence_length: Length of input sequences
            time_to_first_token: Target time to first token (seconds)
            num_gpus: Total number of GPUs
            parallelism_config: Parallelism configuration
            dtype_override: Optional dtype override ('int4', 'int8', 'float16', 'bfloat16', 'float32')
            
        Returns:
            PrefillResources with all metrics
        """
        # Validate GPU count matches config
        if num_gpus != parallelism_config.total_gpus:
            raise ValueError(
                f"num_gpus ({num_gpus}) must match parallelism config "
                f"({parallelism_config.total_gpus})"
            )
        
        # Calculate kernel launch overhead
        num_kernel_launches = self.calculate_num_kernel_launches(parallelism_config)
        kernel_launch_overhead = num_kernel_launches * self.model.kernel_launch_latency
        effective_compute_time = time_to_first_token - kernel_launch_overhead
        
        if effective_compute_time <= 0:
            raise ValueError(
                f"Kernel launch overhead ({kernel_launch_overhead*1000:.2f}ms) exceeds "
                f"target TTFT ({time_to_first_token*1000:.2f}ms). "
                f"Cannot meet timing requirements with {num_kernel_launches} kernel launches."
            )
        
        # Calculate compute (based on effective compute time, not total TTFT)
        total_flops = self.calculate_prefill_compute(batch_size, sequence_length, parallelism_config)
        flops_per_sec = total_flops / effective_compute_time
        
        # Calculate memory
        memory = self.calculate_prefill_memory(batch_size, sequence_length, parallelism_config, dtype_override)
        
        # Calculate memory bandwidth (based on effective compute time)
        memory_bandwidth = self.calculate_prefill_memory_bandwidth(
            batch_size, sequence_length, effective_compute_time, parallelism_config, dtype_override
        )
        
        # Calculate network bandwidth (based on effective compute time)
        network_bandwidth = self.calculate_prefill_network_bandwidth(
            batch_size, sequence_length, effective_compute_time, parallelism_config
        )
        
        # Calculate arithmetic intensity
        # AI = FLOPs / Bytes moved
        arithmetic_intensity = total_flops / (memory_bandwidth * effective_compute_time)
        
        # Determine if compute-bound or memory-bound
        # Rough heuristic: AI > 100 is typically compute-bound for modern GPUs
        compute_bound = arithmetic_intensity > 100
        
        return PrefillResources(
            memory_per_gpu=memory["total"],
            memory_model_weights=memory["model_weights"],
            memory_kv_cache=memory["kv_cache"],
            memory_activations=memory["activations"],
            memory_bandwidth_per_gpu=memory_bandwidth,
            network_bandwidth_per_gpu=network_bandwidth,
            compute_per_gpu=total_flops,
            compute_flops_per_sec=flops_per_sec,
            time_to_first_token=time_to_first_token,
            num_kernel_launches=num_kernel_launches,
            kernel_launch_overhead=kernel_launch_overhead,
            effective_compute_time=effective_compute_time,
            arithmetic_intensity=arithmetic_intensity,
            compute_bound=compute_bound
        )

