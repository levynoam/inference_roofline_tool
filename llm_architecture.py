"""
LLM Architecture Data Structures
Flexible modeling of modern LLM architectures for performance analysis
"""

from dataclasses import dataclass, field
from typing import Optional, Literal, Dict, Any, List
from enum import Enum


class ArchitectureType(Enum):
    """Type of transformer architecture"""
    DECODER_ONLY = "decoder_only"
    ENCODER_DECODER = "encoder_decoder"
    ENCODER_ONLY = "encoder_only"


class AttentionType(Enum):
    """Attention mechanism variants"""
    MULTI_HEAD = "multi_head_attention"  # Standard MHA
    GROUPED_QUERY = "grouped_query_attention"  # GQA
    MULTI_QUERY = "multi_query_attention"  # MQA
    SLIDING_WINDOW = "sliding_window_attention"  # Local attention


class LayerAttentionType(Enum):
    """Per-layer attention type for hybrid architectures"""
    FULL_ATTENTION = "full_attention"  # Full context attention
    SLIDING_ATTENTION = "sliding_attention"  # Sliding window / local attention
    LINEAR_ATTENTION = "linear_attention"  # Linear attention (phi(K)^T @ phi(V))


class ActivationType(Enum):
    """Activation functions"""
    GELU = "gelu"
    SILU = "silu"  # SwiGLU uses SiLU
    RELU = "relu"
    RELU2 = "relu2"  # Squared ReLU: ReLU(x)^2
    SWIGLU = "swiglu"
    GEGLU = "geglu"


class NormalizationType(Enum):
    """Normalization layer types"""
    LAYER_NORM = "layer_norm"
    RMS_NORM = "rms_norm"
    GROUP_NORM = "group_norm"


class PositionEncodingType(Enum):
    """Position encoding methods"""
    ABSOLUTE = "absolute"
    ROTARY = "rotary"  # RoPE
    ALIBI = "alibi"
    RELATIVE = "relative"
    NONE = "none"


@dataclass
class LinearAttentionConfig:
    """Configuration for linear attention layers.
    
    Linear attention replaces softmax(Q @ K^T) @ V with Q @ (phi(K)^T @ phi(V)),
    where phi() is an activation function. This changes the compute complexity
    from O(L^2) to O(L) and uses a fixed-size state instead of growing KV cache.
    
    The state S = sum_t phi(k_t) @ phi(v_t)^T has shape (key_head_dim, value_head_dim)
    per head group, and is constant regardless of sequence length.
    """
    num_key_heads: int  # Number of key heads for linear attention
    key_head_dim: int  # Dimension per key head
    num_value_heads: int  # Number of value heads for linear attention
    value_head_dim: int  # Dimension per value head
    conv_kernel_dim: int = 4  # Short convolution kernel size
    
    @property
    def total_key_dim(self) -> int:
        """Total key dimension = num_key_heads * key_head_dim"""
        return self.num_key_heads * self.key_head_dim
    
    @property
    def total_value_dim(self) -> int:
        """Total value dimension = num_value_heads * value_head_dim"""
        return self.num_value_heads * self.value_head_dim
    
    def get_state_size_bytes(self, batch_size: int, bytes_per_element: int = 2) -> int:
        """
        Calculate linear attention state size per layer in bytes.
        
        State shape per head group: (key_head_dim, value_per_group)
        where value_per_group = (num_value_heads / num_key_heads) * value_head_dim
        Total state: batch * num_key_heads * key_head_dim * value_per_group
        = batch * key_head_dim * num_value_heads * value_head_dim
        
        This is CONSTANT regardless of sequence length (unlike KV cache).
        """
        return (batch_size * self.key_head_dim * self.num_value_heads 
                * self.value_head_dim * bytes_per_element)
    
    def get_prefill_flops(self, sequence_length: int, batch_size: int,
                          num_query_heads: int, hidden_dim: int) -> int:
        """
        Calculate FLOPs for linear attention prefill (one layer).
        
        Linear attention: O = Q @ (phi(K)^T @ phi(V))
        
        Components:
        1. Q projection: H -> num_query_heads * key_head_dim
        2. K projection: H -> num_key_heads * key_head_dim  
        3. V projection: H -> num_value_heads * value_head_dim
        4. Convolution on K/V (short conv with kernel_dim)
        5. State build: phi(K)^T @ phi(V) over L tokens - O(L)
        6. Query output: Q @ state - O(L)
        7. Output projection: total_value_dim -> H
        
        Args:
            sequence_length: L - number of tokens
            batch_size: B - batch size
            num_query_heads: Number of query heads (from main attention config)
            hidden_dim: H - model hidden dimension
            
        Returns:
            Total FLOPs for one linear attention layer prefill
        """
        B = batch_size
        L = sequence_length
        H = hidden_dim
        d_k = self.key_head_dim
        d_v = self.value_head_dim
        n_q = num_query_heads
        n_k = self.num_key_heads
        n_v = self.num_value_heads
        v_per_group = n_v // n_k if n_k > 0 else n_v
        
        flops = 0
        
        # Q projection: (B, L, H) @ (H, n_q * d_k)
        flops += 2 * B * L * H * (n_q * d_k)
        
        # K projection: (B, L, H) @ (H, n_k * d_k)
        flops += 2 * B * L * H * (n_k * d_k)
        
        # V projection: (B, L, H) @ (H, n_v * d_v)
        flops += 2 * B * L * H * (n_v * d_v)
        
        # Short convolution on K and V: approx 2 * B * L * dim * kernel
        conv_flops = 2 * B * L * (n_k * d_k) * self.conv_kernel_dim
        conv_flops += 2 * B * L * (n_v * d_v) * self.conv_kernel_dim
        flops += conv_flops
        
        # State build: phi(K)^T @ phi(V) accumulated over L tokens
        # Per head group: (d_k, L) @ (L, v_per_group * d_v) across L tokens
        # = 2 * L * d_k * v_per_group * d_v per group
        # Total across n_k groups: 2 * B * L * n_k * d_k * v_per_group * d_v
        # = 2 * B * L * d_k * n_v * d_v
        flops += 2 * B * L * d_k * n_v * d_v
        
        # Q @ state: per Q head (L, d_k) @ (d_k, v_per_group * d_v)
        # Total: 2 * B * n_q * L * d_k * v_per_group * d_v
        flops += 2 * B * n_q * L * d_k * v_per_group * d_v
        
        # Output projection: (B, L, n_v * d_v) @ (n_v * d_v, H)
        flops += 2 * B * L * (n_v * d_v) * H
        
        return flops
    
    def get_decode_flops(self, batch_size: int, num_query_heads: int, 
                         hidden_dim: int) -> int:
        """
        Calculate FLOPs for linear attention decode (one layer, one token).
        
        Key advantage: CONSTANT cost regardless of context length.
        - State update: S += phi(k_new) @ phi(v_new)^T
        - Query: o = q @ S
        
        Args:
            batch_size: B - batch size
            num_query_heads: Number of query heads
            hidden_dim: H - model hidden dimension
            
        Returns:
            FLOPs per decode token for one linear attention layer
        """
        B = batch_size
        H = hidden_dim
        d_k = self.key_head_dim
        d_v = self.value_head_dim
        n_q = num_query_heads
        n_k = self.num_key_heads
        n_v = self.num_value_heads
        v_per_group = n_v // n_k if n_k > 0 else n_v
        
        flops = 0
        
        # Q projection: (B, 1, H) @ (H, n_q * d_k)
        flops += 2 * B * H * (n_q * d_k)
        
        # K projection: (B, 1, H) @ (H, n_k * d_k)
        flops += 2 * B * H * (n_k * d_k)
        
        # V projection: (B, 1, H) @ (H, n_v * d_v)
        flops += 2 * B * H * (n_v * d_v)
        
        # Short convolution (1 token, but uses kernel_dim previous tokens)
        conv_flops = 2 * B * (n_k * d_k) * self.conv_kernel_dim
        conv_flops += 2 * B * (n_v * d_v) * self.conv_kernel_dim
        flops += conv_flops
        
        # State update: S += phi(k) @ phi(v)^T (outer product)
        # Per group: d_k * v_per_group * d_v
        # Total: 2 * B * n_k * d_k * v_per_group * d_v = 2 * B * d_k * n_v * d_v
        flops += 2 * B * d_k * n_v * d_v
        
        # Q @ state: per Q head (d_k,) @ (d_k, v_per_group * d_v)
        # Total: 2 * B * n_q * d_k * v_per_group * d_v
        flops += 2 * B * n_q * d_k * v_per_group * d_v
        
        # Output projection: (B, 1, n_v * d_v) @ (n_v * d_v, H)
        flops += 2 * B * (n_v * d_v) * H
        
        return flops
    
    def get_decode_state_traffic(self, batch_size: int, bytes_per_element: int = 2) -> int:
        """
        Memory traffic for reading/writing state during decode.
        
        Must read the full state and write updated state.
        State size = batch * key_head_dim * num_value_heads * value_head_dim
        Traffic = 2x state size (read + write)
        """
        state_size = self.get_state_size_bytes(batch_size, bytes_per_element)
        return 2 * state_size  # Read + write


@dataclass
class AttentionConfig:
    """Configuration for attention mechanism"""
    num_attention_heads: int
    num_key_value_heads: Optional[int] = None  # For GQA/MQA, None = same as attention heads
    attention_type: AttentionType = AttentionType.MULTI_HEAD
    head_dim: Optional[int] = None  # If None, calculated as hidden_dim / num_heads
    sliding_window_size: Optional[int] = None  # For sliding window attention
    attention_dropout: float = 0.0
    attention_bias: bool = False
    
    # Multi-head Latent Attention (MLA) configuration
    use_mla: bool = False  # Enable MLA for compressed KV cache
    mla_kv_lora_rank: Optional[int] = None  # Low-rank dimension for KV compression
    mla_q_lora_rank: Optional[int] = None  # Low-rank dimension for Q (if used)
    
    # DeepSeek Sparse Attention (DSA) configuration
    use_sparse_attention: bool = False  # Enable sparse attention patterns
    sparse_block_size: Optional[int] = None  # Block size for sparse attention
    sparse_local_blocks: Optional[int] = None  # Number of local blocks to attend to
    sparse_global_blocks: Optional[int] = None  # Number of global blocks to attend to
    
    # Dynamic Sparse Attention (DSA) - top-K selection
    use_dsa: bool = False  # Enable Dynamic Sparse Attention (top-K KV selection)
    dsa_q_indexer_dim: Optional[int] = None  # Query indexer dimension (e.g., 128)
    dsa_k_indexer_dim: Optional[int] = None  # Key indexer dimension (e.g., 64)
    dsa_top_k: Optional[int] = None  # Number of top KV pairs to select (e.g., 2048)


@dataclass
class FFNConfig:
    """Configuration for Feed-Forward Network"""
    intermediate_size: int
    activation: ActivationType = ActivationType.GELU
    ffn_dropout: float = 0.0
    use_gating: bool = False  # True for SwiGLU, GeGLU, etc.
    ffn_bias: bool = False
    # For interleaved dense/MoE architectures (e.g., Llama-4):
    # - intermediate_size is used for MoE expert FFN
    # - dense_intermediate_size is used for dense FFN layers
    dense_intermediate_size: Optional[int] = None  # If None, uses intermediate_size


@dataclass
class MoEConfig:
    """Configuration for Mixture of Experts"""
    num_experts: int
    num_experts_per_token: int  # Top-K experts activated
    expert_capacity_factor: float = 1.0
    router_type: Literal["top_k", "switch", "expert_choice"] = "top_k"
    router_jitter: float = 0.0
    router_z_loss_coef: float = 0.0
    shared_expert: bool = False  # Some models have shared experts


class HybridLayerType(Enum):
    """Layer types for hybrid Mamba/Attention architectures"""
    ATTENTION = "attention"  # Standard attention layer (denoted as * in patterns)
    MAMBA = "mamba"  # Mamba-2 SSM layer (denoted as M in patterns)
    MLP = "mlp"  # Pure MLP/FFN layer (denoted as E in patterns)

    # Sublayer types for architectures that decouple SSM, attention, and FFN into
    # separate "sublayers" (e.g., Nemotron-3-Super-120B where each pattern character
    # is an independent sublayer with no other component):
    MAMBA_ONLY = "mamba_only"      # Mamba SSM with NO FFN (Nemotron-Super M sublayer)
    LATENT_MOE = "latent_moe"      # Latent MoE FFN only, no attention/Mamba (Nemotron-Super E sublayer)
    ATTENTION_ONLY = "attention_only"  # Attention only, no FFN (Nemotron-Super * sublayer)


class FFNLayerType(Enum):
    """FFN layer types for dense/MoE interleaved architectures"""
    DENSE = "dense"  # Standard dense FFN layer
    MOE = "moe"  # Mixture of Experts FFN layer


@dataclass
class Mamba2Config:
    """Configuration for Mamba-2 SSM layers"""
    num_heads: int  # H: number of heads in Mamba-2
    head_dim: int  # d_head: dimension per head
    state_size: int  # N: SSM state dimension (e.g., 128)
    conv_kernel: int = 4  # Convolution kernel size
    chunk_size: int = 128  # Scan chunk size for chunked operations
    expand: int = 2  # Expansion factor for inner dimension
    
    @property
    def d_inner(self) -> int:
        """Inner dimension = H * d_head"""
        return self.num_heads * self.head_dim
    
    def get_state_size_bytes(self, batch_size: int, bytes_per_element: int = 2) -> int:
        """
        Calculate Mamba state size in bytes per layer.
        State shape: (batch, H, d_head, N)
        """
        return batch_size * self.num_heads * self.head_dim * self.state_size * bytes_per_element
    
    def get_prefill_flops(self, sequence_length: int, d_model: int) -> int:
        """
        Calculate FLOPs for Mamba-2 prefill.
        
        Args:
            sequence_length: T - number of tokens
            d_model: Model hidden dimension
            
        Returns:
            Total FLOPs for prefill
        """
        T = sequence_length
        H = self.num_heads
        d_head = self.head_dim
        N = self.state_size
        d_inner = self.d_inner
        d_proj = 2 * d_inner + 2 * H * N + H
        
        flops_in_proj = 2 * T * d_model * d_proj
        flops_out_proj = 2 * T * d_inner * d_model
        flops_ssm = T * (6 * H * d_head * N + 2 * H * d_head)
        
        return flops_in_proj + flops_ssm + flops_out_proj
    
    def get_decode_flops(self, d_model: int) -> int:
        """
        Calculate FLOPs for Mamba-2 decode (per token).
        
        Args:
            d_model: Model hidden dimension
            
        Returns:
            FLOPs per decode token
        """
        H = self.num_heads
        d_head = self.head_dim
        N = self.state_size
        d_inner = self.d_inner
        d_proj = 2 * d_inner + 2 * H * N + H
        
        flops_in_proj = 2 * d_model * d_proj
        flops_out_proj = 2 * d_inner * d_model
        flops_ssm = 6 * H * d_head * N + 2 * H * d_head
        
        return flops_in_proj + flops_ssm + flops_out_proj
    
    def get_prefill_kernel_launches(self, sequence_length: int) -> int:
        """
        Calculate kernel launches for Mamba-2 prefill.
        
        Args:
            sequence_length: T - number of tokens
            
        Returns:
            Number of kernel launches
        """
        import math
        return 3 + math.ceil(sequence_length / self.chunk_size)
    
    def get_decode_kernel_launches(self) -> int:
        """Kernel launches per decode token"""
        return 4


@dataclass
class LatentMoEConfig:
    """Configuration for Latent Mixture of Experts (Nemotron-3-Super style).

    Unlike standard MoE where experts work at the full hidden dimension, in Latent MoE
    the routed experts operate in a reduced latent space:
      - Input is projected from hidden_dim (H) down to latent_size (L)
      - Each routed expert computes: L → expert_intermediate → L
      - Results are aggregated and projected back up from L to H

    The shared expert(s) still work at the full hidden dimension H.

    This design dramatically reduces the per-expert parameter count, enabling many more
    experts (e.g., 512) while keeping memory bandwidth manageable.
    """
    num_experts: int                          # Total routed experts (e.g., 512)
    num_experts_per_token: int                # Top-K routing (e.g., 22)
    latent_size: int                          # Reduced input dim for routed experts (e.g., 1024)
    expert_intermediate_size: int             # Per-expert FFN intermediate size in latent space (e.g., 2688)
    shared_expert_intermediate_size: int      # Shared expert intermediate at full hidden dim (e.g., 5376)
    n_shared_experts: int = 1                 # Number of shared experts
    use_gating: bool = False                  # Gated activation (e.g., SwiGLU) in routed experts

    def get_prefill_flops(self, batch_size: int, sequence_length: int, hidden_dim: int) -> int:
        """
        Calculate FLOPs for one Latent MoE layer during prefill.

        Components:
          1. Down-projection:  H → latent  (all tokens)
          2. Router:           latent → E score per expert
          3. Routed experts (top-k): latent → expert_intermediate → latent
          4. Up-projection:    latent → H  (all tokens)
          5. Shared expert(s): H → shared_intermediate → H  (always active)
        """
        B, T = batch_size, sequence_length
        H, d_lat = hidden_dim, self.latent_size
        d_int, d_sh = self.expert_intermediate_size, self.shared_expert_intermediate_size
        k, E = self.num_experts_per_token, self.num_experts

        flops = 0
        # 1. Down projection
        flops += 2 * B * T * H * d_lat
        # 2. Router
        flops += 2 * B * T * d_lat * E
        # 3. Routed experts (up + down per active expert)
        flops += 2 * k * B * T * d_lat * d_int  # up
        flops += 2 * k * B * T * d_int * d_lat  # down
        if self.use_gating:
            flops += 2 * k * B * T * d_lat * d_int  # gate projection
        # 4. Up projection
        flops += 2 * B * T * d_lat * H
        # 5. Shared expert(s): H → shared_intermediate → H (full dimension)
        flops += self.n_shared_experts * 2 * B * T * H * d_sh  # up
        flops += self.n_shared_experts * 2 * B * T * d_sh * H  # down
        return flops

    def get_decode_flops(self, batch_size: int, hidden_dim: int) -> int:
        """FLOPs for one Latent MoE layer during decode (single new token)."""
        return self.get_prefill_flops(batch_size, 1, hidden_dim)

    def get_weight_params(self, hidden_dim: int) -> int:
        """Total weight parameters in a Latent MoE layer (all experts, for memory capacity)."""
        H, d_lat = hidden_dim, self.latent_size
        d_int, d_sh = self.expert_intermediate_size, self.shared_expert_intermediate_size

        per_expert = d_lat * d_int + d_int * d_lat
        if self.use_gating:
            per_expert += d_lat * d_int

        return (
            H * d_lat                                           # down proj
            + d_lat * self.num_experts                          # router
            + self.num_experts * per_expert                     # all routed experts
            + d_lat * H                                         # up proj
            + self.n_shared_experts * (H * d_sh + d_sh * H)   # shared expert(s)
        )


@dataclass
class LLMArchitecture:
    """
    Complete specification of an LLM architecture
    Flexible enough to represent various modern LLMs
    """
    # Basic model identification
    model_name: str
    model_family: str  # e.g., "Llama", "GPT", "DeepSeek", "Mistral"
    version: str
    
    # Architecture type
    architecture_type: ArchitectureType = ArchitectureType.DECODER_ONLY
    
    # Core dimensions
    num_layers: int = 32
    hidden_dim: int = 4096
    vocab_size: int = 32000
    max_sequence_length: int = 2048
    
    # Attention configuration
    attention_config: AttentionConfig = field(default_factory=lambda: AttentionConfig(num_attention_heads=32))
    
    # FFN configuration
    ffn_config: FFNConfig = field(default_factory=lambda: FFNConfig(intermediate_size=11008))
    
    # MoE configuration (optional)
    moe_config: Optional[MoEConfig] = None
    is_moe: bool = False
    
    # Mamba-2 configuration (optional, for hybrid Mamba/Attention architectures)
    mamba_config: Optional[Mamba2Config] = None
    
    # Linear attention configuration (optional, for hybrid linear/full attention architectures)
    # Used when layer_types contains LINEAR_ATTENTION entries
    linear_attention_config: Optional[LinearAttentionConfig] = None

    # Latent MoE configuration (optional, for architectures where routed experts operate
    # in a reduced latent dimension — e.g., Nemotron-3-Super-120B)
    # Used when hybrid_layer_types contains LATENT_MOE entries
    latent_moe_config: Optional[LatentMoEConfig] = None
    
    # Hybrid layer pattern (for Mamba/Attention/MLP hybrid architectures like Nemotron-H)
    # Format: List of HybridLayerType specifying layer type for each layer
    # Example pattern "MEMEM*" means: Mamba, MLP, Mamba, MLP, Mamba, Attention
    # If None, all layers are standard transformer layers (attention + FFN)
    hybrid_layer_types: Optional[List[HybridLayerType]] = None
    
    # Normalization
    normalization_type: NormalizationType = NormalizationType.LAYER_NORM
    normalization_eps: float = 1e-5
    
    # Position encoding
    position_encoding: PositionEncodingType = PositionEncodingType.ROTARY
    rope_theta: float = 10000.0  # Base for RoPE
    
    # Embeddings
    embedding_dropout: float = 0.0
    tie_word_embeddings: bool = False  # Share input/output embeddings
    
    # Additional architectural features
    use_parallel_residual: bool = False  # GPT-J style
    use_post_attention_layernorm: bool = False
    
    # Per-layer attention types (for hybrid sliding/full attention architectures)
    # If None, all layers use the attention_config.attention_type
    # If specified, must be a list of LayerAttentionType with length == num_layers
    layer_types: Optional[List[LayerAttentionType]] = None
    
    # Per-layer FFN types (for interleaved dense/MoE architectures like Llama-4)
    # If None, all layers use is_moe to determine FFN type
    # If specified, must be a list of FFNLayerType with length == num_layers
    # Dense layers use ffn_config.dense_intermediate_size (or intermediate_size if not set)
    # MoE layers use moe_config with ffn_config.intermediate_size per expert
    ffn_layer_types: Optional[List[FFNLayerType]] = None
    
    # Precision and quantization hints
    dtype: str = "float16"  # float32, float16, bfloat16, int8, int4
    
    # Kernel execution parameters
    kernel_launch_latency: float = 5e-6  # Kernel launch overhead in seconds (default: 5 microseconds)
    
    # Computed properties (can be calculated but allow override)
    total_parameters: Optional[int] = None
    active_parameters: Optional[int] = None  # For MoE models
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate derived properties"""
        # Set num_key_value_heads if not specified
        if self.attention_config.num_key_value_heads is None:
            self.attention_config.num_key_value_heads = self.attention_config.num_attention_heads
        
        # Calculate head dimension if not specified
        if self.attention_config.head_dim is None:
            self.attention_config.head_dim = self.hidden_dim // self.attention_config.num_attention_heads
        
        # Calculate total parameters if not provided
        if self.total_parameters is None:
            self.total_parameters = self.estimate_parameters()
        
        # For non-MoE models, active params = total params
        if self.active_parameters is None and not self.is_moe:
            self.active_parameters = self.total_parameters
    
    def estimate_parameters(self) -> int:
        """
        Estimate total number of parameters
        Simplified calculation - can be refined for specific architectures
        """
        params = 0
        
        # Embedding layer
        params += self.vocab_size * self.hidden_dim
        
        # Output layer (if not tied)
        if not self.tie_word_embeddings:
            params += self.vocab_size * self.hidden_dim
        
        # Per-layer parameters
        for _ in range(self.num_layers):
            # Attention QKV projections
            q_params = self.hidden_dim * (self.attention_config.num_attention_heads * self.attention_config.head_dim)
            kv_params = self.hidden_dim * (self.attention_config.num_key_value_heads * self.attention_config.head_dim) * 2
            
            # Attention output projection
            o_params = (self.attention_config.num_attention_heads * self.attention_config.head_dim) * self.hidden_dim
            
            attention_params = q_params + kv_params + o_params
            
            # Add bias if used
            if self.attention_config.attention_bias:
                attention_params += (self.attention_config.num_attention_heads * self.attention_config.head_dim) * 4
            
            # FFN parameters
            if self.is_moe and self.moe_config:
                # MoE: multiply by number of experts
                ffn_params_per_expert = (
                    self.hidden_dim * self.ffn_config.intermediate_size +
                    self.ffn_config.intermediate_size * self.hidden_dim
                )
                if self.ffn_config.use_gating:
                    ffn_params_per_expert += self.hidden_dim * self.ffn_config.intermediate_size
                
                ffn_params = ffn_params_per_expert * self.moe_config.num_experts
                
                # Router parameters
                ffn_params += self.hidden_dim * self.moe_config.num_experts
            else:
                # Dense FFN
                ffn_params = (
                    self.hidden_dim * self.ffn_config.intermediate_size +
                    self.ffn_config.intermediate_size * self.hidden_dim
                )
                if self.ffn_config.use_gating:
                    ffn_params += self.hidden_dim * self.ffn_config.intermediate_size
            
            # Add bias if used
            if self.ffn_config.ffn_bias:
                ffn_params += self.ffn_config.intermediate_size * 2
            
            # Layer normalization parameters (gamma and beta)
            norm_params = self.hidden_dim * 2  # Pre-attention norm
            norm_params += self.hidden_dim * 2  # Pre-FFN norm
            
            params += attention_params + ffn_params + norm_params
        
        return params
    
    def estimate_active_parameters(self) -> int:
        """
        Estimate active parameters during inference
        Important for MoE models where not all experts are active
        """
        if not self.is_moe:
            return self.total_parameters
        
        # Calculate the difference between all experts and active experts
        params = self.total_parameters
        
        if self.moe_config:
            # Per layer MoE params
            ffn_params_per_expert = (
                self.hidden_dim * self.ffn_config.intermediate_size +
                self.ffn_config.intermediate_size * self.hidden_dim
            )
            if self.ffn_config.use_gating:
                ffn_params_per_expert += self.hidden_dim * self.ffn_config.intermediate_size
            
            # Total MoE params per layer
            total_moe_per_layer = ffn_params_per_expert * self.moe_config.num_experts
            
            # Active MoE params per layer
            active_moe_per_layer = ffn_params_per_expert * self.moe_config.num_experts_per_token
            
            # Difference across all layers
            inactive_params = (total_moe_per_layer - active_moe_per_layer) * self.num_layers
            
            params -= inactive_params
        
        return params
    
    def get_num_full_attention_layers(self) -> int:
        """
        Return the number of layers using full attention.
        
        Returns:
            Number of full attention layers
        """
        if self.layer_types is None:
            # Check if global attention type is sliding window
            if self.attention_config.attention_type == AttentionType.SLIDING_WINDOW:
                return 0
            return self.num_layers
        
        return sum(1 for lt in self.layer_types if lt == LayerAttentionType.FULL_ATTENTION)
    
    def get_num_sliding_attention_layers(self) -> int:
        """
        Return the number of layers using sliding window attention.
        
        Returns:
            Number of sliding attention layers
        """
        if self.layer_types is None:
            # Check if global attention type is sliding window
            if self.attention_config.attention_type == AttentionType.SLIDING_WINDOW:
                return self.num_layers
            return 0
        
        return sum(1 for lt in self.layer_types if lt == LayerAttentionType.SLIDING_ATTENTION)
    
    def get_num_linear_attention_layers(self) -> int:
        """
        Return the number of layers using linear attention.
        
        Returns:
            Number of linear attention layers
        """
        if self.layer_types is None:
            return 0
        
        return sum(1 for lt in self.layer_types if lt == LayerAttentionType.LINEAR_ATTENTION)
    
    def get_linear_attention_state_size(self, batch_size: int = 1, 
                                         bytes_per_element: int = 2) -> int:
        """
        Calculate total linear attention state size for all linear attention layers.
        
        Linear attention state per layer: batch * key_head_dim * num_value_heads * value_head_dim
        This is CONSTANT regardless of sequence length.
        
        Args:
            batch_size: Number of sequences in the batch
            bytes_per_element: Bytes per state element (2 for fp16/bf16)
            
        Returns:
            Total state size in bytes
        """
        if not self.linear_attention_config:
            return 0
        
        num_linear_layers = self.get_num_linear_attention_layers()
        state_per_layer = self.linear_attention_config.get_state_size_bytes(
            batch_size, bytes_per_element
        )
        return num_linear_layers * state_per_layer
    
    def get_layer_attention_type(self, layer_idx: int) -> LayerAttentionType:
        """
        Get the attention type for a specific layer.
        
        Args:
            layer_idx: Layer index (0-based)
            
        Returns:
            LayerAttentionType for that layer
        """
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise ValueError(f"Layer index {layer_idx} out of range [0, {self.num_layers})")
        
        if self.layer_types is not None:
            return self.layer_types[layer_idx]
        
        # Infer from global attention type
        if self.attention_config.attention_type == AttentionType.SLIDING_WINDOW:
            return LayerAttentionType.SLIDING_ATTENTION
        return LayerAttentionType.FULL_ATTENTION
    
    def get_kv_cache_size(self, batch_size: int, sequence_length: int, bytes_per_element: int = 2) -> int:
        """
        Calculate KV cache memory requirement in bytes.
        
        For hybrid Mamba/Attention architectures:
        - Mamba layers: no KV cache (use Mamba state instead)
        - Attention layers: full KV cache
        - MLP-only layers: no KV cache
        
        For hybrid sliding/full attention architectures:
        - Full attention layers: store KV for full sequence
        - Sliding attention layers: store KV only for sliding_window_size tokens
        
        Args:
            batch_size: Number of sequences in batch
            sequence_length: Length of sequences
            bytes_per_element: Size of each element (2 for fp16/bf16, 4 for fp32)
        
        Returns:
            Total KV cache size in bytes
        """
        if self.attention_config.use_mla and self.attention_config.mla_kv_lora_rank:
            # MLA: Compressed KV cache using low-rank representation
            # Note: MLA compression applies to both full and sliding layers
            kv_dim = self.attention_config.mla_kv_lora_rank
            kv_size_factor = 2  # K and V both compressed
        else:
            # Standard KV cache dimension
            kv_dim = self.attention_config.num_key_value_heads * self.attention_config.head_dim
            kv_size_factor = 2  # K and V
        
        # Base KV cache calculation per token
        kv_per_token = kv_size_factor * batch_size * kv_dim * bytes_per_element
        
        # Get number of attention layers (accounts for hybrid Mamba/Attention)
        if self.hybrid_layer_types is not None:
            num_attention_layers = self.get_num_attention_layers_hybrid()
        else:
            num_attention_layers = self.num_layers
        
        # If no attention layers, no KV cache needed
        if num_attention_layers == 0:
            return 0
        
        # Determine effective sequence length per layer
        if self.layer_types is not None:
            # Hybrid sliding/full attention: calculate per-layer KV cache
            total_kv_cache = 0
            sliding_window = self.attention_config.sliding_window_size or sequence_length
            
            for layer_type in self.layer_types:
                if layer_type == LayerAttentionType.FULL_ATTENTION:
                    # Full attention: store entire sequence
                    effective_seq_len = sequence_length
                elif layer_type == LayerAttentionType.LINEAR_ATTENTION:
                    # Linear attention: no KV cache (uses fixed-size state instead)
                    continue
                else:
                    # Sliding attention: only store sliding_window tokens
                    effective_seq_len = min(sequence_length, sliding_window)
                
                total_kv_cache += kv_per_token * effective_seq_len
            
            return total_kv_cache
        else:
            # Homogeneous architecture: all layers same type
            if self.attention_config.attention_type == AttentionType.SLIDING_WINDOW:
                # Sliding window attention: cap at window size
                sliding_window = self.attention_config.sliding_window_size or sequence_length
                effective_seq_len = min(sequence_length, sliding_window)
            else:
                # Full attention
                effective_seq_len = sequence_length
            
            return kv_per_token * effective_seq_len * num_attention_layers
    
    def get_kv_cache_size_breakdown(self, batch_size: int, sequence_length: int, 
                                     bytes_per_element: int = 2) -> Dict[str, int]:
        """
        Get a breakdown of KV cache size by layer type.
        
        Args:
            batch_size: Number of sequences in batch
            sequence_length: Length of sequences
            bytes_per_element: Size of each element (2 for fp16/bf16, 4 for fp32)
        
        Returns:
            Dictionary with:
            - 'full_attention_layers': KV cache for full attention layers
            - 'sliding_attention_layers': KV cache for sliding attention layers
            - 'linear_attention_state': State memory for linear attention layers
            - 'total': Total KV cache + linear attention state size
            - 'num_full_layers': Number of full attention layers
            - 'num_sliding_layers': Number of sliding attention layers
            - 'num_linear_layers': Number of linear attention layers
        """
        if self.attention_config.use_mla and self.attention_config.mla_kv_lora_rank:
            kv_dim = self.attention_config.mla_kv_lora_rank
        else:
            kv_dim = self.attention_config.num_key_value_heads * self.attention_config.head_dim
        
        kv_per_token = 2 * batch_size * kv_dim * bytes_per_element
        sliding_window = self.attention_config.sliding_window_size or sequence_length
        
        num_full = self.get_num_full_attention_layers()
        num_sliding = self.get_num_sliding_attention_layers()
        num_linear = self.get_num_linear_attention_layers()
        
        full_kv_cache = kv_per_token * sequence_length * num_full
        sliding_kv_cache = kv_per_token * min(sequence_length, sliding_window) * num_sliding
        linear_state = self.get_linear_attention_state_size(batch_size, bytes_per_element)
        
        return {
            'full_attention_layers': full_kv_cache,
            'sliding_attention_layers': sliding_kv_cache,
            'linear_attention_state': linear_state,
            'total': full_kv_cache + sliding_kv_cache + linear_state,
            'num_full_layers': num_full,
            'num_sliding_layers': num_sliding,
            'num_linear_layers': num_linear,
        }
    
    def get_num_mamba_layers(self) -> int:
        """Get the number of Mamba layers in hybrid architecture.

        Counts both MAMBA (Mamba+FFN) and MAMBA_ONLY (Mamba with no FFN) layer types.
        """
        if not self.hybrid_layer_types:
            return 0
        return sum(1 for t in self.hybrid_layer_types
                   if t in (HybridLayerType.MAMBA, HybridLayerType.MAMBA_ONLY))
    
    def get_num_attention_layers_hybrid(self) -> int:
        """Get the number of attention layers in hybrid architecture.

        Counts both ATTENTION (Attention+FFN) and ATTENTION_ONLY (Attention with no FFN)
        layer types, since both contribute to the KV cache.
        """
        if not self.hybrid_layer_types:
            return self.num_layers
        return sum(1 for t in self.hybrid_layer_types
                   if t in (HybridLayerType.ATTENTION, HybridLayerType.ATTENTION_ONLY))
    
    def get_num_mlp_only_layers(self) -> int:
        """Get the number of MLP-only layers in hybrid architecture"""
        if not self.hybrid_layer_types:
            return 0
        return sum(1 for t in self.hybrid_layer_types if t == HybridLayerType.MLP)

    def get_num_latent_moe_layers(self) -> int:
        """Get the number of Latent MoE sublayers (Nemotron-Super style E sublayers)."""
        if not self.hybrid_layer_types:
            return 0
        return sum(1 for t in self.hybrid_layer_types if t == HybridLayerType.LATENT_MOE)
    
    def get_num_dense_ffn_layers(self) -> int:
        """Get the number of dense FFN layers (non-MoE) in interleaved architectures"""
        if not self.ffn_layer_types:
            # If no ffn_layer_types specified, all layers are same type
            return 0 if self.is_moe else self.num_layers
        return sum(1 for t in self.ffn_layer_types if t == FFNLayerType.DENSE)
    
    def get_num_moe_ffn_layers(self) -> int:
        """Get the number of MoE FFN layers in interleaved architectures"""
        if not self.ffn_layer_types:
            # If no ffn_layer_types specified, all layers are same type
            return self.num_layers if self.is_moe else 0
        return sum(1 for t in self.ffn_layer_types if t == FFNLayerType.MOE)
    
    def get_dense_intermediate_size(self) -> int:
        """Get the FFN intermediate size for dense layers"""
        if self.ffn_config.dense_intermediate_size is not None:
            return self.ffn_config.dense_intermediate_size
        return self.ffn_config.intermediate_size
    
    def get_moe_intermediate_size(self) -> int:
        """Get the per-expert FFN intermediate size for MoE layers"""
        return self.ffn_config.intermediate_size
    
    def get_mamba_state_size(self, batch_size: int = 1, bytes_per_element: int = 2) -> int:
        """
        Calculate total Mamba state size for all Mamba layers.
        
        Mamba-2 state per layer: batch_size * num_heads * head_dim * state_size
        
        Args:
            batch_size: Number of sequences in the batch
            bytes_per_element: Bytes per state element (2 for fp16/bf16)
            
        Returns:
            Total state size in bytes
        """
        if not self.mamba_config:
            return 0
        
        num_mamba_layers = self.get_num_mamba_layers()
        state_per_layer = self.mamba_config.get_state_size_bytes(batch_size, bytes_per_element)
        return num_mamba_layers * state_per_layer
    
    def get_total_inference_state_size(self, batch_size: int = 1, sequence_length: Optional[int] = None, 
                                        bytes_per_element: int = 2) -> Dict[str, int]:
        """
        Get total inference state size including KV cache, Mamba state, and linear attention state.
        
        Args:
            batch_size: Number of sequences in the batch
            sequence_length: Sequence length for KV cache calculation
            bytes_per_element: Bytes per element
            
        Returns:
            Dictionary with state size breakdown
        """
        if sequence_length is None:
            sequence_length = self.max_sequence_length
            
        kv_cache_size = self.get_kv_cache_size(batch_size, sequence_length, bytes_per_element)
        mamba_state_size = self.get_mamba_state_size(batch_size, bytes_per_element)
        linear_attn_state_size = self.get_linear_attention_state_size(batch_size, bytes_per_element)
        
        return {
            'kv_cache': kv_cache_size,
            'mamba_state': mamba_state_size,
            'linear_attention_state': linear_attn_state_size,
            'total': kv_cache_size + mamba_state_size + linear_attn_state_size,
            'num_attention_layers': self.get_num_attention_layers_hybrid(),
            'num_mamba_layers': self.get_num_mamba_layers(),
            'num_linear_attention_layers': self.get_num_linear_attention_layers(),
        }
    
    def get_memory_footprint(self, batch_size: int = 1, sequence_length: Optional[int] = None) -> Dict[str, int]:
        """
        Estimate memory footprint for inference
        
        Returns:
            Dictionary with memory components in bytes
        """
        if sequence_length is None:
            sequence_length = self.max_sequence_length
        
        bytes_per_param = 2 if self.dtype in ["float16", "bfloat16"] else 4
        
        memory = {
            "model_parameters": self.active_parameters * bytes_per_param,
            "kv_cache": self.get_kv_cache_size(batch_size, sequence_length, bytes_per_param),
            "mamba_state": self.get_mamba_state_size(batch_size, bytes_per_param),
            "linear_attention_state": self.get_linear_attention_state_size(batch_size, bytes_per_param),
            "activations": batch_size * sequence_length * self.hidden_dim * bytes_per_param * 4,  # Rough estimate
        }
        
        memory["total"] = sum(memory.values())
        
        return memory
    
    def summary(self) -> str:
        """Generate a human-readable summary of the architecture"""
        lines = [
            f"Model: {self.model_name} ({self.model_family} {self.version})",
            f"Architecture: {self.architecture_type.value}",
            f"Layers: {self.num_layers}",
            f"Hidden Dim: {self.hidden_dim}",
            f"Attention Heads: {self.attention_config.num_attention_heads}",
            f"KV Heads: {self.attention_config.num_key_value_heads}",
            f"FFN Intermediate: {self.ffn_config.intermediate_size}",
            f"Vocab Size: {self.vocab_size}",
            f"Max Sequence Length: {self.max_sequence_length}",
            f"Total Parameters: {self.total_parameters:,}",
        ]
        
        if self.is_moe and self.moe_config:
            lines.extend([
                f"MoE Experts: {self.moe_config.num_experts}",
                f"Active Experts: {self.moe_config.num_experts_per_token}",
                f"Active Parameters: {self.active_parameters:,}",
            ])
        
        lines.extend([
            f"Attention Type: {self.attention_config.attention_type.value}",
            f"Position Encoding: {self.position_encoding.value}",
            f"Normalization: {self.normalization_type.value}",
            f"Activation: {self.ffn_config.activation.value}",
            f"Kernel Launch Latency: {self.kernel_launch_latency * 1e6:.2f} µs",
        ])
        
        if self.attention_config.use_mla:
            lines.append(f"MLA KV Compression Rank: {self.attention_config.mla_kv_lora_rank}")
        
        if self.attention_config.use_sparse_attention:
            lines.append(f"Sparse Attention: Block={self.attention_config.sparse_block_size}, "
                        f"Local={self.attention_config.sparse_local_blocks}, "
                        f"Global={self.attention_config.sparse_global_blocks}")
        
        if self.hybrid_layer_types:
            num_attn = self.get_num_attention_layers_hybrid()
            num_mamba = self.get_num_mamba_layers()
            num_mlp = self.get_num_mlp_only_layers()
            num_latent_moe = self.get_num_latent_moe_layers()
            # Count MAMBA_ONLY and ATTENTION_ONLY separately for clarity
            num_mamba_only = sum(1 for t in self.hybrid_layer_types if t == HybridLayerType.MAMBA_ONLY)
            num_attn_only = sum(1 for t in self.hybrid_layer_types if t == HybridLayerType.ATTENTION_ONLY)
            if num_latent_moe > 0 or num_mamba_only > 0 or num_attn_only > 0:
                # Sublayer-style (e.g., Nemotron-Super)
                lines.append(
                    f"Hybrid Sublayer Architecture: {num_mamba_only} Mamba-only, "
                    f"{num_attn_only} Attention-only, {num_latent_moe} LatentMoE sublayers"
                )
            else:
                lines.append(f"Hybrid Architecture: {num_attn} Attention, {num_mamba} Mamba, {num_mlp} MLP layers")
        
        if self.mamba_config:
            lines.append(f"Mamba-2: heads={self.mamba_config.num_heads}, head_dim={self.mamba_config.head_dim}, "
                        f"state={self.mamba_config.state_size}, chunk={self.mamba_config.chunk_size}")
        
        if self.latent_moe_config:
            lmoe = self.latent_moe_config
            num_latent = self.get_num_latent_moe_layers()
            lines.append(
                f"Latent MoE: {num_latent} layers, experts={lmoe.num_experts}, "
                f"top-k={lmoe.num_experts_per_token}, latent_size={lmoe.latent_size}, "
                f"expert_intermediate={lmoe.expert_intermediate_size}, "
                f"shared_intermediate={lmoe.shared_expert_intermediate_size}"
            )
        
        if self.linear_attention_config:
            lac = self.linear_attention_config
            num_linear = self.get_num_linear_attention_layers()
            num_full = self.get_num_full_attention_layers()
            lines.append(f"Linear/Full Attention: {num_linear} Linear, {num_full} Full attention layers")
            lines.append(f"Linear Attention: key_heads={lac.num_key_heads}, key_dim={lac.key_head_dim}, "
                        f"value_heads={lac.num_value_heads}, value_dim={lac.value_head_dim}, "
                        f"conv_kernel={lac.conv_kernel_dim}")
        
        if self.ffn_layer_types:
            num_dense = self.get_num_dense_ffn_layers()
            num_moe = self.get_num_moe_ffn_layers()
            dense_size = self.get_dense_intermediate_size()
            moe_size = self.get_moe_intermediate_size()
            lines.append(f"Interleaved FFN: {num_dense} Dense (size={dense_size}), {num_moe} MoE (size={moe_size})")
        
        return "\n".join(lines)
