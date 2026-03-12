"""
Tests for persistent storage offloading feature (MoE models that don't fit in DRAM).

This feature handles the scenario where MoE models are too large to fit in DRAM,
requiring active experts to be loaded from persistent storage during inference.
"""

import pytest
from llm_configs import MIXTRAL_8X7B, DEEPSEEK_V3, LLAMA_3_8B
from inference_performance import InferencePerformance, SystemConstraints, ParallelismConfig


class TestPersistentStorageOffloading:
    """Test persistent storage bandwidth for MoE offloading"""
    
    def test_moe_fits_in_memory_no_storage_used(self):
        """When MoE model fits in DRAM, persistent storage should not be used"""
        # Use Mixtral 8x7B (MoE model, ~93 GB)
        model = MIXTRAL_8X7B
        perf = InferencePerformance(model)
        
        # Large DRAM that can hold the model
        system = SystemConstraints(
            memory_capacity=200 * (1024**3),  # 200 GB (plenty)
            memory_bandwidth=2000 * (1024**3),  # 2 TB/s
            compute_throughput=312e12,  # 312 TFLOPS
            network_bandwidth=600 * (1024**3),  # 600 GB/s
            persistent_storage_bandwidth=20 * (1024**3)  # 20 GB/s
        )
        
        # Test decode performance
        result = perf.calculate_decode_performance(
            system_constraints=system,
            batch_size=1,
            prefill_length=1024,
            output_length=100
        )
        
        # Model fits in memory, so storage bandwidth should not be used
        assert result.avg_storage_bw_utilization == 0.0
        assert result.primary_bottleneck != "Persistent Storage Bandwidth"
        
        print(f"\nMixtral 8x7B model size: {model.total_parameters * 2 / 1e9:.1f} GB")
        print(f"DRAM capacity: {system.memory_capacity / 1e9:.1f} GB")
        print(f"Storage BW utilization: {result.avg_storage_bw_utilization * 100:.1f}%")
        print(f"Primary bottleneck: {result.primary_bottleneck}")
        print("✓ Model fits in DRAM - no storage offloading needed")
    
    def test_moe_doesnt_fit_storage_used(self):
        """When MoE model doesn't fit in DRAM, persistent storage should be used"""
        # Use Mixtral 8x7B (MoE model, ~93 GB)
        model = MIXTRAL_8X7B
        perf = InferencePerformance(model)
        
        # Small DRAM that cannot hold the model
        system = SystemConstraints(
            memory_capacity=40 * (1024**3),  # 40 GB (insufficient for 93 GB model)
            memory_bandwidth=1555 * (1024**3),  # 1.5 TB/s
            compute_throughput=312e12,  # 312 TFLOPS
            network_bandwidth=600 * (1024**3),  # 600 GB/s
            persistent_storage_bandwidth=20 * (1024**3)  # 20 GB/s
        )
        
        # Test decode performance
        result = perf.calculate_decode_performance(
            system_constraints=system,
            batch_size=1,
            prefill_length=1024,
            output_length=100
        )
        
        # Model doesn't fit in memory, so storage should be used
        assert result.avg_storage_bw_utilization > 0.0
        print(f"\nModel size: {model.total_parameters * 2 / 1e9:.1f} GB")
        print(f"DRAM capacity: {system.memory_capacity / 1e9:.1f} GB")
        print(f"Storage BW utilization: {result.avg_storage_bw_utilization * 100:.1f}%")
        print(f"Primary bottleneck: {result.primary_bottleneck}")
        print("✓ Model doesn't fit - persistent storage offloading active")
    
    def test_non_moe_never_uses_storage(self):
        """Non-MoE models should never use persistent storage, even if they don't fit"""
        # Use Llama 3 8B (dense model)
        model = LLAMA_3_8B
        perf = InferencePerformance(model)
        
        # Small DRAM that cannot hold the model
        system = SystemConstraints(
            memory_capacity=10 * (1024**3),  # 10 GB (insufficient)
            memory_bandwidth=1555 * (1024**3),  # 1.5 TB/s
            compute_throughput=312e12,  # 312 TFLOPS
            network_bandwidth=600 * (1024**3),  # 600 GB/s
            persistent_storage_bandwidth=20 * (1024**3)  # 20 GB/s
        )
        
        # Test decode performance
        result = perf.calculate_decode_performance(
            system_constraints=system,
            batch_size=1,
            prefill_length=1024,
            output_length=100
        )
        
        # Non-MoE model should never use persistent storage
        assert result.avg_storage_bw_utilization == 0.0
        assert result.primary_bottleneck != "Persistent Storage Bandwidth"
        
        print(f"\nDense model - Storage BW utilization: {result.avg_storage_bw_utilization * 100:.1f}%")
        print("✓ Dense models don't use persistent storage offloading")
    
    def test_storage_bandwidth_becomes_bottleneck(self):
        """Test that slow persistent storage can become the bottleneck"""
        # Use Mixtral 8x7B
        model = MIXTRAL_8X7B
        perf = InferencePerformance(model)
        
        # System with fast compute/memory but slow storage
        system = SystemConstraints(
            memory_capacity=40 * (1024**3),  # 40 GB (insufficient)
            memory_bandwidth=3000 * (1024**3),  # 3 TB/s (fast)
            compute_throughput=1000e12,  # 1 PFLOPS (fast)
            network_bandwidth=600 * (1024**3),  # 600 GB/s
            persistent_storage_bandwidth=5 * (1024**3)  # 5 GB/s (slow!)
        )
        
        # Test decode performance
        result = perf.calculate_decode_performance(
            system_constraints=system,
            batch_size=1,
            prefill_length=1024,
            output_length=100
        )
        
        print(f"\nStorage BW: {system.persistent_storage_bandwidth / 1e9:.1f} GB/s")
        print(f"Storage BW utilization: {result.avg_storage_bw_utilization * 100:.1f}%")
        print(f"Primary bottleneck: {result.primary_bottleneck}")
        
        # With slow storage, it should be the bottleneck
        assert result.primary_bottleneck == "Persistent Storage Bandwidth"
        print("✓ Slow persistent storage correctly identified as bottleneck")
    
    def test_fast_storage_not_bottleneck(self):
        """Test that fast persistent storage doesn't become bottleneck"""
        # Use Mixtral 8x7B
        model = MIXTRAL_8X7B
        perf = InferencePerformance(model)
        
        # System with very fast storage
        system = SystemConstraints(
            memory_capacity=40 * (1024**3),  # 40 GB (insufficient)
            memory_bandwidth=1555 * (1024**3),  # 1.5 TB/s
            compute_throughput=312e12,  # 312 TFLOPS
            network_bandwidth=600 * (1024**3),  # 600 GB/s
            persistent_storage_bandwidth=1000 * (1024**3)  # 1 TB/s (very fast!)
        )
        
        # Test decode performance
        result = perf.calculate_decode_performance(
            system_constraints=system,
            batch_size=1,
            prefill_length=1024,
            output_length=100
        )
        
        print(f"\nStorage BW: {system.persistent_storage_bandwidth / 1e9:.1f} GB/s")
        print(f"Storage BW utilization: {result.avg_storage_bw_utilization * 100:.1f}%")
        print(f"Primary bottleneck: {result.primary_bottleneck}")
        
        # With fast storage, it should not be the bottleneck
        assert result.primary_bottleneck != "Persistent Storage Bandwidth"
        assert result.avg_storage_bw_utilization < 1.0  # Not fully utilized
        print("✓ Fast storage correctly not identified as bottleneck")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
