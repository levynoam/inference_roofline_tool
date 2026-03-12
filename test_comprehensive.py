"""
Comprehensive Test Suite Runner for LLM Inference Performance Modeling

This module serves as documentation and a convenience runner for all tests.
Individual test classes are organized in the tests/ folder:

Test Organization:
==================
- tests/test_prefill_resources.py - Forward direction: TTFT → resources (8 tests)
- tests/test_achievable_ttft.py - Backward direction: resources → TTFT (8 tests)
- tests/test_decode_performance.py - Decode phase performance (15 tests)
- tests/test_cross_function_consistency.py - Consistency between functions (3 tests)
- tests/test_edge_cases.py - Edge cases and boundary conditions (6 tests)
- tests/test_memory_breakdown.py - Memory breakdown analysis (3 tests)
- tests/test_compute_breakdown.py - Compute breakdown analysis (4 tests)
- tests/test_breakdown_with_parallelism.py - Breakdowns with parallelism (1 test)
- tests/test_time_breakdown.py - Time breakdown analysis (4 tests)
- tests/test_mla.py - Multi-head Latent Attention tests (5 tests)
- tests/test_dsa.py - Dynamic Sparse Attention tests (6 tests)
- tests/test_prefill_resources_extended.py - Extended prefill tests (8 tests)
- tests/test_achievable_ttft_extended.py - Extended TTFT tests (5 tests)
- tests/test_decode_performance_extended.py - Extended decode tests (5 tests)

Total: 81 tests across 14 test files

Usage Examples:
===============
    # Run ALL tests (recommended method)
    pytest tests/ -v
    
    # Run all tests with coverage
    pytest tests/ --cov=inference_performance --cov-report=html
    
    # Run specific test file
    pytest tests/test_prefill_resources.py -v
    
    # Run specific test class
    pytest tests/test_decode_performance.py::TestDecodePerformance -v
    
    # Run specific test method
    pytest tests/test_decode_performance.py::TestDecodePerformance::test_basic_decode -v
    
    # Run tests matching a pattern
    pytest tests/ -k "decode" -v
    
    # Run tests with detailed output
    pytest tests/ -vv
    
    # Run tests and stop at first failure
    pytest tests/ -x
    
    # Run tests in parallel (requires pytest-xdist)
    pytest tests/ -n auto

Test Structure:
===============
All tests use shared fixtures defined in tests/conftest.py:
- all_models: List of all pre-configured models
- all_gpus: List of all GPU configurations  
- batch_sizes: Common batch sizes [1, 2, 4, 8, 16, 32]
- sequence_lengths: Common sequence lengths [128, 512, 1024, 2048, 4096, 8192]
- parallelism_configs: Various parallelism configurations

Benefits of Modular Structure:
==============================
1. Easy to locate specific tests
2. Faster test execution (can run individual files)
3. Better organization and maintainability
4. Parallel test execution support
5. Clear test categorization
6. Easier to add new test files

Note: This file exists for documentation purposes. 
Always run tests from the tests/ folder: pytest tests/
"""

# This file is kept for backward compatibility and documentation.
# Tests should be run from the tests/ folder directly.
# Example: pytest tests/ -v
