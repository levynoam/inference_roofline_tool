"""
Shared pytest fixtures for all tests
"""

import pytest
from llm_configs import (
    LLAMA_3_8B, LLAMA_3_70B, LLAMA_2_7B,
    DEEPSEEK_V3, DEEPSEEK_3_2,
    MISTRAL_7B, MIXTRAL_8X7B, GPT3_175B
)
from inference_performance import (
    ParallelismConfig,
    ParallelismType
)


@pytest.fixture
def all_models():
    """All pre-configured models"""
    return [
        LLAMA_3_8B, LLAMA_3_70B, LLAMA_2_7B,
        DEEPSEEK_V3, DEEPSEEK_3_2,
        MISTRAL_7B, MIXTRAL_8X7B, GPT3_175B
    ]


@pytest.fixture
def all_gpus():
    """All GPU configurations"""
    return ["A100-40GB", "A100-80GB", "H100-80GB", "MI300X"]


@pytest.fixture
def batch_sizes():
    """Common batch sizes to test"""
    return [1, 2, 4, 8, 16, 32]


@pytest.fixture
def sequence_lengths():
    """Common sequence lengths to test"""
    return [128, 512, 1024, 2048, 4096, 8192]


@pytest.fixture
def parallelism_configs():
    """Various parallelism configurations"""
    return [
        ParallelismConfig(),  # No parallelism
        ParallelismConfig(
            parallelism_type=ParallelismType.TENSOR_PARALLEL,
            tensor_parallel_size=2
        ),
        ParallelismConfig(
            parallelism_type=ParallelismType.TENSOR_PARALLEL,
            tensor_parallel_size=4
        ),
        ParallelismConfig(
            parallelism_type=ParallelismType.TENSOR_PARALLEL,
            tensor_parallel_size=8
        ),
    ]
