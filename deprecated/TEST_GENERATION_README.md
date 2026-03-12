# Test Generation from Configuration Files

This directory contains infrastructure to automatically generate unit tests from exported configuration files.

## Overview

When debugging performance issues or validating specific configurations, you can:
1. Export a configuration from the web UI
2. Automatically generate a regression test from it
3. Add it to the test suite to prevent regressions

## Workflow

### 1. Export Configuration from Web UI

Click the "Export Configuration" button in the web app to save your current setup as a JSON file. The file will be downloaded with a name like:
```
llm-config-decode-2026-01-29T01-10-21.json
```

Save it to the `logs/` directory in this workspace.

### 2. Generate Tests

Run the test generator script:

```bash
.\venv\Scripts\python.exe generate_tests_from_configs.py
```

This will:
- Scan the `logs/` directory for all `llm-config-*.json` files
- Run each configuration to capture the results
- Generate pytest test functions with assertions
- Create `test_from_configs.py` with all generated tests

**Custom Options:**
```bash
# Generate from a different directory
.\venv\Scripts\python.exe generate_tests_from_configs.py path/to/configs

# Generate to a different output file
.\venv\Scripts\python.exe generate_tests_from_configs.py logs test_my_configs.py
```

### 3. Run Generated Tests

```bash
# Run all generated tests
.\venv\Scripts\python.exe -m pytest test_from_configs.py -v

# Run a specific test
.\venv\Scripts\python.exe -m pytest test_from_configs.py::test_llama_4_scout_decode_tensor_parallel_2gpu -v
```

### 4. Add to Comprehensive Test Suite (Optional)

To include generated tests in the main test suite:

1. Review the generated test in `test_from_configs.py`
2. Copy the test function to `test_comprehensive.py`
3. Adjust test name if needed to avoid conflicts
4. Run full test suite to verify

```bash
.\venv\Scripts\python.exe -m pytest test_comprehensive.py -v
```

## Generated Test Structure

Each generated test includes:

### Configuration Documentation
```python
def test_llama_4_scout_decode_tensor_parallel_2gpu():
    """
    Test llama-4-scout Decode calculation with TENSOR_PARALLEL parallelism.
    
    Configuration:
    - Batch Size: 8
    - Prefill Length: 2048
    - Output Length: 8192
    - Memory: 256 GB
    - Compute: 450000 TFLOPS
    - Memory BW: 8000000 GB/s
    - Network BW: 600 GB/s
    
    Expected Results:
    - Total Time: 107.37 ms
    - TPS: 76293.95
    - Primary Bottleneck: Network Bandwidth
    - Network BW Used: 600.00 GB/s
    """
```

### Setup Code
- Model loading
- System constraints
- Parallelism configuration
- Calculation execution

### Assertions
Tests verify:
- **Performance metrics** (TTFT, decode time, TPS)
- **Bottleneck identification** (Compute, Memory BW, Network)
- **Resource utilization** (Compute %, Memory BW %, Network %)
- **Bandwidth usage** (actual GB/s or TB/s consumed)

## Use Cases

### 1. Regression Testing
Generate tests for critical configurations to ensure future changes don't break them:
```bash
# Export config from web UI for: Llama-4-Scout, TP=8, Batch=16, Decode
# Save to logs/
.\venv\Scripts\python.exe generate_tests_from_configs.py
# Add generated test to test_comprehensive.py
```

### 2. Bug Validation
When fixing a bug, create a test to prevent it from recurring:
```bash
# Reproduce bug in web UI
# Export the problematic configuration
# Fix the bug
# Generate test to validate the fix
.\venv\Scripts\python.exe generate_tests_from_configs.py
.\venv\Scripts\python.exe -m pytest test_from_configs.py -v
```

### 3. Batch Test Generation
Test multiple configurations systematically:
```bash
# Export several configs from web UI (different models, parallelism, batch sizes)
# Generate all tests at once
.\venv\Scripts\python.exe generate_tests_from_configs.py
# Run all generated tests
.\venv\Scripts\python.exe -m pytest test_from_configs.py -v
```

## Files

- **`generate_tests_from_configs.py`** - Test generator script
- **`test_from_configs.py`** - Generated test file (auto-created)
- **`logs/`** - Directory for exported configuration files
- **`analyze_config.py`** - Offline analysis tool for debugging configs

## Debugging Workflow

When a configuration produces unexpected results:

1. **Export the config** from the web UI
2. **Analyze it offline** to see detailed debug info:
   ```bash
   .\venv\Scripts\python.exe analyze_config.py logs/llm-config-decode-*.json
   ```
3. **Fix the issue** in the codebase
4. **Generate a test** to prevent regression:
   ```bash
   .\venv\Scripts\python.exe generate_tests_from_configs.py
   ```
5. **Verify the fix**:
   ```bash
   .\venv\Scripts\python.exe -m pytest test_from_configs.py -v
   ```

## Tips

- **Descriptive file names**: The config filename becomes part of the test name
- **Clean up logs**: Remove old configs you don't want to test
- **Tolerance levels**: Assertions use reasonable tolerances (e.g., ±1ms for timing, ±1 GB/s for bandwidth)
- **Network bandwidth**: Tests specifically validate network BW calculations to catch the zero-traffic bug

## Example Session

```bash
# 1. User exports config from web app to logs/
# 2. Generate tests
PS> .\venv\Scripts\python.exe generate_tests_from_configs.py
Found 1 config file(s)
Processing: llm-config-decode-2026-01-29T01-10-21.json
  ✓ Generated test for llama-4-scout DECODE
Generated test_from_configs.py with 1 test(s)

# 3. Run tests
PS> .\venv\Scripts\python.exe -m pytest test_from_configs.py -v
test_from_configs.py::test_llama_4_scout_decode_tensor_parallel_2gpu PASSED [100%]
1 passed in 0.05s

# 4. Add to comprehensive suite if desired
# (copy test function to test_comprehensive.py)
```

## Maintenance

- **Update tolerances**: If test becomes flaky, adjust assertion tolerances
- **Update assertions**: Add new assertions when new metrics are added
- **Regenerate**: Re-run generator after fixing bugs to update expected values
