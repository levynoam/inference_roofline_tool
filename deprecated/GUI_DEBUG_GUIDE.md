# GUI Debugging & Troubleshooting Guide

## Issues Found and Fixed

### Issue 1: SystemConstraints Parameter Mismatch
**Problem**: GUI was passing `gpu_name` parameter to `SystemConstraints()`, but the dataclass doesn't accept that parameter.

**Symptoms**:
```
TypeError: SystemConstraints.__init__() got an unexpected keyword argument 'gpu_name'
```

**Fix**: Removed `gpu_name` and `num_gpus` from SystemConstraints initialization:
```python
# Before (WRONG):
return SystemConstraints(
    gpu_name=f"Custom ({self.memory_var.get()}GB)",
    num_gpus=1,
    memory_capacity=memory_capacity,
    ...
)

# After (CORRECT):
return SystemConstraints(
    memory_capacity=memory_capacity,
    memory_bandwidth=memory_bandwidth,
    compute_throughput=compute_throughput,
    network_bandwidth=network_bandwidth
)
```

### Issue 2: Attribute Name Mismatch
**Problem**: GUI was using abbreviated attribute names that don't exist in `ResourceUtilization`.

**Symptoms**:
```
AttributeError: 'ResourceUtilization' object has no attribute 'memory_bw_utilization'
```

**Fix**: Changed abbreviated names to full names:
- `memory_bw_utilization` → `memory_bandwidth_utilization`
- `network_bw_utilization` → `network_bandwidth_utilization`

These changes were made in:
1. TTFT results display
2. TTFT visualization
3. Both `gui_app.py` and `gui_app_debug.py`

## How to Debug GUI Issues

### Method 1: Use Debug Version
```bash
.\venv\Scripts\python.exe gui_app_debug.py
```

The debug version prints detailed messages to console:
- `[DEBUG] GUI initialized successfully`
- `[DEBUG] Model selected: Llama 3 8B`
- `[DEBUG] Calculate button clicked`
- `[DEBUG] TTFT params: batch=1, seq_len=2048`
- `[DEBUG] TTFT result: 107.03ms`
- `[DEBUG] Calculation completed successfully`

If an error occurs, it prints full traceback to console.

### Method 2: Run Test Script
```bash
.\venv\Scripts\python.exe test_gui_calculation.py
```

This tests the calculation logic without the GUI:
- Tests SystemConstraints creation
- Tests TTFT calculation
- Tests Decode calculation
- Tests error handling

If this passes, the issue is in the GUI code, not the calculation logic.

### Method 3: Check Terminal Output
When GUI crashes or calculate button doesn't work:
1. Look at terminal where you launched GUI
2. Check for error messages and tracebacks
3. Common errors:
   - `TypeError`: Wrong parameter names
   - `AttributeError`: Wrong attribute names
   - `ValueError`: Invalid input values (non-numeric)
   - `KeyError`: Wrong dictionary keys

### Method 4: Test Individual Components
Create minimal test scripts for specific GUI components:

**Test System Constraints:**
```python
from inference_performance import SystemConstraints

gpu = SystemConstraints(
    memory_capacity=80e9,
    memory_bandwidth=2039e9,
    compute_throughput=312e12,
    network_bandwidth=600e9
)
print(f"Created: {gpu}")
```

**Test Model Loading:**
```python
from llm_configs import LLAMA_3_8B
from inference_performance import InferencePerformance

model = LLAMA_3_8B
perf = InferencePerformance(model)
print(f"Model: {model.model_name}, {model.total_parameters/1e9}B params")
```

## Common Issues and Solutions

### Issue: Calculate Button Does Nothing
**Check**:
1. Is button connected to `calculate_performance` method?
2. Are there errors in terminal output?
3. Run `gui_app_debug.py` to see if button click is detected

**Solution**: Look for `[DEBUG] Calculate button clicked` in console. If missing, button binding is broken.

### Issue: Invalid Input Error
**Check**: Input validation in spinboxes

**Solution**: Ensure all spinbox values are valid numbers:
```python
try:
    batch_size = int(self.batch_size_var.get())
except ValueError:
    messagebox.showerror("Error", "Batch size must be a number")
    return
```

### Issue: Visualization Not Updating
**Check**: 
1. Is `canvas.draw()` being called?
2. Are there errors in visualization code?

**Solution**: Ensure visualization methods complete without errors. Check terminal for matplotlib warnings.

### Issue: Results Text Empty
**Check**:
1. Is calculation completing successfully?
2. Is `display_*_results()` being called?

**Solution**: Add debug prints in display methods to confirm they're called.

## Testing Checklist

Before reporting GUI as broken, verify:

- [ ] Test script passes (`test_gui_calculation.py`)
- [ ] Debug GUI prints messages to console
- [ ] All system parameters have valid values
- [ ] Model is selected
- [ ] Calculation type is selected
- [ ] Parameters are filled in
- [ ] Terminal shows no error messages

## Files for Debugging

1. **gui_app.py** - Production GUI (clean, no debug output)
2. **gui_app_debug.py** - Debug GUI (verbose console output)
3. **test_gui_calculation.py** - Test calculation logic without GUI
4. **test_comprehensive.py** - Full test suite (34 tests)

## Quick Debug Commands

```bash
# Test calculation logic
.\venv\Scripts\python.exe test_gui_calculation.py

# Run full test suite
.\venv\Scripts\python.exe -m pytest test_comprehensive.py -v

# Launch debug GUI
.\venv\Scripts\python.exe gui_app_debug.py

# Launch production GUI
.\venv\Scripts\python.exe gui_app.py
```

## Known Working Configuration

After fixes, this configuration works:

**TTFT Calculation**:
- Model: Llama 3 8B
- GPU: A100-80GB (80GB, 312 TFLOPS, 2039 GB/s)
- Batch: 1
- Sequence: 2048
- Result: ~107ms, Compute-bound

**Decode Calculation**:
- Model: Llama 3 8B
- GPU: A100-80GB
- Batch: 1
- Prefill: 2048
- Output: 128
- Result: ~106 TPS, Memory BW-bound

## What to Do If GUI Still Doesn't Work

1. **Update the repository**: Ensure you have latest fixes
2. **Check Python version**: Should be 3.12
3. **Check dependencies**: 
   ```bash
   .\venv\Scripts\python.exe -m pip list
   # Should have: matplotlib, tkinter (built-in)
   ```
4. **Run test script first**: Verify calculation logic works
5. **Use debug GUI**: Check console for error messages
6. **Check file paths**: Ensure all imports resolve correctly
7. **Try fresh virtual environment**: Sometimes package issues cause problems

## Success Indicators

GUI is working correctly when:
- ✅ GUI window opens without errors
- ✅ Model dropdown shows 8 models
- ✅ GPU preset loads values into spinboxes
- ✅ Calculate button shows results in text area
- ✅ Bar chart visualization updates with colored bars
- ✅ Bottleneck has red border in chart
- ✅ No error popups or console errors
