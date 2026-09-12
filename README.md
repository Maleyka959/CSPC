# CSPC Lab A

# CSPC Lab Reports

## PW1 --- Lab A: Radioactive Decay & Vectorization

### 1. Test Results
All 3 unit tests in `PW1/Lab A/test_decay.py` passed successfully using `pytest`:
- `test_starts_at_N0`: Verified initial atom count.
- `test_negative_rate_raises_value_error`: Confirmed `ValueError` is raised for negative rates (`lam < 0`).
- `test_simulation_average_close_to_theoretical`: Validated that simulated decay aligns with theoretical $N(t) = N_0 e^{-\lambda t}$ within tolerance.

### 2. Speed Comparison
Running `PW1/Lab A/speed.py` with $N_0 = 200,000$ atoms over 200 steps produced the following timings:
- **Pure-Python Loop (`simulate_loop`):** ~2.15 seconds
- **Vectorized NumPy (`simulate`):** ~0.05 seconds
- **Speedup Factor:** NumPy version ran approximately **40x to 50x faster** than the pure Python loop.

### 3. Conclusion
Vectorizing calculations with NumPy significantly reduces runtime compared to standard Python loops. NumPy offloads iteration loops to compiled C code and operates on contiguous memory blocks, avoiding high Python dynamic type checks and loop overhead during array operations.