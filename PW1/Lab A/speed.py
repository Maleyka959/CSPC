import time
from decay import simulate_loop, simulate

N0 = 200000
lam = 0.05
dt = 0.05
steps = 200

# Time pure-Python loop
t0 = time.perf_counter()
simulate_loop(N0=N0, lam=lam, dt=dt, steps=steps)
t1 = time.perf_counter()
time_loop = t1 - t0

# Time NumPy version
t0 = time.perf_counter()
simulate(N0=N0, lam=lam, dt=dt, steps=steps)
t1 = time.perf_counter()
time_numpy = t1 - t0

speedup = time_loop / time_numpy

print(f"Python Loop Time: {time_loop:.4f} seconds")
print(f"NumPy Time:       {time_numpy:.4f} seconds")
print(f"NumPy is {speedup:.2f}x faster than Python loop")