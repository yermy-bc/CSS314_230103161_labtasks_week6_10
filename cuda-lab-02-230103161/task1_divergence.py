"""Task 1: uniform, alternating-lane and warp-aligned branching."""
import time
import numpy as np
from numba import cuda

@cuda.jit
def kernel_uniform(a, out, n):
    i = cuda.grid(1)
    if i < n:
        x = a[i]
        for _ in range(1000):
            x = x * 1.00001 + 0.00001
        out[i] = x

@cuda.jit
def kernel_divergent(a, out, n):
    i = cuda.grid(1)
    if i < n:
        x = a[i]
        if i % 2 == 0:
            for _ in range(1000):
                x = x * 1.00001 + 0.00001
        else:
            for _ in range(1000):
                x = (x - 0.00001) / 1.00001
        out[i] = x

@cuda.jit
def kernel_warp_aligned(a, out, n):
    i = cuda.grid(1)
    if i < n:
        x = a[i]
        if (i // 32) % 2 == 0:
            for _ in range(1000):
                x = x * 1.00001 + 0.00001
        else:
            for _ in range(1000):
                x = (x - 0.00001) / 1.00001
        out[i] = x

def run(n=1_000_000, trials=10):
    a = cuda.to_device(np.ones(n, dtype=np.float32))
    out = cuda.device_array(n, dtype=np.float32)
    threads = 256
    blocks = (n + threads - 1) // threads
    results = {}
    for name, kernel in [('Uniform Path', kernel_uniform), ('Full Divergence', kernel_divergent), ('Warp-Aligned Branching', kernel_warp_aligned)]:
        kernel[blocks, threads](a, out, n)
        cuda.synchronize()
        samples = []
        for _ in range(trials):
            start = time.perf_counter()
            kernel[blocks, threads](a, out, n)
            cuda.synchronize()
            samples.append((time.perf_counter() - start) * 1000)
        results[name] = float(np.mean(samples))
        print(f'{name}: {results[name]:.3f} ms')
    return results

if __name__ == '__main__':
    run()
