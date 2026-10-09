"""Task 3: grid-stride vector transformation for arbitrary sizes."""
import numpy as np
from numba import cuda

@cuda.jit
def grid_stride_kernel(input_array, output_array, n, factor):
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    for j in range(i, n, stride):
        output_array[j] = input_array[j] * factor

def run(n=10_000_000):
    a = np.random.default_rng(42).random(n).astype(np.float32)
    if n == 0:
        print('Task 3: PASS (empty input)')
        return True
    d_a = cuda.to_device(a)
    d_out = cuda.device_array_like(d_a)
    grid_stride_kernel[256, 256](d_a, d_out, n, np.float32(4.25))
    actual = d_out.copy_to_host()
    ok = np.allclose(actual, a * np.float32(4.25), rtol=1e-5, atol=1e-6)
    print('Task 3:', 'PASS' if ok else 'FAIL', 'size:', n)
    return bool(ok)

if __name__ == '__main__':
    assert all(run(n) for n in [0, 1, 7, 257, 65537, 10_000_000])


def run_grid_stride(input_array, factor):
    """Scale arbitrary 1D float32 vector by factor using a grid-stride loop."""
    a = np.asarray(input_array, dtype=np.float32)
    if a.ndim != 1:
        raise ValueError("Expected a 1D input array")
    if a.size == 0:
        return a.copy()
    d_a = cuda.to_device(a)
    d_out = cuda.device_array_like(d_a)
    grid_stride_kernel[256, 256](d_a, d_out, a.size, np.float32(factor))
    return d_out.copy_to_host()
