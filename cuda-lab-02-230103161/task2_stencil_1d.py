"""Task 2: 3-point averaging stencil with CLAMPED boundary indices."""
import numpy as np
from numba import cuda

@cuda.jit
def stencil_1d(input_array, output_array, n):
    i = cuda.grid(1)
    if i < n:
        left = i - 1 if i > 0 else 0
        right = i + 1 if i + 1 < n else n - 1
        output_array[i] = (input_array[left] + input_array[i] + input_array[right]) / 3.0

def reference(a):
    if len(a) == 0:
        return a.copy()
    ids = np.arange(len(a))
    return ((a[np.maximum(ids - 1, 0)] + a[ids] + a[np.minimum(ids + 1, len(a)-1)]) / 3.0).astype(np.float32)

def run(n=1_000_000):
    a = np.random.default_rng(42).random(n).astype(np.float32)
    if n == 0:
        return True
    d_a = cuda.to_device(a)
    d_out = cuda.device_array_like(d_a)
    stencil_1d[(n + 255)//256, 256](d_a, d_out, n)
    actual = d_out.copy_to_host()
    ok = np.allclose(actual, reference(a), rtol=1e-5, atol=1e-6)
    print('Task 2:', 'PASS' if ok else 'FAIL')
    return bool(ok)

if __name__ == '__main__':
    assert run()


def cpu_stencil(input_array):
    """CPU reference used by the instructor verification suite."""
    a = np.asarray(input_array, dtype=np.float32)
    return reference(a)


def run_stencil(input_array):
    """Run clamped 3-point stencil on CUDA and return a host NumPy array."""
    a = np.asarray(input_array, dtype=np.float32)
    if a.ndim != 1:
        raise ValueError("Expected a 1D input array")
    if a.size == 0:
        return a.copy()
    d_a = cuda.to_device(a)
    d_out = cuda.device_array_like(d_a)
    stencil_1d[(a.size + 255) // 256, 256](d_a, d_out, a.size)
    return d_out.copy_to_host()
