"""Task 4: 3x3 Sobel-X convolution, zero-valued outer image border."""
import numpy as np
from numba import cuda

@cuda.jit
def sobel_2d(input_image, output_image, height, width):
    y, x = cuda.grid(2)
    if y < height and x < width:
        if y == 0 or x == 0 or y == height - 1 or x == width - 1:
            output_image[y, x] = 0.0
        else:
            output_image[y, x] = (
                -input_image[y-1, x-1] + input_image[y-1, x+1]
                -2.0 * input_image[y, x-1] + 2.0 * input_image[y, x+1]
                -input_image[y+1, x-1] + input_image[y+1, x+1]
            )

def reference(image):
    out = np.zeros_like(image)
    if image.shape[0] < 3 or image.shape[1] < 3:
        return out
    out[1:-1,1:-1] = (-image[:-2,:-2] + image[:-2,2:] - 2.0*image[1:-1,:-2] + 2.0*image[1:-1,2:] - image[2:,:-2] + image[2:,2:])
    return out

def run(height=512, width=512):
    image = np.random.default_rng(42).random((height, width)).astype(np.float32)
    d_img = cuda.to_device(image)
    d_out = cuda.device_array_like(d_img)
    threads = (16, 16)
    blocks = ((height + 15)//16, (width + 15)//16)
    sobel_2d[blocks, threads](d_img, d_out, height, width)
    actual = d_out.copy_to_host()
    ok = np.allclose(actual, reference(image), rtol=1e-5, atol=1e-5)
    print('Task 4:', 'PASS' if ok else 'FAIL', 'shape:', image.shape)
    return bool(ok)

if __name__ == '__main__':
    assert all(run(h, w) for h,w in [(1,1),(2,5),(5,2),(7,9),(512,512)])


def run_sobel(input_image):
    """Return signed horizontal Sobel-X response with zero outer border."""
    image = np.asarray(input_image, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError("Expected a 2D input image")
    h, w = image.shape
    if h == 0 or w == 0:
        return np.zeros_like(image)
    d_img = cuda.to_device(image)
    d_out = cuda.device_array_like(d_img)
    threads = (16, 16)
    blocks = ((h + 15) // 16, (w + 15) // 16)
    sobel_2d[blocks, threads](d_img, d_out, h, w)
    return d_out.copy_to_host()
