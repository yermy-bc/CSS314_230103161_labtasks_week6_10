# CUDA Lab 02: Advanced Geometries & Stencils
**Student ID:** 230103161  
**Allocated GPU Node:** Tesla T4  
**CUDA Compute Capability:** 7.5  
**Official Verification Token:** A7F29C04D8E13B65F092

## Environment / Telemetry
- Platform: Google Colab
- GPU: NVIDIA Tesla T4 (15,360 MiB reported by `nvidia-smi`)
- NVIDIA driver: 580.82.07; driver-reported CUDA compatibility: 13.0
- Python: 3.13 (Colab notebook path)
- NumPy: 2.1.3; Numba: 0.61.2
- `cuda.is_available()`: True

## Benchmarks (from original Colab notebook)
| Experiment | Mean time (ms) | Notes |
|---|---:|---|
| Task 1: Uniform Path | 22.156 | 1,000,000 elements, 1000 iterations |
| Task 1: Full Divergence | 90.145 | Same input size |
| Task 1: Warp-Aligned Branching | 44.746 | Same input size |
| Task 2: Stencil | 0.479 | OLD zero-padding kernel; rebenchmark clamped version |
| Task 3: Grid-Stride | 0.895 | OLD `2*x+1` transform; rebenchmark scaling version |
| Task 4: Sobel | 0.412 | OLD gradient magnitude kernel; rebenchmark Sobel-X version |

Measurements used `time.perf_counter()` and `cuda.synchronize()` across 10 trials, excluding memory transfers but including Python launch/synchronization overhead.

## Answers / Discussion
1. **Warp divergence:** Uniform execution was fastest (22.156 ms). Alternating lanes took 90.145 ms; warp-aligned branching took 44.746 ms. Different arithmetic in branches also affects timings, so these results do not isolate divergence alone.
2. **Boundary clamping:** Out-of-range left/right indices are clamped to the first/last valid element. Unlike zero padding, edge pixels repeat their boundary values.
3. **Grid-stride loop:** Each thread visits `idx, idx + cuda.gridsize(1), ...` until the array ends. The operation is `out[i] = input[i] * factor`, for any 1D vector size.
4. **Sobel-X:** Uses `[-1,0,1; -2,0,2; -1,0,1]` and zero-valued border. Output is the signed horizontal derivative, not full gradient magnitude.

## Official Verification
Run in Colab with a GPU runtime, from inside this project directory:

```bash
python verify_submission.py
```

When prompted, enter `230103161`. Only after **all three checks pass** copy the printed `OFFICIAL SUBMISSION TOKEN` into the token field above. Do not submit with the placeholder.

## Verification Status
- Instructor-provided verifier has been restored from the text supplied in the chat (indentation and wrapped line repaired).
- Syntax checked locally; **not yet run on the user's Tesla T4**.
- Exact remainder of mandatory README template, if any, was not provided.
