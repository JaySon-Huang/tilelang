import torch
import tilelang
import tilelang.language as T
import tilelang.testing


def _run_pdl_elementwise(N, dtype, torch_dtype, rtol, atol):
    """A kernel that uses CUDA's programmatic dependent launch intrinsics.

    On CUDA these emit cudaTriggerProgrammaticLaunchCompletion() and
    cudaGridDependencySynchronize(). Metal has no PDL: dispatches on a
    command stream are already ordered, so the codegen lowers both calls
    to no-ops and kernels written with them remain portable.
    """

    @T.prim_func
    def main(A: T.Tensor((N,), dtype), B: T.Tensor((N,), dtype)):
        with T.Kernel(T.ceildiv(N, 256), threads=256) as bx:
            T.pdl_trigger()
            for i in T.Parallel(256):
                B[bx * 256 + i] = A[bx * 256 + i] * 2.0
            T.pdl_sync()

    kernel = tilelang.compile(main, target="metal")
    a = torch.randn(N, device="mps", dtype=torch_dtype)
    b = torch.zeros(N, device="mps", dtype=torch_dtype)
    kernel(a, b)
    torch.mps.synchronize()
    torch.testing.assert_close(b, a * 2.0, rtol=rtol, atol=atol)

    # nothing PDL/CUDA-specific may leak into the generated MSL
    src = kernel.get_kernel_source(kernel_only=False)
    assert "pdl" not in src.lower()
    assert "cuda" not in src.lower()


@tilelang.testing.requires_metal
def test_pdl_intrinsics_fp32():
    _run_pdl_elementwise(4096, T.float32, torch.float32, rtol=1e-5, atol=1e-6)


@tilelang.testing.requires_metal
def test_pdl_intrinsics_bf16():
    _run_pdl_elementwise(4096, T.bfloat16, torch.bfloat16, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    if torch.mps.is_available():
        tilelang.testing.main()
