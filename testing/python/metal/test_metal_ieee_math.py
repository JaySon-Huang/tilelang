import pytest
import torch
import tilelang
import tilelang.language as T
import tilelang.testing


def _run_ieee_fmaf(M, N, block_M, block_N, dtype, torch_dtype, rtol, atol):
    """Compile and run a kernel whose body is a pure tl.ieee_fmaf on Metal.

    Note: preallocates the output buffer because the Metal torch adapter does
    not yet support JIT-allocated outputs (out_idx returns None).
    """
    rounding_mode = "rn"

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
        D: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            for i, j in T.Parallel(block_M, block_N):
                D[by * block_M + i, bx * block_N + j] = T.ieee_fmaf(
                    A[by * block_M + i, bx * block_N + j],
                    B[by * block_M + i, bx * block_N + j],
                    C[by * block_M + i, bx * block_N + j],
                    rounding_mode,
                )

    kernel = tilelang.compile(
        main,
        target="metal",
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False},
    )
    a = torch.randn(M, N, device="mps", dtype=torch_dtype)
    b = torch.randn(M, N, device="mps", dtype=torch_dtype)
    c = torch.randn(M, N, device="mps", dtype=torch_dtype)
    d = torch.zeros(M, N, device="mps", dtype=torch_dtype)
    kernel(a, b, c, d)
    torch.mps.synchronize()

    ref = torch.addcmul(c, a, b)  # a * b + c (unfused reference)
    torch.testing.assert_close(d, ref, rtol=rtol, atol=atol)


@tilelang.testing.requires_metal
def test_ieee_fmaf_fp32():
    _run_ieee_fmaf(64, 64, 16, 16, T.float32, torch.float32, rtol=1e-5, atol=1e-6)


@tilelang.testing.requires_metal
def test_ieee_fmaf_fp16():
    _run_ieee_fmaf(64, 64, 16, 16, T.float16, torch.float16, rtol=1e-2, atol=1e-2)


@tilelang.testing.requires_metal
def test_ieee_fmaf_unsupported_rounding_mode_raises():
    """Metal exposes only round-to-nearest fused multiply-add."""

    @T.prim_func
    def main(
        A: T.Tensor((8, 8), T.float32),
        B: T.Tensor((8, 8), T.float32),
        C: T.Tensor((8, 8), T.float32),
        D: T.Tensor((8, 8), T.float32),
    ):
        with T.Kernel(1, 1, threads=64) as (bx, by):
            for i, j in T.Parallel(8, 8):
                D[i, j] = T.ieee_fmaf(A[i, j], B[i, j], C[i, j], "rz")

    with pytest.raises(RuntimeError, match="round-to-nearest"):
        tilelang.compile(main, target="metal")


if __name__ == "__main__":
    if torch.mps.is_available():
        tilelang.testing.main()
