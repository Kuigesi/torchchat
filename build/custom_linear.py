import os
import torch
import torch.nn as nn
from torch import Tensor
import torch
import triton
import triton.language as tl

from my_timer import mytimer

FP32_BLOCK_SIZE_M = 1
FP32_BLOCK_SIZE_N = 256
BF16_BLOCK_SIZE_M = 16
BF16_BLOCK_SIZE_N = 64
"""
Kernel for computing Y = A @ X, where A is a dense matrix with
M rows and N columns.
- Input X has shape (N,)
- A has shape (M, N)
- Output has shape (M,)
"""
@triton.jit
def gemv_kernel_bf16(
    Y,
    A,
    X,
    M,
    N,
    stride_am,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    rm = start_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = tl.arange(0, BLOCK_SIZE_N)
    A = A + (rm[:, None] * stride_am + rn[None, :])
    X = X + rn
    acc = tl.zeros((BLOCK_SIZE_M, ), dtype=tl.float32)
    for n in range(N, 0, -BLOCK_SIZE_N):
        a = tl.load(A)
        x = tl.load(X)
        acc += tl.sum(a * x[None, :], axis=1)
        A += BLOCK_SIZE_N
        X += BLOCK_SIZE_N
    y = acc.to(tl.bfloat16)
    Y = Y + rm
    tl.store(Y, y)
@triton.jit
def gemv_kernel_f32(
    Y,
    A,
    X,
    M,
    N,
    stride_am,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    rm = start_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = tl.arange(0, BLOCK_SIZE_N)
    A = A + (rm[:, None] * stride_am + rn[None, :])
    X = X + rn
    acc = tl.zeros((BLOCK_SIZE_M, ), dtype=tl.float32)
    for n in range(N, 0, -BLOCK_SIZE_N):
        a = tl.load(A)
        x = tl.load(X)
        acc += tl.sum(a * x[None, :], axis=1)
        A += BLOCK_SIZE_N
        X += BLOCK_SIZE_N
    Y = Y + rm
    tl.store(Y, acc)
def gemv(
    weight: torch.Tensor,
    x: torch.Tensor,
    output: torch.Tensor,
):
    assert weight.shape[1] == x.shape[0], "Incompatible dimensions"
    assert weight.is_contiguous() and x.is_contiguous(), "Input and weight must be contiguous"
    assert x.dtype == weight.dtype, f"Input and weight must have the same dtype, got {x.dtype} and {weight.dtype}"
    M, N = weight.shape
    if x.dtype == torch.bfloat16:
        BLOCK_SIZE_M = BF16_BLOCK_SIZE_M
        BLOCK_SIZE_N = BF16_BLOCK_SIZE_N
    else:
        BLOCK_SIZE_M = FP32_BLOCK_SIZE_M
        BLOCK_SIZE_N = FP32_BLOCK_SIZE_N
    # TODO: Currently masked load is not supported yet.
    assert M % BLOCK_SIZE_M == 0 and N % BLOCK_SIZE_N == 0, "Masking currently not supported, Matrix dimensions must be multiples of block size"
    if output is None:
        # Allocates output.
        output = torch.empty((M, ), device=x.device, dtype=x.dtype)
    else:
        assert output.shape == (M, ) and output.dtype == x.dtype, "Incompatible output"
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]), )
    start_t = mytimer.record()
    if x.dtype == torch.bfloat16:
        gemv_kernel_bf16[grid](output, weight, x, M, N, weight.stride(0), BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N)
    else:
        gemv_kernel_f32[grid](output, weight, x, M, N, weight.stride(0), BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N)
    end_t = mytimer.record()
    mytimer.kernel_time += (end_t - start_t)
    return output
USE_TRITON = os.getenv("TORCHCHAT_USE_TRITON", "0") == "1"
class myLinear(nn.Linear):
    def forward(self, input: Tensor) -> Tensor:
        #print("calling my linear with")
        #print(f"weight: shape: {self.weight.shape}")
        #print(f"bias: {self.bias}")
        #print(f"vector: {input.shape}")
        if not USE_TRITON:
            start_t = mytimer.record()
            res = super().forward(input)
            end_t = mytimer.record()
            mytimer.linear_time += (end_t - start_t)
            return res
        elif (input.dim() == 3 and input.shape[0] == 1 and input.shape[1] == 1):
            #print(f"calling triton gemv")
            start_t = mytimer.record()
            #end_t = mytimer.record()
            res = gemv(self.weight, input.view(-1), None).view(1, 1, -1)
            tt_start = mytimer.record()
            tt_end = mytimer.record()
            end_t = mytimer.record()
            mytimer.all_time += (end_t - start_t)
            mytimer.timer_time += (tt_end - tt_start)
            #print(f"all: {mytimer.all_time} , kernel: {mytimer.kernel_time}, timer: {mytimer.timer_time}")
            return res
        else:
            #print(f"calling torch linear")
            start_t = mytimer.record()
            output = torch.empty((input.shape[0], input.shape[1], self.weight.shape[0]), device=input.device, dtype=input.dtype)
            assert input.shape[0] == 1
            for row_n in range(input.shape[1]):
                gemv(self.weight, input[0][row_n], output[0][row_n])
            #res = super().forward(input)
            end_t = mytimer.record()
            mytimer.all_time += (end_t - start_t)
            #return res
            return output
