# pyre-ignore-all-errors
import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from torch._inductor.runtime.triton_compat import libdevice


def check_and_create_outs(x_orig, weight, bias, eps):
    """
    Performs 1D layer normalization on the input tensor using Helion.
    Args:
        x (torch.Tensor): Input tensor of shape [batch_size, dim], expected to be FP16.
        normalized_shape (list[int]): List containing the dimension to normalize over (should be length 1).
        weight (torch.Tensor): Learnable scale parameter of shape [dim].
        bias (torch.Tensor | None): Optional learnable bias parameter of shape [dim].
        eps (float, optional): Small value added to variance for numerical stability. Default is 1e-5.
    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - The layer-normalized output tensor of shape [batch_size, dim], in FP16.
            - Mean tensor of shape [batch_size], in FP32.
            - Reciprocal standard deviation tensor of shape [batch_size], in FP32.
    """

    normalized_shape = (x_orig.shape[-1],)
    x = x_orig.reshape(-1, x_orig.shape[-1])
    m, n = x.size()
    assert weight.size(0) == n, f"weight size mismatch {weight.size(0)} != {n}"
    if bias is not None:
        assert bias.size(0) == n, f"bias size mismatch {bias.size(0)} != {n}"
    assert (
        len(normalized_shape) == 1
    ), "Helion layer norm only supports 1D layer norm currently"
    assert (
        normalized_shape[0] == n
    ), f"normalized shape mismatch {normalized_shape[0]} != {n}"
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    mean = torch.empty([m], dtype=torch.float32, device=x.device)
    rstd = torch.empty([m], dtype=torch.float32, device=x.device)
    return (x, mean, rstd, weight, bias, out, eps)


@triton.jit
def compute_multi_cta_sum(
    x, cta_cluster_rank, BLOCK_SIZE_M: tl.constexpr, num_reduction_ctas: tl.constexpr
):
    dtype_x = tlx.dtype_of(x)
    local_buff = tlx.local_alloc((BLOCK_SIZE_M, 1), dtype_x, num_reduction_ctas)
    local_partial_sum = tl.sum(x, axis=1, keep_dims=True)
    for i in range(num_reduction_ctas, loop_unroll_factor=num_reduction_ctas):
        remote_local_buff_view = tlx.local_view(local_buff, cta_cluster_rank)
        tlx.remote_shmem_store(
            dst=remote_local_buff_view, src=local_partial_sum, remote_cta_rank=i
        )
    tlx.cluster_barrier()

    final_sum = tl.zeros((BLOCK_SIZE_M, 1), dtype=dtype_x)
    for i in range(num_reduction_ctas, loop_unroll_factor=num_reduction_ctas):
        remote_local_buff_view = tlx.local_view(local_buff, i)
        final_sum += tlx.local_load(remote_local_buff_view)
    # tlx.cluster_barrier()
    return final_sum


kernel_configs_multi_cta = [
    triton.Config({"BLOCK_SIZE_M": m, "num_reduction_ctas": ctas}, num_warps=nw)
    for m in [1, 2]
    for nw in [1, 2, 4, 8, 16, 32]
    for ctas in [2, 4, 8]
]


@triton.autotune(
    configs=kernel_configs_multi_cta,
    key=["M", "N"],
)
@triton.heuristics(
    {
        "BLOCK_SIZE_N": lambda args: triton.next_power_of_2(
            args["N"] // args["num_reduction_ctas"]
        )
    }
)
@triton.jit
def kernel_norm_multi_cta(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    Mean_out,  # pointer to the mean
    Rstd_out,  # pointer to the 1/std
    row_stride,  # input row stride
    M,  # number of rows in X
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    num_reduction_ctas: tl.constexpr,
):
    cta_cluster_rank = tlx.cluster_cta_rank()
    tlx.set_num_reduction_ctas(num_reduction_ctas)

    row_offsets = tl.program_id(0) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    # Partition reduction axes over multiple CTAs
    col_offsets = (tl.program_id(1) % num_reduction_ctas) * BLOCK_SIZE_N + tl.arange(
        0, BLOCK_SIZE_N
    )

    # mask_row = row_offsets < M
    # mask_col = col_offsets < N
    read_write_offsets = (row_offsets[:, None] * row_stride) + col_offsets[None, :]
    # read_write_mask = mask_row[:, None] & mask_col[None, :]

    X_ptr = X + read_write_offsets
    Y_ptr = Y + read_write_offsets

    # x = tl.load(X_ptr, mask=read_write_mask, other=0.0).to(tl.float32)
    # x = tl.load(X_ptr).to(tl.float32)
    x_load = tl.load(X_ptr)
    x = tl.cast(x_load, tl.float32)

    tlx.cluster_barrier()
    multi_cta_sum = compute_multi_cta_sum(
        x, cta_cluster_rank, BLOCK_SIZE_M, num_reduction_ctas
    )
    mean = multi_cta_sum / N
    # x_minus_mean = tl.where(read_write_mask, x - mean, 0.0)
    x_minus_mean = x - mean
    x_minus_mean_sq = x_minus_mean * x_minus_mean

    multi_cta_sum_x_minus_mean_sq = compute_multi_cta_sum(
        x_minus_mean_sq, cta_cluster_rank, BLOCK_SIZE_M, num_reduction_ctas
    )
    var = multi_cta_sum_x_minus_mean_sq / N
    rstd = libdevice.rsqrt(var + eps)
    mean_1d = tl.reshape(mean, (BLOCK_SIZE_M,))
    # tl.store(Mean_out + row_offsets, mean_1d, mask=mask_row)
    tl.store(Mean_out + row_offsets, mean_1d)

    rstd_1d = tl.reshape(rstd, (BLOCK_SIZE_M,))
    # tl.store(Rstd_out + row_offsets, rstd_1d, mask=mask_row)
    # w = tl.load(W + col_offsets, mask=mask_col).to(tl.float32)
    # b = tl.load(B + col_offsets, mask=mask_col).to(tl.float32)
    w = tl.load(W + col_offsets).to(tl.float32)
    b = tl.load(B + col_offsets).to(tl.float32)
    tl.store(Rstd_out + row_offsets, rstd_1d)
    x_hat = (x - mean) * rstd
    y = x_hat * w + b
    y = tl.cast(y, Y_ptr.dtype.element_ty)
    # tl.store(Y_ptr, y, mask=read_write_mask)
    tl.store(Y_ptr, y)


def _tlx_multi_cta_layernorm_fwd_with_bias(x, mean, rstd, weight, bias, out, eps):
    def grid_2d(meta):
        M = x.size(0)
        N = x.size(1)
        # print(
        #     f"Using 2D grid. {M=} {N=} {meta["BLOCK_SIZE_M"]=} {meta["BLOCK_SIZE_N"]=}"
        # )
        grid = (
            triton.cdiv(M, meta["BLOCK_SIZE_M"]),
            triton.cdiv(N, meta["BLOCK_SIZE_N"]),
        )
        # print(f"{grid=}")
        return grid

    M = x.size(0)
    N = x.size(1)
    # X, # pointer to the input
    # Y, # pointer to the output
    # W, # pointer to the weights
    # B, # pointer to the biases
    # Mean, # pointer to the mean
    # Rstd, # pointer to the 1/std
    # row_stride, # input row stride
    # M, # number of rows in X
    # N, # number of columns in X
    # eps, # epsilon to avoid division by zero
    # BLOCK_SIZE_N: tl.constexpr
    k = kernel_norm_multi_cta[grid_2d](
        X=x,
        Y=out,
        W=weight,
        B=bias,
        Mean_out=mean,
        Rstd_out=rstd,
        row_stride=x.stride(0),
        M=M,
        N=N,
        eps=eps,
    )


def tlx_multi_cta_layernorm_fwd_with_bias(x, weight, bias, eps):
    x, mean, rstd, weight, bias, out, eps = check_and_create_outs(
        x, weight, bias, eps=1e-5
    )
    _tlx_multi_cta_layernorm_fwd_with_bias(x, mean, rstd, weight, bias, out, eps)
    return (out, mean, rstd)


def _create_input_tensors(B, M, N, elementwise_affine, dtype, device):
        x_shape = (B, M, N)
        x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
        dy = 0.1 * torch.randn_like(x)
        weight, bias = None, None

        if elementwise_affine:
            w_shape = (x_shape[-1],)
            weight = torch.randn(
                w_shape, dtype=dtype, device=device, requires_grad=True
            )

            bias = torch.randn(
                w_shape, dtype=dtype, device=device, requires_grad=True
            )

        x.requires_grad_(True)

        return (x, dy, weight, bias)

def check_correctness_tlx_multi_cta_layernorm(x, weight, bias, eps, dtype) :
        w_shape = (x.shape[-1],)

        torch_x = x.detach().clone().requires_grad_(True)
        zeroBIAS = torch.zeros_like(weight).to(dtype).requires_grad_(True)
        y_ref = torch.nn.functional.layer_norm(torch_x, w_shape, weight, zeroBIAS, eps)
        y_ref = y_ref.reshape(-1, y_ref.shape[-1])

        x = x.reshape(-1, x.shape[-1])
        _x = x.clone().to(dtype).requires_grad_(True)
        _weight = weight.detach().clone().requires_grad_(True)
        y_tri, _ , _ = tlx_multi_cta_layernorm_fwd_with_bias(_x, _weight, zeroBIAS, eps=eps)
        try:
            torch.testing.assert_close(y_ref, y_tri, rtol=1e-5, atol=1e-5)
            print("PASS!")
        except AssertionError as error:
            print("FAIL!")
        
        return lambda: None

B = 1152
M = 1
N = 32768
DEVICE = triton.runtime.driver.active.get_active_torch_device()
DTYPE = torch.float32
x, dy, weight, bias = _create_input_tensors(B, M, N, elementwise_affine=True, dtype=DTYPE, device=DEVICE)
check_correctness_tlx_multi_cta_layernorm(x, weight, bias, eps=1e-5, dtype=DTYPE)
