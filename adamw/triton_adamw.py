import math

import torch
import triton
import triton.language as tl
from torch.optim import Optimizer


@triton.jit
def quantize_to_fp8(x, scale):
    """Simple FP8 E4M3 quantization"""
    # E4M3 max value is 448.0
    x_scaled = x / scale
    # Clamp to E4M3 range [-448, 448]
    x_clipped = tl.minimum(tl.maximum(x_scaled, -448.0), 448.0)
    # Round to nearest using floor(x + 0.5)
    x_rounded = tl.floor(x_clipped + 0.5)
    # Scale back
    return x_rounded * scale


@triton.jit
def adamw_kernel(
    params_ptr,
    grads_ptr,
    exp_avg_ptr,
    exp_avg_sq_ptr,
    scale_ptr,  # Scale factor for FP8 quantization
    lr,
    beta1,
    beta2,
    eps,
    weight_decay,
    bias_correction1,
    bias_correction2,
    n_elements,
    is_bias,
    use_fp8,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load
    params = tl.load(params_ptr + offsets, mask=mask)
    grads = tl.load(grads_ptr + offsets, mask=mask)
    exp_avg = tl.load(exp_avg_ptr + offsets, mask=mask)
    exp_avg_sq = tl.load(exp_avg_sq_ptr + offsets, mask=mask)
    scale = tl.load(scale_ptr)

    # Update momentum (exp_avg)
    exp_avg = beta1 * exp_avg + (1.0 - beta1) * grads

    # Quantize momentum if using FP8
    if use_fp8 == 1:
        exp_avg = quantize_to_fp8(exp_avg, scale)

    # Update second moment
    exp_avg_sq = beta2 * exp_avg_sq + (1.0 - beta2) * grads * grads

    # Bias correction
    step_size = lr / bias_correction1
    exp_avg_sq_corrected = exp_avg_sq / bias_correction2

    # Update parameters
    denom = tl.sqrt(exp_avg_sq_corrected) + eps

    # Weight decay
    if is_bias == 0:
        params = params * (1.0 - lr * weight_decay)

    # Final update
    params = params - step_size * (exp_avg / denom)

    # Store
    tl.store(params_ptr + offsets, params, mask=mask)
    tl.store(exp_avg_ptr + offsets, exp_avg, mask=mask)
    tl.store(exp_avg_sq_ptr + offsets, exp_avg_sq, mask=mask)


class TritonAdamW(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        use_fp8=False,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            use_fp8=use_fp8,
        )
        super().__init__(params, defaults)

    def _get_block_size(self, n_elements):
        """Get optimal block size based on number of elements"""
        if n_elements < 2048:
            return 128
        elif n_elements < 4096:
            return 256
        elif n_elements < 8192:
            return 512
        else:
            return 1024

    @staticmethod
    def _is_bias(param_name, param):
        return (param_name and "bias" in param_name.lower()) or (len(param.shape) == 1)

    def _update_scale(self, exp_avg):
        """Update scale factor for FP8 quantization"""
        with torch.no_grad():
            max_abs = torch.max(torch.abs(exp_avg))
            scale = torch.where(max_abs > 0, max_abs / 448.0, torch.ones_like(max_abs))
            min_scale = torch.full_like(scale, 1e-10)
            return torch.maximum(scale, min_scale)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            use_fp8 = group["use_fp8"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")

                if not p.is_contiguous():
                    p.data = p.data.contiguous()
                if not grad.is_contiguous():
                    grad = grad.contiguous()

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    if not use_fp8:
                        state["exp_avg"] = torch.zeros_like(
                            p,  # memory_format=torch.preserve_format
                        )
                    else:
                        state["exp_avg"] = torch.zeros_like(p, dtype=torch.uint8)
                    state["exp_avg_sq"] = torch.zeros_like(
                        p,  # memory_format=torch.preserve_format
                    )
                    state["scale"] = torch.ones(1, device=p.device)  # For FP8

                    param_state = (
                        [name for name, param in p.named_parameters()]
                        if hasattr(p, "named_parameters")
                        else []
                    )
                    param_name = param_state[0] if param_state else None
                    state["is_bias"] = self._is_bias(param_name, p)

                for key in ["exp_avg", "exp_avg_sq", "scale"]:
                    if state[key].device != p.device:
                        state[key] = state[key].to(p.device)

                state["step"] += 1

                # Update scale for FP8 if needed
                if use_fp8:
                    state["scale"] = self._update_scale(state["exp_avg"])

                bias_correction1 = 1.0 - beta1 ** state["step"]
                bias_correction2 = 1.0 - beta2 ** state["step"]

                n_elements = p.numel()
                block_size = self._get_block_size(n_elements)

                grid = (triton.cdiv(n_elements, block_size),)

                adamw_kernel[grid](
                    p.data,
                    grad,
                    state["exp_avg"],
                    state["exp_avg_sq"],
                    state["scale"],
                    group["lr"],
                    beta1,
                    beta2,
                    group["eps"],
                    group["weight_decay"],
                    bias_correction1,
                    bias_correction2,
                    n_elements,
                    int(state["is_bias"]),
                    int(use_fp8),
                    BLOCK_SIZE=block_size,
                )

        return loss
