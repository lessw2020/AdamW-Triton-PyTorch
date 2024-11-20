import torch
import triton
import triton.language as tl
from torch.optim import Optimizer

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=32),
    ],
    key=['n_elements'],
)
@triton.jit
def adamw_kernel(
    params_ptr,
    grads_ptr,
    exp_avg_ptr,
    exp_avg_sq_ptr,
    lr,
    beta1,
    beta2,
    eps,
    weight_decay,
    step,
    bias_correction1,
    bias_correction2,
    n_elements,
    is_bias,
    BLOCK_SIZE: tl.constexpr,
    NUM_WARPS: tl.constexpr,
):
    # Calculate the starting offset for this program instance
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load parameters and states with block pointer arithmetic
    params = tl.load(params_ptr + offsets, mask=mask)
    grads = tl.load(grads_ptr + offsets, mask=mask)
    exp_avg = tl.load(exp_avg_ptr + offsets, mask=mask)
    exp_avg_sq = tl.load(exp_avg_sq_ptr + offsets, mask=mask)

    # Update biased first moment estimate
    exp_avg = beta1 * exp_avg + (1 - beta1) * grads
    
    # Update biased second raw moment estimate
    exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * (grads * grads)

    # Compute bias-corrected estimates
    exp_avg_corrected = exp_avg / bias_correction1
    exp_avg_sq_corrected = exp_avg_sq / bias_correction2
    
    # Compute denominator with numerical stability
    denom = tl.sqrt(exp_avg_sq_corrected) + eps
    
    # Compute the Adam update
    update = exp_avg_corrected / denom
    
    # Apply weight decay only if not a bias parameter
    if not is_bias:
        update = update + weight_decay * params
        
    # Apply final update with learning rate
    params = params - lr * update

    # Store updated values
    tl.store(params_ptr + offsets, params, mask=mask)
    tl.store(exp_avg_ptr + offsets, exp_avg, mask=mask)
    tl.store(exp_avg_sq_ptr + offsets, exp_avg_sq, mask=mask)

class TritonAdamW(Optimizer):
    def __init__(
        self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
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

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        
        # Initialize cache for autotuned configs
        self._kernel_configs = {}

    @staticmethod
    def _is_bias(param_name, param):
        """
        Determine if a parameter is a bias.
        This checks both the name and shape of the parameter.
        """
        return (param_name and 'bias' in param_name.lower()) or (len(param.shape) == 1)

    def _get_kernel_config(self, n_elements):
        """
        Get or create cached kernel configuration for given tensor size.
        """
        if n_elements not in self._kernel_configs:
            # Run a warmup iteration to trigger autotuning
            dummy_input = torch.zeros(n_elements, device='cuda')
            grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
            
            self._kernel_configs[n_elements] = adamw_kernel.warmup(
                dummy_input, dummy_input, dummy_input, dummy_input,
                1e-3, 0.9, 0.999, 1e-8, 0.01, 1, 1.0, 1.0, n_elements, False,
                grid=grid
            )
        return self._kernel_configs[n_elements]

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                # Ensure data is contiguous and on the correct device
                if not p.is_contiguous():
                    p.data = p.data.contiguous()
                if not grad.is_contiguous():
                    grad = grad.contiguous()
                
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format, device=p.device
                    )
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format, device=p.device
                    )
                    # Store whether this is a bias parameter
                    param_state = [name for name, param in p.named_parameters()] if hasattr(p, 'named_parameters') else []
                    param_name = param_state[0] if param_state else None
                    state["is_bias"] = self._is_bias(param_name, p)

                # Ensure all tensors are on the same device
                for key in ["exp_avg", "exp_avg_sq"]:
                    if state[key].device != p.device:
                        state[key] = state[key].to(p.device)

                # Update step count
                state["step"] += 1

                # Compute bias correction terms
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                # Get number of elements and kernel config
                n_elements = p.numel()
                kernel_config = self._get_kernel_config(n_elements)
                
                # Launch kernel with autotuned configuration
                grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
                adamw_kernel[grid](
                    p.data,
                    grad,
                    state["exp_avg"],
                    state["exp_avg_sq"],
                    group["lr"],
                    beta1,
                    beta2,
                    group["eps"],
                    group["weight_decay"],
                    state["step"],
                    bias_correction1,
                    bias_correction2,
                    n_elements,
                    state["is_bias"],
                )

        return loss
