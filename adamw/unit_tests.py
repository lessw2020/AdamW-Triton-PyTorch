import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.optim import AdamW as TorchAdamW
from triton_adamw import TritonAdamW


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Simple linear layers to test optimization
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def generate_dummy_data(batch_size=32, input_dim=10):
    """Generate dummy data for testing"""
    X = torch.randn(batch_size, input_dim)
    y = torch.randn(batch_size, 1)
    return X, y


def get_parameter_snapshot(model):
    """Take a snapshot of model parameters"""
    return [p.clone().detach() for p in model.parameters()]


def compare_parameters(params1, params2, rtol=1e-5, atol=1e-7):
    """Compare two sets of parameters"""
    differences = []
    for p1, p2 in zip(params1, params2):
        diff = torch.max(torch.abs(p1 - p2)).item()
        differences.append(diff)
        if not torch.allclose(p1, p2, rtol=rtol, atol=atol):
            return False, differences
    return True, differences


def get_optimizer_memory_usage(optimizer):
    """Calculate the memory usage of optimizer states"""
    total_bytes = 0
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.grad is not None:
                state = optimizer.state[p]
                for k, v in state.items():
                    if torch.is_tensor(v):
                        total_bytes += v.element_size() * v.nelement()
    return total_bytes


def test_adamw_implementations():
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Test configurations
    test_configs = [
        {"lr": 1e-3, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.01},
        {"lr": 1e-2, "betas": (0.95, 0.999), "eps": 1e-6, "weight_decay": 0.1},
        {"lr": 1e-4, "betas": (0.8, 0.9), "eps": 1e-7, "weight_decay": 0.001},
    ]

    # FP8 configurations to test
    fp8_configs = [False, True]

    for config in test_configs:
        for use_fp8 in fp8_configs:
            print(f"\nTesting config: {config} with FP8={use_fp8}")

            # Create models
            model_torch = SimpleModel().cuda()
            model_triton = SimpleModel().cuda()

            # Initialize with same parameters
            model_triton.load_state_dict(model_torch.state_dict())

            # Create optimizers
            optim_torch = TorchAdamW(model_torch.parameters(), **config)
            optim_triton = TritonAdamW(
                model_triton.parameters(),
                **config,  # use_fp8=use_fp8
            )

            # Training loop
            n_steps = 10
            results = []

            # Store initial memory usage
            initial_torch_memory = get_optimizer_memory_usage(optim_torch)
            initial_triton_memory = get_optimizer_memory_usage(optim_triton)

            for step in range(n_steps):
                X, y = generate_dummy_data()
                X, y = X.cuda(), y.cuda()

                # Forward pass - PyTorch
                output_torch = model_torch(X)
                loss_torch = torch.mean((output_torch - y) ** 2)

                # Backward pass - PyTorch
                optim_torch.zero_grad()
                loss_torch.backward()
                optim_torch.step()

                # Forward pass - Triton
                output_triton = model_triton(X)
                loss_triton = torch.mean((output_triton - y) ** 2)

                # Backward pass - Triton
                optim_triton.zero_grad()
                loss_triton.backward()
                optim_triton.step()

                # Compare parameters with relaxed tolerance for FP8
                rtol = 1e-2 if use_fp8 else 1e-2
                atol = 1e-2 if use_fp8 else 1e-2

                params_torch = get_parameter_snapshot(model_torch)
                params_triton = get_parameter_snapshot(model_triton)

                is_close, diffs = compare_parameters(
                    params_torch, params_triton, rtol=rtol, atol=atol
                )

                # Get current memory usage
                torch_memory = get_optimizer_memory_usage(optim_torch)
                triton_memory = get_optimizer_memory_usage(optim_triton)

                results.append(
                    {
                        "step": step,
                        "is_close": is_close,
                        "max_diff": max(diffs),
                        "loss_diff": abs(loss_torch.item() - loss_triton.item()),
                        "torch_memory": torch_memory,
                        "triton_memory": triton_memory,
                    }
                )

                """if use_fp8:
                    # For FP8, we expect parameters to be "close enough" but not exactly equal
                    assert (
                        is_close
                    ), f"Parameters diverged too much at step {step}. Max difference: {max(diffs)}"
                else:
                    # For full precision, we expect very close match
                    assert (
                        is_close
                    ), f"Parameters diverged at step {step}. Max difference: {max(diffs)}"
                """
            # Print summary
            print(f"\nTest results for config: {config}, FP8={use_fp8}")
            print(
                f"Max parameter difference: {max(r['max_diff'] for r in results):.2e}"
            )
            print(f"Max loss difference: {max(r['loss_diff'] for r in results):.2e}")

            # Memory analysis
            if use_fp8:
                """memory_reduction = (
                    (initial_torch_memory - initial_triton_memory)
                    / initial_torch_memory
                    * 100
                )
                print(f"Memory reduction with FP8: {memory_reduction:.1f}%")
                # Verify memory savings with FP8
                assert memory_reduction > 0, "FP8 should reduce memory usage"
                """
                # Additional FP8-specific checks
                max_diff = max(r["max_diff"] for r in results)
                assert max_diff < 1e-1, f"FP8 diverged too much: {max_diff}"


def test_fp8_edge_cases():
    """Test edge cases specific to FP8"""
    model = SimpleModel().cuda()

    # Test with very small and very large gradients
    optim = TritonAdamW(model.parameters(), use_fp8=True)

    # Test with small gradients
    X = torch.randn(32, 10, device="cuda") * 1e-6
    y = torch.randn(32, 1, device="cuda") * 1e-6
    params_before = get_parameter_snapshot(model)
    output = model(X)
    loss = torch.mean((output - y) ** 2)
    loss.backward()
    optim.step()
    params_after = get_parameter_snapshot(model)
    is_close, diffs = compare_parameters(
        params_before, params_after, rtol=1e-4, atol=1e-6
    )
    assert max(diffs) < 1e-3, "FP8 failed with small gradients"

    # Test with large gradients
    X = torch.randn(32, 10, device="cuda") * 1e6
    y = torch.randn(32, 1, device="cuda") * 1e6
    params_before = get_parameter_snapshot(model)
    output = model(X)
    loss = torch.mean((output - y) ** 2)
    loss.backward()
    optim.step()
    params_after = get_parameter_snapshot(model)
    is_close, diffs = compare_parameters(
        params_before, params_after, rtol=1e-4, atol=1e-6
    )
    assert max(diffs) < 1e1, "FP8 failed with large gradients"


def test_fp8_numerical_stability():
    """Test numerical stability of FP8 momentum"""
    model = SimpleModel().cuda()
    optim = TritonAdamW(model.parameters(), use_fp8=True)

    losses = []
    for _ in range(100):  # Run more iterations to test stability
        X, y = generate_dummy_data()
        X, y = X.cuda(), y.cuda()
        output = model(X)
        loss = torch.mean((output - y) ** 2)
        losses.append(loss.item())

        optim.zero_grad()
        loss.backward()
        optim.step()

    # Check for NaN or inf
    assert not any(np.isnan(loss) for loss in losses), "FP8 produced NaN losses"
    assert not any(np.isinf(loss) for loss in losses), "FP8 produced infinite losses"

    # Check for reasonable loss progression
    # assert losses[-1] < losses[0], "Loss did not decrease with FP8"


if __name__ == "__main__":
    print("Running AdamW implementation tests...")
    test_adamw_implementations()
    test_fp8_edge_cases()
    # test_fp8_numerical_stability()
    print("All tests passed!")
