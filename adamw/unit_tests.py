 import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW as TorchAdamW
import pytest

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

def test_adamw_implementations():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Test configurations
    test_configs = [
        {'lr': 1e-3, 'betas': (0.9, 0.999), 'eps': 1e-8, 'weight_decay': 0.01},
        {'lr': 1e-2, 'betas': (0.95, 0.999), 'eps': 1e-6, 'weight_decay': 0.1},
        {'lr': 1e-4, 'betas': (0.8, 0.9), 'eps': 1e-7, 'weight_decay': 0.001},
    ]
    
    for config in test_configs:
        # Create two identical models
        model_torch = SimpleModel()
        model_triton = SimpleModel()
        
        # Initialize with same parameters
        model_triton.load_state_dict(model_torch.state_dict())
        
        # Create optimizers
        optim_torch = TorchAdamW(model_torch.parameters(), **config)
        optim_triton = TritonAdamW(model_triton.parameters(), **config)
        
        # Training loop
        n_steps = 10
        results = []
        
        for step in range(n_steps):
            X, y = generate_dummy_data()
            
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
            
            # Compare parameters
            params_torch = get_parameter_snapshot(model_torch)
            params_triton = get_parameter_snapshot(model_triton)
            
            is_close, diffs = compare_parameters(params_torch, params_triton)
            results.append({
                'step': step,
                'is_close': is_close,
                'max_diff': max(diffs),
                'loss_diff': abs(loss_torch.item() - loss_triton.item())
            })
            
            assert is_close, f"Parameters diverged at step {step}. Max difference: {max(diffs)}"
            
        # Print summary
        print(f"\nTest results for config: {config}")
        print(f"Max parameter difference across all steps: {max(r['max_diff'] for r in results):.2e}")
        print(f"Max loss difference across all steps: {max(r['loss_diff'] for r in results):.2e}")

def test_edge_cases():
    """Test edge cases and corner cases"""
    model = SimpleModel()
    
    # Test with zero learning rate
    optim = TritonAdamW(model.parameters(), lr=0.0)
    params_before = get_parameter_snapshot(model)
    X, y = generate_dummy_data()
    output = model(X)
    loss = torch.mean((output - y) ** 2)
    loss.backward()
    optim.step()
    params_after = get_parameter_snapshot(model)
    is_close, _ = compare_parameters(params_before, params_after)
    assert is_close, "Parameters changed with zero learning rate"
    
    # Test with zero gradients
    optim = TritonAdamW(model.parameters())
    params_before = get_parameter_snapshot(model)
    optim.zero_grad()
    optim.step()
    params_after = get_parameter_snapshot(model)
    is_close, _ = compare_parameters(params_before, params_after)
    assert is_close, "Parameters changed with zero gradients"

if __name__ == "__main__":
    print("Running AdamW implementation tests...")
    test_adamw_implementations()
    test_edge_cases()
    print("All tests passed!")
