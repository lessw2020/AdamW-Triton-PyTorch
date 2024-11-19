import torch 
from triton_adamw import TritonAdamW

def test_triton_adamw():
    # Create identical models
    torch_model = torch.nn.Linear(10, 10)
    triton_model = torch.nn.Linear(10, 10)
    triton_model.load_state_dict(torch_model.state_dict())

    # Create optimizers
    torch_opt = torch.optim.AdamW(torch_model.parameters())
    triton_opt = TritonAdamW(triton_model.parameters())

    # Create identical inputs
    x = torch.randn(32, 10)
    y = torch.randn(32, 10)

    # Training loop
    for _ in range(5):
        # PyTorch
        torch_out = torch_model(x)
        torch_loss = torch.nn.functional.mse_loss(torch_out, y)
        torch_loss.backward()
        torch_opt.step()
        torch_opt.zero_grad()

        # Triton
        triton_out = triton_model(x)
        triton_loss = torch.nn.functional.mse_loss(triton_out, y)
        triton_loss.backward()
        triton_opt.step()
        triton_opt.zero_grad()

        # Compare parameters
        for p1, p2 in zip(torch_model.parameters(), triton_model.parameters()):
            assert torch.allclose(p1, p2, rtol=1e-5, atol=1e-5)
