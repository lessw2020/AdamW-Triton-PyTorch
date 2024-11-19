import torch 
from triton_adamw import TritonAdamW

def test_triton_adamw():
    # Create identical models
    torch_model = torch.nn.Linear(10, 10)
    triton_model = torch.nn.Linear(10, 10)
    triton_model.load_state_dict(torch_model.state_dict())

    # move to gpu
    torch_model.cuda()
    triton_model.cuda()

    lr = 0.1  # make it easy to test
    betas = (0.9, 0.999)

    # Create optimizers
    torch_opt = torch.optim.AdamW(torch_model.parameters(), betas = betas, lr=lr, eps=1e-6)
    triton_opt = TritonAdamW(triton_model.parameters(), lr=lr, betas=betas, eps=1e-6)

    # Create identical inputs
    x = torch.randn(32, 10).to('cuda')
    y = torch.randn(32, 10).to('cuda')

    # Training loop
    for i in range(5):
        # PyTorch
        torch_out = torch_model(x)
        torch_loss = torch.nn.functional.mse_loss(torch_out, y)
        torch_loss.backward()
        torch_opt.step()
        torch_opt.zero_grad()

        # Triton
        print(f"\npre-optimizer {triton_model.parameters()=}\n")
        triton_out = triton_model(x)
        triton_loss = torch.nn.functional.mse_loss(triton_out, y)
        triton_loss.backward()
        triton_opt.step()
        triton_opt.zero_grad()
        print(f"\npost-optimizer {triton_model.parameters()=}\n")

        # Compare parameters
        print(f"Comparing parameters...{i=}")
        for p1, p2 in zip(torch_model.parameters(), triton_model.parameters()):
            assert torch.allclose(p1, p2, rtol=1e-4, atol=1e-4)

if __name__ == "__main__":
    test_triton_adamw()
    print("All tests passed!")
