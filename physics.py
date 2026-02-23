import torch

def physics_loss(model, t, r=0.2, K=1.0):
    t.requires_grad_(True)
    N = model(t)

    dN_dt = torch.autograd.grad(N, t, torch.ones_like(N), create_graph=True)[0]
    eq = dN_dt - r * N * (1 - N / K)

    return torch.mean(eq**2)
