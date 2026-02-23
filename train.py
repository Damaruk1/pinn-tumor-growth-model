import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from model import PINN
from physics import physics_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PINN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

t = torch.linspace(0, 5, 50).view(-1, 1).to(device)
N_true = 0.1 * torch.exp(0.2 * t) / (1 + 0.1 * (torch.exp(0.2 * t) - 1))

for epoch in range(2000):
    t_col = torch.rand(100, 1).to(device) * 5

    data_loss = torch.mean((model(t) - N_true)**2)
    phys_loss = physics_loss(model, t_col)

    loss = data_loss + phys_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 200 == 0:
        print(epoch, loss.item())

t_test = torch.linspace(0, 5, 100).view(-1, 1).to(device)
pred = model(t_test).detach().cpu().numpy()

plt.plot(t_test.cpu(), pred, label="Predicted Tumor Size")
plt.plot(t.cpu(), N_true.cpu(), "o", label="Data")
plt.legend()
plt.title("Tumor Growth PINN")
plt.show()
