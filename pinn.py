import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt

# -------------------- data --------------------
data = pd.read_excel(r"C:\Users\Aaron\Desktop\MAT292\Project\data\trial4_fall.xlsx")

t_raw = torch.tensor(data["delta_t_s"].values, dtype=torch.float32).unsqueeze(1)
y_raw = torch.tensor(data["bpm"].values, dtype=torch.float32).unsqueeze(1)

# ---------- normalization (z-score) ----------
t_mean, t_std = t_raw.mean(), t_raw.std()
y_mean, y_std = y_raw.mean(), y_raw.std()

t_data = (t_raw - t_mean) / t_std
y_data = (y_raw - y_mean) / y_std

tmin, tmax = float(t_raw.min()), float(t_raw.max())

# subject-level HR limits (raw scale)
hr_min = 65.0
hr_max = 200.0

# convert into normalized scale for ODE computation
hr_min_n = (torch.tensor([[hr_min]]) - y_mean) / y_std
hr_max_n = (torch.tensor([[hr_max]]) - y_mean) / y_std

# ---------------- physics collocation points ----------------
N_phys = 2000
t_phys_raw = torch.empty(N_phys, 1).uniform_(tmin, tmax)
t_phys = (t_phys_raw - t_mean) / t_std
t_phys.requires_grad_(True)

# -------------------- model --------------------
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)
        self.act = nn.Tanh()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.fc3(x)
        return x

model = Net()

# PINN ODE parameters (RAW scale for D; exponents dimensionless)
A = nn.Parameter(torch.tensor(0.54))
B = nn.Parameter(torch.tensor(1.6))
C = nn.Parameter(torch.tensor(1.75))
E = nn.Parameter(torch.tensor(1.0))
D = nn.Parameter(torch.tensor(100.0))  # target HR for this trial

# all trainable parameters
params = list(model.parameters()) + [A, B, C, E, D]
optimizer = optim.Adam(params, lr=1e-3)

lambda_data = 1.0
lambda_ode = 0.1

# -------------------- training --------------------
epochs = 10000
for epoch in range(epochs):

    # -------- data loss --------
    y_pred_data = model(t_data)
    data_loss = torch.mean((y_pred_data - y_data) ** 2)

    # -------- ODE loss --------
    H = model(t_phys)
    dydt = torch.autograd.grad(
        outputs=H,
        inputs=t_phys,
        grad_outputs=torch.ones_like(H),
        create_graph=True
    )[0]

    # convert D to normalized
    D_n = (D - y_mean) / y_std

    # ODE terms
    term1 = torch.clamp(H - hr_min_n, min=1e-6)
    term2 = torch.clamp(hr_max_n - H, min=1e-6)
    term3 = torch.clamp(D_n - H, min=1e-6)

    rhs = A * term1**B * term2**C * term3**E

    ode_res = dydt - rhs
    ode_loss = torch.mean(ode_res ** 2)

    loss = lambda_data * data_loss + lambda_ode * ode_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # print summary
    if (epoch + 1) % 500 == 0:
        print(
            f"Epoch {epoch+1:5d} | Loss={loss.item():.6f} "
            f"(data={data_loss.item():.4f}, ode={ode_loss.item():.4f}) "
            f"A={A.item():.3e}, B={B.item():.2f}, C={C.item():.2f}, "
            f"E={E.item():.2f}, D ={D.item():.2f}"
        )


# -------------------- plot --------------------
t_plot_raw = torch.linspace(tmin, tmax, 400).unsqueeze(1)
t_plot = (t_plot_raw - t_mean) / t_std

with torch.no_grad():
    y_plot_norm = model(t_plot)
    y_plot_raw = y_plot_norm * y_std + y_mean

plt.figure(figsize=(8, 5))
plt.scatter(t_raw.numpy(), y_raw.numpy(), s=12, label="Data")
plt.plot(t_plot_raw.numpy(), y_plot_raw.numpy(), label="PINN solution", linewidth=2)
plt.xlabel("t (s)")
plt.ylabel("Heart rate (bpm)")
plt.legend()
plt.title("PINN Fitted HR vs. Time")
plt.show()
