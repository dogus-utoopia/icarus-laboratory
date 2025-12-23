import torch
import torch.nn.functional as F

# -----------------------------
# Dimensionality & steps
# -----------------------------
d = 64
steps = 500

# -----------------------------
# Base parameters
# -----------------------------
lambda_base = 0.1
alpha, beta, gamma = 1.0, 0.5, 0.3
noise_base = 1.0

# -----------------------------
# Modulation parameters
# -----------------------------
k_lambda = 0.3
mu_lambda = 0.9

# -----------------------------
# Initial states
# -----------------------------
U = torch.zeros(d)
lambda_mem = torch.tensor(lambda_base)

# -----------------------------
# Logs
# -----------------------------
U_norm_log = []
lambda_log = []

# -----------------------------
# ICARUS loop
# -----------------------------
for t in range(steps):

    # Expansion
    noise = noise_base * torch.randn(d)
    X = U + noise

    # Integration
    Y = F.softmax(X, dim=0)

    # Residual
    R = X - Y

    # Metrics
    C = torch.norm(Y) / (torch.norm(X) + 1e-6)
    D = torch.norm(R)
    R_load = torch.norm(X)

    # Value
    V = alpha * C - beta * D - gamma * R_load

    
    lambda_inst = lambda_base * (1 + k_lambda * torch.tanh(V))
    lambda_inst = torch.clamp(lambda_inst, 0.01, 0.5)

    
    lambda_mem = mu_lambda * lambda_mem + (1 - mu_lambda) * lambda_inst

    # State update
    U = (1 - lambda_mem) * U + lambda_mem * Y

    # Logging
    U_norm_log.append(torch.norm(U).item())
    lambda_log.append(lambda_mem.item())

# -----------------------------
# Diagnostics
# -----------------------------
print("Final UIS norm:", torch.norm(U).item())
print("Final lambda:", lambda_mem.item())
