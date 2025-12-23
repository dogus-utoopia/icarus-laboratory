"""
ICARUS — PyTorch Prototype v0.3
Closed-Cycle Internal Dynamics
Active Homeostatic Modulation

Author: Luis Javier Carrasco Pérez
Date: December 2025

Purpose:
Evaluate whether introducing active homeostatic modulation
into the closed HRM–ICM–UIS loop preserves stability,
induces regulation, or causes collapse.

This prototype makes no claims about consciousness,
intelligence, or task performance.
"""

import torch
import torch.nn.functional as F

# -----------------------------
# Hyperparameters (fixed)
# -----------------------------
d = 64                  # UIS dimensionality
lambda_base = 0.1       # base UIS integration rate
alpha, beta, gamma = 1.0, 0.5, 0.3
f_base = 1.0
k_freq = 0.25
noise_scale_base = 1.0
steps = 500

# -----------------------------
# Initial State
# -----------------------------
U = torch.zeros(d)      # Unified Internal State

# -----------------------------
# Metrics logging
# -----------------------------
V_log = []
U_norm_log = []
lambda_log = []
noise_log = []

# -----------------------------
# ICARUS closed-cycle
# -----------------------------
for t in range(steps):

    # ---------- Expansion ----------
    noise_scale = noise_scale_base
    noise = noise_scale * torch.randn(d)
    X = U + noise

    # ---------- Integration (ICM) ----------
    W = torch.eye(d)                  # identity integration (v0.x)
    Y = F.softmax(W @ X, dim=0)

    # ---------- Residual ----------
    R = X - Y

    # ---------- Coherence ----------
    C = torch.norm(Y) / (torch.norm(X) + 1e-6)

    # ---------- Deviation ----------
    D = torch.norm(R)

    # ---------- Resource proxy ----------
    R_load = torch.norm(X)

    # ---------- Homeostatic value ----------
    V = alpha * C - beta * D - gamma * R_load

    # ---------- Active modulation ----------
    f = f_base + k_freq * V
    f = torch.clamp(f, 0.2, 2.0)

    # Modulate integration rate
    lambda_u = torch.clamp(lambda_base * f, 0.01, 0.5)

    # Modulate expansion amplitude
    noise_scale = noise_scale_base / f

    # ---------- UIS update ----------
    U = (1 - lambda_u) * U + lambda_u * Y

    # ---------- Logging ----------
    V_log.append(V.item())
    U_norm_log.append(torch.norm(U).item())
    lambda_log.append(lambda_u.item())
    noise_log.append(noise_scale)

# -----------------------------
# Results
# -----------------------------
print("Final UIS norm:", torch.norm(U).item())
print("Value function (last 10):", V_log[-10:])
print("Lambda_u (last 10):", lambda_log[-10:])
print("Noise scale (last 10):", noise_log[-10:])
