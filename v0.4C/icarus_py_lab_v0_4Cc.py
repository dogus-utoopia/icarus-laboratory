# ICARUS — Laboratory Implementation
# Version: v0.4Cc
# Hypothesis: Absence of Proportional Modulation
# Status: LAB — NON-FINAL
# Author: Luis Javier Carrasco Pérez
# Description:
# Continuation direct of ICARUS v0.3.
# No proportional modulation (phi) is present or required.
# The modulation parameter lambda is constant and independent of internal viability V.

import torch
import torch.nn.functional as F


class ICARUS:
    def __init__(
        self,
        dim=16,
        noise_scale=0.05,
        constant_lambda=0.1,
        device="cpu"
    ):
        self.dim = dim
        self.noise_scale = noise_scale
        self.constant_lambda = constant_lambda
        self.device = device

        self.U = F.softmax(torch.randn(dim, device=device), dim=0)

    def expand(self):
        noise = self.noise_scale * torch.randn(self.dim, device=self.device)
        X = self.U + noise
        return X

    def integrate(self, X):
        Y = F.softmax(X, dim=0)
        return Y

    def evaluate_viability(self, X, Y):
        dist = torch.norm(X - Y, p=2)
        V = 1.0 / (1.0 + dist)
        return V

    def update_state(self, Y):
        lam = self.constant_lambda
        self.U = F.softmax((1.0 - lam) * self.U + lam * Y, dim=0)
        return lam

    def step(self):
        X = self.expand()
        Y = self.integrate(X)
        V = self.evaluate_viability(X, Y)
        lam = self.update_state(Y)
        return {
            "U": self.U.clone(),
            "V": V.item(),
            "lambda": lam
        }
