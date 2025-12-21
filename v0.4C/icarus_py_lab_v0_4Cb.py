# ICARUS — Laboratory Implementation
# Version: v0.4Cb
# Hypothesis: Imposed Proportional Modulation
# Status: LAB — NON-FINAL
# Author: Luis Javier Carrasco Pérez
# Description:
# Continuation direct of ICARUS v0.3.
# Proportional modulation (phi) is explicitly imposed as a structural constraint.
# The modulation parameter lambda is a predefined proportional function of internal viability V.

import torch
import torch.nn.functional as F


class ICARUS_v0_4Cb:
    def __init__(
        self,
        dim=16,
        noise_scale=0.05,
        base_rate=0.05,
        gain=0.4,
        lambda_min=0.01,
        lambda_max=0.5,
        seed=None,
    ):
        if seed is not None:
            torch.manual_seed(seed)

        self.dim = dim
        self.noise_scale = noise_scale

        # Explicit proportional parameters
        self.base_rate = base_rate
        self.gain = gain
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max

        # Unified Internal State
        self.U = F.softmax(torch.randn(dim), dim=0)

    def expand(self):
        noise = torch.randn(self.dim) * self.noise_scale
        X = self.U + noise
        return X

    def integrate(self, X):
        Y = F.softmax(X, dim=0)
        return Y

    def evaluate_viability(self, X, Y):
        # Viability inversely proportional to divergence
        dist = torch.norm(X - Y, p=2)
        V = 1.0 / (1.0 + dist)
        return V

    def derive_lambda_imposed(self, V):
        # Explicit proportional rule
        lam = self.base_rate + self.gain * V
        lam = torch.clamp(lam, self.lambda_min, self.lambda_max)
        return lam

    def update_state(self, Y, lam):
        U_next = (1.0 - lam) * self.U + lam * Y
        U_next = F.softmax(U_next, dim=0)
        self.U = U_next

    def step(self):
        X = self.expand()
        Y = self.integrate(X)
        V = self.evaluate_viability(X, Y)
        lam = self.derive_lambda_imposed(V)
        self.update_state(Y, lam)
        return {
            "U": self.U.clone(),
            "X": X.clone(),
            "Y": Y.clone(),
            "V": V.item(),
            "lambda": lam.item(),
        }


if __name__ == "__main__":
    model = ICARUS_v0_4Cb(dim=8, seed=42)
    for t in range(10):
        out = model.step()
        print(
            f"t={t} | V={out['V']:.4f} | lambda={out['lambda']:.4f}"
        )
