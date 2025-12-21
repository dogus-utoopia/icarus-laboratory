# ICARUS v0.4Ca
# Emergent proportional modulation
# Direct continuation of v0.3
# No imposed proportionality, no explicit phi parameter

import torch
import torch.nn.functional as F


class ICARUS_v04Ca:
    def __init__(
        self,
        dim=16,
        noise_scale_base=0.05,
        lambda_min=0.01,
        lambda_max=0.5,
        device="cpu",
    ):
        self.dim = dim
        self.device = device

        # Unified Internal State
        self.U = F.softmax(torch.randn(dim, device=device), dim=0)

        # Base parameters
        self.noise_scale_base = noise_scale_base
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max

        # No imposed structure (identity integration)
        self.W = torch.eye(dim, device=device)

    def expand(self, U):
        noise = torch.randn_like(U) * self.noise_scale_base
        return U + noise

    def integrate(self, X):
        Y = self.W @ X
        return F.softmax(Y, dim=0)

    def evaluate_internal_viability(self, X, Y):
        # Coherence proxy: inverse L2 distance
        coherence = 1.0 / (1.0 + torch.norm(X - Y, p=2))
        # Deviation proxy
        deviation = torch.norm(X - Y, p=2)
        # Internal viability signal
        V = coherence - deviation
        return V

    def derive_lambda_emergent(self, V):
        # No proportional rule imposed
        # Lambda emerges indirectly from internal signal scaling
        raw = torch.tanh(V)
        lambda_t = (raw + 1.0) / 2.0
        lambda_t = self.lambda_min + lambda_t * (self.lambda_max - self.lambda_min)
        return lambda_t

    def step(self):
        X = self.expand(self.U)
        Y = self.integrate(X)
        V = self.evaluate_internal_viability(X, Y)

        lambda_t = self.derive_lambda_emergent(V)

        self.U = (1.0 - lambda_t) * self.U + lambda_t * Y
        self.U = F.softmax(self.U, dim=0)

        return {
            "U": self.U.clone(),
            "V": V.item(),
            "lambda": lambda_t.item(),
        }


if __name__ == "__main__":
    system = ICARUS_v04Ca(dim=16)

    for t in range(100):
        out = system.step()
        if t % 10 == 0:
            print(
                f"t={t} | V={out['V']:.4f} | lambda={out['lambda']:.4f}"
            )
