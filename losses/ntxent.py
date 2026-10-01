# losses/ntxent.py
# =================
# NT-Xent (Normalized Temperature-scaled Cross-Entropy) loss.
# Used in Stage 1: self-supervised contrastive pretraining.
#
# Reference: Chen et al., "A Simple Framework for Contrastive Learning
#            of Visual Representations" (SimCLR), ICML 2020.

import torch
import torch.nn as nn


class NTXentLoss(nn.Module):
    """
    For a batch of N images we have 2N views (2 per image).
    Each view i's positive is its other augmented view j.
    The remaining 2(N-1) views are negatives.

    L_i = -log( exp(sim(z_i, z_j) / tau)
                / sum_{k != i} exp(sim(z_i, z_k) / tau) )

    Temperature tau controls hardness:
        Low tau  (0.07): sharp distribution, hard negatives dominate.
        High tau (0.50): soft distribution, all negatives contribute.

    We use tau=0.07 following SimCLR's ablation.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.tau = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z1, z2: (N, D) L2-normalized embeddings from view 1 and view 2.
        Returns:
            Scalar loss.
        """
        N = z1.size(0)
        z = torch.cat([z1, z2], dim=0)                      # (2N, D)
        sim = torch.mm(z, z.t()) / self.tau                  # (2N, 2N)

        # Mask out self-similarity on the diagonal
        mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, -1e9)

        # Positive similarities: z1[i] <-> z2[i] and z2[i] <-> z1[i]
        pos = torch.cat([torch.diag(sim, N), torch.diag(sim, -N)])  # (2N,)

        loss = -pos + torch.logsumexp(sim, dim=1)
        return loss.mean()
