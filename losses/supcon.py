# losses/supcon.py
# =================
# Supervised Contrastive Loss.
# Used in Stage 2: supervised fine-tuning with category labels.
#
# Extension of NT-Xent that leverages class labels: ALL images of the
# same class are positives, not just the augmented pair.
#
# Reference: Khosla et al., "Supervised Contrastive Learning", NeurIPS 2020.

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """
    For anchor i, P(i) = all other views with the same label (excluding self).

    L_i = -(1/|P(i)|) sum_{p in P(i)} log(
              exp(sim(z_i, z_p) / tau)
              / sum_{k != i} exp(sim(z_i, z_k) / tau)
          )

    This teaches the model that different images of the same saree
    category should cluster together, adding semantic discrimination
    on top of Stage 1's color-invariant features.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.tau = temperature

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z1, z2: (N, D) L2-normalized embeddings.
            labels:  (N,) class labels.
        Returns:
            Scalar loss.
        """
        N = z1.size(0)
        z = torch.cat([z1, z2], dim=0)            # (2N, D)
        labels = labels.repeat(2)                  # (2N,)

        sim = torch.mm(z, z.t()) / self.tau        # (2N, 2N)

        # Masks
        self_mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
        sim.masked_fill_(self_mask, -1e9)

        pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~self_mask

        # Per-anchor loss
        log_sum_exp = torch.logsumexp(sim, dim=1)                # (2N,)
        num_pos = pos_mask.sum(dim=1).clamp(min=1)               # (2N,)
        pos_sum = (sim * pos_mask.float()).sum(dim=1)             # (2N,)

        loss = -(pos_sum / num_pos) + log_sum_exp
        return loss.mean()
