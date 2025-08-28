# filename: VAEAC.py
import math
from typing import Optional, Tuple

import torch
from torch import nn
from torch.distributions import kl_divergence
from torch.nn import Module

from prob_utils import normal_parse_params


def _kl(q, p, mode: str = "standard") -> torch.Tensor:
    """
    KL utilities for diagonal Gaussians from torch.distributions.
    mode = "standard"  -> KL(q || p)
    mode = "symmetric" -> 0.5 * ( KL(q||p) + KL(p||q) )
    """
    if mode == "standard":
        return kl_divergence(q, p)
    if mode == "symmetric":
        return 0.5 * (kl_divergence(q, p) + kl_divergence(p, q))
    raise ValueError(f"Unknown KL mode: {mode}")


def softplus_inv(y: float, eps: float = 1e-8) -> float:
    """
    Inverse softplus: returns x such that softplus(x) ~= y (for y>0).
    Useful to initialize a raw parameter so that softplus(raw) = y.
    """
    y = max(y, eps)
    return math.log(math.exp(y) - 1.0)


class VAEAC(Module):
    """
    Original VAEAC core (no network swap).

    Mask semantics: mask==1 means "missing / to be inpainted".
    - Proposal q(z|x,mask) sees [x, mask]
    - Prior    p(z|x_obs,mask) sees [x*(1-mask), mask]
    """

    def __init__(
        self,
        rec_log_prob,
        proposal_network,
        prior_network,
        generative_network,
        *,
        # regularizers (unchanged)
        sigma_mu: float = 1e4,
        sigma_sigma: float = 1e-4,
        # diagnostics
        debug_asserts: bool = False,
        # ---- KL controls ----
        kl_mode: str = "standard",          # "standard" or "symmetric"
        kl_alpha: Optional[float] = 1.0,    # numeric weight; use None if learnable_alpha=True
        learnable_alpha: bool = False,      # learn alpha during training
        alpha_init: float = 1.0,            # initial alpha when learnable
        alpha_max: float = 1e6,             # hard cap for the weight
        free_bits: float = 0.0,             # "free bits" floor (nats) per-sample KL
    ):
        super().__init__()
        self.rec_log_prob = rec_log_prob
        self.proposal_network = proposal_network
        self.prior_network = prior_network
        self.generative_network = generative_network

        self.sigma_mu = sigma_mu
        self.sigma_sigma = sigma_sigma
        self.debug_asserts = debug_asserts

        self.kl_mode = kl_mode
        self.free_bits = float(free_bits)
        self.alpha_max = float(alpha_max)

        if learnable_alpha:
            raw0 = softplus_inv(alpha_init if alpha_init is not None else 1.0)
            self.raw_alpha = nn.Parameter(torch.tensor(raw0, dtype=torch.float32))
            self.learnable_alpha = True
            self.kl_alpha = None
        else:
            self.learnable_alpha = False
            if isinstance(kl_alpha, float) and math.isfinite(kl_alpha):
                self.kl_alpha = float(kl_alpha)
            else:
                self.kl_alpha = 1.0  # safe default

    # ----------------- masking -----------------
    @staticmethod
    def make_observed(batch: torch.Tensor, mask: torch.Tensor, fill_value: float = 0.0) -> torch.Tensor:
        """
        batch: [B, C, H, W]
        mask : [B, 1 or C, H, W], 1 = missing
        return: x_obs = x * (1 - mask) + fill * mask
        """
        if mask.dtype != torch.bool:
            mask = mask.to(dtype=torch.bool, device=batch.device)
        if mask.shape[1] == 1 and batch.shape[1] != 1:
            mask = mask.repeat(1, batch.shape[1], 1, 1)
        return batch * (~mask) + float(fill_value) * mask

    # ----------------- latent distributions -----------------
    def make_latent_distributions(
        self, batch: torch.Tensor, mask: torch.Tensor, no_proposal: bool = False
    ):
        """
        Return (q, p) where q = q(z|x,mask), p = p(z|x_obs,mask).
        If no_proposal=True, return (None, p).
        """
        if mask.dtype != torch.bool:
            mask = mask.to(dtype=torch.bool, device=batch.device)
        if mask.shape[1] == 1 and batch.shape[1] != 1:
            mask = mask.repeat(1, batch.shape[1], 1, 1)

        q = None
        if not no_proposal:
            full_info = torch.cat([batch, mask.float()], 1)
            q_params = self.proposal_network(full_info)
            q = normal_parse_params(q_params, min_sigma=1e-3)

        x_obs = self.make_observed(batch, mask, 0.0)
        if self.debug_asserts and mask.any():
            with torch.no_grad():
                leaked = x_obs[mask].abs().max()
                assert float(leaked) == 0.0, "Prior input contains non-zero values in masked region!"

        p_in = torch.cat([x_obs, mask.float()], 1)
        p_params = self.prior_network(p_in)
        p = normal_parse_params(p_params, min_sigma=1e-3)

        return q, p

    # ----------------- helpers -----------------
    @torch.no_grad()
    def latent_means(self, batch: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Returns per-sample flattened mean of q(z|x,mask): [B, D]."""
        q, _ = self.make_latent_distributions(batch, mask, no_proposal=False)
        return q.mean.view(q.mean.shape[0], -1)

    def prior_regularization(self, prior) -> torch.Tensor:
        """Same as the original repository."""
        B = prior.mean.shape[0]
        mu = prior.mean.view(B, -1)
        sigma = prior.scale.view(B, -1)
        mu_reg = -(mu ** 2).sum(-1) / (2 * (self.sigma_mu ** 2))
        sigma_reg = (sigma.log() - sigma).sum(-1) * self.sigma_sigma
        return mu_reg + sigma_reg

    # ----------------- loss -----------------
    def _alpha_value(self) -> torch.Tensor:
        """
        Returns a scalar tensor α; grads flow when learnable.
        """
        if self.learnable_alpha:
            alpha = torch.nn.functional.softplus(self.raw_alpha)
            return torch.clamp(alpha, max=self.alpha_max)
        # numeric alpha as tensor on correct device
        dev = next(self.parameters()).device
        return torch.as_tensor(min(self.kl_alpha, self.alpha_max), dtype=torch.float32, device=dev)

    def _kl_term(self, q, p) -> torch.Tensor:
        # [B, latent_dims...] -> sum over dims
        kl = _kl(q, p, self.kl_mode).view(q.mean.shape[0], -1).sum(-1)
        if self.free_bits > 0.0:
            kl = torch.clamp(kl, min=self.free_bits)
        return kl

    def batch_vlb(self, batch: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        ELBO with configurable KL:
          ELBO = E_q[ log p(x|z,mask) ] - alpha * KL(q||p) + prior_reg
        """
        q, p = self.make_latent_distributions(batch, mask)
        z = q.rsample()
        rec_params = self.generative_network(z)
        rec_ll = self.rec_log_prob(batch, rec_params, mask)

        alpha = self._alpha_value()
        kl = self._kl_term(q, p)
        prior_reg = self.prior_regularization(p)

        return rec_ll - alpha * kl + prior_reg

    def batch_iwae(self, batch: torch.Tensor, mask: torch.Tensor, K: int) -> torch.Tensor:
        """
        IWAE estimate (uses standard importance weights).
        """
        q, p = self.make_latent_distributions(batch, mask)
        parts = []
        for _ in range(K):
            z = q.rsample()
            rec_params = self.generative_network(z)
            rec_ll = self.rec_log_prob(batch, rec_params, mask)

            log_pz = p.log_prob(z).view(batch.shape[0], -1).sum(-1)
            log_qz = q.log_prob(z).view(batch.shape[0], -1).sum(-1)
            parts.append((rec_ll + log_pz - log_qz)[:, None])
        return torch.logsumexp(torch.cat(parts, 1), 1) - math.log(K)

    # ----------------- sampling -----------------
    @torch.no_grad()
    def generate_samples_params(self, batch: torch.Tensor, mask: torch.Tensor, K: int = 1) -> torch.Tensor:
        _, p = self.make_latent_distributions(batch, mask, no_proposal=True)
        outs = []
        for _ in range(K):
            z = p.rsample()
            outs.append(self.generative_network(z).unsqueeze(1))
        return torch.cat(outs, 1)

    @torch.no_grad()
    def generate_reconstructions_params(self, batch: torch.Tensor, mask: torch.Tensor, K: int = 1) -> torch.Tensor:
        _, p = self.make_latent_distributions(batch, mask, no_proposal=True)
        outs = []
        for _ in range(K):
            z = p.rsample()
            outs.append(self.generative_network(z).unsqueeze(1))
        return torch.cat(outs, 1)

