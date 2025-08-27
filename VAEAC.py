import math
import torch
from torch.distributions import kl_divergence
from torch.nn import Module
from prob_utils import normal_parse_params




class VAEAC(Module):
    """
    Variational Autoencoder with Arbitrary Conditioning (VAEAC).


    Masking semantics: mask==1 means "missing / to be inpainted".
    - Prior network sees ONLY observed pixels: observed = x * (1 - mask)
    - Proposal network sees full (x, mask)
    """


    def __init__(
        self,
        rec_log_prob,
        proposal_network,
        prior_network,
        generative_network,
        sigma_mu: float = 1e4,
        sigma_sigma: float = 1e-4,
        debug_asserts: bool = False,
    ):
        super().__init__()
        self.rec_log_prob = rec_log_prob
        self.proposal_network = proposal_network
        self.prior_network = prior_network
        self.generative_network = generative_network
        self.sigma_mu = sigma_mu
        self.sigma_sigma = sigma_sigma
        self.debug_asserts = debug_asserts


    # --------- masking ---------
    @staticmethod
    def make_observed(batch: torch.Tensor, mask: torch.Tensor, fill_value: float = 0.0) -> torch.Tensor:
        """
        batch:  [B, C, H, W]
        mask:   [B, C or 1, H, W] with 1 = missing, 0 = observed
        returns: batch with missing region filled with fill_value (default 0)
        """
        if mask.dtype != torch.bool:
            mask = mask.to(dtype=torch.bool)
        if mask.shape[1] == 1 and batch.shape[1] != 1:
            mask = mask.repeat(1, batch.shape[1], 1, 1)
        return batch * (~mask) + float(fill_value) * mask


    # --------- latent dists ---------
    def make_latent_distributions(self, batch: torch.Tensor, mask: torch.Tensor, no_proposal: bool = False):
        """
        Builds q(z|x,mask) and p(z|x_obs,mask) with correct masking.
        """
        if mask.dtype != torch.bool:
            mask = mask.to(dtype=torch.bool, device=batch.device)
        if mask.shape[1] == 1 and batch.shape[1] != 1:
            mask = mask.repeat(1, batch.shape[1], 1, 1)


        # proposal q(z|x,mask) sees full (x, mask)
        proposal = None
        if not no_proposal:
            full_info = torch.cat([batch, mask.float()], 1)
            proposal_params = self.proposal_network(full_info)
            proposal = normal_parse_params(proposal_params, min_sigma=1e-3)


        # prior p(z|x_obs,mask) sees only observed pixels
        observed = self.make_observed(batch, mask, 0.0)
        if self.debug_asserts:
            # hard anti-leakage check: masked positions must be zero in "observed"
            with torch.no_grad():
                leaked = observed[mask].abs().max() if mask.any() else torch.tensor(0.0, device=observed.device)
                assert float(leaked) == 0.0, "Prior input contains non-zero values in masked region!"


        prior_in = torch.cat([observed, mask.float()], 1)
        prior_params = self.prior_network(prior_in)
        prior = normal_parse_params(prior_params, min_sigma=1e-3)


        return proposal, prior


    # --------- tiny utility used by logging/viz ---------
    @torch.no_grad()
    def latent_means(self, batch: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Returns the per-sample flattened mean of q(z|x,mask) -> [B, D]
        """
        q, _ = self.make_latent_distributions(batch, mask, no_proposal=False)
        flat = q.mean.view(q.mean.shape[0], -1)
        return flat


    # --------- losses ---------
    def prior_regularization(self, prior):
        num = prior.mean.shape[0]
        mu = prior.mean.view(num, -1)
        sigma = prior.scale.view(num, -1)
        mu_reg = -(mu ** 2).sum(-1) / (2 * (self.sigma_mu ** 2))
        sigma_reg = (sigma.log() - sigma).sum(-1) * self.sigma_sigma
        return mu_reg + sigma_reg


    def batch_vlb(self, batch, mask):
        q, p = self.make_latent_distributions(batch, mask)
        prior_reg = self.prior_regularization(p)
        z = q.rsample()
        rec_params = self.generative_network(z)
        rec_ll = self.rec_log_prob(batch, rec_params, mask)
        kl = kl_divergence(q, p).view(batch.shape[0], -1).sum(-1)
        return rec_ll - kl + prior_reg


    def batch_iwae(self, batch, mask, K: int):
        q, p = self.make_latent_distributions(batch, mask)
        est = []
        for _ in range(K):
            z = q.rsample()
            rec_params = self.generative_network(z)
            rec_ll = self.rec_log_prob(batch, rec_params, mask)
            log_pz = p.log_prob(z).view(batch.shape[0], -1).sum(-1)
            log_qz = q.log_prob(z).view(batch.shape[0], -1).sum(-1)
            est.append((rec_ll + log_pz - log_qz)[:, None])
        return torch.logsumexp(torch.cat(est, 1), 1) - math.log(K)


    # --------- sampling ---------
    @torch.no_grad()
    def generate_samples_params(self, batch, mask, K: int = 1):
        _, p = self.make_latent_distributions(batch, mask, no_proposal=True)
        out = []
        for _ in range(K):
            z = p.rsample()
            out.append(self.generative_network(z).unsqueeze(1))
        return torch.cat(out, 1)


    @torch.no_grad()
    def generate_reconstructions_params(self, batch, mask, K: int = 1):
        _, p = self.make_latent_distributions(batch, mask, no_proposal=True)
        out = []
        for _ in range(K):
            z = p.rsample()
            out.append(self.generative_network(z).unsqueeze(1))
        return torch.cat(out, 1)
