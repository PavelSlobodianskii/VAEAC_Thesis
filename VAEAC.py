import math
import torch
from torch.distributions import kl_divergence
from torch.nn import Module

from prob_utils import normal_parse_params

class VAEAC(Module):
    """
    Variational Autoencoder with Arbitrary Conditioning core model.
    Correct masking logic is implemented: during test/inpainting, the prior network only sees observed pixels.
    """
    def __init__(self, rec_log_prob, proposal_network, prior_network,
                 generative_network, sigma_mu=1e4, sigma_sigma=1e-4):
        super().__init__()
        self.rec_log_prob = rec_log_prob
        self.proposal_network = proposal_network
        self.prior_network = prior_network
        self.generative_network = generative_network
        self.sigma_mu = sigma_mu
        self.sigma_sigma = sigma_sigma

    def make_observed(self, batch, mask, fill_value=0.0):
        """
        Returns the batch with missing features filled by fill_value (default 0).
        mask: 1 = missing, 0 = observed
        """
        # Only observed pixels are kept; missing pixels filled with fill_value (default: 0)
        return batch * (1 - mask) + fill_value * mask

    def make_latent_distributions(self, batch, mask, no_proposal=False):
        """
        Make latent distributions for the given batch and mask.
        If no_proposal is True, return None instead of proposal distribution.
        """
        # Proper masking: prior sees only observed, proposal sees all
        observed = self.make_observed(batch, mask)  # zeros in missing region
        if no_proposal:
            proposal = None
        else:
            full_info = torch.cat([batch, mask], 1)
            proposal_params = self.proposal_network(full_info)
            proposal = normal_parse_params(proposal_params, 1e-3)
        # Prior net input: observed pixels only (missing regions zeroed)
        prior_params = self.prior_network(torch.cat([observed, mask], 1))
        prior = normal_parse_params(prior_params, 1e-3)
        # Sanity check: no info leakage in prior input
        # assert torch.allclose(observed[mask.bool()], torch.zeros_like(observed[mask.bool()])), "Prior input contains info in masked region!"
        return proposal, prior

    def prior_regularization(self, prior):
        num_objects = prior.mean.shape[0]
        mu = prior.mean.view(num_objects, -1)
        sigma = prior.scale.view(num_objects, -1)
        mu_regularizer = -(mu ** 2).sum(-1) / 2 / (self.sigma_mu ** 2)
        sigma_regularizer = (sigma.log() - sigma).sum(-1) * self.sigma_sigma
        return mu_regularizer + sigma_regularizer

    def batch_vlb(self, batch, mask):
        proposal, prior = self.make_latent_distributions(batch, mask)
        prior_regularization = self.prior_regularization(prior)
        latent = proposal.rsample()
        rec_params = self.generative_network(latent)
        rec_loss = self.rec_log_prob(batch, rec_params, mask)
        kl = kl_divergence(proposal, prior).view(batch.shape[0], -1).sum(-1)
        return rec_loss - kl + prior_regularization

    def batch_iwae(self, batch, mask, K):
        proposal, prior = self.make_latent_distributions(batch, mask)
        estimates = []
        for i in range(K):
            latent = proposal.rsample()
            rec_params = self.generative_network(latent)
            rec_loss = self.rec_log_prob(batch, rec_params, mask)
            prior_log_prob = prior.log_prob(latent)
            prior_log_prob = prior_log_prob.view(batch.shape[0], -1)
            prior_log_prob = prior_log_prob.sum(-1)
            proposal_log_prob = proposal.log_prob(latent)
            proposal_log_prob = proposal_log_prob.view(batch.shape[0], -1)
            proposal_log_prob = proposal_log_prob.sum(-1)
            estimate = rec_loss + prior_log_prob - proposal_log_prob
            estimates.append(estimate[:, None])
        return torch.logsumexp(torch.cat(estimates, 1), 1) - math.log(K)

    def generate_samples_params(self, batch, mask, K=1):
        """
        Generate parameters of generative distributions for samples from the given batch.
        Proper masking is always applied to the prior.
        """
        _, prior = self.make_latent_distributions(batch, mask)
        samples_params = []
        for i in range(K):
            latent = prior.rsample()
            sample_params = self.generative_network(latent)
            samples_params.append(sample_params.unsqueeze(1))
        return torch.cat(samples_params, 1)

    def generate_reconstructions_params(self, batch, mask, K=1):
        """
        Generate parameters of generative distributions for reconstructions from the given batch.
        Proper masking is always applied to the prior.
        """
        _, prior = self.make_latent_distributions(batch, mask)
        reconstructions_params = []
        for i in range(K):
            latent = prior.rsample()
            rec_params = self.generative_network(latent)
            reconstructions_params.append(rec_params.unsqueeze(1))
        return torch.cat(reconstructions_params, 1)
