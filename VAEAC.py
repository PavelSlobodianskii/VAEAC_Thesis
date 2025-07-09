import math
import torch
from torch.distributions import kl_divergence
from torch.nn import Module
from prob_utils import normal_parse_params

class VAEAC(Module):
    """
    Variational Autoencoder with Arbitrary Conditioning core model.
    Assumes:
    + batch and mask are same shape
    + Prior and proposal output [B, 2*latent_dim, h, w] (mean/logvar concatenated)
    + Generative network maps [B, latent_dim, h, w] → [B, 3, 128, 128]
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

    def make_observed(self, batch, mask):
        """ Zero-out unobserved pixels in the batch. """
        observed = batch.clone() if isinstance(batch, torch.Tensor) else torch.tensor(batch)
        observed[mask.bool()] = 0
        return observed

    def make_latent_distributions(self, batch, mask, no_proposal=False):
        """
        Calls networks and splits their output as (mean, logvar) for proposal/prior.
        If no_proposal: proposal = None.
        Returns: proposal, prior (both are torch.distributions)
        """
        observed = self.make_observed(batch, mask)

        if no_proposal:
            proposal = None
        else:
            full_info = torch.cat([batch, mask], 1)
            proposal_params = self.proposal_network(full_info)
            latent_dim = proposal_params.shape[1] // 2
            proposal_mean = proposal_params[:, :latent_dim]
            proposal_logvar = proposal_params[:, latent_dim:]
            # Optionally print shapes for debug:
            # print("Proposal mean:", proposal_mean.shape, "logvar:", proposal_logvar.shape)
            proposal = normal_parse_params((proposal_mean, proposal_logvar), 1e-3)

        prior_params = self.prior_network(torch.cat([observed, mask], 1))
        latent_dim = prior_params.shape[1] // 2
        prior_mean = prior_params[:, :latent_dim]
        prior_logvar = prior_params[:, latent_dim:]
        # print("Prior mean:", prior_mean.shape, "logvar:", prior_logvar.shape)
        prior = normal_parse_params((prior_mean, prior_logvar), 1e-3)
        return proposal, prior

    def prior_regularization(self, prior):
        """
        Regularizes prior to avoid degenerate values. Standard VAEAC.
        """
        num_objects = prior.mean.shape[0]
        mu = prior.mean.view(num_objects, -1)
        sigma = prior.scale.view(num_objects, -1)
        mu_regularizer = -(mu ** 2).sum(-1) / 2 / (self.sigma_mu ** 2)
        sigma_regularizer = (sigma.log() - sigma).sum(-1) * self.sigma_sigma
        return mu_regularizer + sigma_regularizer

    def batch_vlb(self, batch, mask):
        """
        Computes stochastic variational lower bound for batch.
        """
        proposal, prior = self.make_latent_distributions(batch, mask)
        prior_reg = self.prior_regularization(prior)
        latent = proposal.rsample()
        rec_params = self.generative_network(latent)
        rec_loss = self.rec_log_prob(batch, rec_params, mask)
        kl = kl_divergence(proposal, prior).view(batch.shape[0], -1).sum(-1)
        return rec_loss - kl + prior_reg

    def batch_iwae(self, batch, mask, K):
        """
        Importance-Weighted Autoencoder bound (evaluation, not training).
        """
        proposal, prior = self.make_latent_distributions(batch, mask)
        estimates = []
        for _ in range(K):
            latent = proposal.rsample()
            rec_params = self.generative_network(latent)
            rec_loss = self.rec_log_prob(batch, rec_params, mask)
            prior_log_prob = prior.log_prob(latent).view(batch.shape[0], -1).sum(-1)
            proposal_log_prob = proposal.log_prob(latent).view(batch.shape[0], -1).sum(-1)
            estimate = rec_loss + prior_log_prob - proposal_log_prob
            estimates.append(estimate[:, None])
        return torch.logsumexp(torch.cat(estimates, 1), 1) - math.log(K)

    def generate_samples_params(self, batch, mask, K=1):
        """
        Samples from the prior, then decodes (K times per batch element).
        Returns tensor [batch, K, ...]
        """
        _, prior = self.make_latent_distributions(batch, mask)
        samples_params = []
        for _ in range(K):
            latent = prior.rsample()
            sample_params = self.generative_network(latent)
            samples_params.append(sample_params.unsqueeze(1))
        return torch.cat(samples_params, 1)

    def generate_reconstructions_params(self, batch, mask, K=1):
        """
        Samples from the prior, then decodes (K times per batch element).
        """
        _, prior = self.make_latent_distributions(batch, mask)
        reconstructions_params = []
        for _ in range(K):
            latent = prior.rsample()
            rec_params = self.generative_network(latent)
            reconstructions_params.append(rec_params.unsqueeze(1))
        return torch.cat(reconstructions_params, 1)

