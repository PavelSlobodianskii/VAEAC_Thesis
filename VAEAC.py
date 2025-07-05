import math
import torch
from torch.distributions import kl_divergence
from torch.nn import Module, Parameter

from prob_utils import normal_parse_params

torch.set_printoptions(precision=3, sci_mode=False)

class VAEAC(Module):
    def __init__(self, rec_log_prob, proposal_network, prior_network,
                 generative_network, sigma_mu=1e4, sigma_sigma=1e-4):
        super().__init__()
        self.rec_log_prob = rec_log_prob
        self.proposal_network = proposal_network
        self.prior_network = prior_network
        self.generative_network = generative_network
        self.sigma_mu = sigma_mu
        self.sigma_sigma = sigma_sigma

        # ---- Learnable parameter for symmetric KL weighting ----
        self._alpha_param = Parameter(torch.tensor(0.0))

    @property
    def alpha(self):
        # Constrain alpha between (0,1) using sigmoid
        return torch.sigmoid(self._alpha_param)

    def make_observed(self, batch, mask):
        observed = torch.tensor(batch)
        observed[mask.bool()] = 0
        return observed

    def make_latent_distributions(self, batch, mask, no_proposal=False):
        observed = self.make_observed(batch, mask)
        if no_proposal:
            proposal = None
        else:
            full_info = torch.cat([batch, mask], 1)
            proposal_params = self.proposal_network(full_info)
            proposal = normal_parse_params(proposal_params, 1e-3)

        prior_params = self.prior_network(torch.cat([observed, mask], 1))
        prior = normal_parse_params(prior_params, 1e-3)

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
        prior_reg = self.prior_regularization(prior)
        latent = proposal.rsample()
        rec_params = self.generative_network(latent)
        rec_loss = self.rec_log_prob(batch, rec_params, mask)

        # ---- Symmetric KL with learnable weighting ----
        kl_forward = kl_divergence(proposal, prior).view(batch.shape[0], -1).sum(-1)
        kl_reverse = kl_divergence(prior, proposal).view(batch.shape[0], -1).sum(-1)
        kl = self.alpha * kl_forward + (1.0 - self.alpha) * kl_reverse

        return rec_loss - kl + prior_reg

    def batch_iwae(self, batch, mask, K):
        proposal, prior = self.make_latent_distributions(batch, mask)
        estimates = []

        for i in range(K):
            latent = proposal.rsample()
            rec_params = self.generative_network(latent)
            rec_loss = self.rec_log_prob(batch, rec_params, mask)

            prior_log_prob = prior.log_prob(latent).view(batch.shape[0], -1).sum(-1)
            proposal_log_prob = proposal.log_prob(latent).view(batch.shape[0], -1).sum(-1)

            estimate = rec_loss + prior_log_prob - proposal_log_prob
            estimates.append(estimate[:, None])

        result = torch.logsumexp(torch.cat(estimates, 1), 1) - math.log(K)
        return result

    def generate_samples_params(self, batch, mask, K=1):
        _, prior = self.make_latent_distributions(batch, mask)
        samples_params = []

        for i in range(K):
            latent = prior.rsample()
            sample_params = self.generative_network(latent)
            samples_params.append(sample_params.unsqueeze(1))

        out = torch.cat(samples_params, 1)
        return out

    def generate_reconstructions_params(self, batch, mask, K=1):
        _, prior = self.make_latent_distributions(batch, mask)
        reconstructions_params = []

        for i in range(K):
            latent = prior.rsample()
            rec_params = self.generative_network(latent)
            reconstructions_params.append(rec_params.unsqueeze(1))

        out = torch.cat(reconstructions_params, 1)
        return out
