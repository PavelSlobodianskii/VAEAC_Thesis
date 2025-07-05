import math
import torch
from torch.distributions import kl_divergence
from torch.nn import Module, Parameter

from prob_utils import normal_parse_params

torch.set_printoptions(precision=3, sci_mode=False)

def compute_mmd(z, z_prior, sigma=1.0):
    """Compute MMD (RBF kernel) between z and z_prior."""
    def rbf_kernel(a, b, sigma):
        a = a.view(a.size(0), -1)
        b = b.view(b.size(0), -1)
        dist2 = torch.cdist(a, b, p=2) ** 2
        return torch.exp(-dist2 / (2 * sigma ** 2))
    Kzz = rbf_kernel(z, z, sigma)
    Kpp = rbf_kernel(z_prior, z_prior, sigma)
    Kzp = rbf_kernel(z, z_prior, sigma)
    mmd = Kzz.mean() + Kpp.mean() - 2 * Kzp.mean()
    return mmd

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
        self.beta = Parameter(torch.tensor(10.0), requires_grad=True)

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

        # WAE-MMD
        z_prior = torch.randn_like(latent)
        mmd = compute_mmd(latent, z_prior, sigma=1.0)

        # Learnable beta
        loss = rec_loss - self.beta * mmd + prior_reg

        print(f"[VAEAC] Batch MMD: {mmd.item():.4f}, beta: {self.beta.item():.4f}")
        return loss

    def batch_iwae(self, batch, mask, K):
        proposal, prior = self.make_latent_distributions(batch, mask)
        estimates = []
        for i in range(K):
            latent = proposal.rsample()
            rec_params = self.generative_network(latent)
            rec_loss = self.rec_log_prob(batch, rec_params, mask)
            # MMD for each sample
            z_prior = torch.randn_like(latent)
            mmd = compute_mmd(latent, z_prior, sigma=1.0)
            estimate = rec_loss - self.beta * mmd
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
