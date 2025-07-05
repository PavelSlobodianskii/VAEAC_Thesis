import math
import torch
from torch.nn import Module, Parameter
from torch.nn import functional as F
from prob_utils import normal_parse_params
from realnvp import RealNVP

torch.set_printoptions(precision=3, sci_mode=False)

class LatentDiscriminator(Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.main = torch.nn.Sequential(
            torch.nn.Conv2d(latent_dim, 128, 1), torch.nn.ReLU(),
            torch.nn.Conv2d(128, 32, 1), torch.nn.ReLU(),
            torch.nn.Conv2d(32, 1, 1)
        )

    def forward(self, z):
        return self.main(z).view(z.size(0), -1).mean(dim=1)  # (batch,)

def info_nce_loss(z1, z2, temperature=0.1):
    z1, z2 = z1.view(z1.size(0), -1), z2.view(z2.size(0), -1)
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    logits = torch.mm(z1, z2.t()) / temperature
    labels = torch.arange(z1.size(0), device=z1.device)
    return F.cross_entropy(logits, labels)

class VAEAC(Module):
    def __init__(self, rec_log_prob, proposal_network, generative_network, sigma_mu=1e4, sigma_sigma=1e-4):
        super().__init__()
        self.rec_log_prob = rec_log_prob
        self.proposal_network = proposal_network
        self.generative_network = generative_network
        self.sigma_mu = sigma_mu
        self.sigma_sigma = sigma_sigma

        self.prior_flow = RealNVP(256)
        self.discriminator = LatentDiscriminator(256)
        self.beta_adv = Parameter(torch.tensor(0.1), requires_grad=True)
        self.beta_nce = Parameter(torch.tensor(0.1), requires_grad=True)

    def make_observed(self, batch, mask):
        observed = torch.tensor(batch)
        observed[mask.bool()] = 0
        return observed

    def make_latent_distribution(self, batch, mask):
        full_info = torch.cat([batch, mask], 1)
        proposal_params = self.proposal_network(full_info)
        if isinstance(proposal_params, tuple):
            proposal_params = proposal_params[0]
        proposal = normal_parse_params(proposal_params, 1e-3)
        return proposal

    def batch_vlb(self, batch, mask):
        from nn_utils import MemoryLayer
        MemoryLayer.storage = {}
        full_info = torch.cat([batch, mask], 1)
        proposal_params = self.proposal_network(full_info)
        if isinstance(proposal_params, tuple):
            proposal_params = proposal_params[0]
        proposal = normal_parse_params(proposal_params, 1e-3)
        latent = proposal.rsample()
        rec_params = self.generative_network(latent)
        if isinstance(rec_params, tuple):
            rec_params = rec_params[0]
        rec_loss = self.rec_log_prob(batch, rec_params, mask)

        z_flat = latent
        log_pz = self.prior_flow.log_prob(z_flat)
        log_qz_x = proposal.log_prob(latent).view(latent.size(0), -1).sum(-1)

        # GAN (adversarial) loss
        real_prior = self.prior_flow.sample(latent.size(0), spatial_shape=(1,1))
        d_real = self.discriminator(real_prior)
        d_fake = self.discriminator(latent.detach())
        d_loss = -torch.mean(F.logsigmoid(d_real) + F.logsigmoid(-d_fake))
        g_loss = -torch.mean(F.logsigmoid(d_fake))

        # Contrastive (instance-level, mask augmentation)
        MemoryLayer.storage = {}
        batch_perm = batch
        mask_perm = torch.stack([mask[i][torch.randperm(mask.size(1))] for i in range(mask.size(0))])
        full_info2 = torch.cat([batch_perm, mask_perm], 1)
        proposal_params2 = self.proposal_network(full_info2)
        if isinstance(proposal_params2, tuple):
            proposal_params2 = proposal_params2[0]
        proposal2 = normal_parse_params(proposal_params2, 1e-3)
        latent2 = proposal2.rsample()
        nce_loss = info_nce_loss(latent, latent2)

        loss = rec_loss + log_pz - log_qz_x \
               - self.beta_adv * g_loss \
               + self.beta_nce * nce_loss

        return loss, {'rec_loss': rec_loss.mean().item(),
                      'log_pz': log_pz.mean().item(),
                      'log_qz_x': log_qz_x.mean().item(),
                      'g_loss': g_loss.item(),
                      'nce_loss': nce_loss.item()}

    def discriminator_loss(self, prior_samples, posterior_samples):
        d_real = self.discriminator(prior_samples)
        d_fake = self.discriminator(posterior_samples)
        return -torch.mean(F.logsigmoid(d_real) + F.logsigmoid(-d_fake))

    def generate_samples_params(self, batch, mask, K=1):
        from nn_utils import MemoryLayer
        MemoryLayer.storage = {}
        full_info = torch.cat([batch, mask], 1)
        proposal_params = self.proposal_network(full_info)
        if isinstance(proposal_params, tuple):
            proposal_params = proposal_params[0]
        proposal = normal_parse_params(proposal_params, 1e-3)
        samples_params = []
        for _ in range(K):
            latent = proposal.rsample()
            rec_params = self.generative_network(latent)
            if isinstance(rec_params, tuple):
                rec_params = rec_params[0]
            samples_params.append(rec_params.unsqueeze(1))
        out = torch.cat(samples_params, 1)
        return out

    def generate_reconstructions_params(self, batch, mask, K=1):
        return self.generate_samples_params(batch, mask, K)

    def batch_iwae(self, batch, mask, K=10):
        """
        Importance Weighted Autoencoder estimator for marginal log-likelihood.
        Returns tensor of shape (batch_size,)
        """
        proposal = self.make_latent_distribution(batch, mask)
        batch_size = batch.size(0)
        latents = proposal.rsample((K,))  # (K, batch, latent_dim, ...)
        latents_ = latents.view(K * batch_size, *latents.shape[2:])
        rec_params = self.generative_network(latents_)
        if isinstance(rec_params, tuple):
            rec_params = rec_params[0]
        rec_params = rec_params.view(K, batch_size, *rec_params.shape[1:])
        # Compute rec_log_prob for each sample
        log_p_x_given_z = []
        for k in range(K):
            logp = self.rec_log_prob(batch, rec_params[k], mask)
            log_p_x_given_z.append(logp)
        log_p_x_given_z = torch.stack(log_p_x_given_z, 0)  # (K, batch)
        # Prior log prob
        z_flat = latents_
        log_p_z = self.prior_flow.log_prob(z_flat).view(K, batch_size)
        # Posterior log prob
        log_q_z_x = proposal.log_prob(latents).view(K, batch_size, -1).sum(-1)
        # IWAE weights
        iwae_weight = log_p_x_given_z + log_p_z - log_q_z_x  # (K, batch)
        # Log-mean-exp over K
        iwae = torch.logsumexp(iwae_weight, dim=0) - math.log(K)
        return iwae  # shape: (batch,)





