import torch
import torch.nn as nn
from torch.optim import Adam

from mask_generators import ImageMaskGenerator
from nn_utils import ResBlock, MemoryLayer
from prob_utils import normal_parse_params, GaussianLoss

# ======== Hyperparameters ========
batch_size = 64
vlb_scale_factor = 128 ** 2

def sampler(params):
    # Returns the mean of the Gaussian for inpainting (no added noise)
    return normal_parse_params(params).mean

def optimizer(parameters):
    return Adam(parameters, lr=2e-4)

reconstruction_log_prob = GaussianLoss()
mask_generator = ImageMaskGenerator()

latent_dim = 32  # Latent channels at the bottleneck

# ======== Proposal (Posterior) Network: U-Net Encoder ========
class ProposalUNet(nn.Module):
    """
    Proposal (posterior) network: U-Net encoder with ResBlock & MemoryLayer.
    Input: [B, 6, 128, 128] (image + mask)
    Output: [B, 2*latent_dim, 8, 8] (mean/logvar)
    """
    def __init__(self, in_channels=6, latent_dim=32):
        super().__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1),  # 128x128
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            ResBlock(64, 64),
            MemoryLayer('#d1')
        )
        self.down2 = nn.Sequential(
            nn.AvgPool2d(2),  # 64x64
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            ResBlock(128, 128),
            MemoryLayer('#d2')
        )
        self.down3 = nn.Sequential(
            nn.AvgPool2d(2),  # 32x32
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            ResBlock(256, 256),
            MemoryLayer('#d3')
        )
        self.down4 = nn.Sequential(
            nn.AvgPool2d(2),  # 16x16
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            ResBlock(512, 512),
            MemoryLayer('#d4')
        )
        self.bottleneck = nn.Sequential(
            nn.AvgPool2d(2),  # 8x8
            nn.Conv2d(512, latent_dim, 3, 1, 1),
            nn.BatchNorm2d(latent_dim),
            nn.LeakyReLU(),
            ResBlock(latent_dim, latent_dim)
        )
        self.fc_mean = nn.Conv2d(latent_dim, latent_dim, 3, 1, 1)
        self.fc_logvar = nn.Conv2d(latent_dim, latent_dim, 3, 1, 1)

    def forward(self, x):
        x1 = self.down1(x)      # 128x128
        x2 = self.down2(x1)     # 64x64
        x3 = self.down3(x2)     # 32x32
        x4 = self.down4(x3)     # 16x16
        bottleneck = self.bottleneck(x4)  # 8x8
        mean = self.fc_mean(bottleneck)
        logvar = self.fc_logvar(bottleneck)
        # Debug: Print shapes (remove/comment out for production)
        return torch.cat([mean, logvar], dim=1)  # [B, 2*latent_dim, 8, 8]

# ======== Prior Network: Same as Proposal, different weights ========
class PriorUNet(nn.Module):
    """
    Prior network: U-Net encoder, same as Proposal.
    Input: [B, 6, 128, 128]
    Output: [B, 2*latent_dim, 8, 8]
    """
    def __init__(self, in_channels=6, latent_dim=32):
        super().__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            ResBlock(64, 64),
            MemoryLayer('#pd1')
        )
        self.down2 = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            ResBlock(128, 128),
            MemoryLayer('#pd2')
        )
        self.down3 = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            ResBlock(256, 256),
            MemoryLayer('#pd3')
        )
        self.down4 = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            ResBlock(512, 512),
            MemoryLayer('#pd4')
        )
        self.bottleneck = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(512, latent_dim, 3, 1, 1),
            nn.BatchNorm2d(latent_dim),
            nn.LeakyReLU(),
            ResBlock(latent_dim, latent_dim)
        )
        self.fc_mean = nn.Conv2d(latent_dim, latent_dim, 3, 1, 1)
        self.fc_logvar = nn.Conv2d(latent_dim, latent_dim, 3, 1, 1)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        bottleneck = self.bottleneck(x4)
        mean = self.fc_mean(bottleneck)
        logvar = self.fc_logvar(bottleneck)
        return torch.cat([mean, logvar], dim=1)

# ======== Decoder (Generative Network): U-Net upsampling ========
class DecoderUNet(nn.Module):
    """
    Decoder (generative network) with upsampling (no skip concat for simplicity).
    Input: [B, latent_dim, 8, 8]
    Output: [B, 6, 128, 128] (mean and logvar for 3 RGB channels)
    """
    def __init__(self, latent_dim=32, out_channels=3):
        super().__init__()
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 2, 1), # 8x8 -> 16x16
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            ResBlock(512, 512)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1), # 16x16 -> 32x32
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            ResBlock(256, 256)
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), # 32x32 -> 64x64
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            ResBlock(128, 128)
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), # 64x64 -> 128x128
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            ResBlock(64, 64)
        )
        # Output: 6 channels (3 for mean, 3 for logvar)
        self.final = nn.Conv2d(64, out_channels * 2, 3, 1, 1)

    def forward(self, z):
        x = self.up1(z)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.final(x)
        return x

# ======== Instantiate Networks ========
proposal_network = ProposalUNet(in_channels=6, latent_dim=latent_dim)
prior_network = PriorUNet(in_channels=6, latent_dim=latent_dim)
generative_network = DecoderUNet(latent_dim=latent_dim, out_channels=3)


