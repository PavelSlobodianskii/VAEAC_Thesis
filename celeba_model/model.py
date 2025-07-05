from torch import nn
from torch.optim import Adam

from mask_generators import ImageMaskGenerator
from nn_utils import ResBlock, SkipConnection
from prob_utils import normal_parse_params, GaussianLoss
# RealNVP is NOT a prior_network, it's imported for use in VAEAC.py

# --- Sampler from the model generative distribution (unchanged) ---
def sampler(params):
    return normal_parse_params(params).mean

def optimizer(parameters):
    return Adam(parameters, lr=2e-4)

def discriminator_optimizer(parameters):
    return Adam(parameters, lr=2e-4)

batch_size = 16
reconstruction_log_prob = GaussianLoss()
mask_generator = ImageMaskGenerator()
vlb_scale_factor = 128 ** 2

def MLPBlock(dim):
    return SkipConnection(
        nn.BatchNorm2d(dim),
        nn.LeakyReLU(),
        nn.Conv2d(dim, dim, 1)
    )

# No "prior_network" needed: the RealNVP is constructed inside VAEAC, not here

proposal_network = nn.Sequential(
    nn.Conv2d(6, 8, 1),
    ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
    nn.AvgPool2d(2, 2),
    ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
    nn.AvgPool2d(2, 2), nn.Conv2d(8, 16, 1),
    ResBlock(16, 8), ResBlock(16, 8), ResBlock(16, 8), ResBlock(16, 8),
    nn.AvgPool2d(2, 2), nn.Conv2d(16, 32, 1),
    ResBlock(32, 16), ResBlock(32, 16), ResBlock(32, 16), ResBlock(32, 16),
    nn.AvgPool2d(2, 2), nn.Conv2d(32, 64, 1),
    ResBlock(64, 32), ResBlock(64, 32), ResBlock(64, 32), ResBlock(64, 32),
    nn.AvgPool2d(2, 2), nn.Conv2d(64, 128, 1),
    ResBlock(128, 64), ResBlock(128, 64),
    ResBlock(128, 64), ResBlock(128, 64),
    nn.AvgPool2d(2, 2), nn.Conv2d(128, 256, 1),
    ResBlock(256, 128), ResBlock(256, 128),
    ResBlock(256, 128), ResBlock(256, 128),
    nn.AvgPool2d(2, 2), nn.Conv2d(256, 512, 1),
    MLPBlock(512), MLPBlock(512), MLPBlock(512), MLPBlock(512),
)

generative_network = nn.Sequential(
    nn.Conv2d(256, 512, 1), nn.LeakyReLU(),  # 1x1
    nn.Upsample(scale_factor=2),  # 2x2
    nn.Conv2d(512, 256, 3, padding=1), nn.LeakyReLU(),
    nn.Upsample(scale_factor=2),  # 4x4
    nn.Conv2d(256, 128, 3, padding=1), nn.LeakyReLU(),
    nn.Upsample(scale_factor=2),  # 8x8
    nn.Conv2d(128, 128, 3, padding=1), nn.LeakyReLU(),
    nn.Upsample(scale_factor=2),  # 16x16
    nn.Conv2d(128, 64, 3, padding=1), nn.LeakyReLU(),
    nn.Upsample(scale_factor=2),  # 32x32
    nn.Conv2d(64, 32, 3, padding=1), nn.LeakyReLU(),
    nn.Upsample(scale_factor=2),  # 64x64
    nn.Conv2d(32, 16, 3, padding=1), nn.LeakyReLU(),
    nn.Upsample(scale_factor=2),  # 128x128
    nn.Conv2d(16, 6, 3, padding=1),  # Output: (B, 6, 128, 128)
)


