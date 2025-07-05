import torch
import torch.nn as nn
import torch.nn.functional as F

class RealNVPBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Sequential(nn.Conv2d(dim // 2, 128, 1), nn.ReLU(), nn.Conv2d(128, dim // 2, 1))
        self.shift = nn.Sequential(nn.Conv2d(dim // 2, 128, 1), nn.ReLU(), nn.Conv2d(128, dim // 2, 1))

    def forward(self, x, reverse=False):
        x1, x2 = torch.chunk(x, 2, dim=1)
        if not reverse:
            s = self.scale(x1)
            t = self.shift(x1)
            y2 = x2 * torch.exp(s) + t
            y = torch.cat([x1, y2], 1)
            log_det_jacobian = s.flatten(1).sum(-1)
            return y, log_det_jacobian
        else:
            s = self.scale(x1)
            t = self.shift(x1)
            y2 = (x2 - t) * torch.exp(-s)
            y = torch.cat([x1, y2], 1)
            log_det_jacobian = -s.flatten(1).sum(-1)
            return y, log_det_jacobian

class RealNVP(nn.Module):
    def __init__(self, dim, num_blocks=6):
        super().__init__()
        self.blocks = nn.ModuleList([RealNVPBlock(dim) for _ in range(num_blocks)])

    def forward(self, z, reverse=False):
        log_det_jacobian = 0
        if not reverse:
            for block in self.blocks:
                z, ldj = block(z, reverse=False)
                log_det_jacobian += ldj
            return z, log_det_jacobian
        else:
            for block in reversed(self.blocks):
                z, ldj = block(z, reverse=True)
                log_det_jacobian += ldj
            return z, log_det_jacobian

    def log_prob(self, z):
        x, ldj = self.forward(z, reverse=True)
        x_flat = x.flatten(1)
        log_p_x = -0.5 * (x_flat ** 2).sum(dim=1) - 0.5 * x_flat.shape[1] * torch.log(torch.tensor(2 * torch.pi, device=x.device))
        return log_p_x + ldj

    def sample(self, n, spatial_shape=(1, 1)):
        device = next(self.parameters()).device
        x = torch.randn(n, 256, *spatial_shape, device=device)
        z, _ = self.forward(x)
        return z
