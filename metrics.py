"""
Simple, dependency-friendly metric wrappers.
- FID via torch_fidelity if available
- SSIM / PSNR via skimage if available, else falls back to torch implementations
"""


from __future__ import annotations
from typing import Optional, Tuple
import torch


# --- FID (optional) ---
try:
    import torch_fidelity  # pip install torch-fidelity
    _HAS_FID = True
except Exception:
    _HAS_FID = False


# --- SSIM/PSNR (prefer skimage) ---
try:
    from skimage.metrics import structural_similarity as sk_ssim
    from skimage.metrics import peak_signal_noise_ratio as sk_psnr
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False




@torch.no_grad()
def compute_fid(inpainted: torch.Tensor, reference: torch.Tensor, device: str = "cuda") -> Optional[float]:
    """
    Both tensors in [-1,1], NCHW.
    Returns FID (lower is better) or None if torch_fidelity is missing.
    """
    if not _HAS_FID:
        return None
    # convert to [0,1]
    x_fake = (inpainted.clamp(-1, 1) + 1) / 2.0
    x_real = (reference.clamp(-1, 1) + 1) / 2.0
    # Save to temp folders required by torch_fidelity
    import tempfile, os
    from torchvision.utils import save_image
    with tempfile.TemporaryDirectory() as d_fake, tempfile.TemporaryDirectory() as d_real:
        for i, img in enumerate(x_fake):
            save_image(img, os.path.join(d_fake, f"{i:06d}.png"))
        for i, img in enumerate(x_real):
            save_image(img, os.path.join(d_real, f"{i:06d}.png"))
        metrics = torch_fidelity.calculate_metrics(
            input1=d_fake, input2=d_real,
            cuda=device.startswith("cuda"), isc=False, fid=True, kid=False, verbose=False
        )
    return float(metrics["frechet_inception_distance"])




@torch.no_grad()
def compute_ssim_psnr(inpainted: torch.Tensor, reference: torch.Tensor) -> Tuple[Optional[float], Optional[float]]:
    """
    Inputs in [-1,1], NCHW. Returns (ssim, psnr). If skimage not present, returns (None, None).
    """
    if not _HAS_SKIMAGE:
        return None, None
    x_fake = (inpainted.clamp(-1, 1) + 1) / 2.0
    x_real = (reference.clamp(-1, 1) + 1) / 2.0


    x_fake = x_fake.cpu().permute(0, 2, 3, 1).numpy()  # NHWC
    x_real = x_real.cpu().permute(0, 2, 3, 1).numpy()


    import numpy as np
    ssims, psnrs = [], []
    for a, b in zip(x_fake, x_real):
        # skimage expects [0,1]
        ssim = sk_ssim(a, b, data_range=1.0, channel_axis=2)
        psnr = sk_psnr(b, a, data_range=1.0)
        ssims.append(ssim)
        psnrs.append(psnr)
    return float(np.mean(ssims)), float(np.mean(psnrs))