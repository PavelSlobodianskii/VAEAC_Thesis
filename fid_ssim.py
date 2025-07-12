import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.functional.image.ssim import structural_similarity_index_measure

def compute_fid_ssim(real_images, generated_images, device="cuda"):
    # Both tensors: [N, 3, 128, 128], values in [-1, 1]
    # Unnormalize to [0, 1] for FID/SSIM
    real_images = (real_images + 1) / 2
    generated_images = (generated_images + 1) / 2

    # FID
    fid = FrechetInceptionDistance(normalize=True).to(device)
    fid.update(real_images, real=True)
    fid.update(generated_images, real=False)
    fid_value = fid.compute().item()

    # SSIM (mean over all pairs)
    ssim = 0.0
    for x, y in zip(real_images, generated_images):
        ssim += structural_similarity_index_measure(x.unsqueeze(0), y.unsqueeze(0)).item()
    ssim_value = ssim / real_images.size(0)
    return fid_value, ssim_value


