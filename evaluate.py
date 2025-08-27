"""
External inpainting demo for 4 Google Drive images, saving triptychs.
Paths are fixed to the user's structure in Colab.
"""


from __future__ import annotations
import os
from typing import Iterable
import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt


def _preprocess() -> T.Compose:
    return T.Compose([
        T.Resize(128),
        T.CenterCrop(128),
        T.ToTensor(),
        T.Normalize((0.5,)*3, (0.5,)*3),
    ])


def _denorm(x: torch.Tensor):
    return (x.clamp(-1,1)*0.5+0.5).cpu().permute(1,2,0).numpy()


@torch.no_grad()
def inpaint_triptychs(model, sampler, dense_mask: bool = True):
    device = next(model.parameters()).device
    preprocess = _preprocess()


    gdrive_root = "/content/drive/MyDrive/VAEAC_Thesis/external"
    names = ["Cat.png", "Dog.png", "building.png", "car.png"]
    imgs = []
    for n in names:
        p = os.path.join(gdrive_root, n)
        imgs.append(preprocess(Image.open(p).convert("RGB")))
    batch = torch.stack(imgs).to(device)


    # masks
    if dense_mask:
        # ~85% random pixel dropout
        prob = 0.85
        m = (torch.rand(batch.shape[:], device=device) < prob)
        masks = m
    else:
        masks = torch.zeros_like(batch, dtype=torch.bool, device=device)
        masks[:, :, 40:88, 40:88] = True


    params = model.generate_samples_params(batch, masks, K=1)
    inpainted = sampler(params[:, 0])


    out_dir = "/content/drive/MyDrive/VAEAC_Thesis/custom_inpaint_results_August"
    os.makedirs(out_dir, exist_ok=True)


    for i, n in enumerate(names):
        plt.figure(figsize=(12,4))
        plt.subplot(1,3,1); plt.imshow(_denorm(batch[i])); plt.axis("off"); plt.title("Original")
        masked = batch[i].clone(); masked[masks[i]] = 0
        plt.subplot(1,3,2); plt.imshow(_denorm(masked)); plt.axis("off"); plt.title("Masked")
        plt.subplot(1,3,3); plt.imshow(_denorm(inpainted[i])); plt.axis("off"); plt.title("Inpainted")
        out = os.path.join(out_dir, f"inpaint_{'dense' if dense_mask else 'center'}_{n.lower().replace('.png','').replace('.jpg','')}.png")
        plt.tight_layout(); plt.savefig(out, dpi=200); plt.close()