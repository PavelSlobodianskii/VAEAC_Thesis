# filename: evaluate_alphas.py
from argparse import ArgumentParser
from importlib import import_module
from os import makedirs
from os.path import join, exists
from typing import List, Tuple, Dict

import os
import numpy as np
import torch
from torchvision import transforms as T
from PIL import Image
from tqdm import tqdm

from VAEAC import VAEAC
from mask_generators import ImageMCARGenerator
from metrics import compute_fid, compute_ssim_psnr


# ------------------------- I/O utils -------------------------
def _preprocess() -> T.Compose:
    return T.Compose([
        T.Resize(128),
        T.CenterCrop(128),
        T.ToTensor(),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])


def _denorm(x: torch.Tensor) -> torch.Tensor:
    # x: [C,H,W] in [-1,1] -> [0,1]
    return (x / 2 + 0.5).clamp(0, 1)


def load_external_images(folder: str, names: List[str]) -> List[torch.Tensor]:
    tfm = _preprocess()
    out = []
    for n in names:
        p = join(folder, n)
        if not exists(p):
            raise FileNotFoundError(f"External image not found: {p}")
        img = Image.open(p).convert("RGB")
        out.append(tfm(img))
    return out


def save_triptych(out_path: str, original: torch.Tensor, masked: torch.Tensor, inpainted: torch.Tensor):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    grid = torch.stack([_denorm(original), _denorm(masked), _denorm(inpainted)], 0)  # [3,C,H,W]
    # simple side-by-side concat
    C, H, W = original.shape
    canvas = torch.zeros(3, H, 3 * W)
    for i in range(3):
        canvas[:, :, i * W:(i + 1) * W] = grid[i]
    Image.fromarray((canvas.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).save(out_path)


# ------------------------- masks -------------------------
def make_center_mask_like(x: torch.Tensor) -> torch.Tensor:
    # x: [C,H,W] or [B,C,H,W], output [C,H,W] or [B,C,H,W] with 1 inside 64x64 center
    if x.dim() == 3:
        C, H, W = x.shape
        m = torch.zeros(C, H, W, dtype=torch.float32, device=x.device)
        m[:, 32:96, 32:96] = 1.0
        return m
    else:
        B, C, H, W = x.shape
        m = torch.zeros(B, C, H, W, dtype=torch.float32, device=x.device)
        m[:, :, 32:96, 32:96] = 1.0
        return m


def make_dense_mask_like(x: torch.Tensor, p: float = 0.95) -> torch.Tensor:
    gen = ImageMCARGenerator(p=p)
    if x.dim() == 3:
        xb = x.unsqueeze(0)
        m = gen(xb.cpu()).to(x.device).squeeze(0)
        return m
    else:
        return gen(x.cpu()).to(x.device)


# ------------------------- runner -------------------------
def run_eval(
    model_dir: str,
    alphas: List[str],
    external_dir: str,
    out_root: str,
    compute_fid_flag: bool,
    compute_ssimpsnr_flag: bool,
    masks: List[str],
    device: str = "cuda",
):
    os.makedirs(out_root, exist_ok=True)

    # ORIGINAL (no swap) networks & utilities
    model_module = import_module(model_dir + ".model")
    sampler = getattr(model_module, "sampler")

    # external image set (fixed order)
    ext_names = ["Cat.png", "Dog.png", "building.png", "car.png"]
    imgs = load_external_images(external_dir, ext_names)  # list of [3,128,128]
    imgs_t = torch.stack(imgs, 0)  # [4,3,128,128]
    imgs_t = imgs_t.to(device if torch.cuda.is_available() else "cpu")

    # pre-create dense mask generator on CPU (for reproducibility)
    # (we’ll still call our helpers per-image, to keep the same API)
    results_rows = []

    for alpha_name in alphas:
        alpha_dir = join(model_dir, "alpha_runs", alpha_name)
        ckpt_path = join(alpha_dir, "best.tar")
        if not exists(ckpt_path):
            print(f"[WARN] Best checkpoint not found for {alpha_name}: {ckpt_path}")
            continue

        # reconstruct alpha config from folder name (best-effort)
        kl_mode = "standard"
        learnable_alpha = False
        kl_alpha = 1.0
        if alpha_name.startswith("symmetric"):
            kl_mode = "symmetric"
            if "_" in alpha_name:
                kl_alpha = float(alpha_name.split("_", 1)[1])
        elif alpha_name == "learnable":
            learnable_alpha = True
            kl_alpha = None
        elif alpha_name == "alpha_inf":
            kl_mode = "standard"; kl_alpha = 1e6
        elif alpha_name == "alpha_ninf":
            kl_mode = "standard"; kl_alpha = 0.0
        elif alpha_name.startswith("alpha_"):
            kl_alpha = float(alpha_name.split("_", 1)[1])

        # build model & load weights
        model = VAEAC(
            model_module.reconstruction_log_prob,
            model_module.proposal_network,
            model_module.prior_network,
            model_module.generative_network,
            kl_mode=kl_mode,
            kl_alpha=kl_alpha,
            learnable_alpha=learnable_alpha,
        )
        model = model.to(imgs_t.device)
        ckpt = torch.load(ckpt_path, map_location=imgs_t.device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        for mtype in masks:
            # build masks batch
            if mtype == "center":
                masks_t = make_center_mask_like(imgs_t)
            elif mtype == "dense":
                masks_t = make_dense_mask_like(imgs_t, p=0.95)
            else:
                raise ValueError(f"Unknown mask type: {mtype}")

            with torch.no_grad():
                params = model.generate_samples_params(imgs_t, masks_t, K=1)  # [B,1,6,128,128]
                gen = sampler(params[:, 0])  # [B,3,128,128]

                # combine with observed part
                obs = imgs_t.clone()
                obs[masks_t.bool()] = 0
                comp = gen.clone()
                comp[~masks_t.bool()] = 0
                comp = comp + obs  # final inpainted

            # save triptychs
            save_dir = join(out_root, alpha_name, mtype)
            os.makedirs(save_dir, exist_ok=True)
            for i, nm in enumerate(ext_names):
                nm_base = os.path.splitext(nm)[0].lower()
                out_path = join(save_dir, f"inpaint_{mtype}_{nm_base}.png")
                save_triptych(out_path, imgs_t[i].cpu(), (imgs_t[i]*(1-masks_t[i]) + 0.5*masks_t[i]).cpu(), comp[i].cpu())

            # metrics (optional) — computed on the 4 images only (quick diagnostic)
            fid = ssim = psnr = None
            if compute_fid_flag:
                fid = compute_fid(comp, imgs_t, device=("cuda" if torch.cuda.is_available() else "cpu"))
            if compute_ssimpsnr_flag:
                ssim, psnr = compute_ssimpsnr(comp, imgs_t)

            results_rows.append(dict(alpha=alpha_name, mask=mtype, fid=fid, ssim=ssim, psnr=psnr))

    # write summary CSV
    import csv as _csv
    csv_path = join(out_root, "summary_metrics_alphas.csv")
    with open(csv_path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["alpha", "mask", "fid", "ssim", "psnr"])
        w.writeheader()
        for r in results_rows:
            w.writerow(r)
    print(f"[INFO] Wrote summary: {csv_path}")


if __name__ == "__main__":
    ap = ArgumentParser("Evaluate alpha variants on 4 external images; save triptychs + summary metrics.")
    ap.add_argument("--model_dir", type=str, required=True)
    ap.add_argument("--alphas", type=str, default="", help="Comma-separated. If empty, list folders under alpha_runs/")
    ap.add_argument("--external_dir", type=str, required=True, help="Folder with Cat.png, Dog.png, building.png, car.png")
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--masks", type=str, default="center,dense")
    ap.add_argument("--compute_fid", action="store_true", default=False)
    ap.add_argument("--compute_ssimpsnr", action="store_true", default=True)

    args = ap.parse_args()

    if args.alphas.strip():
        alpha_list = [t.strip() for t in args.alphas.split(",") if t.strip()]
    else:
        # discover alphas from folder structure
        root = join(args.model_dir, "alpha_runs")
        alpha_list = sorted([d for d in os.listdir(root) if os.path.isdir(join(root, d))])

    mask_list = [t.strip() for t in args.masks.split(",") if t.strip()]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    run_eval(
        model_dir=args.model_dir,
        alphas=alpha_list,
        external_dir=args.external_dir,
        out_root=args.out_dir,
        compute_fid_flag=args.compute_fid,
        compute_ssimpsnr_flag=args.compute_ssimpsnr,
        masks=mask_list,
        device=dev,
    )
