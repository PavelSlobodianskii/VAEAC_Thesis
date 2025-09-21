# evaluate_many.py
# Evaluate VAEAC on many images (CelebA* splits) with on-the-fly masks.
# - Handles learnable alpha checkpoints automatically.
# - Generates masks on CPU (as in mask_generators.py), then moves to device.
# - Supports multiple splits via comma-separated --dataset (e.g., "celeba_val,celeba_train").
# - SSIM/PSNR are averaged over K samples; FID is computed on one sample per image (k=0).
#
# Usage:
#   python evaluate_many.py \
#     --model_dir celeba_model \
#     --checkpoint celeba_model/alpha_runs/learnable/best.tar \
#     --dataset celeba_train \
#     --out_csv celeba_model/alpha_runs/learnable/metrics_many.csv \
#     --max_images 10000 --K 4

import argparse
import csv
import tempfile
from importlib import import_module
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.utils import save_image
from tqdm import tqdm

from datasets import load_dataset
from train_utils import extend_batch
from VAEAC import VAEAC

# metrics
from skimage.metrics import structural_similarity as sk_ssim
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from torch_fidelity import calculate_metrics


def build_dataset(splits_csv: str):
    """Allow comma-separated splits, e.g. 'celeba_val,celeba_train'."""
    names = [s.strip() for s in splits_csv.split(",") if s.strip()]
    dsets = [load_dataset(n) for n in names]
    return dsets[0] if len(dsets) == 1 else ConcatDataset(dsets)


def denorm01(x: torch.Tensor) -> torch.Tensor:
    # x in [-1,1] -> [0,1]
    return (x.clamp(-1, 1) + 1) / 2.0


@torch.no_grad()
def main(args):
    # ----- setup -----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # import model module (provides networks, sampler, batch_size, mask_generator)
    mm = import_module(args.model_dir + ".model")
    sampler = mm.sampler
    batch_size = mm.batch_size
    mask_gen = mm.mask_generator   # generators in mask_generators.py expect CPU tensors

    # ----- build model & load checkpoint (supports learnable alpha) -----
    model = VAEAC(
        mm.reconstruction_log_prob,
        mm.proposal_network,
        mm.prior_network,
        mm.generative_network,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    if "raw_alpha" in state and not hasattr(model, "raw_alpha"):
        # checkpoint came from learnable-alpha training → re-init with learnable_alpha=True
        model = VAEAC(
            mm.reconstruction_log_prob,
            mm.proposal_network,
            mm.prior_network,
            mm.generative_network,
            learnable_alpha=True,
            kl_alpha=None,
        ).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()

    # ----- data (no masks dataset; masks are generated per batch) -----
    ds = build_dataset(args.dataset)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)
    total_len = len(ds)
    print(f"[INFO] Split(s): {args.dataset} | images available: {total_len}")
    if args.max_images > 0:
        print(f"[INFO] Will evaluate up to max_images={args.max_images}")

    # ----- temp folders for FID -----
    tmp_root = Path(tempfile.mkdtemp(prefix="fid_many_"))
    dir_fake = tmp_root / "inpainted"
    dir_real = tmp_root / "real"
    dir_fake.mkdir(parents=True, exist_ok=True)
    dir_real.mkdir(parents=True, exist_ok=True)

    # ----- streaming aggregates -----
    ssim_sum = 0.0
    psnr_sum = 0.0
    n_imgs = 0
    save_idx = 0

    pbar = tqdm(dl, desc="Evaluating", dynamic_ncols=True)
    for batch in pbar:
        if args.max_images > 0 and n_imgs >= args.max_images:
            break

        # keep batch on CPU for mask generation; then move both to device
        batch = extend_batch(batch, dl, batch_size)  # shape: [B,3,H,W] CPU
        init = batch.shape[0]

        masks_cpu = mask_gen(batch)  # generators produce CPU tensors by design
        batch = batch.to(device, non_blocking=True)
        masks = masks_cpu.to(device, non_blocking=True)

        # K samples per image
        params = model.generate_samples_params(batch, masks, K=args.K)   # [B,K,6,H,W]
        params = params[:init]

        gens = []
        for k in range(args.K):
            gens.append(sampler(params[:, k]))                           # list of [B,3,H,W] in [-1,1]
        gen_stack = torch.stack(gens, 1)                                  # [B,K,3,H,W]

        # mean over K for SSIM/PSNR; single sample (k=0) for FID
        gen_mean = gen_stack.mean(1)                                      # [B,3,H,W]
        comp_mean = gen_mean.clone()
        comp_mean[~masks.bool()] = batch[~masks.bool()]                   # correct composition

        comp_fid = gen_stack[:, 0].clone()
        comp_fid[~masks.bool()] = batch[~masks.bool()]

        # to [0,1] for metrics/saving
        comp_mean_01 = denorm01(comp_mean)
        comp_fid_01  = denorm01(comp_fid)
        real_01      = denorm01(batch[:init])

        # loop images
        for i in range(init):
            if args.max_images > 0 and n_imgs >= args.max_images:
                break

            # FID files (PNG, lossless)
            save_image(comp_fid_01[i], str(dir_fake / f"{save_idx:06d}.png"))
            save_image(real_01[i],     str(dir_real / f"{save_idx:06d}.png"))

            # SSIM/PSNR on averaged composite
            a = comp_mean_01[i].permute(1, 2, 0).cpu().numpy()
            b = real_01[i].permute(1, 2, 0).cpu().numpy()
            ssim_sum += sk_ssim(a, b, data_range=1.0, channel_axis=2)
            psnr_sum += sk_psnr(b, a, data_range=1.0)

            n_imgs += 1
            save_idx += 1

        pbar.set_postfix(images=n_imgs)

    # ----- compute FID on all saved images -----
    metrics = calculate_metrics(
        input1=str(dir_fake),
        input2=str(dir_real),
        fid=True, kid=False, verbose=False,
        cuda=torch.cuda.is_available(),
    )
    fid = float(metrics["frechet_inception_distance"])
    ssim = float(ssim_sum / max(1, n_imgs))
    psnr = float(psnr_sum / max(1, n_imgs))

    # ----- write CSV row -----
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_csv.exists()
    with open(out_csv, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["num_images", "K", "fid", "ssim", "psnr", "checkpoint", "splits", "seed"])
        w.writerow([n_imgs, args.K, fid, ssim, psnr, args.checkpoint, args.dataset, args.seed])

    # cleanup
    import shutil
    shutil.rmtree(tmp_root, ignore_errors=True)

    print(f"[RESULT] images={n_imgs} | K={args.K} | FID={fid:.2f} | SSIM={ssim:.4f} | PSNR={psnr:.2f} dB")
    print(f"[SAVED]  {out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser("Evaluate VAEAC metrics on many images (masks on-the-fly).")
    ap.add_argument("--model_dir", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--dataset", type=str, required=True, help="Split name(s); comma-separated allowed.")
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--max_images", type=int, default=10000, help="-1 for all available.")
    ap.add_argument("--K", type=int, default=1, help="Samples per image (mean for SSIM/PSNR; FID uses k=0).")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()
    main(args)

