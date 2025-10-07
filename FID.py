import argparse
import os
import re
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import lpips
from torchvision import transforms
from torchvision.utils import save_image
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from torch_fidelity import calculate_metrics
import matplotlib.pyplot as plt
import csv

# === Image loading helper ===
def load_image(path):
    img = Image.open(path).convert("RGB").resize((128, 128), Image.BICUBIC)
    return transforms.ToTensor()(img).unsqueeze(0)  # [1, 3, H, W]

def apply_mask(img, mask):
    return img * mask + 0.5 * (1 - mask)  # masked region set to gray

def extract_common_ids(gt_dir, sample_dir, mask_dir):
    id_rx = re.compile(r"(\d{5})")
    gt_ids     = {id_rx.search(f.name).group(1) for f in gt_dir.iterdir() if "groundtruth" in f.name}
    sample_ids = {id_rx.search(f.name).group(1) for f in sample_dir.iterdir() if "sample_000" in f.name}
    mask_ids   = {id_rx.search(f.name).group(1) for f in mask_dir.iterdir() if "input" in f.name}
    return sorted(gt_ids & sample_ids & mask_ids)

def evaluate_all(gt_dir, sample_dir, mask_dir, output_csv="masked_metrics.csv"):
    gt_dir = Path(gt_dir)
    sample_dir = Path(sample_dir)
    mask_dir = Path(mask_dir)

    common_ids = extract_common_ids(gt_dir, sample_dir, mask_dir)
    assert common_ids, "No matching image IDs found in all 3 sets"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = lpips.LPIPS(net='alex').to(device)

    lpips_scores, ssim_scores, psnr_scores = [], [], []
    fid_fake_dir = Path("temp_masked_fid/fake")
    fid_real_dir = Path("temp_masked_fid/real")
    fid_fake_dir.mkdir(parents=True, exist_ok=True)
    fid_real_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Evaluating {len(common_ids)} images...")

    for i, img_id in enumerate(common_ids):
        gt_path     = gt_dir / f"{img_id}_groundtruth.jpg"
        sample_path = sample_dir / f"{img_id}_sample_000.jpg"
        mask_path   = mask_dir / f"{img_id}_input.jpg"

        gt     = load_image(gt_path).to(device)
        sample = load_image(sample_path).to(device)
        mask   = load_image(mask_path).to(device)

        mask_bin = (mask.mean(dim=1, keepdim=True) > 0.6).float()  # 1 where visible, 0 where masked

        masked_gt     = apply_mask(gt, mask_bin)
        masked_sample = apply_mask(sample, mask_bin)

        # Save for FID
        save_image(masked_gt, fid_real_dir / f"{i:06d}.png")
        save_image(masked_sample, fid_fake_dir / f"{i:06d}.png")

        # LPIPS
        lp = loss_fn(masked_gt, masked_sample).item()
        lpips_scores.append(lp)

        # SSIM / PSNR
        gt_np     = masked_gt.squeeze().permute(1, 2, 0).cpu().numpy()
        sample_np = masked_sample.squeeze().permute(1, 2, 0).cpu().numpy()
        ssim_scores.append(ssim(gt_np, sample_np, data_range=1.0, channel_axis=2))
        psnr_scores.append(psnr(gt_np, sample_np, data_range=1.0))

    # Masked FID
    fid_result = calculate_metrics(
        input1=str(fid_fake_dir), input2=str(fid_real_dir),
        fid=True, kid=False, verbose=False, cuda=torch.cuda.is_available(),
    )
    fid_value = float(fid_result["frechet_inception_distance"])

    shutil.rmtree("temp_masked_fid", ignore_errors=True)

    # Save CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Image ID", "SSIM", "PSNR", "LPIPS"])
        for img_id, ssim_val, psnr_val, lp in zip(common_ids, ssim_scores, psnr_scores, lpips_scores):
            writer.writerow([img_id, ssim_val, psnr_val, lp])
        writer.writerow(["Mean", np.mean(ssim_scores), np.mean(psnr_scores), np.mean(lpips_scores)])
        writer.writerow(["FID_masked", "", "", fid_value])

    print(f"[RESULT] Masked FID = {fid_value:.2f} | LPIPS = {np.mean(lpips_scores):.4f} | SSIM = {np.mean(ssim_scores):.4f} | PSNR = {np.mean(psnr_scores):.2f} dB")

    # Plotting
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.plot(lpips_scores, label="LPIPS")
    plt.axhline(np.mean(lpips_scores), color='r', linestyle='--', label="Mean")
    plt.title("LPIPS")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(ssim_scores, label="SSIM", color='green')
    plt.axhline(np.mean(ssim_scores), color='r', linestyle='--', label="Mean")
    plt.title("SSIM")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(psnr_scores, label="PSNR", color='orange')
    plt.axhline(np.mean(psnr_scores), color='r', linestyle='--', label="Mean")
    plt.title("PSNR")
    plt.legend()

    plt.tight_layout()
    plt.savefig("masked_metrics_plot.png")
    print("[INFO] Plots saved as masked_metrics_plot.png")

# === Entry Point ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluate Masked FID, SSIM, PSNR, LPIPS")
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--sample_dir", required=True)
    parser.add_argument("--mask_dir", required=True)
    parser.add_argument("--output_csv", default="masked_metrics.csv")
    args = parser.parse_args()

    print("[INFO] Starting full masked evaluation")
    evaluate_all(args.gt_dir, args.sample_dir, args.mask_dir, args.output_csv)

