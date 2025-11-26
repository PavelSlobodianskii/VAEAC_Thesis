# filename: train_alphas.py
# Train ORIGINAL VAEAC for multiple KL variants (alpha values) with full Option‑4 logging.
from argparse import ArgumentParser
from importlib import import_module
from math import ceil
from os import makedirs, replace
from os.path import exists, join
from shutil import copy
from sys import stderr
import csv
import os
import random
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# Optional libraries (safe guards)
try:
    from metrics import compute_fid, compute_ssim_psnr
except Exception:
    compute_fid = None
    compute_ssim_psnr = None

try:
    from viz import tsne_latents
except Exception:
    tsne_latents = None

from datasets import load_dataset, LengthBounder
from train_utils import extend_batch, get_validation_iwae, make_mask_on_batch_device
from VAEAC import VAEAC


# ================================================================
# Utilities
# ================================================================
def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_alpha_token(tok: str):
    tok = tok.strip().lower()
    if tok.startswith("symmetric"):
        if ":" in tok:
            _, w = tok.split(":")
            return dict(name=f"symmetric_{w}",
                        kl_mode="symmetric",
                        kl_alpha=float(w),
                        learnable_alpha=False)
        else:
            return dict(name="symmetric_1.0",
                        kl_mode="symmetric",
                        kl_alpha=1.0,
                        learnable_alpha=False)

    if tok == "learnable":
        return dict(name="learnable",
                    kl_mode="standard",
                    kl_alpha=None,
                    learnable_alpha=True)

    if tok in ("inf", "+inf"):
        return dict(name="alpha_inf",
                    kl_mode="standard",
                    kl_alpha=1e6,
                    learnable_alpha=False)

    if tok in ("-inf", "ninf"):
        return dict(name="alpha_ninf",
                    kl_mode="standard",
                    kl_alpha=0.0,
                    learnable_alpha=False)

    try:
        val = float(tok)
    except Exception:
        raise ValueError(f"Unrecognized alpha token: {tok}")

    return dict(name=f"alpha_{val}",
                kl_mode="standard",
                kl_alpha=val,
                learnable_alpha=False)


def compute_recon_samples(model, batch, mask, sampler):
    """Generate reconstruction samples for metric computation (safe)."""
    try:
        with torch.no_grad():
            params = model.generate_samples_params(batch, mask, K=1)
            return sampler(params[:, 0])
    except Exception:
        return None


def get_grad_norms(params):
    """Compute gradient norms (safe logging)."""
    total_sq = 0.0
    for p in params:
        if p.grad is not None:
            total_sq += float((p.grad.detach() ** 2).sum().item())
    total_norm = float(total_sq ** 0.5) if total_sq > 0 else 0.0
    return {"grad_total": total_norm, "grad_model": total_norm}


# ================================================================
# Main training loop for a SINGLE alpha variant
# ================================================================
def run_one_alpha(args, alpha_cfg, device="cuda"):

    # ------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------
    model_module = import_module(args.model_dir + '.model')
    out_root = join(args.model_dir, f"alpha_runs/{alpha_cfg['name']}")
    makedirs(out_root, exist_ok=True)

    train_dataset = load_dataset(args.train_dataset)
    if args.max_train_images > 0:
        train_dataset = LengthBounder(train_dataset, args.max_train_images)
    val_dataset = load_dataset(args.validation_dataset)

    dl = DataLoader(train_dataset, batch_size=model_module.batch_size,
                    shuffle=True, drop_last=False, num_workers=args.num_workers)
    val_dl = DataLoader(val_dataset, batch_size=model_module.batch_size,
                        shuffle=True, drop_last=False, num_workers=args.num_workers)

    validation_batches = ceil(len(dl) / args.validations_per_epoch)

    # ------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------
    model = VAEAC(
        model_module.reconstruction_log_prob,
        model_module.proposal_network,
        model_module.prior_network,
        model_module.generative_network,
        debug_asserts=args.debug_asserts,
        kl_mode=alpha_cfg["kl_mode"],
        kl_alpha=alpha_cfg["kl_alpha"],
        learnable_alpha=alpha_cfg["learnable_alpha"],
        alpha_init=args.alpha_init,
        alpha_max=args.alpha_max,
        free_bits=args.free_bits,
    ).to(device if torch.cuda.is_available() else "cpu")

    optimizer = model_module.optimizer(model.parameters())
    sampler = getattr(model_module, "sampler", None)
    vlb_scale = getattr(model_module, "vlb_scale_factor", 1)
    mask_gen = model_module.mask_generator

    # ------------------------------------------------------------
    # Resume checkpoint if exists
    # ------------------------------------------------------------
    last_ckpt = join(out_root, "last.tar")

    validation_iwae = []
    train_vlb = []
    rec_errors = []
    kl_terms = []
    alpha_log = []           # ADDED
    epoch_loss_logs = []     # ADDED

    start_epoch = 0
    if exists(last_ckpt):
        ckpt = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        validation_iwae = ckpt.get("validation_iwae", [])
        train_vlb = ckpt.get("train_vlb", [])
        rec_errors = ckpt.get("rec_errors", [])
        kl_terms = ckpt.get("kl_terms", [])
        alpha_log = ckpt.get("alpha_log", [])
        epoch_loss_logs = ckpt.get("loss_logs", [])
        start_epoch = ckpt.get("epoch", 0) + 1

    # ------------------------------------------------------------
    # CSV Metrics
    # ------------------------------------------------------------
    csv_path = join(out_root, "metrics_alphas.csv")
    if not exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow([
                "epoch", "step", "split",
                "train_vlb", "val_iwae",
                "recon", "kl",
                "fid", "ssim", "psnr",
                "alpha",
                "lr",
                "grad_total", "grad_model"
            ])

    # ------------------------------------------------------------
    def save_ckpt(epoch):
        tmp = last_ckpt + ".bak"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_iwae": validation_iwae,
            "train_vlb": train_vlb,
            "rec_errors": rec_errors,
            "kl_terms": kl_terms,
            "alpha_log": alpha_log,
            "loss_logs": epoch_loss_logs,
        }, tmp)
        replace(tmp, last_ckpt)

    # ============================================================
    # Training loop
    # ============================================================
    for epoch in range(start_epoch, args.epochs):

        iterator = tqdm(dl, desc=f"[{alpha_cfg['name']}] Epoch {epoch+1}/{args.epochs}") if args.verbose else dl
        avg_vlb = 0.0
        batch_logs = []

        last_batch = None
        last_mask = None

        for i, batch in enumerate(iterator):

            # ===================================================================
            # Validation checkpoint (IWAE, metrics, CSV, checkpoint)
            # ===================================================================
            if any([i == 0 and epoch == start_epoch,
                    i % validation_batches == validation_batches - 1,
                    i + 1 == len(dl)]):

                val_i = get_validation_iwae(val_dl, mask_gen,
                                            model_module.batch_size, model,
                                            args.validation_iwae_num_samples,
                                            verbose=args.verbose)
                validation_iwae.append(val_i)
                train_vlb.append(avg_vlb)

                # -------------- Metrics at validation checkpoint --------------
                fid_score = ssim_val = psnr_val = None
                if sampler is not None and hasattr(model, "generate_samples_params"):
                    try:
                        vb = next(iter(val_dl))
                        if torch.cuda.is_available(): vb = vb.cuda()
                        vmask = make_mask_on_batch_device(mask_gen, vb)

                        gen = compute_recon_samples(model, vb, vmask, sampler)

                        if args.compute_fid and compute_fid is not None and gen is not None:
                            fid_score = compute_fid(gen, vb, device=("cuda" if torch.cuda.is_available() else "cpu"))
                        if args.compute_ssimpsnr and compute_ssim_psnr is not None and gen is not None:
                            ssim_val, psnr_val = compute_ssim_psnr(gen, vb)
                    except Exception as e:
                        print(f"[WARN] Validation FID/SSIM/PSNR failed: {e}")

                # Latest known logs
                last_rec = rec_errors[-1] if rec_errors else None
                last_kl = kl_terms[-1] if kl_terms else None
                last_alpha = alpha_log[-1] if alpha_log else None
                last_grad_total = batch_logs[-1]["grad_total"] if batch_logs else None
                last_grad_model = batch_logs[-1]["grad_model"] if batch_logs else None

                with open(csv_path, "a", newline="") as f:
                    csv.writer(f).writerow([
                        epoch+1, i, "val",
                        avg_vlb, val_i,
                        last_rec, last_kl,
                        fid_score, ssim_val, psnr_val,
                        last_alpha,
                        optimizer.param_groups[0]["lr"],
                        last_grad_total, last_grad_model
                    ])

                save_ckpt(epoch)

                # Best checkpoint
                best_path = join(out_root, "best.tar")
                if max(validation_iwae[::-1]) <= val_i:
                    tmp = best_path + ".bak"
                    copy(last_ckpt, tmp)
                    replace(tmp, best_path)

                if args.verbose:
                    print(file=stderr)
                    print(file=stderr)

            # ===================================================================
            # Normal batch
            # ===================================================================
            batch = extend_batch(batch, dl, model_module.batch_size)
            mask = mask_gen(batch)

            if torch.cuda.is_available():
                batch = batch.cuda(non_blocking=True)
                mask = mask.cuda(non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            vlb = model.batch_vlb(batch, mask).mean()
            loss = -vlb / vlb_scale
            loss.backward()

            grad_norms = get_grad_norms(model.parameters())
            optimizer.step()

            # Running average VLB
            avg_vlb += (float(vlb) - avg_vlb) / (i + 1)

            # Detailed metrics per batch (safe)
            rec = kl = alpha_val = None
            with torch.no_grad():
                # Reconstruction & KL
                try:
                    q, p = model.make_latent_distributions(batch, mask)
                    z = q.rsample()
                    rec_params = model.generative_network(z)

                    rec_val = model.rec_log_prob(batch, rec_params, mask)
                    rec = float(rec_val.mean().item())

                    kl_val = torch.distributions.kl_divergence(q, p)
                    kl = float(kl_val.view(batch.shape[0], -1).sum(-1).mean().item())
                except Exception:
                    pass

                if hasattr(model, "_alpha_value"):
                    try:
                        alpha_val = float(model._alpha_value().item())
                    except:
                        alpha_val = None

            rec_errors.append(rec)
            kl_terms.append(kl)
            alpha_log.append(alpha_val)

            # Save per-batch logs
            batch_logs.append({
                "epoch": epoch + 1,
                "step": i,
                "vlb": float(vlb.item()),
                "rec": rec,
                "kl": kl,
                "alpha": alpha_val,
                "grad_total": grad_norms["grad_total"],
                "grad_model": grad_norms["grad_model"],
            })

            if args.verbose and isinstance(iterator, tqdm):
                iterator.set_postfix(
                    vlb=f"{avg_vlb:.2f}",
                    rec=f"{rec:.2f}",
                    kl=f"{kl:.2f}",
                    alpha=f"{alpha_val}",
                    grad=f"{grad_norms['grad_total']:.3f}"
                )

            last_batch = batch
            last_mask = mask

        # ------------------------------------------------------------------
        # End of epoch: epoch-level CSV, latent save, t-SNE, pretty print
        # ------------------------------------------------------------------
        epoch_loss_logs.extend(batch_logs)
        with open(join(out_root, f"loss_logs_epoch_{epoch+1}.pkl"), "wb") as f:
            pickle.dump(batch_logs, f)

        # Pretty summary
        if last_batch is not None:
            with torch.no_grad():
                q, p = model.make_latent_distributions(last_batch, last_mask)
                z = q.rsample()
                rec_params = model.generative_network(z)

            bar = "-" * 72
            print(f"+{bar}+")
            print(f"| Alpha: {alpha_cfg['name']:<20} | Epoch: {epoch+1}/{args.epochs:<4} | Avg VLB: {avg_vlb:>10.3f} |")
            print(f"+{bar}+")
            print(f"| Last batch : {tuple(last_batch.shape)}")
            print(f"| Mask       : {tuple(last_mask.shape)}")
            print(f"| q.mean     : {tuple(q.mean.shape)}")
            print(f"| p.mean     : {tuple(p.mean.shape)}")
            print(f"| z          : {tuple(z.shape)}")
            print(f"| rec_params : {tuple(rec_params.shape)}")
            print(f"+{bar}+")

        # Epoch-end metrics
        fid_score = ssim_val = psnr_val = None
        if sampler is not None and hasattr(model, "generate_samples_params"):
            try:
                vb = next(iter(val_dl))
                if torch.cuda.is_available(): vb = vb.cuda()
                vmask = make_mask_on_batch_device(mask_gen, vb)
                gen = compute_recon_samples(model, vb, vmask, sampler)
                if args.compute_fid and compute_fid is not None and gen is not None:
                    fid_score = compute_fid(gen, vb, device=("cuda" if torch.cuda.is_available() else "cpu"))
                if args.compute_ssimpsnr and compute_ssim_psnr is not None and gen is not None:
                    ssim_val, psnr_val = compute_ssim_psnr(gen, vb)
            except Exception as e:
                print(f"[WARN] Epoch-end metrics failed: {e}")

        last_rec = rec_errors[-1] if rec_errors else None
        last_kl = kl_terms[-1] if kl_terms else None
        last_alpha = alpha_log[-1] if alpha_log else None
        last_grad_total = batch_logs[-1]["grad_total"] if batch_logs else None
        last_grad_model = batch_logs[-1]["grad_model"] if batch_logs else None

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch+1, "epoch_end", "train",
                avg_vlb, "",
                last_rec, last_kl,
                fid_score, ssim_val, psnr_val,
                last_alpha,
                optimizer.param_groups[0]["lr"],
                last_grad_total, last_grad_model
            ])

        # Save latent means & t-SNE
        if hasattr(model, "latent_means"):
            try:
                with torch.no_grad():
                    lat_means = model.latent_means(last_batch, last_mask).cpu().numpy()
                np.savez_compressed(join(out_root, f"latents_epoch_{epoch+1}.npz"),
                                    latents=lat_means)
                if tsne_latents is not None and args.tsne_every > 0 and (epoch+1) % args.tsne_every == 0:
                    tsne_latents(lat_means, labels=None,
                                 out_path=join(out_root, f"tsne_epoch_{epoch+1}.png"))
            except:
                pass

    # ============================================================
    # Save global history + plots
    # ============================================================
    print(f"[INFO] Finished: {alpha_cfg['name']} — Saving history & plots…")

    with open(join(out_root, "history.pkl"), "wb") as f:
        pickle.dump({
            "validation_iwae": validation_iwae,
            "train_vlb": train_vlb,
            "rec_errors": rec_errors,
            "kl_terms": kl_terms,
            "alpha_log": alpha_log,
            "loss_logs": epoch_loss_logs,
        }, f)

    # Plot: Train VLB vs Validation IWAE
    try:
        plt.figure(figsize=(12, 5))
        plt.plot(validation_iwae, label="Validation IWAE", marker="o")
        plt.plot(train_vlb, label="Train VLB", marker="x")
        plt.grid(True, alpha=0.4)
        plt.xlabel("Validation checkpoint")
        plt.ylabel("Value")
        plt.title(f"[{alpha_cfg['name']}] VLB / IWAE")
        plt.legend()
        plt.tight_layout()
        plt.savefig(join(out_root, "loss_curves.png"), dpi=150)
        plt.close()
    except:
        pass

    # Plot: Reconstruction vs KL (per batch)
    try:
        plt.figure(figsize=(12, 5))
        plt.plot(rec_errors, label="Reconstruction", linestyle="--")
        plt.plot(kl_terms, label="KL", linestyle="-.")
        plt.grid(True, alpha=0.4)
        plt.xlabel("Batch")
        plt.ylabel("Value")
        plt.title(f"[{alpha_cfg['name']}] Reconstruction & KL per batch")
        plt.legend()
        plt.tight_layout()
        plt.savefig(join(out_root, "recon_kl_per_batch.png"), dpi=150)
        plt.close()
    except:
        pass


# ================================================================
# Main CLI
# ================================================================
if __name__ == "__main__":
    p = ArgumentParser("Train ORIGINAL VAEAC across multiple KL alpha regimes.")

    # Required
    p.add_argument("--model_dir", type=str, required=True)
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--train_dataset", type=str, required=True)
    p.add_argument("--validation_dataset", type=str, required=True)

    # Alpha list
    p.add_argument("--alphas", type=str,
                   default="0,0.5,1,inf,-inf,learnable,symmetric",
                   help="Comma-separated alpha tokens.")

    # Dataset trimming
    p.add_argument("--max_train_images", type=int, default=25000)

    # IWAE eval frequency
    p.add_argument("--validation_iwae_num_samples", type=int, default=25)
    p.add_argument("--validations_per_epoch", type=int, default=5)

    # General
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--verbose", action="store_true", default=True)

    # Metrics
    p.add_argument("--compute_fid", action="store_true", default=False)
    p.add_argument("--compute_ssimpsnr", action="store_true", default=False)
    p.add_argument("--tsne_every", type=int, default=0)

    # Stability
    p.add_argument("--amp", action="store_true", default=False)
    p.add_argument("--debug_asserts", action="store_true", default=False)

    # KL options
    p.add_argument("--alpha_init", type=float, default=1.0)
    p.add_argument("--alpha_max", type=float, default=1e6)
    p.add_argument("--free_bits", type=float, default=0.0)

    args = p.parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokens = [t for t in args.alphas.split(",") if t.strip()]
    cfgs = [parse_alpha_token(t) for t in tokens]

    print("[INFO] Running alpha variants:", ", ".join([c["name"] for c in cfgs]))

    for cfg in cfgs:
        run_one_alpha(args, cfg, device=device)
