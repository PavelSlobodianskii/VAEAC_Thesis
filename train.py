# filename: train_alphas.py
# Train ORIGINAL VAEAC for a list of KL variants/weights (alphas) and save per-alpha plots & metrics.
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

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle

from datasets import load_dataset, LengthBounder
from train_utils import extend_batch, get_validation_iwae, make_mask_on_batch_device
from VAEAC import VAEAC
from metrics import compute_fid, compute_ssim_psnr
from viz import tsne_latents


def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_alpha_token(tok: str):
    """
    Map CLI tokens to a config dict for the VAEAC alpha/KL mode.
    Allowed tokens:
      0, 0.5, 1, inf, -inf, learnable, symmetric, symmetric:0.5 (etc)
    """
    tok = tok.strip().lower()
    # symmetric optionally with weight (default 1.0)
    if tok.startswith("symmetric"):
        if ":" in tok:
            _, w = tok.split(":")
            return dict(name=f"symmetric_{w}", kl_mode="symmetric", kl_alpha=float(w), learnable_alpha=False)
        else:
            return dict(name="symmetric_1.0", kl_mode="symmetric", kl_alpha=1.0, learnable_alpha=False)

    if tok == "learnable":
        return dict(name="learnable", kl_mode="standard", kl_alpha=None, learnable_alpha=True)

    if tok in ("inf", "+inf"):
        return dict(name="alpha_inf", kl_mode="standard", kl_alpha=1e6, learnable_alpha=False)

    if tok in ("-inf", "ninf"):
        # equivalent to "ignore KL"
        return dict(name="alpha_ninf", kl_mode="standard", kl_alpha=0.0, learnable_alpha=False)

    # numeric
    try:
        val = float(tok)
    except Exception:
        raise ValueError(f"Unrecognized alpha token: {tok}")

    return dict(name=f"alpha_{val}", kl_mode="standard", kl_alpha=val, learnable_alpha=False)


def run_one_alpha(args, alpha_cfg, device="cuda"):
    # ---------------- paths & bookkeeping ----------------
    model_module = import_module(args.model_dir + '.model')  # ORIGINAL model.py (no swap)
    out_root = join(args.model_dir, f"alpha_runs/{alpha_cfg['name']}")
    makedirs(out_root, exist_ok=True)

    # datasets
    train_dataset = load_dataset(args.train_dataset)
    if args.max_train_images > 0:
        train_dataset = LengthBounder(train_dataset, args.max_train_images)
    val_dataset = load_dataset(args.validation_dataset)

    dl = DataLoader(train_dataset, batch_size=model_module.batch_size, shuffle=True,
                    drop_last=False, num_workers=args.num_workers)
    val_dl = DataLoader(val_dataset, batch_size=model_module.batch_size, shuffle=True,
                        drop_last=False, num_workers=args.num_workers)
    validation_batches = ceil(len(dl) / args.validations_per_epoch)

    # model
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
    sampler = getattr(model_module, "sampler")
    vlb_scale = getattr(model_module, "vlb_scale_factor", 1)
    mask_gen = model_module.mask_generator

    # resume?
    last_ckpt = join(out_root, "last.tar")
    validation_iwae, train_vlb, rec_errors, kl_terms = [], [], [], []
    start_epoch = 0
    if exists(last_ckpt):
        ckpt = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        validation_iwae = ckpt.get("validation_iwae", [])
        train_vlb = ckpt.get("train_vlb", [])
        rec_errors = ckpt.get("rec_errors", [])
        kl_terms = ckpt.get("kl_terms", [])
        start_epoch = ckpt.get("epoch", 0) + 1

    # CSV metrics
    csv_path = join(out_root, "metrics_alphas.csv")
    if not exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["epoch", "step", "split", "train_vlb", "val_iwae", "recon", "kl", "fid", "ssim", "psnr", "lr"]
            )

    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and torch.cuda.is_available()))

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
        }, tmp)
        replace(tmp, last_ckpt)

    # ----------------------- training loop -----------------------
    for epoch in range(start_epoch, args.epochs):
        iterator = tqdm(dl, desc=f"[{alpha_cfg['name']}] Epoch {epoch+1}/{args.epochs}") if args.verbose else dl
        avg_vlb = 0.0
        last_batch = last_mask = None

        for i, batch in enumerate(iterator):
            # periodic validation
            if any([i == 0 and epoch == start_epoch, i % validation_batches == validation_batches - 1, i + 1 == len(dl)]):
                val_i = get_validation_iwae(val_dl, mask_gen, model_module.batch_size, model,
                                            args.validation_iwae_num_samples, verbose=args.verbose)
                validation_iwae.append(val_i)
                train_vlb.append(avg_vlb)
                save_ckpt(epoch)

                # best snapshot
                best_path = join(out_root, "best.tar")
                if max(validation_iwae[::-1]) <= val_i:
                    tmp = best_path + ".bak"
                    copy(last_ckpt, tmp)
                    replace(tmp, best_path)

                # CSV row
                with open(csv_path, "a", newline="") as f:
                    csv.writer(f).writerow([epoch+1, i, "val", avg_vlb, val_i, "", "", "", "", "", optimizer.param_groups[0]["lr"]])
                if args.verbose: print(file=stderr); print(file=stderr)

            batch = extend_batch(batch, dl, model_module.batch_size)
            mask = mask_gen(batch)

            if torch.cuda.is_available():
                batch = batch.cuda(non_blocking=True)
                mask = mask.cuda(non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(args.amp and torch.cuda.is_available())):
                vlb = model.batch_vlb(batch, mask).mean()
                loss = -vlb / vlb_scale
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                q, p = model.make_latent_distributions(batch, mask)
                z = q.rsample()
                rec_params = model.generative_network(z)
                rec = float(model.rec_log_prob(batch, rec_params, mask).mean().item())
                kl = float(torch.distributions.kl_divergence(q, p).view(batch.shape[0], -1).sum(-1).mean().item())
                rec_errors.append(rec)
                kl_terms.append(kl)

            avg_vlb += (float(vlb) - avg_vlb) / (i + 1)
            if args.verbose and isinstance(iterator, tqdm):
                iterator.set_postfix(vlb=f"{avg_vlb:.1f}", alpha=model._alpha_value())

            last_batch, last_mask = batch, mask

        # epoch end pretty print
        with torch.no_grad():
            q, p = model.make_latent_distributions(last_batch, last_mask)
            z = q.rsample()
            rec_params = model.generative_network(z)

        bar = "-" * 72
        print(f"+{bar}+")
        print(f"| Alpha: {alpha_cfg['name']:<20} | Epoch: {epoch+1}/{args.epochs:<4} | Avg VLB: {avg_vlb:>10.3f} |")
        print(f"+{bar}+")
        print(f"| Batch           : {tuple(last_batch.shape)}")
        print(f"| Mask            : {tuple(last_mask.shape)}")
        print(f"| q.mean          : {tuple(q.mean.shape)}")
        print(f"| p.mean          : {tuple(p.mean.shape)}")
        print(f"| z               : {tuple(z.shape)}")
        print(f"| rec_params      : {tuple(rec_params.shape)}")
        print(f"+{bar}+")

        # quick validation metrics on small batch
        fid_score = ssim = psnr = None
        if args.compute_fid or args.compute_ssimpsnr:
            val_it = iter(val_dl)
            try: vb = next(val_it)
            except StopIteration:
                val_it = iter(val_dl); vb = next(val_it)
            vb = vb.cuda() if torch.cuda.is_available() else vb
            vmask = make_mask_on_batch_device(mask_gen, vb)
            with torch.no_grad():
                params = model.generate_samples_params(vb, vmask, K=1)
                gen = sampler(params[:, 0])
            if args.compute_fid:
                fid_score = compute_fid(gen, vb, device=("cuda" if torch.cuda.is_available() else "cpu"))
            if args.compute_ssimpsnr:
                ssim, psnr = compute_ssim_psnr(gen, vb)

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch+1, "epoch_end", "train", avg_vlb, "", "", "", fid_score, ssim, psnr, optimizer.param_groups[0]["lr"]])

        # save latents & optional TSNE
        with torch.no_grad():
            lat_means = model.latent_means(last_batch, last_mask).cpu().numpy()
        np.savez_compressed(join(out_root, f"latents_epoch_{epoch+1}_alphas.npz"), latents=lat_means)
        if args.tsne_every and ((epoch + 1) % args.tsne_every == 0):
            tsne_latents(lat_means, labels=None, out_path=join(out_root, f"tsne_epoch_{epoch+1}_alphas.png"))

    # --------------- save plots after training ---------------
    print(f"[INFO] ({alpha_cfg['name']}) training complete – saving history & plots…")
    with open(join(out_root, "history.pkl"), "wb") as f:
        pickle.dump(
            {"validation_iwae": validation_iwae, "train_vlb": train_vlb, "rec_errors": rec_errors, "kl_terms": kl_terms},
            f,
        )

    # (a) Train VLB vs Val IWAE
    plt.figure(figsize=(14, 6))
    plt.plot(validation_iwae, label="Validation IWAE", marker="o")
    plt.plot(train_vlb, label="Train VLB", marker="x")
    plt.xlabel("Validation checkpoint")
    plt.ylabel("Loss / Value")
    plt.title(f"[{alpha_cfg['name']}] Validation IWAE and Train VLB")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(join(out_root, "loss_curves.png"), dpi=200)
    plt.close()

    # (b) Reconstruction & KL per batch
    plt.figure(figsize=(14, 6))
    plt.plot(rec_errors, label="Reconstruction Error", linestyle="--")
    plt.plot(kl_terms, label="KL Divergence", linestyle="-.")
    plt.xlabel("Batch")
    plt.ylabel("Loss / Value")
    plt.title(f"[{alpha_cfg['name']}] Reconstruction Error and KL (per batch)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(join(out_root, "recon_kl_per_batch.png"), dpi=200)
    plt.close()

    # (c) Combined quick chart
    plt.figure(figsize=(12, 5))
    plt.plot(validation_iwae, label="Validation IWAE", marker="o")
    plt.plot(train_vlb, label="Train VLB", marker="x")
    plt.plot(np.linspace(0, len(validation_iwae) - 1, num=len(rec_errors)), rec_errors, alpha=0.5, label="Recon (per batch)")
    plt.plot(np.linspace(0, len(validation_iwae) - 1, num=len(kl_terms)), kl_terms, alpha=0.5, label="KL (per batch)")
    plt.xlabel("Checkpoint")
    plt.ylabel("Value")
    plt.title(f"[{alpha_cfg['name']}] All metrics")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(join(out_root, "all_metrics_alphas.png"), dpi=200)
    plt.close()


if __name__ == "__main__":
    p = ArgumentParser("Train ORIGINAL VAEAC with multiple KL weights/variants.")
    # core datasets/paths
    p.add_argument("--model_dir", type=str, required=True)
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--train_dataset", type=str, required=True)
    p.add_argument("--validation_dataset", type=str, required=True)

    # alpha list (comma-separated tokens)
    p.add_argument(
        "--alphas",
        type=str,
        default="0,0.5,1,inf,-inf,learnable,symmetric",
        help='Comma-separated: e.g. "0,0.5,1,inf,-inf,learnable,symmetric,symmetric:0.5"',
    )

    # misc
    p.add_argument("--max_train_images", type=int, default=25000)
    p.add_argument("--validation_iwae_num_samples", type=int, default=25)
    p.add_argument("--validations_per_epoch", type=int, default=5)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--verbose", action="store_true", default=True)

    # metrics & viz
    p.add_argument("--compute_fid", action="store_true", default=False)
    p.add_argument("--compute_ssimpsnr", action="store_true", default=False)
    p.add_argument("--tsne_every", type=int, default=0)

    # stability
    p.add_argument("--amp", action="store_true", default=False)
    p.add_argument("--debug_asserts", action="store_true", default=False)

    # KL extra knobs
    p.add_argument("--alpha_init", type=float, default=1.0)
    p.add_argument("--alpha_max", type=float, default=1e6)
    p.add_argument("--free_bits", type=float, default=0.0)

    args = p.parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokens = [t for t in args.alphas.split(",") if t.strip()]
    cfgs = [parse_alpha_token(t) for t in tokens]

    print("[INFO] KL variants to run:", ", ".join([c["name"] for c in cfgs]))
    for cfg in cfgs:
        run_one_alpha(args, cfg, device=device)

