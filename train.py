from __future__ import annotations

import argparse
import pickle
import time
from collections import deque
from importlib import import_module
from math import ceil
from os import replace
from os.path import exists, join
from shutil import copy

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from datasets import load_dataset
from train_utils import extend_batch, get_validation_iwae
from VAEAC import VAEAC
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from importlib import import_module
model_module = import_module("model")

# ----------------------------- CLI ----------------------------------------- #

parser = argparse.ArgumentParser(description="Train VAEAC to inpaint.")

parser.add_argument("--model_dir", required=True,
                    help="Directory containing model.py and where checkpoints "
                         "will be stored.")
parser.add_argument("--epochs", type=int, required=True,
                    help="Number of training epochs.")
parser.add_argument("--train_dataset", required=True,
                    help="Name/key of the training dataset – see datasets.py.")
parser.add_argument("--validation_dataset", required=True,
                    help="Name/key of the validation dataset – see datasets.py.")
parser.add_argument("--validation_iwae_num_samples", type=int, default=25,
                    help="#samples per object to estimate IWAE. Default: 25.")
parser.add_argument("--validations_per_epoch", type=int, default=5,
                    help="How many times to run IWAE per epoch. Default: 5.")
parser.add_argument("--smooth_window", type=int, default=200,
                    help="Rolling window (in batches) for smoothed VLB in tqdm.")

# New argument: set a fixed alpha for symmetric KL (default: learnable)
parser.add_argument("--fixed_alpha", type=float, default=None,
                    help="Set fixed alpha for symmetric KL (otherwise learnable).")

# New arguments: to limit number of images in train/val
parser.add_argument("--limit_train_images", type=int, default=None,
                    help="Limit number of images from training set (default: all).")
parser.add_argument("--limit_val_images", type=int, default=None,
                    help="Limit number of images from validation set (default: all).")

args = parser.parse_args()

# ...after parser.parse_args() and args = parser.parse_args()
if not os.path.exists(args.model_dir):
    os.makedirs(args.model_dir, exist_ok=True)

# -------------------------- Torch & environment ---------------------------- #

torch.set_printoptions(precision=3, sci_mode=False)
use_cuda = torch.cuda.is_available()
num_workers = 0  # Set >0 if your system supports

# ----------------------------- Model --------------------------------------- #

model_module = import_module("model")

model = VAEAC(
    model_module.reconstruction_log_prob,
    model_module.proposal_network,
    model_module.prior_network,
    model_module.generative_network,
    alpha=args.fixed_alpha  # Pass fixed alpha or use learnable by default
)
if use_cuda:
    model.cuda()

optimizer = model_module.optimizer(model.parameters())
batch_size: int = model_module.batch_size
vlb_scale_factor: float = getattr(model_module, "vlb_scale_factor", 1.0)
mask_generator = model_module.mask_generator

# --------------------------- Datasets & loaders ---------------------------- #

train_dataset = load_dataset(args.train_dataset)
val_dataset = load_dataset(args.validation_dataset)

# Limit train/val size if requested
if args.limit_train_images is not None:
    train_dataset = Subset(train_dataset, range(args.limit_train_images))
if args.limit_val_images is not None:
    val_dataset = Subset(val_dataset, range(args.limit_val_images))

dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                        drop_last=False, num_workers=num_workers)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,
                            drop_last=False, num_workers=num_workers)

validation_batches = ceil(len(dataloader) / args.validations_per_epoch)

# ------------------------------ Checkpointing ------------------------------ #

validation_iwae: list[float] = []
train_vlb: list[float] = []
alpha_history: list[float] = []

# --- Added: Track rec_error and KL term ---
rec_errors: list[float] = []
kl_terms: list[float] = []

ckpt_path = join(args.model_dir, "last_checkpoint.tar")

if exists(ckpt_path):
    location = "cuda" if use_cuda else "cpu"
    ckpt = torch.load(ckpt_path, map_location=location)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    validation_iwae = ckpt.get("validation_iwae", [])
    train_vlb = ckpt.get("train_vlb", [])
    alpha_history = ckpt.get("alpha_history", [])
    print(f"[INFO] Resumed from {ckpt_path}")

def save_checkpoint(epoch_idx: int) -> None:
    tmp = ckpt_path + ".bak"
    torch.save({
        "epoch": epoch_idx,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "validation_iwae": validation_iwae,
        "train_vlb": train_vlb,
        "alpha_history": alpha_history,
    }, tmp)
    replace(tmp, ckpt_path)

# ------------------------------- Training ---------------------------------- #

for epoch in range(args.epochs):
    epoch_start = time.time()

    iterator = tqdm(enumerate(dataloader), total=len(dataloader),
                    desc="Train VLB")

    # Running statistics
    avg_vlb = 0.0
    vlb_window: deque[float] = deque(maxlen=args.smooth_window)

    last_batch = last_mask = None  # for after-epoch tensor-shape print

    for i, batch in iterator:
        batch = extend_batch(batch, dataloader, batch_size)
        mask = mask_generator(batch)

        if use_cuda:
            batch = batch.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        # ------ CHANGED: get rec_loss and kl value ------
        vlb_out = model.batch_vlb(batch, mask, return_details=True)
        vlb, rec_loss_val, kl_val = vlb_out
        vlb = vlb.mean()
        (-vlb / vlb_scale_factor).backward()
        optimizer.step()

        # Track reconstruction error and KL for each batch
        rec_errors.append(rec_loss_val)
        kl_terms.append(kl_val)

        # ----- stats -----
        avg_vlb += (float(vlb) - avg_vlb) / (i + 1)
        vlb_window.append(float(vlb))
        smooth_vlb = sum(vlb_window) / len(vlb_window)

        iterator.set_description(f"Train VLB (avg{args.smooth_window}): {smooth_vlb:.3f}")

        last_batch, last_mask = batch, mask

        # ---- Validation & checkpoints ----
        if any([
            (i == 0 and epoch == 0),
            (i % validation_batches == validation_batches - 1),
            (i + 1 == len(dataloader)),
        ]):
            val_iwae = get_validation_iwae(
                val_dataloader, mask_generator, batch_size, model,
                args.validation_iwae_num_samples, verbose=True,
            )
            validation_iwae.append(val_iwae)
            train_vlb.append(avg_vlb)

            # --- Track alpha (always, fixed or learned) ---
            alpha_value = float(model.alpha.item()) if hasattr(model, "alpha") else None
            alpha_history.append(alpha_value)
            print(f"Epoch {epoch + 1}, alpha (KL weight): {alpha_value:.4f}")

            save_checkpoint(epoch)

            # Save best checkpoint by validation IWAE
            if val_iwae >= max(validation_iwae):
                best_path = join(args.model_dir, "best_checkpoint.tar")
                copy(ckpt_path, best_path + ".bak")
                replace(best_path + ".bak", best_path)

    # ----------------- end of epoch: pretty print ------------------------- #
    with torch.no_grad():
        proposal, prior = model.make_latent_distributions(last_batch, last_mask)
        latent = proposal.rsample()
        rec_params = model.generative_network(latent)

    bar = "-" * 70
    print(f"+{bar}+")
    print(f"| Epoch: {epoch + 1}/{args.epochs:<6} | Average Loss: {avg_vlb:.3f} |")
    print(f"+{bar}+")
    print(f"| Batch shape                 : {tuple(last_batch.shape)} |")
    print(f"| Mask shape                  : {tuple(last_mask.shape)} |")
    print(f"| Proposal mean shape         : {tuple(proposal.mean.shape)} |")
    print(f"| Prior mean shape            : {tuple(prior.mean.shape)} |")
    print(f"| Latent sample shape         : {tuple(latent.shape)} |")
    print(f"| Reconstruction params shape : {tuple(rec_params.shape)} |")
    print(f"+{bar}+")

    elapsed_min = (time.time() - epoch_start) / 60
    print(f"*** Epoch {epoch + 1}/{args.epochs} | Train Loss: {avg_vlb:.3f} | "
          f"Time: {elapsed_min:.2f} min\n")

    # Extra safety: sync checkpoint at the *very* end of epoch as well
    save_checkpoint(epoch)

# ----------------------------- After training ------------------------------ #
print("[INFO] Training complete – saving history & plots…")

hist_path = join(args.model_dir, "iwae_and_vlb.pkl")
with open(hist_path, "wb") as f:
    pickle.dump({
        "validation_iwae": validation_iwae,
        "train_vlb": train_vlb,
        "alpha_history": alpha_history,
    }, f)
print(f"[INFO] Saved history to {hist_path}")

# -------- plot IWAE/VLB/Alpha
plt.figure(figsize=(8, 4))
plt.plot(validation_iwae, label="Validation IWAE", marker="o")
plt.plot(train_vlb, label="Train VLB", marker="x")
if len(alpha_history) > 0:
    plt.plot(alpha_history, label="KL Alpha", marker="*")
plt.xlabel("Validation checkpoint")
plt.ylabel("Loss / Alpha")
plt.title("Validation IWAE vs Train VLB vs KL Alpha")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plot_path = join(args.model_dir, "iwae_vs_vlb_alpha.png")
plt.savefig(plot_path)
plt.close()
print(f"[INFO] Saved plot to {plot_path}")

# -------- plot rec_error vs KL term
plt.figure(figsize=(8, 4))
plt.plot(rec_errors, label="Reconstruction Error")
plt.plot(kl_terms, label="KL Term")
plt.xlabel("Training batch")
plt.ylabel("Loss component value")
plt.title("Reconstruction Error vs KL Term (per batch, training)")
plt.legend()
plt.tight_layout()
plot_path_components = join(args.model_dir, "rec_vs_kl.png")
plt.savefig(plot_path_components)
plt.close()
print(f"[INFO] Saved plot to {plot_path_components}")
