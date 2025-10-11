import os
import random
import csv
import pickle
from math import ceil
from os.path import exists, join
from shutil import copy
from os import replace
from argparse import ArgumentParser
from importlib import import_module

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

import lpips  # pip install lpips

# ---------------- GradNorm -----------------
class GradNormLoss(nn.Module):
    def __init__(self, n_tasks, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.log_weights = nn.Parameter(torch.zeros(n_tasks, dtype=torch.float32))
        self.initial_task_losses = None

    def get_weights(self):
        return F.softplus(self.log_weights)

    def forward(self, losses, shared_params, step, reset_l0=False):
        n_tasks = len(losses)
        weights = self.get_weights()
        normed_grads = []
        for i, loss in enumerate(losses):
            grads = torch.autograd.grad(loss, shared_params, retain_graph=True, allow_unused=True)
            grads = [g for g in grads if g is not None]
            if len(grads) == 0:
                normed_grads.append(torch.tensor(0.0, device=weights.device))
                continue
            grad_norm = torch.cat([g.detach().view(-1) for g in grads]).norm()
            normed_grads.append(grad_norm)
        normed_grads = torch.stack(normed_grads)

        if (self.initial_task_losses is None) or reset_l0:
            self.initial_task_losses = torch.tensor([l.item() for l in losses], device=weights.device)

        L_ratios = torch.tensor([l.item() / l0 for l, l0 in zip(losses, self.initial_task_losses)], device=weights.device)
        avg_ratio = L_ratios.mean()
        target_norms = normed_grads.mean() * (L_ratios / avg_ratio) ** self.alpha

        loss_gradnorm = F.l1_loss(normed_grads, target_norms.detach(), reduction='sum')
        return loss_gradnorm

def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3, ndf=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, ndf, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf*2, 4, 2, 1), nn.BatchNorm2d(ndf*2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1), nn.BatchNorm2d(ndf*4), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*4, 1, 4, 1, 0),
        )
    def forward(self, x):
        return self.net(x)

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
    def forward(self, z_i, z_j):
        z_i = F.normalize(z_i, dim=-1)
        z_j = F.normalize(z_j, dim=-1)
        batch_size = z_i.size(0)
        representations = torch.cat([z_i, z_j], dim=0)
        sim_matrix = torch.mm(representations, representations.t()) / self.temperature
        mask = torch.eye(2*batch_size, dtype=torch.bool, device=sim_matrix.device)
        sim_matrix = sim_matrix.masked_fill(mask, -1e4)
        positives = torch.cat([torch.diag(sim_matrix, batch_size), torch.diag(sim_matrix, -batch_size)])
        negatives = sim_matrix[~mask].view(2*batch_size, -1)
        logits = torch.cat([positives.unsqueeze(1), negatives], dim=1)
        labels = torch.zeros(2*batch_size, dtype=torch.long, device=z_i.device)
        loss = F.cross_entropy(logits, labels)
        return loss

from datasets import load_dataset, LengthBounder
from train_utils import extend_batch, get_validation_iwae, make_mask_on_batch_device
from VAEAC import VAEAC
from metrics import compute_fid, compute_ssim_psnr
from viz import tsne_latents

def parse_alpha_token(tok: str):
    tok = tok.strip().lower()
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
        return dict(name="alpha_ninf", kl_mode="standard", kl_alpha=0.0, learnable_alpha=False)
    try:
        val = float(tok)
    except Exception:
        raise ValueError(f"Unrecognized alpha token: {tok}")
    return dict(name=f"alpha_{val}", kl_mode="standard", kl_alpha=val, learnable_alpha=False)

def run_one_alpha(args, alpha_cfg, device="cuda"):
    model_module = import_module(args.model_dir + '.model')
    out_root = join(args.model_dir, f"alpha_runs/{alpha_cfg['name']}")
    os.makedirs(out_root, exist_ok=True)

    train_dataset = load_dataset(args.train_dataset)
    if args.max_train_images > 0:
        train_dataset = LengthBounder(train_dataset, args.max_train_images)
    val_dataset = load_dataset(args.validation_dataset)

    dl = DataLoader(train_dataset, batch_size=model_module.batch_size, shuffle=True, drop_last=False, num_workers=args.num_workers)
    val_dl = DataLoader(val_dataset, batch_size=model_module.batch_size, shuffle=True, drop_last=False, num_workers=args.num_workers)
    validation_batches = ceil(len(dl) / args.validations_per_epoch)

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

    gradnorm = GradNormLoss(n_tasks=3, alpha=0.5).to(device)
    params = list(model.parameters()) + list(gradnorm.parameters())
    optimizer = model_module.optimizer(params)
    sampler = getattr(model_module, "sampler")
    vlb_scale = getattr(model_module, "vlb_scale_factor", 1)
    mask_gen = model_module.mask_generator

    last_ckpt = join(out_root, "last.tar")
    validation_iwae, train_vlb, rec_errors, kl_terms, alpha_log = [], [], [], [], []
    epoch_loss_logs = []
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
        if "gradnorm_state_dict" in ckpt:
            gradnorm.load_state_dict(ckpt["gradnorm_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1

    csv_path = join(out_root, "metrics_alphas.csv")
    if not exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["epoch", "step", "split", "train_vlb", "val_iwae", "recon", "kl", "fid", "ssim", "psnr", "alpha",
                 "gradnorm_weight_lpips", "gradnorm_weight_contrastive", "gradnorm_weight_adv", "lr"]
            )

    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and torch.cuda.is_available()))
    lpips_loss_fn = lpips.LPIPS(net='vgg').to(device)
    discriminator = PatchDiscriminator(in_channels=3).to(device)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    nt_xent_loss_fn = NTXentLoss(temperature=0.5)

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
            "gradnorm_state_dict": gradnorm.state_dict(),
        }, tmp)
        replace(tmp, last_ckpt)

    def get_phase(epoch):
        if epoch < args.phase1_epochs:
            return 1
        elif epoch < args.phase1_epochs + args.phase2_epochs:
            return 2
        else:
            return 3

    for epoch in range(start_epoch, args.epochs):
        phase = get_phase(epoch)
        iterator = tqdm(dl, desc=f"[{alpha_cfg['name']}] Epoch {epoch+1}/{args.epochs} (Phase {phase})") if args.verbose else dl
        avg_vlb = 0.0
        last_batch = last_mask = None
        last_q = last_p = last_z = last_rec_params = None
        loss_logs = []
        for i, batch in enumerate(iterator):
            if any([i == 0 and epoch == start_epoch, i % validation_batches == validation_batches - 1, i + 1 == len(dl)]):
                val_i = get_validation_iwae(val_dl, mask_gen, model_module.batch_size, model,
                                            args.validation_iwae_num_samples, verbose=args.verbose)
                validation_iwae.append(val_i)
                train_vlb.append(avg_vlb)
                save_ckpt(epoch)
                best_path = join(out_root, "best.tar")
                if max(validation_iwae[::-1]) <= val_i:
                    tmp = best_path + ".bak"
                    copy(last_ckpt, tmp)
                    replace(tmp, best_path)
                with open(csv_path, "a", newline="") as f:
                    weights = gradnorm.get_weights().detach().cpu().numpy()
                    csv.writer(f).writerow([epoch+1, i, "val", avg_vlb, val_i, "", "", "", "", "", "",
                        weights[0], weights[1], weights[2], optimizer.param_groups[0]["lr"]])

            batch = extend_batch(batch, dl, model_module.batch_size)
            mask = mask_gen(batch)
            batch = batch.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(args.amp and torch.cuda.is_available())):
                vlb = model.batch_vlb(batch, mask).mean()
                elbo_loss = -vlb / vlb_scale

                q, p = model.make_latent_distributions(batch, mask)
                z = q.rsample()
                rec_params = model.generative_network(z)
                rec_img = rec_params[:, :3, :, :].contiguous()
                gt_img = batch[:, :3, :, :].contiguous()
                rec_img_lpips = (rec_img * 2 - 1).clamp(-1, 1)
                gt_img_lpips = (gt_img * 2 - 1).clamp(-1, 1)
                lpips_loss = lpips_loss_fn(rec_img_lpips, gt_img_lpips).mean()

                aux_losses, aux_names = [lpips_loss], ["lpips"]
                contrastive_loss = torch.tensor(0.0, device=batch.device)
                adv_loss = torch.tensor(0.0, device=batch.device)

                if phase >= 2:
                    mask2 = mask_gen(batch)
                    mask2 = mask2.to(device, non_blocking=True)
                    q2, _ = model.make_latent_distributions(batch, mask2)
                    z1 = z.view(batch.size(0), -1)
                    z2 = q2.rsample().view(batch.size(0), -1)
                    contrastive_loss = nt_xent_loss_fn(z1, z2)
                    aux_losses.append(contrastive_loss)
                    aux_names.append("contrastive")
                if phase == 3:
                    real_out = discriminator(gt_img)
                    fake_out = discriminator(rec_img.detach())
                    d_loss = (F.binary_cross_entropy_with_logits(real_out, torch.ones_like(real_out)) +
                              F.binary_cross_entropy_with_logits(fake_out, torch.zeros_like(fake_out))) * 0.5
                    d_optimizer.zero_grad()
                    d_loss.backward()
                    d_optimizer.step()
                    adv_logits = discriminator(rec_img)
                    adv_loss = F.binary_cross_entropy_with_logits(adv_logits, torch.ones_like(adv_logits))
                    aux_losses.append(adv_loss)
                    aux_names.append("adversarial")

                shared_params = [p for n, p in model.named_parameters() if "generative_network" in n]
                gradnorm_loss = gradnorm(aux_losses, shared_params, i, reset_l0=(i==0))
                total_aux = sum(w * l for w, l in zip(gradnorm.get_weights()[:len(aux_losses)], aux_losses))
                total_loss = elbo_loss + total_aux + gradnorm_loss

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                rec = float(model.rec_log_prob(batch, rec_params, mask).mean().item())
                kl = float(torch.distributions.kl_divergence(q, p).view(batch.shape[0], -1).sum(-1).mean().item())
                alpha_val = float(model._alpha_value().item())
                weights = gradnorm.get_weights().detach().cpu().numpy()
                logdict = {
                    "elbo_loss": float(elbo_loss.item()),
                    "lpips_loss": float(lpips_loss.item()),
                    "contrastive_loss": float(contrastive_loss.item() if phase >= 2 else 0.),
                    "adv_loss": float(adv_loss.item() if phase == 3 else 0.),
                    "total_loss": float(total_loss.item()),
                    "rec": rec,
                    "kl": kl,
                    "alpha": alpha_val,
                    "gradnorm_weight_lpips": weights[0],
                    "gradnorm_weight_contrastive": weights[1] if phase >= 2 else 0.,
                    "gradnorm_weight_adv": weights[2] if phase == 3 else 0.
                }
                loss_logs.append(logdict)
                rec_errors.append(rec)
                kl_terms.append(kl)
                alpha_log.append(alpha_val)
                # Save last batch and outputs for summary printing
                last_batch, last_mask, last_q, last_p, last_z, last_rec_params = batch, mask, q, p, z, rec_params
            avg_vlb += (float(vlb) - avg_vlb) / (i + 1)
            if args.verbose and isinstance(iterator, tqdm):
                iterator.set_postfix(vlb=f"{avg_vlb:.1f}", alpha=alpha_val)

        # Save batch losses for this epoch
        epoch_loss_logs.extend(loss_logs)
        with open(join(out_root, f"loss_logs_epoch_{epoch+1}.pkl"), "wb") as f:
            pickle.dump(loss_logs, f)

        # Plot GradNorm weights at epoch end
        weights_arr = np.stack([
            [log.get("gradnorm_weight_lpips", 0) for log in loss_logs],
            [log.get("gradnorm_weight_contrastive", 0) for log in loss_logs],
            [log.get("gradnorm_weight_adv", 0) for log in loss_logs]
        ], axis=1)
        plt.figure(figsize=(10,5))
        plt.plot(weights_arr[:,0], label="GradNorm LPIPS")
        plt.plot(weights_arr[:,1], label="GradNorm Contrastive")
        plt.plot(weights_arr[:,2], label="GradNorm Adversarial")
        plt.legend(), plt.grid(True, alpha=0.3)
        plt.title(f"GradNorm Weights (epoch {epoch+1})")
        plt.tight_layout()
        plt.savefig(join(out_root, f"gradnorm_weights_epoch_{epoch+1}.png"))
        plt.close()

        # ---- Epoch summary block: Shapes and weights ----
        bar = "-" * 72
        print(f"+{bar}+")
        print(f"| Alpha: {alpha_cfg['name']:<20} | Epoch: {epoch+1}/{args.epochs:<4} | Avg VLB: {avg_vlb:>10.3f} |")
        print(f"+{bar}+")
        print(f"| Batch           : {tuple(last_batch.shape) if last_batch is not None else 'N/A'}")
        print(f"| Mask            : {tuple(last_mask.shape) if last_mask is not None else 'N/A'}")
        print(f"| q.mean          : {tuple(getattr(last_q, 'mean', torch.empty(0)).shape) if last_q is not None else 'N/A'}")
        print(f"| p.mean          : {tuple(getattr(last_p, 'mean', torch.empty(0)).shape) if last_p is not None else 'N/A'}")
        print(f"| z               : {tuple(getattr(last_z, 'shape', torch.empty(0).shape)) if last_z is not None else 'N/A'}")
        print(f"| rec_params      : {tuple(last_rec_params.shape) if last_rec_params is not None else 'N/A'}")
        gradnorm_weights = gradnorm.get_weights().detach().cpu().numpy()
        print(f"| gradnorm_weight_lpips       : {gradnorm_weights[0]:>10.4f}")
        print(f"| gradnorm_weight_contrastive : {gradnorm_weights[1]:>10.4f}")
        print(f"| gradnorm_weight_adv         : {gradnorm_weights[2]:>10.4f}")
        print(f"+{bar}+")

        # (optional) Plot or print other diagnostics

    # Save all batch losses for all epochs
    with open(join(out_root, "all_loss_logs.pkl"), "wb") as f:
        pickle.dump(epoch_loss_logs, f)
    print(f"[INFO] ({alpha_cfg['name']}) training complete – saving history & plots…")
    with open(join(out_root, "history.pkl"), "wb") as f:
        pickle.dump(
            {"validation_iwae": validation_iwae, "train_vlb": train_vlb, "rec_errors": rec_errors, "kl_terms": kl_terms, "alpha_log": alpha_log},
            f,
        )

if __name__ == "__main__":
    p = ArgumentParser("Train VAEAC + GradNorm + Staged Strategy")
    p.add_argument("--model_dir", type=str, required=True)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--train_dataset", type=str, required=True)
    p.add_argument("--validation_dataset", type=str, required=True)
    p.add_argument("--alphas", type=str, default="1.0")
    p.add_argument("--max_train_images", type=int, default=45000)
    p.add_argument("--validation_iwae_num_samples", type=int, default=25)
    p.add_argument("--validations_per_epoch", type=int, default=5)
    p.add_argument("--phase1_epochs", type=int, default=10)
    p.add_argument("--phase2_epochs", type=int, default=8)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--verbose", action="store_true", default=True)
    p.add_argument("--compute_fid", action="store_true", default=False)
    p.add_argument("--compute_ssimpsnr", action="store_true", default=False)
    p.add_argument("--tsne_every", type=int, default=0)
    p.add_argument("--amp", action="store_true", default=False)
    p.add_argument("--debug_asserts", action="store_true", default=False)
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





