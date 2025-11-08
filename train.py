import os, random, csv, pickle
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

class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3, ndf=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf*2, 4, 2, 1),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*4, 1, 4, 1, 0),
        )
    def forward(self, x): return self.net(x)

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
        return F.cross_entropy(logits, labels)

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
    try: val = float(tok)
    except Exception: raise ValueError(f"Unrecognized alpha token: {tok}")
    return dict(name=f"alpha_{val}", kl_mode="standard", kl_alpha=val, learnable_alpha=False)

def get_grad_norms(model_params):
    def norm(p_list):
        return float(torch.sqrt(sum([(p.grad**2).sum() for p in p_list if p.grad is not None])).item()) if p_list else 0.0
    model_norm = norm(model_params)
    return {"grad_total": model_norm, "grad_model": model_norm}

class RunningMaxNormalizer:
    def __init__(self, n_losses, eps=1e-8):
        self.max_vals = torch.ones(n_losses, dtype=torch.float32)
        self.eps = eps
    def update_and_normalize(self, losses):
        out = []
        for i, L in enumerate(losses):
            val = L.detach()
            self.max_vals[i] = torch.max(self.max_vals[i].to(val.device), val)
            divisor = self.max_vals[i] + self.eps
            if divisor < 1e-5:
                print(f"Warning: normalization divisor for loss {i} is too small ({divisor.item()})!")
                divisor = torch.tensor(1.0, device=val.device)
            out.append(L / divisor)
        return out

def pc_backward(losses, model, optimizer):
    grads = []
    params = [p for p in model.parameters() if p.requires_grad]
    for loss in losses:
        optimizer.zero_grad(set_to_none=True)
        loss.backward(retain_graph=True)
        grads.append([p.grad.detach().clone() if p.grad is not None else None for p in params])
    optimizer.zero_grad(set_to_none=True)
    logs = {"cosines": []}
    n = len(losses)
    for i in range(n):
        for j in range(i+1, n):
            g1, g2 = grads[i], grads[j]
            dot, norm_sq = 0., 0.
            for a, b in zip(g1, g2):
                if a is not None and b is not None:
                    dot += (a * b).sum().item()
                    norm_sq += (b * b).sum().item()
            if dot < 0 and norm_sq > 0:
                for k in range(len(g1)):
                    if g1[k] is not None and g2[k] is not None:
                        g1[k] -= (dot / (norm_sq + 1e-8)) * g2[k]
            logs["cosines"].append(dot / (norm_sq ** 0.5 + 1e-8) if norm_sq > 0 else 0.)
    for p in params:
        p.grad = None
    for i, p in enumerate(params):
        gsum = None
        for g in grads:
            if g[i] is not None:
                gsum = g[i] if gsum is None else gsum + g[i]
        if gsum is not None:
            p.grad = gsum.clone() / len(grads)
    return logs

def print_model_optimizer_sanity(model, optimizer, header="[MODEL/OPTIMIZER PARAM CHECK]"):
    print("\n" + "="*20 + f" {header} " + "="*20)
    print("-- Model parameters --")
    for n, p in model.named_parameters():
        print(f"  {n}: shape={tuple(p.shape)}, requires_grad={p.requires_grad}, id={id(p)}")
    print("-- Optimizer param groups --")
    for gi, g in enumerate(optimizer.param_groups):
        for p in g['params']:
            found = False
            for mn, mp in model.named_parameters():
                if id(mp) == id(p):
                    found = True
                    print(f"  OPT group {gi}: {mn} (id={id(p)})")
                    break
            if not found:
                print(f"  OPT group {gi}: UNKNOWN PARAM (id={id(p)})")
    print("="*54 + "\n")

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
        model_module.reconstruction_log_prob, model_module.proposal_network, model_module.prior_network, model_module.generative_network,
        debug_asserts=args.debug_asserts, kl_mode=alpha_cfg["kl_mode"], kl_alpha=alpha_cfg["kl_alpha"], learnable_alpha=alpha_cfg["learnable_alpha"],
        alpha_init=args.alpha_init, alpha_max=args.alpha_max, free_bits=args.free_bits,
    ).to(device if torch.cuda.is_available() else "cpu")
    model_params = list(model.parameters())
    param_groups = [{"params": model_params, "lr": 2e-4}]
    optimizer = torch.optim.Adam(param_groups)

    print_model_optimizer_sanity(model, optimizer, header="PARAMS AFTER OPTIMIZER CREATION")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs) if getattr(args, "use_scheduler", False) else None
    sampler = getattr(model_module, "sampler")
    vlb_scale = getattr(model_module, "vlb_scale_factor", 1)
    mask_gen = model_module.mask_generator

    last_ckpt = join(out_root, "last.tar")
    best_ckpt = join(out_root, "best.tar")
    best_val = float('-inf')
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
        start_epoch = ckpt.get("epoch", 0) + 1
        if scheduler is not None and "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        print_model_optimizer_sanity(model, optimizer, header="AFTER CHECKPOINT RELOAD")

    csv_path = join(out_root, "metrics_alphas.csv")
    if not exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["epoch", "step", "split", "train_vlb", "val_iwae", "recon", "kl", "fid", "ssim", "psnr", "alpha",
                 "lpips_loss", "adv_loss", "contrastive_loss", "lr",
                 "pc_cosine_lpips_adv", "pc_cosine_lpips_con", "pc_cosine_adv_con",
                 "grad_total", "grad_model"])

    lpips_loss_fn = lpips.LPIPS(net='vgg').to(device)
    discriminator = PatchDiscriminator(in_channels=3).to(device)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    nt_xent_loss_fn = NTXentLoss(temperature=0.5)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and torch.cuda.is_available()))

    normalizer = RunningMaxNormalizer(3)
    aux_loss_buffers = [[] for _ in range(3)]

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
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None
        }, tmp)
        replace(tmp, last_ckpt)

    for epoch in range(start_epoch, args.epochs):
        iterator = tqdm(dl, desc=f"[{alpha_cfg['name']}] Epoch {epoch+1}/{args.epochs}") if args.verbose else dl
        avg_vlb = 0.0
        last_batch = last_mask = None
        loss_logs = []
        for i, batch in enumerate(iterator):
            # Validation checkpoint
            if any([i == 0 and epoch == start_epoch, i % validation_batches == validation_batches - 1, i + 1 == len(dl)]):
                val_i = get_validation_iwae(val_dl, mask_gen, model_module.batch_size, model, args.validation_iwae_num_samples, verbose=args.verbose)
                validation_iwae.append(val_i)
                train_vlb.append(avg_vlb)
                save_ckpt(epoch)
                if val_i > best_val:
                    best_val = val_i
                    tmp = best_ckpt + ".bak"
                    copy(last_ckpt, tmp)
                    replace(tmp, best_ckpt)
                with open(csv_path, "a", newline="") as f:
                    csv.writer(f).writerow([epoch+1, i, "val", avg_vlb, val_i, "", "", "", "", "", "", "", "", "", "", "", "", optimizer.param_groups[0]["lr"], "", "", "", "", ""])

            batch = extend_batch(batch, dl, model_module.batch_size)
            mask = mask_gen(batch)
            mask = mask.to(batch.device)
            if torch.cuda.is_available():
                batch = batch.cuda(non_blocking=True)
                mask = mask.cuda(non_blocking=True)

            # --- Discriminator update ---
            q, p = model.make_latent_distributions(batch, mask)
            z = q.rsample()
            rec_params = model.generative_network(z)
            rec_img = rec_params[:, :3, :, :].contiguous()
            gt_img = batch[:, :3, :, :].contiguous()
            real_out = discriminator(gt_img)
            fake_out = discriminator(rec_img.detach())
            d_loss = (F.binary_cross_entropy_with_logits(real_out, torch.ones_like(real_out)) +
                      F.binary_cross_entropy_with_logits(fake_out, torch.zeros_like(fake_out))) * 0.5
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            use_amp = args.amp and torch.cuda.is_available()
            if use_amp:
                with torch.cuda.amp.autocast():
                    vlb = model.batch_vlb(batch, mask).mean()
                    elbo_loss = -vlb / vlb_scale
                    rec_img_lpips = (rec_img * 2 - 1).clamp(-1, 1)
                    gt_img_lpips = (gt_img * 2 - 1).clamp(-1, 1)
                    mask_rgb = mask[:, :3, :, :]
                    lpips_loss = lpips_loss_fn(rec_img_lpips * mask_rgb, gt_img_lpips * mask_rgb).mean()
                    adv_logits = discriminator(rec_img)
                    adv_loss = F.binary_cross_entropy_with_logits(adv_logits, torch.ones_like(adv_logits))
                    z1 = z.view(batch.size(0), -1)
                    z2 = q.rsample().view(batch.size(0), -1)
                    contrastive_loss = nt_xent_loss_fn(z1, z2)
            else:
                vlb = model.batch_vlb(batch, mask).mean()
                elbo_loss = -vlb / vlb_scale
                rec_img_lpips = (rec_img * 2 - 1).clamp(-1, 1)
                gt_img_lpips = (gt_img * 2 - 1).clamp(-1, 1)
                mask_rgb = mask[:, :3, :, :]
                lpips_loss = lpips_loss_fn(rec_img_lpips * mask_rgb, gt_img_lpips * mask_rgb).mean()
                adv_logits = discriminator(rec_img)
                adv_loss = F.binary_cross_entropy_with_logits(adv_logits, torch.ones_like(adv_logits))
                z1 = z.view(batch.size(0), -1)
                z2 = q.rsample().view(batch.size(0), -1)
                contrastive_loss = nt_xent_loss_fn(z1, z2)

            # --- Normalization logic ---
            aux_losses = [lpips_loss, adv_loss, contrastive_loss]
            normalized = normalizer.update_and_normalize(aux_losses)

            # ------ Main PCGrad Step ------
            losses_for_pcgrad = [elbo_loss] + normalized
            pcgrad_logs = pc_backward(losses_for_pcgrad, model, optimizer)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            grad_norms = get_grad_norms(model_params)
            pc_cosines = pcgrad_logs["cosines"] if (pcgrad_logs is not None and "cosines" in pcgrad_logs) else [None, None, None]

            with torch.no_grad():
                rec = float(model.rec_log_prob(batch, rec_params, mask).mean().item())
                kl = float(torch.distributions.kl_divergence(q, p).view(batch.shape[0], -1).sum(-1).mean().item())
                alpha_val = float(model._alpha_value().item())
                logdict = {
                    "elbo_loss": float(elbo_loss.item()),
                    "lpips_loss": float(lpips_loss.item()),
                    "adv_loss": float(adv_loss.item()),
                    "contrastive_loss": float(contrastive_loss.item()),
                    "total_loss": float(elbo_loss.item() + normalized[0].item() + normalized[1].item() + normalized[2].item()),
                    "rec": rec, "kl": kl, "alpha": alpha_val,
                    "pc_cosine_lpips_adv": float(pc_cosines[1]) if len(pc_cosines) > 1 else None,
                    "pc_cosine_lpips_con": float(pc_cosines[2]) if len(pc_cosines) > 2 else None,
                    "pc_cosine_adv_con": float(pc_cosines[5]) if len(pc_cosines) > 5 else None,
                    "grad_total": grad_norms["grad_total"],
                    "grad_model": grad_norms["grad_model"]
                }
                loss_logs.append(logdict)
                rec_errors.append(rec)
                kl_terms.append(kl)
                alpha_log.append(alpha_val)
                avg_vlb += (float(vlb) - avg_vlb) / (i + 1)
            if args.verbose and isinstance(iterator, tqdm):
                iterator.set_postfix(vlb=f"{avg_vlb:.1f}", alpha=alpha_val, grad_total=f"{grad_norms['grad_total']:.3g}")

            last_batch, last_mask = batch, mask

        epoch_loss_logs.extend(loss_logs)
        with open(join(out_root, f"loss_logs_epoch_{epoch+1}.pkl"), "wb") as f:
            pickle.dump(loss_logs, f)

        if scheduler is not None:
            scheduler.step()

        if last_batch is not None and last_mask is not None:
            with torch.no_grad():
                q, p = model.make_latent_distributions(last_batch, last_mask)
                z = q.rsample()
                rec_params = model.generative_network(z)
                bar = "-" * 72
                print(f"+{bar}+")
                print(f"| Alpha: {alpha_cfg['name']:<20} | Epoch: {epoch+1}/{args.epochs:<4} | Avg VLB: {avg_vlb:>10.3f} |")
                print(f"+{bar}+")
                print(f"| Batch : {tuple(last_batch.shape)}")
                print(f"| Mask : {tuple(last_mask.shape)}")
                print(f"| q.mean : {tuple(q.mean.shape)}")
                print(f"| p.mean : {tuple(p.mean.shape)}")
                print(f"| z : {tuple(z.shape)}")
                print(f"| rec_params : {tuple(rec_params.shape)}")
                print(f"+{bar}+")
                print(f"| grad_total: {grad_norms['grad_total']:.4f} | grad_model: {grad_norms['grad_model']:.4f} |")
                print(f"+{bar}+")

            fid_score = ssim = psnr = None
            if args.compute_fid or args.compute_ssimpsnr:
                val_it = iter(val_dl)
                try:
                    vb = next(val_it)
                except StopIteration:
                    val_it = iter(val_dl)
                    vb = next(val_it)
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
                csv.writer(f).writerow([
                    epoch+1, "epoch_end", "train", avg_vlb, "", "", "", fid_score, ssim, psnr, alpha_val,
                    float(lpips_loss.item()), float(adv_loss.item()), float(contrastive_loss.item()), optimizer.param_groups[0]["lr"],
                    float(pc_cosines[1]) if len(pc_cosines) > 1 else None,
                    float(pc_cosines[2]) if len(pc_cosines) > 2 else None,
                    float(pc_cosines[5]) if len(pc_cosines) > 5 else None,
                    grad_norms["grad_total"], grad_norms["grad_model"]
                ])

            with torch.no_grad():
                lat_means = model.latent_means(last_batch, last_mask).cpu().numpy()
                np.savez_compressed(join(out_root, f"latents_epoch_{epoch+1}_alphas.npz"), latents=lat_means)
            if args.tsne_every and ((epoch + 1) % args.tsne_every == 0):
                tsne_latents(lat_means, labels=None, out_path=join(out_root, f"tsne_epoch_{epoch+1}_alphas.png"))
        else:
            print(f"WARNING: No batch was seen this epoch (epoch {epoch+1}). Skipping latent logging and final visualizations.")

    with open(join(out_root, "all_loss_logs.pkl"), "wb") as f:
        pickle.dump(epoch_loss_logs, f)
    print(f"[INFO] ({alpha_cfg['name']}) training complete – saving history & plots…")
    with open(join(out_root, "history.pkl"), "wb") as f:
        pickle.dump({"validation_iwae": validation_iwae, "train_vlb": train_vlb, "rec_errors": rec_errors, "kl_terms": kl_terms, "alpha_log": alpha_log}, f)
    plt.figure(figsize=(14, 6))
    plt.plot(validation_iwae, label="Validation IWAE", marker="o")
    plt.plot(train_vlb, label="Train VLB", marker="x")
    plt.xlabel("Validation checkpoint"); plt.ylabel("Loss / Value")
    plt.title(f"[{alpha_cfg['name']}] Validation IWAE and Train VLB")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(); plt.tight_layout()
    plt.savefig(join(out_root, "loss_curves.png"), dpi=200)
    plt.close()
    plt.figure(figsize=(14, 6))
    plt.plot(rec_errors, label="Reconstruction Error", linestyle="--")
    plt.plot(kl_terms, label="KL Divergence", linestyle="-.")
    plt.xlabel("Batch"); plt.ylabel("Loss / Value")
    plt.title(f"[{alpha_cfg['name']}] Reconstruction Error and KL (per batch)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(); plt.tight_layout()
    plt.savefig(join(out_root, "recon_kl_per_batch.png"), dpi=200)
    plt.close()

if __name__ == "__main__":
    p = ArgumentParser("Train ORIGINAL VAEAC with multiple KL weights/variants.")
    p.add_argument("--model_dir", type=str, required=True)
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--train_dataset", type=str, required=True)
    p.add_argument("--validation_dataset", type=str, required=True)
    p.add_argument("--alphas", type=str, default="0,0.5,1,inf,-inf,learnable,symmetric",
        help='Comma-separated: e.g. "0,0.5,1,inf,-inf,learnable,symmetric,symmetric:0.5"')
    p.add_argument("--max_train_images", type=int, default=25000)
    p.add_argument("--validation_iwae_num_samples", type=int, default=25)
    p.add_argument("--validations_per_epoch", type=int, default=5)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--verbose", action="store_true", default=True)
    p.add_argument("--compute_fid", action="store_true", default=False)
    p.add_argument("--compute_ssimpsnr", action="store_true", default=False)
    p.add_argument("--tsne_every", type=int, default=0)
    p.add_argument("--amp", action="store_true", default=False)
    p.add_argument("--debug_asserts", action="store_true", default=False)
    p.add_argument("--alpha_init", type=float, default=1.0)
    p.add_argument("--alpha_max", type=float, default=1e6)
    p.add_argument("--free_bits", type=float, default=0.0)
    p.add_argument("--use_scheduler", action="store_true", default=False)
    p.add_argument("--normalize_epochs", type=int, default=5)
    args = p.parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokens = [t for t in args.alphas.split(",") if t.strip()]
    cfgs = [parse_alpha_token(t) for t in tokens]
    print("[INFO] KL variants to run:", ", ".join([c["name"] for c in cfgs]))
    for cfg in cfgs:
        run_one_alpha(args, cfg, device=device)








