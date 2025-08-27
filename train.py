# [MODIFIED FOR SWAPPED_NETS EXPERIMENT + metrics/exports, Aug 2025]
from argparse import ArgumentParser
from importlib import import_module
from math import ceil
from os import replace
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
from train_utils import extend_batch, get_validation_iwae
from VAEAC import VAEAC
from metrics import compute_fid, compute_ssim_psnr
from viz import tsne_latents


from train_utils import make_mask_on_batch_device




def set_seed(seed: int = 1337):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




parser = ArgumentParser(description='Train VAEAC to inpaint (swapped-nets exp).')


# --- required (keep your interface) ---
parser.add_argument('--model_dir', type=str, required=True)
parser.add_argument('--epochs', type=int, required=True)
parser.add_argument('--train_dataset', type=str, required=True)
parser.add_argument('--validation_dataset', type=str, required=True)


# --- unchanged defaults ---
parser.add_argument('--validation_iwae_num_samples', type=int, default=25)
parser.add_argument('--validations_per_epoch', type=int, default=5)
parser.add_argument('--max_train_images', type=int, default=3500)


# --- NEW FLAGS (all optional) ---
parser.add_argument('--seed', type=int, default=1337)
parser.add_argument('--debug_asserts', action='store_true', default=False, help='extra anti-leakage assertions')
parser.add_argument('--compute_fid', action='store_true', default=False, help='per-epoch FID on validation subset')
parser.add_argument('--compute_ssimpsnr', action='store_true', default=False, help='per-epoch SSIM/PSNR')
parser.add_argument('--amp', action='store_true', default=False, help='use torch.cuda.amp during training')
parser.add_argument('--tsne_every', type=int, default=0, help='if >0, run t-SNE every N epochs on val subset')


args = parser.parse_args()
set_seed(args.seed)


use_cuda = torch.cuda.is_available()
verbose = True
num_workers = 4


# import the module with the model networks definitions
model_module = import_module(args.model_dir + '.model')


# build VAEAC
model = VAEAC(
    model_module.reconstruction_log_prob,
    model_module.proposal_network,
    model_module.prior_network,
    model_module.generative_network,
    debug_asserts=args.debug_asserts,
)
if use_cuda:
    model = model.cuda()


optimizer = model_module.optimizer(model.parameters())
batch_size = model_module.batch_size
vlb_scale_factor = getattr(model_module, 'vlb_scale_factor', 1)
mask_generator = model_module.mask_generator
sampler = getattr(model_module, 'sampler')


# datasets/dataloaders (+subset limit)
train_dataset = load_dataset(args.train_dataset)
train_dataset = LengthBounder(train_dataset, args.max_train_images)
validation_dataset = load_dataset(args.validation_dataset)


dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_workers)
val_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_workers)


validation_batches = ceil(len(dataloader) / args.validations_per_epoch)


# history buffers
validation_iwae, train_vlb = [], []
rec_errors, kl_terms = [], []


# resume
ckpt_last = join(args.model_dir, 'checkpoint_swapped_nets_last_AUG.tar')
if exists(ckpt_last):
    location = 'cuda' if use_cuda else 'cpu'
    checkpoint = torch.load(ckpt_last, map_location=location)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    validation_iwae = checkpoint.get('validation_iwae', [])
    train_vlb = checkpoint.get('train_vlb', [])
    rec_errors = checkpoint.get('rec_errors', [])
    kl_terms = checkpoint.get('kl_terms', [])


# CSV metrics file
metrics_csv = join(args.model_dir, 'metrics_AUG.csv')
if not exists(metrics_csv):
    with open(metrics_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'step', 'split', 'train_vlb', 'val_iwae', 'recon', 'kl', 'fid', 'ssim', 'psnr', 'lr'])


scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and use_cuda))




def make_checkpoint(epoch: int):
    filename = join(args.model_dir, 'checkpoint_swapped_nets_last_AUG.tar')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'validation_iwae': validation_iwae,
        'train_vlb': train_vlb,
        'rec_errors': rec_errors,
        'kl_terms': kl_terms,
    }, filename + '.bak')
    replace(filename + '.bak', filename)




# ------------------------------- TRAIN LOOP -------------------------------- #
for epoch in range(args.epochs):
    iterator = dataloader
    avg_vlb = 0.0
    if verbose:
        print('Epoch %d...' % (epoch + 1), file=stderr, flush=True)
        iterator = tqdm(iterator)


    last_batch = None
    last_mask = None


    for i, batch in enumerate(iterator):
        # validations
        if any([
            i == 0 and epoch == 0,
            i % validation_batches == validation_batches - 1,
            i + 1 == len(dataloader)
        ]):
            val_iwae = get_validation_iwae(
                val_dataloader, mask_generator, batch_size, model,
                args.validation_iwae_num_samples, verbose
            )
            validation_iwae.append(val_iwae)
            train_vlb.append(avg_vlb)
            make_checkpoint(epoch)


            # best copy
            if max(validation_iwae[::-1]) <= val_iwae:
                src_filename = join(args.model_dir, 'checkpoint_swapped_nets_last_AUG.tar')
                dst_filename = join(args.model_dir, 'checkpoint_swapped_nets_best_AUG.tar')
                copy(src_filename, dst_filename + '.bak')
                replace(dst_filename + '.bak', dst_filename)


            # write CSV row (validation)
            with open(metrics_csv, 'a', newline='') as f:
                csv.writer(f).writerow([epoch+1, i, 'val', avg_vlb, val_iwae, '', '', '', '', '', optimizer.param_groups[0]['lr']])


            if verbose:
                print(file=stderr); print(file=stderr)


        batch = extend_batch(batch, dataloader, batch_size)
        mask = mask_generator(batch)


        if use_cuda:
            batch = batch.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)


        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(args.amp and use_cuda)):
            vlb = model.batch_vlb(batch, mask).mean()
            loss = -vlb / vlb_scale_factor
        scaler.scale(loss).backward()
        scaler.step(optimizer); scaler.update()


        # per-batch logging (no grad)
        with torch.no_grad():
            q, p = model.make_latent_distributions(batch, mask)
            z = q.rsample()
            rec_params = model.generative_network(z)
            rec_err = float(model.rec_log_prob(batch, rec_params, mask).mean().item())
            kl = float(torch.distributions.kl_divergence(q, p).view(batch.shape[0], -1).sum(-1).mean().item())
            rec_errors.append(rec_err); kl_terms.append(kl)


        avg_vlb += (float(vlb) - avg_vlb) / (i + 1)
        if verbose:
            iterator.set_description('Train VLB: %g' % avg_vlb)


        last_batch = batch
        last_mask = mask


    # -------- end epoch pretty print and extra exports --------
    with torch.no_grad():
        q, p = model.make_latent_distributions(last_batch, last_mask)
        z = q.rsample()
        rec_params = model.generative_network(z)


    bar = "-" * 70
    print(f"+{bar}+")
    print(f"| Epoch: {epoch + 1}/{args.epochs:<6} | Average Loss: {avg_vlb:.3f} |")
    print(f"+{bar}+")
    print(f"| Batch shape                 : {tuple(last_batch.shape)} |")
    print(f"| Mask shape                  : {tuple(last_mask.shape)} |")
    print(f"| Proposal mean shape         : {tuple(q.mean.shape)} |")
    print(f"| Prior mean shape            : {tuple(p.mean.shape)} |")
    print(f"| Latent sample shape         : {tuple(z.shape)} |")
    print(f"| Reconstruction params shape : {tuple(rec_params.shape)} |")
    print(f"+{bar}+")
    # ---------------- extra metrics on a small val subset ----------------
    fid_score = None; ssim = None; psnr = None
    if args.compute_fid or args.compute_ssimpsnr:
        # build a small val batch
        val_iter = iter(val_dataloader)
        try:
            val_batch = next(val_iter)
        except StopIteration:
            val_iter = iter(val_dataloader); val_batch = next(val_iter)
        val_batch = val_batch.cuda() if use_cuda else val_batch
        # new (always generate mask on CPU, then move to the batch device)
        val_mask = make_mask_on_batch_device(mask_generator, val_batch)


        with torch.no_grad():
            params = model.generate_samples_params(val_batch, val_mask, K=1)
            inp = sampler(params[:, 0])
        if args.compute_fid:
            fid_score = compute_fid(inp, val_batch, device='cuda' if use_cuda else 'cpu')
        if args.compute_ssimpsnr:
            ssim, psnr = compute_ssim_psnr(inp, val_batch)
        # CSV row (train epoch summary)
    with open(metrics_csv, 'a', newline='') as f:
        csv.writer(f).writerow([epoch+1, 'epoch_end', 'train', avg_vlb, '', '', '', fid_score, ssim, psnr, optimizer.param_groups[0]['lr']])


    # save latent means of the last training epoch batch (for later KL/analysis)
    with torch.no_grad():
        lat_means = model.latent_means(last_batch, last_mask).cpu().numpy()
    os.makedirs(args.model_dir, exist_ok=True)
    npz_path = join(args.model_dir, f'latents_after_swap_epoch_{epoch+1}_AUG.npz')
    np.savez_compressed(npz_path, latents=lat_means)
    # optional t-SNE
    if args.tsne_every and ((epoch + 1) % args.tsne_every == 0):
        tsne_latents(lat_means, labels=None,
                     out_path=join(args.model_dir, f"tsne_latent_celeba_swapped_epoch{epoch+1}_AUG.png"))


# ----------------------------- After training ------------------------------ #
print("[INFO] Training complete – saving history & plots…")


hist_path = join(args.model_dir, "iwae_and_vlb_AUG.pkl")
with open(hist_path, "wb") as f:
    pickle.dump({
        "validation_iwae": validation_iwae,
        "train_vlb": train_vlb,
        "rec_errors": rec_errors,
        "kl_terms": kl_terms,
    }, f)
print(f"[INFO] Saved history to {hist_path}")


# (a) Train VLB vs Val IWAE
plt.figure(figsize=(14,6))
plt.plot(validation_iwae, label="Validation IWAE", marker="o")
plt.plot(train_vlb, label="Train VLB", marker="x")
plt.xlabel("Validation checkpoint"); plt.ylabel("Loss / Value")
plt.title("Validation IWAE and Train VLB")
plt.grid(True, linestyle="--", alpha=0.5); plt.legend()
plt.tight_layout(); plt.savefig(join(args.model_dir, "loss_curves_AUG.png"), dpi=200); plt.close()


# (b) Reconstruction & KL per batch
plt.figure(figsize=(14,6))
plt.plot(rec_errors, label="Reconstruction Error", linestyle="--")
plt.plot(kl_terms, label="KL Divergence", linestyle="-.")
plt.xlabel("Batch"); plt.ylabel("Loss / Value")
plt.title("Reconstruction Error and KL Divergence (per batch)")
plt.grid(True, linestyle="--", alpha=0.5); plt.legend()
plt.tight_layout(); plt.savefig(join(args.model_dir, "loss_curves_for_thesis_AUG.png"), dpi=200); plt.close()


# (c) Combined chart (quick)
plt.figure(figsize=(12,5))
plt.plot(validation_iwae, label="Validation IWAE", marker="o")
plt.plot(train_vlb, label="Train VLB", marker="x")
plt.plot(np.linspace(0, len(validation_iwae)-1, num=len(rec_errors)), rec_errors, alpha=0.5, label="Recon (per batch)")
plt.plot(np.linspace(0, len(validation_iwae)-1, num=len(kl_terms)), kl_terms, alpha=0.5, label="KL (per batch)")
plt.xlabel("Checkpoint"); plt.ylabel("Value")
plt.title("All metrics")
plt.grid(True, alpha=0.3); plt.legend()
plt.tight_layout(); plt.savefig(join(args.model_dir, "loss_curves_AUG_combined.png"), dpi=200); plt.close()


print("[INFO] Saved plots & metrics CSV.")
