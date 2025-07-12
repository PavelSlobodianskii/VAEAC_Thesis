from argparse import ArgumentParser
from importlib import import_module
from math import ceil
from os import replace
from os.path import exists, join
from shutil import copy
from sys import stderr

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import load_dataset
from train_utils import extend_batch, get_validation_iwae
from VAEAC import VAEAC

import numpy as np
import matplotlib.pyplot as plt
import os
import time

# --------- Metrics: FID/SSIM ---------
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.functional.image.ssim import structural_similarity_index_measure
from prob_utils import normal_parse_params

def compute_fid_ssim(real_images, generated_images, device="cuda"):
    real_images = (real_images + 1) / 2
    generated_images = (generated_images + 1) / 2
    fid = FrechetInceptionDistance(normalize=True).to(device)
    fid.update(real_images, real=True)
    fid.update(generated_images, real=False)
    fid_value = fid.compute().item()
    ssim = 0.0
    for x, y in zip(real_images, generated_images):
        ssim += structural_similarity_index_measure(x.unsqueeze(0), y.unsqueeze(0)).item()
    ssim_value = ssim / real_images.size(0)
    return fid_value, ssim_value

parser = ArgumentParser(description='Train VAEAC to inpaint.')
parser.add_argument('--model_dir', type=str, action='store', required=True)
parser.add_argument('--epochs', type=int, action='store', required=True)
parser.add_argument('--train_dataset', type=str, action='store', required=True)
parser.add_argument('--validation_dataset', type=str, action='store', required=True)
parser.add_argument('--validation_iwae_num_samples', type=int, action='store', default=25)
parser.add_argument('--validations_per_epoch', type=int, action='store', default=5)

args = parser.parse_args()
os.makedirs(args.model_dir, exist_ok=True)

use_cuda = torch.cuda.is_available()
verbose = True
num_workers = 4

model_module = import_module(args.model_dir + '.model')

model = VAEAC(
    model_module.reconstruction_log_prob,
    model_module.proposal_network,
    model_module.prior_network,
    model_module.generative_network
)
if use_cuda:
    model = model.cuda()

optimizer = model_module.optimizer(model.parameters())
batch_size = model_module.batch_size
vlb_scale_factor = getattr(model_module, 'vlb_scale_factor', 1)
mask_generator = model_module.mask_generator

train_dataset = load_dataset(args.train_dataset)
validation_dataset = load_dataset(args.validation_dataset)

dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_workers)
val_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_workers)

validation_iwae = []
train_vlb = []
fid_history = []
ssim_history = []

def make_checkpoint():
    filename = join(args.model_dir, 'last_checkpoint.tar')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'validation_iwae': validation_iwae,
        'train_vlb': train_vlb,
        'fid_history': fid_history,
        'ssim_history': ssim_history,
    }, filename + '.bak')
    replace(filename + '.bak', filename)

# load the last checkpoint, if it exists
start_epoch = 0
if exists(join(args.model_dir, 'last_checkpoint.tar')):
    location = 'cuda' if use_cuda else 'cpu'
    checkpoint = torch.load(join(args.model_dir, 'last_checkpoint.tar'), map_location=location)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    validation_iwae = checkpoint.get('validation_iwae', [])
    train_vlb = checkpoint.get('train_vlb', [])
    fid_history = checkpoint.get('fid_history', [])
    ssim_history = checkpoint.get('ssim_history', [])
    start_epoch = checkpoint.get('epoch', 0)

for epoch in range(start_epoch, args.epochs):
    start_time = time.time()
    iterator = dataloader
    avg_vlb = 0

    if verbose:
        print('Epoch %d...' % (epoch + 1), file=stderr, flush=True)
        iterator = tqdm(iterator)

    for i, batch in enumerate(iterator):
        batch = extend_batch(batch, dataloader, batch_size)
        mask = mask_generator(batch)
        optimizer.zero_grad()
        if use_cuda:
            batch = batch.cuda()
            mask = mask.cuda()
        vlb = model.batch_vlb(batch, mask).mean()
        (-vlb / vlb_scale_factor).backward()
        optimizer.step()

        avg_vlb += (float(vlb) - avg_vlb) / (i + 1)
        if verbose:
            iterator.set_description('Train VLB: %g' % avg_vlb)

    epoch_time = (time.time() - start_time) / 60

    # --- END OF EPOCH: do validation, FID/SSIM, append metrics ---
    val_iwae = get_validation_iwae(val_dataloader, mask_generator, batch_size, model, args.validation_iwae_num_samples, verbose)
    validation_iwae.append(val_iwae)
    train_vlb.append(avg_vlb)
    make_checkpoint()
    if max(validation_iwae[::-1]) <= val_iwae:
        src_filename = join(args.model_dir, 'last_checkpoint.tar')
        dst_filename = join(args.model_dir, 'best_checkpoint.tar')
        copy(src_filename, dst_filename + '.bak')
        replace(dst_filename + '.bak', dst_filename)

    # === FID/SSIM: generate reconstructions on validation ===
    val_iter = iter(val_dataloader)
    real_images = next(val_iter)
    if use_cuda:
        real_images = real_images.cuda()
    model.eval()
    with torch.no_grad():
        mask = mask_generator(real_images)
        if mask.device != real_images.device:
            mask = mask.to(real_images.device)
        # Prior as distribution:
        prior_params = model.prior_network(torch.cat([real_images * (1 - mask), mask], 1))
        prior_dist = normal_parse_params(prior_params, 1e-3)
        latent = prior_dist.rsample()
        recon_params = model.generative_network(latent)
        recon_images = recon_params[:, :3, :, :]
    model.train()
    fid_value, ssim_value = compute_fid_ssim(real_images[:, :3], recon_images, device='cuda' if use_cuda else 'cpu')
    fid_history.append(fid_value)
    ssim_history.append(ssim_value)
    np.save(f"{args.model_dir}/fid_history.npy", np.array(fid_history))
    np.save(f"{args.model_dir}/ssim_history.npy", np.array(ssim_history))
    np.save(f"{args.model_dir}/train_vlb.npy", np.array(train_vlb))
    np.save(f"{args.model_dir}/validation_iwae.npy", np.array(validation_iwae))

    # === PRINT SUMMARY TABLE ===
    batch_shape = tuple(batch.shape)
    mask_shape = tuple(mask.shape)
    with torch.no_grad():
        prop_mean_shape = tuple(model.proposal_network(torch.cat([batch, mask], 1)).shape)
        prior_mean_shape = tuple(model.prior_network(torch.cat([batch * (1 - mask), mask], 1)).shape)
        prior_dist2 = normal_parse_params(model.prior_network(torch.cat([batch * (1 - mask), mask], 1)), 1e-3)
        latent_sample_shape = tuple(prior_dist2.rsample().shape)
        recon_shape = tuple(recon_params.shape)
    print("+----------------------------+")
    print(f"| Epoch: {epoch+1}/{args.epochs} | Average Loss: {avg_vlb:.3f} ")
    print("+----------------------------+")
    print(f"| Batch shape               : {batch_shape} |")
    print(f"| Mask shape                : {mask_shape} |")
    print(f"| Proposal mean shape       : {prop_mean_shape} |")
    print(f"| Prior mean shape          : {prior_mean_shape} |")
    print(f"| Latent sample shape       : {latent_sample_shape} |")
    print(f"| Reconstruction params shape: {recon_shape} |")
    print("+----------------------------+")
    print(f"*** Epoch {epoch+1}/{args.epochs} | Train Loss: {avg_vlb:.3f} | Time: {epoch_time:.2f} min")
    print(f"*** FID: {fid_value:.3f} | SSIM: {ssim_value:.4f} | Val IWAE: {validation_iwae[-1]:.3f}")

# === After training: plot overlay of Validation IWAE and Train VLB ===
epochs_range = np.arange(1, len(validation_iwae) + 1)

plt.figure(figsize=(12,5))
plt.plot(epochs_range, validation_iwae, marker='o', linestyle='-', color='tab:blue', label='Validation IWAE')
plt.plot(epochs_range, train_vlb, marker='x', linestyle='-', color='orange', label='Train VLB')
plt.xlabel('Validation checkpoint')
plt.ylabel('Loss')
plt.title('Validation IWAE vs Train VLB')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{args.model_dir}/vlb_iwae_overlay.png", dpi=150)
plt.show()

# You can still plot FID/SSIM if you wish (as before)
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(epochs_range, fid_history, marker='o')
plt.title('FID over Epochs')
plt.xlabel('Epoch')
plt.ylabel('FID (lower is better)')
plt.grid(True)
plt.subplot(1,2,2)
plt.plot(epochs_range, ssim_history, marker='o', color='orange')
plt.title('SSIM over Epochs')
plt.xlabel('Epoch')
plt.ylabel('SSIM (higher is better)')
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{args.model_dir}/fid_ssim_curves.png", dpi=150)
plt.show()





