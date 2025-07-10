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
import matplotlib.pyplot as plt
import pickle

from datasets import load_dataset
from train_utils import extend_batch, get_validation_iwae
from VAEAC import VAEAC

parser = ArgumentParser(description='Train VAEAC to inpaint.')

parser.add_argument('--model_dir', type=str, action='store', required=True)
parser.add_argument('--epochs', type=int, action='store', required=True)
parser.add_argument('--train_dataset', type=str, action='store', required=True)
parser.add_argument('--validation_dataset', type=str, action='store', required=True)
parser.add_argument('--validation_iwae_num_samples', type=int, action='store', default=25)
parser.add_argument('--validations_per_epoch', type=int, action='store', default=5)
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
verbose = True
num_workers = 0

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

validation_batches = ceil(len(dataloader) / args.validations_per_epoch)

validation_iwae = []
train_vlb = []
rec_errors = []
kl_terms = []

if exists(join(args.model_dir, 'last_checkpoint.tar')):
    location = 'cuda' if use_cuda else 'cpu'
    checkpoint = torch.load(join(args.model_dir, 'last_checkpoint.tar'), map_location=location)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    validation_iwae = checkpoint.get('validation_iwae', [])
    train_vlb = checkpoint.get('train_vlb', [])
    rec_errors = checkpoint.get('rec_errors', [])
    kl_terms = checkpoint.get('kl_terms', [])

def make_checkpoint(epoch):
    filename = join(args.model_dir, 'last_checkpoint.tar')
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

for epoch in range(args.epochs):

    iterator = dataloader
    avg_vlb = 0
    if verbose:
        print('Epoch %d...' % (epoch + 1), file=stderr, flush=True)
        iterator = tqdm(iterator)

    last_batch = None
    last_mask = None

    for i, batch in enumerate(iterator):

        # Checkpoint/validation logic
        if any([
                i == 0 and epoch == 0,
                i % validation_batches == validation_batches - 1,
                i + 1 == len(dataloader)
            ]):
            val_iwae = get_validation_iwae(val_dataloader, mask_generator,
                                           batch_size, model,
                                           args.validation_iwae_num_samples,
                                           verbose)
            validation_iwae.append(val_iwae)
            train_vlb.append(avg_vlb)
            make_checkpoint(epoch)
            if max(validation_iwae[::-1]) <= val_iwae:
                src_filename = join(args.model_dir, 'last_checkpoint.tar')
                dst_filename = join(args.model_dir, 'best_checkpoint.tar')
                copy(src_filename, dst_filename + '.bak')
                replace(dst_filename + '.bak', dst_filename)
            if verbose:
                print(file=stderr)
                print(file=stderr)

        batch = extend_batch(batch, dataloader, batch_size)
        mask = mask_generator(batch)
        optimizer.zero_grad()
        if use_cuda:
            batch = batch.cuda()
            mask = mask.cuda()

        # Standard VLB forward/backward
        vlb = model.batch_vlb(batch, mask)
        vlb = vlb.mean()
        (-vlb / vlb_scale_factor).backward()
        optimizer.step()

        # --- Logging: compute rec_error and KL per batch (NO grad needed) ---
        with torch.no_grad():
            proposal, prior = model.make_latent_distributions(batch, mask)
            latent = proposal.rsample()
            rec_params = model.generative_network(latent)
            rec_error = float(model.rec_log_prob(batch, rec_params, mask).mean().item())
            kl = float(torch.distributions.kl_divergence(proposal, prior).view(batch.shape[0], -1).sum(-1).mean().item())
            rec_errors.append(rec_error)
            kl_terms.append(kl)

        avg_vlb += (float(vlb) - avg_vlb) / (i + 1)
        if verbose:
            iterator.set_description('Train VLB: %g' % avg_vlb)

        last_batch = batch
        last_mask = mask

    # ---- end of epoch: pretty print ----
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

# ----------------------------- After training ------------------------------ #
print("[INFO] Training complete – saving history & plots…")

hist_path = join(args.model_dir, "iwae_and_vlb.pkl")
with open(hist_path, "wb") as f:
    pickle.dump({
        "validation_iwae": validation_iwae,
        "train_vlb": train_vlb,
        "rec_errors": rec_errors,
        "kl_terms": kl_terms,
    }, f)
print(f"[INFO] Saved history to {hist_path}")

# ----------- Plot curves
plt.figure(figsize=(10,5))
plt.plot(validation_iwae, label="Validation IWAE", marker="o")
plt.plot(train_vlb, label="Train VLB", marker="x")
plt.plot(rec_errors, label="Reconstruction Error", alpha=0.7)
plt.plot(kl_terms, label="KL Divergence", alpha=0.7)
plt.xlabel("Batch / Validation checkpoint")
plt.ylabel("Loss/Value")
plt.title("Validation IWAE, Train VLB, Reconstruction Error, KL Term")
plt.legend()
plt.tight_layout()
plot_path = join(args.model_dir, "loss_curves.png")
plt.savefig(plot_path)
plt.close()
print(f"[INFO] Saved plot to {plot_path}")

