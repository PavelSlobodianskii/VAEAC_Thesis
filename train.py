# train.py: VAEAC + RealNVP Prior + Contrastive Loss + IWAE
import argparse
import os
import torch
from torch.utils.data import DataLoader, Subset
from importlib import import_module
from tqdm import tqdm

from datasets import load_dataset
from VAEAC import VAEAC

parser = argparse.ArgumentParser(description='Train VAEAC with advanced priors and losses')

parser.add_argument('--model_dir', type=str, required=True,
                    help='Directory to save model/checkpoints')
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--train_dataset', type=str, required=True)
parser.add_argument('--validation_dataset', type=str, required=True)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--iwae_samples', type=int, default=10,
                    help='IWAE K samples for validation')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--print_every', type=int, default=50)
args = parser.parse_args()

# Load model definitions from the model_dir
model_module = import_module(args.model_dir + '.model')

model = VAEAC(
    model_module.reconstruction_log_prob,
    model_module.proposal_network,
    model_module.generative_network
)
model = model.to(args.device)
optimizer = model_module.optimizer(model.parameters())

# Loss scale (ICLR2019 code does this)
vlb_scale_factor = 128 ** 2

# Load datasets and dataloaders
train_dataset = load_dataset(args.train_dataset)
val_dataset = load_dataset(args.validation_dataset)

# --- LIMIT TRAIN DATASET TO FIRST 2600 IMAGES ---
train_dataset = Subset(train_dataset, range(2600))

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

os.makedirs(args.model_dir, exist_ok=True)

def train_one_epoch(model, dataloader, optimizer, device, print_every=50):
    model.train()
    running_loss = 0.0
    mask_generator = model_module.mask_generator
    for i, batch in enumerate(tqdm(dataloader, desc="Training")):
        batch = batch.to(device)
        mask = mask_generator(batch).to(device)
        optimizer.zero_grad()
        loss, metrics = model.batch_vlb(batch, mask)  # <-- unpack tuple
        loss = -loss.mean() / vlb_scale_factor
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if (i + 1) % print_every == 0:
            print(f"Batch {i+1}/{len(dataloader)} | Avg Loss: {running_loss/(i+1):.4f}")
            # Optionally print metrics
            # print(metrics)
    return running_loss / len(dataloader)

def evaluate(model, dataloader, device, K_iwae=10):
    model.eval()
    vlb_losses = []
    iwae_losses = []
    mask_generator = model_module.mask_generator
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            mask = mask_generator(batch).to(device)
            vlb, _ = model.batch_vlb(batch, mask)  # <-- unpack tuple
            iwae = model.batch_iwae(batch, mask, K=K_iwae)
            vlb = vlb.mean().item() / vlb_scale_factor
            iwae = iwae.mean().item() / vlb_scale_factor
            vlb_losses.append(vlb)
            iwae_losses.append(iwae)
    mean_vlb = sum(vlb_losses) / len(vlb_losses)
    mean_iwae = sum(iwae_losses) / len(iwae_losses)
    return mean_vlb, mean_iwae

best_val_iwae = float('-inf')
for epoch in range(1, args.epochs + 1):
    print(f"\n==== Epoch {epoch} ====")
    train_loss = train_one_epoch(model, train_loader, optimizer, args.device, print_every=args.print_every)
    print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}")

    val_vlb, val_iwae = evaluate(model, val_loader, args.device, K_iwae=args.iwae_samples)
    print(f"Validation VLB: {val_vlb:.4f} | IWAE (K={args.iwae_samples}): {val_iwae:.4f}")

    # Save checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, os.path.join(args.model_dir, 'last_checkpoint.tar'))

    # Save best (by IWAE)
    if val_iwae > best_val_iwae:
        best_val_iwae = val_iwae
        torch.save(checkpoint, os.path.join(args.model_dir, 'best_checkpoint.tar'))
        print(f"New best model (by IWAE) saved.")

print("\nTraining complete!")



