# inpaint.py (patched for alpha-runs / explicit checkpoints)

from argparse import ArgumentParser
from importlib import import_module
from os import makedirs
from os.path import join
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
from tqdm import tqdm

from datasets import load_dataset, ZipDatasets
from train_utils import extend_batch_tuple
from VAEAC import VAEAC

# ---------------- ARGPARSE ----------------
p = ArgumentParser(description='Inpaint images using a trained VAEAC model.')

p.add_argument('--model_dir', type=str, required=True,
               help='Directory with model.py and checkpoints (root dir).')
p.add_argument('--num_samples', type=int, default=5,
               help='Number of inpaintings per image.')
p.add_argument('--dataset', type=str, required=True,
               help='Dataset of images to inpaint (see datasets.py).')
p.add_argument('--masks', type=str, required=True,
               help='Masks dataset (white=to inpaint).')
p.add_argument('--out_dir', type=str, required=True,
               help='Output directory for results.')
p.add_argument('--use_last_checkpoint', action='store_true', default=False,
               help='Use last.tar instead of best.tar.')
# NEW
p.add_argument('--checkpoint_dir', type=str, default=None,
               help='Folder containing best.tar/last.tar (e.g. /alpha_runs/alpha_INF)')
p.add_argument('--checkpoint', type=str, default=None,
               help='Explicit path to a checkpoint .tar (overrides the above).')

args = p.parse_args()

# ---------------- SETUP ----------------
use_cuda = torch.cuda.is_available()
num_workers = 4
verbose = True

# import the networks
model_module = import_module(args.model_dir + '.model')

model = VAEAC(
    model_module.reconstruction_log_prob,
    model_module.proposal_network,
    model_module.prior_network,
    model_module.generative_network,
    learnable_alpha=False,   # <-- add this
    kl_alpha=None           # <-- match training: None when learnable
)

if use_cuda:
    model = model.cuda()

batch_size = model_module.batch_size
sampler = model_module.sampler

# ---------------- CHECKPOINT LOGIC ----------------
if args.checkpoint is not None:
    ckpt_path = args.checkpoint
else:
    root = args.checkpoint_dir if args.checkpoint_dir is not None else args.model_dir
    ckpt_path = join(root, 'last (4).tar' if args.use_last_checkpoint else 'best (4).tar')


print(f"[INFO] Loading checkpoint: {ckpt_path}")
checkpoint = torch.load(ckpt_path, map_location=('cuda' if use_cuda else 'cpu'))
model.load_state_dict(checkpoint['model_state_dict'])

# ---------------- DATA ----------------
dataset = load_dataset(args.dataset)
masks = load_dataset(args.masks)
dataloader = DataLoader(ZipDatasets(dataset, masks), batch_size=batch_size,
                        shuffle=False, drop_last=False, num_workers=num_workers)

makedirs(args.out_dir, exist_ok=True)

def save_img(img, path):
    ToPILImage()((img / 2 + 0.5).clamp(0, 1).cpu()).save(path)

# ---------------- INPAINT LOOP ----------------
iterator = tqdm(dataloader) if verbose else dataloader
img_id = 0

for batch, masks in iterator:
    init_size = batch.shape[0]
    batch, masks = extend_batch_tuple((batch, masks), dataloader, batch_size)

    if use_cuda:
        batch, masks = batch.cuda(), masks.cuda()

    with torch.no_grad():
        params = model.generate_samples_params(batch, masks, args.num_samples)
        params = params[:init_size]

    for gt, mask, samp_params in zip(batch[:init_size], masks[:init_size], params):
        save_img(gt, join(args.out_dir, f"{img_id:05d}_groundtruth.jpg"))

        inp_vis = gt.clone()
        inp_vis[mask.bool()] = 0.5
        save_img(inp_vis, join(args.out_dir, f"{img_id:05d}_input.jpg"))

        inp = gt.clone()
        inp[mask.bool()] = 0

        samples = sampler(samp_params)
        for i, s in enumerate(samples):
            s[~mask.bool()] = 0
            s += inp
            save_img(s, join(args.out_dir, f"{img_id:05d}_sample_{i:03d}.jpg"))
        img_id += 1

print(f"[INFO] Inpainting complete. Results saved to {args.out_dir}")

