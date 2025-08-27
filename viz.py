from __future__ import annotations
import os
import numpy as np
import torch
import matplotlib.pyplot as plt


def tsne_latents(latents: np.ndarray, labels: list[str] | None, out_path: str):
    """
    latents: [N, D] numpy
    """
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=max(5, min(30, (latents.shape[0]-1)//3)))
    xy = tsne.fit_transform(latents)


    plt.figure(figsize=(10, 7))
    plt.scatter(xy[:, 0], xy[:, 1], c="tab:blue", alpha=0.65, label="samples")
    if labels:
        for (x, y), name in zip(xy, labels):
            plt.text(x + 0.8, y + 0.8, str(name), fontsize=8, color="black")
    plt.title("t-SNE of Latent Means (q(z|x,mask))")
    plt.xlabel("t-SNE dim 1"); plt.ylabel("t-SNE dim 2")
    plt.grid(True, alpha=0.25)
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
