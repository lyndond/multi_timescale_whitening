import array
import os
import os.path as op
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import seaborn as sns


def load_images(
    data_dir: str = "/mnt/home/tyerxa/ceph/datasets/datasets/vanhateren_imc", 
    n_images: int = 1,
    crop_size: int = 256, 
    rng: Optional[np.random.Generator] = None,
    ) -> List[npt.NDArray[np.uint16]]:
    """Loads randomly cropped img from van hateren dataset."""

    if rng is None:
        rng = np.random.default_rng()

    files = sorted(os.listdir(data_dir))
    rand_idx = rng.choice(range(len(files)), n_images, replace=False)

    images = []
    for i in rand_idx:
        filename = files[i]
        with open(op.join(data_dir, filename), 'rb') as handle:
            s = handle.read()
            arr = array.array('H', s)
            arr.byteswap()
        img = np.array(arr, dtype='uint16').reshape(1024, 1536)
        H, W = img.shape

        rand_h = rng.integers(0, H-crop_size, 1)[0]
        rand_w = rng.integers(0, W-crop_size, 1)[0]
        img = img[rand_h:rand_h + crop_size, rand_w:rand_w + crop_size]
        images.append(img)
    return images


def random_walk(
    n_steps: int, 
    sigma: float = 1.0,
    rng: Optional[np.random.Generator] = None,
    ) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """2D Gaussian random walk."""
    if rng is None:
        rng = np.random.default_rng()
    x = rng.normal(0, sigma, n_steps)
    y = rng.normal(0, sigma, n_steps)
    return np.cumsum(x).astype(int), np.cumsum(y).astype(int)


def get_patches(
    img: npt.NDArray, 
    patch_h: int, 
    patch_w: int,
    walk_h: npt.NDArray[np.int64], 
    walk_w: npt.NDArray[np.int64]
    ) -> List[npt.NDArray]:
    """Get patches within a context."""
    all_images = []
    for di, dj in zip(walk_h, walk_w):
        all_images.append(img[di:di+patch_h, dj:dj+patch_w])
    return all_images


def get_contexts(
    img: npt.NDArray, 
    patch_h: int, 
    patch_w: int, 
    n_contexts: int, 
    sigma: float, 
    n_steps: int, 
    pad_factor: int = 1,
    rng: Optional[np.random.Generator] = None,
    ) -> Tuple[npt.NDArray, List[npt.NDArray[np.int64]]]:

    if rng is None:
        rng = np.random.default_rng()

    img_h, img_w = img.shape
    pad_h, pad_w = pad_factor * patch_h, pad_factor * patch_w

    all_contexts = []
    walk_coords = []

    for _ in range(n_contexts):
        i = rng.integers(pad_h, img_h-pad_w, 1)[0]
        j = rng.integers(pad_h, img_w-pad_w, 1)[0]
        walk_h, walk_w = random_walk(n_steps, sigma, rng)
        walk_h = np.clip(walk_h+i, 0+patch_h, img_h-patch_h)
        walk_w = np.clip(walk_w+j, 0+patch_w, img_w-patch_w)
        all_contexts.append(get_patches(img, patch_h, patch_w, walk_h, walk_w))
        walk_coords.append(np.stack([walk_h, walk_w], axis=1))

    return np.array(all_contexts), walk_coords


# sample 5 images without replacement from all_images and plot
def add_subplot_border(ax, width=1, color=None ):
    """from https://stackoverflow.com/questions/45441909/how-to-add-a-fixed-width-border-to-subplot"""

    fig = ax.get_figure()

    # Convert bottom-left and top-right to display coordinates
    x0, y0 = ax.transAxes.transform((0, 0))
    x1, y1 = ax.transAxes.transform((1, 1))

    # Convert back to Axes coordinates
    x0, y0 = ax.transAxes.inverted().transform((x0, y0))
    x1, y1 = ax.transAxes.inverted().transform((x1, y1))

    rect = plt.Rectangle(
        (x0, y0), x1-x0, y1-y0,
        color=color,
        transform=ax.transAxes,
        zorder=-1,
        lw=2*width+1,
        fill=None,
    )
    fig.patches.append(rect)


def plot_context_samples(
    all_contexts: npt.NDArray, 
    n_samples: int, 
    cmap: str = "gray",
    palette: str = "Set1", 
    plot_border: bool = True,
    rng: Optional[np.random.Generator] = None,
    dpi: Optional[int] = None,
    ):
    if rng is None:
        rng = np.random.default_rng()
    n_contexts, n_steps, _, _ = all_contexts.shape
    sampled_idx = rng.choice(n_steps, n_samples, replace=False)

    cols = sns.color_palette(palette, n_contexts)
    fig, ax = plt.subplots(n_contexts, n_samples, figsize=(n_samples, 4), dpi=dpi, squeeze=False)
    for ctx in range(n_contexts):
        VMIN, VMAX = np.min(all_contexts[ctx]), np.max(all_contexts[ctx])

        for i in range(n_samples):
            ax[ctx, i].imshow(all_contexts[ctx][sampled_idx[i]], cmap=cmap, vmin=VMIN, vmax=VMAX)

            if plot_border:
                add_subplot_border(ax[ctx, i], width=3, color=cols[ctx])
            ax[ctx, i].axis("off")

    fig.tight_layout()

    return fig, ax


def plot_patch_stats(all_images: npt.NDArray) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharex="all", sharey="all")

    im = ax[0].imshow(np.mean(all_images, 0), cmap="bone")
    plt.colorbar(im)
    ax[0].set(title="Cross-patch mean")

    im = ax[1].imshow(np.var(all_images, 0), cmap="bone")
    ax[1].set(title="Cross-patch variance", xticklabels=[], yticklabels=[])
    plt.colorbar(im)
    fig.tight_layout()
