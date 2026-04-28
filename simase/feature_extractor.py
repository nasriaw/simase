import io
from typing import Tuple

import numpy as np
from PIL import Image, ImageOps
from skimage import color, transform
from skimage.feature import hog


def load_image_from_bytes(file_bytes: bytes) -> np.ndarray:
    """
    Load image from raw bytes, handle EXIF orientation, and return as RGB numpy array [H,W,3] in [0,1].
    """
    with Image.open(io.BytesIO(file_bytes)) as im:
        im = ImageOps.exif_transpose(im)  # fix orientation
        im = im.convert("RGB")
        arr = np.asarray(im).astype(np.float32) / 255.0  # scale to [0,1]
        return arr


def preprocess_image(img: np.ndarray, size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """
    Ensure image is RGB float32 in [0,1] and resize to given size.
    """
    if img.ndim == 2:
        # grayscale -> RGB by stacking
        img = np.stack([img, img, img], axis=-1)
    elif img.shape[-1] == 4:
        # RGBA -> RGB (drop alpha)
        img = img[..., :3]

    # skimage.transform.resize returns float64 by default and scales to [0,1]
    img_resized = transform.resize(
        img, size, order=1, mode="reflect", anti_aliasing=True, preserve_range=False
    ).astype(np.float32)
    return img_resized


def extract_features(img_rgb: np.ndarray) -> np.ndarray:
    """
    Extract a simple feature vector using skimage:
    - HOG on grayscale
    - Color histogram (RGB, 32 bins per channel)
    Returns 1D float32 numpy array.
    """
    # Grayscale for HOG
    gray = color.rgb2gray(img_rgb)

    hog_vec = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    ).astype(np.float32)

    # Color histograms
    bins = 32
    hist_feats = []
    for c in range(3):
        h, _ = np.histogram(img_rgb[..., c], bins=bins, range=(0.0, 1.0), density=True)
        hist_feats.append(h.astype(np.float32))
    hist_vec = np.concatenate(hist_feats, axis=0)

    feats = np.concatenate([hog_vec, hist_vec], axis=0).astype(np.float32)
    return feats


def extract_from_bytes(file_bytes: bytes) -> np.ndarray:
    """Convenience function: bytes -> RGB -> resize -> features."""
    arr = load_image_from_bytes(file_bytes)
    arr = preprocess_image(arr, size=(256, 256))
    feats = extract_features(arr)
    return feats
