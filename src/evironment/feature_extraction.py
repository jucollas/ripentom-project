import cv2
import numpy as np
from tqdm import tqdm
from skimage.feature import (
    graycomatrix,
    graycoprops,
    local_binary_pattern
)


def extract_features(image_path):
    """
    Extract handcrafted features from a tomato image.

    Features included:
    - HSV statistics
    - LAB statistics
    - HSV histograms
    - Texture descriptors (GLCM + LBP)
    - Dark/brown spot analysis
    - Edge density
    - Shape descriptors

    Parameters
    ----------
    image_path : str or Path
        Path to the image.

    Returns
    -------
    np.ndarray
        Feature vector as float32.
    """

    # ============================================================
    # 1. LOAD IMAGE
    # ============================================================
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    # ============================================================
    # 2. DETECT OBJECT (BLACK BACKGROUND ASSUMPTION)
    # ============================================================
    # Pixels different from black are considered part of the tomato.
    mask = np.any(image > 10, axis=2).astype(np.uint8)

    # ============================================================
    # 3. CROP ONLY THE TOMATO REGION
    # ============================================================
    ys, xs = np.where(mask)

    if len(xs) == 0 or len(ys) == 0:
        raise ValueError(f"No object detected in image: {image_path}")

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    image = image[y_min:y_max, x_min:x_max]

    # Resize to fixed size
    image = cv2.resize(image, (224, 224))

    # ============================================================
    # 4. RECOMPUTE MASK AFTER RESIZE
    # ============================================================
    mask = np.any(image > 10, axis=2).astype(np.uint8)
    mask_bool = mask > 0

    # ============================================================
    # 5. COLOR SPACE CONVERSIONS
    # ============================================================
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    features = []

    # ============================================================
    # AUXILIARY FUNCTION
    # ============================================================
    def channel_statistics(channel, valid_mask):
        """
        Compute statistical descriptors for a color channel.
        """

        pixels = channel[valid_mask]

        return [
            np.mean(pixels),
            np.std(pixels),
            np.percentile(pixels, 25),
            np.percentile(pixels, 75)
        ]

    # ============================================================
    # 6. HSV CHANNEL STATISTICS
    # ============================================================
    for i in range(3):
        channel = hsv[:, :, i]
        features.extend(channel_statistics(channel, mask_bool))

    # ============================================================
    # 7. LAB CHANNEL STATISTICS
    # ============================================================
    for i in range(3):
        channel = lab[:, :, i]
        features.extend(channel_statistics(channel, mask_bool))

    # ============================================================
    # 8. HSV HISTOGRAM FEATURES
    # ============================================================
    hue_bins = 16
    sv_bins = 16

    hist_h = cv2.calcHist(
        [hsv],
        [0],
        mask,
        [hue_bins],
        [0, 180]
    )

    hist_s = cv2.calcHist(
        [hsv],
        [1],
        mask,
        [sv_bins],
        [0, 256]
    )

    hist_v = cv2.calcHist(
        [hsv],
        [2],
        mask,
        [sv_bins],
        [0, 256]
    )

    # Normalize histograms
    hist_h = cv2.normalize(hist_h, hist_h).flatten()
    hist_s = cv2.normalize(hist_s, hist_s).flatten()
    hist_v = cv2.normalize(hist_v, hist_v).flatten()

    features.extend(hist_h)
    features.extend(hist_s)
    features.extend(hist_v)

    # ============================================================
    # 9. GLCM TEXTURE FEATURES
    # ============================================================
    glcm = graycomatrix(
        gray,
        distances=[1, 2],
        angles=[0, np.pi / 4],
        levels=256,
        symmetric=True,
        normed=True
    )

    glcm_properties = [
        "contrast",
        "correlation",
        "energy",
        "homogeneity",
        "dissimilarity"
    ]

    for prop in glcm_properties:

        values = graycoprops(glcm, prop)

        features.append(values.mean())
        features.append(values.std())

    # ============================================================
    # 10. LOCAL BINARY PATTERN (LBP)
    # ============================================================
    lbp = local_binary_pattern(
        gray,
        P=8,
        R=1,
        method="uniform"
    )

    lbp_pixels = lbp[mask_bool]

    hist_lbp, _ = np.histogram(
        lbp_pixels,
        bins=np.arange(0, 11),
        range=(0, 10)
    )

    hist_lbp = hist_lbp.astype(np.float32)

    # Normalize histogram
    hist_lbp /= (hist_lbp.sum() + 1e-6)

    features.extend(hist_lbp)

    # ============================================================
    # 11. DARK REGION RATIO
    # ============================================================
    # Detect very dark areas inside the tomato.
    dark_mask = (
        (hsv[:, :, 2] < 50) &
        mask_bool
    )

    dark_ratio = np.sum(dark_mask) / np.sum(mask_bool)

    features.append(dark_ratio)

    # ============================================================
    # 12. BROWN REGION RATIO
    # ============================================================
    # Approximate brown color range in HSV space.
    lower_brown = np.array([5, 40, 20])
    upper_brown = np.array([25, 255, 200])

    brown_mask = cv2.inRange(
        hsv,
        lower_brown,
        upper_brown
    )

    brown_ratio = (
        np.sum(brown_mask > 0) /
        np.sum(mask_bool)
    )

    features.append(brown_ratio)

    # ============================================================
    # 13. NUMBER OF BROWN REGIONS
    # ============================================================
    # Count connected brown components.
    num_labels, _ = cv2.connectedComponents(
        (brown_mask > 0).astype(np.uint8)
    )

    features.append(num_labels)

    # ============================================================
    # 14. EDGE DENSITY
    # ============================================================
    edges = cv2.Canny(gray, 100, 200)

    edge_density = (
        np.sum(edges > 0) /
        np.sum(mask_bool)
    )

    features.append(edge_density)

    # ============================================================
    # 15. SHAPE FEATURES
    # ============================================================
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) > 0:

        # Use the largest contour
        contour = max(contours, key=cv2.contourArea)

        area = cv2.contourArea(contour)

        perimeter = cv2.arcLength(contour, True)

        circularity = (
            4 * np.pi * area /
            (perimeter ** 2 + 1e-6)
        )

        x, y, width, height = cv2.boundingRect(contour)

        aspect_ratio = width / (height + 1e-6)

    else:
        area = 0
        perimeter = 0
        circularity = 0
        aspect_ratio = 0

    features.extend([
        area,
        perimeter,
        circularity,
        aspect_ratio
    ])

    # ============================================================
    # RETURN FINAL FEATURE VECTOR
    # ============================================================
    return np.array(features, dtype=np.float32)


def split_feature_extraction(split_path):
    CLASE_NAME = ["damaged", "ripe", "unripe"]
    X_train, y_train, X_val, y_val = [], [], [], []

    for split, X, y in [("train", X_train, y_train), ("val", X_val, y_val)]:
        for idx, clase in enumerate(CLASE_NAME):
            carpeta = split_path / split / clase
            imgs = list(carpeta.glob("*.*"))
            print(f"Extrayendo {split}/{clase}: {len(imgs)} imágenes...")
            for img_path in tqdm(imgs, leave=False):
                X.append(extract_features(img_path))
                y.append(idx)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_val = np.array(X_val)
    y_val = np.array(y_val)

    print(f"\n✓ X_train: {X_train.shape} | X_val: {X_val.shape}")
    return X_train, y_train, X_val, y_val
  
if __name__ == "__main__":
  from pathlib import Path
  split_path = Path("data/processed/")
  X_train, y_train, X_val, y_val = split_feature_extraction(split_path)
  
  
