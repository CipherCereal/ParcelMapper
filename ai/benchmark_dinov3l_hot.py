import os
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


# =========================================================
# CONFIG
# =========================================================

MODEL = "models/dinov3-hot-buildings/dinov3l_buildings.onnx"
INPUT_DIR = Path("data/input")
OUT_DIR = Path("runs/dinov3l_hot_benchmark")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TILE = 256
STRIDE = 192

THRESHOLDS = [0.20, 0.25, 0.30]

MIN_AREA = 150

MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32,
)

STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32,
)


# =========================================================
# TENSORRT LIBRARIES
# =========================================================

venv = os.environ.get("VIRTUAL_ENV", "")

trt_lib = f"{venv}/lib/python3.11/site-packages/tensorrt_libs"
nvidia_lib = f"{venv}/lib/python3.11/site-packages/nvidia/cu13/lib"
cudnn_lib = f"{venv}/lib/python3.11/site-packages/nvidia/cudnn/lib"

os.environ["LD_LIBRARY_PATH"] = (
    f"{trt_lib}:{nvidia_lib}:{cudnn_lib}:"
    + os.environ.get("LD_LIBRARY_PATH", "")
)


# =========================================================
# LOAD MODEL
# =========================================================

print("=" * 60)
print("DINOv3-L HOT BUILDINGS BENCHMARK")
print("=" * 60)

print("\nLoading model...")

session = ort.InferenceSession(
    MODEL,
    providers=[
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ],
)

print("Providers:")
for p in session.get_providers():
    print(" ", p)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name


# =========================================================
# TILE INFERENCE
# =========================================================

def predict_tile(tile_rgb):

    tile = cv2.resize(
        tile_rgb,
        (TILE, TILE),
        interpolation=cv2.INTER_LINEAR,
    )

    x = tile.astype(np.float32) / 255.0

    x = (x - MEAN) / STD

    x = np.transpose(x, (2, 0, 1))

    x = np.expand_dims(x, axis=0)

    x = np.ascontiguousarray(x)

    logits = session.run(
        [output_name],
        {input_name: x},
    )[0][0]

    # Stable softmax
    logits -= np.max(
        logits,
        axis=0,
        keepdims=True,
    )

    exp_logits = np.exp(logits)

    probs = exp_logits / np.sum(
        exp_logits,
        axis=0,
        keepdims=True,
    )

    # Class 0 = building
    return probs[0]


def run_tiled(image_rgb):

    H, W = image_rgb.shape[:2]

    probability_sum = np.zeros(
        (H, W),
        dtype=np.float32,
    )

    weight_sum = np.zeros(
        (H, W),
        dtype=np.float32,
    )

    ys = list(
        range(
            0,
            max(1, H - TILE + 1),
            STRIDE,
        )
    )

    xs = list(
        range(
            0,
            max(1, W - TILE + 1),
            STRIDE,
        )
    )

    final_y = max(0, H - TILE)
    final_x = max(0, W - TILE)

    if not ys or ys[-1] != final_y:
        ys.append(final_y)

    if not xs or xs[-1] != final_x:
        xs.append(final_x)

    total_tiles = len(xs) * len(ys)

    tile_count = 0

    start = time.perf_counter()

    for y in ys:

        for x in xs:

            tile = image_rgb[
                y:y + TILE,
                x:x + TILE,
            ]

            th, tw = tile.shape[:2]

            if th != TILE or tw != TILE:

                padded = np.zeros(
                    (TILE, TILE, 3),
                    dtype=np.uint8,
                )

                padded[:th, :tw] = tile

                tile = padded

            prob = predict_tile(tile)

            prob = prob[:th, :tw]

            probability_sum[
                y:y + th,
                x:x + tw
            ] += prob

            weight_sum[
                y:y + th,
                x:x + tw
            ] += 1.0

            tile_count += 1

    elapsed = time.perf_counter() - start

    probability = (
        probability_sum
        / np.maximum(weight_sum, 1e-6)
    )

    return probability, elapsed, total_tiles


# =========================================================
# MASK CLEANUP
# =========================================================

def make_mask(probability, threshold):

    mask = (
        probability >= threshold
    ).astype(np.uint8) * 255

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5),
    )

    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3),
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        close_kernel,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        open_kernel,
    )

    # Remove tiny connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    cleaned = np.zeros_like(mask)

    for label in range(1, num_labels):

        area = stats[label, cv2.CC_STAT_AREA]

        if area >= MIN_AREA:

            cleaned[labels == label] = 255

    return cleaned


# =========================================================
# OVERLAY
# =========================================================

def make_overlay(image_bgr, mask):

    overlay = image_bgr.copy()

    colored = np.zeros_like(
        image_bgr
    )

    # Green building overlay
    colored[:, :, 1] = mask

    return cv2.addWeighted(
        overlay,
        0.65,
        colored,
        0.35,
        0,
    )


# =========================================================
# FIND INPUT IMAGES
# =========================================================

images = sorted(
    p for p in INPUT_DIR.iterdir()
    if p.suffix.lower()
    in [".jpg", ".jpeg", ".png"]
)

print(f"\nImages found: {len(images)}")

for p in images:
    print(" ", p.name)


# =========================================================
# PROCESS
# =========================================================

results = []

total_start = time.perf_counter()

for image_path in images:

    print("\n" + "-" * 60)
    print(f"Processing: {image_path.name}")
    print("-" * 60)

    image_bgr = cv2.imread(
        str(image_path)
    )

    if image_bgr is None:
        print("ERROR: could not read image")
        continue

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )

    H, W = image_rgb.shape[:2]

    print(f"Resolution: {W} x {H}")

    probability, elapsed, tiles = run_tiled(
        image_rgb
    )

    print(
        f"Tiles: {tiles} | "
        f"Inference: {elapsed:.2f}s"
    )

    print(
        f"Probability range: "
        f"{probability.min():.3f} -> "
        f"{probability.max():.3f}"
    )

    # Save probability heatmap
    heat = np.clip(
        probability * 255,
        0,
        255,
    ).astype(np.uint8)

    heatmap = cv2.applyColorMap(
        heat,
        cv2.COLORMAP_JET,
    )

    cv2.imwrite(
        str(
            OUT_DIR
            / f"{image_path.stem}_probability.jpg"
        ),
        heatmap,
    )

    row = {
        "image": image_path.name,
        "width": W,
        "height": H,
        "tiles": tiles,
        "seconds": elapsed,
    }

    for threshold in THRESHOLDS:

        mask = make_mask(
            probability,
            threshold,
        )

        pixels = np.count_nonzero(mask)

        percentage = (
            pixels / (H * W) * 100
        )

        print(
            f"  threshold={threshold:.2f} "
            f"building={percentage:.2f}%"
        )

        mask_path = (
            OUT_DIR
            / f"{image_path.stem}_mask_{threshold:.2f}.png"
        )

        overlay_path = (
            OUT_DIR
            / f"{image_path.stem}_overlay_{threshold:.2f}.jpg"
        )

        cv2.imwrite(
            str(mask_path),
            mask,
        )

        overlay = make_overlay(
            image_bgr,
            mask,
        )

        cv2.imwrite(
            str(overlay_path),
            overlay,
        )

        row[
            f"coverage_{threshold:.2f}"
        ] = percentage

    results.append(row)


# =========================================================
# SUMMARY
# =========================================================

total_elapsed = (
    time.perf_counter()
    - total_start
)

print("\n")
print("=" * 60)
print("SUMMARY")
print("=" * 60)

for row in results:

    print(
        f"\n{row['image']}"
    )

    print(
        f"  {row['width']}x{row['height']} "
        f"| {row['tiles']} tiles "
        f"| {row['seconds']:.2f}s"
    )

    for threshold in THRESHOLDS:

        print(
            f"  {threshold:.2f}: "
            f"{row[f'coverage_{threshold:.2f}']:.2f}%"
        )


print(
    f"\nTotal time: "
    f"{total_elapsed:.2f}s"
)

print(
    f"\nResults saved to:\n"
    f"{OUT_DIR}"
)

