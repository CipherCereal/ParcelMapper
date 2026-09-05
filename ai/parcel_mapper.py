from pathlib import Path
import sys
import csv
import time

import cv2
import numpy as np
import onnxruntime as ort
import torch

# ============================================================
# ParcelMapper
# DINOv3-L HOT + SAM2.1 Large
#
# Pipeline:
#   Image
#     -> overlapping DINOv3-L tiles
#     -> building candidates
#     -> expanded boxes
#     -> SAM2 box + center point
#     -> overlapping mask merge
#     -> polygons
# ============================================================

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = BASE_DIR / "data" / "input"

DINO_MODEL = (
    BASE_DIR
    / "models"
    / "dinov3-hot-buildings"
    / "dinov3l_buildings.onnx"
)

SAM2_CHECKPOINT = (
    BASE_DIR
    / "models"
    / "sam2"
    / "sam2.1_hiera_large.pt"
)

OUTPUT_DIR = BASE_DIR / "runs" / "parcel_mapper"

SAM2_REPO = BASE_DIR / "sam2"

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

TILE_SIZE = 256

# More overlap than the original 192 stride.
STRIDE = 160

# Building recall is the priority.
DINO_THRESHOLD = 0.18

# Ignore tiny DINO regions.
MIN_AREA = 250

# Give SAM2 some surrounding context.
BOX_EXPAND = 0.15

# Maximum number of candidate regions per image.
MAX_CANDIDATES = 80

# Minimum SAM2 mask area.
MIN_SAM_AREA = 250

# Merge masks when one substantially overlaps another.
MERGE_OVERLAP = 0.35

# Polygon simplification.
POLYGON_EPSILON = 0.01

# Supported image formats.
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".JPG",
    ".JPEG",
    ".PNG",
}

# ------------------------------------------------------------
# SAM2 import setup
# ------------------------------------------------------------

sys.path.insert(0, str(SAM2_REPO))


# ============================================================
# Utility functions
# ============================================================

def sigmoid(x):
    return 1.0 / (
        1.0 + np.exp(-np.clip(x, -50, 50))
    )


def normalize_dino(image):
    """
    HOT DINOv3-L normalization.

    The HOT model does NOT use ImageNet normalization.
    """

    image = image.astype(np.float32) / 255.0

    mean = np.array(
        [0.4297, 0.4002, 0.3433],
        dtype=np.float32,
    )

    std = np.array(
        [0.2056, 0.1674, 0.1599],
        dtype=np.float32,
    )

    image = (image - mean) / std

    return image.transpose(2, 0, 1)[None]


def save_mask(path, mask):
    """
    Save binary mask as PNG.
    """

    output = (
        mask.astype(np.uint8) * 255
    )

    cv2.imwrite(str(path), output)


def expand_box(box, width, height, amount):
    """
    Expand a bounding box by a percentage.
    """

    x1, y1, x2, y2 = box

    box_width = x2 - x1
    box_height = y2 - y1

    x1 = int(x1 - box_width * amount)
    y1 = int(y1 - box_height * amount)

    x2 = int(x2 + box_width * amount)
    y2 = int(y2 + box_height * amount)

    x1 = max(0, x1)
    y1 = max(0, y1)

    x2 = min(width - 1, x2)
    y2 = min(height - 1, y2)

    return [
        x1,
        y1,
        x2,
        y2,
    ]


def mask_overlap(mask_a, mask_b):
    """
    Calculate intersection relative to the smaller mask.

    This is intentionally different from IoU:
    if one mask is mostly contained by another, they merge.
    """

    intersection = np.logical_and(
        mask_a,
        mask_b,
    ).sum()

    smaller = min(
        mask_a.sum(),
        mask_b.sum(),
    )

    if smaller == 0:
        return 0.0

    return intersection / smaller


def mask_to_polygon(mask):
    """
    Convert a binary mask into polygon coordinates.

    Returns the largest external contour.
    """

    mask_uint8 = (
        mask.astype(np.uint8) * 255
    )

    contours, _ = cv2.findContours(
        mask_uint8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return None

    contour = max(
        contours,
        key=cv2.contourArea,
    )

    area = cv2.contourArea(contour)

    if area < MIN_AREA:
        return None

    perimeter = cv2.arcLength(
        contour,
        True,
    )

    epsilon = (
        POLYGON_EPSILON * perimeter
    )

    polygon = cv2.approxPolyDP(
        contour,
        epsilon,
        True,
    )

    points = polygon.reshape(-1, 2)

    if len(points) < 3:
        return None

    return [
        [int(x), int(y)]
        for x, y in points
    ]


def polygon_area(points):
    """
    Shoelace polygon area.
    """

    if len(points) < 3:
        return 0.0

    pts = np.asarray(
        points,
        dtype=np.float64,
    )

    x = pts[:, 0]
    y = pts[:, 1]

    return abs(
        np.sum(
            x * np.roll(y, -1)
            - y * np.roll(x, -1)
        )
        / 2.0
    )


# ============================================================
# DINOv3-L
# ============================================================

def run_dino(
    image_rgb,
    session,
):
    """
    Run overlapping DINOv3-L HOT inference.

    HOT output:
        channel 0 = building mask logit
        channel 1 = boundary logit
        channel 2 = signed distance

    For candidate generation we use channel 0.
    """

    height, width = image_rgb.shape[:2]

    probability = np.zeros(
        (height, width),
        dtype=np.float32,
    )

    weights = np.zeros(
        (height, width),
        dtype=np.float32,
    )

    input_name = session.get_inputs()[0].name

    tile_count = 0

    for y in range(
        0,
        height,
        STRIDE,
    ):

        for x in range(
            0,
            width,
            STRIDE,
        ):

            x2 = min(
                x + TILE_SIZE,
                width,
            )

            y2 = min(
                y + TILE_SIZE,
                height,
            )

            crop = image_rgb[
                y:y2,
                x:x2,
            ]

            crop_height, crop_width = crop.shape[:2]

            padded = np.zeros(
                (
                    TILE_SIZE,
                    TILE_SIZE,
                    3,
                ),
                dtype=np.uint8,
            )

            padded[
                :crop_height,
                :crop_width,
            ] = crop

            tensor = normalize_dino(
                padded
            )

            output = session.run(
                None,
                {
                    input_name: tensor
                },
            )[0]

            building_logits = (
                output[0, 0]
            )

            tile_probability = sigmoid(
                building_logits
            )

            probability[
                y:y2,
                x:x2,
            ] += tile_probability[
                :crop_height,
                :crop_width,
            ]

            weights[
                y:y2,
                x:x2,
            ] += 1.0

            tile_count += 1

    probability /= np.maximum(
        weights,
        1e-6,
    )

    return probability, tile_count


def get_candidates(probability):
    """
    Convert DINO probability map into candidate building regions.
    """

    mask = (
        probability >= DINO_THRESHOLD
    ).astype(np.uint8)

    kernel = np.ones(
        (3, 3),
        dtype=np.uint8,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1,
    )

    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )
    )

    candidates = []

    for label in range(
        1,
        num_labels,
    ):

        x, y, w, h, area = (
            stats[label]
        )

        if area < MIN_AREA:
            continue

        x2 = x + w
        y2 = y + h

        region = (
            labels == label
        )

        score = float(
            probability[region].mean()
        )

        candidates.append(
            {
                "box": [
                    int(x),
                    int(y),
                    int(x2),
                    int(y2),
                ],
                "area": int(area),
                "score": score,
            }
        )

    candidates.sort(
        key=lambda item: item["area"],
        reverse=True,
    )

    return candidates[:MAX_CANDIDATES], mask


# ============================================================
# SAM2
# ============================================================

def refine_with_sam2(
    image_rgb,
    candidates,
    predictor,
):
    """
    Refine DINO candidate regions using:

        expanded box + DINO candidate center point
    """

    height, width = image_rgb.shape[:2]

    results = []

    for candidate in candidates:

        original_box = candidate["box"]

        expanded_box = expand_box(
            original_box,
            width,
            height,
            BOX_EXPAND,
        )

        x1, y1, x2, y2 = (
            expanded_box
        )

        if (
            x2 - x1 < 20
            or y2 - y1 < 20
        ):
            continue

        ox1, oy1, ox2, oy2 = (
            original_box
        )

        point_x = int(
            (ox1 + ox2) / 2
        )

        point_y = int(
            (oy1 + oy2) / 2
        )

        box = np.array(
            expanded_box,
            dtype=np.float32,
        )

        point_coords = np.array(
            [[point_x, point_y]],
            dtype=np.float32,
        )

        point_labels = np.array(
            [1],
            dtype=np.int32,
        )

        masks, scores, _ = (
            predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                multimask_output=False,
            )
        )

        mask = (
            masks[0]
            .astype(np.uint8)
        )

        score = float(
            scores[0]
        )

        area = int(
            mask.sum()
        )

        if area < MIN_SAM_AREA:
            continue

        results.append(
            {
                "mask": mask,
                "score": score,
                "dino_score": candidate[
                    "score"
                ],
                "box": expanded_box,
            }
        )

    return results


# ============================================================
# Mask merging
# ============================================================

def merge_masks(mask_results):
    """
    Merge substantially overlapping SAM2 masks.
    """

    merged = []

    # Highest SAM confidence first.
    ordered = sorted(
        mask_results,
        key=lambda item: item["score"],
        reverse=True,
    )

    for item in ordered:

        mask = item["mask"]

        merged_into_existing = False

        for existing in merged:

            overlap = mask_overlap(
                mask,
                existing["mask"],
            )

            if overlap >= MERGE_OVERLAP:

                existing["mask"] = (
                    np.logical_or(
                        existing["mask"],
                        mask,
                    ).astype(np.uint8)
                )

                existing["score"] = max(
                    existing["score"],
                    item["score"],
                )

                merged_into_existing = True

                break

        if not merged_into_existing:

            merged.append(
                {
                    "mask": mask.copy(),
                    "score": item["score"],
                }
            )

    return merged


# ============================================================
# Visualization
# ============================================================

def create_overlay(
    image,
    mask,
):
    """
    Create final building overlay.
    """

    output = image.copy()

    mask_bool = mask > 0

    # Red building fill.
    red = np.zeros_like(output)
    red[:, :, 2] = 255

    output[mask_bool] = (
        0.35 * output[mask_bool]
        + 0.65 * red[mask_bool]
    ).astype(np.uint8)

    # Yellow building boundaries.
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    cv2.drawContours(
        output,
        contours,
        -1,
        (0, 255, 255),
        2,
    )

    return output


# ============================================================
# GeoJSON
# ============================================================

def save_geojson(
    path,
    image_name,
    polygons,
):
    """
    Save pixel-coordinate building polygons as GeoJSON.

    NOTE:
    These are image pixel coordinates, not geographic
    coordinates. Georeferencing can be added later if
    the source imagery has a camera/GPS transform.
    """

    features = []

    for index, polygon in enumerate(
        polygons
    ):

        # Close polygon.
        ring = [
            [float(x), float(y)]
            for x, y in polygon
        ]

        if ring[0] != ring[-1]:
            ring.append(
                ring[0]
            )

        features.append(
            {
                "type": "Feature",
                "properties": {
                    "id": index,
                    "image": image_name,
                    "area_pixels": polygon_area(
                        polygon
                    ),
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        ring
                    ],
                },
            }
        )

    geojson = {
        "type": "FeatureCollection",
        "features": features,
    }

    import json

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            geojson,
            file,
            indent=2,
        )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("ParcelMapper")
    print("DINOv3-L HOT + SAM2.1 Large")
    print("=" * 70)

    print(f"Input : {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Find images
    # --------------------------------------------------------

    images = sorted(
        [
            path
            for path in INPUT_DIR.iterdir()
            if path.suffix in IMAGE_EXTENSIONS
        ]
    )

    if not images:
        raise RuntimeError(
            f"No images found in {INPUT_DIR}"
        )

    print(
        f"\nFound {len(images)} images."
    )

    # --------------------------------------------------------
    # Load DINO
    # --------------------------------------------------------

    print(
        "\nLoading DINOv3-L HOT..."
    )

    providers = [
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]

    dino_session = (
        ort.InferenceSession(
            str(DINO_MODEL),
            providers=providers,
        )
    )

    print(
        "DINO providers:",
        dino_session.get_providers(),
    )

    # --------------------------------------------------------
    # Load SAM2
    # --------------------------------------------------------

    print(
        "\nLoading SAM2.1 Large..."
    )

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import (
        SAM2ImagePredictor,
    )

    config = (
        "configs/sam2.1/sam2.1_hiera_l.yaml"
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    sam2_model = build_sam2(
        config,
        str(SAM2_CHECKPOINT),
        device=device,
    )

    predictor = SAM2ImagePredictor(
        sam2_model
    )

    print(
        f"SAM2 device: {device}"
    )

    # --------------------------------------------------------
    # CSV summary
    # --------------------------------------------------------

    summary_path = (
        OUTPUT_DIR
        / "summary.csv"
    )

    summary_rows = []

    total_start = time.perf_counter()

    # --------------------------------------------------------
    # Process images
    # --------------------------------------------------------

    for image_index, image_path in enumerate(
        images,
        start=1,
    ):

        print("\n" + "=" * 70)
        print(
            f"[{image_index}/{len(images)}] "
            f"{image_path.name}"
        )
        print("=" * 70)

        start_time = (
            time.perf_counter()
        )

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            print(
                "WARNING: Could not load image."
            )
            continue

        image_rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        height, width = (
            image.shape[:2]
        )

        # ----------------------------------------------------
        # Per-image directory
        # ----------------------------------------------------

        image_output = (
            OUTPUT_DIR
            / image_path.stem
        )

        image_output.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ----------------------------------------------------
        # DINO
        # ----------------------------------------------------

        print(
            "Running DINOv3-L..."
        )

        dino_start = (
            time.perf_counter()
        )

        probability, tile_count = (
            run_dino(
                image_rgb,
                dino_session,
            )
        )

        dino_time = (
            time.perf_counter()
            - dino_start
        )

        # Save probability.
        probability_image = (
            np.clip(
                probability * 255,
                0,
                255,
            ).astype(np.uint8)
        )

        cv2.imwrite(
            str(
                image_output
                / "dino_probability.jpg"
            ),
            probability_image,
        )

        # ----------------------------------------------------
        # Candidates
        # ----------------------------------------------------

        candidates, dino_mask = (
            get_candidates(
                probability
            )
        )

        save_mask(
            image_output
            / "dino_mask.png",
            dino_mask,
        )

        print(
            f"DINO candidates: "
            f"{len(candidates)}"
        )

        # ----------------------------------------------------
        # SAM2
        # ----------------------------------------------------

        predictor.set_image(
            image_rgb
        )

        print(
            "Running SAM2 refinement..."
        )

        sam_start = (
            time.perf_counter()
        )

        sam_results = (
            refine_with_sam2(
                image_rgb,
                candidates,
                predictor,
            )
        )

        sam_time = (
            time.perf_counter()
            - sam_start
        )

        print(
            f"SAM2 masks: "
            f"{len(sam_results)}"
        )

        # ----------------------------------------------------
        # Merge
        # ----------------------------------------------------

        merged_results = merge_masks(
            sam_results
        )

        print(
            f"Merged buildings: "
            f"{len(merged_results)}"
        )

        # ----------------------------------------------------
        # Final combined mask
        # ----------------------------------------------------

        final_mask = np.zeros(
            (height, width),
            dtype=np.uint8,
        )

        for item in merged_results:

            final_mask = (
                np.logical_or(
                    final_mask,
                    item["mask"],
                ).astype(np.uint8)
            )

        save_mask(
            image_output
            / "building_mask.png",
            final_mask,
        )

        # ----------------------------------------------------
        # Final overlay
        # ----------------------------------------------------

        overlay = create_overlay(
            image,
            final_mask,
        )

        cv2.imwrite(
            str(
                image_output
                / "overlay.jpg"
            ),
            overlay,
        )

        # ----------------------------------------------------
        # Polygons
        # ----------------------------------------------------

        polygons = []

        for item in merged_results:

            polygon = mask_to_polygon(
                item["mask"]
            )

            if polygon is not None:
                polygons.append(
                    polygon
                )

        print(
            f"Polygons: "
            f"{len(polygons)}"
        )

        save_geojson(
            image_output
            / "buildings.geojson",
            image_path.name,
            polygons,
        )

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        dino_pixels = int(
            dino_mask.sum()
        )

        final_pixels = int(
            final_mask.sum()
        )

        total_pixels = (
            width * height
        )

        dino_coverage = (
            dino_pixels
            / total_pixels
            * 100
        )

        final_coverage = (
            final_pixels
            / total_pixels
            * 100
        )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        print(
            f"DINO coverage : "
            f"{dino_coverage:.2f}%"
        )

        print(
            f"Final coverage: "
            f"{final_coverage:.2f}%"
        )

        print(
            f"DINO time     : "
            f"{dino_time:.2f}s"
        )

        print(
            f"SAM2 time     : "
            f"{sam_time:.2f}s"
        )

        print(
            f"Total time    : "
            f"{elapsed:.2f}s"
        )

        summary_rows.append(
            {
                "image": image_path.name,
                "width": width,
                "height": height,
                "tiles": tile_count,
                "dino_candidates": len(
                    candidates
                ),
                "sam_masks": len(
                    sam_results
                ),
                "merged_buildings": len(
                    merged_results
                ),
                "polygons": len(
                    polygons
                ),
                "dino_coverage_percent": round(
                    dino_coverage,
                    3,
                ),
                "final_coverage_percent": round(
                    final_coverage,
                    3,
                ),
                "dino_seconds": round(
                    dino_time,
                    3,
                ),
                "sam2_seconds": round(
                    sam_time,
                    3,
                ),
                "total_seconds": round(
                    elapsed,
                    3,
                ),
            }
        )

        # Clear SAM2 image state before next frame.
        try:
            predictor.reset_predictor()
        except Exception:
            pass

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    if summary_rows:

        fieldnames = list(
            summary_rows[0].keys()
        )

        with open(
            summary_path,
            "w",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )

            writer.writeheader()
            writer.writerows(
                summary_rows
            )

    total_time = (
        time.perf_counter()
        - total_start
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PARCELMAPPER COMPLETE")
    print("=" * 70)

    print(
        f"Images processed: "
        f"{len(summary_rows)}"
    )

    print(
        f"Total runtime: "
        f"{total_time:.2f}s"
    )

    print(
        f"\nResults:"
    )

    print(
        OUTPUT_DIR
    )

    print(
        f"\nSummary:"
    )

    print(
        summary_path
    )

    print("\nPer-image outputs:")
    print(
        "  dino_probability.jpg"
    )
    print(
        "  dino_mask.png"
    )
    print(
        "  building_mask.png"
    )
    print(
        "  overlay.jpg"
    )
    print(
        "  buildings.geojson"
    )


if __name__ == "__main__":
    main()
