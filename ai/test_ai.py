from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
from ultralytics import YOLO


# ============================================================
# Paths
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = PROJECT_DIR / "data" / "input"
OUTPUT_DIR = PROJECT_DIR / "runs" / "dinov3s"

DINO_MODEL = PROJECT_DIR / "models" / "dinov3s-buildings.onnx"
YOLO_MODEL = PROJECT_DIR / "ai" / "yolo26n-seg.pt"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# YOLO
# ============================================================

print("Loading YOLO...")
yolo_model = YOLO(str(YOLO_MODEL))


# ============================================================
# DINOv3 ONNX
# ============================================================

print("Loading DINOv3 ONNX...")

dino_session = ort.InferenceSession(
    str(DINO_MODEL),
    providers=["CPUExecutionProvider"],
)

dino_input = dino_session.get_inputs()[0]
dino_output = dino_session.get_outputs()[0]

print(f"DINO input : {dino_input.name} {dino_input.shape}")
print(f"DINO output: {dino_output.name} {dino_output.shape}")


# ============================================================
# DINOv3 inference
# ============================================================

def run_dino(image: Image.Image):
    """
    Run DINOv3 building segmentation on a single 256x256 image.

    Returns:
        logits: numpy array with shape [3, 256, 256]
    """

    # Convert to RGB
    image = image.convert("RGB")

    # Model expects exactly 256x256
    image = image.resize((256, 256), Image.Resampling.BILINEAR)

    # RGB -> numpy float32
    image_np = np.asarray(image).astype(np.float32)

    # Scale 0..255 -> 0..1
    image_np /= 255.0

    # HWC -> CHW
    image_np = np.transpose(image_np, (2, 0, 1))

    # Add batch dimension
    image_np = np.expand_dims(image_np, axis=0)

    # Inference
    logits = dino_session.run(
        [dino_output.name],
        {dino_input.name: image_np},
    )[0]

    # [1, 3, 256, 256] -> [3, 256, 256]
    return logits[0]


# ============================================================
# Process images
# ============================================================

image_files = sorted(
    path
    for path in INPUT_DIR.iterdir()
    if path.suffix.lower() in [".jpg", ".jpeg", ".png"]
)


if not image_files:
    print(f"No images found in {INPUT_DIR}")
    raise SystemExit(1)


for image_path in image_files:

    print()
    print("=" * 60)
    print(f"Processing: {image_path.name}")

    # --------------------------------------------------------
    # YOLO
    # --------------------------------------------------------

    print("Running YOLO...")

    yolo_results = yolo_model(
        str(image_path),
        save=True,
    )

    for result in yolo_results:
        print(f"YOLO result: {result}")

    # --------------------------------------------------------
    # DINOv3
    # --------------------------------------------------------

    print("Running DINOv3...")

    image = Image.open(image_path)

    logits = run_dino(image)

    # --------------------------------------------------------
    # Convert logits to class prediction
    # --------------------------------------------------------

    class_map = np.argmax(logits, axis=0).astype(np.uint8)

    # Save raw class map
    class_map_path = OUTPUT_DIR / f"{image_path.stem}_classes.png"

    cv2.imwrite(
        str(class_map_path),
        class_map,
    )

    # --------------------------------------------------------
    # Create a simple building mask
    #
    # NOTE:
    # We are initially assuming class 1 represents buildings.
    # We will verify this against the model documentation/output.
    # --------------------------------------------------------

    building_mask = (class_map == 1).astype(np.uint8) * 255

    mask_path = OUTPUT_DIR / f"{image_path.stem}_building_mask.png"

    cv2.imwrite(
        str(mask_path),
        building_mask,
    )

    print(f"DINO classes : {class_map_path}")
    print(f"Building mask: {mask_path}")


print()
print("=" * 60)
print("Finished.")
print(f"Outputs saved to: {OUTPUT_DIR}")
