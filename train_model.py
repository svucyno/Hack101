"""
train_model.py  —  STEP 2  (CORRECTED)
=======================================
Trains YOLOv8n on the 3-class helmet/rider dataset.

Classes trained:
  0 = No_Helmet    (violation — rider without helmet)
  1 = With_Helmet  (clean — rider WITH helmet, needed for contrast)
  2 = Rider        (person on bike, used for triple-riding count)

WHY 3 CLASSES INSTEAD OF 2:
  The previous version dropped With_Helmet entirely.  That meant the
  model was only shown "no helmet" examples with no negative examples.
  It could not learn the visual difference between helmet/no-helmet,
  causing it to label helmeted riders as "No_Helmet".
  With_Helmet (class 1) is the negative class that teaches the model
  what a helmeted head looks like so it stops misfiring.

RUN AFTER prepare_dataset.py:
    python train_model.py

OUTPUT:
    runs/detect/violation_model/weights/best.pt
"""

import os, sys, json
from pathlib import Path
import torch

BASE = Path(os.getcwd())

# ── PyTorch 2.6 patch ─────────────────────────────────────────
_orig = torch.load
torch.load = lambda f, *a, **kw: _orig(f, **{**kw, "weights_only": False})

try:
    import torchvision.ops as _tv
    import numpy as np
    def _nms(boxes, scores, iou_threshold):
        if boxes.numel() == 0:
            return torch.empty(0, dtype=torch.long)
        b, s = boxes.cpu().numpy(), scores.cpu().numpy()
        x1, y1, x2, y2 = b[:,0], b[:,1], b[:,2], b[:,3]
        areas = (x2-x1)*(y2-y1)
        order = s.argsort()[::-1]; keep = []
        while order.size > 0:
            i = order[0]; keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0., xx2-xx1) * np.maximum(0., yy2-yy1)
            iou_v = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            order = order[np.where(iou_v <= iou_threshold)[0] + 1]
        return torch.tensor(keep, dtype=torch.long)
    _tv.nms = _nms
    print("✔  NMS patched.")
except Exception as e:
    print(f"⚠  NMS patch skipped: {e}")

from ultralytics import YOLO

# ── Load dataset summary ──────────────────────────────────────
SUM_PATH = BASE / "dataset_summary.json"
if not SUM_PATH.exists():
    print("❌  dataset_summary.json not found!")
    print("    Run: python prepare_dataset.py  first")
    sys.exit(1)

summary = json.loads(SUM_PATH.read_text())
YAML    = summary["yaml"]
NC      = summary.get("nc", 3)

# Verify we have the 3-class dataset (not the old 2-class)
if NC != 3:
    print(f"⚠  dataset_summary.json shows nc={NC} but we need nc=3.")
    print("   Please re-run:  python prepare_dataset.py")
    sys.exit(1)

N_TRAIN = summary["train_images"]
print(f"\n  Classes : {summary['classes']}")
print(f"  nc=3 confirmed ✔")

# ── Check for existing model ──────────────────────────────────
TRAIN_DIR = BASE / "runs" / "detect" / "violation_model"
BEST_PT   = TRAIN_DIR / "weights" / "best.pt"

print(f"\n━━━  Training YOLOv8 on {N_TRAIN} images  ━━━")
print(f"  Dataset : {YAML}")
print(f"  Device  : {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}")

if BEST_PT.exists():
    print(f"\n  ✔  Trained model already exists: {BEST_PT}")
    print("  Delete that file to retrain.")
    summary["model_path"] = str(BEST_PT)
    SUM_PATH.write_text(json.dumps(summary, indent=4))
    print("  ✔  model_path saved.")
    sys.exit(0)

if N_TRAIN < 10:
    print(f"\n  ⚠  Only {N_TRAIN} training images — quality may be low.")

print(f"\n  Epochs:25  Batch:8  ImgSize:640")
print("  Estimated: ~4 min GPU  /  ~15 min CPU\n")

model = YOLO("yolov8n.pt")
model.train(
    data     = YAML,
    epochs   = 25,         # reduced for faster training
    imgsz    = 640,
    batch    = 8,
    name     = "violation_model",
    project  = str(BASE / "runs" / "detect"),
    exist_ok = True,
    device   = 0 if torch.cuda.is_available() else "cpu",
    patience = 10,         # reduced patience to match lower epoch count
    augment  = True,
    verbose  = False,
)

if not BEST_PT.exists():
    print("❌  Training failed — best.pt not found.")
    sys.exit(1)

print(f"\n  ✅  Model saved: {BEST_PT}")

# ── Validate ──────────────────────────────────────────────────
print("\n━━━  Validating  ━━━")
try:
    metrics = model.val(data=YAML, verbose=False)
    print(f"  mAP50    : {metrics.box.map50:.3f}")
    print(f"  mAP50-95 : {metrics.box.map:.3f}")
    # Per-class AP
    if hasattr(metrics.box, 'ap_class_index'):
        names = {0:"No_Helmet", 1:"With_Helmet", 2:"Rider"}
        for i, ap in enumerate(metrics.box.ap50):
            print(f"  AP50 [{names.get(i,i)}]: {ap:.3f}")
except Exception as e:
    print(f"  Validation skipped: {e}")

summary["model_path"] = str(BEST_PT)
SUM_PATH.write_text(json.dumps(summary, indent=4))
print(f"\n  ✔  model_path written to dataset_summary.json")
print("  Next: python main_fixed.py")