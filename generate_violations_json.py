"""
adding generate_violations_json.py  —  OFFLINE BATCH MODE  (CORRECTED v3)
====================================================================
IMPORTANT — READ BEFORE RUNNING:
  If you see "Model has 2 classes" warning, the OLD model is still loaded.
  Fix:
    1. Delete:  runs\detect\violation_model\weights\best.pt
    2. Run:     python train_model.py
    3. Run:     python generate_violations_json.py

  The 3-class model is REQUIRED for correct helmet detection.
  Class layout that must be in data.yaml / model:
    0 = No_Helmet    → VIOLATION
    1 = With_Helmet  → clean  (blocks false No_Helmet labels)
    2 = Rider        → used for triple-riding

HOW FALSE "No Helmet" IS PREVENTED:
  Even when the model predicts No_Helmet, we apply these guards:
    G1. No_Helmet confidence must be >= NH_CONF_MIN (0.65 for 3-class,
        0.80 for old 2-class model as stricter fallback)
    G2. No_Helmet box height <= 50% of Rider box height (head is small)
    G3. No_Helmet center must be in top 55% of Rider box (head is at top)
    G4. If a With_Helmet detection overlaps the same spot -> BLOCK the flag
    G5. If No_Helmet conf - With_Helmet conf < 0.20 -> BLOCK (model unsure)

RUN:
    python generate_violations_json.py

OUTPUT:
    violations/violations.json
    evidence/images/V_*.jpg
"""

import os, json, time, sys, cv2
from pathlib import Path
from datetime import datetime

BASE = Path(os.getcwd())

SUMMARY_PATH = BASE / "dataset_summary.json"
JSON_PATH    = BASE / "violations" / "violations.json"
IMAGE_DIR    = BASE / "evidence"   / "images"
JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════
MODEL_CONF           = 0.50
NH_CONF_MIN          = 0.65   # raised to 0.80 automatically for 2-class model
NH_MAX_HEIGHT_RATIO  = 0.50
NH_MAX_CENTER_Y_FRAC = 0.55
NH_RIDER_IOU_MIN     = 0.10
WH_BLOCK_IOU         = 0.25
WH_NH_CONF_GAP_MIN   = 0.20
TRIPLE_MIN           = 3
CLUSTER_DIST_FRAC    = 0.25
MIN_BOX_AREA         = 800

CLS_NO_HELMET   = 0
CLS_WITH_HELMET = 1
CLS_RIDER       = 2

VCOLOURS = {
    "No_Helmet":     (0,  140, 255),
    "Triple_Riding": (0,  0,   255),
}

# ══════════════════════════════════════════════════════════════
#  LOAD MODEL
# ══════════════════════════════════════════════════════════════
import torch
_orig = torch.load
torch.load = lambda f, *a, **kw: _orig(f, **{**kw, "weights_only": False})

from ultralytics import YOLO

def find_model():
    if SUMMARY_PATH.exists():
        try:
            d  = json.loads(SUMMARY_PATH.read_text())
            mp = d.get("model_path", "").strip()
            if mp and Path(mp).exists():
                return Path(mp)
        except Exception:
            pass
    detect_dir = BASE / "runs" / "detect"
    if detect_dir.exists():
        cands = sorted(detect_dir.rglob("best.pt"),
                       key=lambda x: x.stat().st_mtime, reverse=True)
        if cands:
            return cands[0]
    return None

MODEL_PATH = find_model()
if not MODEL_PATH:
    print("No trained model found.")
    print("Run: python prepare_dataset.py  then  python train_model.py")
    sys.exit(1)

model = YOLO(str(MODEL_PATH))
model.overrides["task"] = "detect"
print(f"Model loaded: {MODEL_PATH.name}")

nc = getattr(model.model, 'nc', None)
IS_3CLASS = (nc == 3)
IS_2CLASS = (nc == 2)

if IS_3CLASS:
    print("3-class model confirmed (No_Helmet / With_Helmet / Rider)")
elif IS_2CLASS:
    print()
    print("=" * 60)
    print("  OLD 2-CLASS MODEL DETECTED")
    print("  This model does NOT know what a helmet looks like.")
    print("  It will misclassify helmeted riders as No_Helmet.")
    print()
    print("  TO FIX:")
    print("    1. Delete runs\\detect\\violation_model\\weights\\best.pt")
    print("    2. python train_model.py")
    print("    3. python generate_violations_json.py")
    print()
    print("  Running in STRICT mode (conf >= 0.80) to reduce false positives.")
    print("=" * 60)
    print()
    NH_CONF_MIN = 0.80
else:
    print(f"Model has {nc} classes (unexpected). Proceeding cautiously.")

# ══════════════════════════════════════════════════════════════
#  FIND IMAGES
# ══════════════════════════════════════════════════════════════
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

val_dir   = BASE / "yolo_dataset" / "images" / "val"
train_dir = BASE / "yolo_dataset" / "images" / "train"

if val_dir.exists() and any(p for p in val_dir.iterdir()
                            if p.suffix.lower() in IMG_EXTS):
    img_dir = val_dir
    print(f"Using val split: {val_dir}")
else:
    img_dir = train_dir
    print(f"Using train split: {train_dir}")

image_paths = sorted(
    p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXTS
)
if not image_paths:
    print(f"No images found in {img_dir}")
    sys.exit(1)

print(f"{len(image_paths)} images to process\n")

# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def valid_box(x1, y1, x2, y2):
    w, h = x2 - x1, y2 - y1
    return w > 0 and h > 0 and w * h >= MIN_BOX_AREA

def box_iou(a, b):
    ax1,ay1,ax2,ay2 = float(a[0]),float(a[1]),float(a[2]),float(a[3])
    bx1,by1,bx2,by2 = float(b[0]),float(b[1]),float(b[2]),float(b[3])
    ix1 = max(ax1,bx1); iy1 = max(ay1,by1)
    ix2 = min(ax2,bx2); iy2 = min(ay2,by2)
    inter = max(0., ix2-ix1) * max(0., iy2-iy1)
    if inter == 0:
        return 0.
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter + 1e-6
    return inter / union

def no_helmet_spatial_ok(nh_box, rider_box):
    if box_iou(nh_box, rider_box) < NH_RIDER_IOU_MIN:
        return False
    rider_h = rider_box[3] - rider_box[1]
    if rider_h <= 0:
        return False
    nh_h  = nh_box[3] - nh_box[1]
    nh_cy = (nh_box[1] + nh_box[3]) / 2.0
    if nh_h > NH_MAX_HEIGHT_RATIO * rider_h:
        return False
    rel_y = (nh_cy - rider_box[1]) / rider_h
    if rel_y > NH_MAX_CENTER_Y_FRAC:
        return False
    return True

def with_helmet_blocks(wh_dets_conf, nh_box, nh_conf):
    for (wx1, wy1, wx2, wy2, wh_conf) in wh_dets_conf:
        if box_iou(nh_box, [wx1,wy1,wx2,wy2]) > WH_BLOCK_IOU:
            if (nh_conf - wh_conf) < WH_NH_CONF_GAP_MIN:
                return True
    return False

def cluster_riders(rider_dets, fw, fh):
    if not rider_dets:
        return []
    diag = (fw ** 2 + fh ** 2) ** 0.5
    used = [False] * len(rider_dets)
    clusters = []
    for i, r in enumerate(rider_dets):
        if used[i]:
            continue
        cluster = [r]
        used[i] = True
        cx1 = (r[0]+r[2])/2.0; cy1 = (r[1]+r[3])/2.0
        for j, r2 in enumerate(rider_dets):
            if used[j]:
                continue
            cx2 = (r2[0]+r2[2])/2.0; cy2 = (r2[1]+r2[3])/2.0
            dist = ((cx1-cx2)**2 + (cy1-cy2)**2)**0.5
            if dist < diag * CLUSTER_DIST_FRAC or box_iou(r, r2) > 0.05:
                cluster.append(r2)
                used[j] = True
        clusters.append(cluster)
    return clusters

def save_evidence(img, annotations, ev_name):
    out = img.copy()
    h, w = out.shape[:2]
    for (x1, y1, x2, y2, label, col, conf_val) in annotations:
        cv2.rectangle(out, (x1, max(y1-38,0)), (x2, y1), col, -1)
        txt = f"{label}  {conf_val:.0%}" if conf_val is not None else label
        cv2.putText(out, txt, (x1+4, max(y1-10,14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
        cv2.rectangle(out, (x1,y1), (x2,y2), col, 3)
    cv2.putText(out, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                (6, h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)
    cv2.imwrite(str(IMAGE_DIR / ev_name), out)

# ══════════════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════════════
violations_db = []
processed = skipped = 0

print("Processing images...")
print("-" * 42)

for img_path in image_paths:
    img = cv2.imread(str(img_path))
    if img is None:
        skipped += 1
        continue

    fh_img, fw_img = img.shape[:2]
    processed += 1

    try:
        results = model(img, verbose=False, conf=MODEL_CONF, task="detect")
        dets = results[0].boxes.data.tolist()
    except Exception as e:
        print(f"  {img_path.name}: {e}")
        skipped += 1
        continue

    nh_dets    = []
    wh_dets    = []
    rider_dets = []

    for det in dets:
        x1,y1,x2,y2,conf,cls = det
        cls  = int(cls)
        bx1  = clamp(int(x1), 0, fw_img)
        by1  = clamp(int(y1), 0, fh_img)
        bx2  = clamp(int(x2), 0, fw_img)
        by2  = clamp(int(y2), 0, fh_img)
        if not valid_box(bx1,by1,bx2,by2):
            continue
        conf = float(conf)
        if   cls == CLS_NO_HELMET:
            nh_dets.append((bx1,by1,bx2,by2,conf))
        elif cls == CLS_WITH_HELMET:
            wh_dets.append((bx1,by1,bx2,by2,conf))
        elif cls == CLS_RIDER:
            rider_dets.append((bx1,by1,bx2,by2))

    # For 2-class model: class 1 = Rider (not With_Helmet)
    if IS_2CLASS:
        for (wx1,wy1,wx2,wy2,wc) in wh_dets:
            rider_dets.append((wx1,wy1,wx2,wy2))
        wh_dets = []

    image_violations = []

    # ── A. No_Helmet ──────────────────────────────────────────
    for (nhx1,nhy1,nhx2,nhy2,nh_conf) in nh_dets:
        nh_box = [nhx1,nhy1,nhx2,nhy2]

        # G1: confidence gate
        if nh_conf < NH_CONF_MIN:
            continue

        # G4+G5: mutual exclusion (3-class only)
        if IS_3CLASS and with_helmet_blocks(wh_dets, nh_box, nh_conf):
            continue

        # Find best matching Rider box
        matched_rider = None
        best_iou = 0.0
        for (rx1,ry1,rx2,ry2) in rider_dets:
            rider_box = [rx1,ry1,rx2,ry2]
            iou = box_iou(nh_box, rider_box)
            if iou > best_iou:
                best_iou = iou
                matched_rider = rider_box

        # G2+G3: spatial guard
        if matched_rider and not no_helmet_spatial_ok(nh_box, matched_rider):
            matched_rider = None

        if matched_rider:
            image_violations.append(("No_Helmet", matched_rider, nh_conf))
        elif nh_conf >= 0.80:
            # Very high confidence with no rider box — still flag
            image_violations.append(("No_Helmet", nh_box, nh_conf))

    # ── B. Triple Riding ──────────────────────────────────────
    if len(rider_dets) >= TRIPLE_MIN:
        clusters = cluster_riders(rider_dets, fw_img, fh_img)
        for cluster in clusters:
            if len(cluster) >= TRIPLE_MIN:
                all_x1 = min(r[0] for r in cluster)
                all_y1 = min(r[1] for r in cluster)
                all_x2 = max(r[2] for r in cluster)
                all_y2 = max(r[3] for r in cluster)
                image_violations.append(
                    ("Triple_Riding", [all_x1,all_y1,all_x2,all_y2], None)
                )

    if not image_violations:
        continue

    # Build evidence image
    annotations = []
    for vtype, vbox, vconf in image_violations:
        col = VCOLOURS.get(vtype, (0,255,0))
        annotations.append((vbox[0],vbox[1],vbox[2],vbox[3],
                             vtype.replace("_"," "), col, vconf))
    # Draw With_Helmet in green for reference (3-class only)
    if IS_3CLASS:
        for (wx1,wy1,wx2,wy2,wc) in wh_dets:
            annotations.append((wx1,wy1,wx2,wy2,
                                 f"With Helmet {wc:.0%}",
                                 (0,200,60), None))

    ts      = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ev_name = f"V_{processed}_{img_path.stem}_{int(time.time()*1000)}.jpg"
    save_evidence(img, annotations, ev_name)

    for vtype, vbox, vconf in image_violations:
        entry = {
            "track_id":   processed,
            "source_img": img_path.name,
            "type":       vtype,
            "time":       ts,
            "image":      ev_name,
            "confidence": round(vconf,3) if vconf is not None else None,
            "box":        [int(c) for c in vbox],
        }
        violations_db.append(entry)
        c = f"  conf={vconf:.2f}" if vconf else ""
        print(f"  [{vtype.replace('_',' '):<14}]  {img_path.name}{c}")

    if processed % 20 == 0:
        print(f"  ... {processed}/{len(image_paths)} | total: {len(violations_db)}")

# ══════════════════════════════════════════════════════════════
#  WRITE violations.json
# ══════════════════════════════════════════════════════════════
tmp = str(JSON_PATH) + ".tmp"
with open(tmp, "w") as f:
    json.dump(violations_db, f, indent=4)
os.replace(tmp, str(JSON_PATH))

type_counts: dict = {}
for v in violations_db:
    type_counts[v["type"]] = type_counts.get(v["type"], 0) + 1

print(f"\n{'='*42}")
print(f"Done!")
print(f"  Images processed : {processed}")
print(f"  Violations found : {len(violations_db)}")
for vt, cnt in sorted(type_counts.items()):
    print(f"    {vt.replace('_',' '):<22}: {cnt}")
print(f"  JSON   -> {JSON_PATH}")
print(f"  Images -> {IMAGE_DIR}")

if IS_2CLASS:
    print()
    print("  REMINDER: You are on the old 2-class model.")
    print("  For correct results:")
    print("    del runs\\detect\\violation_model\\weights\\best.pt")
    print("    python train_model.py")
    print("    python generate_violations_json.py")

print(f"\n  View: streamlit run dashboard/app.py")
