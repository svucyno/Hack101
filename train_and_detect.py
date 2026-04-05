"""
train_and_detect.py
===================
COMPLETE PIPELINE — run this ONE script to:

  STEP 1 → Download datasets from Kaggle
  STEP 2 → Prepare / convert to YOLOv8 format
  STEP 3 → Train YOLOv8 on No Helmet + Speeding + Triple Riding
  STEP 4 → Run detection on your video with the trained model

DATASETS USED:
  • Rider + Helmet/No-Helmet (Kaggle: aneesarom)
    kaggle.com/datasets/aneesarom/rider-with-helmet-without-helmet-number-plate
  • Highway Traffic Videos for Speeding (Kaggle: aryashah2k)
    kaggle.com/datasets/aryashah2k/highway-traffic-videos-dataset

PREREQUISITES (run once):
    pip install kaggle ultralytics deep-sort-realtime opencv-python yt-dlp

HOW TO RUN:
    1. Get your kaggle.json from https://www.kaggle.com/settings → API → Create New Token
    2. Place kaggle.json in  C:\\Users\\<YourName>\\.kaggle\\kaggle.json
    3. python train_and_detect.py
"""

import os, sys, json, shutil, math, time, subprocess, zipfile, glob
import cv2, numpy as np
from datetime import datetime
from pathlib import Path

BASE = Path(os.getcwd())

# ─────────────────────────────────────────────
def log(msg):  print(f"\n{'═'*55}\n  {msg}\n{'═'*55}")
def ok(msg):   print(f"  ✔  {msg}")
def info(msg): print(f"  ℹ  {msg}")
def warn(msg): print(f"  ⚠  {msg}")
def err(msg):  print(f"  ✖  {msg}"); sys.exit(1)

# ══════════════════════════════════════════════
#  STEP 1 — DOWNLOAD DATASETS FROM KAGGLE
# ══════════════════════════════════════════════
log("STEP 1 — Downloading Kaggle datasets")

KAGGLE_JSON = Path.home() / ".kaggle" / "kaggle.json"
if not KAGGLE_JSON.exists():
    err(
        "kaggle.json not found!\n"
        "  1. Go to https://www.kaggle.com/settings\n"
        "  2. Click API → Create New Token\n"
        "  3. Save the downloaded kaggle.json to:\n"
        f"     {KAGGLE_JSON}"
    )

try:
    import kaggle
    ok("kaggle package ready")
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle", "-q"])
    import kaggle
    ok("kaggle installed")

DATASETS = {
    # Dataset slug                                       → local folder name
    "aneesarom/rider-with-helmet-without-helmet-number-plate": "helmet_dataset",
    "aryashah2k/highway-traffic-videos-dataset":               "highway_dataset",
}

RAW_DIR = BASE / "raw_datasets"
RAW_DIR.mkdir(exist_ok=True)

for slug, folder in DATASETS.items():
    dest = RAW_DIR / folder
    if dest.exists() and any(dest.iterdir()):
        ok(f"Already downloaded: {folder}")
        continue
    dest.mkdir(exist_ok=True)
    print(f"  ⬇  Downloading {slug} …")
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", slug,
             "-p", str(dest), "--unzip"],
            check=True
        )
        ok(f"Saved to: {dest}")
    except Exception as e:
        warn(f"Download failed for {slug}: {e}")
        info("Make sure your kaggle.json is valid and you've accepted dataset terms.")

# ══════════════════════════════════════════════
#  STEP 2 — PREPARE YOLO FORMAT DATASET
# ══════════════════════════════════════════════
log("STEP 2 — Preparing YOLOv8 format dataset")

"""
Target class map (what we train on):
    0 = No_Helmet      (from helmet_dataset)
    1 = Triple_Riding  (synthetic from rider count ≥ 3)
    2 = Speeding       (only detected at inference via tracker — not trained)

YOLOv8 label format per line:
    class_id  cx  cy  w  h   (all normalised 0-1)
"""

YOLO_DIR   = BASE / "yolo_dataset"
IMG_TRAIN  = YOLO_DIR / "images" / "train"
IMG_VAL    = YOLO_DIR / "images" / "val"
LBL_TRAIN  = YOLO_DIR / "labels" / "train"
LBL_VAL    = YOLO_DIR / "labels" / "val"
for d in [IMG_TRAIN, IMG_VAL, LBL_TRAIN, LBL_VAL]:
    d.mkdir(parents=True, exist_ok=True)

HELMET_SRC = RAW_DIR / "helmet_dataset"

# ── Discover label files in the Kaggle helmet dataset ─────────────────────────
# The aneesarom dataset uses YOLO-style .txt labels
# Classes in the original dataset:
#   0 = Rider   1 = With Helmet   2 = Without Helmet   3 = Number Plate
# We remap:  Without Helmet (2) → our class 0 (No_Helmet)
#            All others dropped (we only want No_Helmet)

CLASS_REMAP = {2: 0}   # original class 2 (Without Helmet) → our class 0

def remap_label_file(src_label: Path, dst_label: Path):
    """Read a YOLO .txt label, keep only remapped classes, write to dst."""
    lines_out = []
    try:
        with open(src_label) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5: continue
                orig_cls = int(parts[0])
                if orig_cls in CLASS_REMAP:
                    new_cls = CLASS_REMAP[orig_cls]
                    lines_out.append(f"{new_cls} " + " ".join(parts[1:]))
    except Exception:
        return False
    if not lines_out:
        return False   # no relevant labels in this image → skip
    with open(dst_label, "w") as f:
        f.write("\n".join(lines_out) + "\n")
    return True

# Walk helmet dataset and collect image/label pairs
all_pairs = []
for img_path in sorted(HELMET_SRC.rglob("*.jpg")) + \
                sorted(HELMET_SRC.rglob("*.jpeg")) + \
                sorted(HELMET_SRC.rglob("*.png")):
    # Look for matching label (.txt same stem)
    lbl_path = img_path.with_suffix(".txt")
    if not lbl_path.exists():
        # Try labels/ subfolder at same level
        lbl_path = img_path.parent.parent / "labels" / (img_path.stem + ".txt")
    if not lbl_path.exists():
        lbl_path = img_path.parent / "obj_train_data" / (img_path.stem + ".txt")
    if lbl_path.exists():
        all_pairs.append((img_path, lbl_path))

info(f"Found {len(all_pairs)} labelled images in helmet dataset")

if len(all_pairs) == 0:
    warn("No labelled images found — checking folder structure…")
    for root, dirs, files in os.walk(HELMET_SRC):
        info(f"  {root}: {len(files)} files")

# 80/20 split
import random; random.seed(42); random.shuffle(all_pairs)
split = int(len(all_pairs) * 0.8)
train_pairs = all_pairs[:split]
val_pairs   = all_pairs[split:]

copied = 0
for pairs, img_dir, lbl_dir in [
    (train_pairs, IMG_TRAIN, LBL_TRAIN),
    (val_pairs,   IMG_VAL,   LBL_VAL),
]:
    for img_src, lbl_src in pairs:
        dst_lbl = lbl_dir / (img_src.stem + ".txt")
        if remap_label_file(lbl_src, dst_lbl):
            dst_img = img_dir / img_src.name
            shutil.copy2(img_src, dst_img)
            copied += 1

ok(f"Prepared {copied} images with No_Helmet labels ({split} train / {len(val_pairs)} val)")

# ── Write data.yaml ────────────────────────────────────────────────────────────
YAML_PATH = YOLO_DIR / "data.yaml"
yaml_content = f"""path: {YOLO_DIR.as_posix()}
train: images/train
val:   images/val

nc: 2
names:
  0: No_Helmet
  1: Triple_Riding
"""
YAML_PATH.write_text(yaml_content)
ok(f"data.yaml written → {YAML_PATH}")

# ══════════════════════════════════════════════
#  STEP 3 — TRAIN YOLOv8 MODEL
# ══════════════════════════════════════════════
log("STEP 3 — Training YOLOv8 on No_Helmet dataset")

import torch
_orig_load = torch.load
torch.load = lambda f, *a, **kw: _orig_load(f, **{**kw, "weights_only": False})

try:
    import torchvision.ops as _tv
    def _nms(boxes, scores, iou_threshold):
        if boxes.numel() == 0: return torch.empty(0, dtype=torch.long)
        b,s=boxes.cpu().numpy(),scores.cpu().numpy()
        x1,y1,x2,y2=b[:,0],b[:,1],b[:,2],b[:,3]
        areas=(x2-x1)*(y2-y1); order=s.argsort()[::-1]; keep=[]
        while order.size>0:
            i=order[0]; keep.append(i)
            xx1=np.maximum(x1[i],x1[order[1:]]); yy1=np.maximum(y1[i],y1[order[1:]])
            xx2=np.minimum(x2[i],x2[order[1:]]); yy2=np.minimum(y2[i],y2[order[1:]])
            inter=np.maximum(0.,xx2-xx1)*np.maximum(0.,yy2-yy1)
            iou_v=inter/(areas[i]+areas[order[1:]]-inter+1e-6)
            order=order[np.where(iou_v<=iou_threshold)[0]+1]
        return torch.tensor(keep,dtype=torch.long)
    _tv.nms=_nms
except Exception: pass

from ultralytics import YOLO

TRAINED_MODEL = BASE / "runs" / "detect" / "violation_model" / "weights" / "best.pt"

if TRAINED_MODEL.exists():
    ok(f"Trained model already exists: {TRAINED_MODEL}")
else:
    if copied < 10:
        warn(f"Only {copied} training images prepared — training may be poor.")
        warn("Consider downloading more data or checking dataset structure.")

    model_train = YOLO("yolov8n.pt")   # start from pretrained nano weights
    info("Training for 50 epochs on No_Helmet classes…")
    info("This will take ~5-30 minutes depending on your GPU/CPU.")

    model_train.train(
        data    = str(YAML_PATH),
        epochs  = 50,
        imgsz   = 640,
        batch   = 8,          # reduce to 4 if you get OOM errors
        name    = "violation_model",
        project = str(BASE / "runs" / "detect"),
        exist_ok= True,
        device  = 0 if torch.cuda.is_available() else "cpu",
        patience= 10,         # early stop if no improvement for 10 epochs
        verbose = False,
    )
    ok(f"Training complete → {TRAINED_MODEL}")

# ══════════════════════════════════════════════
#  STEP 4 — RUN DETECTION ON VIDEO
# ══════════════════════════════════════════════
log("STEP 4 — Running violation detection on video")

from deep_sort_realtime.deepsort_tracker import DeepSort
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ── Model setup ───────────────────────────────────────────────────────────────
violation_model = YOLO(str(TRAINED_MODEL)) if TRAINED_MODEL.exists() else None
coco_model      = YOLO("yolov8n.pt")

if violation_model:
    ok(f"Using trained violation model: {TRAINED_MODEL.name}")
else:
    warn("No trained model found — using COCO fallback (rider overlap logic only)")

tracker = DeepSort(max_age=40, n_init=5, max_cosine_distance=0.35)

# ── Find video ────────────────────────────────────────────────────────────────
VIDEO_CANDIDATES = [
    BASE/"sample_video1.mp4", BASE/"sample_video2.mp4",
    BASE/"sample_video3.mp4", BASE/"sample_video.mp4",
]
# Also check highway dataset for MP4 files
for mp4 in (RAW_DIR / "highway_dataset").rglob("*.mp4"):
    VIDEO_CANDIDATES.insert(0, mp4)
    break

VIDEO_FILE = next((v for v in VIDEO_CANDIDATES if Path(v).exists()), None)
if not VIDEO_FILE:
    err(
        "No video file found!\n"
        "  Place sample_video1.mp4 in: " + str(BASE) + "\n"
        "  OR the Kaggle highway dataset contains MP4 files to use."
    )

# ── Output paths ──────────────────────────────────────────────────────────────
JSON_PATH = BASE / "violations" / "violations.json"
IMAGE_DIR = BASE / "evidence" / "images"
OUT_VIDEO = BASE / "output_annotated.mp4"
JSON_PATH.parent.mkdir(exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# ── Load existing records ─────────────────────────────────────────────────────
violations_db = []
if JSON_PATH.exists():
    try:
        violations_db = json.loads(JSON_PATH.read_text())
        ok(f"Loaded {len(violations_db)} existing records")
    except Exception: pass

logged_violations = {}
for v in violations_db:
    logged_violations.setdefault(v["track_id"], set()).add(v["type"])

# ── Per-track state ───────────────────────────────────────────────────────────
speed_hist   = {}
prev_center  = {}
speed_consec = {}

# ── Config ────────────────────────────────────────────────────────────────────
CONF_THRESH      = 0.45
COCO_CONF        = 0.45
IOU_RIDER        = 0.10
TRIPLE_MIN       = 3
SPEED_PX_FRAME   = 15
SPEED_SMOOTH_WIN = 6
SPEED_CONFIRM_N  = 4
MIN_BOX_AREA     = 1800
MAX_AR           = 4.0
MIN_AR           = 0.25
CLS_PERSON       = 0
CLS_BIKE         = 3
CLS_CAR          = 2

VCOLOURS = {
    "No_Helmet":     (0,  140, 255),
    "Triple_Riding": (0,  0,   255),
    "Speeding":      (200, 0,  255),
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def clamp(v,lo,hi): return max(lo,min(hi,v))

def valid_box(x1,y1,x2,y2):
    w,h=x2-x1,y2-y1
    if w<=0 or h<=0: return False
    if w*h < MIN_BOX_AREA: return False
    ar=w/h
    return MIN_AR<=ar<=MAX_AR

def box_iou(a,b):
    ax1,ay1,ax2,ay2=float(a[0]),float(a[1]),float(a[2]),float(a[3])
    bx1,by1,bx2,by2=float(b[0]),float(b[1]),float(b[2]),float(b[3])
    ix1,iy1=max(ax1,bx1),max(ay1,by1); ix2,iy2=min(ax2,bx2),min(ay2,by2)
    inter=max(0.,ix2-ix1)*max(0.,iy2-iy1)
    if inter==0: return 0.
    return inter/((ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter+1e-6)

def get_riders(people,bx):
    x1,y1,x2,y2=bx
    return [p for p in people if
            ((x1<=(p[0]+p[2])/2<=x2 and y1<=(p[1]+p[3])/2<=y2)
             or box_iou(bx,[p[0],p[1],p[2],p[3]])>IOU_RIDER)]

def update_speed(tid,cx,cy):
    prev=prev_center.get(tid); prev_center[tid]=(cx,cy)
    if prev is None: speed_consec[tid]=0; return 0.,0
    raw=math.hypot(cx-prev[0],cy-prev[1])
    h=speed_hist.setdefault(tid,[])
    h.append(raw)
    if len(h)>SPEED_SMOOTH_WIN: h.pop(0)
    avg=sum(h)/len(h)
    speed_consec[tid] = speed_consec.get(tid,0)+1 if avg>SPEED_PX_FRAME else 0
    return avg, speed_consec[tid]

def already_logged(tid,vtype): return vtype in logged_violations.get(tid,set())
def mark_logged(tid,vtype):    logged_violations.setdefault(tid,set()).add(vtype)

def save_violation(frame,box,vtype,tid):
    if already_logged(tid,vtype): return
    x1,y1,x2,y2=map(int,box)
    ev=frame.copy(); col=VCOLOURS.get(vtype,(0,255,0))
    cv2.rectangle(ev,(x1,max(y1-36,0)),(x2,y1),col,-1)
    cv2.putText(ev,f"{vtype.replace('_',' ')}  ID:{tid}",
                (x1+4,max(y1-10,14)),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
    cv2.rectangle(ev,(x1,y1),(x2,y2),col,3)
    cv2.putText(ev,datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                (6,ev.shape[0]-8),cv2.FONT_HERSHEY_SIMPLEX,0.48,(220,220,220),1)
    img_name=f"V_{tid}_{vtype}_{int(time.time()*1000)}.jpg"
    cv2.imwrite(str(IMAGE_DIR/img_name),ev)
    entry={"track_id":tid,"type":vtype,
           "time":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"image":img_name}
    violations_db.append(entry)
    mark_logged(tid,vtype)
    tmp=str(JSON_PATH)+".tmp"
    with open(tmp,"w") as f: json.dump(violations_db,f,indent=4)
    os.replace(tmp,str(JSON_PATH))
    print(f"  📄  [{vtype.replace('_',' ')}]  ID:{tid}")

def draw_label(frame,x1,y1,x2,y2,label,col):
    font=cv2.FONT_HERSHEY_SIMPLEX; fs=0.48; th=1
    (lw,lh),_=cv2.getTextSize(label,font,fs,th)
    pad=4
    if y1>lh+pad+4:
        bx1,by1,bx2,by2=x1,y1-lh-pad-2,x1+lw+pad,y1
        tx,ty=x1+2,y1-pad
    else:
        bx1,by1,bx2,by2=x1,y1,x1+lw+pad,y1+lh+pad+2
        tx,ty=x1+2,y1+lh+2
    overlay=frame.copy()
    cv2.rectangle(overlay,(bx1,by1),(bx2,by2),(0,0,0),-1)
    cv2.addWeighted(overlay,0.65,frame,0.35,0,frame)
    cv2.putText(frame,label,(tx,ty),font,fs,col,th,cv2.LINE_AA)

# ── GUI probe ─────────────────────────────────────────────────────────────────
_gui=True
try:
    cv2.imshow("__p__",np.zeros((4,4,3),dtype=np.uint8))
    cv2.waitKey(1); cv2.destroyWindow("__p__")
except Exception: _gui=False

# ── Open video ────────────────────────────────────────────────────────────────
cap=cv2.VideoCapture(str(VIDEO_FILE))
if not cap.isOpened(): err(f"Cannot open: {VIDEO_FILE}")
fps    =cap.get(cv2.CAP_PROP_FPS) or 25
fh     =int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fw     =int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
ok(f"Video: {Path(VIDEO_FILE).name} | {fw}x{fh} @ {fps:.0f}fps")

writer=None
if not _gui:
    writer=cv2.VideoWriter(str(OUT_VIDEO),cv2.VideoWriter_fourcc(*"mp4v"),fps,(fw,fh))
    info(f"Headless → saving to {OUT_VIDEO}")

frame_no=0
print("🚦  Running…  Press Q to quit.\n")

# ── Main detection loop ───────────────────────────────────────────────────────
while cap.isOpened():
    ret,frame=cap.read()
    if not ret: break
    frame_no+=1

    # COCO detections → bikes, cars, people
    coco_raw=coco_model(frame,verbose=False,conf=COCO_CONF)[0].boxes.data.tolist()
    people =[d for d in coco_raw if int(d[5])==CLS_PERSON]
    bikes_r=[d for d in coco_raw if int(d[5])==CLS_BIKE]
    cars_r =[d for d in coco_raw if int(d[5])==CLS_CAR]

    # Trained model detections → {(x1,y1,x2,y2): [vtype,...]}
    model_dets={}
    if violation_model:
        for det in violation_model(frame,verbose=False,conf=CONF_THRESH)[0].boxes.data.tolist():
            x1,y1,x2,y2,conf,cls=det
            # Map trained class index to name
            cls_names={0:"No_Helmet",1:"Triple_Riding"}
            vtype=cls_names.get(int(cls),"")
            if not vtype: continue
            bx1,by1,bx2,by2=(clamp(int(x1),0,fw),clamp(int(y1),0,fh),
                              clamp(int(x2),0,fw),clamp(int(y2),0,fh))
            if not valid_box(bx1,by1,bx2,by2): continue
            model_dets.setdefault((bx1,by1,bx2,by2),[]).append(vtype)

    # DeepSort — bikes + cars only (NO persons)
    ds_in=[]
    for d in bikes_r+cars_r:
        x1,y1,x2,y2,conf,cls=d
        if not valid_box(int(x1),int(y1),int(x2),int(y2)): continue
        ds_in.append(([x1,y1,x2-x1,y2-y1],conf,int(cls)))
    tracks=tracker.update_tracks(ds_in,frame=frame)

    active_tids=set()
    for track in tracks:
        if not track.is_confirmed(): continue
        tid=int(track.track_id); ltrb=track.to_ltrb()
        x1=clamp(int(ltrb[0]),0,fw); y1=clamp(int(ltrb[1]),0,fh)
        x2=clamp(int(ltrb[2]),0,fw); y2=clamp(int(ltrb[3]),0,fh)
        if not valid_box(x1,y1,x2,y2): continue

        cls_id=getattr(track,"det_class",CLS_BIKE)
        is_bike=(cls_id==CLS_BIKE)
        cx=(x1+x2)//2; cy=(y1+y2)//2
        active_tids.add(tid)

        violations_now=[]

        # A. Match trained model detections to this track
        for (dx1,dy1,dx2,dy2),vtypes in model_dets.items():
            if box_iou([x1,y1,x2,y2],[dx1,dy1,dx2,dy2])>0.25:
                for v in vtypes:
                    if v not in violations_now: violations_now.append(v)

        # B. COCO fallback rider logic (when no trained model)
        if not violation_model and is_bike:
            riders=get_riders(people,[x1,y1,x2,y2])
            if len(riders)>=TRIPLE_MIN and "Triple_Riding" not in violations_now:
                violations_now.append("Triple_Riding")
            if len(riders)>=1 and "No_Helmet" not in violations_now:
                violations_now.append("No_Helmet")

        # C. Speeding (tracker-based, always active)
        avg_spd,consec=update_speed(tid,cx,cy)
        if consec>=SPEED_CONFIRM_N and "Speeding" not in violations_now:
            violations_now.append("Speeding")

        # D. Save new violations
        for vtype in set(violations_now):
            if not already_logged(tid,vtype):
                save_violation(frame.copy(),[x1,y1,x2,y2],vtype,tid)

        # E. Draw
        new_v =set(violations_now)
        past_v=logged_violations.get(tid,set())

        if "Triple_Riding" in new_v: box_col=(0,0,255)
        elif "No_Helmet" in new_v:   box_col=(0,140,255)
        elif "Speeding" in new_v:    box_col=(200,0,255)
        elif past_v:                 box_col=(0,120,255)
        else:                        box_col=(0,210,60)

        cv2.rectangle(frame,(x1,y1),(x2,y2),box_col,2)

        if violations_now:
            lbl=", ".join(v.replace("_"," ") for v in sorted(set(violations_now)))
        elif past_v:
            lbl="!"+ "+".join(v.replace("_"," ") for v in sorted(past_v))
        else:
            lbl="OK"
        draw_label(frame,x1,y1,x2,y2,f"ID:{tid} {lbl}",box_col)

        # Rider count badge
        if is_bike:
            riders_now=get_riders(people,[x1,y1,x2,y2])
            if riders_now:
                cv2.putText(frame,f"R:{len(riders_now)}",(x2-44,y1+16),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),2)

    # HUD
    cv2.rectangle(frame,(0,fh-26),(fw,fh),(15,15,15),-1)
    cv2.putText(frame,
        f"Frame:{frame_no}  Violations:{len(violations_db)}  Vehicles:{len(active_tids)}",
        (8,fh-8),cv2.FONT_HERSHEY_SIMPLEX,0.48,(200,200,200),1)

    legend=[("No Helmet",(0,140,255)),("Triple Riding",(0,0,255)),
            ("Speeding",(200,0,255)),("Clean",(0,210,60))]
    for i,(lbl,lc) in enumerate(legend):
        lx=fw-148; ly=8+i*20
        cv2.rectangle(frame,(lx,ly),(lx+12,ly+12),lc,-1)
        cv2.putText(frame,lbl,(lx+16,ly+11),cv2.FONT_HERSHEY_SIMPLEX,0.42,lc,1)

    if _gui:
        cv2.imshow("SIGCE AI Traffic System",frame)
        if cv2.waitKey(1)&0xFF==ord('q'): break
    else:
        writer.write(frame)
        if frame_no%150==0: print(f"  ⏳ frame {frame_no} | violations:{len(violations_db)}")

cap.release()
if writer: writer.release(); ok(f"Saved → {OUT_VIDEO}")
if _gui:   cv2.destroyAllWindows()

print(f"\n🎉  Done!  Violations: {len(violations_db)}")
print(f"    JSON   → {JSON_PATH}")
print(f"    Images → {IMAGE_DIR}")
print(f"    Model  → {TRAINED_MODEL}")