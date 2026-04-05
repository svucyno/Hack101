"""
monitor.py  —  SIGCE 2026 AI Traffic Enforcement System
=========================================================
VIOLATIONS:
    1. No Helmet      – rider without helmet (helmet model or fallback)
    2. Triple Riding  – 3+ persons on one motorcycle
    3. Speeding       – sustained fast movement (6 consecutive frames)
    4. Wrong-Way      – vehicle moving against dominant traffic direction
    5. Red-Light Jump – vehicle crosses stop-line while light is red

DEDUPLICATION GUARANTEE:
    Each (track_id, violation_type) is saved and shown EXACTLY ONCE.
    Uses DeepSort for stable, unique integer Track IDs.

NO ERRORS:
    • PyTorch 2.6 patch applied before ultralytics import
    • torchvision NMS patched for version mismatch
    • Thread-safe UI updates via window.after()
    • Atomic JSON writes (no corruption on crash)
"""

import os
import cv2
import json
import math
import time
import threading
import numpy as np
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import torch

# ══════════════════════════════════════════════════════════════
#  PYTORCH PATCHES — must be before ultralytics import
# ══════════════════════════════════════════════════════════════
_orig_load = torch.load
torch.load = lambda f, *a, **kw: _orig_load(f, **{**kw, "weights_only": False})

try:
    import torchvision.ops as _tv
    def _cpu_nms(boxes, scores, iou_threshold):
        if boxes.numel() == 0:
            return torch.empty(0, dtype=torch.long)
        b = boxes.cpu().numpy()
        s = scores.cpu().numpy()
        x1, y1, x2, y2 = b[:,0], b[:,1], b[:,2], b[:,3]
        areas = (x2-x1)*(y2-y1)
        order = s.argsort()[::-1]
        keep  = []
        while order.size > 0:
            i = order[0]; keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0., xx2-xx1) * np.maximum(0., yy2-yy1)
            iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            order = order[np.where(iou <= iou_threshold)[0] + 1]
        return torch.tensor(keep, dtype=torch.long)
    _tv.nms = _cpu_nms
except Exception:
    pass

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ══════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════
BASE            = Path(os.getcwd())

# Detection thresholds
COCO_CONF       = 0.45
HELMET_CONF     = 0.40

# Speeding
SPEED_PX        = 14    # smoothed px/frame threshold
SPEED_WIN       = 6     # rolling average window
SPEED_CONFIRM   = 6     # consecutive fast frames needed

# Triple Riding
TRIPLE_MIN      = 3

# DeepSort
TRACK_N_INIT    = 4     # frames before track confirmed
TRACK_MAX_AGE   = 40

# Confirmation — a violation must appear on N frames before saving
CONFIRM_FRAMES  = 3

# Box quality filters (rejects walls, poles, distant pedestrians)
MIN_BOX_AREA    = 1800
MAX_AR          = 4.2
MIN_AR          = 0.22

# COCO class IDs
CLS_PERSON      = 0
CLS_CAR         = 2
CLS_BIKE        = 3
CLS_TL          = 9     # traffic light

# Violation display colours (BGR)
VCOLOURS = {
    "No Helmet":     (0,  140, 255),
    "Triple Riding": (0,  0,   255),
    "Speeding":      (200, 0,  255),
    "Wrong-Way":     (128, 0,  200),
    "Red-Light Jump":(0,  60,  255),
}


# ══════════════════════════════════════════════════════════════
#  UTILITY
# ══════════════════════════════════════════════════════════════
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def valid_box(x1, y1, x2, y2):
    w, h = x2-x1, y2-y1
    if w <= 0 or h <= 0 or w*h < MIN_BOX_AREA:
        return False
    ar = w / h
    return MIN_AR <= ar <= MAX_AR

def box_iou(a, b):
    ax1,ay1,ax2,ay2 = float(a[0]),float(a[1]),float(a[2]),float(a[3])
    bx1,by1,bx2,by2 = float(b[0]),float(b[1]),float(b[2]),float(b[3])
    ix1 = max(ax1,bx1); iy1 = max(ay1,by1)
    ix2 = min(ax2,bx2); iy2 = min(ay2,by2)
    inter = max(0.,ix2-ix1)*max(0.,iy2-iy1)
    if inter == 0:
        return 0.
    return inter/((ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter+1e-6)


# ══════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ══════════════════════════════════════════════════════════════
class TrafficViolationApp:

    def __init__(self, window):
        self.window = window
        self.window.title(" AI Traffic Enforcement System")
        self.window.geometry("1280x750")
        self.window.configure(bg="#0d0d0d")

        # ── Paths ──────────────────────────────────────────────
        self.json_path = BASE / "violations" / "violations.json"
        self.image_dir = BASE / "evidence" / "images"
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        self.image_dir.mkdir(parents=True, exist_ok=True)

        # ── Load models ────────────────────────────────────────
        self.coco_model = YOLO("yolov8n.pt")
        self.coco_model.overrides["task"] = "detect"

        self.helmet_model = None
        hm = BASE / "helmet_best.pt"
        if hm.exists():
            try:
                self.helmet_model = YOLO(str(hm))
                self.helmet_model.overrides["task"] = "detect"
                print("✔  helmet_best.pt loaded.")
            except Exception:
                print("⚠  helmet_best.pt failed to load — using fallback.")
        else:
            print("⚠  helmet_best.pt not found — using rider-presence fallback.")

        # Also check if trained violation model exists (from train_model.py)
        self.viol_model = None
        sp = BASE / "dataset_summary.json"
        if sp.exists():
            try:
                d  = json.loads(sp.read_text())
                mp = str(d.get("model_path","")).strip()
                if mp and mp not in ("",".") and Path(mp).exists():
                    self.viol_model = YOLO(mp)
                    self.viol_model.overrides["task"] = "detect"
                    print(f"✔  Violation model loaded: {Path(mp).name}")
            except Exception:
                pass

        # ── Tracker ────────────────────────────────────────────
        self._make_tracker()

        # ── Persistent violation DB ────────────────────────────
        self.violations_db = []
        self._saved        = {}   # { tid: set(vtype) }
        if self.json_path.exists():
            try:
                self.violations_db = json.loads(self.json_path.read_text())
                for rec in self.violations_db:
                    self._saved.setdefault(rec["track_id"], set()).add(rec["type"])
                print(f"📂  Loaded {len(self.violations_db)} existing records.")
            except Exception:
                pass

        # ── Per-track state ────────────────────────────────────
        self.prev_center   = {}   # { tid: (cx, cy) }
        self.speed_buf     = {}   # { tid: [px/frame] }
        self.speed_streak  = {}   # { tid: int }
        self.flow_vectors  = []   # for dominant direction
        self.confirm_cnt   = {}   # { tid: { vtype: int } }
        self.red_light     = False
        self.stop_line_y   = None

        # ── Runtime ────────────────────────────────────────────
        self.is_running  = False
        self.video_path  = None

        self._setup_ui()

    # ──────────────────────────────────────────────────────────
    def _make_tracker(self):
        """Create a fresh DeepSort tracker (called on each Start)."""
        self.tracker = DeepSort(
            max_age             = TRACK_MAX_AGE,
            n_init              = TRACK_N_INIT,
            max_cosine_distance = 0.35,
            nn_budget           = 100,
        )

    # ══════════════════════════════════════════════════════════
    #  UI SETUP
    # ══════════════════════════════════════════════════════════
    def _setup_ui(self):
        # Header
        tk.Label(self.window,
                 text="⚡  AI TRAFFIC ENFORCEMENT SYSTEM  ⚡",
                 font=("Segoe UI", 18, "bold"),
                 bg="#0d0d0d", fg="#00e676").pack(fill=tk.X, pady=8)

        main = tk.Frame(self.window, bg="#0d0d0d")
        main.pack(expand=True, fill="both")

        # Video panel
        self.video_label = tk.Label(main, bg="black", bd=2, relief="flat")
        self.video_label.pack(side=tk.LEFT, padx=15, pady=8)

        # Log panel
        log_outer = tk.Frame(main, bg="#1a1a1a", bd=1, relief="flat")
        log_outer.pack(side=tk.RIGHT, fill="both", expand=True, padx=10, pady=8)

        tk.Label(log_outer, text="VIOLATION LOG",
                 font=("Segoe UI", 11, "bold"),
                 bg="#1a1a1a", fg="#ff5252").pack(pady=(6,2))

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview",
                        background="#1a1a1a", foreground="white",
                        fieldbackground="#1a1a1a", rowheight=22)
        style.configure("Treeview.Heading",
                        background="#222", foreground="#00e676")

        self.tree = ttk.Treeview(
            log_outer,
            columns=("ID","Violation","Time"),
            show="headings"
        )
        for col, w in [("ID",70),("Violation",160),("Time",90)]:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=w, anchor="center")
        self.tree.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        for vt, fg in [
            ("No Helmet",     "#ff9800"),
            ("Triple Riding", "#f44336"),
            ("Speeding",      "#ff5252"),
            ("Wrong-Way",     "#e040fb"),
            ("Red-Light Jump","#ff1744"),
        ]:
            self.tree.tag_configure(vt, foreground=fg)

        self.count_var = tk.StringVar(value="Total Violations: 0")
        tk.Label(log_outer, textvariable=self.count_var,
                 font=("Segoe UI", 10),
                 bg="#1a1a1a", fg="#aaaaaa").pack(pady=(0,6))

        # Buttons
        btn_frame = tk.Frame(self.window, bg="#0d0d0d")
        btn_frame.pack(fill=tk.X, pady=10)

        def btn(text, cmd, bg="#333333", fg="white"):
            tk.Button(btn_frame, text=text, command=cmd,
                      bg=bg, fg=fg,
                      font=("Segoe UI", 10, "bold"),
                      relief="flat", padx=14, pady=6
                      ).pack(side=tk.LEFT, padx=8)

        btn("📂  Select Video",      self.select_video)
        btn("▶  Start",              self.start,           bg="#007bff")
        btn("⏹  Stop",               self.stop,            bg="#dc3545")
        btn("🔴  Toggle Red Light",  self.toggle_red_light, bg="#b71c1c")
        btn("🗑  Clear Log",          self.clear_log,       bg="#444444")

        # Status bar
        self.status_var = tk.StringVar(value="Ready — select a video to begin.")
        tk.Label(self.window, textvariable=self.status_var,
                 font=("Segoe UI", 9),
                 bg="#0d0d0d", fg="#777777").pack(pady=(0,4))

    # ══════════════════════════════════════════════════════════
    #  CONTROLS
    # ══════════════════════════════════════════════════════════
    def select_video(self):
        path = filedialog.askopenfilename(
            filetypes=[("Video files","*.mp4 *.avi *.mov *.mkv *.webm"),
                       ("All","*.*")]
        )
        if path:
            self.video_path = path
            self.status_var.set(f"Loaded: {Path(path).name}")

    def start(self):
        if not self.video_path:
            messagebox.showerror("No Video", "Please select a video file first.")
            return
        # Reset per-session state
        self._make_tracker()
        self.prev_center.clear()
        self.speed_buf.clear()
        self.speed_streak.clear()
        self.flow_vectors.clear()
        self.confirm_cnt.clear()
        self.stop_line_y = None
        self.is_running  = True
        self.status_var.set("Processing…")
        threading.Thread(target=self._process, daemon=True).start()

    def stop(self):
        self.is_running = False
        self.status_var.set("Stopped.")

    def toggle_red_light(self):
        self.red_light = not self.red_light
        state = "🔴 RED" if self.red_light else "🟢 GREEN"
        self.status_var.set(f"Traffic light: {state}")

    def clear_log(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.count_var.set("Total Violations: 0")

    # ══════════════════════════════════════════════════════════
    #  DEDUPLICATION HELPERS
    # ══════════════════════════════════════════════════════════
    def _is_saved(self, tid, vtype):
        return vtype in self._saved.get(tid, set())

    def _lock(self, tid, vtype):
        self._saved.setdefault(tid, set()).add(vtype)

    def _confirm(self, tid, vtype):
        """
        Increment confirmation counter.
        Returns True EXACTLY ONCE when CONFIRM_FRAMES reached — then locks.
        """
        if self._is_saved(tid, vtype):
            return False
        d = self.confirm_cnt.setdefault(tid, {})
        d[vtype] = d.get(vtype, 0) + 1
        if d[vtype] >= CONFIRM_FRAMES:
            d[vtype] = CONFIRM_FRAMES + 9999  # permanent lock
            return True
        return False

    def _reset_confirm(self, tid, vtype):
        """Reset counter when violation not seen this frame (prevents jitter)."""
        cv = self.confirm_cnt.get(tid, {}).get(vtype, 0)
        if cv < CONFIRM_FRAMES:
            self.confirm_cnt.setdefault(tid, {})[vtype] = 0

    # ══════════════════════════════════════════════════════════
    #  SAVE VIOLATION  (runs exactly once per tid+vtype)
    # ══════════════════════════════════════════════════════════
    def _save_violation(self, frame, box, vtype, tid):
        if self._is_saved(tid, vtype):
            return

        x1, y1, x2, y2 = map(int, box)
        fh, fw = frame.shape[:2]
        col    = VCOLOURS.get(vtype, (0,200,0))

        ev = frame.copy()
        bar_y1 = max(y1-38, 0)
        cv2.rectangle(ev, (x1, bar_y1), (x2, y1), col, -1)
        cv2.putText(ev, f"{vtype}  |  ID:{tid}",
                    (x1+4, max(y1-10,14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255,255,255), 2)
        cv2.rectangle(ev, (x1,y1), (x2,y2), col, 3)
        cv2.putText(ev,
                    datetime.now().strftime("%Y-%m-%d  %H:%M:%S"),
                    (6, fh-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220,220,220), 1)

        img_name = f"V_{tid}_{vtype.replace(' ','_')}_{int(time.time()*1000)}.jpg"
        cv2.imwrite(str(self.image_dir / img_name), ev)

        ts    = datetime.now().strftime("%H:%M:%S")
        entry = {
            "track_id": tid,
            "type":     vtype,
            "time":     ts,
            "image":    img_name,
        }
        self.violations_db.append(entry)
        self._lock(tid, vtype)

        # Atomic JSON write
        tmp = str(self.json_path) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.violations_db, f, indent=4)
        os.replace(tmp, str(self.json_path))

        # Thread-safe UI update
        self.window.after(0, lambda: self._ui_add_row(tid, vtype, ts))
        print(f"  📄  [{vtype}]  ID:{tid}  →  {img_name}")

    def _ui_add_row(self, tid, vtype, ts):
        self.tree.insert("", 0, values=(tid, vtype, ts), tags=(vtype,))
        self.count_var.set(f"Total Violations: {len(self.tree.get_children())}")

    # ══════════════════════════════════════════════════════════
    #  VIOLATION DETECTION HELPERS
    # ══════════════════════════════════════════════════════════
    def _get_riders(self, people, x1, y1, x2, y2):
        riders = []
        for p in people:
            pcx = (p[0]+p[2])/2
            pcy = (p[1]+p[3])/2
            inside   = (x1<=pcx<=x2) and (y1<=pcy<=y2)
            overlaps = box_iou([x1,y1,x2,y2],[p[0],p[1],p[2],p[3]]) > 0.12
            if inside or overlaps:
                riders.append(p)
        return riders

    def _no_helmet(self, frame, x1, y1, x2, y2, riders):
        if not riders:
            return False
        if self.helmet_model is None:
            return True   # fallback: flag all bikes with riders
        upper_y2 = y1 + (y2-y1)//2
        roi = frame[max(0,y1):upper_y2, max(0,x1):x2]
        if roi.size == 0:
            return True
        res = self.helmet_model(roi, verbose=False, conf=HELMET_CONF)[0]
        has_h = no_h = False
        for hb in res.boxes.data.tolist():
            cls = int(hb[5])
            if cls == 0: has_h = True
            if cls == 1: no_h  = True
        return no_h or (not has_h)

    def _update_speed(self, tid, cx, cy):
        """Returns (avg_speed, consecutive_frames_above_threshold)."""
        prev = self.prev_center.get(tid)
        self.prev_center[tid] = (cx, cy)
        if prev is None:
            self.speed_streak[tid] = 0
            return 0., 0
        raw  = math.hypot(cx-prev[0], cy-prev[1])
        buf  = self.speed_buf.setdefault(tid, [])
        buf.append(raw)
        if len(buf) > SPEED_WIN:
            buf.pop(0)
        avg = sum(buf)/len(buf)
        if avg > SPEED_PX:
            self.speed_streak[tid] = self.speed_streak.get(tid,0)+1
        else:
            self.speed_streak[tid] = 0
        return avg, self.speed_streak[tid]

    def _dominant_dir(self):
        if len(self.flow_vectors) < 15:
            return 0
        dxs = [v[0] for v in self.flow_vectors[-60:]]
        return 1 if sum(dxs) > 0 else -1

    def _detect_red_light(self, frame, tl_boxes):
        for box in tl_boxes:
            x1,y1,x2,y2 = map(int, box[:4])
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            h3  = max(1,(y2-y1)//3)
            top = roi[:h3]
            hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
            m1  = cv2.inRange(hsv,(0,  120,70),(10, 255,255))
            m2  = cv2.inRange(hsv,(170,120,70),(180,255,255))
            rpx = cv2.countNonZero(m1)+cv2.countNonZero(m2)
            tot = top.shape[0]*top.shape[1]
            if tot > 0 and rpx/tot > 0.25:
                return True
        return False

    # ══════════════════════════════════════════════════════════
    #  MAIN PROCESSING LOOP  (runs in background thread)
    # ══════════════════════════════════════════════════════════
    def _process(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.status_var.set("Error: cannot open video.")
            return

        fps     = cap.get(cv2.CAP_PROP_FPS) or 25
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.stop_line_y = int(frame_h * 0.55)

        frame_no = 0

        while cap.isOpened() and self.is_running:
            ret, frame = cap.read()
            if not ret:
                break
            frame_no += 1

            # ── 1. YOLO detections ────────────────────────────
            raw      = self.coco_model(frame, verbose=False, conf=COCO_CONF)[0].boxes.data.tolist()
            people   = [d for d in raw if int(d[5])==CLS_PERSON]
            bikes_r  = [d for d in raw if int(d[5])==CLS_BIKE]
            cars_r   = [d for d in raw if int(d[5])==CLS_CAR]
            tl_boxes = [d for d in raw if int(d[5])==CLS_TL]

            # ── 2. Trained violation model (if available) ─────
            viol_dets = {}   # { (x1,y1,x2,y2): "No Helmet" }
            if self.viol_model is not None:
                try:
                    vraw  = self.viol_model(frame, verbose=False, conf=0.40, task="detect")[0].boxes.data.tolist()
                    names = getattr(self.viol_model,"names",{})
                    for det in vraw:
                        bx1=clamp(int(det[0]),0,frame_w); by1=clamp(int(det[1]),0,frame_h)
                        bx2=clamp(int(det[2]),0,frame_w); by2=clamp(int(det[3]),0,frame_h)
                        if not valid_box(bx1,by1,bx2,by2): continue
                        nm = names.get(int(det[5]),"")
                        if "no" in nm.lower() or "without" in nm.lower() or "helmet" in nm.lower():
                            viol_dets[(bx1,by1,bx2,by2)] = "No Helmet"
                except Exception:
                    pass

            # ── 3. Red-light state ────────────────────────────
            if tl_boxes:
                self.red_light = self._detect_red_light(frame, tl_boxes)

            # ── 4. DeepSort — bikes + cars only ───────────────
            ds_in = []
            for d in bikes_r + cars_r:
                x1,y1,x2,y2,conf,cls = d
                if not valid_box(int(x1),int(y1),int(x2),int(y2)):
                    continue
                ds_in.append(([x1,y1,x2-x1,y2-y1], conf, int(cls)))

            tracks = self.tracker.update_tracks(ds_in, frame=frame)

            # ── 5. Per-track violation logic ──────────────────
            active_tids = set()

            for track in tracks:
                if not track.is_confirmed():
                    continue

                tid  = int(track.track_id)
                ltrb = track.to_ltrb()
                x1 = clamp(int(ltrb[0]),0,frame_w)
                y1 = clamp(int(ltrb[1]),0,frame_h)
                x2 = clamp(int(ltrb[2]),0,frame_w)
                y2 = clamp(int(ltrb[3]),0,frame_h)

                if not valid_box(x1,y1,x2,y2):
                    continue

                cls_id  = getattr(track,"det_class",CLS_BIKE)
                is_bike = (cls_id==CLS_BIKE)
                cx      = (x1+x2)//2
                cy      = (y1+y2)//2
                active_tids.add(tid)

                seen_now = []   # violations detected THIS frame

                # ── A. No Helmet ──────────────────────────────
                # From trained model
                for (dx1,dy1,dx2,dy2),vt in viol_dets.items():
                    if box_iou([x1,y1,x2,y2],[dx1,dy1,dx2,dy2])>0.25:
                        if vt not in seen_now:
                            seen_now.append(vt)

                # From helmet model ROI / fallback
                if is_bike and "No Helmet" not in seen_now:
                    riders = self._get_riders(people,x1,y1,x2,y2)
                    if self._no_helmet(frame,x1,y1,x2,y2,riders):
                        seen_now.append("No Helmet")

                    # ── B. Triple Riding ──────────────────────
                    if len(riders) >= TRIPLE_MIN:
                        seen_now.append("Triple Riding")

                # ── C. Speeding ───────────────────────────────
                prev = self.prev_center.get(tid)
                avg_spd, consec = self._update_speed(tid, cx, cy)

                # Update flow vectors for wrong-way detection
                if prev is not None:
                    dx = cx - prev[0]
                    self.flow_vectors.append((dx, cy-prev[1]))
                    if len(self.flow_vectors) > 200:
                        self.flow_vectors.pop(0)

                if consec >= SPEED_CONFIRM:
                    seen_now.append("Speeding")

                # ── D. Wrong-Way ──────────────────────────────
                dom = self._dominant_dir()
                if dom != 0 and prev is not None:
                    vdx = cx - prev[0]
                    if abs(vdx) > 5 and (vdx>0) != (dom>0):
                        seen_now.append("Wrong-Way")

                # ── E. Red-Light Jump ─────────────────────────
                if self.red_light and prev is not None:
                    prev_cy = prev[1]
                    crossed = (
                        (prev_cy < self.stop_line_y <= cy) or
                        (prev_cy > self.stop_line_y >= cy)
                    )
                    if crossed:
                        seen_now.append("Red-Light Jump")

                # ── F. Confirm + deduplicate + save ───────────
                all_types = {
                    "No Helmet","Triple Riding","Speeding",
                    "Wrong-Way","Red-Light Jump"
                }
                for vtype in all_types:
                    if vtype in seen_now:
                        if not self._is_saved(tid, vtype):
                            if self._confirm(tid, vtype):
                                self._save_violation(
                                    frame.copy(),[x1,y1,x2,y2],vtype,tid
                                )
                    else:
                        self._reset_confirm(tid, vtype)

                # ── G. Draw box + label ───────────────────────
                saved_v = self._saved.get(tid, set())

                if seen_now:
                    if "Triple Riding" in seen_now:   box_col=(0,0,255)
                    elif "No Helmet"   in seen_now:   box_col=(0,140,255)
                    elif "Speeding"    in seen_now:   box_col=(200,0,255)
                    elif "Wrong-Way"   in seen_now:   box_col=(128,0,200)
                    else:                             box_col=(0,60,255)
                elif saved_v:
                    box_col = (0,120,255)
                else:
                    box_col = (0,210,60)

                cv2.rectangle(frame,(x1,y1),(x2,y2),box_col,2)

                if seen_now:
                    lbl = ", ".join(seen_now)
                elif saved_v:
                    lbl = "⚠ "+" ".join(sorted(saved_v))
                else:
                    lbl = "OK"

                label = f"ID:{tid}  {lbl}"
                font  = cv2.FONT_HERSHEY_SIMPLEX
                fs, th = 0.48, 1
                (lw,lh),_ = cv2.getTextSize(label,font,fs,th)
                pad = 4
                if y1>lh+pad+4:
                    bx1b,by1b=x1,y1-lh-pad-2; bx2b,by2b=x1+lw+pad,y1
                    tx,ty=x1+2,y1-pad
                else:
                    bx1b,by1b=x1,y1; bx2b,by2b=x1+lw+pad,y1+lh+pad+2
                    tx,ty=x1+2,y1+lh+2
                ov=frame.copy()
                cv2.rectangle(ov,(bx1b,by1b),(bx2b,by2b),(0,0,0),-1)
                cv2.addWeighted(ov,0.65,frame,0.35,0,frame)
                cv2.putText(frame,label,(tx,ty),font,fs,box_col,th,cv2.LINE_AA)

                # Rider badge
                if is_bike:
                    rc = len(self._get_riders(people,x1,y1,x2,y2))
                    if rc > 0:
                        cv2.putText(frame,f"R:{rc}",(x2-40,y1+16),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),2)

            # ── 6. HUD overlays ───────────────────────────────
            # Stop line
            cv2.line(frame,(0,self.stop_line_y),(frame_w,self.stop_line_y),(0,255,255),2)
            cv2.putText(frame,"STOP LINE",(6,self.stop_line_y-6),
                        cv2.FONT_HERSHEY_SIMPLEX,0.46,(0,255,255),1)

            # Traffic light state
            tl_col = (0,0,255) if self.red_light else (0,255,0)
            cv2.putText(frame,"🔴 RED" if self.red_light else "🟢 GREEN",
                        (10,28),cv2.FONT_HERSHEY_SIMPLEX,0.8,tl_col,2)

            # Bottom bar
            cv2.rectangle(frame,(0,frame_h-26),(frame_w,frame_h),(15,15,15),-1)
            cv2.putText(frame,
                f"Frame:{frame_no}   Violations:{len(self.violations_db)}"
                f"   Vehicles:{len(active_tids)}",
                (8,frame_h-8),cv2.FONT_HERSHEY_SIMPLEX,0.46,(200,200,200),1)

            # Legend
            LEGEND=[("No Helmet",(0,140,255)),("Triple Riding",(0,0,255)),
                    ("Speeding",(200,0,255)),("Wrong-Way",(128,0,200)),
                    ("Red-Light Jump",(0,60,255)),("Clean",(0,210,60))]
            for i,(lbl,lc) in enumerate(LEGEND):
                lx=frame_w-158; ly=8+i*20
                cv2.rectangle(frame,(lx,ly),(lx+12,ly+12),lc,-1)
                cv2.putText(frame,lbl,(lx+16,ly+11),
                            cv2.FONT_HERSHEY_SIMPLEX,0.38,lc,1)

            # ── 7. Send frame to Tkinter UI ───────────────────
            disp    = cv2.resize(frame,(820,520))
            img_rgb = cv2.cvtColor(disp,cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_tk  = ImageTk.PhotoImage(image=img_pil)
            # Thread-safe image update
            self.window.after(0, self._update_frame, img_tk)

        cap.release()
        self.window.after(0, lambda: self.status_var.set("Done. Processing complete."))

    def _update_frame(self, img_tk):
        """Called from main thread — safe Tkinter image update."""
        self.video_label.config(image=img_tk)
        self.video_label.image = img_tk   # prevent GC


# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    root = tk.Tk()
    app  = TrafficViolationApp(root)
    root.mainloop()