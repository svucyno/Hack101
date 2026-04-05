"""
dashboard/app.py  —  STEP 4
============================
Streamlit dashboard to view all logged violations.

RUN AFTER main_fixed.py:
    streamlit run dashboard/app.py

SHOWS:
  • Total violation counts per type
  • Filterable table of all violations
  • Evidence images for each violation
  • Download button for the JSON report
"""

import streamlit as st
import json, os
from pathlib import Path
from datetime import datetime
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────
BASE      = Path(__file__).parent.parent
JSON_PATH = BASE/"violations"/"violations.json"
IMG_DIR   = BASE/"evidence"/"images"

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="SIGCE Traffic Violation Dashboard",
    page_icon="🚦",
    layout="wide",
)

# ── Load data ──────────────────────────────────────────────────
@st.cache_data(ttl=10)   # refresh every 10 seconds
def load_data():
    if not JSON_PATH.exists():
        return []
    try:
        return json.loads(JSON_PATH.read_text())
    except Exception:
        return []

data = load_data()

# ── Header ─────────────────────────────────────────────────────
st.title("🚦 SIGCE AI Traffic Violation Dashboard")
st.caption(f"Data source: `{JSON_PATH}`  |  Last loaded: {datetime.now().strftime('%H:%M:%S')}")

if st.button("🔄 Refresh"):
    st.cache_data.clear()
    st.rerun()

if not data:
    st.warning("No violations logged yet. Run `python main_fixed.py` first.")
    st.stop()

# ── Build DataFrame ────────────────────────────────────────────
df = pd.DataFrame(data)
df.columns = [c.lower() for c in df.columns]

# Normalise column names  (JSON uses "type" or "violation_type")
if "violation_type" in df.columns and "type" not in df.columns:
    df["type"] = df["violation_type"]
if "type" not in df.columns:
    df["type"] = "Unknown"
if "track_id" not in df.columns:
    df["track_id"] = range(len(df))
if "time" not in df.columns:
    df["time"] = ""
if "image" not in df.columns:
    df["image"] = ""

# Clean type names
df["type"] = df["type"].str.replace("_"," ")

# ── KPI cards ──────────────────────────────────────────────────
violation_types = ["No Helmet","Triple Riding","Speeding"]
counts = {vt: int((df["type"]==vt).sum()) for vt in violation_types}
total  = len(df)

c0,c1,c2,c3 = st.columns(4)
c0.metric("🔢 Total Violations", total)
c1.metric("⛑️ No Helmet",     counts["No Helmet"],     delta=None)
c2.metric("👥 Triple Riding", counts["Triple Riding"],  delta=None)
c3.metric("💨 Speeding",      counts["Speeding"],       delta=None)

st.divider()

# ── Filters ────────────────────────────────────────────────────
col_f1, col_f2 = st.columns([1,2])
with col_f1:
    type_filter = st.multiselect(
        "Filter by violation type",
        options=df["type"].unique().tolist(),
        default=df["type"].unique().tolist(),
    )
with col_f2:
    search_id = st.text_input("Search by Track ID", "")

filtered = df[df["type"].isin(type_filter)]
if search_id.strip():
    filtered = filtered[filtered["track_id"].astype(str).str.contains(search_id.strip())]

st.caption(f"Showing {len(filtered)} of {total} records")

# ── Table ──────────────────────────────────────────────────────
st.dataframe(
    filtered[["track_id","type","time","image"]].rename(columns={
        "track_id":"Track ID","type":"Violation","time":"Timestamp","image":"Evidence File"
    }),
    use_container_width=True,
    height=280,
)

st.divider()

# ── Evidence images ────────────────────────────────────────────
st.subheader("📸 Evidence Images")

# Deduplicated: show only one image per (track_id, type)
seen = set()
unique_rows = []
for _, row in filtered.iterrows():
    key=(row["track_id"],row["type"])
    if key not in seen:
        seen.add(key)
        unique_rows.append(row)

if not unique_rows:
    st.info("No images to show for the current filter.")
else:
    cols_per_row = 4
    for row_start in range(0, len(unique_rows), cols_per_row):
        cols = st.columns(cols_per_row)
        for col_idx, row in enumerate(unique_rows[row_start:row_start+cols_per_row]):
            img_file = row["image"]
            img_path = IMG_DIR / img_file
            with cols[col_idx]:
                vtype = str(row["type"])
                tid   = row["track_id"]
                ts    = str(row.get("time",""))
                if img_path.exists():
                    st.image(
                        str(img_path),
                        caption=f"ID:{tid} | {vtype}\n{ts}",
                        use_container_width=True,
                    )
                else:
                    st.warning(f"Image not found:\n{img_file}")
                    st.caption(f"ID:{tid} | {vtype}")

st.divider()

# ── Download ────────────────────────────────────────────────────
st.subheader("⬇️  Export")
col_dl1, col_dl2 = st.columns(2)

with col_dl1:
    st.download_button(
        label="📥 Download violations.json",
        data=JSON_PATH.read_bytes(),
        file_name="violations.json",
        mime="application/json",
    )

with col_dl2:
    csv = filtered.to_csv(index=False).encode()
    st.download_button(
        label="📊 Download as CSV",
        data=csv,
        file_name="violations.csv",
        mime="text/csv",
    )

# ── Chart ──────────────────────────────────────────────────────
st.divider()
st.subheader("📊 Violation Breakdown")
chart_data = df["type"].value_counts().reset_index()
chart_data.columns = ["Violation Type","Count"]
st.bar_chart(chart_data.set_index("Violation Type"))