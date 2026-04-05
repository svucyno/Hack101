import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
from pathlib import Path
from PIL import Image
import plotly.graph_objects as go

st.set_page_config(page_title="Traffic Violation Dashboard", layout="wide")

# Load violations
VIOLATIONS_LOG = "violations.json"
EVIDENCE_DIR = "evidence"

@st.cache_data
def load_violations():
    if os.path.exists(VIOLATIONS_LOG):
        with open(VIOLATIONS_LOG, "r") as f:
            return json.load(f)
    return []

def get_violation_counts(violations):
    counts = {
        "No Helmet": 0,
        "Triple Riding": 0,
        "Speeding": 0
    }
    for v in violations:
        violation_type = v.get("violation", "Unknown")
        if violation_type in counts:
            counts[violation_type] += 1
    return counts

def get_violation_images(violations, violation_type):
    """Get all images for a specific violation type"""
    images = []
    for v in violations:
        if v.get("violation") == violation_type:
            image_path = v.get("image_path")
            if image_path and os.path.exists(image_path):
                images.append({
                    "path": image_path,
                    "track_id": v.get("track_id"),
                    "timestamp": v.get("timestamp"),
                    "confidence": v.get("confidence", 0)
                })
    return images

# Main dashboard
st.title("🚦 Traffic Violation Detection Dashboard")

violations = load_violations()
counts = get_violation_counts(violations)

# Summary cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Violations", len(violations), delta=None)

with col2:
    st.metric("🪖 No Helmet", counts["No Helmet"], delta=None)

with col3:
    st.metric("🏍️ Triple Riding", counts["Triple Riding"], delta=None)

with col4:
    st.metric("⚡ Speeding", counts["Speeding"], delta=None)

st.divider()

# Filters
col1, col2 = st.columns([2, 2])

with col1:
    violation_filter = st.selectbox(
        "Filter by Violation Type",
        ["All Violations", "No Helmet", "Triple Riding", "Speeding"]
    )

with col2:
    track_id_search = st.text_input("Search by Track ID")

# Apply filters
filtered_violations = violations

if violation_filter != "All Violations":
    filtered_violations = [v for v in filtered_violations 
                          if v.get("violation") == violation_filter]

if track_id_search:
    try:
        track_id = int(track_id_search)
        filtered_violations = [v for v in filtered_violations 
                              if v.get("track_id") == track_id]
    except ValueError:
        st.warning("Please enter a valid Track ID")

# Display results
if filtered_violations:
    st.subheader(f"Results: {len(filtered_violations)} violations found")
    
    # Create DataFrame
    df_data = []
    for v in filtered_violations:
        df_data.append({
            "Track ID": v.get("track_id"),
            "Violation": v.get("violation"),
            "Timestamp": v.get("timestamp"),
            "Confidence": f"{v.get('confidence', 0):.2%}",
            "Image": v.get("image_path")
        })
    
    df = pd.DataFrame(df_data)
    st.dataframe(df.drop("Image", axis=1), use_container_width=True)
    
    st.divider()
    
    # Display images in grid
    st.subheader("Evidence Images")
    
    cols = st.columns(3)
    col_idx = 0
    
    for v in filtered_violations:
        image_path = v.get("image_path")
        if image_path and os.path.exists(image_path):
            col = cols[col_idx % 3]
            with col:
                image = Image.open(image_path)
                st.image(image, use_column_width=True)
                st.caption(f"**ID:** {v.get('track_id')} | **Type:** {v.get('violation')}\n"
                          f"**Conf:** {v.get('confidence', 0):.2%}")
            col_idx += 1

else:
    st.info("No violations found matching your filters")

# Statistics
st.divider()
st.subheader("Statistics")

col1, col2 = st.columns(2)

with col1:
    # Pie chart
    fig = go.Figure(data=[go.Pie(
        labels=["No Helmet", "Triple Riding", "Speeding"],
        values=[counts["No Helmet"], counts["Triple Riding"], counts["Speeding"]],
        hole=0.3
    )])
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Violations over time
    if filtered_violations or violations:
        data_list = filtered_violations if filtered_violations else violations
        violation_types = [v.get("violation") for v in data_list]
        
        counts_series = pd.Series(violation_types).value_counts()
        fig = go.Figure(data=[go.Bar(
            x=counts_series.index,
            y=counts_series.values,
            marker_color=['#FF6B6B', '#FFA500', '#FF4444']
        )])
        fig.update_layout(
            title="Violations by Type",
            xaxis_title="Violation Type",
            yaxis_title="Count",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

# Refresh button
if st.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()