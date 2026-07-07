import streamlit as st
import cv2
import os
import time
import math
import numpy as np
import tempfile
from ultralytics import YOLO

def cluster_heads(heads, max_dist=180):
    clusters = []
    for head in heads:
        box, cls, t_id = head
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        inserted = False
        for cluster in clusters:
            for member in cluster:
                mx1, my1, mx2, my2 = member[0]
                mcx, mcy = (mx1 + mx2) / 2, (my1 + my2) / 2
                dist = math.sqrt((cx - mcx)**2 + (cy - mcy)**2)
                if dist < max_dist:
                    cluster.append(head)
                    inserted = True
                    break
            if inserted:
                break
        if not inserted:
            clusters.append([head])
    return clusters

st.set_page_config(page_title="Traffic Safety Analytics", layout="wide")
st.title("🚦 Smart Traffic Enforcement System")
st.subheader("Helmet Compliance & Triple Riding Detection Dashboard")

MODEL_WEIGHTS = 'weights/custom_biker_model.pt'
OUTPUT_FOLDER = 'violations'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@st.cache_resource
def load_yolo_model():
    return YOLO(MODEL_WEIGHTS)

try:
    model = load_yolo_model()
except Exception as e:
    st.error(f"Could not load YOLO model from {MODEL_WEIGHTS}. Please check the path.")
    st.stop()

VIOLATION_CLASSES = []
for idx, name in model.names.items():
    n = name.lower()
    if 'no' in n or 'without' in n or 'unhelmet' in n:
        VIOLATION_CLASSES.append(idx)
if not VIOLATION_CLASSES:
    VIOLATION_CLASSES = [1]

uploaded_file = st.sidebar.file_uploader(
    "Upload Source Material",
    type=['png', 'jpg', 'jpeg', 'webp', 'mp4', 'avi', 'mov', 'mkv']
)

col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("### 🖥️ Live Processing Feed")
    video_placeholder = st.empty()

with col2:
    st.markdown("### 📋 Real-Time Activity Log")
    log_placeholder = st.empty()

log_entries = []

def update_dashboard_log(message):
    log_entries.insert(0, f"⏱️ {time.strftime('%H:%M:%S')} - {message}")
    log_placeholder.markdown("\n".join(log_entries[:15]))

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    is_image = uploaded_file.name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
    
    if is_image:
        nparr = np.frombuffer(file_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        results = model(frame, verbose=False, conf=0.25)
        
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            current_frame_heads = []
            
            for box, cls in zip(boxes, classes):
                x1, y1, x2, y2 = box
                class_name = model.names.get(cls, f"Class {cls}")
                current_frame_heads.append((box, cls, None))
                
                if cls in VIOLATION_CLASSES:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, "ALERT: NO HELMET", (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    evidence_view = frame.copy()
                    stamp = int(time.time())
                    cv2.imwrite(f"{OUTPUT_FOLDER}/img_violator_{stamp}.jpg", evidence_view)
                    with open(f"{OUTPUT_FOLDER}/violation_log.txt", "a") as log_file:
                        log_file.write(f"Timestamp: {stamp} | Source: {uploaded_file.name} | Type: NO-HELMET ({class_name})\n")
                    update_dashboard_log(f"Violation: Unhelmeted Rider Detected")
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Safe: {class_name}", (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            head_clusters = cluster_heads(current_frame_heads, max_dist=180)
            for cluster in head_clusters:
                rider_count = len(cluster)
                if rider_count >= 3:
                    cx1 = min([h[0][0] for h in cluster])
                    cy1 = min([h[0][1] for h in cluster])
                    cx2 = max([h[0][2] for h in cluster])
                    cy2 = max([h[0][3] for h in cluster])
                    cv2.rectangle(frame, (cx1 - 10, cy1 - 10), (cx2 + 10, cy2 + 10), (255, 0, 255), 2)
                    cv2.putText(frame, f"CRITICAL: TRIPLE RIDING ({rider_count} Pax)", (cx1 - 10, cy1 - 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                    evidence_view = frame.copy()
                    stamp = int(time.time())
                    cv2.imwrite(f"{OUTPUT_FOLDER}/img_triple_{stamp}.jpg", evidence_view)
                    with open(f"{OUTPUT_FOLDER}/violation_log.txt", "a") as log_file:
                        log_file.write(f"Timestamp: {stamp} | Source: {uploaded_file.name} | Type: TRIPLE-RIDING\n")
                    update_dashboard_log(f"Violation: Triple Riding Confirmed ({rider_count} People)")
                    
        video_placeholder.image(frame, channels="BGR", use_container_width=True)
        st.success("🎯 Static Analysis Finished.")
    else:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(file_bytes)
        tfile.close()
        cap = cv2.VideoCapture(tfile.name)
        processed_violators = set()
        processed_clusters = set()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model.track(frame, persist=True, verbose=False, conf=0.25)
            
            if results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                classes = results[0].boxes.cls.cpu().numpy().astype(int)
                track_ids = results[0].boxes.id
                track_ids = track_ids.cpu().numpy().astype(int) if track_ids is not None else [None] * len(boxes)
                current_frame_heads = []
                
                for box, cls, t_id in zip(boxes, classes, track_ids):
                    x1, y1, x2, y2 = box
                    class_name = model.names.get(cls, f"Class {cls}")
                    current_frame_heads.append((box, cls, t_id))
                    
                    if cls in VIOLATION_CLASSES:
                        label_text = f"ALERT: NO HELMET" + (f" (ID: {t_id})" if t_id is not None else "")
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, label_text, (x1, y1 - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        if t_id is not None and t_id not in processed_violators:
                            evidence_view = frame.copy()
                            stamp = int(time.time())
                            cv2.imwrite(f"{OUTPUT_FOLDER}/violator_id_{t_id}_{stamp}.jpg", evidence_view)
                            with open(f"{OUTPUT_FOLDER}/violation_log.txt", "a") as log_file:
                                log_file.write(f"Timestamp: {stamp} | Track ID: {t_id} | Type: NO-HELMET ({class_name})\n")
                            processed_violators.add(t_id)
                            update_dashboard_log(f"Logged Track ID {t_id} (No Helmet)")
                    else:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"Safe: {class_name}", (x1, y1 - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                    
                head_clusters = cluster_heads(current_frame_heads, max_dist=180)
                for cluster in head_clusters:
                    rider_count = len(cluster)
                    if rider_count >= 3:
                        cx1 = min([h[0][0] for h in cluster])
                        cy1 = min([h[0][1] for h in cluster])
                        cx2 = max([h[0][2] for h in cluster])
                        cy2 = max([h[0][3] for h in cluster])
                        member_ids = sorted([str(h[2]) for h in cluster if h[2] is not None])
                        cluster_key = "_".join(member_ids)
                        cv2.rectangle(frame, (cx1 - 10, cy1 - 10), (cx2 + 10, cy2 + 10), (255, 0, 255), 2)
                        cv2.putText(frame, f"CRITICAL: TRIPLE RIDING ({rider_count} Pax)", (cx1 - 10, cy1 - 22),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                        if cluster_key and cluster_key not in processed_clusters:
                            evidence_view = frame.copy()
                            stamp = int(time.time())
                            cv2.imwrite(f"{OUTPUT_FOLDER}/triple_riding_{stamp}.jpg", evidence_view)
                            with open(f"{OUTPUT_FOLDER}/violation_log.txt", "a") as log_file:
                                log_file.write(f"Timestamp: {stamp} | Group IDs: {cluster_key} | Type: TRIPLE-RIDING\n")
                            processed_clusters.add(cluster_key)
                            update_dashboard_log(f"Logged Triple Riding Group [{cluster_key}]")
            video_placeholder.image(frame, channels="BGR", use_container_width=True)
            
        cap.release()
        os.unlink(tfile.name)
        st.success("🎯 Video Stream Processing Completed.")
else:
    st.info("👈 Use the sidebar panel to upload a traffic photo or video stream file.")