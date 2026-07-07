import cv2
import os
import time
import math
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

# Configurable Input (Accepts video file extensions or image file extensions)
INPUT_SOURCE = 'three_girls.mp4'  
MODEL_WEIGHTS = 'weights/custom_biker_model.pt'
OUTPUT_FOLDER = 'violations'

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
model = YOLO(MODEL_WEIGHTS)

print("\n--- DETECTED MODEL CLASSES ---")
print(model.names)
print("------------------------------\n")

VIOLATION_CLASSES = []
for idx, name in model.names.items():
    n = name.lower()
    if 'no' in n or 'without' in n or 'unhelmet' in n:
        VIOLATION_CLASSES.append(idx)

if not VIOLATION_CLASSES:
    print("⚠️ Defaulting Class ID 1 as the violation target.")
    VIOLATION_CLASSES = [1]

IS_IMAGE = INPUT_SOURCE.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp'))

if IS_IMAGE:
    cap = None
else:
    cap = cv2.VideoCapture(INPUT_SOURCE)
    if not cap.isOpened():
        print(f"❌ ERROR: Could not open video file '{INPUT_SOURCE}'")
        exit()

WINDOW_NAME = "Traffic Safety Monitoring Feed"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 960, 540)

processed_violators = set()
processed_clusters = set() 

while True:
    if IS_IMAGE:
        frame = cv2.imread(INPUT_SOURCE)
        if frame is None:
            print(f"❌ ERROR: Could not open image file '{INPUT_SOURCE}'")
            break
    else:
        ret, frame = cap.read()
        if not ret:
            break
        
    if IS_IMAGE:
        results = model(frame, verbose=False, conf=0.25)
    else:
        results = model.track(frame, persist=True, verbose=False, conf=0.25)
    
    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        classes = results[0].boxes.cls.cpu().numpy().astype(int)
        
        track_ids = results[0].boxes.id
        track_ids = track_ids.cpu().numpy().astype(int) if track_ids is not None else [None] * len(boxes)
        
        current_frame_heads = []
        
        # --- FEATURE 1: Individual Head Processing ---
        for box, cls, t_id in zip(boxes, classes, track_ids):
            x1, y1, x2, y2 = box
            class_name = model.names.get(cls, f"Class {cls}")
            current_frame_heads.append((box, cls, t_id))
            
            if cls in VIOLATION_CLASSES:
                label_text = f"ALERT: NO HELMET" + (f" (ID: {t_id})" if t_id is not None else "")
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, label_text, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                if IS_IMAGE or (t_id is not None and t_id not in processed_violators):
                    print(f"[VIOLATION TRIPPED] Logging" + (f" Track ID {t_id}" if t_id is not None else " Static Target"))
                    
                    evidence_view = frame.copy()
                    cv2.rectangle(evidence_view, (x1, y1), (x2, y2), (0, 0, 255), 4)
                    cv2.putText(evidence_view, f"VIOLATION: NO HELMET" + (f" | ID {t_id}" if t_id is not None else ""), 
                                (x1, y1 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    stamp = int(time.time())
                    file_prefix = f"img_violator_{stamp}" if IS_IMAGE else f"violator_id_{t_id}_{stamp}"
                    cv2.imwrite(f"{OUTPUT_FOLDER}/{file_prefix}.jpg", evidence_view)
                    
                    with open(f"{OUTPUT_FOLDER}/violation_log.txt", "a") as log_file:
                        source_info = f"Source: {INPUT_SOURCE}" if IS_IMAGE else f"Track ID: {t_id}"
                        log_file.write(f"Timestamp: {stamp} | {source_info} | Type: NO-HELMET ({class_name})\n")
                        
                    if not IS_IMAGE and t_id is not None:
                        processed_violators.add(t_id)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Safe: {class_name}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            
        # --- FEATURE 2: Spatial Clustering For Triple+ Riding ---
        head_clusters = cluster_heads(current_frame_heads, max_dist=180)
        
        for cluster in head_clusters:
            rider_count = len(cluster)
            
            if rider_count >= 3:
                cx1 = min([h[0][0] for h in cluster])
                cy1 = min([h[0][1] for h in cluster])
                cx2 = max([h[0][2] for h in cluster])
                cy2 = max([h[0][3] for h in cluster])
                
                if IS_IMAGE:
                    cluster_key = f"static_{cx1}_{cy1}"
                else:
                    member_ids = sorted([str(h[2]) for h in cluster if h[2] is not None])
                    cluster_key = "_".join(member_ids)
                
                cv2.rectangle(frame, (cx1 - 10, cy1 - 10), (cx2 + 10, cy2 + 10), (255, 0, 255), 2)
                cv2.putText(frame, f"CRITICAL: TRIPLE RIDING ({rider_count} Pax)", (cx1 - 10, cy1 - 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                
                if IS_IMAGE or (cluster_key and cluster_key not in processed_clusters):
                    print(f"[TRIPLE RIDING TRIPPED] Logging Group Combination [{cluster_key}]")
                    
                    evidence_view = frame.copy()
                    cv2.rectangle(evidence_view, (cx1 - 15, cy1 - 15), (cx2 + 15, cy2 + 15), (255, 0, 255), 4)
                    cv2.putText(evidence_view, f"VIOLATION: TRIPLE RIDING | Count: {rider_count}", (cx1 - 15, cy1 - 32),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                    
                    stamp = int(time.time())
                    file_prefix = f"img_triple_{stamp}" if IS_IMAGE else f"triple_riding_{stamp}"
                    cv2.imwrite(f"{OUTPUT_FOLDER}/{file_prefix}.jpg", evidence_view)
                    
                    with open(f"{OUTPUT_FOLDER}/violation_log.txt", "a") as log_file:
                        log_file.write(f"Timestamp: {stamp} | Group IDs: {cluster_key} | Type: TRIPLE-RIDING\n")
                        
                    if not IS_IMAGE:
                        processed_clusters.add(cluster_key)

    cv2.imshow(WINDOW_NAME, frame)
    
    if IS_IMAGE:
        cv2.waitKey(0)
        break
    else:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if cap is not None:
    cap.release()
cv2.destroyAllWindows()