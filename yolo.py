import cv2
import torch
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
from ultralytics import YOLO
import csv


# =============================================================================
# SECTION 1 — CONFIGURATION
# =============================================================================

ENTRY_VIDEO_PATH = 'entry_file.mp4'
EXIT_VIDEO_PATH  = 'exit_file.mp4'
OUTPUT_CSV       = 'results.csv'
FPS              = 10
SIM_THRESHOLD    = 0.7
MATCH_TIMEOUT    = 150
CROP_NUM         = 5
CROP_DELAY_SEC   = 1     # seconds to wait after first seeing a car before collecting crops

# ROI: (left, top, right, bottom) as fractions of frame size (0.0 - 1.0)
ENTRY_ROI          = (0.0,  0.167, 1.0, 1.0)   # bottom 5/6, full width
EXIT_ROI           = (0.25, 0.0,   1.0, 1.0)   # rightmost 3/4, full height
EXIT_MIN_CAR_SIZE  = 3 / 8  # car bounding box must cover this fraction of the ROI area


# =============================================================================
# SECTION 2 — LOAD MODELS
# =============================================================================


ultra_model = YOLO('yolov8n.pt')

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

resnet_model = models.resnet50(weights='IMAGENET1K_V2')
resnet_model.fc = torch.nn.Identity()
resnet_model.eval()
resnet_model = resnet_model.to(device)


# =============================================================================
# SECTION 3 -- HELPER FUNCTIONS
# =============================================================================

def extract_embedding(crop, resnet_model, device):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    tensor = transform(crop)
    tensor = tensor.unsqueeze(0)
    tensor = tensor.to(device)
    with torch.no_grad():
        embedding = resnet_model(tensor)
    return embedding.squeeze().cpu().numpy()


def get_average_embedding(crops, resnet_model, device):
    embeddings = [extract_embedding(crop, resnet_model, device) for crop in crops]
    stacked = np.stack(embeddings)
    averaged = np.mean(stacked, axis=0)
    norm = np.linalg.norm(averaged)
    normalized = averaged / norm
    return normalized


def get_vehicle_crop(frame, box):
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    pad = 10
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(frame.shape[1], x2 + pad)
    y2 = min(frame.shape[0], y2 + pad)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop



def match_car(exit_embedding, exit_time, entry_buffer):
    best_match_key = None
    best_similarity = -1.0
    for key, data in entry_buffer.items():
        entry_time = data["timestamp"]
        transit = exit_time - entry_time
        if transit < 0 or transit > MATCH_TIMEOUT:
            continue
        similarity = np.dot(exit_embedding, data["embedding"])
        if similarity > best_similarity:
            best_similarity = similarity
            best_match_key = key
    if best_similarity >= SIM_THRESHOLD:
        return best_match_key
    return None


# =============================================================================
# SECTION 4 -- PER-CAMERA PROCESSING FUNCTION
# =============================================================================

def process_camera(video_path, yolo_model, resnet_model, device, role, roi, min_car_size=0.0):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: could not open video {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Compute ROI pixel bounds once
    roi_x1 = int(roi[0] * w)
    roi_y1 = int(roi[1] * h)
    roi_x2 = int(roi[2] * w)
    roi_y2 = int(roi[3] * h)
    roi_area = (roi_x2 - roi_x1) * (roi_y2 - roi_y1)
    delay_frames = int(CROP_DELAY_SEC * fps)

    crop_buffer  = {}
    first_seen   = {}
    logged_ids   = set()
    results      = []
    frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1
        roi_frame = frame[roi_y1:roi_y2, roi_x1:roi_x2]

        results_yolo = yolo_model.track(
            roi_frame,
            persist=True,
            classes=[2, 5, 7],
            verbose=False
        )

        result = results_yolo[0]

        if result.boxes is None or result.boxes.id is None:
            continue

        boxes     = result.boxes.xyxy.cpu().numpy()
        track_ids = result.boxes.id.cpu().numpy().astype(int)

        for i, track_id in enumerate(track_ids):
            box = boxes[i]

            if track_id not in first_seen:
                first_seen[track_id] = frame_number

            # Wait CROP_DELAY_SEC before collecting crops
            if frame_number - first_seen[track_id] < delay_frames:
                continue

            # Skip if car is too small relative to ROI area
            box_area = (box[2] - box[0]) * (box[3] - box[1])
            if box_area / roi_area < min_car_size:
                continue

            crop = get_vehicle_crop(roi_frame, box)
            if crop is None:
                continue

            if track_id not in crop_buffer:
                crop_buffer[track_id] = []
            if len(crop_buffer[track_id]) < CROP_NUM:
                crop_buffer[track_id].append(crop)

            if track_id not in logged_ids and len(crop_buffer.get(track_id, [])) >= CROP_NUM:
                embedding = get_average_embedding(
                    crop_buffer[track_id], resnet_model, device
                )
                timestamp = frame_number / fps
                results.append({
                    "track_id":  track_id,
                    "timestamp": timestamp,
                    "embedding": embedding
                })
                logged_ids.add(track_id)
                print(f"[{role.upper()}] Car {track_id} logged at {timestamp:.2f}s")

    cap.release()
    return results


# =============================================================================
# SECTION 5 -- MAIN MATCHING LOGIC
# =============================================================================

def run_matching():
    print("Processing entry camera...")
    entry_events = process_camera(
        ENTRY_VIDEO_PATH, ultra_model, resnet_model, device, "entry", roi=ENTRY_ROI
    )

    print("Processing exit camera...")
    exit_events = process_camera(
        EXIT_VIDEO_PATH, ultra_model, resnet_model, device, "exit", roi=EXIT_ROI, min_car_size=EXIT_MIN_CAR_SIZE
    )

    entry_buffer = {}
    for event in entry_events:
        key = event["track_id"]
        entry_buffer[key] = {
            "embedding": event["embedding"],
            "timestamp": event["timestamp"]
        }

    matched_results = []
    unmatched_exits = []

    for exit_event in exit_events:
        exit_embedding = exit_event["embedding"]
        exit_time      = exit_event["timestamp"]
        exit_id        = exit_event["track_id"]

        match_key = match_car(exit_embedding, exit_time, entry_buffer)

        if match_key is not None:
            entry_time   = entry_buffer[match_key]["timestamp"]
            transit_time = exit_time - entry_time
            matched_results.append({
                "entry_track_id":  match_key,
                "exit_track_id":   exit_id,
                "entry_time":      round(entry_time, 2),
                "exit_time":       round(exit_time, 2),
                "transit_seconds": round(transit_time, 2)
            })
            print(f"MATCH: Entry car {match_key} -> Exit car {exit_id} | Transit: {transit_time:.1f}s")
            del entry_buffer[match_key]
        else:
            unmatched_exits.append(exit_id)
            print(f"NO MATCH: Exit car {exit_id} at {exit_time:.2f}s")

    unmatched_entries = list(entry_buffer.keys())
    for key in unmatched_entries:
        t = entry_buffer[key]["timestamp"]
        print(f"UNMATCHED ENTRY: Car {key} entered at {t:.2f}s -- no exit found")

    print(f"\n--- RESULTS ---")
    print(f"Matched pairs:     {len(matched_results)}")
    print(f"Unmatched exits:   {len(unmatched_exits)}")
    print(f"Unmatched entries: {len(unmatched_entries)}")

    with open(OUTPUT_CSV, "w", newline="") as f:
        fieldnames = ["entry_track_id", "exit_track_id", "entry_time", "exit_time", "transit_seconds"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(matched_results)
    print(f"Results saved to {OUTPUT_CSV}")


# =============================================================================
# SECTION 6 -- VISUALIZATION
# =============================================================================

def save_sample(video_path, output_path, roi, min_car_size=0.0, duration_seconds=15):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_frames = int(duration_seconds * fps)

    roi_x1  = int(roi[0] * w)
    roi_y1  = int(roi[1] * h)
    roi_x2  = int(roi[2] * w)
    roi_y2  = int(roi[3] * h)
    roi_area = (roi_x2 - roi_x1) * (roi_y2 - roi_y1)

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    delay_frames = int(CROP_DELAY_SEC * fps)
    first_seen   = {}
    crop_counts  = {}
    logged_ids   = set()

    frame_number = 0
    while frame_number < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_number += 1

        roi_frame = frame[roi_y1:roi_y2, roi_x1:roi_x2]

        results_yolo = ultra_model.track(
            roi_frame, persist=True, classes=[2, 5, 7], verbose=False
        )
        result = results_yolo[0]

        if result.boxes is not None and result.boxes.id is not None:
            boxes     = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.cpu().numpy().astype(int)

            for i, track_id in enumerate(track_ids):
                x1 = int(boxes[i][0]) + roi_x1
                y1 = int(boxes[i][1]) + roi_y1
                x2 = int(boxes[i][2]) + roi_x1
                y2 = int(boxes[i][3]) + roi_y1

                if track_id not in first_seen:
                    first_seen[track_id] = frame_number
                frames_since_first = frame_number - first_seen[track_id]

                box_area = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1])
                size_frac = box_area / roi_area
                too_small = size_frac < min_car_size

                count = crop_counts.get(track_id, 0)

                # Simulate crop collection to keep counts in sync
                if (track_id not in logged_ids and frames_since_first >= delay_frames
                        and not too_small and count < CROP_NUM):
                    crop_counts[track_id] = count + 1
                    if crop_counts[track_id] >= CROP_NUM:
                        logged_ids.add(track_id)

                # Color and label:
                #   red    = waiting (delay) or too small
                #   yellow = actively collecting
                #   green  = done
                if track_id in logged_ids:
                    color = (0, 255, 0)
                    label = f"ID:{track_id} done"
                elif frames_since_first < delay_frames:
                    color = (0, 0, 255)
                    label = f"ID:{track_id} wait"
                elif too_small:
                    color = (0, 0, 255)
                    label = f"ID:{track_id} small {size_frac:.2f}"
                else:
                    color = (0, 220, 255)
                    label = f"ID:{track_id} crop {count+1}/{CROP_NUM}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label,
                            (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # Draw ROI boundary in blue
        cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 100, 0), 2)

        timestamp = frame_number / fps
        cv2.putText(frame, f"{timestamp:.1f}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
        out.write(frame)

    cap.release()
    out.release()
    print(f"Sample saved to {output_path}")


# =============================================================================
# SECTION 7 -- ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print("Saving annotated samples...")
    save_sample(ENTRY_VIDEO_PATH, "sample_entry.mp4", roi=ENTRY_ROI, duration_seconds=90)
    save_sample(EXIT_VIDEO_PATH,  "sample_exit.mp4",  roi=EXIT_ROI, min_car_size=EXIT_MIN_CAR_SIZE, duration_seconds=90)
    print("Done. Open sample_entry.mp4 and sample_exit.mp4 to review tracking.")
    run_matching()
