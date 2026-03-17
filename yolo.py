# =============================================================================
# ROUNDABOUT CAR TRANSIT TIMER
# YOLOv8 + Vehicle Re-ID across two camera feeds
# =============================================================================

# -----------------------------------------------------------------------------
# DEPENDENCIES — install via pip
# -----------------------------------------------------------------------------
# pip install ultralytics          # YOLOv8 detection + tracking
# pip install opencv-python        # video reading, frame handling, drawing
# pip install torch torchvision    # PyTorch backend for ReID model
# pip install numpy                # array math, embedding operations
# pip install scipy                # cosine similarity comparisons
# pip install torchreid            # pretrained vehicle ReID models (by KaiyangZhou)
#   └─ install guide: https://kaiyangzhou.github.io/deep-person-reid/install.html
# pip install pandas               # optional, for logging results to CSV

import cv2
import torch
import numpy as np
from ultralytics import YOLO
from scipy.spatial.distance import cosine
import torchreid
import pandas as pd


# =============================================================================
# SECTION 1 — CONFIGURATION
# =============================================================================
# - Set the file paths to your two video files (entry_video, exit_video)
# - Set the FPS of your videos (or read it dynamically from cv2.VideoCapture)
# - Define a SIMILARITY_THRESHOLD (float between 0-1) for ReID matching
#   └─ start around 0.4, tune based on false matches in testing
# - Define a MATCH_TIMEOUT in seconds — how long to keep an unmatched entry
#   before dropping it (e.g., car that backed out or was missed by exit cam)
# - Define how many crops to collect per car for embedding averaging (e.g., 5)


# =============================================================================
# SECTION 2 — LOAD MODELS
# =============================================================================

# --- 2A: Load YOLOv8 model ---
# - Use ultralytics YOLO() to load a pretrained model
# - 'yolov8n.pt' is the nano (fastest) model, good for starting out
# - If detection quality is poor, step up to yolov8m.pt or yolov8l.pt
# - LEARN: https://docs.ultralytics.com/models/yolov8/

# --- 2B: Load a pretrained vehicle ReID model via torchreid ---
# - Use torchreid.models.build_model() to load OSNet (osnet_x1_0)
#   which is fast, lightweight, and works well for vehicles
# - Load pretrained weights — torchreid can download them automatically
# - Set the model to eval() mode and move to GPU if available (torch.device)
# - LEARN: https://kaiyangzhou.github.io/deep-person-reid/user_guide.html
# - PRETRAINED VEHICLE WEIGHTS: search "VeRi-776 OSNet weights torchreid"
#   on the torchreid GitHub releases page


# =============================================================================
# SECTION 3 — HELPER FUNCTIONS
# =============================================================================

# --- 3A: extract_embedding(crop, reid_model) ---
# - Takes a single cropped car image (numpy array from cv2)
# - Resize to 256x128 (standard ReID input size)
# - Normalize using ImageNet mean/std values
# - Convert to a PyTorch tensor, add batch dimension, move to device
# - Run through the ReID model (no_grad) to get a 1D feature vector
# - Return the vector as a numpy array
# - LEARN: https://pytorch.org/tutorials/beginner/basics/transforms_tutorial.html

# --- 3B: get_average_embedding(list_of_crops, reid_model) ---
# - Takes a list of crop images collected over multiple frames for one car
# - Calls extract_embedding() on each crop
# - Stacks all vectors and takes np.mean() across them
# - Normalizes the result (divide by its L2 norm) so cosine similarity works correctly
# - Returns a single averaged embedding representing that car

# --- 3C: get_vehicle_crop(frame, bounding_box) ---
# - Takes the full video frame and a YOLOv8 bounding box (x1, y1, x2, y2)
# - Slices the frame array using the bounding box coordinates
# - Adds a small padding margin around the box so you don't clip the car edges
# - Returns the cropped image

# --- 3D: crosses_trigger_zone(bounding_box, trigger_line) ---
# - Defines a horizontal (or vertical) line in pixel coordinates for each camera
# - Checks if the bottom-center point of the bounding box has crossed that line
# - Returns True/False
# - LEARN: Think of this as a simple coordinate comparison — no special library needed

# --- 3E: match_car(exit_embedding, entry_buffer) ---
# - Takes the averaged embedding of an exiting car
# - Iterates over all entries currently stored in the entry_buffer dictionary
# - Computes cosine similarity between exit_embedding and each stored embedding
#   using scipy.spatial.distance.cosine (note: cosine() returns distance, so 1 - cosine() = similarity)
# - Returns the track/car ID of the best match if its similarity exceeds SIMILARITY_THRESHOLD
# - Returns None if no good match found
# - LEARN: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cosine.html


# =============================================================================
# SECTION 4 — PER-CAMERA PROCESSING FUNCTION
# =============================================================================

# --- process_camera(video_path, yolo_model, reid_model, role) ---
#   role is either "entry" or "exit"
#
# - Open the video with cv2.VideoCapture
# - Read FPS from cap.get(cv2.CAP_PROP_FPS) for accurate timestamp calculation
# - Loop over frames using cap.read()
#
# - On each frame:
#     - Run YOLOv8 in tracking mode: model.track(frame, persist=True, classes=[2,5,7])
#         └─ classes 2=car, 5=bus, 7=truck (COCO class IDs — filter to just cars if you want)
#         └─ persist=True keeps track IDs consistent across frames
#     - For each detected box:
#         - Get the track ID assigned by the tracker
#         - Get the bounding box coordinates
#         - Call get_vehicle_crop() to extract the car image
#         - Accumulate crops in a per-track dictionary (track_id → list of crops)
#           until you have reached your target crop count
#         - Once you have enough crops, call get_average_embedding()
#         - Check crosses_trigger_zone() to see if this car has hit the line
#         - If yes and we haven't already logged this track_id:
#             - Record timestamp = frame_number / fps
#             - Store { track_id: { embedding, timestamp } } in a results list
#
# - After processing all frames, return the results list
#
# - LEARN (YOLOv8 tracking): https://docs.ultralytics.com/modes/track/
# - LEARN (cv2 video):        https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html


# =============================================================================
# SECTION 5 — MAIN MATCHING LOGIC
# =============================================================================

# - Run process_camera() on the entry video → get entry_results list
# - Run process_camera() on the exit video  → get exit_results list
#
# - Build entry_buffer: a dictionary of { entry_key: {embedding, entry_timestamp} }
#   for every car that crossed the entry trigger
#
# - For each car in exit_results:
#     - Call match_car() against the entry_buffer
#     - If a match is found:
#         - Compute transit_time = exit_timestamp - entry_timestamp
#         - Log: { matched_entry_id, exit_id, entry_time, exit_time, transit_time }
#         - Remove the matched entry from the buffer (so it can't double-match)
#     - If no match:
#         - Log as unmatched exit (car not seen at entry, or ReID failed)
#
# - After all exits processed, anything remaining in entry_buffer with
#   (current_time - entry_timestamp) > MATCH_TIMEOUT is logged as unmatched entry
#
# - Print or save results to a CSV using pandas


# =============================================================================
# SECTION 6 — OUTPUT & VISUALIZATION (optional but useful for debugging)
# =============================================================================

# - Use cv2 to draw bounding boxes and track IDs on frames while processing
# - Draw the trigger line on each video feed so you can visually verify placement
# - Print a summary table: car ID | entry time | exit time | transit duration
# - Optionally write annotated frames to a new video file using cv2.VideoWriter
# - LEARN: https://docs.opencv.org/4.x/dc/da5/tutorial_py_drawing_functions.html


# =============================================================================
# SECTION 7 — ENTRY POINT
# =============================================================================

# if __name__ == "__main__":
#     - Load config values
#     - Load both models (Section 2)
#     - Run process_camera() on both videos
#     - Run matching logic (Section 5)
#     - Output results


# =============================================================================
# KEY RESOURCES SUMMARY
# =============================================================================
# YOLOv8 docs:            https://docs.ultralytics.com/
# YOLOv8 tracking guide:  https://docs.ultralytics.com/modes/track/
# torchreid install:      https://kaiyangzhou.github.io/deep-person-reid/install.html
# torchreid user guide:   https://kaiyangzhou.github.io/deep-person-reid/user_guide.html
# torchreid GitHub:       https://github.com/KaiyangZhou/deep-person-reid
# OSNet paper:            https://arxiv.org/abs/1905.00953  (understand what the model is doing)
# VeRi-776 dataset:       https://vehiclereid.github.io/VeRi/  (what the weights were trained on)
# PyTorch transforms:     https://pytorch.org/vision/stable/transforms.html
# OpenCV video I/O:       https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html
# Cosine similarity:      https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cosine.html
# ByteTrack paper:        https://arxiv.org/abs/2110.06864  (understand the tracker YOLOv8 uses)