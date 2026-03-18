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
# - Set the file paths to your two video files (entry_video, exit_video)
# - Set the FPS of your videos (or read it dynamically from cv2.VideoCapture)
# - Define a SIMILARITY_THRESHOLD (float between 0-1) for ReID matching
#   └─ start around 0.4, tune based on false matches in testing
# - Define a MATCH_TIMEOUT in seconds — how long to keep an unmatched entry
#   before dropping it (e.g., car that backed out or was missed by exit cam)
# - Define how many crops to collect per car for embedding averaging (e.g., 5)

VID_PATH = 'video.mp4'
FPS = 30
SIM_THRESHOLD = 0.4 
MATCH_TIMEOUT = 150
CROP_NUM = 5


# =============================================================================
# SECTION 2 — LOAD MODELS
# =============================================================================

# --- 2A: Load YOLOv8 model ---
# - Use ultralytics YOLO() to load a pretrained model
# - 'yolov8n.pt' is the nano (fastest) model, good for starting out
# - If detection quality is poor, step up to yolov8m.pt or yolov8l.pt
# - LEARN: https://docs.ultralytics.com/models/yolov8/

ultra_model = YOLO('yolov8n.pt')

# --- 2B: Load a pretrained ResNet50 as a feature extractor ---
# - Load ResNet50 using torchvision.models.resnet50(weights="IMAGENET1K_V1")
# - Strip the final fully-connected classification layer by replacing it with
#   torch.nn.Identity() — this makes the model output a raw 2048-dim feature
#   vector instead of ImageNet class probabilities
# - Set the model to eval() mode and move to GPU if available (torch.device)
# - No extra downloads needed — torchvision fetches the weights automatically
# - LEARN: https://pytorch.org/vision/stable/models/resnet.html
# - NOTE: if matching accuracy feels low in testing, swap to CLIP as a drop-in
#   upgrade: pip install openai-clip  (richer embeddings, same usage pattern)
 
resnet_model = models.resnet50(weights='IMAGENET1K_V2')

resnet_model.eval()



# =============================================================================
# SECTION 3 -- HELPER FUNCTIONS
# =============================================================================
# Helper functions are small reusable pieces of logic you'll call repeatedly
# throughout the program. Write each one as a standalone def block.
# Test each one individually before moving on -- print their output to verify
# they're doing what you expect.
 
def extract_embedding(crop, resnet_model, device):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224,224),
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



# --- 3A: def extract_embedding(crop, resnet_model, device) ---
#
# PURPOSE: Takes a single car crop image and returns a 2048-number fingerprint.
#
# PARAMETERS:
#   crop         -> a numpy array (H x W x 3) -- a cropped image of one car
#                  from one video frame. This is what cv2 gives you when you
#                  slice a frame using bounding box coordinates.
#   resnet_model -> the ResNet50 model you loaded in Section 2B
#   device       -> the torch.device("cuda" or "cpu") from Section 2B
#
# STEP BY STEP:
#
#   Step 1 -- Define a transform pipeline using torchvision.transforms.Compose:
#     This is a list of preprocessing steps applied in order to the crop image.
#     ResNet50 was trained on images in a specific format, so you must match
#     that exact format or the embeddings will be garbage.
#
#     transform = transforms.Compose([
#         transforms.ToPILImage(),
#         # cv2 gives you a numpy array. PyTorch transforms expect a PIL Image.
#         # This converts the numpy array to PIL format. Always first.
#
#         transforms.Resize((224, 224)),
#         # Resize the crop to 224x224 pixels -- ResNet50's expected input size.
#         # All crops become the same size regardless of how big the car was
#         # in the original frame.
#
#         transforms.ToTensor(),
#         # Convert the PIL Image to a PyTorch tensor and scale pixel values
#         # from 0-255 integers to 0.0-1.0 floats. Shape becomes (3, 224, 224).
#
#         transforms.Normalize(
#             mean=[0.485, 0.456, 0.406],
#             std=[0.229, 0.224, 0.225]
#         ),
#         # These exact numbers are the mean and standard deviation of the
#         # ImageNet dataset that ResNet50 was trained on. Normalizing with
#         # these values puts your image in the same "scale" the model expects.
#         # Always use these exact values when working with ImageNet models.
#     ])
#
#     NOTE: Define this transform inside the function body, OR define it once
#     globally outside all functions and just reference it here. Either works,
#     but defining it globally is slightly more efficient.
#
#   Step 2 -- Apply the transform to the crop:
#     tensor = transform(crop)
#     # tensor is now shape (3, 224, 224) -- 3 color channels, 224x224 pixels
#
#   Step 3 -- Add a batch dimension:
#     tensor = tensor.unsqueeze(0)
#     # Neural networks always expect a batch of images, even if it's just one.
#     # unsqueeze(0) adds a dimension at position 0, making shape (1, 3, 224, 224)
#     # which means "a batch of 1 image with 3 channels at 224x224"
#
#   Step 4 -- Move tensor to the same device as the model:
#     tensor = tensor.to(device)
#
#   Step 5 -- Run the model and get the embedding:
#     with torch.no_grad():
#         embedding = resnet_model(tensor)
#     # torch.no_grad() tells PyTorch not to track gradients during this forward
#     # pass. Since we're not training, we don't need gradients. This saves memory
#     # and makes inference faster. Always use it during inference.
#     # embedding is a tensor of shape (1, 2048)
#
#   Step 6 -- Convert to a flat numpy array and return:
#     return embedding.squeeze().cpu().numpy()
#     # .squeeze() removes the batch dimension -> shape becomes (2048,)
#     # .cpu() moves it back from GPU to regular memory (needed before numpy)
#     # .numpy() converts from PyTorch tensor to a regular numpy array
#     # Final result: a 1D array of 2048 floats -- the car's fingerprint
#
# LEARN MORE: https://pytorch.org/vision/stable/transforms.html

def get_average_embedding(crops, resnet_model, device):
    embeddings = [extract_embedding(crop, resnet_model, device) for crop in crops]
    stacked = np.stack(embeddings)
    averaged = np.mean(stacked, axis=0)
    norm = np.linalg.norm(averaged)
    normalized = averaged / norm
    return normalized

 
# --- 3B: def get_average_embedding(crops, resnet_model, device) ---
#
# PURPOSE: Takes a list of several crop images of the same car and returns
#          one single averaged fingerprint that represents that car.
#          More robust than using a single crop because it smooths out
#          motion blur, partial occlusion, and angle changes.
#
# PARAMETERS:
#   crops        -> a Python list of numpy array crops, e.g. [crop1, crop2, crop3]
#   resnet_model -> the ResNet50 model from Section 2B
#   device       -> the torch.device from Section 2B
#
# STEP BY STEP:
#
#   Step 1 -- Call extract_embedding() on every crop in the list:
#     embeddings = [extract_embedding(crop, resnet_model, device) for crop in crops]
#     # This is a list comprehension -- it's shorthand for a for loop that
#     # builds a new list. Each item in embeddings is a (2048,) numpy array.
#
#   Step 2 -- Stack all embeddings into a 2D array and average across them:
#     stacked = np.stack(embeddings)
#     # np.stack turns a list of (2048,) arrays into one (N, 2048) array
#     # where N = number of crops.
#
#     averaged = np.mean(stacked, axis=0)
#     # np.mean with axis=0 averages across all N rows, giving back one
#     # (2048,) array that represents the average fingerprint.
#
#   Step 3 -- Normalize the averaged embedding (L2 normalization):
#     norm = np.linalg.norm(averaged)
#     normalized = averaged / norm
#     # np.linalg.norm computes the "length" of the vector.
#     # Dividing by the length scales the vector so its total length = 1.
#     # This is called L2 normalization. It's important because it makes
#     # cosine similarity in Section 3E reduce to a simple dot product,
#     # and it ensures all fingerprints are on the same scale regardless
#     # of the car's size or brightness in the video.
#
#   Step 4 -- Return the normalized embedding:
#     return normalized
 
def get_vehicle_crop(frame, box):
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    pad = 10
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(frame.shape[1], x2 + pad)
    y2 = min(frame.shape[0], y2 + pad)
    crop = frame[y1:y2, x1:x2]
    return crop
 
# --- 3C: def get_vehicle_crop(frame, box) ---
#
# PURPOSE: Cuts out just the car from a full video frame using the bounding
#          box coordinates that YOLOv8 gives you.
#
# PARAMETERS:
#   frame -> the full video frame as a numpy array (H x W x 3), from cap.read()
#   box   -> a list or array of [x1, y1, x2, y2] pixel coordinates
#            where (x1, y1) is the top-left corner and (x2, y2) is bottom-right
#
# STEP BY STEP:
#
#   Step 1 -- Unpack the bounding box and convert to integers:
#     x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
#     # YOLOv8 gives coordinates as floats. You need integers to use them
#     # as array indices when slicing a numpy array.
#
#   Step 2 -- Add a small padding margin so you don't clip the car's edges:
#     pad = 10
#     x1 = max(0, x1 - pad)
#     y1 = max(0, y1 - pad)
#     x2 = min(frame.shape[1], x2 + pad)
#     y2 = min(frame.shape[0], y2 + pad)
#     # frame.shape is (height, width, channels), so:
#     #   frame.shape[0] = frame height (max valid Y coordinate)
#     #   frame.shape[1] = frame width  (max valid X coordinate)
#     # max(0, ...) and min(frame.shape..., ...) clamp the coordinates so
#     # you never go outside the frame boundaries, which would crash the slice.
#
#   Step 3 -- Slice the frame to get just the car:
#     crop = frame[y1:y2, x1:x2]
#     # NumPy array slicing. In image arrays, the first index is Y (rows)
#     # and the second is X (columns). This gives you a smaller array
#     # containing just the pixels inside the bounding box.
#
#   Step 4 -- Return the crop:
#     return crop
#     # If the crop is empty (e.g., box was invalid), it's good practice to
#     # check: if crop.size == 0: return None
#     # Then check for None in any function that calls get_vehicle_crop()
 
 
# --- 3D: def crosses_trigger_zone(box, trigger_y) ---
#
# PURPOSE: Checks whether a car has crossed the virtual trigger line.
#          Returns True if yes, False if no.
#
# PARAMETERS:
#   box       -> [x1, y1, x2, y2] bounding box coordinates from YOLOv8
#   trigger_y -> the Y pixel coordinate of your trigger line (from Section 1)
#
# STEP BY STEP:
#
#   Step 1 -- Find the bottom edge of the bounding box:
#     bottom_y = int(box[3])
#     # box[3] is y2, the bottom edge of the bounding box.
#     # We use the bottom of the box (where the wheels are) rather than the
#     # center so that the car "crosses" the line when it physically reaches it.
#
#   Step 2 -- Check if bottom_y has passed the trigger line:
#     return bottom_y >= trigger_y
#     # In video coordinates, Y=0 is at the TOP of the frame and increases
#     # downward. So "crossing downward" means the Y value is increasing.
#     # If the bottom of the car is at or below the trigger line, return True.
#     # If your camera is positioned differently and cars approach from below,
#     # flip this to: return bottom_y <= trigger_y
 
 
# --- 3E: def match_car(exit_embedding, entry_buffer) ---
#
# PURPOSE: Given a car's fingerprint from the exit camera, find the best
#          matching car in the entry buffer (cars that already entered).
#          Returns the key of the best match, or None if no good match found.
#
# PARAMETERS:
#   exit_embedding -> a normalized (2048,) numpy array from get_average_embedding()
#   entry_buffer   -> a dictionary where each key is a unique car identifier
#                     and each value is a dict containing at least:
#                     { "embedding": <numpy array>, "timestamp": <float> }
#                     This buffer is built and maintained in Section 5.
#
# STEP BY STEP:
#
#   Step 1 -- Set up tracking variables:
#     best_match_key = None
#     best_similarity = -1.0
#     # We'll loop through all entries and keep track of the best one found.
#
#   Step 2 -- Loop through every car currently in the entry buffer:
#     for key, data in entry_buffer.items():
#
#       Step 2a -- Get the stored entry embedding:
#         entry_embedding = data["embedding"]
#
#       Step 2b -- Compute cosine similarity using numpy dot product:
#         similarity = np.dot(exit_embedding, entry_embedding)
#         # Because both embeddings are L2-normalized (from Step 3 in 3B),
#         # the dot product equals the cosine similarity directly.
#         # Result is between -1.0 and 1.0. For similar cars expect 0.5-0.9.
#         # For completely different cars expect 0.0-0.3.
#
#       Step 2c -- Check if this is the best match so far:
#         if similarity > best_similarity:
#             best_similarity = similarity
#             best_match_key = key
#
#   Step 3 -- After the loop, check if the best match meets the threshold:
#     if best_similarity >= SIMILARITY_THRESHOLD:
#         return best_match_key
#     else:
#         return None
#     # If no entry in the buffer scored above the threshold, return None.
#     # The calling code in Section 5 will handle this as an unmatched exit.
#
# LEARN MORE: https://numpy.org/doc/stable/reference/generated/numpy.dot.html
 
 
# =============================================================================
# SECTION 4 -- PER-CAMERA PROCESSING FUNCTION
# =============================================================================
#
# --- def process_camera(video_path, yolo_model, resnet_model, device, trigger_y, role) ---
#
# PURPOSE: Processes one entire video from start to finish. For every car that
#          crosses the trigger line, it records the timestamp and a fingerprint.
#          Returns a list of all such events, one per car.
#
# PARAMETERS:
#   video_path   -> string path to the video file (from your config in Section 1)
#   yolo_model   -> the YOLO model loaded in Section 2A
#   resnet_model -> the ResNet50 model loaded in Section 2B
#   device       -> torch.device from Section 2B
#   trigger_y    -> the Y coordinate trigger line for this camera (from Section 1)
#   role         -> a string, either "entry" or "exit", just for logging/printing
#
# WHAT IT RETURNS:
#   A list of dictionaries. Each dict represents one car that crossed the line:
#   [
#     { "track_id": 3, "timestamp": 12.4, "embedding": <numpy array> },
#     { "track_id": 7, "timestamp": 31.8, "embedding": <numpy array> },
#     ...
#   ]
#
# STEP BY STEP:
#
#   Step 1 -- Open the video file:
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print(f"Error: could not open video {video_path}")
#         return []
#     # cv2.VideoCapture opens the file and prepares it for reading.
#     # Always check isOpened() -- if the file path is wrong it fails silently.
#
#   Step 2 -- Read the FPS (frames per second) from the video:
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     # FPS is needed to convert frame numbers to real timestamps.
#     # A frame number alone means nothing -- you need to know how many
#     # frames happen per second to calculate elapsed time.
#
#   Step 3 -- Set up data structures you'll use while looping:
#     crop_buffer = {}
#     # A dictionary: { track_id: [crop1, crop2, crop3, ...] }
#     # As you see a car across multiple frames, you accumulate its crops here
#     # until you have enough to build a good average embedding.
#
#     logged_ids = set()
#     # A set of track IDs that have already been recorded as crossing the line.
#     # Prevents logging the same car multiple times if it lingers near the line.
#
#     results = []
#     # The list you'll return at the end. Each entry is one car event.
#
#     frame_number = 0
#     # You'll manually increment this each iteration to track position in video.
#
#   Step 4 -- Start the main frame loop:
#     while True:
#         ret, frame = cap.read()
#         # cap.read() grabs the next frame.
#         # ret is True if a frame was successfully read, False at end of video.
#         # frame is a numpy array (H x W x 3) of the raw image pixels.
#
#         if not ret:
#             break
#         # When ret is False, we've hit the end of the video. Exit the loop.
#
#         frame_number += 1
#
#   Step 5 -- Inside the loop: run YOLOv8 tracking on the current frame:
#         results_yolo = yolo_model.track(
#             frame,
#             persist=True,
#             classes=[2, 5, 7],
#             verbose=False
#         )
#         # yolo_model.track() runs detection AND tracking in one call.
#         # persist=True is critical -- it tells the tracker to remember track
#         #   IDs from previous frames. Without this, every frame gets new IDs.
#         # classes=[2, 5, 7] filters to only cars(2), buses(5), trucks(7).
#         #   These are COCO dataset class IDs. Remove buses/trucks if you want.
#         # verbose=False silences the per-frame console output (gets very noisy).
#         #
#         # results_yolo is a list. For a single frame input, use results_yolo[0]
#         # to get the detection result object for that frame.
#
#   Step 6 -- Inside the loop: extract boxes and track IDs from YOLOv8 output:
#         result = results_yolo[0]
#         # result.boxes contains all detected bounding boxes for this frame.
#         # Each box has:
#         #   .xyxy  -> tensor of shape (N, 4) with [x1, y1, x2, y2] per box
#         #   .id    -> tensor of shape (N, 1) with track IDs -- can be None
#         #             if no objects were tracked this frame
#         #   .cls   -> tensor of shape (N, 1) with class IDs
#         #   .conf  -> tensor of shape (N, 1) with confidence scores
#
#         if result.boxes is None or result.boxes.id is None:
#             continue
#         # If nothing was detected or tracking hasn't assigned IDs yet, skip frame.
#
#         boxes = result.boxes.xyxy.cpu().numpy()
#         # Convert bounding boxes from GPU tensor to numpy array.
#         # Shape: (N, 4) where N = number of detected objects this frame.
#
#         track_ids = result.boxes.id.cpu().numpy().astype(int)
#         # Convert track IDs to a numpy array of integers.
#         # Shape: (N,) -- one ID per detected box.
#
#   Step 7 -- Inside the loop: process each detected car:
#         for i, track_id in enumerate(track_ids):
#             box = boxes[i]
#             # boxes[i] = [x1, y1, x2, y2] for the i-th detected object.
#             # track_id is the consistent integer ID for this car across frames.
#
#             # Get the cropped car image using your helper from Section 3C:
#             crop = get_vehicle_crop(frame, box)
#             if crop is None or crop.size == 0:
#                 continue
#             # Skip if the crop is empty (malformed bounding box).
#
#             # Add this crop to the buffer for this track_id:
#             if track_id not in crop_buffer:
#                 crop_buffer[track_id] = []
#             if len(crop_buffer[track_id]) < CROPS_NEEDED:
#                 crop_buffer[track_id].append(crop)
#             # Only collect up to CROPS_NEEDED crops. Once you have enough,
#             # stop adding so you don't use too much memory.
#
#             # Check if this car has crossed the trigger line:
#             if crosses_trigger_zone(box, trigger_y) and track_id not in logged_ids:
#
#                 # Only proceed if we have enough crops to build a good embedding:
#                 if len(crop_buffer.get(track_id, [])) >= CROPS_NEEDED:
#
#                     # Build the average embedding for this car:
#                     embedding = get_average_embedding(
#                         crop_buffer[track_id], resnet_model, device
#                     )
#
#                     # Calculate the real-world timestamp in seconds:
#                     timestamp = frame_number / fps
#                     # Example: frame 450 at 30fps = 15.0 seconds into the video.
#
#                     # Record this car event:
#                     results.append({
#                         "track_id": track_id,
#                         "timestamp": timestamp,
#                         "embedding": embedding
#                     })
#
#                     # Mark this ID so we don't log it again:
#                     logged_ids.add(track_id)
#
#                     print(f"[{role.upper()}] Car {track_id} crossed at {timestamp:.2f}s")
#                     # Useful for debugging -- you'll see events print in real time.
#
#   Step 8 -- After the loop finishes, release the video and return results:
#     cap.release()
#     # Always release the VideoCapture when done. This frees the file handle.
#     return results
#
# LEARN MORE (YOLOv8 tracking): https://docs.ultralytics.com/modes/track/
# LEARN MORE (cv2 video):       https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html
 
 
# =============================================================================
# SECTION 5 -- MAIN MATCHING LOGIC
# =============================================================================
#
# This section ties everything together. It runs both cameras, then matches
# exit events to entry events to compute transit times.
#
# STEP BY STEP:
#
#   Step 1 -- Process both videos:
#     print("Processing entry camera...")
#     entry_events = process_camera(
#         ENTRY_VIDEO_PATH, yolo_model, resnet_model, device, ENTRY_TRIGGER_Y, "entry"
#     )
#
#     print("Processing exit camera...")
#     exit_events = process_camera(
#         EXIT_VIDEO_PATH, yolo_model, resnet_model, device, EXIT_TRIGGER_Y, "exit"
#     )
#
#     NOTE: This processes them sequentially (one after the other). If you want
#     to process both in parallel (true real-time), look into Python's
#     threading or multiprocessing module -- but do sequential first.
#
#   Step 2 -- Build the entry buffer from entry_events:
#     entry_buffer = {}
#     for event in entry_events:
#         key = event["track_id"]
#         entry_buffer[key] = {
#             "embedding": event["embedding"],
#             "timestamp": event["timestamp"]
#         }
#     # This is just reorganizing the list into a dictionary for fast lookup.
#     # Key = track ID, value = the embedding + timestamp for that car.
#
#   Step 3 -- Set up lists to store matched and unmatched results:
#     matched_results = []
#     unmatched_exits = []
#
#   Step 4 -- Loop through every exit event and try to find a matching entry:
#     for exit_event in exit_events:
#         exit_embedding = exit_event["embedding"]
#         exit_time      = exit_event["timestamp"]
#         exit_id        = exit_event["track_id"]
#
#         # Call your matching function from Section 3E:
#         match_key = match_car(exit_embedding, entry_buffer)
#
#         if match_key is not None:
#             # A match was found!
#             entry_time   = entry_buffer[match_key]["timestamp"]
#             transit_time = exit_time - entry_time
#
#             matched_results.append({
#                 "entry_track_id":  match_key,
#                 "exit_track_id":   exit_id,
#                 "entry_time":      round(entry_time, 2),
#                 "exit_time":       round(exit_time, 2),
#                 "transit_seconds": round(transit_time, 2)
#             })
#
#             print(f"MATCH: Entry car {match_key} -> Exit car {exit_id} | Transit: {transit_time:.1f}s")
#
#             # Remove matched entry from buffer so it can't match again:
#             del entry_buffer[match_key]
#
#         else:
#             # No match found for this exit car
#             unmatched_exits.append(exit_id)
#             print(f"NO MATCH: Exit car {exit_id} at {exit_time:.2f}s")
#
#   Step 5 -- Handle leftover entries that never matched:
#     unmatched_entries = list(entry_buffer.keys())
#     # Whatever is still in entry_buffer never found an exit match.
#     for key in unmatched_entries:
#         t = entry_buffer[key]["timestamp"]
#         print(f"UNMATCHED ENTRY: Car {key} entered at {t:.2f}s -- no exit found")
#
#   Step 6 -- Print a final summary:
#     print(f"\n--- RESULTS ---")
#     print(f"Matched pairs:     {len(matched_results)}")
#     print(f"Unmatched exits:   {len(unmatched_exits)}")
#     print(f"Unmatched entries: {len(unmatched_entries)}")
#
#   Step 7 -- Save matched results to CSV:
#     with open(OUTPUT_CSV, "w", newline="") as f:
#         fieldnames = ["entry_track_id", "exit_track_id", "entry_time", "exit_time", "transit_seconds"]
#         writer = csv.DictWriter(f, fieldnames=fieldnames)
#         writer.writeheader()
#         writer.writerows(matched_results)
#     print(f"Results saved to {OUTPUT_CSV}")
#     #
#     # csv.DictWriter writes a list of dictionaries as rows in a CSV file.
#     # fieldnames defines the column order.
#     # writeheader() writes the first row (the column names).
#     # writerows() writes all the data rows at once.
#     #
#     # LEARN MORE: https://docs.python.org/3/library/csv.html
 
 
# =============================================================================
# SECTION 6 -- VISUALIZATION (optional but very helpful for debugging)
# =============================================================================
#
# This section is optional but highly recommended while you're developing.
# It lets you visually verify that detection, tracking, and trigger zones
# are working correctly before you trust the matching results.
#
# You can add this drawing code inside your process_camera() loop in Section 4,
# right after Step 6 where you extract boxes and track_ids.
#
# DRAWING BOUNDING BOXES:
#   for i, track_id in enumerate(track_ids):
#       box = boxes[i]
#       x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
#       cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#       # Draws a green rectangle around each detected car.
#       # (0, 255, 0) is BGR color for green. 2 is line thickness in pixels.
#
#       cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 10),
#                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#       # Draws the track ID label just above each bounding box.
#       # (x1, y1 - 10) positions it 10 pixels above the top-left corner.
#
# DRAWING THE TRIGGER LINE:
#   cv2.line(frame, (0, trigger_y), (frame.shape[1], trigger_y), (0, 0, 255), 2)
#   # Draws a red horizontal line across the full width of the frame
#   # at the trigger_y position. Lets you visually verify line placement.
#
# SHOWING THE FRAME IN A WINDOW:
#   cv2.imshow(f"{role} camera", frame)
#   if cv2.waitKey(1) & 0xFF == ord('q'):
#       break
#   # cv2.imshow() pops up a window showing the current frame with annotations.
#   # cv2.waitKey(1) waits 1ms between frames (keeps the window responsive).
#   # Pressing 'q' breaks out of the loop early so you can stop the video.
#   # Call cv2.destroyAllWindows() after the loop to close the window.
#
# SAVING ANNOTATED VIDEO TO FILE:
#   # Set this up BEFORE the loop (after you open the VideoCapture):
#   frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#   frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#   fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#   out = cv2.VideoWriter("annotated_entry.mp4", fourcc, fps, (frame_width, frame_height))
#
#   # Inside the loop, after drawing, write the frame:
#   out.write(frame)
#
#   # After the loop, before returning:
#   out.release()
#
# LEARN MORE: https://docs.opencv.org/4.x/dc/da5/tutorial_py_drawing_functions.html
 
 
# =============================================================================
# SECTION 7 -- ENTRY POINT
# =============================================================================
#
# In Python, the block below only runs when you execute this file directly
# (e.g., "python roundabout_tracker.py"). It does NOT run if this file is
# imported as a module by another file. This is standard Python practice
# and you should always structure your scripts this way.
#
# if __name__ == "__main__":
#
#     Step 1 -- Print a startup message so you know the script started:
#       print("Starting roundabout transit timer...")
#
#     Step 2 -- Load both models (from Section 2):
#       yolo_model   = YOLO("yolov8n.pt")
#       device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#       resnet_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
#       resnet_model.fc = torch.nn.Identity()
#       resnet_model.eval()
#       resnet_model = resnet_model.to(device)
#       print(f"Models loaded. Using device: {device}")
#
#     Step 3 -- Run the matching logic from Section 5.
#       You can either write Section 5's code directly here inside this block,
#       or wrap it all in its own function called run_matching() and call that
#       function here. Wrapping it in a function is cleaner and easier to debug.
 
 
# =============================================================================
# KEY RESOURCES SUMMARY
# =============================================================================
# YOLOv8 docs:               https://docs.ultralytics.com/
# YOLOv8 tracking guide:     https://docs.ultralytics.com/modes/track/
# ResNet50 in torchvision:   https://pytorch.org/vision/stable/models/resnet.html
# PyTorch transforms:        https://pytorch.org/vision/stable/transforms.html
# torch.nn.Identity:         https://pytorch.org/docs/stable/generated/torch.nn.Identity.html
# torch.no_grad:             https://pytorch.org/docs/stable/generated/torch.no_grad.html
# OpenCV VideoCapture:       https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html
# OpenCV drawing functions:  https://docs.opencv.org/4.x/dc/da5/tutorial_py_drawing_functions.html
# numpy dot product:         https://numpy.org/doc/stable/reference/generated/numpy.dot.html
# numpy linalg.norm:         https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html
# Python csv module:         https://docs.python.org/3/library/csv.html
# ByteTrack paper:           https://arxiv.org/abs/2110.06864  (understand the tracker YOLOv8 uses)
# CLIP (upgrade option):     https://github.com/openai/CLIP