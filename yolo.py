# =============================================================================
# IMPORTS
# What is an import? Python doesn't load every tool automatically — you have to
# explicitly ask for the ones you need. Each line below brings in an external
# library (a package of pre-written code) so we can use its functions.
# =============================================================================

import cv2
# cv2 is the OpenCV library. OpenCV (Open Computer Vision) is a massive toolkit
# for anything involving images or video. It can open video files, read them
# frame by frame, draw colored boxes and text on frames, and save new video files.
# The name "cv2" is just the Python package name — it's always imported as cv2.

import torch
# PyTorch is a deep learning framework made by Meta. Think of it as a math engine
# that's specially designed to run neural networks. It handles the heavy numerical
# math (multiplying giant matrices of numbers) and can do it extremely fast on a
# GPU. We use it to run our ResNet model.

import numpy as np
# NumPy (Numerical Python) is the foundation of almost all Python data science.
# It gives us "arrays" — grids of numbers that support fast math operations.
# Instead of looping through numbers one at a time, NumPy can add, multiply,
# or average thousands of numbers in one line. We alias it as "np" so we can
# write np.something instead of numpy.something everywhere.

import torchvision.models as models
# torchvision is a companion library to PyTorch specifically for computer vision.
# torchvision.models contains dozens of famous pre-built neural network
# architectures (ResNet, VGG, EfficientNet, etc.) with the option to load
# weights that were already trained on huge datasets. We import it as "models"
# so we can write models.resnet50() instead of the full path.

import torchvision.transforms as transforms
# torchvision.transforms is a toolkit of image preprocessing operations.
# Neural networks are picky — they need images in a very specific format,
# size, and numerical range before they can process them. transforms gives us
# a clean way to chain those preprocessing steps together. We alias it as
# "transforms" for short.

from ultralytics import YOLO
# Ultralytics is the company that makes YOLOv8. YOLO stands for
# "You Only Look Once" — it's a family of extremely fast object detection
# models. We import just the YOLO class from the ultralytics package.
# This class handles loading the model, running detection, AND tracking
# objects across frames (assigning consistent IDs to the same car over time).

import csv
# csv is a built-in Python standard library module (no installation needed).
# It handles reading and writing Comma-Separated Value files, which are
# plain text files that spreadsheet apps (Excel, Google Sheets) can open.
# Each row is a line of text; each column is separated by a comma.


# =============================================================================
# SECTION 1 — CONFIGURATION
# These are all the settings that control how the pipeline behaves.
# They're defined at the top as constants (ALL_CAPS by convention) so you can
# easily find and change them without digging through the rest of the code.
# =============================================================================

ENTRY_VIDEO_PATH = 'entry_file.mp4'
# The file path to the video recorded by the ENTRY camera.
# This camera watches cars as they arrive and pull into the roundabout.
# The string 'entry_file.mp4' assumes the file is in the same folder as
# this script. You could also use an absolute path like '/Users/you/videos/entry.mp4'.

EXIT_VIDEO_PATH  = 'exit_file.mp4'
# The file path to the video recorded by the EXIT camera.
# This camera watches cars as they leave the roundabout.

OUTPUT_CSV       = 'results.csv'
# The name of the file we'll write our final results into.
# Each row will contain one matched pair: entry car ID, exit car ID,
# when it entered, when it exited, and how long the transit took.

FPS              = 10
# Frames Per Second — how many individual still images the video contains
# per second of footage. This is defined here as a fallback reference value,
# but the actual FPS is read directly from each video file's metadata using
# cap.get(cv2.CAP_PROP_FPS) later on.

SIM_THRESHOLD    = 0.7
# Similarity threshold for matching cars between cameras.
# When we compare the "fingerprint" of an exit car to an entry car, we get
# a similarity score between 0.0 (completely different) and 1.0 (identical).
# If the score is 0.7 or higher, we call it a match. If it's below 0.7,
# we don't match — the cars look too different to be the same vehicle.
# Raise this to be stricter (fewer false matches), lower it to be looser
# (fewer missed matches, but more false positives).

MATCH_TIMEOUT    = 150
# Maximum number of seconds a car can take to travel from entry to exit
# before we stop considering it as a potential match.
# If a car entered 151 seconds ago, we won't try to match it anymore —
# it's been too long and it's probably not still in the roundabout.
# This prevents stale entry events from being incorrectly matched.

CROP_NUM         = 5
# How many photos (crops) of each car we collect before computing its fingerprint.
# We take multiple photos of the same car from different frames rather than
# just one, then average the resulting fingerprints together. This makes the
# final fingerprint more stable — one blurry frame won't ruin everything.

CROP_DELAY_SEC   = 1
# How many seconds to wait after we first spot a car before we start taking photos.
# When a car first appears at the edge of the camera's view, it's usually small
# and partially out of frame. Waiting 1 second gives it time to move closer and
# be more fully visible, so our crop photos are cleaner and more useful.

# --------------------------------------------------------------------------
# ROI — Region of Interest
# --------------------------------------------------------------------------
# The full video frame contains a lot of irrelevant stuff — buildings, parked
# cars, trees, pedestrians, etc. We define a rectangular "Region of Interest"
# for each camera so that YOLO only analyzes the part of the frame where cars
# actually pass through. Everything outside this box is ignored.
#
# Each ROI is defined as four fractions (0.0 to 1.0):
#   (left, top, right, bottom)
# where 0.0 means the very left/top edge and 1.0 means the very right/bottom
# edge of the full frame. Fractions let us define the ROI in a way that works
# regardless of the video's pixel resolution.
#
# Example: (0.25, 0.0, 1.0, 1.0) means:
#   left   = 25% from the left edge
#   top    = 0% from the top (the very top)
#   right  = 100% (the very right edge)
#   bottom = 100% (the very bottom edge)
#   → This gives you the rightmost 75% of the frame, full height.

ENTRY_ROI          = (0.0,  0.167, 1.0, 1.0)
# Entry camera ROI: the bottom 5/6 of the frame, full width.
# The top 1/6 (0.0 to 0.167 in the vertical axis) is excluded — it probably
# shows sky or a distant road that we don't care about.

EXIT_ROI           = (0.25, 0.0,   1.0, 1.0)
# Exit camera ROI: the rightmost 3/4 of the frame, full height.
# The leftmost 1/4 is excluded — it probably shows the wrong lane or
# an area where exiting cars haven't reached yet.

EXIT_MIN_CAR_SIZE  = 3 / 8
# At the exit camera, a car's bounding box must cover at least 3/8 (37.5%)
# of the ROI's total area before we'll collect crops from it.
# This filters out cars that are still far away (small bounding box) and
# ensures we only fingerprint cars that are close to and clearly exiting.
# We don't apply this at the entry camera (its default is 0.0 = no minimum).


# =============================================================================
# SECTION 2 — LOAD MODELS
# Here we load the two neural network models into memory and prepare them
# for inference (running predictions). Loading happens once at startup —
# it would be way too slow to reload the models for every single frame.
# =============================================================================

ultra_model = YOLO('yolov8n.pt')
# Load the YOLOv8 "nano" model from a local file called yolov8n.pt.
# .pt is the file extension for PyTorch model weight files.
# "nano" is the smallest and fastest YOLOv8 variant — good for real-time use.
# Other sizes exist: small (s), medium (m), large (l), extra-large (xl).
# The weights file contains millions of numbers that the model learned during
# training on the COCO dataset (a huge labeled image dataset with 80 object types).
# When we call ultra_model.track() later, YOLO uses these weights to detect cars.

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# This line picks which piece of hardware will run our ResNet model.
# Neural networks do enormous amounts of matrix multiplication. Modern GPUs
# are designed to do this kind of math in parallel — thousands of times faster
# than a regular CPU. If we're on an Apple Silicon Mac, we can use the built-in
# GPU via Apple's MPS (Metal Performance Shaders) backend for PyTorch.
#
# torch.backends.mps.is_available() returns True if we're on Apple Silicon
# with the right software installed, False otherwise.
# If MPS isn't available, we fall back to "cpu" — slower, but always works.
# On a Windows/Linux machine with an NVIDIA GPU, you'd use "cuda" instead of "mps".
#
# torch.device() creates a device object that we pass around to tell PyTorch
# where to put tensors and models. Everything has to be on the SAME device —
# you can't multiply a CPU tensor by a GPU tensor.

resnet_model = models.resnet50(weights='IMAGENET1K_V2')
# Load a ResNet-50 neural network with pre-trained weights.
#
# ResNet-50 is a "deep" convolutional neural network with 50 layers, designed
# by Microsoft Research in 2015. It was originally trained to classify images
# into 1,000 categories (dog, car, chair, etc.) from the ImageNet dataset.
#
# weights='IMAGENET1K_V2' means: load the version of ResNet-50 that was trained
# on ImageNet (1,000 classes, version 2 of the training recipe). These weights
# encode a general understanding of visual features — edges, textures, shapes,
# and object parts — that transfers well to our car-matching task even though
# ResNet was never specifically trained to recognize cars across cameras.
#
# This is called "transfer learning": reusing a model trained on one task
# (image classification) for a different but related task (visual similarity).

resnet_model.fc = torch.nn.Identity()
# ResNet's final layer (called "fc" for "fully connected") normally takes the
# 2048 internal features it computed and squishes them into 1,000 class scores
# (one per ImageNet category). We don't want class scores — we want the raw
# 2048 features, because THOSE are what make a good fingerprint for comparing cars.
#
# torch.nn.Identity() is literally a "do nothing" layer. It takes whatever
# comes in and passes it through unchanged. By replacing the fc layer with
# Identity, we're saying: "skip the final classification step and just give
# us the 2048 raw features." Now ResNet acts as a pure feature extractor.
#
# Those 2048 numbers are our car's "embedding" — a compact numeric representation
# of what the car looks like. Two photos of the same car should produce similar
# embeddings (similar lists of 2048 numbers). Two different cars should produce
# different embeddings.

resnet_model.eval()
# Put ResNet into "evaluation mode" (as opposed to "training mode").
# Neural networks behave slightly differently during training vs. inference:
#
# - Dropout layers: during training, dropout randomly zeroes out some neurons
#   to prevent overfitting. During eval, dropout is disabled so every neuron
#   always contributes. We want consistent results, so eval mode is correct.
#
# - BatchNorm layers: during training, BatchNorm computes statistics from
#   the current batch of data. During eval, it uses fixed statistics learned
#   during training. Again, eval mode gives consistent results.
#
# Always call .eval() before running inference. If you forget, results may
# be slightly random due to dropout, which would make fingerprints unreliable.

resnet_model = resnet_model.to(device)
# Move all of ResNet's weights (millions of numbers) from CPU RAM onto the
# device we selected earlier (MPS GPU or CPU). If device is "mps", this
# transfers the data to the GPU's memory. All subsequent operations on this
# model will then happen on that device. This is what enables GPU acceleration.


# =============================================================================
# SECTION 3 -- HELPER FUNCTIONS
# Smaller functions that do one specific job. The main processing functions
# below call these to keep the code clean and avoid repetition.
# =============================================================================

def extract_embedding(crop, resnet_model, device):
    """
    Given a cropped photo of a car (a NumPy array of pixel values), preprocess
    it into the format ResNet expects, run it through ResNet, and return a
    1D NumPy array of 2048 numbers (the car's embedding / fingerprint).
    """

    # transforms.Compose() chains multiple transform operations into a single
    # callable object. When you call transform(image), it applies each step
    # in order, passing the output of one step as the input to the next.
    transform = transforms.Compose([

        transforms.ToPILImage(),
        # Our crop is a NumPy array with shape [height, width, 3] where 3 is
        # the number of color channels (BGR in OpenCV's format). PyTorch's
        # image transforms expect a PIL Image object, not a NumPy array.
        # PIL (Python Imaging Library, now called Pillow) is Python's standard
        # image format. ToPILImage() handles the conversion and also
        # automatically flips the channel order from BGR (OpenCV default)
        # to RGB (what ResNet was trained on).

        transforms.Resize((224, 224)),
        # Resize the image to exactly 224×224 pixels. ResNet-50 was designed
        # and trained with 224×224 inputs. If we feed it a different size,
        # the internal math breaks (the layers expect specific dimensions).
        # Our car crops could be any size depending on how close the car is
        # to the camera, so we always resize to standardize them.

        transforms.ToTensor(),
        # Convert the PIL Image into a PyTorch Tensor.
        # A tensor is PyTorch's core data structure — essentially a multi-
        # dimensional array (like NumPy, but GPU-compatible and with gradient
        # tracking built in).
        # ToTensor() does two things:
        #   1. Rearranges the shape from [height, width, channels] (PIL format)
        #      to [channels, height, width] (PyTorch's expected format).
        #      So [224, 224, 3] becomes [3, 224, 224].
        #   2. Scales pixel values from the range 0–255 (standard image range)
        #      down to 0.0–1.0 (neural networks train better with small values).

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        # Normalize each color channel by subtracting the mean and dividing
        # by the standard deviation. This centers the pixel values around zero
        # and scales them to a consistent range.
        #
        # Why? ResNet was trained on the ImageNet dataset with these specific
        # mean and std values (computed from millions of ImageNet images).
        # If we feed it images with different statistics (e.g., from our cameras),
        # the model "expects" a different distribution and performs worse.
        # Normalizing with the training stats ensures our images "look" the
        # same way to the model as the images it was trained on.
        #
        # Formula per channel: normalized = (pixel - mean) / std
        # The three values correspond to the Red, Green, and Blue channels.
    ])

    tensor = transform(crop)
    # Apply the full pipeline: NumPy array → PIL Image → resized PIL →
    # PyTorch tensor (scaled 0–1) → normalized tensor.
    # Result shape: [3, 224, 224]  (channels, height, width)

    tensor = tensor.unsqueeze(0)
    # ResNet (and all PyTorch models) always expect input in "batches" —
    # a batch is a group of multiple images processed at once.
    # Even if we only have ONE image, we still need to wrap it in a batch
    # of size 1. unsqueeze(0) inserts a new dimension at position 0:
    #   [3, 224, 224]  →  [1, 3, 224, 224]
    #      ↑                 ↑
    #   (channels)       (batch size of 1, then channels)
    # Now we have "a batch of one image."

    tensor = tensor.to(device)
    # Move the tensor from CPU RAM onto the same device as the model.
    # If the model is on MPS (GPU) and the tensor is on CPU, PyTorch will
    # throw an error when we try to run the model. They must be on the same device.

    with torch.no_grad():
        # torch.no_grad() is a context manager that tells PyTorch:
        # "for everything inside this block, do NOT track gradients."
        #
        # Normally, PyTorch records every math operation it does so it can
        # later compute gradients (the math needed to update model weights
        # during training). This gradient tracking uses extra memory and time.
        #
        # Since we're NOT training (we're just running the model to get a
        # fingerprint), we don't need gradients at all. Disabling tracking
        # with no_grad() makes inference faster and uses less memory.
        embedding = resnet_model(tensor)
        # Pass the tensor through ResNet. PyTorch calls the model like a function.
        # Internally, the image is passed through all 50 layers:
        #   conv layers → batch norm → ReLU activations → residual connections
        #   → ... → average pooling → Identity (our replaced fc layer)
        # Output shape: [1, 2048]
        # (batch size of 1, and 2048 extracted features)

    return embedding.squeeze().cpu().numpy()
    # embedding.squeeze(): removes all dimensions of size 1.
    #   [1, 2048] → [2048]   (we just want the flat list of 2048 numbers)
    #
    # .cpu(): move the tensor from GPU back to CPU RAM. We need it in CPU
    #   memory to convert to NumPy — NumPy doesn't know about GPUs.
    #
    # .numpy(): convert from a PyTorch tensor to a NumPy array.
    #   They store the same data but NumPy is what the rest of our code uses
    #   for math operations like np.dot() and np.mean().


def get_average_embedding(crops, resnet_model, device):
    """
    Given a list of crop images (multiple photos of the same car),
    compute an embedding for each one, average all the embeddings together,
    then normalize the result into a unit vector.

    Why average? Because any single photo might be slightly blurry, oddly lit,
    or captured at an awkward angle. Averaging across 5 photos smooths out
    these imperfections and gives a more reliable fingerprint.
    """

    embeddings = [extract_embedding(crop, resnet_model, device) for crop in crops]
    # List comprehension: for each crop in our list, call extract_embedding()
    # to get its 2048-number fingerprint. The result is a Python list of
    # NumPy arrays, each with shape [2048].

    stacked = np.stack(embeddings)
    # np.stack() takes a list of arrays with the same shape and stacks them
    # into a new 2D array. If we have 5 embeddings each of shape [2048]:
    #   list of 5 arrays, each [2048]  →  one 2D array of shape [5, 2048]
    # Think of it like stacking 5 rows into a table with 5 rows and 2048 columns.

    averaged = np.mean(stacked, axis=0)
    # np.mean() computes the average. axis=0 means: average along the first
    # dimension (across the 5 rows). For each of the 2048 columns, we get
    # the average value across all 5 crops.
    #   [5, 2048]  →  [2048]   (one averaged embedding)
    # This is equivalent to: add up all 5 embeddings element-wise, then divide by 5.

    norm = np.linalg.norm(averaged)
    # np.linalg.norm() computes the "L2 norm" (also called the Euclidean length)
    # of a vector. This is the same as: sqrt(a[0]² + a[1]² + ... + a[2047]²)
    # It tells us the "length" of our 2048-dimensional vector in geometric space.

    normalized = averaged / norm
    # Divide every element of the averaged embedding by its length.
    # This produces a "unit vector" — a vector with length exactly 1.0.
    #
    # WHY IS THIS IMPORTANT?
    # Later, we compare cars using the dot product (np.dot) of their embeddings.
    # For unit vectors specifically, the dot product equals the cosine similarity:
    #   cosine_similarity = dot(A, B) = cos(angle between A and B)
    # This gives a value from -1 to 1 that measures how "similar in direction"
    # two vectors are, independent of their magnitude.
    #
    # If we didn't normalize, a brighter/closer car would have a larger-magnitude
    # embedding and would dominate comparisons unfairly. Normalization ensures
    # we're comparing the PATTERN of features, not the scale.

    return normalized


def get_vehicle_crop(frame, box):
    """
    Given a full video frame and a bounding box [x1, y1, x2, y2] from YOLO,
    slice out just the car's pixels from the frame. Adds a small padding
    around the box so we don't accidentally clip the edges of the car.
    Returns the cropped NumPy array, or None if the crop is empty.
    """

    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    # YOLO returns bounding box coordinates as floating point numbers.
    # We convert them to integers because we're going to use them as
    # array indices (you can't slice an array with float indices).
    # box[0]=x1 (left edge), box[1]=y1 (top edge),
    # box[2]=x2 (right edge), box[3]=y2 (bottom edge)

    pad = 10
    # We'll expand the bounding box by 10 pixels in every direction.
    # YOLO's boxes are tight around the car, but there might be pixels
    # just outside the box that are still part of the car (mirrors, hood edges).
    # The padding gives us a bit of breathing room.

    x1 = max(0, x1 - pad)
    # Subtract pad from the left edge, but don't go below 0 (can't index
    # negative pixels). max(0, ...) clamps the value at 0.

    y1 = max(0, y1 - pad)
    # Same logic for the top edge.

    x2 = min(frame.shape[1], x2 + pad)
    # frame.shape returns (height, width, channels).
    # frame.shape[1] is the frame WIDTH. We can't go beyond the right edge.
    # min(...) clamps the value so it doesn't exceed the frame width.

    y2 = min(frame.shape[0], y2 + pad)
    # frame.shape[0] is the frame HEIGHT. Clamp the bottom edge similarly.

    crop = frame[y1:y2, x1:x2]
    # NumPy array slicing: extract the rectangular region from the frame.
    # frame is a 3D array with shape [height, width, 3].
    # frame[y1:y2, x1:x2] gives us all rows from y1 to y2 and all columns
    # from x1 to x2. This is our cropped car image.

    if crop.size == 0:
        # crop.size is the total number of elements in the array (h × w × 3).
        # If it's 0, the crop is empty — this can happen if the bounding box
        # was somehow invalid (e.g., x1 >= x2 or y1 >= y2). Skip it.
        return None

    return crop
    # Returns a NumPy array of shape [crop_height, crop_width, 3] — the car pixels.


def match_car(exit_embedding, exit_time, entry_buffer):
    """
    Given the fingerprint and timestamp of an exit car, search through all
    logged entry cars to find the best match.

    The matching process:
    1. Filter out entry cars that are impossible matches (wrong time window)
    2. For remaining candidates, compute cosine similarity between fingerprints
    3. Return the entry car with the highest similarity if it clears the threshold

    Parameters:
        exit_embedding : NumPy array [2048] — the exit car's fingerprint
        exit_time      : float — when (in seconds) the exit car was logged
        entry_buffer   : dict  — { track_id: {"embedding": ..., "timestamp": ...} }
                                  for all unmatched entry cars

    Returns:
        The track_id (key) of the best matching entry car, or None if no match found.
    """

    best_match_key  = None   # Will hold the track_id of the best match we find
    best_similarity = -1.0   # Worst possible similarity; will be updated as we find better matches
    # We start at -1.0 because cosine similarity ranges from -1 to 1.
    # Any real candidate will score higher than -1.0.

    for key, data in entry_buffer.items():
        # Iterate over every unmatched entry car.
        # key   = the track_id (an integer like 3, 7, 12...)
        # data  = a dict with "embedding" (the car's fingerprint) and "timestamp" (when it entered)

        entry_time = data["timestamp"]
        transit = exit_time - entry_time
        # Compute how many seconds elapsed between entry and exit.
        # If this is negative, the car "exited before it entered" — impossible.
        # If this exceeds MATCH_TIMEOUT, the car has been in there too long.

        if transit < 0 or transit > MATCH_TIMEOUT:
            continue
            # Skip this entry car — the timing makes it an impossible match.
            # continue jumps to the next iteration of the for loop.

        similarity = np.dot(exit_embedding, data["embedding"])
        # np.dot() computes the dot product of two vectors:
        #   dot(A, B) = A[0]*B[0] + A[1]*B[1] + ... + A[2047]*B[2047]
        # Because both embeddings are unit vectors (we normalized them in
        # get_average_embedding), this dot product equals the cosine similarity.
        # cosine similarity = 1.0   → vectors point in the same direction → very similar cars
        # cosine similarity = 0.0   → vectors are perpendicular → unrelated cars
        # cosine similarity = -1.0  → opposite directions → very dissimilar

        if similarity > best_similarity:
            # If this candidate is better than our current best, update it.
            best_similarity = similarity
            best_match_key  = key

    if best_similarity >= SIM_THRESHOLD:
        # Only return a match if the best candidate actually clears our threshold.
        # If no candidate exceeded SIM_THRESHOLD, best_match_key is still None.
        return best_match_key

    return None
    # If we get here, either no candidates were in the valid time window,
    # or the best similarity was below the threshold. No match found.


# =============================================================================
# SECTION 4 -- PER-CAMERA PROCESSING FUNCTION
# This is the core workhorse function. It opens a video, processes it frame
# by frame, runs YOLO to detect and track cars, collects crop photos,
# and computes a fingerprint for each car. Returns a list of results.
# =============================================================================

def process_camera(video_path, yolo_model, resnet_model, device, role, roi, min_car_size=0.0):
    """
    Process a single camera's video file and return a list of car events.
    Each event is a dict: { "track_id": int, "timestamp": float, "embedding": np.array }

    Parameters:
        video_path   : str   — path to the video file
        yolo_model   : YOLO  — the YOLO object detection + tracking model
        resnet_model : nn    — the ResNet feature extractor
        device       : torch.device — "mps" or "cpu"
        role         : str   — "entry" or "exit" (used only for print labels)
        roi          : tuple — (left, top, right, bottom) as fractions 0.0–1.0
        min_car_size : float — minimum car bounding box size as fraction of ROI area
    """

    cap = cv2.VideoCapture(video_path)
    # cv2.VideoCapture opens a video file and prepares it for reading frame by frame.
    # It doesn't load the whole video into memory — it reads frames on demand.
    # Internally it creates a "capture object" we'll use to pull frames in the loop.

    if not cap.isOpened():
        # Check if the file was actually opened successfully.
        # If the file doesn't exist, the path is wrong, or the format is unsupported,
        # cap.isOpened() returns False. We print an error and return an empty list
        # rather than crashing with a confusing error later.
        print(f"Error: could not open video {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    # Read the actual FPS from the video file's metadata.
    # cv2.CAP_PROP_FPS is a constant that tells cap.get() "I want the FPS property."
    # This is more accurate than our global FPS constant because different videos
    # can have different frame rates.

    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # Similarly read the frame dimensions from the video metadata.
    # We need these to convert our ROI fractions into actual pixel coordinates.
    # cap.get() returns floats, so we cast to int with int().

    # Convert the ROI fractions into pixel coordinates for this specific video resolution
    roi_x1 = int(roi[0] * w)     # left edge:   fraction × width
    roi_y1 = int(roi[1] * h)     # top edge:    fraction × height
    roi_x2 = int(roi[2] * w)     # right edge:  fraction × width
    roi_y2 = int(roi[3] * h)     # bottom edge: fraction × height
    # Example: if w=1920, h=1080 and roi=(0.25, 0.0, 1.0, 1.0):
    #   roi_x1 = 480, roi_y1 = 0, roi_x2 = 1920, roi_y2 = 1080

    roi_area = (roi_x2 - roi_x1) * (roi_y2 - roi_y1)
    # Total pixel area of the ROI box. We'll use this to compute what fraction
    # of the ROI a car's bounding box covers (for the min_car_size filter).

    delay_frames = int(CROP_DELAY_SEC * fps)
    # Convert the delay from seconds to a number of frames.
    # Example: CROP_DELAY_SEC=1.0 and fps=30.0 → delay_frames=30
    # We wait 30 frames (1 second) after first seeing a car before collecting crops.

    # --------------------------------------------------------------------------
    # State tracking dictionaries and sets
    # These track what's happening with each car across frames.
    # --------------------------------------------------------------------------

    crop_buffer  = {}
    # Dictionary: { track_id: [crop1, crop2, crop3, ...] }
    # Stores the collected crop photos for each car.
    # A "crop" here is a NumPy array of the car's pixels from one frame.

    first_seen   = {}
    # Dictionary: { track_id: frame_number }
    # Records the first frame number in which we ever saw each car.
    # Used to enforce the CROP_DELAY_SEC waiting period.

    logged_ids   = set()
    # A Python set of track IDs that have already been fully processed.
    # Once we've collected CROP_NUM crops and computed the fingerprint for a car,
    # we add its ID here so we don't process it again in future frames.
    # Sets have O(1) lookup — checking "is X in logged_ids?" is instant.

    results      = []
    # Our output list. Each entry is a dict with track_id, timestamp, and embedding.
    # This is what the function returns at the end.

    frame_number = 0
    # Counter that increments by 1 for each frame we read.
    # We use this (divided by fps) to compute timestamps in seconds.

    # --------------------------------------------------------------------------
    # Main frame-reading loop
    # --------------------------------------------------------------------------

    while True:
        # Loop forever — we'll break out when we reach the end of the video.

        ret, frame = cap.read()
        # cap.read() reads the next frame from the video.
        # Returns two values:
        #   ret   : bool  — True if a frame was successfully read, False at end of video
        #   frame : np.array — the frame's pixel data, shape [height, width, 3]
        #                      (3 channels: Blue, Green, Red — OpenCV uses BGR not RGB)

        if not ret:
            break
            # We've hit the end of the video (or an error). Exit the loop.

        frame_number += 1
        # Increment our frame counter. We do this before processing so frame_number
        # starts at 1 (not 0), which is more natural for timestamp calculation.

        roi_frame = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        # Slice the full frame down to just the Region of Interest.
        # This is a NumPy array slice — it's zero-copy (no data is duplicated),
        # it just creates a "view" into the original frame's memory.
        # We pass roi_frame to YOLO instead of the full frame so YOLO only
        # analyzes the relevant part of the video.

        results_yolo = yolo_model.track(
            roi_frame,       # The image (or video frame) to run detection on
            persist=True,    # Keep track IDs consistent across frames
                             # (car #3 in frame 50 is the same car as #3 in frame 51)
            classes=[2, 5, 7], # Only detect these COCO dataset classes:
                             # 2=car, 5=bus, 7=truck. Ignores people, bikes, etc.
            verbose=False    # Don't print YOLO's detection results to the console
                             # for every frame (would be extremely noisy)
        )
        # YOLO's .track() method does two things in one call:
        #   1. DETECTION: finds all vehicles in the frame and draws bounding boxes
        #   2. TRACKING: assigns a persistent integer ID to each vehicle and tries
        #      to maintain that ID across frames even as the car moves
        # The tracking algorithm uses appearance + position to re-identify cars.
        # Result: a list of "Results" objects (one per input image; we gave it one).

        result = results_yolo[0]
        # We passed one frame, so results_yolo is a list with one element.
        # Grab index [0] to get the Results object for our frame.

        if result.boxes is None or result.boxes.id is None:
            continue
            # result.boxes contains all detected bounding boxes.
            # result.boxes.id contains the track IDs (None if tracking failed this frame).
            # If either is None, YOLO found no vehicles or couldn't assign IDs.
            # Skip this frame and move to the next one.

        boxes     = result.boxes.xyxy.cpu().numpy()
        # result.boxes.xyxy: bounding boxes in [x1, y1, x2, y2] format (top-left and bottom-right corners)
        # as a PyTorch tensor on the GPU.
        # .cpu(): move the tensor from GPU to CPU memory.
        # .numpy(): convert from PyTorch tensor to NumPy array for easier math.
        # Final shape: [num_detections, 4] — one row per detected car, 4 coordinates per row.

        track_ids = result.boxes.id.cpu().numpy().astype(int)
        # result.boxes.id: the persistent track IDs assigned to each detection.
        # .cpu().numpy(): same GPU→CPU→NumPy conversion as above.
        # .astype(int): IDs are returned as floats, but they're logically integers.
        #   Convert so we can use them as dictionary keys cleanly.
        # Final shape: [num_detections] — one ID per detected car.

        for i, track_id in enumerate(track_ids):
            # Loop over each detected car in this frame.
            # enumerate() gives us both the index (i) and the value (track_id)
            # so we can use i to grab the corresponding bounding box from boxes[i].

            box = boxes[i]
            # The [x1, y1, x2, y2] bounding box for this specific car.

            if track_id not in first_seen:
                first_seen[track_id] = frame_number
            # If this is the very first time we've seen this track_id,
            # record the current frame number. This marks when the car first appeared.
            # "not in" checks for membership in the dictionary's keys — O(1) operation.

            if frame_number - first_seen[track_id] < delay_frames:
                continue
            # If fewer frames have passed than our delay threshold, skip this car.
            # The car just appeared — we're waiting for it to get into a better position
            # before we start photographing it. Continue to the next car in this frame.

            box_area = (box[2] - box[0]) * (box[3] - box[1])
            # Compute the pixel area of the bounding box.
            # width  = box[2] - box[0]  =  x2 - x1
            # height = box[3] - box[1]  =  y2 - y1
            # area   = width × height

            if box_area / roi_area < min_car_size:
                continue
            # Compute what fraction of the ROI area this car's box covers.
            # If it's less than min_car_size (3/8 for exit, 0 for entry), skip.
            # A tiny bounding box means the car is far from the camera and probably
            # not a good candidate for fingerprinting (blurry, too small for ResNet).

            crop = get_vehicle_crop(roi_frame, box)
            # Cut out the car's pixels from the ROI frame using the bounding box.
            # This returns a NumPy array of just the car, or None if something went wrong.

            if crop is None:
                continue
            # If get_vehicle_crop returned None (empty crop), skip this car this frame.

            if track_id not in crop_buffer:
                crop_buffer[track_id] = []
            # If this is the first valid crop for this car, create an empty list
            # in crop_buffer to hold its crops.

            if len(crop_buffer[track_id]) < CROP_NUM:
                crop_buffer[track_id].append(crop)
            # If we haven't yet collected enough crops (< CROP_NUM = 5),
            # add this frame's crop to the car's collection.
            # len() returns the current count, and we only add if it's under the limit.

            if track_id not in logged_ids and len(crop_buffer.get(track_id, [])) >= CROP_NUM:
                # Two conditions must BOTH be true to proceed:
                # 1. We haven't already logged this car (not in logged_ids)
                # 2. We have at least CROP_NUM crops collected for it
                #
                # crop_buffer.get(track_id, []) safely retrieves the crop list,
                # returning an empty list [] as default if the key doesn't exist
                # (avoids a KeyError crash).

                embedding = get_average_embedding(
                    crop_buffer[track_id], resnet_model, device
                )
                # Compute the averaged, normalized fingerprint for this car
                # using all CROP_NUM collected photos. Returns a [2048] NumPy array.

                timestamp = frame_number / fps
                # Convert the current frame number to a timestamp in seconds.
                # Example: frame 300, fps 30 → 10.0 seconds into the video.

                results.append({
                    "track_id":  track_id,    # The integer ID YOLO assigned to this car
                    "timestamp": timestamp,    # When (in seconds) this car was fully logged
                    "embedding": embedding     # The 2048-number fingerprint for this car
                })
                # Add this car's event record to our output list.

                logged_ids.add(track_id)
                # Mark this car as done. In all future frames, if YOLO detects this
                # same car again (same track_id), we'll skip it — we already have
                # everything we need from it.

                print(f"[{role.upper()}] Car {track_id} logged at {timestamp:.2f}s")
                # Print a progress message. role.upper() prints "ENTRY" or "EXIT".
                # :.2f formats the float with 2 decimal places (e.g., "12.34s").

    cap.release()
    # Close the video file and release the memory/file handles associated with it.
    # Always release captures when you're done — not doing so can cause file
    # locking issues or memory leaks.

    return results
    # Return the list of car event dicts to the caller (run_matching or wherever).


# =============================================================================
# SECTION 5 -- MAIN MATCHING LOGIC
# This function ties everything together: process both videos, then try to
# pair each exit car with the entry car that looks most like it.
# =============================================================================

def run_matching():
    """
    Full end-to-end pipeline:
    1. Process entry video → fingerprints for each entry car
    2. Process exit video  → fingerprints for each exit car
    3. For each exit car, find the best-matching entry car
    4. Compute transit time = exit time - entry time
    5. Write matched pairs to a CSV file
    """

    print("Processing entry camera...")
    entry_events = process_camera(
        ENTRY_VIDEO_PATH,   # Video file to process
        ultra_model,        # YOLO model for detecting/tracking cars
        resnet_model,       # ResNet model for extracting fingerprints
        device,             # Which hardware to run ResNet on (mps/cpu)
        "entry",            # Label for print messages
        roi=ENTRY_ROI       # The region of interest for the entry camera
        # No min_car_size argument → defaults to 0.0 (no size filter)
    )
    # entry_events is now a list of dicts, one per car that entered:
    # [ {"track_id": 3, "timestamp": 4.2, "embedding": array([...])}, ... ]

    print("Processing exit camera...")
    exit_events = process_camera(
        EXIT_VIDEO_PATH,
        ultra_model,
        resnet_model,
        device,
        "exit",
        roi=EXIT_ROI,
        min_car_size=EXIT_MIN_CAR_SIZE   # Only fingerprint cars covering 3/8 of the ROI
    )

    entry_buffer = {}
    # Build a lookup dictionary from the entry_events list.
    # Structure: { track_id: {"embedding": ..., "timestamp": ...} }
    # Using a dictionary (instead of keeping it as a list) lets us quickly
    # look up and delete entries by track_id during matching.

    for event in entry_events:
        key = event["track_id"]
        entry_buffer[key] = {
            "embedding": event["embedding"],
            "timestamp": event["timestamp"]
        }
    # After this loop, entry_buffer contains one entry per entry car,
    # keyed by its YOLO track_id.

    matched_results  = []
    # List of successfully matched pairs. Each element is a dict with
    # entry_track_id, exit_track_id, entry_time, exit_time, transit_seconds.

    unmatched_exits  = []
    # List of exit track IDs that we couldn't find an entry match for.
    # Useful for debugging — a high number of unmatched exits suggests
    # our threshold is too strict or the cameras are misaligned.

    for exit_event in exit_events:
        # Process each exit car one at a time.

        exit_embedding = exit_event["embedding"]  # This exit car's fingerprint
        exit_time      = exit_event["timestamp"]  # When it appeared at the exit camera
        exit_id        = exit_event["track_id"]   # Its YOLO ID at the exit camera
        # Note: exit IDs and entry IDs are completely separate numbering systems!
        # YOLO assigns IDs independently per camera. Car "3" at entry and car "3"
        # at exit are NOT necessarily the same vehicle. That's why we use the
        # embedding similarity to match them — not the ID.

        match_key = match_car(exit_embedding, exit_time, entry_buffer)
        # Search entry_buffer for the best matching entry car.
        # Returns the entry track_id if a match was found, or None if not.

        if match_key is not None:
            # A match was found!

            entry_time   = entry_buffer[match_key]["timestamp"]
            transit_time = exit_time - entry_time
            # Compute how many seconds elapsed from entry to exit.

            matched_results.append({
                "entry_track_id":  match_key,               # Which car entered
                "exit_track_id":   exit_id,                 # Which car (at exit camera) exited
                "entry_time":      round(entry_time, 2),    # Entry timestamp (seconds)
                "exit_time":       round(exit_time, 2),     # Exit timestamp (seconds)
                "transit_seconds": round(transit_time, 2)   # How long the trip took
                # round(..., 2) limits to 2 decimal places for cleaner output
            })

            print(f"MATCH: Entry car {match_key} -> Exit car {exit_id} | Transit: {transit_time:.1f}s")

            del entry_buffer[match_key]
            # CRITICAL: remove this entry car from the buffer after matching it.
            # This ensures one entry car can only be matched to ONE exit car.
            # Without this, multiple exit cars could all claim the same entry car.

        else:
            # No suitable entry car was found for this exit car.
            unmatched_exits.append(exit_id)
            print(f"NO MATCH: Exit car {exit_id} at {exit_time:.2f}s")

    unmatched_entries = list(entry_buffer.keys())
    # Any entry cars still remaining in entry_buffer were never matched.
    # This could mean:
    #   - The car didn't make it to the exit camera within MATCH_TIMEOUT
    #   - The exit camera simply didn't detect it
    #   - The similarity was below threshold for all exit cars
    # list() converts the dict_keys view into a regular Python list.

    for key in unmatched_entries:
        t = entry_buffer[key]["timestamp"]
        print(f"UNMATCHED ENTRY: Car {key} entered at {t:.2f}s -- no exit found")

    # Print a summary of results
    print(f"\n--- RESULTS ---")
    print(f"Matched pairs:     {len(matched_results)}")
    print(f"Unmatched exits:   {len(unmatched_exits)}")
    print(f"Unmatched entries: {len(unmatched_entries)}")

    with open(OUTPUT_CSV, "w", newline="") as f:
        # open() opens a file for writing. "w" = write mode (creates the file
        # if it doesn't exist, overwrites it if it does).
        # newline="" is important for csv.writer on Windows — without it,
        # the writer would add extra blank lines between rows.
        # "as f" gives us a file object named f to write to.
        # The "with" block automatically closes the file when we're done,
        # even if an error occurs mid-way through.

        fieldnames = ["entry_track_id", "exit_track_id", "entry_time", "exit_time", "transit_seconds"]
        # The column headers for our CSV file. Each string becomes a column name.

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        # DictWriter is a CSV writer that accepts rows as dictionaries.
        # The keys in each dict must match the fieldnames list.
        # This is more readable than positional writing because column order
        # doesn't matter as long as the keys match.

        writer.writeheader()
        # Write the first row of the CSV: the column names (e.g., "entry_track_id,exit_track_id,...").
        # Without this, the CSV has no headers and is harder to read/import.

        writer.writerows(matched_results)
        # Write all matched pairs as rows. writerows() takes a list of dicts
        # and writes each one as a row in the CSV file.

    print(f"Results saved to {OUTPUT_CSV}")


# =============================================================================
# SECTION 6 -- VISUALIZATION
# Creates annotated debug videos showing exactly how the pipeline processes
# each camera. Each car gets a colored bounding box that shows its current
# processing state. Invaluable for tuning ROI, min_car_size, and delay settings.
# =============================================================================

def save_sample(video_path, output_path, roi, min_car_size=0.0, duration_seconds=15):
    """
    Read up to duration_seconds of a video, run YOLO tracking, draw annotated
    bounding boxes on each car showing its processing state, and save the
    result as a new video file.

    Box colors:
      RED    = car is in the delay period (too early) OR bounding box too small
      YELLOW = car is being actively photographed (collecting crops)
      GREEN  = car is fully logged (fingerprint computed)

    This function doesn't actually compute embeddings — it simulates the
    crop-counting logic to keep the color states in sync with what process_camera
    would do in the real pipeline.
    """

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_frames = int(duration_seconds * fps)
    # How many frames to process before stopping.
    # duration_seconds × fps = total number of frames in that time span.

    # Convert ROI fractions to pixel coordinates (same as in process_camera)
    roi_x1  = int(roi[0] * w)
    roi_y1  = int(roi[1] * h)
    roi_x2  = int(roi[2] * w)
    roi_y2  = int(roi[3] * h)
    roi_area = (roi_x2 - roi_x1) * (roi_y2 - roi_y1)

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    # cv2.VideoWriter creates a new video file we can write frames into.
    #
    # cv2.VideoWriter_fourcc(*"mp4v"):
    #   fourcc stands for "four character code" — it identifies the video codec.
    #   "mp4v" is the codec for MPEG-4 video, compatible with .mp4 files.
    #   The *"mp4v" unpacks the string into four separate characters: 'm','p','4','v'.
    #
    # fps: the output video's frame rate (matches the input video).
    # (w, h): the output video's frame dimensions (matches the input).
    # All three (fps, size, codec) must match what you're writing to avoid corruption.

    delay_frames = int(CROP_DELAY_SEC * fps)
    first_seen   = {}    # track_id → first frame number
    crop_counts  = {}    # track_id → number of crops "collected" so far (simulated)
    logged_ids   = set() # Set of fully logged car IDs

    frame_number = 0

    while frame_number < max_frames:
        # Read frames until we hit our time limit.
        # Using "while frame_number < max_frames" instead of "while True"
        # so we automatically stop after duration_seconds seconds.

        ret, frame = cap.read()
        if not ret:
            break
        frame_number += 1

        roi_frame = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        # Crop the frame to the ROI before passing to YOLO (same as process_camera)

        results_yolo = ultra_model.track(
            roi_frame, persist=True, classes=[2, 5, 7], verbose=False
        )
        result = results_yolo[0]

        if result.boxes is not None and result.boxes.id is not None:
            boxes     = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.cpu().numpy().astype(int)

            for i, track_id in enumerate(track_ids):

                # Translate bounding box coordinates from ROI-space back to full-frame space.
                # YOLO was given the roi_frame (cropped), so its coordinates are relative
                # to the top-left corner of the ROI (0,0 = top-left of ROI).
                # But we want to draw on the full frame, so we add the ROI's offset back.
                x1 = int(boxes[i][0]) + roi_x1   # roi_x1 is how far right the ROI starts
                y1 = int(boxes[i][1]) + roi_y1   # roi_y1 is how far down the ROI starts
                x2 = int(boxes[i][2]) + roi_x1
                y2 = int(boxes[i][3]) + roi_y1

                if track_id not in first_seen:
                    first_seen[track_id] = frame_number
                frames_since_first = frame_number - first_seen[track_id]
                # How many frames have passed since we first spotted this car.
                # We compare this to delay_frames to decide if we're still waiting.

                box_area  = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1])
                # Note: use boxes[i] (ROI-space coords) for area calculation,
                # NOT x1/y1/x2/y2 (full-frame coords), because the ROI-space
                # values give us the actual pixel dimensions of the bounding box.
                size_frac = box_area / roi_area
                too_small = size_frac < min_car_size
                # Boolean: True if this car's bounding box is below the minimum size.

                count = crop_counts.get(track_id, 0)
                # crop_counts.get(track_id, 0): retrieve the crop count for this car,
                # returning 0 as default if we haven't seen this car before.

                # Simulate the crop-collection logic from process_camera()
                # so our color states match what the real pipeline would do.
                # We don't actually compute embeddings here — just count.
                if (track_id not in logged_ids          # Not already done
                        and frames_since_first >= delay_frames  # Past the delay
                        and not too_small                # Big enough to use
                        and count < CROP_NUM):           # Still need more crops
                    crop_counts[track_id] = count + 1   # Increment the simulated crop count
                    if crop_counts[track_id] >= CROP_NUM:
                        logged_ids.add(track_id)         # Mark as fully logged

                # Determine the box color and label text based on state:
                if track_id in logged_ids:
                    color = (0, 255, 0)             # OpenCV colors are BGR, not RGB!
                    label = f"ID:{track_id} done"   # (0, 255, 0) = green in BGR
                elif frames_since_first < delay_frames:
                    color = (0, 0, 255)             # (0, 0, 255) = red in BGR
                    label = f"ID:{track_id} wait"   # Waiting for delay to pass
                elif too_small:
                    color = (0, 0, 255)             # Red: car is too small/far
                    label = f"ID:{track_id} small {size_frac:.2f}"
                    # :.2f formats the float as 2 decimal places, e.g., "0.23"
                else:
                    color = (0, 220, 255)           # (0, 220, 255) = yellow in BGR
                    label = f"ID:{track_id} crop {count+1}/{CROP_NUM}"
                    # Shows progress: "crop 3/5" means we've collected 3 of 5 needed crops

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                # Draw a filled-outline rectangle on the frame.
                # (x1, y1) = top-left corner of the box
                # (x2, y2) = bottom-right corner
                # color     = BGR tuple from above
                # 2         = line thickness in pixels

                cv2.putText(frame, label,
                            (x1, y1 - 8),           # Position: just above the box's top edge
                            cv2.FONT_HERSHEY_SIMPLEX, # A clean, readable sans-serif font
                            0.55,                   # Font scale (size multiplier)
                            color,                  # Text color (same as box color)
                            2)                      # Text thickness in pixels

        # Draw the ROI boundary as a blue rectangle on the full frame.
        # This makes it easy to see exactly which area YOLO is analyzing.
        # (255, 100, 0) in BGR = a dark blue-ish color (255 blue, 100 green, 0 red)
        cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 100, 0), 2)

        timestamp = frame_number / fps
        cv2.putText(frame, f"{timestamp:.1f}s",  # "12.5s" style timestamp
                    (10, 30),                    # Top-left area of the frame
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,                         # Slightly larger font than the car labels
                    (0, 200, 255),               # Bright cyan color (visible on most backgrounds)
                    2)

        out.write(frame)
        # Write this annotated frame to the output video file.
        # This adds one frame to the output video. After max_frames calls,
        # we'll have a complete video of the annotated footage.

    cap.release()   # Close the input video file
    out.release()   # Finalize and close the output video file.
                    # This is essential — without it, the output file may be
                    # corrupted or unplayable (the video codec needs to write
                    # a final end-of-stream marker).

    print(f"Sample saved to {output_path}")


# =============================================================================
# SECTION 7 -- ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # This is a standard Python pattern.
    #
    # When Python runs a file, it sets a special variable called __name__.
    # If you run the file directly (python main.py), __name__ is set to "__main__".
    # If another file imports this file as a module, __name__ is set to the
    # module's name (e.g., "main"), NOT "__main__".
    #
    # This check means: "only run the following code if this file was executed
    # directly, not imported." It prevents the pipeline from running automatically
    # when some other script imports our functions.

    print("Saving annotated samples...")

    save_sample(ENTRY_VIDEO_PATH, "sample_entry.mp4", roi=ENTRY_ROI, duration_seconds=90)
    # Process the first 90 seconds of the entry video and save an annotated version.
    # Open sample_entry.mp4 afterward to check:
    #   - Are the ROI boundaries in the right place?
    #   - Are cars being detected? Do they get green boxes?
    #   - Are any cars stuck on red too long (wrong delay setting)?

    save_sample(EXIT_VIDEO_PATH,  "sample_exit.mp4",  roi=EXIT_ROI, min_car_size=EXIT_MIN_CAR_SIZE, duration_seconds=90)
    # Same for the exit video. Also check:
    #   - Are cars getting "small" labels too often? (lower EXIT_MIN_CAR_SIZE if so)
    #   - Are distant non-exiting cars being filtered out properly? (raise it if not)

    print("Done. Open sample_entry.mp4 and sample_exit.mp4 to review tracking.")

    run_matching()
    # Run the full matching pipeline:
    # process entry → process exit → match pairs → save CSV