# Open yaml
import yaml

with open("nyu.yaml", "r") as file:
    config = yaml.safe_load(file)
print(config)

INTRINSICS = [config["fx"], config["fy"], config["cx"], config["cy"]]
R = config["depth_max"]  # Maximum range of sensor
EPSILON = config["resolution"]  # Resolution of the sensor

# Image dir
import torch
import numpy as np
from PIL import Image
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "/scratchdata/nyu_plane"
INDEX = 10

rgb = Image.open(os.path.join(DATA_DIR, "rgb", f"{INDEX}.png")).convert("RGB")
rgb = np.array(rgb)

depth = Image.open(os.path.join(DATA_DIR, "depth", f"{INDEX}.png")).convert("I")
depth = np.array(depth) * EPSILON
print(depth.max(), depth.min())

# Tuning parameters


TARGET_FOLDER = "sam"

SIGMA_RATIO = 0.01

CONFIDENCE = 0.99
INLIER_THRESHOLD = 0.1
MAX_PLANE = 16

USE_SAM = True

SAM_CONFIDENCE = 0.99
SAM_INLIER_THRESHOLD = 0.2
SAM_MAX_PLANE = 4

POST_PROCESSING = False

# Use SAM to partition the image
if USE_SAM: 
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    sam = sam_model_registry["default"](checkpoint="/scratchdata/sam_vit_h_4b8939.pth").to(DEVICE)
    mask_generator = SamAutomaticMaskGenerator(sam, stability_score_thresh=0.98)

    sam_masks = mask_generator.generate(rgb)
    print(len(sam_masks))
    masks = sorted(sam_masks, key=lambda x: x["stability_score"])