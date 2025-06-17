import torch
import numpy as np
from PIL import Image
import os

from information_estimation import information_estimation
from utils.process_depth import get_3d
# Open yaml
import yaml

with open("nyu.yaml", "r") as file:
    config = yaml.safe_load(file)

INTRINSICS = [config["fx"], config["fy"], config["cx"], config["cy"]]
R = config["depth_max"]  # Maximum range of sensor
EPSILON = config["resolution"]  # Resolution of the sensor

# Image dir

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "/scratchdata/nyu_plane"
INDEX = 10

rgb = Image.open(os.path.join(DATA_DIR, "rgb", f"{INDEX}.png")).convert("RGB")
rgb = np.array(rgb)

depth = Image.open(os.path.join(DATA_DIR, "depth", f"{INDEX}.png")).convert("I")
depth = np.array(depth) * EPSILON

pts_3d = get_3d(depth, INTRINSICS)
# Tuning parameters

TARGET_FOLDER = "sam"

SIGMA_RATIO = 0.01
SIGMA = SIGMA_RATIO * depth

CONFIDENCE = 0.99
INLIER_RATIO= 0.1
MAX_PLANE = 9

USE_SAM = False

SAM_CONFIDENCE = 0.99
SAM_INLIER_RATIO = 0.2
SAM_MAX_PLANE = 4

POST_PROCESSING = False

global_mask = np.zeros_like(depth, dtype=np.int32).flatten()
global_planes = []

# Use SAM to partition the image
if USE_SAM: 
    print("Using SAM to partition the image...")

    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    sam = sam_model_registry["default"](checkpoint="/scratchdata/sam_vit_h_4b8939.pth").to(DEVICE)
    mask_generator = SamAutomaticMaskGenerator(sam, stability_score_thresh=0.98)

    sam_masks = mask_generator.generate(rgb)
    print("SAM of Regions:", len(sam_masks))
    masks = sorted(sam_masks, key=lambda x: x["stability_score"])


valid_mask = (global_mask == 0) & (depth > 0).flatten()
mask, plane = information_estimation(pts_3d, R, EPSILON, SIGMA.flatten(), CONFIDENCE, INLIER_RATIO, MAX_PLANE, valid_mask=valid_mask, verbose=True)

print(mask.max())
global_mask = np.where(mask > 0, mask+global_mask.max(), global_mask)
print(global_mask.max())
global_planes.append(plane)
print(global_planes)