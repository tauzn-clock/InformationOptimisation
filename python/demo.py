import torch
import numpy as np
from PIL import Image
import os
import csv
import yaml

from information_optimisation import information_optimisation
from utils.process_depth import get_3d


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Open yaml
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "nyu.yaml"), "r") as file:
    config = yaml.safe_load(file)

INTRINSICS = [config["camera_params"]["fx"], config["camera_params"]["fy"], config["camera_params"]["cx"], config["camera_params"]["cy"]]
R = config["depth_max"]  # Maximum range of sensor
EPSILON = config["resolution"]  # Resolution of the sensor

# Image dir
DATA_DIR = "/scratchdata/nyu_plane"
INDEX = 0

rgb = Image.open(os.path.join(DATA_DIR, "rgb", f"{INDEX}.png")).convert("RGB")
rgb = np.array(rgb)

depth = Image.open(os.path.join(DATA_DIR, "depth", f"{INDEX}.png")).convert("I;16")
depth = np.array(depth) * EPSILON

pts_3d = get_3d(depth, INTRINSICS)

# Tuning parameters
TARGET_FOLDER = "mask"

SIGMA_RATIO = 0.01
SIGMA = SIGMA_RATIO * depth # Proportional noise model
#SIGMA = 0.0012 + 0.0019 * (depth - 0.4)**2 # Empirical noise model
SIGMA = SIGMA.flatten()

CONFIDENCE = 0.99##
INLIER_RATIO= 0.1
MAX_PLANE = 8

USE_SAM = True

SAM_CONFIDENCE = 0.99
SAM_INLIER_RATIO = 0.2
SAM_MAX_PLANE = 4

POST_PROCESSING = False

global_mask = np.zeros_like(depth, dtype=np.uint8).flatten()
global_planes = np.array([], dtype=np.float32).reshape(0, 4)

# Use SAM to partition the image
if USE_SAM: 
    print("Using SAM to partition the image...")

    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    sam = sam_model_registry["default"](checkpoint="/scratchdata/sam_vit_h_4b8939.pth").to(DEVICE)
    mask_generator = SamAutomaticMaskGenerator(sam, stability_score_thresh=0.98)

    sam_masks = mask_generator.generate(rgb)
    print("SAM of Regions:", len(sam_masks))
    masks = sorted(sam_masks, key=lambda x: x["stability_score"])

    for sam_i, sam_mask in enumerate(sam_masks):
        valid_mask = sam_mask["segmentation"] & (depth > 0)
        valid_mask = valid_mask.flatten()

        mask, plane = information_optimisation(pts_3d, R, EPSILON, SIGMA, SAM_CONFIDENCE, SAM_INLIER_RATIO, SAM_MAX_PLANE, valid_mask=valid_mask, verbose=False)

        if len(plane) > 0:
            global_mask = np.where(mask > 0, mask+global_mask.max(), global_mask)
            global_planes = np.vstack((global_planes, plane))

valid_mask = (global_mask == 0) & (depth > 0).flatten()
mask, plane = information_optimisation(pts_3d, R, EPSILON, SIGMA, CONFIDENCE, INLIER_RATIO, MAX_PLANE, valid_mask=valid_mask, verbose=True)

global_mask = np.where(mask > 0, mask+global_mask.max(), global_mask)
global_planes = np.vstack((global_planes, plane))

global_mask = global_mask.reshape(depth.shape)

mask_PIL = Image.fromarray(global_mask)
mask_PIL.save(os.path.join(DATA_DIR, TARGET_FOLDER, f"{INDEX}.png"))

# Save the plane
with open(os.path.join(DATA_DIR, TARGET_FOLDER, f"{INDEX}.csv"), 'w') as f:
    writer = csv.writer(f)
    writer.writerows(global_planes)
