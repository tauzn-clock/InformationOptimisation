import torch
import numpy as np
from PIL import Image
import os
import yaml

from utils.visualise import mask_over_img

# Open yaml
with open("nyu.yaml", "r") as file:
    config = yaml.safe_load(file)

INTRINSICS = [config["fx"], config["fy"], config["cx"], config["cy"]]
R = config["depth_max"]  # Maximum range of sensor
EPSILON = config["resolution"]  # Resolution of the sensor

# Image dir
DATA_DIR = "/scratchdata/nyu_plane"
INDEX = 0

rgb = Image.open(os.path.join(DATA_DIR, "rgb", f"{INDEX}.png")).convert("RGB")
rgb = np.array(rgb)

depth = Image.open(os.path.join(DATA_DIR, "depth", f"{INDEX}.png")).convert("I;16")
depth = np.array(depth) * EPSILON

mask = Image.open(os.path.join(DATA_DIR, "mask", f"{INDEX}.png")).convert("L")
mask = np.array(mask)

mask_over_img(rgb, mask, "tmp.png")