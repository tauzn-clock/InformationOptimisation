import sys
import os 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt

DATA_DIR = "/scratchdata/test"

with open(os.path.join(DATA_DIR, "camera_info.json"), "r") as f:
    camera_info = json.load(f)

INTRINSICS = [camera_info["K"][0], camera_info["K"][4], camera_info["K"][2], camera_info["K"][5]]
print(INTRINSICS)

N = 1

NOISE_LEVEL = 10
R = 10
EPSILON = 0.001
CONFIDENCE = 0.99
INLIER_THRESHOLD = 0.1
MAX_PLANE = 24
SIGMA_RATIO = 0.01

COLOR_DEPTH = True
OPEN3D_MASK = True
OPEN3D_COLOR = True
OUR_MASK = False
OUR_COLOR = True
COMBINE = True

if COLOR_DEPTH:
    def depth_to_color(depth):
        depth = depth / depth.max()
        H, W = depth.shape
        color = np.zeros((H, W, 3))
        for i in range(H):
            for j in range(W):
                color[i,j] = plt.cm.viridis(depth[i,j])[0:3]
        color[depth == 0] = 0
        return color
    DEPTH_COLOR = os.path.join(DATA_DIR, "depth_color")
    if not os.path.exists(DEPTH_COLOR):
        os.makedirs(DEPTH_COLOR)
        print(f"Directory '{DEPTH_COLOR}' created.")
    for i in range(N):
        depth = Image.open(os.path.join(DATA_DIR, "depth", f"{i}.png"))
        depth = np.array(depth)
        depth = depth_to_color(depth)
        plt.imsave(f"{DATA_DIR}/depth_color/{i}.png", depth)

if OPEN3D_MASK:
    from utils.open3d_ransac import open3d_ransac
    OPEN3D_PTH = os.path.join(DATA_DIR, "open3d")
    if not os.path.exists(OPEN3D_PTH):
        os.makedirs(OPEN3D_PTH)
        print(f"Directory '{OPEN3D_PTH}' created.")
    for i in range(N):
        depth = Image.open(os.path.join(DATA_DIR, "depth", f"{i}.png"))
        depth = np.array(depth) * EPSILON

        mask, planes = open3d_ransac(depth, INTRINSICS, EPSILON * NOISE_LEVEL, CONFIDENCE, INLIER_THRESHOLD, MAX_PLANE, verbose=True, zero_depth=True)

        mask_PIL = Image.fromarray(mask.astype(np.uint8))
        mask_PIL.save(os.path.join(DATA_DIR, "open3d", f"{i}.png"))

if OPEN3D_COLOR:
    from utils.visualise import mask_to_hsv
    OPEN3D_PTH = os.path.join(DATA_DIR, "open3d_color")
    if not os.path.exists(OPEN3D_PTH):
        os.makedirs(OPEN3D_PTH)
        print(f"Directory '{OPEN3D_PTH}' created.")

    for i in range(N):
        mask = Image.open(os.path.join(DATA_DIR, "open3d", f"{i}.png"))
        mask = np.array(mask)

        mask_color = mask_to_hsv(mask)
        plt.imsave(os.path.join(DATA_DIR, "open3d_color", f"{i}.png"), mask_color)

if OUR_MASK:
    # Faster to use cpp implementation

    from information_optimisation import information_optimisation
    from utils.process_depth import get_3d

    OUR_PTH = os.path.join(DATA_DIR, "our")
    if not os.path.exists(OUR_PTH):
        os.makedirs(OUR_PTH)
        print(f"Directory '{OUR_PTH}' created.")

    for i in range(N):
        depth = Image.open(os.path.join(DATA_DIR, "depth", f"{i}.png"))
        depth = np.array(depth) * EPSILON

        pts_3d = get_3d(depth, INTRINSICS)

        SIGMA = SIGMA_RATIO * depth # Proportional noise model
        SIGMA = SIGMA.flatten()

        valid_mask = (depth > 0).flatten()
        mask, plane = information_optimisation(pts_3d, R, EPSILON, SIGMA, CONFIDENCE, INLIER_THRESHOLD, MAX_PLANE, valid_mask=valid_mask, verbose=True)

        mask_PIL = Image.fromarray(mask.astype(np.uint8).reshape(depth.shape))
        mask_PIL.save(os.path.join(DATA_DIR, "our", f"{i}.png"))

if OUR_COLOR:
    from utils.visualise import mask_to_hsv
    OUR_PTH = os.path.join(DATA_DIR, "our_color")
    if not os.path.exists(OUR_PTH):
        os.makedirs(OUR_PTH)
        print(f"Directory '{OUR_PTH}' created.")

    for i in range(N):
        mask = Image.open(os.path.join(DATA_DIR, "our", f"{i}.png"))
        mask = np.array(mask)

        mask_color = mask_to_hsv(mask)
        plt.imsave(os.path.join(DATA_DIR, "our_color", f"{i}.png"), mask_color)

if COMBINE:
    COMBINED = os.path.join(DATA_DIR, "combined")
    if not os.path.exists(COMBINED):
        os.makedirs(COMBINED)
        print(f"Directory '{COMBINED}' created.")
    for i in range(N):
        img = plt.imread(os.path.join(DATA_DIR, "rgb", f"{i}.png"))
        depth = plt.imread(os.path.join(DATA_DIR, "depth_color", f"{i}.png"))
        open3d = plt.imread(os.path.join(DATA_DIR, "open3d_color", f"{i}.png"))
        our = plt.imread(os.path.join(DATA_DIR, "our_color", f"{i}.png"))

        H, W, _ = img.shape

        combined = np.zeros((2*H, 2*W, 3))

        combined[:H, :W] = img
        combined[:H, W:] = depth[:,:,:3]
        combined[H:, :W] = open3d[:,:,:3]
        combined[H:, W:] = our[:,:,:3]

        plt.imsave(f"{DATA_DIR}/combined/{i}.png", combined)