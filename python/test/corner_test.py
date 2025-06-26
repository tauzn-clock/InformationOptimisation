import sys
import os 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from set_depth import set_depth
from information_optimisation import information_optimisation
from metric import plane_ordering
from utils.open3d_ransac import open3d_ransac
from utils.visualise import img_over_pcd, mask_to_hsv
from utils.process_depth import get_3d
np.random.seed(0)

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corner")
# Create folder if it doesnt exist
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

print(f"Saving to {SAVE_DIR}")

# Param
SIGMA_PROP = 5 # Change this param

H = 480
W = 640
INTRINSICS = [500, 500, W//2, H//2]

R = 10
EPSILON = 0.001

SIGMA = np.ones(H*W) * EPSILON * SIGMA_PROP
MAX_PLANE = 8
CONFIDENCE = 0.99
INLIER_THRESHOLD = 0.25

# Generate Depth
depth = np.zeros((H,W))
N = 3
distance = np.array([-1,-1,-1])
gt = np.zeros((H,W),dtype=int)
for i in range(3):
    mask = np.ones_like(depth,dtype=bool)
    angle = (120*i) * np.pi/180

    normal = np.array([np.cos(angle), np.sin(angle),1])
    distance = -0.7

    new_depth = set_depth(np.ones((H,W)),INTRINSICS, mask, normal, distance)
    gt[new_depth>depth] = i+1
    depth = np.maximum(depth, new_depth)

gt[depth>1.5] = 4
depth = np.clip(depth,0,1.5)
print(depth.max(), depth.min())

#Add noise
depth += np.random.normal(0,5 * EPSILON,(H,W))
depth = np.array(depth/EPSILON,dtype=int) * EPSILON

print(depth.max(), depth.min())

pcd = get_3d(depth, INTRINSICS)

plt.imsave(f"{SAVE_DIR}/corner.png",depth,cmap='gray')

mask, planes = open3d_ransac(depth, INTRINSICS, EPSILON * SIGMA_PROP, CONFIDENCE, INLIER_THRESHOLD, MAX_PLANE, verbose=True)

img_over_pcd(pcd, mask_to_hsv(mask), filepath=f"{SAVE_DIR}/{SIGMA_PROP}_default_pcd_corner.png")

R = depth.max() - depth.min()
print(R)
mask, plane = information_optimisation(pcd, R, EPSILON, SIGMA, CONFIDENCE, INLIER_THRESHOLD, MAX_PLANE, verbose=True)
mask, planes = plane_ordering(pcd, mask, planes, R, EPSILON, SIGMA, keep_index=mask.max())
print(mask.max())

img_over_pcd(pcd, mask_to_hsv(mask.reshape(depth.shape)), filepath=f"{SAVE_DIR}/{SIGMA_PROP}_our_pcd_corner.png")
