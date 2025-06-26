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
from utils.process_depth import get_3d, get_normal_adj
np.random.seed(0)

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "limitation")
# Create folder if it doesnt exist
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
    
print(f"Saving to {SAVE_DIR}")

# Param
SIGMA_PROP = 20 # Change this param

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
gt = np.zeros((H,W))
N = 2
normal = [[0.707,0, 0.707],[-0.707, 0, 0.707]]
distance = [-1,-2]

mask = np.zeros((H,W))
mask[:,:int(0.5*W)] = 1
gt[mask==1] = 0+1
depth += set_depth(np.ones((H,W)),INTRINSICS, mask, normal[0], distance[0])

mask = np.zeros((H,W))
mask[:,int(0.5*W):] = 1
gt[mask==1] = 1+1
depth += set_depth(np.ones((H,W)),INTRINSICS, mask, normal[1], distance[1])

depth = np.array(depth/EPSILON,dtype=int) * EPSILON

pcd = get_3d(depth, INTRINSICS)

plt.imsave(f"{SAVE_DIR}/limitation.png",depth,cmap='gray')

R = depth.max() - depth.min()
print(R)

mask, plane = information_optimisation(pcd, R, EPSILON, SIGMA, CONFIDENCE, INLIER_THRESHOLD, MAX_PLANE, verbose=True)
print(mask.max())

img_over_pcd(pcd, mask_to_hsv(mask.reshape(depth.shape)), filepath=f"{SAVE_DIR}/{SIGMA_PROP}_our_pcd_limitation.png")

normal = get_normal_adj(depth, INTRINSICS)
mask, plane = information_optimisation(pcd, R, EPSILON, SIGMA, CONFIDENCE, INLIER_THRESHOLD, MAX_PLANE, verbose=True, normal_remap=normal)
print(mask.max())

img_over_pcd(pcd, mask_to_hsv(mask.reshape(depth.shape)), filepath=f"{SAVE_DIR}/{SIGMA_PROP}_our_pcd_limitation_normal.png")

#Add noise
depth += np.random.normal(0, 10 * EPSILON,(H,W))
depth = np.array(depth/EPSILON,dtype=int) * EPSILON

pcd = get_3d(depth, INTRINSICS)

normal = get_normal_adj(depth, INTRINSICS)
mask, plane = information_optimisation(pcd, R, EPSILON, SIGMA, CONFIDENCE, INLIER_THRESHOLD, MAX_PLANE, verbose=True, normal_remap=normal)
print(mask.max())

img_over_pcd(pcd, mask_to_hsv(mask.reshape(depth.shape)), filepath=f"{SAVE_DIR}/{SIGMA_PROP}_our_pcd_limitation_noise.png")
