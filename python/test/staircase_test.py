import sys
import os 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from set_depth import set_depth
from utils.open3d_ransac import open3d_ransac

np.random.seed(0)

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "staircase")
print(f"Saving to {SAVE_DIR}")

# Param
NOISE_LEVEL = 5 # Change this param

H = 480
W = 640
INTRINSICS = [500, 500, W//2, H//2]

R = 10
EPSILON = 0.001

SIGMA = np.ones(H*W) * EPSILON * NOISE_LEVEL
MAX_PLANE = 8
CONFIDENCE = 0.99
INLIER_THRESHOLD = 0.25

# Generate Depth
depth = np.zeros((H,W))
gt = np.zeros((H,W),dtype=int)
N = 4
distance = [1.63,1,1,1.63]
distance = np.array(distance) * -(0.5)**0.5
for i in range(N):
    mask = np.zeros((H,W))
    mask[i*H//N:(i+1)*H//N,:] = 1
    gt[mask==1] = i+1

    angle = -(-1)**i * np.pi/4

    normal = np.array([0,np.sin(angle), np.cos(angle)])

    depth += set_depth(np.ones((H,W)),INTRINSICS, mask, normal, distance[i])

#Add noise
depth += np.random.normal(0, 5 * EPSILON,(H,W))
depth = np.array(depth/EPSILON,dtype=int) * EPSILON

print(depth.max(), depth.min())

plt.imsave(f"{SAVE_DIR}/staircase.png",depth,cmap='gray')

mask, planes = open3d_ransac(depth, INTRINSICS, EPSILON * NOISE_LEVEL, CONFIDENCE, INLIER_THRESHOLD, MAX_PLANE, verbose=True)

R = depth.max() - depth.min()
print(R)
information, mask, planes = plane_ransac(depth, INTRINSICS, R, EPSILON, SIGMA, CONFIDENCE, INLIER_THRESHOLD, MAX_PLANE, verbose=True)
print(information)
print(planes)