import sys
import os 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from set_depth import set_depth
from information_estimation import information_estimation
from metric import plane_ordering
from utils.open3d_ransac import open3d_ransac
from utils.visualise import img_over_pcd, mask_to_hsv
from utils.process_depth import get_3d
np.random.seed(0)

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flatness")
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
noise_level = [0,2,10,100]
N = 4
for i in range(N):
    plane_mask = np.zeros((H,W),dtype=bool)
    plane_mask[:,i*W//N:(i+1)*W//N] = True
    depth += set_depth(np.ones((H,W)),INTRINSICS, plane_mask, [0,0,1], -0.2 * i - 2.5)
    #mask[plane_mask] = i+1

    if i!=0:
        amplitude = NOISE_LEVEL * EPSILON
        x = np.linspace(0,1,H)
        y = np.sin(2*np.pi*x*noise_level[i]) * amplitude
        noise = np.tile(y,(W,1)).T
        print(noise.shape)
        #noise = np.random.normal(0,EPSILON*25,(H,W))
        #noise_mask = (random_mask < noise_level[i]) & plane_mask
        depth[plane_mask] += noise[plane_mask]

depth = np.array(depth/EPSILON,dtype=int) * EPSILON

print(depth.max(), depth.min())

pcd = get_3d(depth, INTRINSICS)

plt.imsave(f"{SAVE_DIR}/our.png",depth,cmap='gray')

mask, planes = open3d_ransac(depth, INTRINSICS, EPSILON * NOISE_LEVEL, CONFIDENCE, INLIER_THRESHOLD, MAX_PLANE, verbose=True)

img_over_pcd(pcd, mask_to_hsv(mask), filepath=f"{SAVE_DIR}/{NOISE_LEVEL}_default_pcd_our.png")

R = depth.max() - depth.min()
print(R)
mask, plane = information_estimation(pcd, R, EPSILON, SIGMA, CONFIDENCE, INLIER_THRESHOLD, MAX_PLANE, verbose=True)
mask, planes = plane_ordering(pcd, mask, planes, R, EPSILON, SIGMA, keep_index=mask.max())
print(mask.max())

img_over_pcd(pcd, mask_to_hsv(mask.reshape(depth.shape)), filepath=f"{SAVE_DIR}/{NOISE_LEVEL}_our_pcd_our.png")
