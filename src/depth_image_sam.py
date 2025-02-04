import sys
sys.path.append('/HighResMDE/segment-anything')

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from PIL import Image
import numpy as np
from process_depth_img import depth_to_pcd
from information_estimation import default_ransac, plane_ransac
import open3d as o3d
import csv
import os
from PIL import Image
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from post_processing import post_processing

DEVICE="cuda:0"
root = "/scratchdata/nyu_plane"
data_csv = "/HighResMDE/get_planes/ransac/config/nyu.csv"

sam = sam_model_registry["vit_h"](checkpoint="/scratchdata/sam_vit_h_4b8939.pth").to(DEVICE)
mask_generator = SamAutomaticMaskGenerator(sam, stability_score_thresh=0.95)

with open(data_csv, 'r') as f:
    reader = csv.reader(f)
    data = list(reader)

data = data[0]
#data = ["rgb/90.png", "depth/90.png", 306.75604248046875, 306.7660827636719, 322.9314270019531, 203.91506958007812, 1, 2**16]

INTRINSICS = [float(data[2]), 0, float(data[4]), 0, 0, float(data[3]), float(data[5]), 0] # fx, fy, cx, cy
INTRINSICS = np.array(INTRINSICS)

img = Image.open(os.path.join(root, data[0]))
img = np.array(img)
sam_masks = mask_generator.generate(img)

depth = Image.open(os.path.join(root, data[1]))
depth = np.array(depth) /float(data[6])
H, W = depth.shape
#depth = get_plane(H,W,INTRINSICS)

START = time.time()

valid_mask = depth > 0

EPSILON = 1/float(data[6]) # Resolution
R = float(data[7]) # Maximum Range
SIGMA = EPSILON * 5 # Normal std

CONFIDENCE = 0.99
INLIER_THRESHOLD = 0.2#5e4/(H*W)
MAX_PLANE = 4

points, index = depth_to_pcd(depth, INTRINSICS)
SIGMA = 0.02 * points[:,2]

global_mask = np.zeros((H, W)).flatten()
total_planes = 0

masks = sorted(sam_masks, key=lambda x: x["stability_score"])

print(len(masks))

remaining_masks = np.ones_like(depth, dtype=bool)
remaining_masks = remaining_masks & (depth > 0)

for sam_i, sam_mask in enumerate(sam_masks):
    valid_mask = sam_mask["segmentation"] & (depth > 0)
    if valid_mask.sum() <= 200: continue
    plt.imsave("mask.png", sam_mask["segmentation"])
    remaining_masks = remaining_masks & ~valid_mask

    information, mask, plane = plane_ransac(depth, INTRINSICS, R, EPSILON, SIGMA, CONFIDENCE, INLIER_THRESHOLD, MAX_PLANE, valid_mask.flatten(),verbose=False)

    # Post Processing
    information, mask, plane = post_processing(depth, INTRINSICS, R, EPSILON, SIGMA, information, mask, plane, valid_mask)

    print(information)

    min_idx = np.argmin(information)
    print("Min Planes: ", min_idx)
    for i in range(1, min_idx+1):
        global_mask[mask==i] = total_planes + i

    total_planes += min_idx

MAX_PLANE = 16
if True:
    information, mask, plane = plane_ransac(depth, INTRINSICS, R, EPSILON, SIGMA, CONFIDENCE, INLIER_THRESHOLD, MAX_PLANE, remaining_masks.flatten(),verbose=True)

    # Post Processing
    information, mask, plane = post_processing(depth, INTRINSICS, R, EPSILON, SIGMA, information, mask, plane, remaining_masks)

    print(information)

    min_idx = np.argmin(information)
    print("Min Planes: ", min_idx)
    for i in range(1, min_idx+1):
        global_mask[mask==i] = total_planes + i

    total_planes += min_idx

plt.imsave("mask.png", global_mask.reshape(H, W))

print(total_planes)

# Visualize the point cloud
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)
color = np.zeros((points.shape[0], 3))

for i in range(1, total_planes+1):
    color[global_mask==i] = np.random.rand(3)
point_cloud.colors = o3d.utility.Vector3dVector(color)

o3d.visualization.draw_geometries([point_cloud])
