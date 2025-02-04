from process_depth_img import depth_to_pcd
from information_estimation import default_ransac, plane_ransac
import open3d as o3d
import numpy as np
import csv
import os
from PIL import Image
import time
import matplotlib.pyplot as plt
from post_processing import post_processing
from test_pcd import get_plane

root = "/scratchdata/nyu_plane"
data_csv = "/HighResMDE/get_planes/ransac/config/nyu.csv"

with open(data_csv, 'r') as f:
    reader = csv.reader(f)
    data = list(reader)

data = data[0]
#data = ["rgb/90.png", "depth/90.png", 306.75604248046875, 306.7660827636719, 322.9314270019531, 203.91506958007812, 1, 2**16]

INTRINSICS = [float(data[2]), 0, float(data[4]), 0, 0, float(data[3]), float(data[5]), 0] # fx, fy, cx, cy
INTRINSICS = np.array(INTRINSICS)

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
INLIER_THRESHOLD = 5e4/(H*W)
MAX_PLANE = 10

points, index = depth_to_pcd(depth, INTRINSICS)
SIGMA = 0.02 * points[:,2]
#SIGMA = 2 * points[:,2]**2 + 1.4 * points[:,2] + 1.1057
#SIGMA = 9 * points[:,2]**2 - 26.5 * points[:,2] + 20.237
#SIGMA *= 1e-3
#information, mask, plane = default_ransac(points, R, EPSILON, SIGMA, CONFIDENCE, INLIER_THRESHOLD, MAX_PLANE, valid_mask.flatten())
information, mask, plane = plane_ransac(depth, INTRINSICS, R, EPSILON, SIGMA, CONFIDENCE, INLIER_THRESHOLD, MAX_PLANE, valid_mask.flatten(),verbose=True)

print("Time Taken: ", time.time()-START)

print("Total Points: ", valid_mask.sum())

for i in range(1,MAX_PLANE+1):
    print(f"Cnt {i}", np.sum(mask==i))
print("Planes: ", plane)

#Find index of smallest information
min_idx = np.argmin(information)
print("Found Planes", min_idx)

print("Information:", information)

# Post Processing
information, mask, plane = post_processing(depth, INTRINSICS, R, EPSILON, SIGMA, information, mask, plane, valid_mask)

#Find index of smallest information
min_idx = np.argmin(information)
print("Found Planes", min_idx)

print("Information:", information)

# Visualize the point cloud
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)
color = np.zeros((points.shape[0], 3))

for i in range(1, min_idx+1):
    color[mask==i] = np.random.rand(3)
point_cloud.colors = o3d.utility.Vector3dVector(color)

o3d.visualization.draw_geometries([point_cloud])
