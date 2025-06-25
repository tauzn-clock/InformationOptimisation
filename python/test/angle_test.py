# Old script used to get CSV
# Kept for documentation purposes


import sys
sys.path.append("/HighResMDE/get_planes/ransac")

import numpy as np
from information_optimisation import plane_ransac
from visualise import visualise_mask, save_mask
from synthetic_test import set_depth, open3d_find_planes
from metrics import plane_ordering
from depth_to_pcd import depth_to_pcd
np.random.seed(0)

ROOT = "/HighResMDE/get_planes/angle"
NOISE_LEVEL = 2
ANGLE = 150

H = 480
W = 640
EPSILON = 0.001
SIGMA = np.ones(H*W) * EPSILON * NOISE_LEVEL
MAX_PLANE = 2
CONFIDENCE = 0.99
INLIER_THRESHOLD = 0.15
INTRINSICS = np.array([500, 0, W//2, 0, 0, 500, H//2])

Z = np.ones((H,W)).flatten()

x, y = np.meshgrid(np.arange(W), np.arange(H))
x = x.flatten()
y = y.flatten()
fx, fy, cx, cy = INTRINSICS[0], INTRINSICS[5], INTRINSICS[2], INTRINSICS[6]
x_3d = (x - cx) * Z / fx
y_3d = (y - cy) * Z / fy
POINTS = np.vstack((x_3d, y_3d, Z)).T
DIRECTION_VECTOR = POINTS / (np.linalg.norm(POINTS, axis=1)[:, None]+1e-7)

DIRECTION_VECTOR = DIRECTION_VECTOR.reshape(H,W,3)

depth = np.zeros((H,W))
gt = np.zeros((H,W))
N = 2
angle = (180 - ANGLE) * np.pi/180 /2
normal = [[0,np.sin(angle), np.cos(angle)],[0,-np.sin(angle), np.cos(angle)]]
distance = [-normal[0][2],-normal[1][2]]
for i in range(N):
    mask = np.zeros((H,W))
    mask[i*H//N:(i+1)*H//N,:] = 1
    gt[mask==1] = i+1

    depth += set_depth(np.ones((H,W)),INTRINSICS, mask, normal[i], distance[i])

#Add noise
depth += np.random.normal(0, 5 * EPSILON,(H,W))
depth = np.array(depth/EPSILON,dtype=int) * EPSILON

print(depth.max(), depth.min())
#visualise_mask(depth, np.zeros_like(depth,dtype=np.int8), INTRINSICS)

import matplotlib.pyplot as plt
plt.imsave(f"{ROOT}/staircase.png",depth,cmap='gray')
#visualise_mask(depth, np.zeros(H*W,dtype=int), INTRINSICS, filepath=f"{ROOT}/stair_gt.png",skip_color=True)

mask, open3d_planes = open3d_find_planes(depth, INTRINSICS, EPSILON * NOISE_LEVEL, CONFIDENCE, INLIER_THRESHOLD, MAX_PLANE, verbose=True)

#save_mask(mask, f"{ROOT}/{NOISE_LEVEL}_default_stair.png")
#visualise_mask(depth, mask, INTRINSICS, filepath=f"{ROOT}/{NOISE_LEVEL}_default_pcd_stair.png")

R = depth.max() - depth.min()
print(R)
information, mask, planes = plane_ransac(depth, INTRINSICS, R, EPSILON, SIGMA, CONFIDENCE, INLIER_THRESHOLD, MAX_PLANE, verbose=True)
print(information)
print(planes)

smallest = np.argmin(information)
mask[mask>smallest] = 0
planes = planes[1:smallest+1]
print(mask.max())

points, index = depth_to_pcd(depth, INTRINSICS)
mask, planes = plane_ordering(points, mask, planes, R, EPSILON, SIGMA,keep_index=mask.max())

store = []

print("Open3D error")
sum_angle_error = 0
sum_dist_error = 0
for i in range(len(open3d_planes)):
    if open3d_planes[i,2] < 0:
        open3d_planes[i] = -open3d_planes[i]

    angle_error = 180
    dist_error = np.inf
    for j in range(N):
        normal_error = np.arccos(np.abs(np.dot(normal[j], open3d_planes[i,:3])))*180/np.pi
        if normal_error < angle_error:
            angle_error = normal_error
            dist_error = np.abs(open3d_planes[i,3] - distance[j])

    print(f"Plane {i+1}: Angle Error: {angle_error}, Distance Error: {dist_error}")
    sum_angle_error += angle_error
    sum_dist_error += dist_error

print(f"Average Angle Error: {sum_angle_error/len(open3d_planes)}, Average Distance Error: {sum_dist_error/len(open3d_planes)}")
store.append(sum_angle_error/len(open3d_planes))
store.append(sum_dist_error/len(open3d_planes))

print("Our error")
sum_angle_error = 0
sum_dist_error = 0
for i in range(len(planes)):
    if planes[i,2] < 0:
        planes[i] = -planes[i]

    angle_error = 180
    dist_error = np.inf
    for j in range(N):
        normal_error = np.arccos(np.abs(np.dot(normal[j], planes[i,:3])))*180/np.pi
        if normal_error < angle_error:
            angle_error = normal_error
            dist_error = np.abs(planes[i,3] - distance[j])

    print(f"Plane {i+1}: Angle Error: {angle_error}, Distance Error: {dist_error}")
    sum_angle_error += angle_error
    sum_dist_error += dist_error

print(f"Average Angle Error: {sum_angle_error/len(planes)}, Average Distance Error: {sum_dist_error/len(planes)}")
store.append(sum_angle_error/len(planes))
store.append(sum_dist_error/len(planes))

if True:
    import csv

    with open("2.csv", mode='r') as file:
        data = list(csv.reader(file))
    
    data.append(store)

    with open("2.csv", mode='w') as file:
        writer = csv.writer(file)
        writer.writerows(data)

#save_mask(mask.reshape(H,W), f"{ROOT}/{NOISE_LEVEL}_ours_stair.png")
#visualise_mask(depth, mask, INTRINSICS, filepath=f"{ROOT}/{NOISE_LEVEL}_ours_pcd_stair.png")