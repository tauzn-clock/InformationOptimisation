from test_pcd import get_plane
from information_estimation import default_ransac
import open3d as o3d
import numpy as np

EPSILON = 0.1 # Resolution
R = 10 # Maximum Range
SIGMA = EPSILON  # Normal std

CONFIDENCE = 0.999
INLIER_THRESHOLD = 0.167
MAX_PLANE = 5

plane_params = [[0, 0, 1, 0], [0.2, 0, 1, 0]]

points = np.zeros((0,3))
for param in plane_params:
    points = np.vstack((points, get_plane(R, EPSILON, param, 1)))

print(points.shape)

information, mask, plane = default_ransac(points, R, EPSILON, SIGMA, CONFIDENCE, INLIER_THRESHOLD, MAX_PLANE)

for i in range(MAX_PLANE+1):
    print(f"Cnt {i}", np.sum(mask[i]==i))
print("Planes: ", plane)

#Find index of smallest information
min_idx = np.argmin(information)
print("Found Planes", min_idx)

print("Information:", information)

dist = points @ plane[1:min_idx+1,:3].T + np.stack([plane[1:min_idx+1,3]]*points.shape[0], axis=0)
dist = np.abs(dist)
isPartofPlane = mask != 0
mask = np.argmin(dist, axis=1) + 1
mask = mask * isPartofPlane

# Visualize the point cloud
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)
color = np.zeros((points.shape[0], 3))

for i in range(1, min_idx+1):
    color[mask[i]==i] = np.random.rand(3)
point_cloud.colors = o3d.utility.Vector3dVector(color)

o3d.visualization.draw_geometries([point_cloud])
