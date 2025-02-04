from process_depth_img import depth_to_pcd
from information_estimation import default_ransac
import open3d as o3d
import numpy as np

EPSILON = 1 # Resolution
R = 2**16 # Maximum Range
SIGMA = EPSILON  # Normal std

CONFIDENCE = 0.999
INLIER_THRESHOLD = 0.25
MAX_PLANE = 4

INTRINSICS = [525.0, 0, 319.5, 0, 0, 525.0, 239.5, 0] # fx, fy, cx, cy
H = 480
W = 640

depth = np.random.randint(0, R, (H, W))
depth[240:480, 320:640] = R/2
depth[120:360, 160:480] = R/4

points, index = depth_to_pcd(depth, INTRINSICS)

information, mask, plane = default_ransac(points, R, EPSILON, SIGMA, CONFIDENCE, INLIER_THRESHOLD, MAX_PLANE)

for i in range(MAX_PLANE+1):
    print(f"Cnt {i}", np.sum(mask[i]==i))
print("Planes: ", plane)

print("Information:", information)

# Visualize the point cloud
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)
color = np.zeros((points.shape[0], 3))
#Find index of smallest information
min_idx = np.argmin(information)
print("Found Planes", min_idx)

for i in range(1, min_idx+1):
    color[mask[i]==i] = np.random.rand(3)
point_cloud.colors = o3d.utility.Vector3dVector(color)

o3d.visualization.draw_geometries([point_cloud])

