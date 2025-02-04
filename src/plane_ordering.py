from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
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
from visualise_mask import merge_mask

def plane_ordering(POINTS, mask, param, R, EPSILON, SIGMA):
    SPACE_STATES = np.log(R/EPSILON)
    PER_POINT_INFO = np.log(SIGMA/(EPSILON+1e-7) + 1e-7)
    PER_POINT_INFO += 0.5 * np.log(2*np.pi) - SPACE_STATES
    TWO_SIGMA_SQUARE = 2 * (SIGMA**2 + 1e-7)

    direction_vector = POINTS / (np.linalg.norm(POINTS, axis=1)[:, None]+1e-7)

    store = []

    for i in range(len(param)):
        norm = param[i,:3]
        d = param[i,3]
        masked_pts = POINTS[mask==i+1]
        masked_direction_vector = direction_vector[mask==i+1]
        masked_z = POINTS[mask==i+1][:,2]

        error = ((-d/(np.dot(masked_direction_vector, norm.T)+1e-7))*masked_direction_vector[:,2] - masked_z) ** 2
        error = error / TWO_SIGMA_SQUARE[mask==i+1] + PER_POINT_INFO[mask==i+1]

        store.append((len(error), error.sum(), error.mean()))
    
    store = np.array(store)

    new_mask = np.zeros_like(mask)
    new_param = np.zeros_like(param)

    print(store[:,0])
    index = np.argsort(store[:,0])[::-1]
    index = np.argsort(store[:,1])
    #index = np.argsort(store[:,2])

    print(index)

    for i in range(len(index)):
        print(store[index[i],0])
        new_mask[mask==index[i]+1] = i+1
        new_param[i] = param[index[i]]

    return new_mask, new_param

ROOT = "/scratchdata/nyu_plane" 
data_csv = "/HighResMDE/get_planes/ransac/config/nyu.csv"

with open(data_csv, 'r') as f:
    reader = csv.reader(f)
    DATA = list(reader)

for frame_cnt in range(0,len(DATA)):
    data = DATA[frame_cnt]
    #data = ["rgb/90.png", "depth/90.png", 306.75604248046875, 306.7660827636719, 322.9314270019531, 203.91506958007812, 1, 2**16]

    INTRINSICS = [float(data[2]), 0, float(data[4]), 0, 0, float(data[3]), float(data[5]), 0] # fx, fy, cx, cy
    INTRINSICS = np.array(INTRINSICS)

    EPSILON = 1/float(data[6]) # Resolution
    R = float(data[7]) # Maximum Range

    depth = Image.open(os.path.join(ROOT, data[1]))
    depth = np.array(depth) /float(data[6])
    H, W = depth.shape


    mask = np.array(Image.open(os.path.join(ROOT, "new_gt", f"{frame_cnt}.png")))

    with open(os.path.join(ROOT, "new_gt", f"{frame_cnt}.csv"), 'r') as f:
        reader = csv.reader(f)
        csv_data = list(reader)

    csv_data = np.array(csv_data, dtype=np.float32)

    print(len(csv_data))
    
    mask, csv_data = merge_mask(mask, csv_data,0.05,0.05)

    print(len(csv_data))

    points, index = depth_to_pcd(depth, INTRINSICS)
    SIGMA = 0.02 * points[:,2]
    
    mask, csv_data = plane_ordering(points, mask.flatten(), csv_data, R, EPSILON, SIGMA)
    
    break

points, _ = depth_to_pcd(depth, INTRINSICS) 

# Visualize the point cloud
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)
color = np.zeros((points.shape[0], 3))

mask = mask.flatten()
for i in range(1,10):
    color[mask==i] = np.random.rand(3)
point_cloud.colors = o3d.utility.Vector3dVector(color)

o3d.visualization.draw_geometries([point_cloud])