import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import open3d as o3d
from process_depth import get_3d

def open3d_ransac(depth, INTRINSICS, SIGMA, CONFIDENCE, INLIER_THRESHOLD, MAX_PLANE, verbose=True, zero_depth=False):
    final_mask = np.zeros_like(depth,dtype=int)
    final_planes = []

    points = get_3d(depth, INTRINSICS)

    height, width = depth.shape
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    x = x.flatten()
    y = y.flatten()
    index = np.column_stack((y,x))

    if zero_depth:
        final_mask[depth == 0] = -1

    pcd = o3d.geometry.PointCloud()
    
    ITERATION = int(np.log(1 - CONFIDENCE) / np.log(1 - INLIER_THRESHOLD**3))
    print("Iteration: ", ITERATION)

    for i in range(MAX_PLANE):
        if (len(points[final_mask.flatten()==0])<3):
            break
        pcd.points = o3d.utility.Vector3dVector(points[final_mask.flatten()==0])
        plane_model, plane_inliers = pcd.segment_plane(SIGMA, 3, ITERATION)
        [a, b, c, d] = plane_model
        if verbose:
            print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

        plane_index = index[final_mask.flatten()==0][plane_inliers]
        final_mask[plane_index[:,0],plane_index[:,1]] = i+1
        
        final_planes.append([a,b,c,d])

    if zero_depth:
        final_mask[final_mask == -1] = 0

    return final_mask, np.array(final_planes)