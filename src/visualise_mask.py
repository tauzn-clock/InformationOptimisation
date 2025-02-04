from process_depth_img import depth_to_pcd
import numpy as np
import open3d as o3d
from PIL import Image
import csv
import os

def filter_mask(mask, csv_data):
    new_mask = np.zeros_like(mask)
    new_csv_data = []
    next_label = 1
    for i in range(1, mask.max()+1):
        if np.sum(mask==i) > 0:
            new_mask[mask==i] = next_label
            new_csv_data.append(csv_data[i-1])
            next_label += 1
    return new_mask, np.array(new_csv_data)

def merge_mask(mask, csv_data, ANGLE_THRESHOLD=0.1, DIST_THRESHOLD=0.1):
    N = csv_data.shape[0]

    new_mask = np.zeros_like(mask)
    new_csv_data = np.zeros_like(csv_data)
    index = np.linspace(0, N-1, N, dtype=np.int32)

    csv_data[csv_data[:,3] <0] *= -1

    for i in range(N):
        closest = i
        best_angle = ANGLE_THRESHOLD
        best_dist = DIST_THRESHOLD
        for j in range(i):
            angle = 1 - np.dot(csv_data[i,:3], csv_data[index[j],:3])
            dist = abs(csv_data[i,3] - csv_data[index[j],3])
            if angle < best_angle and dist < best_dist:
                closest = j
                best_angle = angle
                best_dist = dist
        new_mask[mask==i+1] = index[closest]+1
        index[i] = index[closest]
        if i == closest:
            new_csv_data[i] = csv_data[closest]
        else:
            total = mask[mask==i+1].sum() + mask[mask==closest+1].sum()
            new_csv_data[i] = (csv_data[i]*mask[mask==i+1].sum() + csv_data[closest]*mask[mask==closest+1].sum())/total
        
    return filter_mask(new_mask,new_csv_data)

def visualise_mask(depth, mask, intrinsics):
    points, _ = depth_to_pcd(depth, intrinsics) 

    # Visualize the point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    color = np.zeros((points.shape[0], 3))

    mask = mask.flatten()
    for i in range(1, mask.max()+1):
        color[mask==i] = np.random.rand(3)
    point_cloud.colors = o3d.utility.Vector3dVector(color)

    o3d.visualization.draw_geometries([point_cloud])

if __name__ == '__main__':
    root = "/scratchdata/nyu_plane"
    data_csv = "/HighResMDE/get_planes/ransac/config/nyu.csv"

    with open(data_csv, 'r') as f:
        reader = csv.reader(f)
        DATA = list(reader)

    INDEX = 6

    data = DATA[INDEX]

    INTRINSICS = [float(data[2]), 0, float(data[4]), 0, 0, float(data[3]), float(data[5]), 0] # fx, fy, cx, cy
    INTRINSICS = np.array(INTRINSICS)

    depth = np.array(Image.open(os.path.join(root, data[1])))/float(data[6])
    mask = np.array(Image.open(os.path.join(root, "original_gt", f"{INDEX}.png")))
    mask = np.array(Image.open(os.path.join(root, "new_gt", f"{INDEX}.png")))

    with open(os.path.join(root, "new_gt", f"{INDEX}.csv"), 'r') as f:
        reader = csv.reader(f)
        csv_data = list(reader)

    csv_data = np.array(csv_data, dtype=np.float32)

    print(mask.max())
    mask, csv_data = merge_mask(mask, csv_data)
    print(mask.max())
    visualise_mask(depth, mask, INTRINSICS)