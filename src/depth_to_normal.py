import numpy as np
import csv
import os
from PIL import Image
import matplotlib.pyplot as plt
from process_depth_img import depth_to_pcd
from scipy.ndimage import gaussian_filter

def Depth2Normal(pt, s, gaussian_kernel):
    padded_pt = np.pad(pt, ((s, s), (s, s), (0, 0)), mode='constant')
    dx = (padded_pt[s:-s, 2*s:, :] - padded_pt[s:-s, :-2*s, :])
    dy = (padded_pt[2*s:, s:-s, :] - padded_pt[:-2*s, s:-s, :])
    
    normal = np.cross(dx, dy)
    normal = normal/(np.linalg.norm(normal, axis=2, keepdims=True)+1e-7)

    if gaussian_kernel is not None:
        normal = gaussian_filter(normal, sigma=gaussian_kernel)
        normal = normal / (np.linalg.norm(normal, axis=2, keepdims=True)+1e-7)

    return normal

if __name__ == "__main__":
    root = "/scratchdata/nyu_plane"
    data_csv = "/HighResMDE/get_planes/ransac/config/nyu.csv"

    with open(data_csv, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)

    data = data[0]

    depth = Image.open(os.path.join(root, data[1]))
    depth = np.array(depth) /float(data[6])

    INTRINSICS = [float(data[2]), 0, float(data[4]), 0, 0, float(data[3]), float(data[5]), 0] # fx, fy, cx, cy

    pt, _ = depth_to_pcd(depth, INTRINSICS)
    H, W = depth.shape
    pt = pt.reshape(H,W, 3)
    normal = Depth2Normal(pt, 1, None)

    plt.imsave("normal.png", (normal+1)/2)