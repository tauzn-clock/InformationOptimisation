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

DEVICE="cuda:0"
ROOT = "/scratchdata/processed/hgx"

sam = sam_model_registry["vit_b"](checkpoint="/scratchdata/sam_vit_b_01ec64.pth").to(DEVICE)

for frame_cnt in tqdm(range(107)):
    #print("Image", i)
    data = [f"rgb/{frame_cnt}.png", f"depth/{frame_cnt}.png", 306.75604248046875, 306.7660827636719, 322.9314270019531, 203.91506958007812, 1, 2**16]

    INTRINSICS = [float(data[2]), 0, float(data[4]), 0, 0, float(data[3]), float(data[5]), 0] # fx, fy, cx, cy
    INTRINSICS = np.array(INTRINSICS)

    img = Image.open(os.path.join(ROOT, data[0]))
    img = np.array(img)

    depth = Image.open(os.path.join(ROOT, data[1]))
    depth = np.array(depth) /float(data[6])

    H, W = depth.shape

    points, index = depth_to_pcd(depth, INTRINSICS)

    EPSILON = 1/float(data[6]) # Resolution
    R = float(data[7]) # Maximum Range
    SIGMA = EPSILON * 5 # Normal std
    SIGMA = 0.02 * points[:,2]

    CONFIDENCE = 0.999
    INLIER_THRESHOLD = 1e5/(H*W)
    MAX_PLANE = 3

    x, y = np.meshgrid(np.arange(W), np.arange(H))
    x = x.flatten()
    y = y.flatten()
    z = np.ones_like(x)
    fx, fy, cx, cy = INTRINSICS[0], INTRINSICS[5], INTRINSICS[2], INTRINSICS[6]
    x_3d = (x - cx) / fx
    y_3d = (y - cy) / fy
    direction_vector = np.vstack((x_3d, y_3d, z)).T
    direction_vector = direction_vector / (np.linalg.norm(direction_vector, axis=1)[:, None]+1e-7)

    mask_generator = SamAutomaticMaskGenerator(sam, stability_score_thresh=0.95)

    masks = mask_generator.generate(img)

    #print("Number of masks:", len(masks))

    global_mask = np.zeros((H, W)).flatten()
    total_planes = 0

    # Sort masks by stability score

    masks = sorted(masks, key=lambda x: x["stability_score"])

    """
    visualise_mask = np.zeros((H, W))
    for i, sam_mask in enumerate(masks):
        visualise_mask += sam_mask["segmentation"] * (i+1)
    plt.imsave("visualise_mask.png", visualise_mask)
    """

    new_depth = depth.copy()
    for sam_i, sam_mask in enumerate(masks):
        valid_mask = sam_mask["segmentation"] & (depth > 0)
        #valid_mask = depth > 0

        information, mask, plane = plane_ransac(depth, INTRINSICS, R, EPSILON, SIGMA, CONFIDENCE, INLIER_THRESHOLD, MAX_PLANE, valid_mask.flatten())

        min_idx = np.argmin(information)

        if min_idx==1:
            dist = points @ plane[1:min_idx+1,:3].T + np.stack([plane[1:min_idx+1,3]]*points.shape[0], axis=0)
            dist = np.abs(dist)
            isPartofPlane = mask != 0
            mask = np.argmin(dist, axis=1) + 1
            mask = mask * isPartofPlane

            best_mask = np.zeros_like(global_mask)
            for i in range(1, min_idx+1):
                best_mask[mask[i]==i] = i + total_planes
                pred_depth = (-plane[i,3]/(np.dot(direction_vector, plane[i,:3].T)+1e-7))*direction_vector[:,2]
                pred_depth = pred_depth.reshape(H, W)
                pred_depth *= (sam_mask["segmentation"]& (depth == 0))
                if pred_depth.max() > R or pred_depth.min() < 0: break
                new_depth += pred_depth
        
            global_mask += best_mask
            total_planes += min_idx

    print(depth.max())
    print(new_depth.max())
    # Round new depth to int
    new_depth = np.round(new_depth * float(data[6])).astype(np.uint16)
    print(new_depth.max())
    # Save with Image
    Image.fromarray(new_depth).save(os.path.join(ROOT,"repair",f"{frame_cnt}.png"))
