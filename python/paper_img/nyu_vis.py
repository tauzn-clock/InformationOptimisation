import sys
import os 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import open3d as o3d
import csv
from utils.process_depth import get_3d
from utils.visualise import mask_over_img, img_over_pcd, mask_to_hsv
from metric import plane_ordering


def pcd_to_img(pcd):
    # Visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Add the point cloud to the visualizer
    vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.point_size = 2

    view_control = vis.get_view_control()
    view_control.set_lookat([0, 0, 0])
    view_control.set_zoom(0.6) 
    view_control.rotate(100.0, 100.0)

    return np.array(vis.capture_screen_float_buffer(True))
def shrink_pcd_img(ori,gt,pred,SAVE_DIR,INDEX):
    def get_bounding_box(mask):
        left = mask.shape[1]
        right = 0
        top = mask.shape[0]
        bottom = 0

        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if np.sum(mask[i,j]) < 3:
                    left = min(left, j)
                    right = max(right, j)
                    top = min(top, i)
                    bottom = max(bottom, i)
        
        return left, right, top, bottom

    left_ori, right_ori, top_ori, bottom_ori = get_bounding_box(ori)
    left_gt, right_gt, top_gt, bottom_gt = get_bounding_box(gt)
    left_pred, right_pred, top_pred, bottom_pred = get_bounding_box(pred)

    left = min(left_ori, left_gt, left_pred)
    right = max(right_ori, right_gt, right_pred)
    top = min(top_ori, top_gt, top_pred)
    bottom = max(bottom_ori, bottom_gt, bottom_pred)

    print(left, right, top, bottom)

    plt.imsave(f"{SAVE_DIR}/{INDEX}_pcd_ori.png", ori[top:bottom, left:right])
    plt.imsave(f"{SAVE_DIR}/{INDEX}_pcd_gt.png", gt[top:bottom, left:right])
    plt.imsave(f"{SAVE_DIR}/{INDEX}_pcd_pred.png", pred[top:bottom, left:right])

def csv_to_depth(mask, csv, INTRINSICS):
    H, W = mask.shape

    POINTS = get_3d(np.ones_like(mask), INTRINSICS)
    DIRECTION_VECTOR = POINTS / (np.linalg.norm(POINTS, axis=1)[:, None]+1e-7)

    mask = mask.flatten()

    coord = np.zeros((H*W, 3))
    
    for i in range(1, mask.max()+1):
        norm = csv[i-1,:3]
        d = csv[i-1,3]
        plane_coord = (-d/(np.dot(DIRECTION_VECTOR, norm.T)+1e-7))[:,None]*DIRECTION_VECTOR
        coord[mask==i] = plane_coord[mask==i]

    return coord

def find_distance_for_gt_planes(coord, csv, mask):
    mask = mask.flatten()

    new_csv = []

    for i in range(1,mask.max()+1):
        norm = csv[i-1,:3]
        norm = norm / np.linalg.norm(norm)
        valid_mask = np.zeros_like(mask)
        valid_mask[mask==i] = 1
        valid_mask[coord[:,2] == 0] = 0
        dist = np.dot(coord[valid_mask==True], norm.T)
        new_csv.append([norm[0], norm[1], norm[2], -dist.mean()])

    return np.array(new_csv)

def transform(pcd):
    # Flip the point cloud
    tf = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    pcd.transform(tf)
    # Put all transformations here
    return pcd

def get_mask_img(mask):
    color = np.zeros((mask.shape[0], mask.shape[1], 3))
    for i in range(1, mask.max()+1):
        color[mask==i] = hsv_to_rgb((i-1)/mask.max()*360, 1, 1)
    return color

if __name__ == "__main__":

    DATA_DIR = "/scratchdata/nyu_plane"
    SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
    FOLDER = "mask_proportional"

    INDEX = 1207

    # Open yaml
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "nyu.yaml"), "r") as file:
        config = yaml.safe_load(file)

    INTRINSICS = [config["camera_params"]["fx"], config["camera_params"]["fy"], config["camera_params"]["cx"], config["camera_params"]["cy"]]
    R = config["depth_max"]  # Maximum range of sensor
    EPSILON = config["resolution"]  # Resolution of the sensor

    rgb = np.array(Image.open(f"{DATA_DIR}/rgb/{INDEX}.png"))
    depth = np.array(Image.open(f"{DATA_DIR}/depth/{INDEX}.png")) * EPSILON
    coord = get_3d(depth, INTRINSICS)
    gt_mask = np.array(Image.open(f"{DATA_DIR}/original_gt/{INDEX}.png"))
    pred_mask = np.array(Image.open(f"{DATA_DIR}/{FOLDER}/{INDEX}.png"))

    with open(f"{DATA_DIR}/original_gt/{INDEX}.csv", 'r') as f:
        reader = csv.reader(f)
        gt_csv = np.array(list(reader), dtype=np.float32)
    with open(f"{DATA_DIR}/{FOLDER}/{INDEX}.csv", 'r') as f:
        reader = csv.reader(f)
        pred_csv = np.array(list(reader), dtype=np.float32)

    points = get_3d(depth, INTRINSICS)
    SIGMA = 0.01 * points[:,2]
    print(pred_mask.max(),len(pred_csv))
    pred_mask, pred_csv = plane_ordering(points, pred_mask.flatten(), pred_csv, R, EPSILON, SIGMA,keep_index=16, merge_planes=True)
    pred_mask = pred_mask.reshape(depth.shape)
    print(pred_mask.max(),len(pred_csv))

    #Original RGB
    plt.imsave(f"{SAVE_DIR}/{INDEX}_rgb.png", rgb[45:471, 41:601])

    #GT Mask overlay
    mask_over_img(rgb[45:471, 41:601], gt_mask[45:471, 41:601], f"{SAVE_DIR}/{INDEX}_gt_mask.png")

    #Pred Mask overlay
    mask_over_img(rgb[45:471, 41:601], pred_mask[45:471, 41:601], f"{SAVE_DIR}/{INDEX}_pred_mask.png")

    #Original PCD
    coord = get_3d(depth, INTRINSICS)
    pcd = img_over_pcd(coord, rgb)
    pcd_ori = transform(pcd)
    ori_pcd_img = pcd_to_img(pcd_ori)

    #GT PCD
    coord = get_3d(depth, INTRINSICS)
    full_gt_csv = find_distance_for_gt_planes(coord, gt_csv, gt_mask)
    coord = csv_to_depth(gt_mask, full_gt_csv, INTRINSICS)
    pcd = img_over_pcd(coord, rgb)
    pcd_gt = transform(pcd)
    gt_pcd_img = pcd_to_img(pcd_gt)

    #Pred PCD
    coord = get_3d(depth, INTRINSICS)
    pcd = img_over_pcd(coord, rgb)
    pcd_pred = transform(pcd)
    pred_pcd_img = pcd_to_img(pcd_pred)

    shrink_pcd_img(ori_pcd_img,gt_pcd_img,pred_pcd_img,SAVE_DIR, INDEX)
