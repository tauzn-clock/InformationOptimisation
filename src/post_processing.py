from depth_to_normal import Depth2Normal
from process_depth_img import depth_to_pcd

import numpy as np
import matplotlib.pyplot as plt

def post_processing(depth, INTRINSICS, R, EPSILON, SIGMA, information, mask, plane, valid_mask=None):

    information_mask = information == np.inf
    information = information[~information_mask]
    plane = plane[~information_mask,:]

    H,W = depth.shape

    pts, _ = depth_to_pcd(depth, INTRINSICS)
    direction_vector = pts / (np.linalg.norm(pts, axis=1, keepdims=True)+1e-7)

    pts_normal = Depth2Normal(pts.reshape(H,W,3), 1, 0.2)
        
    #plt.imsave("normal.png", (pts_normal+1)/2)

    # Calculate Information

    SPACE_STATES = np.log(R/EPSILON)
    PER_POINT_INFO = np.log(SIGMA/EPSILON+1e-7) 
    PER_POINT_INFO += 0.5 * np.log(2*np.pi) - SPACE_STATES
    TWO_SIGMA_SQUARE = 2 * (SIGMA**2 + 1e-7)

    distance = plane[1:,3]
    normal = plane[1:,:3]

    Z = depth.flatten()
    error = ((-distance/(np.dot(direction_vector, normal.T)+1e-7))*direction_vector[:,2, None] - Z[:,None]) ** 2
    error = error / TWO_SIGMA_SQUARE[:,None] + PER_POINT_INFO[:,None]   

    new_mask = np.argmin(error, axis=1) + 1
    new_mask = new_mask * (mask > 0)

    pts_normal = pts_normal.reshape(H*W, 3)
    normal_error = np.abs(np.dot(pts_normal, normal.T))
    normal_error[error > 0] = - np.inf
    new_mask = np.argmax(normal_error, axis=1) + 1
    new_mask = new_mask * (mask > 0)

    if valid_mask is not None: TOTAL_NO_PTS = valid_mask.sum()
    else: TOTAL_NO_PTS = H * W

    new_information = np.zeros_like(information)
    new_information[0] = information[0]
    for plane_cnt in range(1, len(plane)):
        new_information[plane_cnt] =  new_information[plane_cnt-1]
        new_information[plane_cnt] -= TOTAL_NO_PTS * np.log(plane_cnt) # Remove previous mask 
        new_information[plane_cnt] += TOTAL_NO_PTS * np.log(plane_cnt+1) # New Mask that classify points
        new_information[plane_cnt] += 3 * SPACE_STATES # New Plane

        new_information[plane_cnt] += error[new_mask == plane_cnt,plane_cnt-1].sum()

    # Visualise
    min_idx = np.argmin(new_information)
    
    tmp_mask = new_mask.reshape(H, W)
    tmp_mask[tmp_mask > min_idx] = 0
    #plt.imsave("mask.png", tmp_mask)
    """
    tmp_normal = np.zeros_like(normal)
    for i in range(1, min_idx+1):
        print((tmp_mask==i).shape)
        tmp_normal[tmp_mask == i,:] = plane[i][:3]
    plt.imsave("plane.png", (tmp_normal+1)/2)
    """

    return new_information, new_mask, plane