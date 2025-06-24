
import numpy as np
import torch
from PIL import Image
import os
import csv

def plane_ordering(POINTS, mask, param, R, EPSILON, SIGMA, keep_index=10000, merge_planes=False):
    def remove_mask_with_zero_area(mask, param):
        new_mask = np.zeros_like(mask)
        new_param = []
        for i in range(1, mask.max()+1):
            if np.sum(mask==i) > 0:
                new_mask[mask==i] = len(new_param)+1
                new_param.append(param[i-1])
        return new_mask, np.array(new_param)

    mask, param = remove_mask_with_zero_area(mask, param)

    SPACE_STATES = np.log(R/EPSILON)
    PER_POINT_INFO = np.log(SIGMA+1e-16) - np.log(EPSILON) + 0.5 * np.log(2*np.pi) - SPACE_STATES
    TWO_SIGMA_SQUARE = 2 * SIGMA**2

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

    #index = np.argsort(store[:,0])[::-1]
    index = np.argsort(store[:,1])
    #index = np.argsort(store[:,2])

    for i in range(len(index)):
        new_mask[mask==index[i]+1] = i+1
        new_param[i] = param[index[i]]

    if merge_planes:
        for i in range(len(new_param)-1, -1, -1):
            masked_direction_vector = direction_vector[mask==i+1]
            masked_z = POINTS[mask==i+1][:,2]

            for j in range(i):
                norm = new_param[j,:3]
                d = new_param[j,3]

                error = ((-d/(np.dot(masked_direction_vector, norm.T)+1e-7))*masked_direction_vector[:,2] - masked_z) ** 2
                error = error / TWO_SIGMA_SQUARE[mask==i+1] + PER_POINT_INFO[mask==i+1]

                if error.sum() < 0:
                    new_mask[mask==i+1] = j+1
                    break

        new_mask, new_param = plane_ordering(POINTS, new_mask, new_param, R, EPSILON, SIGMA, keep_index=keep_index, merge_planes=False)
    
    new_mask[new_mask > keep_index] = 0
    new_param = new_param[:keep_index]

    return new_mask, new_param

def evaluateMasks(predSegmentations, gtSegmentations, device, printInfo=False):
    """
    :param predSegmentations:
    :param gtSegmentations:
    :param device:
    :param pred_non_plane_idx:
    :param gt_non_plane_idx:
    :param printInfo:
    :return:
    """
    predSegmentations = torch.from_numpy(predSegmentations).to(device) # (h, w)
    gtSegmentations = torch.from_numpy(gtSegmentations).to(device) # (h, w)

    pred_masks = []
    for i in range(1,predSegmentations.max() + 1):
        mask_i = predSegmentations == i
        mask_i = mask_i.float()
        if mask_i.sum() > 0:
            pred_masks.append(mask_i)
    predMasks = torch.stack(pred_masks, dim=0)

    gt_masks = []
    for i in range(1,gtSegmentations.max() + 1):
        mask_i = gtSegmentations == i
        mask_i = mask_i.float()
        if mask_i.sum() > 0:
            gt_masks.append(mask_i)
    gtMasks = torch.stack(gt_masks, dim=0)

    valid_mask = (gtMasks.max(0)[0]).unsqueeze(0)

    gtMasks = torch.cat([gtMasks, torch.clamp(1 - gtMasks.sum(0, keepdim=True), min=0)], dim=0)  # M+1, H, W
    predMasks = torch.cat([predMasks, torch.clamp(1 - predMasks.sum(0, keepdim=True), min=0)], dim=0)  # N+1, H, W

    intersection = (gtMasks.unsqueeze(1) * predMasks * valid_mask).sum(-1).sum(-1).float()  # torch.Size([M+1, N+1])
    union = (torch.max(gtMasks.unsqueeze(1), predMasks) * valid_mask).sum(-1).sum(-1).float() # torch.Size([M+1, N+1])

    N = intersection.sum()

    RI = 1 - ((intersection.sum(0).pow(2).sum() + intersection.sum(1).pow(2).sum()) / 2 - intersection.pow(2).sum()) / (
            N * (N - 1) / 2)
    joint = intersection / N
    marginal_2 = joint.sum(0) # torch.Size([N+1])
    marginal_1 = joint.sum(1) # torch.Size([M+1])
    H_1 = (-marginal_1 * torch.log2(marginal_1 + (marginal_1 == 0).float())).sum()
    H_2 = (-marginal_2 * torch.log2(marginal_2 + (marginal_2 == 0).float())).sum()

    B = (marginal_1.unsqueeze(-1) * marginal_2) # torch.Size([M+1, N+1])
    log2_quotient = torch.log2(torch.clamp(joint, 1e-8) / torch.clamp(B, 1e-8)) * (torch.min(joint, B) > 1e-8).float() # torch.Size([M+1, N+1])
    MI = (joint * log2_quotient).sum()
    voi = H_1 + H_2 - 2 * MI

    IOU = intersection / torch.clamp(union, min=1) # torch.Size([M+1, N+1])
    SC = ((IOU.max(-1)[0] * torch.clamp((gtMasks * valid_mask).sum(-1).sum(-1), min=1e-4)).sum() / N + (
            IOU.max(0)[0] * torch.clamp((predMasks * valid_mask).sum(-1).sum(-1), min=1e-4)).sum() / N) / 2
    info = [RI.item(), voi.item(), SC.item()]
    if printInfo:
        print('mask statistics', info)
        pass
    return info

if __name__ == "__main__":
    ROOT = "/scratchdata/nyu_plane"
    FOLDER = "new_gt_sigma_1"
    SIGMA_RATIO = 0.01
    data_csv = "/HighResMDE/get_planes/ransac/config/nyu.csv"

    with open(data_csv, 'r') as f:
        reader = csv.reader(f)
        DATA = list(reader)

    avg = []

    for frame_cnt in range(0,len(DATA)):
        data = DATA[frame_cnt]
        INTRINSICS = np.array([float(data[2]), 0, float(data[4]), 0, 0, float(data[3]), float(data[5]), 0]) # fx, fy, cx, cy

        EPSILON = 1/float(data[6]) # Resolution
        R = float(data[7]) # Maximum Range

        depth = np.array(Image.open(os.path.join(ROOT, data[1])))/float(data[6])
        H, W = depth.shape

        points, index = depth_to_pcd(depth, INTRINSICS)
        SIGMA = SIGMA_RATIO * points[:,2]

        pred = np.array(Image.open(f"{ROOT}/{FOLDER}/{frame_cnt}.png")).flatten()
        gt = np.array(Image.open(f"{ROOT}/original_gt/{frame_cnt}.png")).flatten()
        with open(f"{ROOT}/{FOLDER}/{frame_cnt}.csv", 'r') as f:
            reader = csv.reader(f)
            csv_data = np.array(list(reader), dtype=np.float32)

        pred, csv_data = plane_ordering(points, pred, csv_data, R, EPSILON, SIGMA,keep_index=gt.max(),merge_planes=True)

        valid_mask = depth[45:471, 41:601].flatten() > 0
        H,W = depth.shape
        pred = pred.reshape(H,W)[45:471, 41:601].flatten()
        pred = pred[valid_mask]
        gt = gt.reshape(H,W)[45:471, 41:601].flatten()
        gt = gt[valid_mask]

        pred = pred.reshape(-1, 1)
        gt = gt.reshape(-1, 1)

        avg.append(evaluateMasks(pred, gt, torch.device("cpu")))
    
    avg = np.array(avg)
    print(avg.mean(0))

    #Get index of smallest in each column
    min_idx = avg.argmin(axis=0)
    max_idx = avg.argmax(axis=0)

    print(min_idx)
    print(max_idx)
    
    print(avg[min_idx[0]])
    print(avg[max_idx[1]])
    print(avg[min_idx[2]])

    print("Smallest RI: ", min_idx[0])
    print("Largest VOI: ", max_idx[1])
    print("Smallest SC: ", min_idx[2])