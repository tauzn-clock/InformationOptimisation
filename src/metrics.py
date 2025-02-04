
import numpy as np
import torch
from PIL import Image
import os
import csv
from visualise_mask import merge_mask

def reassign_mask(mask, gt):
    new_mask = mask.copy()
    for i in range(1, mask.max()+1):
        best_intersection = 0
        best_j = 0
        for j in range(1, gt.max()+1):
            intersection = np.sum((mask==i)&(gt==j))
            if intersection > best_intersection:
                best_intersection = intersection
                best_j = j
        new_mask[mask==best_j] = i
        new_mask[mask==i] = best_j
        
    return new_mask

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

    avg = []

    for index in range(1449):
        depth = Image.open(f"{ROOT}/depth/{index}.png")
        pred = Image.open(f"{ROOT}/new_gt/{index}.png")
        gt = Image.open(f"{ROOT}/original_gt/{index}.png")
        with open(f"{ROOT}/new_gt/{index}.csv", 'r') as f:
            reader = csv.reader(f)
            csv_data = list(reader)
        csv_data = np.array(csv_data, dtype=np.float32)

        depth = np.array(depth)
        pred = np.array(pred)
        pred, _ = merge_mask(pred, csv_data)
        gt = np.array(gt)

        valid_mask = gt > 0
        pred = pred[valid_mask]
        gt = gt[valid_mask]

        #pred = reassign_mask(pred, gt)

        pred = pred.reshape(-1, 1)
        gt = gt.reshape(-1, 1)
        
        avg.append(evaluateMasks(pred, gt, torch.device("cpu")))
        print(index)
    
    avg = np.array(avg)
    print(avg.mean(0))
        