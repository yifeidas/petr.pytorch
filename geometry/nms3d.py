import torch
from pytorch3d.ops.iou_box3d import box3d_overlap
from geometry.boxes3d import GenericBoxes3D

def box3d_iou(box1, box2):
    b1 = GenericBoxes3D(box1[:, :4], box1[:, 4:7], box1[:, 7:10])
    corners1 = b1.corners.clone()
    b2 = GenericBoxes3D(box2[:, :4], box2[:, 4:7], box2[:, 7:10])
    corners2 = b2.corners.clone()
    vol, iou =  box3d_overlap(corners1, corners2)
    return iou

def nms3d(box3d_tensor, nms_thresh=0.5):
    # box3d_tensor: Nx12 [[cls, prob, q0,q1,q2,q3, x,y,z, w,l,h]]
    N = box3d_tensor.size(0)
    if N <= 1:
        return box3d_tensor
    else:
        mask = torch.ones(N).bool()
        p_obj = box3d_tensor[:, 1]
        p_obj, idx = p_obj.sort(descending=True)
        box3d_sorted = box3d_tensor.index_select(0, idx).clone()
        box_pred = box3d_sorted[:, 2:].clone() 
        for i in range(N-1):
            if mask[i] > 0:
                iou = box3d_iou(box_pred[i, :].unsqueeze(0), box_pred[(i+1):, :])
                iou = iou.squeeze(0)
                mask_box = iou > nms_thresh
                need_mask = mask[(i+1):]
                need_mask[mask_box] = 0
                mask[(i+1):] = need_mask
        return box3d_sorted[mask, :].clone()

def box3d_distance(box1, box2):
    #box1: Nx10  box2: Mx10
    # return NxM
    xyz1 = box1[:, 4:7].unsqueeze(1)
    xyz2 = box2[:, 4:7].unsqueeze(0)
    dis1 = (xyz1 - xyz2) **2
    # dis2 = dis1.sum(dim=2)
    dis2 = dis1[:,:,:2].sum(dim=2)
    return torch.sqrt(dis2)

def nms_distance(box3d_tensor, distance_thresh=0.5):
    # box3d_tensor: Nx12 [[cls, prob, q0,q1,q2,q3, x,y,z, w,l,h]]
    N = box3d_tensor.size(0)
    if N <= 1:
        return box3d_tensor
    else:
        mask = torch.ones(N).bool()
        p_obj = box3d_tensor[:, 1]
        p_obj, idx = p_obj.sort(descending=True)
        box3d_sorted = box3d_tensor.index_select(0, idx).clone()
        box_pred = box3d_sorted[:, 2:].clone() 
        for i in range(N-1):
            if mask[i] > 0:
                distance = box3d_distance(box_pred[i, :].unsqueeze(0), box_pred[(i+1):, :])
                distance = distance.squeeze(0)
                mask_box = distance < distance_thresh
                need_mask = mask[(i+1):]
                need_mask[mask_box] = 0
                mask[(i+1):] = need_mask
        return box3d_sorted[mask, :].clone()