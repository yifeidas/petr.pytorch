import torch
from torch import nn
from torch.nn import functional as F 
import numpy as np
from geometry.boxes3d import GenericBoxes3D

class FocalLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.reduction = reduction

    def focal_loss(self, x, y, eps=1e-9):
        alpha = 0.25
        gamma = 2.0
        p = x.sigmoid()
        weight = (p - y).pow(gamma)
        alpha_t = alpha * y + (1 - alpha) * (1 - y)
        weight *= alpha_t
        p = p.to(torch.float32)
        eps = 1e-9
        loss = y * (0 - torch.log(p + eps)) + \
               (1.0 - y) * (0 - torch.log(1.0 - p + eps))
        loss *= weight
        num_pos = y.sum().clamp(min=1.0)
        loss = loss.sum()
        if self.reduction == 'mean':
            loss = loss / num_pos
        return loss

    def forward(self, keypoint_pred, keypoint_truth):
        loss = self.focal_loss(keypoint_pred, keypoint_truth)
        return loss

def sigmoid_focal_loss(inputs, targets, num_boxes, alpha=0.25, gamma=2):
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.sum() / num_boxes

class DisentangledBox3DLoss(nn.Module):
    def __init__(self, smooth_l1_loss_beta=0.05, max_loss_per_group=5.0):
        super(DisentangledBox3DLoss, self).__init__()
        self.smooth_l1_loss_beta = smooth_l1_loss_beta
        self.max_loss_per_group = max_loss_per_group
        self.l1loss = nn.SmoothL1Loss(reduction="none", beta=self.smooth_l1_loss_beta)
        
    def forward(self, box3d_pred, box3d_targets):
        box3d_pred = box3d_pred.to(torch.float32)
        box3d_targets = box3d_targets.to(torch.float32)
        target_corners = box3d_targets.corners

        disentangled_losses = {}
        for component_key in ["quat", "tvec", "size"]:
            disentangled_boxes = box3d_targets.clone()
            setattr(disentangled_boxes, component_key, getattr(box3d_pred, component_key))
            pred_corners = disentangled_boxes.to(torch.float32).corners
            loss = self.l1loss(pred_corners, target_corners)
            # Bound the loss
            loss.clamp(max=self.max_loss_per_group)
            loss = torch.sum(loss.reshape(-1, 24).mean(dim=1))
            disentangled_losses["loss_box3d_" + component_key] = loss

        entangled_l1_dist = (target_corners - box3d_pred.corners).detach().abs().reshape(-1, 24).mean(dim=1)
        return disentangled_losses, entangled_l1_dist

def tensor_to_box3d(tensor):
    #input: Nx10
    quat = tensor[:, :4]
    xyz = tensor[:, 4:7]
    wlh = tensor[:, 7:10]
    box3d = GenericBoxes3D(quat, xyz, wlh)
    return box3d

def box3d_entangled_loss(box1, box2):
    M = box1.size(0)
    N = box2.size(0)
    box1_ = box1.unsqueeze(1).repeat(1, N, 1)
    box2_ = box2.unsqueeze(0).repeat(M, 1, 1)
    box3d_1 = tensor_to_box3d(box1_.view(-1, 10))
    box3d_2 = tensor_to_box3d(box2_.view(-1, 10))
    # wl = (box3d_1.size[:, 0] * box3d_1.size[:, 1]).sqrt()
    corners1 = box3d_1.corners
    corners2 = box3d_2.corners
    loss_l1 = (corners1 - corners2).detach().abs().reshape(-1, 24).mean(dim=1) #/ wl
    loss = loss_l1.reshape(M, N)
    return loss