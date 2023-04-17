import torch
import torch.nn as nn
import torch.nn.functional as F
from model.image_encoder import CamEncoder
from model.network_blocks import BaseConv
from model.transformer import Transformer, MLP, inverse_sigmoid
from model.matcher import HungarianMatcher
from model.loss import sigmoid_focal_loss
from geometry.nms3d import nms3d, nms_distance
import copy
import math

def sin_positional_encoding3D(size, device, num_feats=128, offset=0.0, temperature=10000):
    #size should be [B, N, H, W]
    scale = 2 * math.pi
    eps = 1e-6
    mask = torch.ones(size, dtype=torch.float32, device=device)
    n_embed = mask.cumsum(1, dtype=torch.float32)
    y_embed = mask.cumsum(2, dtype=torch.float32)
    x_embed = mask.cumsum(3, dtype=torch.float32)
    n_embed = (n_embed + offset) / (n_embed[:, -1:, :, :] + eps) * scale
    y_embed = (y_embed + offset) / (y_embed[:, :, -1:, :] + eps) * scale
    x_embed = (x_embed + offset) / (x_embed[:, :, :, -1:] + eps) * scale
    dim_t = torch.arange(num_feats, dtype=torch.float32, device=device)
    dim_t = temperature**(2 * (dim_t // 2) / num_feats)
    pos_n = n_embed[:, :, :, :, None] / dim_t
    pos_x = x_embed[:, :, :, :, None] / dim_t
    pos_y = y_embed[:, :, :, :, None] / dim_t
    B, N, H, W = mask.size()
    pos_n = torch.stack((pos_n[:, :, :, :, 0::2].sin(), pos_n[:, :, :, :, 1::2].cos()), dim=4).view(B, N, H, W, -1)
    pos_x = torch.stack((pos_x[:, :, :, :, 0::2].sin(), pos_x[:, :, :, :, 1::2].cos()), dim=4).view(B, N, H, W, -1)
    pos_y = torch.stack((pos_y[:, :, :, :, 0::2].sin(), pos_y[:, :, :, :, 1::2].cos()), dim=4).view(B, N, H, W, -1)
    pos = torch.cat((pos_n, pos_y, pos_x), dim=4)
    return pos

class PETR(nn.Module):
    def __init__(self, grid_conf, cam_num, cam_C, D, cls_num, backbone='yolo-m', bev_aug=False, w_cls=1, w_box=1, aux_loss=False):
        super(PETR, self).__init__()
        self.name = 'petr'
        self.img_encoder = CamEncoder(cam_C, backbone=backbone)
        self.N = cam_num
        self.cls_num = cls_num
        self.D = D
        self.xbound = grid_conf['xbound']
        self.ybound = grid_conf['ybound']
        self.zbound = grid_conf['zbound'] 
        self.bev_aug = bev_aug

        self.feat_conv = nn.Conv2d(cam_C, 256, 1, 1)
        # self.feat_conv = nn.Conv2d(cam_C, 256, 1, 1)
        self.depth_conv = nn.Sequential(
            nn.Conv2d(cam_C, 256, 1, 1),
            nn.ReLU(),
            nn.Conv2d(256, self.D, 1, 1),
        )
        self.pos_conv = nn.Sequential(
            nn.Conv2d(self.D * 3, 1024, 1, 1),
            nn.ReLU(),
            nn.Conv2d(1024, 256, 1, 1),
        )
        self.adapt_pos2d = nn.Sequential(
            nn.Conv2d(384, 1024, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0),
        )
        self.num_decoder_layers = 6
        self.transformer = Transformer(num_decoder_layers=self.num_decoder_layers)
        # self.class_embed = nn.Linear(256, cls_num+1) # include background
        # self.bbox_embed = MLP(256, 256, 8, 3)
        self.class_embed = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, cls_num)
        )
        self.bbox_embed = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 8)
        )

        self.aux_loss = aux_loss
        self.init_parameters()
        self.w_cls = w_cls
        self.w_box = w_box
        self.matcher = HungarianMatcher()
        empty_weight = torch.ones(self.cls_num + 1)
        empty_weight[-1] = 0.1 # empty weight
        self.register_buffer('empty_weight', empty_weight)

    def init_parameters(self):
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed[-1].bias.data = torch.ones(self.cls_num) * bias_value
        nn.init.constant_(self.bbox_embed[-1].bias.data, 0)
        if self.aux_loss:
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(self.num_decoder_layers)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(self.num_decoder_layers)])

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.img_encoder.trunk.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval() 

    def forward(self, input, target=None):
        B,N,C,H_raw,W_raw = input['image'].size()
        x = self.img_encoder(input['image'].view(-1, C, H_raw, W_raw))
        BN, C, H, W = x.size()

        feat = self.feat_conv(x)
        feat2d = feat.permute(0,2,3,1) # BN x H x W x C
        feat2d = feat2d.reshape(B, -1, 256)

        depth_dist = self.depth_conv(x).sigmoid()

        intrins = input['intrins'].clone()
        intrins[:, :, 0, :] *= ( W / W_raw)
        intrins[:, :, 1, :] *= ( H / H_raw)
        img_pos = self.make_coordinate(W, H, self.D, intrins, input['rots'], input['trans'])
        if self.bev_aug and self.training:
            bev_rot = input['bev_rot'].view(B, N, 1, 1, 1, 3, 3)
            img_pos = bev_rot.matmul(img_pos.unsqueeze(-1)).squeeze(-1)

        img_pos_norm = self.norm_pos(img_pos) # BxNxDxHxWx3
        # img_pos_norm = inverse_sigmoid(img_pos_norm)
        img_pos_norm = img_pos_norm.permute(0,1,2,5,3,4).contiguous().view(BN,-1,H,W)        
        pos_emb = self.pos_conv(img_pos_norm)
        pos_emb = pos_emb * depth_dist
        pos3d_emb = pos_emb.permute(0,2,3,1) # BN x H x W x C
        pos3d_emb = pos3d_emb.reshape(B, -1, 256)

        pos2d_feat = sin_positional_encoding3D((B,N,H,W), x.device) # B x N x H x W x 384
        pos2d_feat = pos2d_feat.permute(0, 1, 4, 2, 3).contiguous().view(BN,-1,H,W)
        pos2d_feat_emb = self.adapt_pos2d(pos2d_feat)
        pos2d_feat_emb = pos2d_feat_emb.permute(0,2,3,1)
        pos2d_feat_emb = pos2d_feat_emb.reshape(B, -1, 256)
        # pos3d = pos3d_emb + pos2d_feat_emb

        outputs, reference_points = self.transformer(feat2d, posemb_3d=pos3d_emb, posemb_2d=pos2d_feat_emb)
        loss_aux = 0
        if self.aux_loss:
            if target is not None:
                for i in range(self.num_decoder_layers - 1):
                    outputs_class = self.class_embed[i](outputs[i])
                    outputs_coord = self.bbox_embed[i](outputs[i])
                    output_boxes = self.decode_boxes(outputs_coord, reference_points[i])
                    pred = {'pred_logits': outputs_class.float(), 'pred_boxes': output_boxes.clone().float()}
                    with torch.cuda.amp.autocast(enabled=False):
                        loss_aux_i, _ = self.get_loss(pred, target.float())
                    loss_aux = loss_aux + loss_aux_i
            outputs_class = self.class_embed[-1](outputs[-1])
            outputs_coord = self.bbox_embed[-1](outputs[-1])
        else:
            outputs_class = self.class_embed(outputs[-1])
            outputs_coord = self.bbox_embed(outputs[-1])
        # decode box
        output_boxes = self.decode_boxes(outputs_coord, reference_points[-1])
        pred = {'pred_logits': outputs_class.float(), 'pred_boxes': output_boxes.float()}
        if target is None:
            return pred
        else:
            with torch.cuda.amp.autocast(enabled=False):
                loss, loss_tags = self.get_loss(pred, target.float())
            if self.aux_loss:
                # loss_aux *= 0.2
                loss += loss_aux
                loss_tags['aux_loss'] = loss_aux.clone().detach()
            return loss, loss_tags

    def decode_boxes(self, outputs_coord, reference_points):
        ref_p = inverse_sigmoid(reference_points)
        output_xyz = (outputs_coord[..., 0:3] + ref_p).sigmoid()
        output_x = output_xyz[..., 0] * (self.xbound[1] - self.xbound[0]) + self.xbound[0]
        output_y = output_xyz[..., 1] * (self.ybound[1] - self.ybound[0]) + self.ybound[0]
        output_z = output_xyz[..., 2] * (self.zbound[1] - self.zbound[0]) + self.zbound[0]
        output_xyz = torch.stack([output_x, output_y, output_z], dim=-1)
        output_wlh = outputs_coord[..., 3:6]
        output_ang = outputs_coord[..., 6:8]
        # output_ang_norm = outputs_coord[..., 6:8].norm(dim=2, keepdim=True).clamp(min=1e-6)
        # output_ang = outputs_coord[..., 6:8] / output_ang_norm
        output_box = torch.cat([output_xyz, output_wlh, output_ang], dim=2)
        return  output_box

    def get_loss(self, pred, target):
        label_list = []
        for i in range(target.shape[0]):
            tgt = target[i]
            mask = tgt[:, 4:7].sum(dim=1) > 0 # size > 0, mask empty gt
            masked_tgt = tgt[mask, :].clone()
            tgt_box = self.make_target_box(masked_tgt[:, 1:].clone())
            label = {'labels':masked_tgt[:, 0].long(), 'boxes':tgt_box}
            label_list.append(label)
        indices, cost_tags = self.matcher(pred, label_list)

        num_boxes = sum(len(t["labels"]) for t in label_list)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(pred.values())).device)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / torch.distributed.get_world_size(), min=1).item()

        lcls = self.loss_labels(pred, label_list, indices, num_boxes) * self.w_cls 
        lbox = self.loss_boxes(pred, label_list, indices, num_boxes) * self.w_box
        loss = lcls + lbox
        loss_tags = {
            'cls': lcls.clone().detach(),
            'box': lbox.clone().detach(),
            'loss': loss.clone().detach()
        }
        loss_tags.update(cost_tags)
        return loss, loss_tags

    def make_target_box(self, tgt):
        xyz = tgt[:, 0:3]
        wlh = tgt[:, 3:6].log()
        ang = tgt[:, 6:8]
        boxl1_tgt = torch.cat([xyz, wlh, ang], dim=-1)
        return boxl1_tgt

    def loss_labels(self, outputs, targets, indices, num_boxes):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.cls_num,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=0.25, gamma=2)
        # loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        return loss_ce

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        loss_bbox = loss_bbox.sum() / num_boxes
        return loss_bbox

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    @torch.no_grad()
    def norm_pos(self, pos):
        x = (pos[..., 0] - self.xbound[0]) / (self.xbound[1] - self.xbound[0])
        y = (pos[..., 1] - self.ybound[0]) / (self.ybound[1] - self.ybound[0])
        z = (pos[..., 2] - self.zbound[0]) / (self.zbound[1] - self.zbound[0])
        norm_pos = torch.stack([x,y,z], dim=-1)
        norm_pos = norm_pos.clamp(min=0.0, max=1.0)
        return norm_pos

    @torch.no_grad()
    def make_coordinate(self, W, H, D, intrins, rots, trans):
        frustum = self.create_frustum(W, H, D)
        frustum = frustum.to(intrins.device)
        B, N, _ = trans.shape
        points = frustum.view(1, 1, D, H, W, 3).repeat(B, N, 1, 1, 1, 1)
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1)).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)
        return points

    @torch.no_grad()
    def create_frustum(self, W, H, D):
        ys, xs = torch.meshgrid([torch.arange(H), torch.arange(W)])
        ones = torch.ones((H, W))
        ds = torch.arange(D, dtype=torch.float).view(-1, 1, 1, 1) + 0.5
        xys = torch.stack([xs+0.5, ys+0.5, ones], dim=-1).view(1, H, W, 3)
        # D x H x W x 3
        frustum = xys * ds
        return frustum

    @torch.no_grad()
    def decode(self, pred, thresh=0.3):
        result_list = []
        for i in range(pred['pred_logits'].size(0)):
            print(pred['pred_logits'])
        #pred = {'pred_logits': outputs_class.float(), 'pred_boxes': outputs_coord.float()}
            cls_prob, cls_idx = torch.topk(pred['pred_logits'][i].sigmoid(), k=1, dim=1)
            print(cls_prob.max())
            mask = cls_prob.squeeze() > thresh
            pred_cls = cls_idx[mask]
            pred_prob = cls_prob[mask]
            boxes = pred['pred_boxes'][i]
            boxes_xyz = boxes[:, 0:3]
            boxes_wlh = boxes[:, 3:6].exp()
            boxes_ang = boxes[..., 6:8]
            boxes_dec = torch.cat([boxes_xyz, boxes_wlh, boxes_ang], dim=-1).clone()
            pred_box = boxes_dec[mask, :].clone()
            result = torch.cat([pred_cls.float(), pred_prob, pred_box], dim=1).detach().cpu()
            result_list.append(result)
        return result_list
