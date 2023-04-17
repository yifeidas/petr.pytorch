import cv2
import math
import torch
import numpy as np 

def sample_to_cuda(data):
    if isinstance(data, str):
        return data
    elif isinstance(data, dict):
        return {key: sample_to_cuda(data[key]) for key in data.keys()}
    elif isinstance(data, list):
        return [sample_to_cuda(val) for val in data]
    else:
        return data.cuda()

def image_to_tensor(image, width, height, cuda=1):
    img_raw1= image[:, :, ::-1]
    img = cv2.resize(img_raw1,(width, height))
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
    if cuda:
        img = img.cuda()
    img = img / 255
    return img

def tensor_to_image(tensor):
    tensor = tensor.cpu().squeeze(0)
    img = tensor.permute(1, 2, 0).contiguous()
    img = img.numpy() * 255
    img = img[:,:,::-1].astype(np.uint8)
    return img

def box_xyxy2xywh(boxes):
    boxes_out = boxes.clone()
    boxes_out[:, 0] = (boxes[:, 0] + boxes[:, 2])/2
    boxes_out[:, 1] = (boxes[:, 1] + boxes[:, 3])/2
    boxes_out[:, 2] = boxes[:, 2] - boxes[:, 0]
    boxes_out[:, 3] = boxes[:, 3] - boxes[:, 1]
    return boxes_out

def box_xywh2xyxy(boxes):
    boxes_out = boxes.clone()
    boxes_out[:, 0] = boxes[:, 0] - boxes[:, 2]/2
    boxes_out[:, 1] = boxes[:, 1] - boxes[:, 3]/2
    boxes_out[:, 2] = boxes[:, 0] + boxes[:, 2]/2
    boxes_out[:, 3] = boxes[:, 1] + boxes[:, 3]/2
    return boxes_out

def merge_six_image_tensors(image_tensor):
    assert image_tensor.size(0) == 6
    img_top = torch.cat([image_tensor[0], image_tensor[1], image_tensor[2]], dim=2)
    img_bot = torch.cat([image_tensor[3], image_tensor[4], image_tensor[5]], dim=2)
    img_merged = torch.cat([img_top, img_bot], dim=1)
    img = tensor_to_image(img_merged)
    return img

def merge_six_images(img_list):
    img_top = np.concatenate(img_list[:3], axis=1)
    img_bot = np.concatenate(img_list[3:], axis=1)
    img_merged = np.concatenate([img_top, img_bot], axis=0)
    return img_merged

def get_rot_matrix(theta):
    rot = torch.Tensor([[math.cos(theta / 180 * np.pi), math.sin(-theta / 180 * np.pi)], \
        [math.sin(theta / 180 * np.pi), math.cos(theta / 180 * np.pi)]])
    return rot

    