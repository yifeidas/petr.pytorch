import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torchvision
import time
import argparse
import numpy as np
import random
import os
import math

def make_optimizer(model, hyp, args):
    pg0, pg1, pg2, pg3 = [], [], [], []  # optimizer parameter groups
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d) or isinstance(v, nn.LayerNorm):
            pg0.append(v.weight)  # no decay
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            if hasattr(v, 'img_encoder'):
                pg3.append(v.weight)
            else:
                pg1.append(v.weight)  # apply decay

    params = [{'params': pg1, 'lr':hyp['lr0'], 'weight_decay': hyp['weight_decay']},
            {'params': pg3, 'lr':(hyp['lr0'] * 0.1), 'weight_decay': hyp['weight_decay']},
            {'params': pg0, 'lr':hyp['lr0'], 'weight_decay': 0.},
            {'params': pg2, 'lr':hyp['lr0'], 'weight_decay': 0.}]

    optimizer = optim.AdamW(params, betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    # optimizer = optim.Adam(params, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999), weight_decay=hyp['weight_decay'])
    # optimizer = optim.SGD(params, lr=hyp['lr0'], momentum=hyp['momentum'], weight_decay=hyp['weight_decay']) 

    if hyp['scheduler'] == 'lamda':
        lf = one_cycle(1, hyp['lrf'], hyp['maxepoch'])  # cosine 1->hyp['lrf']
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    else:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=hyp['steps'], gamma=hyp['lrdecay'])
    return optimizer, scheduler

def one_cycle(y1=0.0, y2=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1

def warmup(n_iter, optimizer, hyp, max_iter):
    xi = [0, max_iter]  # x interp
    # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
    for j, x in enumerate(optimizer.param_groups):
        x['lr'] = np.interp(n_iter, xi, [0.0, x['initial_lr']])

def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True

def reduce_dict_mean(input_dict):
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values, op=torch.distributed.ReduceOp.SUM)
        values = values / dist.get_world_size()
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict

class Trainer(object):
    def __init__(self, net, trainLoader, valLoader, args, hyp, rank, optimizer=None, random_resize=False):
        self.net = net
        # initialize_weights(self.net)
        self.netname = net.name + '-' + valLoader.dataset.name
        self.loader_train = trainLoader
        self.max_iter = len(self.loader_train)
        self.loader_val = valLoader
        self.args = args
        self.hyp = hyp
        self.rank = rank
        self.n_iter = 0
        
        self.start_epoch = 0
        self.max_epoch = hyp['maxepoch']
        self.random_resize = random_resize
        self.width = self.hyp['width']
        self.height = self.hyp['height']
        self.max_warmup_iter = min(round(hyp['warmup_epochs'] * len(trainLoader)), 3000)

        if self.rank == 0:
            save_dir = os.path.join(args.logdir, self.netname)
            i = 0 
            while os.path.exists(save_dir):
                save_dir = os.path.join(args.logdir, (self.netname + '_%02d'%i))
                i += 1
            self.writer = SummaryWriter(save_dir)
        
        self.optimizer, self.scheduler = make_optimizer(self.net, hyp, args)
        self.scaler = amp.GradScaler(enabled=bool(self.args.half))
        self.min_loss = 1e5

        if args.resume:
            self.resume_train(args)

        self.net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.net)
        print('Using SyncBatchNorm default')
        self.net =DDP(self.net, find_unused_parameters=False, device_ids=[rank])

    def run(self):
        for epoch in range(self.start_epoch, self.max_epoch):
            if self.rank == 0:
                print('\n  Start Training Epoch: %03d/%03d'%(epoch, self.max_epoch))
            t0 = time.time()
            # Train
            loss_train = self.train_epoch(epoch)
            # Validation
            loss_val = self.validate_epoch(epoch)
            # Check and save model
            self.save_checkpoint(epoch, loss_val)
            # Take a scheduler step
            self.scheduler.step()

            t1 = time.time()
            if self.rank == 0:
                self.writer.add_scalars('Loss/train-val', {'train_loss': loss_train, 'val_loss':loss_val}, epoch)
            print('one training epoch time ', t1 - t0)
        print('Training completed')

    def random_resize_input(self, data_dict, epoch):
        if self.random_resize :
            w_raw, h_raw = self.width, self.height
            if self.n_iter%10 == 0:
                self.height = (self.hyp['height'] // 32 + random.randint(-4, 4)) * 32
                self.width = self.height * 2
                dist_tensor = torch.Tensor([self.width, self.height]).int().cuda(self.rank) if self.rank == 0 else torch.zeros(2).int().cuda(self.rank)
                dist.barrier()
                dist.broadcast(dist_tensor, 0)
                if self.rank != 0:
                    size = dist_tensor.cpu().numpy()
                    self.width, self.height = int(size[0]), int(size[1])
            B, N, C, H, W = data_dict['image'].shape
            image = F.interpolate(data_dict['image'].view(-1,C,H,W), size=(self.height, self.width), mode='bilinear')
            intrins = data_dict['intrins']
            intrins[:, :, 0, :] *= ( self.width / w_raw)
            intrins[:, :, 1, :] *= ( self.height / h_raw)
            data_dict['image'] = image.view(B,N,C,self.height,self.width)
            data_dict['intrins'] = intrins
        return data_dict
    
    def train_epoch(self, epoch):
        self.net.train()
        self.net.module.freeze_bn()

        # if epoch == (self.max_epoch - 2):
        #     self.loader_train.dataset.close_bev_aug()

        self.loader_train.sampler.set_epoch(epoch)
        loss_train = 0.0
        pbar = tqdm(enumerate(self.loader_train), total=len(self.loader_train))
        for i, batch in pbar:
            if self.n_iter < self.max_warmup_iter:
                warmup(self.n_iter, self.optimizer, self.hyp, self.max_warmup_iter)
            self.optimizer.zero_grad()
            data_dict = self.sample_to_cuda(batch)
            self.random_resize_input(data_dict, epoch)

            with amp.autocast(enabled=bool(self.args.half)):
                loss, loss_tags = self.net(data_dict, target=data_dict['label'])
                # loss, loss_tags = self.criterion(pred, data_dict)

            # loss *= self.args.worldsize

            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=35, norm_type=2.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            loss_tags = reduce_dict_mean(loss_tags)
            if self.rank == 0  and (self.n_iter % 1 == 0):
                for k in loss_tags.keys():
                    self.writer.add_scalar('Loss/%s'%k, loss_tags[k].cpu().numpy(), self.n_iter)
            loss_train += loss_tags['loss']
            self.n_iter += 1
        return loss_train / (i+1)

    def validate_epoch(self, epoch):
        self.net.eval()
        self.loader_val.sampler.set_epoch(epoch)
        loss_val = 0.0
        pbar = tqdm(enumerate(self.loader_val), total=len(self.loader_val))
        for i, batch in pbar:
            data_dict = self.sample_to_cuda(batch)
            with amp.autocast(enabled=bool(self.args.half)):
                loss, loss_tags = self.net(data_dict, target=data_dict['label'])
                # loss, loss_tags = self.criterion(pred, data_dict)
            loss_tags = reduce_dict_mean(loss_tags)
            if self.rank == 0 and (i%10 == 0):
                for k in loss_tags.keys():
                    self.writer.add_scalar('Loss/val_%s'%k, loss_tags[k].cpu().numpy(), (i + epoch*len(self.loader_val)))
            loss_val += loss_tags['loss']
        return loss_val / (i+1)

    def sample_to_cuda(self, data):
        if isinstance(data, str):
            return data
        elif isinstance(data, dict):
            return {key: self.sample_to_cuda(data[key]) for key in data.keys()}
        elif isinstance(data, list):
            return [self.sample_to_cuda(val) for val in data]
        else:
            return data.cuda(self.rank)
        
    def save_checkpoint(self, i, loss_val):
        if self.rank == 0:
            print('Saving weights...')
            net_dict = self.net.module.state_dict()
            state = {
                'net': net_dict,
                'loss': loss_val,
                'epoch': i,
                # "optimizer": self.optimizer.state_dict(),
                'args': self.args,
                'hyp': self.hyp,
                'iter': self.n_iter
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            if (i+1) %  (self.max_epoch//5) == 0:
                torch.save(state, '%s/%s-%03d.pth'%(self.args.ckpt_output, self.netname, i))
            if loss_val < self.min_loss:
                torch.save(state, '%s/%s-best.pth'%(self.args.ckpt_output, self.netname))
                self.min_loss = loss_val
            torch.save(state, '%s/%s-backup.pth'%(self.args.ckpt_output, self.netname))

    def resume_train(self, args):
        ckpt = torch.load(args.ckpt, map_location='cpu')
        self.net.load_state_dict(ckpt["net"])
        # self.optimizer.load_state_dict(ckpt["optimizer"])
        self.start_epoch = ckpt['epoch'] + 1
        self.n_iter = ckpt['iter']
        self.scheduler.last_epoch = ckpt['epoch']