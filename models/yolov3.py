import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quantization import DeQuantStub, QuantStub

from pytorch_modules.backbones import (mobilenet, resnet34, resnet50,
                                       resnext50_32x4d)
#from pytorch_modules.backbones import (resnext50_32x4d, resnet34, resnet50)                                      
from pytorch_modules.backbones.mobilenet import ConvBNReLU, InvertedResidual
from pytorch_modules.nn import ConvNormAct, SeparableConvNormAct
from pytorch_modules.utils import initialize_weights, replace_relu6
#from pytorch_modules.utils import initialize_weights

from .fpn import FPN
from .spp import SPP


class YOLOLayer(nn.Module):
    def __init__(self, anchors, nc, img_size, yolo_index):
        super(YOLOLayer, self).__init__()

        self.anchors = torch.Tensor(anchors)
        self.na = len(anchors)  # number of anchors (3)
        self.nc = nc  # number of classes (80)
        self.nx = 0  # initialize number of x gridpoints
        self.ny = 0  # initialize number of y gridpoints

    def forward(self, p, img_size):
        bs, ny, nx = p.shape[0], p.shape[-2], p.shape[-1]
        if (self.nx, self.ny) != (nx, ny):
            create_grids(self, img_size, (nx, ny), p.device, p.dtype)

        # p.view(bs, 255, 13, 13) -- > (bs, 3, 13, 13, 85)  # (bs, anchors, grid, grid, classes + xywh)
        p = p.view(bs, self.na, self.nc + 6, self.ny,
                   self.nx).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        if self.training:
            return p


        else:  # inference
            io = p.clone()  # inference output
            io[..., 0:2] = torch.sigmoid(io[..., 0:2]) + self.grid_xy  # xy
            io[..., 2:4] = torch.exp(
                io[..., 2:4]) * self.anchor_wh  # wh yolo method
            io[..., :4] *= self.stride

            torch.sigmoid_(io[..., 4:-1])

            if self.nc == 1:
                io[...,
                   5] = 1  # single-class model https://github.com/ultralytics/yolov3/issues/235

            return io.view(bs, -1, 6 + self.nc), p


class YOLOV3(nn.Module):
    # YOLOv3 object detection model

    def __init__(
            self,
            num_classes,
            img_size=(416, 416),
#23 10 38 15 52 22 90 25 93 54 185 97 207 31 436 289 452 92
            anchors=[
                #[[116, 90], [156, 198], [373, 326]],
                #[[30, 61], [62, 45], [59, 119]],
                #[[10, 13], [16, 30], [33, 23]],
                [[207, 31],[436, 289],[452, 92]],
                [[90, 25],[93, 54],[185, 97]],
                [[23, 10],[38, 15],[52, 22]],
            ]):
        super(YOLOV3, self).__init__()
        self.backbone = mobilenet.mobilenet_v2(pretrained=True)
        #self.backbone = resnet50(pretrained=True)
        depth = 5
        width = [512, 256, 128]
        planes_list = [1280, 96, 32]
        self.spp = nn.Sequential(ConvNormAct(1280, 320, 1, activate=nn.ReLU(True)), SPP())
        self.fpn = FPN(planes_list, width, depth)
        self.head = nn.ModuleList([])
        self.yolo_layers = nn.ModuleList([])
        for i in range(3):
            self.head.append(
                nn.Sequential(
                    SeparableConvNormAct(width[i],
                                         width[i],
                                         activate=nn.ReLU(True)),
                    nn.Conv2d(width[i],
                              len(anchors[i]) * (6 + num_classes), 1),
                ))
            self.yolo_layers.append(
                YOLOLayer(
                    anchors=np.float32(anchors[i]),
                    nc=num_classes,
                    img_size=img_size,
                    yolo_index=i,
                ))
        initialize_weights(self.fpn)
        initialize_weights(self.head)

    def forward(self, x):
        img_size = x.shape[-2:]
        if hasattr(self, 'quant'):
            x = self.quant(x)
            # print(x)
        features = self.backbone(x)
        # features = [features[-1], features[-2], features[-3]]
        features = [self.spp(features[-1]), features[-2], features[-3]]
        features = self.fpn(features)
        features = [
            head(feature) for feature, head in zip(features, self.head)
        ]
        if os.environ.get('MODEL_EXPORT'):
            return features
        if hasattr(self, 'dequant'):
            features = [self.dequant(feature) for feature in features]
        output = [
            yolo(feature, img_size)
            for feature, yolo in zip(features, self.yolo_layers)
        ]
        if self.training:
            return tuple(output)
        else:
            io, p = list(zip(*output))  # inference output, training output
            return torch.cat(io, 1), p

    def fuse_model(self):
        # replace_relu6(self)
        for m in self.modules():
            if type(m) == ConvBNReLU:
                torch.quantization.fuse_modules(m, ['0', '1', '2'], inplace=True)
            if type(m) == ConvNormAct:
                torch.quantization.fuse_modules(m, ['0', '1', '2'], inplace=True)
            if type(m) == InvertedResidual:
                for idx in range(len(m.conv)):
                    if type(m.conv[idx]) == nn.Conv2d:
                        torch.quantization.fuse_modules(m.conv, [str(idx), str(idx + 1)], inplace=True)


def create_grids(self,
                 img_size=416,
                 ng=(13, 13),
                 device='cpu',
                 type=torch.float32):
    nx, ny = ng  # x and y grid size
    self.img_size = max(img_size)
    self.stride = self.img_size / max(ng)

    # build xy offsets
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
    self.grid_xy = torch.stack((xv, yv), 2).to(device).type(type).view(
        (1, 1, ny, nx, 2))

    # build wh gains
    self.anchor_vec = self.anchors.to(device) / self.stride
    self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1,
                                          2).to(device).type(type)
    self.ng = torch.Tensor(ng).to(device)
    self.nx = nx
    self.ny = ny


if __name__ == '__main__':
    model = YOLOV3(80)
    a = torch.rand([4, 3, 416, 416])
    b = model(a)
    print(b[0].shape)
    b[0].mean().backward()
