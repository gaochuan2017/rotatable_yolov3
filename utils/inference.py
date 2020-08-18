import cv2
import numpy as np
import torch

from pytorch_modules.utils import device

from .utils import non_max_suppression, scale_coords ,xywht2polygon


@torch.no_grad()
def inference(model, imgs, img_size=(416, 416), conf_thres=0.3, nms_thres=0.5):
    shapes = [img.shape for img in imgs]
    img_size = (img_size[0] , img_size[1])
    imgs = [
        cv2.resize(img, img_size)[:, :, ::-1].transpose(2, 0, 1)
        for img in imgs
    ]
    imgs = torch.FloatTensor(imgs).to(device) / 255.
    preds = model(imgs)

    dets = []
    for pred, shape in zip(preds, shapes):
        # Apply NMS
        '''
        ps = pred[0][0:2,:]
        ps[0,0] = 400
        ps[0,1] = 300
        ps[0,2] = 200
        ps[0,3] = 100
        ps[0,-1] = 0
        for i in range(20):
            ps[1,i] = ps[0,i] + 1
        ps[:,7:8] = 1
        '''

        det = non_max_suppression(pred, conf_thres, nms_thres)[0]

        # Process detections
        if det is None:
            det = []
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :8] = scale_coords(imgs.shape[2:], det[:, :8],
                                      shape[:2]).round()
        dets.append(det)
    return dets
