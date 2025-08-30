import cv2
from tensorrt_yolo.libs import py_trtyolo
from time import time
import numpy as np
import torch
import torchvision

def wh2xy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2 
    y[:, 1] = x[:, 1] - x[:, 3] / 2 
    y[:, 2] = x[:, 0] + x[:, 2] / 2 
    y[:, 3] = x[:, 1] + x[:, 3] / 2 
    return y

def non_max_suppression(outputs, conf_threshold, iou_threshold):
    max_wh = 7680
    max_det = 300
    max_nms = 30000

    bs = outputs.shape[0]  # batch size
    nc = outputs.shape[1] - 4  # number of classes
    xc = outputs[:, 4:4 + nc].amax(1) > conf_threshold  # candidates

    start = time()
    limit = 0.5 + 0.05 * bs  # seconds to quit after

    output = [torch.zeros((0, 6), device=outputs.device)] * bs
    for index, x in enumerate(outputs):  # image index, image inference
        x = x.transpose(0, -1)[xc[index]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        box, cls = x.split((4, nc), 1)
        box = wh2xy(box)  # (cx, cy, w, h) to (x1, y1, x2, y2)
        if nc > 1:
            i, j = (cls > conf_threshold).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_threshold]

        if not x.shape[0]:  # no boxes
            continue
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * max_wh  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_threshold)  # NMS
        i = i[:max_det]  # limit detections

        output[index] = x[i]
        if (time() - start) > limit:
            break  # time limit exceeded

    return output

def cacl_meta(img, new_shape = (640, 640), auto = False, 
              scale_fill = False, scaleup = False, stride = 32):
    
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scale_fill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  
    dh /= 2

    width, height  = new_unpad
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    return dw, dh, width, height

def letterbox(img, new_shape = (640, 640), color = (114, 114, 114), 
              auto = False, scale_fill = False, scaleup = False, stride = 32):
    shape = img.shape[:2] 
    dw, dh, width, height = cacl_meta(img, new_shape, auto, scale_fill, scaleup, stride)
    new_unpad = (width, height)
    if shape[::-1] != new_unpad:  
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, dw, dh, width, height


def post(outputs, w, h, height, width, shape):
    outputs = non_max_suppression(outputs, 0.25, 0.7)[0]
    outputs[:, [0, 2]] -= w  # x padding
    outputs[:, [1, 3]] -= h  # y padding
    outputs[:, :4] /= min(height / shape[0], width / shape[1])

    outputs[:, 0].clamp_(0, shape[1])  # x1
    outputs[:, 1].clamp_(0, shape[0])  # y1
    outputs[:, 2].clamp_(0, shape[1])  # x2
    outputs[:, 3].clamp_(0, shape[0])  # y2
    return outputs


if __name__ == '__main__':

    option = py_trtyolo.option.InferOption()
    option.enable_swap_rb()

    model = py_trtyolo.model.DetectModel("../models/slow_only_dy.engine", option)
    img = cv2.imread("../imgs/conlon.jpg")
    # _, w, h, width, height = letterbox(img)
    # shape = img.shape[:2]
    # o = model.predict(img).to_numpy()
    # o = o.reshape(1, -1, 8400)
    # outputs = torch.from_numpy(o)
    # outputs = post(outputs, w, h, height, width, shape).numpy()
    # bboxs = outputs[:, :4]
    # for i, box in enumerate(bboxs):
    #     cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
    # cv2.imwrite("result.jpg", img)

    o = model.predict(img)


    # _, w, h, width, height = letterbox(img)
    # shape = img.shape[:2]
    # for _ in range(100):
    #     t1 = time()
    #     o = model.predict(img).to_numpy()
    #     o = o.reshape(1, -1, 8400)
    #     outputs = torch.from_numpy(o)
    #     outputs = post(outputs, w, h, height, width, shape).numpy()
    #     t2 = time()
    #     print("Infer time: ", (t2 - t1))