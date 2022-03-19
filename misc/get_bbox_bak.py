import numpy as np
import torch
import torch.nn.functional as F
from .nms import nms
import torch.nn as nn
import cv2
def local_maximum(input):
    input = torch.from_numpy(input).unsqueeze(0).unsqueeze(0)
    kernel = [[1/9., 1/9., 1/9.], [1/9., 1/9., 1/9.], [1/9., 1/9.,1/9.]]
    kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
    weight = nn.Parameter(data=kernel, requires_grad=False)
    input = F.conv2d(input, weight, stride=1,padding=1)

    keep = nn.functional.max_pool2d(input, (3, 3), stride=1, padding=1)
    keep = (keep == input).float()
    input = keep * input

    input[input < 150.0 / 255.0 * torch.max(input)] = 0
    input[input > 0] = 1
    count = int(torch.sum(input).item())
    kpoint = input.data.squeeze(0).squeeze(0).cpu().numpy()

    points = np.array(list(zip(np.nonzero(kpoint)[1], np.nonzero(kpoint)[0])))

    # f_loc.write('{} {} '.format(w_fname, count))
    boxes = np.zeros((len(points), 5)).astype(np.int32)
    for i in range(len(points)):
        x, y = points[i]
        de_h = 1
        de_w = 1
        new_x_s = max(x - de_w, 0)
        new_y_s = max(y - de_h, 0)
        boxes[i] = [new_x_s, new_y_s, 2, 2,4]
    pre_data = {'num': count, 'points': points, 'boxes': boxes}
    return pre_data

def get_boxInfo_from_Binar_map(pred_map, Binar_map, scale_map, min_area=2,scale_factor = 1., polish =True):
    def Noise_box_detection(recs):
        maintain_list = []
        recs[:, 2] = recs[:, 0] + recs[:, 2]
        recs[:, 3] = recs[:, 1] + recs[:, 3]
        # print(recs)
        length = len(recs)

        for i in range(length):
            if i < length - 1:
                j = i + 1
                index = (recs[i][0] >= recs[j:][:, 0]) & (recs[i][1] >= recs[j:][:, 1]) \
                        & (recs[i][2] <= recs[j:][:, 2]) & (recs[i][3] <= recs[j:][:, 3])
                index = np.where(index == True)[0]
                if index.size > 0:
                    continue
                else:
                    maintain_list.append(i)
            else:
                maintain_list.append(i)
        return maintain_list

    Binar_map = Binar_map.squeeze().astype(np.uint8)
    scale_map = scale_map.squeeze()
    pred_map = pred_map.squeeze()
    assert Binar_map.ndim == scale_map.ndim == 2
    cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(Binar_map, connectivity=4)  # centriod (w,h)

    boxes = stats[1:, :]
    points = centroids[1:, :]
    index = (boxes[:, 4] >= min_area)
    boxes = boxes[index]
    points = points[index]

    order = np.argsort(boxes[:, 4])
    points = points[order]
    boxes = boxes[order]

    maintain_list = Noise_box_detection(boxes.copy())
    boxes = boxes[maintain_list]
    points = points[maintain_list]
    if polish:
        boxes_app = []
        points_app = []
        for id in range(len(boxes)):
            w_s, h_s, w, h, area = boxes[id]
            sub_pre = pred_map[h_s:h_s + h, w_s:w_s + w].copy()
            iou = boxes[id, 4] / (w * h)
            ration = h / w
            if  area>16:
                if ration > 2 or ration < 0.5 or iou < 0.6:
                    pred_data = local_maximum(sub_pre)
                    if pred_data['num'] >= 1:
                        pred_data['boxes'][:, 0] += w_s
                        pred_data['boxes'][:, 1] += h_s
                        pred_data['points'][:, 0] += w_s
                        pred_data['points'][:, 1] += h_s
                        boxes[id, :] = pred_data['boxes'][0, :]
                        points[id, :] = pred_data['points'][0, :]
                        for k in range(1, pred_data['num']):
                            boxes_app.append(pred_data['boxes'][k, :])
                            points_app.append(pred_data['points'][k, :])

        # print('original:{}, add_boxes:{}, final_boxes:{}'.format(len(boxes), len(boxes_app), len(boxes) + len(boxes_app)))

        if len(boxes_app) > 0:
            boxes = np.concatenate((boxes, np.array(boxes_app)))
            points = np.concatenate((points, np.array(points_app)))
        #
    new_boxes = np.zeros((len(points), 5)).astype(np.float32)

    for i in range(len(boxes)):
        x_s, y_s, w, h, area = boxes[i]
        x, y = points[i]
        _scale = scale_map[y_s:y_s + h, x_s:x_s + w]
        _mask = Binar_map[y_s:y_s + h, x_s:x_s + w]
        _pred = pred_map[y_s:y_s + h, x_s:x_s + w]
        score = _pred.max()

        sigma = np.sqrt(w ** 2 + h ** 2)
        sin = h / sigma
        cos = w / sigma
        index = _mask > 0
        scale = max(_scale[index].max(), sigma / 2) if  index.sum()>0 else sigma / 2

        de_h,  de_w = scale * sin, scale * cos

        new_x_s, new_x_e = x - de_w, x + de_w
        new_y_s, new_y_e = y - de_h, y + de_h

        new_boxes[i] = [new_x_s, new_y_s, new_x_e, new_y_e, score]

    boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
    boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
    boxes = boxes[:, :4]

    if polish:
        keep = nms(new_boxes,0.3)
        points = points[keep]
        boxes = boxes[keep]
        new_boxes = new_boxes[keep]

    pre_data = {'num': len(points), 'points': points*scale_factor, 'boxes': boxes*scale_factor, 'new_boxes': new_boxes*scale_factor}
    return pre_data