
video_path = 'T:\CVPR2022/video_ori'
label_path = 'T:\CVPR2022/label_list_all'

# video_path = '/media/D/GJY/ht/CVPR2022/dataset/SENSE/video_ori'
# label_path = '/media/D/GJY/ht/CVPR2022/dataset/SENSE/label_list_all'

scene = '1102_IMG_5764_cut_02'

import json
import os
import os.path as osp
from numpy import array
import numpy as np
import pylab as pl
from collections import defaultdict
import  cv2
import torch
def plot_id(img0,img1,kpts0, kpts1, match_gt):
    point_r_value = 20
    thickness = 4

    kpts0, kpts1 = kpts0.numpy().astype(int), kpts1.numpy().astype(int)
    white = (255, 255, 255)
    black = (0, 0, 0)
    green = (0,255,0)
    red = (0, 0, 255)
    blue = (255,0,0)
    # import pdb
    # pdb.set_trace()
    for x, y in kpts0[match_gt['a2b'][:,0]]:
        cv2.circle(img0, (x, y), point_r_value, green, thickness, lineType=cv2.LINE_AA)
        # cv2.circle(img0, (x, y), 3, white, -1, lineType=cv2.LINE_AA)

    for x, y in kpts1[match_gt['a2b'][:,1]]:
        cv2.circle(img1, (x, y), point_r_value, green, thickness, lineType=cv2.LINE_AA)
        # cv2.circle(img1, (x, y), 3, white, -1, lineType=cv2.LINE_AA)

    for x, y in kpts0[match_gt['un_a']]:
        cv2.circle(img0, (x, y), point_r_value, red, thickness, lineType=cv2.LINE_AA)

    for x, y in kpts1[match_gt['un_b']]:
        cv2.circle(img1, (x, y), point_r_value, blue, thickness, lineType=cv2.LINE_AA)

    return  img0, img1

def plot_intro():
    Info_dict={}
    gts = defaultdict(list)
    with open(os.path.join(label_path,scene+'.txt')) as f:
        lines = f.readlines()
        for line in lines:
            lin_list = [i for i in line.rstrip().split(' ')]
            ind = lin_list[0]
            # print(lin_list)
            lin_list = [float(i) for i in lin_list[3:] if i != '']
            assert  len(lin_list)%7==0
            gts[ind]=lin_list

    root  = osp.join(video_path, scene)
    img_ids = sorted(os.listdir(root))
    id_list = []
    pair_img = []
    pair_target = []
    for idx, img_id in enumerate(img_ids):
        if not img_id.endswith("jpg"):
            continue
        if idx==0 or idx == 28:
            img_path=osp.join(root, img_id)
            label = gts[img_id]
            box_and_point = torch.tensor(label).view(-1,7)
            boxes = box_and_point[:,0:4]
            points = box_and_point[:,4:6]
            ids = box_and_point[:,6].long()
            id_list.append(ids)

            img = cv2.imread(img_path)

            pair_img.append(img)
            pair_target.append({'person_id':ids, 'points':points})

            # plot_img = plot_boxes(img, boxes, points, ids)
            # cv2.imshow(img_id, plot_img)
            # cv2.waitKey()
        # print(lines)

    a_ids = pair_target[0]['person_id']
    b_ids = pair_target[1]['person_id']
    dis = a_ids.unsqueeze(1).expand(-1, len(b_ids)) - b_ids.unsqueeze(0).expand(len(a_ids), -1)
    dis = dis.abs()
    # import pdb
    # pdb.set_trace()
    matched_a, matched_b = torch.where(dis == 0)
    matched_a2b = torch.stack([matched_a, matched_b], 1)
    unmatched0 = torch.where(dis.min(1)[0] > 0)[0]
    unmatched1 = torch.where(dis.min(0)[0] > 0)[0]
    match_gt = {'a2b': matched_a2b, 'un_a': unmatched0, 'un_b': unmatched1}
    img0, img1 = plot_id(pair_img[0], pair_img[1], pair_target[0]['points'], pair_target[1]['points'], match_gt)
    cv2.imwrite('0.png',img0.copy())
    cv2.imwrite('30.png', img1.copy())
    cv2.imshow('0', img0)
    cv2.imshow('1', img1)

    cv2.waitKey()

if __name__ == '__main__':
    plot_intro()