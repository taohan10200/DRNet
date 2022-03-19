#!/usr/bin/env python
# coding: utf-8

import csv
import os.path as osp
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data

from torchvision.ops.boxes import clip_boxes_to_image
from PIL import Image
try:
    from scipy.misc import imread
except ImportError:
    from scipy.misc.pilutil import imread
import random
import json
class Dataset(data.Dataset):
    """
    Dataset class.
    """
    def __init__(self, txt_path, base_path,main_transform=None,img_transform=None,train=True, datasetname='Empty'):
        self.base_path = base_path
        self.bboxes = defaultdict(list)
        self.imgs_path = []
        self.labels = []
        self.datasetname = datasetname
        if train:
            with open(osp.join(base_path, txt_path), 'r') as txt:
                scene_names = txt.readlines()
        else:
            scene_names = txt_path             # for val and test
        if datasetname == 'NWPU':
            for line in scene_names:
                splited = line.strip().split()
                self.imgs_path.append(os.path.join(self.base_path, 'images', splited[0] + '.jpg'))
                self.labels.append(os.path.join(self.base_path, 'jsons', splited[0] + '.json'))
        else:
            for i in scene_names:
                if datasetname == 'HT21':
                    img_path, label= HT21_ImgPath_and_Target(base_path,i.strip())
                elif datasetname == 'SENSE':
                    img_path, label = SENSE_ImgPath_and_Target(base_path,i.strip())
                else:
                    raise NotImplementedError
                self.imgs_path+=img_path
                self.labels +=label

        self.is_train = train
        self.main_transforms = main_transform
        self.img_transforms = img_transform

    def __len__(self):
        return len(self.imgs_path)

    def filter_targets(self, boxes, ignore_ar, im):
        """
        Remove boxes with 0 or negative area
        """
        filtered_targets = []
        filtered_ignorear = []
        for bx, ig_ar in zip(boxes, ignore_ar):
            clipped_im = clip_boxes_to_image(torch.tensor(bx), im.shape[:2]).cpu().numpy()
            area_cond = self.get_area(clipped_im) <= 1
            dim_cond = clipped_im[2] - clipped_im[0] <= 0 and clipped_im[3] - clipped_im[1] <= 0
            # if width_cond or height_cond or area_cond or dim_cond:
            if area_cond or dim_cond:
                continue
            filtered_targets.append(clipped_im)
            filtered_ignorear.append(ig_ar)
        return np.array(filtered_targets), filtered_ignorear


    def get_area(self, boxes):
        """
        Area of BB
        """
        boxes = np.array(boxes)
        if len(boxes.shape) != 2:
            area = np.product(boxes[2:4] - boxes[0:2])
        else:
            area = np.product(boxes[:, 2:4] - boxes[:, 0:2], axis=1)
        return area

    def create_target_dict(self, img, target, index, ignore_ar=None):
        """
        Create the GT dictionary in similar style to COCO.
        For empty boxes, use [1,2,3,4] as box dimension, but with
        background class label. Refer to __getitem__ comment.
        """
        n_target = len(target['points'])
        image_id = torch.tensor([index])
        visibilities = torch.ones((n_target), dtype=torch.float32)
        iscrowd = torch.zeros((n_target,), dtype=torch.int64)

        # When there are no targets, set the BBOxes to 1pixel wide
        # and assign background label
        if n_target == 0:
            target, n_target = [[1, 2, 3, 4]], 1
            boxes = torch.tensor(target, dtype=torch.float32)
            labels = torch.zeros((n_target,), dtype=torch.int64)

        else:
            boxes = torch.tensor(target['points'], dtype=torch.float32)
            labels = torch.ones((n_target,), dtype=torch.int64)

        # area = torch.tensor(self.get_area(target))

        target_dict = {
                        'image' : img,
                        'points': target['points'],
                        'person_id': target['person_id'],
                        'frame_id': image_id,
                        'scene_name':target['scene_name']
                        # 'area': area,
                        # 'iscrowd': iscrowd,
                        # 'visibilities': visibilities,
         }

        # Need ignore label for CHuman evaluation
        if self.is_train:
            return target_dict
        else:
            assert len(ignore_ar)== len(target)
            target_dict['ignore'] = ignore_ar
            return target_dict


    def __getitem__(self, index):

        img = Image.open(self.imgs_path[index])
        if img.mode is not 'RGB':
            img=img.convert('RGB')

        if self.datasetname == 'NWPU':
            target = self.NWPU_Imgpath_and_Target(json_path=self.labels[index])
        else:
            target = self.labels[index].copy()

        if self.main_transforms is not None:
            img, target = self.main_transforms(img, target)
        if self.img_transforms is not None:
            img = self.img_transforms(img)

        return  img,target

    def NWPU_Imgpath_and_Target(self,json_path):

        with open(json_path, 'r') as f:
            info = json.load(f)
        boxes = torch.tensor(info['boxes'],dtype=torch.float32).view(-1,4).contiguous()
        points = torch.zeros((boxes.size(0), 2), dtype=torch.float32)
        if boxes.size(0) > 0:
            sigma = 0.6 * torch.stack([(boxes[:, 2] - boxes[:, 0]) / 2, (boxes[:, 3] - boxes[:, 1]) / 2], 1).min(1)[0]
        else:
            sigma = torch.tensor([])
        # sigma = (((boxes[:, 2] - boxes[:, 0]) / 2.)**2 +((boxes[:, 3] - boxes[:, 1]) / 2.)**2).sqrt()
        points[:, 0] = (boxes[:, 2] + boxes[:, 0]) / 2.
        points[:, 1] = (boxes[:, 3] + boxes[:, 1]) / 2.
        ids = torch.arange(points.size(0),dtype=torch.int64)

        return {'scene_name': 'nwpu', 'frame': json_path.split('/')[-1], 'person_id': ids, 'points': points, 'sigma':sigma}

def HT21_ImgPath_and_Target(base_path,i):
    img_path = []
    labels=[]
    root  = osp.join(base_path, i + '/img1')
    img_ids = os.listdir(root)
    img_ids.sort()
    gts = defaultdict(list)
    with open(osp.join(root.replace('img1', 'gt'), 'gt.txt'), 'r') as f:
        lines = f.readlines()
        for lin in lines:
            lin_list = [float(i) for i in lin.rstrip().split(',')]
            ind = int(lin_list[0])
            gts[ind].append(lin_list)

    for img_id in img_ids:
        img_id = img_id.strip()
        single_path = osp.join(root, img_id)
        labels_point = gts[int(img_id.split('.')[0])]
        points = torch.zeros((len(labels_point), 2),dtype=torch.float32)
        ids = torch.zeros(len(labels_point),dtype=torch.int64)
        sigma = torch.zeros(len(labels_point),dtype=torch.float32)
        for idx, label in enumerate(labels_point):
            points[idx, 0] = label[2] + label[4] / 2  # x1
            points[idx, 1] = label[3] + label[5] / 2  # y1
            ids[idx] = int(label[1])
            sigma[idx] = min(label[4], label[5]) /2. #torch.tensor((label[4]/2) **2 + (label[5]/2)**2 ).sqrt()
        img_path.append(single_path)
        if len(points)<30:
            print('saaaaaaaaaaaaadddddddddasda')
        labels.append({'scene_name':i,'frame':int(img_id.split('.')[0]), 'person_id':ids, 'points':points, 'sigma':sigma})
    return img_path, labels

def SENSE_ImgPath_and_Target(base_path,i):
    img_path = []
    labels=[]
    root  = osp.join(base_path, 'video_ori', i )
    img_ids = os.listdir(root)
    img_ids.sort()
    gts = defaultdict(list)
    with open(root.replace('video_ori', 'label_list_all')+'.txt', 'r') as f: #label_list_all_rmInvalid
        lines = f.readlines()
        for lin in lines:
            lin_list = [i for i in lin.rstrip().split(' ')]
            ind = lin_list[0]
            lin_list = [float(i) for i in lin_list[3:] if i != '']
            assert len(lin_list) % 7 == 0
            gts[ind] = lin_list

    for img_id in img_ids:
        img_id = img_id.strip()
        single_path = osp.join(root, img_id)
        label = gts[img_id]
        box_and_point = torch.tensor(label).view(-1, 7).contiguous()
        # points = torch.zeros((box_and_point.size(0), 2),dtype=torch.float32)
        # ids = torch.zeros(box_and_point.size(0),dtype=torch.int64)

        points = box_and_point[:, 4:6].float()
        ids = (box_and_point[:, 6]).long()
        # import pdb
        # pdb.set_trace()
        if ids.size(0)>0:
            sigma = 0.6*torch.stack([(box_and_point[:,2]-box_and_point[:,0])/2,(box_and_point[:,3]-box_and_point[:,1])/2],1).min(1)[0]  #torch.sqrt(((box_and_point[:,2]-box_and_point[:,0])/2)**2 + ((box_and_point[:,3]-box_and_point[:,1])/2)**2)
        else:
            sigma = torch.tensor([])
        img_path.append(single_path)

        # print(sigma)
        labels.append({'scene_name':i,'frame':int(img_id.split('.')[0]), 'person_id':ids, 'points':points, 'sigma':sigma})
    return img_path, labels


class TestDataset(data.Dataset):
    """
    Dataset class.
    """
    def __init__(self,scene_name, base_path, main_transform=None, img_transform=None, interval=1, target=True, datasetname='Empty'):
        self.base_path = base_path
        self.target = target

        # import  pdb
        # pdb.set_trace()
        if self.target:
            if datasetname == 'HT21':
                self.imgs_path, self.label = HT21_ImgPath_and_Target(self.base_path, scene_name)
            elif datasetname == 'SENSE':
                self.imgs_path, self.label = SENSE_ImgPath_and_Target(self.base_path, scene_name)
            else:
                raise NotImplementedError
        else:
            if datasetname == 'HT21':
                self.imgs_path = self.generate_imgPath_label(scene_name)
            elif datasetname == 'SENSE':
                self.imgs_path, self.label = SENSE_ImgPath_and_Target(self.base_path, scene_name)
            else:
                raise NotImplementedError
        self.interval =interval

        self.main_transforms = main_transform
        self.img_transforms = img_transform
        self.length =  len(self.imgs_path)
    def __len__(self):
        return len(self.imgs_path) - self.interval


    def __getitem__(self, index):
        index1 = index
        index2 = index + self.interval
        img1 = Image.open(self.imgs_path[index1])
        img2 = Image.open(self.imgs_path[index2])
        # import pdb
        # pdb.set_trace()
        if img1.mode is not 'RGB':
            img1=img1.convert('RGB')
        if img2.mode is not 'RGB':
            img2 = img2.convert('RGB')
        if self.img_transforms is not None:
            img1 = self.img_transforms(img1)
            img2 = self.img_transforms(img2)
        if self.target:
            target1 = self.label[index1]
            target2 = self.label[index2]
            return  [img1,img2], [target1,target2]
        # img, target = self.refine_transformation(transformed_dict)


        return [img1,img2], None

    def generate_imgPath_label(self, i):

        import re
        def myc(string):
            p = re.compile("\d+")
            return int(p.findall(string)[0])

        import re
        def mykey(string):
            p = re.compile("\d+")  # \d 是转数字 +是多个数字
            return int(p.findall(string)[1])

        img_path = []
        root = osp.join(self.base_path, i +'/img1')
        img_ids = os.listdir(root)
        img_ids.sort(key=myc)


        for img_id in img_ids:
            img_id = img_id.strip()
            single_path = osp.join(root, img_id)
            img_path.append(single_path)
        # print(img_path)
        # import  pdb
        # pdb.set_trace()
        return img_path
