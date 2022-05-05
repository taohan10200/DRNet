import cv2
import os
from collections import defaultdict
import  numpy as np
import os.path as osp
def plot_boxes(cur_frame, head_map, points, ids,body_map={}, text=True):
    plotting_im = cur_frame.copy()
    for index,  t_dim in enumerate(head_map):
        (startX, startY, endX, endY) = [int(i) for i in t_dim]
        cv2.rectangle(plotting_im, (startX, startY), (endX, endY),
                      (0, 255, 0), 2)
        cur_centroid = tuple([(startX+endX)//2,
                              (startY+endY)//2])

        # cv2.circle(plotting_im, cur_centroid, 2,
        #               (255, 0, 0), 2)

        if text:
            cv2.putText(plotting_im, str(ids[index]), cur_centroid,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    for index,  t_dim in enumerate(points):
        X, Y, = [int(i) for i in t_dim]
        cv2.circle(plotting_im, (X, Y), 2,
                      (0, 0, 255), 2)

    for index, (t_id, t_dim) in enumerate(body_map.items()):
        (startX, startY, endX, endY) = [int(i) for i in t_dim]
        cv2.rectangle(plotting_im, (startX, startY), (endX, endY),
                      (0, 255, 0), 2)
    return plotting_im

def CroHead():
    root = '../../dataset/HT21/train'
    sub_scenes = os.listdir(root)
    print(sub_scenes)

    for sub_scene in sub_scenes[2:]:
        imgs_path = os.path.join(root, sub_scene, 'img1')
        imgs_id = os.listdir(imgs_path)
        det_path = os.path.join(imgs_path.replace('img1', 'det'), 'det.txt')

        bboxes = defaultdict(list)
        with open(det_path, 'r') as f:
            lines = f.readlines()
            # imgs_path = [i.rstrip().strip("#").lstrip()
            #                   for i in lines if i.startswith('#')]
            for lin in lines:
                lin_list = [float(i) for i in lin.rstrip().split(',')]
                ind = int(lin_list[0])
                bboxes[ind].append(lin_list)
        f.close()
        gts = defaultdict(list)
        with open(os.path.join(imgs_path.replace('img1','gt'), 'gt.txt'), 'r') as f:
            lines = f.readlines()
            for lin in lines:
                lin_list = [float(i) for i in lin.rstrip().split(',')]
                ind = int(lin_list[0])
                gts[ind].append(lin_list)
        f.close()
        # print(gts)
        # print(imgs_id)

        for img_id in imgs_id:
            img_path=os.path.join(imgs_path,img_id)
            labels = bboxes[int(img_id.split('.')[0])]
            labels_point = gts[int(img_id.split('.')[0])]
            annotations = np.zeros((0, 4))
            points =  np.zeros((0, 2))
            if len(labels) == 0:
                label = [[0, 0, 0, 0, 0]]
            ignore_ar = []
            for idx, label in enumerate(labels):
                annotation = np.zeros((1, 4))
                # bbox
                annotation[0, 0] = label[2]  # x1
                annotation[0, 1] = label[3]  # y1
                annotation[0, 2] = label[4] +label[2] # x2
                annotation[0, 3] = label[5] +label[3]# y2
                annotations = np.append(annotations, annotation, axis=0)
            for idx, label in enumerate(labels_point):
                point = np.zeros((1, 2))
                # bbox
                point[0, 0] = label[2]  + label[4]/2# x1
                point[0, 1] = label[3] + label[5]/2  # y1
                points = np.append(points, point, axis=0)
            # print(annotations)
            print(len(points))
            img = cv2.imread(img_path)
            img = plot_boxes(img,{},points)
            # cv2.imshow(img_id, img)
            save_path = img_path.replace('img1','vis')
            cv2.imwrite(save_path,img)
            # cv2.waitKey()

video_path = 'E:/netdisk\SenseCrowd/video_ori'
label_path = 'E:/netdisk\SenseCrowd/label_list_all_rmInvalid'
import json
import os
from numpy import array
import numpy as np
import pylab as pl
def SensorCrowd():
    Info_dict={}
    time = 0
    for scene in sorted(os.listdir(video_path)[51:]):
        print(scene)
        gts = defaultdict(list)
        with open(os.path.join(label_path,scene+'.txt')) as f:
            lines = f.readlines()
            for line in lines:
                lin_list = [i for i in line.rstrip().split(' ')]
                ind = lin_list[0]
                lin_list = [float(i) for i in lin_list[3:] if i != '']
                assert  len(lin_list)%7==0
                gts[ind]=lin_list

        root  = osp.join(video_path, scene)
        img_ids = os.listdir(root)
        print(img_ids)
        id_list = []
        for img_id in img_ids:
            if not img_id.endswith("jpg"):
                continue
            time+=1/5
            img_path=osp.join(root, img_id)
            label = gts[img_id]
            box_and_point = np.array(label).reshape(-1,7)
            boxes = box_and_point[:,0:4]
            points = box_and_point[:,4:6]
            ids = box_and_point[:,6].astype(np.int)

            id_list.append(ids)

            img = cv2.imread(img_path)
            print(img_path)
            plot_img = plot_boxes(img, boxes, points, ids)
            cv2.imshow(img_id, plot_img)
            cv2.waitKey()
        all_id = np.concatenate(id_list)
        Info_dict.update({scene:len(set(all_id))})


    print(time)
    with open('info.json','w') as f:
        json.dump(Info_dict,f)

    # print(Info_dict)

def SENSE_train_val_test():
    import random
    random.seed(0)
    scenarios = ['1_cut', '']
    all_scenarios = []
    with open('./info.json','r') as f:
        a = json.load(f)
    for k, v in a.items():
        all_scenarios.append(k)
    print(len(all_scenarios))
    train_val = random.sample(all_scenarios, int(len(all_scenarios)*0.6))
    # print(train_val)
    test = list(set(all_scenarios)-set(train_val))

    val = random.sample(train_val, int(0.1*len(all_scenarios)))
    # print(val)
    train = list(set(train_val)-set(val))
    data = ''
    with open('./train.txt', 'w') as f:
        for i in train: data += i+'\n'
        f.write(data)
    data = ''
    with open('./val.txt', 'w') as f:
        for i in val: data += i+'\n'
        f.write(data)
    data = ''
    with open('./test.txt', 'w') as f:
        for i in test: data += i+'\n'
        f.write(data)


    print(len(train) +len(val)+len(test))

def Infor_statistics():
    with open('./info.json','r') as f:
        a = json.load(f)
    data = []
    number = np.zeros(5)
    cat = ['0~50', '50~100', '100~150', '150~200', '200~400']
    for k, v in a.items():
        data.append(v)
        if v in range(0,50):
            number[0]+=1
        elif v in range(50,100):
            number[1]+=1
        elif v in range(100,150):
            number[2]+=1
        elif v in range(150, 200):
            number[3] += 1
        elif v in range(200, 400):
            number[4] += 1
    data = np.array(data)
    import pdb
    pdb.set_trace()

    print(data, data.sum())
    draw_hist(data)



def draw_hist(lenths):
    data = lenths

    bins = np.linspace(min(data), 400, 10)
    bins = [0,100, 200, 400]
    pl.hist(data, bins)

    pl.xlabel('Number of people')

    pl.ylabel('Number of occurences')

    pl.title('Frequency distribution of number of people in SensorCrowd (634 Seq)')

    pl.show()



if __name__ =='__main__':
    SensorCrowd()
    Infor_statistics()
    # SENSE_train_val_test()