import argparse
import os
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as standard_transforms

import pprint
import tqdm

from model.crowd_ldc import Crowd_locator
from eval.utils import hungarian,AverageMeter, AverageCategoryMeter
from scipy import spatial as ss
from config import cfg
import numpy as np
from PIL import Image, ImageOps
import  cv2
import pdb

dataset = 'NWPU'
saved_img_path = './xian_pred/'

GPU_ID = '2,3'
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
torch.backends.cudnn.benchmark = True

model_path = '../exp/10-30_18-05_NWPU_HR_Net_1e-05_Pixel(0.802)_revised_grad/ep_241_F1_0.802_Pre_0.841_Rec_0.766_mae_55.6_mse_330.9.pth'

if not os.path.exists(saved_img_path):
    os.makedirs(saved_img_path)

def main():

    file_list = os.listdir('./xian')
    print(file_list)
    test(file_list)


def test(file_list):
    if dataset == 'NWPU':
        mean_std = ([0.446139603853, 0.409515678883, 0.395083993673], [0.288205742836, 0.278144598007, 0.283502370119])
    if dataset == 'SHHB':
        mean_std = ([0.452016860247, 0.447249650955, 0.431981861591], [0.23242045939, 0.224925786257, 0.221840232611])
    if dataset == 'SHHA':
        mean_std = ([0.410824894905, 0.370634973049, 0.359682112932], [0.278580576181, 0.26925137639, 0.27156367898])

    img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])

    pil_to_tensor = standard_transforms.ToTensor()

    model = Crowd_locator(cfg, pretrained=True)
    if model_path is not None:
        print(model_path)
        model.load_state_dict(torch.load(model_path))

    model.eval()
    num_classes = 6
    metrics_s = {'tp': AverageMeter(), 'fp': AverageMeter(), 'fn': AverageMeter(), 'tp_c': AverageCategoryMeter(num_classes),
                 'fn_c': AverageCategoryMeter(num_classes)}
    metrics_l = {'tp': AverageMeter(), 'fp': AverageMeter(), 'fn': AverageMeter(), 'tp_c': AverageCategoryMeter(num_classes),
                 'fn_c': AverageCategoryMeter(num_classes)}
    cnt_errors = {'mae': AverageMeter(), 'mse': AverageMeter(), 'nae': AverageMeter()}
    file_list = tqdm.tqdm(file_list)
    for infos in file_list:
        imgname = os.path.join('./xian', infos)
        img = Image.open(imgname)
        # img = img.resize((768,1024))
        vis_img = np.array(img)
        w,h = img.size

        if img.mode == 'L':
            img = img.convert('RGB')
        img = img_transform(img)[None, :, :, :]
        slice_h, slice_w = 1080,1920
        with torch.no_grad():
            img = Variable(img).cuda()
            b, c, h, w = img.shape
            [pred_threshold, pred_map, __] = [i.cpu() for i in model(img, mask_gt=None, mode='val')]
            a = torch.ones_like(pred_map)
            b = torch.zeros_like(pred_map)
            binar_map = torch.where(pred_map >= pred_threshold, a, b)

            pred_data, boxes = get_boxInfo_from_Binar_map(binar_map.numpy())

            ##====================get a binar map for visiual ===
            # binar_color_map = cv2.applyColorMap((255 * binar_map.numpy()).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)
            # pil_binar = Image.fromarray(cv2.cvtColor(binar_color_map, cv2.COLOR_BGR2RGB))
            # pil_binar.save('3152_binar_map.jpg', quality=95, subsampling=0)
            for i in pred_data['boxes']:
                x, y, width, heigt, __ = i
                # print(x, y , width, heigt )
                r_large  = int(np.sqrt (width * width + heigt * heigt) / 2)
                cv2.circle(vis_img, (int(x+width/2), int(y+heigt/2.0)), r_large, (0, 255, 0), 2)  # tp: green
            font=cv2. FONT_HERSHEY_SIMPLEX
            cv2.putText(vis_img, 'pred num:'+str(pred_data['num']), (760, 100), font, 2, (0, 0, 255), 3, cv2.LINE_AA)  # 画文字

            save_path = os.path.join(saved_img_path,infos)
            print(save_path)
            cv2.imwrite(save_path,vis_img)
            #=======================================================

            # print(pred_data)
            print(' pred_num: %d, image:%s' %  ( pred_data['num'], infos))


def get_boxInfo_from_Binar_map(Binar_numpy, min_area=3):
    Binar_numpy = Binar_numpy.squeeze().astype(np.uint8)
    assert Binar_numpy.ndim == 2
    cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(Binar_numpy, connectivity=4)  # centriod (w,h)

    boxes = stats[1:, :]
    points = centroids[1:, :]
    index = (boxes[:, 4] >= min_area)
    boxes = boxes[index]
    points = points[index]
    pre_data = {'num': len(points), 'boxes': boxes}
    return pre_data, boxes

def read_box_gt(box_gt_file):
    gt_data = {}
    with open(box_gt_file) as f:
        for line in f.readlines():
            line = line.strip().split(' ')

            line_data = [int(i) for i in line]
            idx, num = [line_data[0], line_data[1]]
            points_r = []
            if num > 0:
                points_r = np.array(line_data[2:]).reshape(((len(line) - 2) // 5, 5))
                gt_data[idx] = {'num': num, 'points': points_r[:, 0:2], 'sigma': points_r[:, 2:4], 'level': points_r[:, 4]}
            else:
                gt_data[idx] = {'num': 0, 'points': [], 'sigma': [], 'level': []}

    return gt_data
def eval_metrics(num_classes, pred_data, gt_data_T):
    # print(gt_data_T)
    if gt_data_T['num']>0:
        gt_data = {'num':gt_data_T['num'], 'points':gt_data_T['points'],\
                   'sigma':gt_data_T['sigma'], 'level':gt_data_T['level']}
    else:
        gt_data = {'num':0, 'points':[],'sigma':[], 'level':[]}

    # print(gt_data)
    tp_s,fp_s,fn_s,tp_l,fp_l,fn_l = [0,0,0,0,0,0]
    tp_c_s = np.zeros([num_classes])
    fn_c_s = np.zeros([num_classes])
    tp_c_l = np.zeros([num_classes])
    fn_c_l = np.zeros([num_classes])

    if gt_data['num'] ==0 and pred_data['num'] !=0:
        pred_p =  pred_data['points']
        fp_pred_index = np.array(range(pred_p.shape[0]))
        fp_s = fp_pred_index.shape[0]
        fp_l = fp_pred_index.shape[0]

    if pred_data['num'] ==0 and gt_data['num'] !=0:
        gt_p = gt_data['points']
        level = gt_data['level']

        fn_gt_index = np.array(range(gt_p.shape[0]))
        fn_s = fn_gt_index.shape[0]
        fn_l = fn_gt_index.shape[0]
        for i_class in range(num_classes):
            fn_c_s[i_class] = (level[fn_gt_index]==i_class).sum()
            fn_c_l[i_class] = (level[fn_gt_index]==i_class).sum()

    if gt_data['num'] !=0 and pred_data['num'] !=0:
        pred_p =  pred_data['points']
        gt_p = gt_data['points']
        sigma_s = gt_data['sigma'][:,0]
        sigma_l = gt_data['sigma'][:,1]
        level = gt_data['level']

        # dist
        dist_matrix = ss.distance_matrix(pred_p,gt_p,p=2)
        match_matrix = np.zeros(dist_matrix.shape,dtype=bool)

        # sigma_s and sigma_l
        tp_s,fp_s,fn_s,tp_c_s,fn_c_s = compute_metrics(dist_matrix,match_matrix,pred_p.shape[0],gt_p.shape[0],sigma_s,level)
        tp_l,fp_l,fn_l,tp_c_l,fn_c_l = compute_metrics(dist_matrix,match_matrix,pred_p.shape[0],gt_p.shape[0],sigma_l,level)
    return tp_s,fp_s,fn_s,tp_c_s,fn_c_s, tp_l,fp_l,fn_l,tp_c_l,fn_c_l


def compute_metrics(dist_matrix, match_matrix, pred_num, gt_num, sigma, level):
    num_classes = 6
    for i_pred_p in range(pred_num):
        pred_dist = dist_matrix[i_pred_p, :]
        match_matrix[i_pred_p, :] = pred_dist <= sigma

    tp, assign = hungarian(match_matrix)
    fn_gt_index = np.array(np.where(assign.sum(0) == 0))[0]
    tp_pred_index = np.array(np.where(assign.sum(1) == 1))[0]
    tp_gt_index = np.array(np.where(assign.sum(0) == 1))[0]
    fp_pred_index = np.array(np.where(assign.sum(1) == 0))[0]
    level_list = level[tp_gt_index]

    tp = tp_pred_index.shape[0]
    fp = fp_pred_index.shape[0]
    fn = fn_gt_index.shape[0]

    tp_c = np.zeros([num_classes])
    fn_c = np.zeros([num_classes])

    for i_class in range(num_classes):
        tp_c[i_class] = (level[tp_gt_index] == i_class).sum()
        fn_c[i_class] = (level[fn_gt_index] == i_class).sum()

    return tp, fp, fn, tp_c, fn_c


def save_visual_img(saved_img_path,ori_img_path, gt_data, pred_data ):
    img_id = ori_img_path.split('/')[-1].split('.')[0]

    gt_p, pred_p, fn_gt_index, tp_pred_index, fp_pred_index, ap, ar = [], [], [], [], [], [], []

    if gt_data['num'] == 0 and pred_data['num'] != 0:
        pred_p = pred_data['points'].astype('int')
        fp_pred_index = np.array(range(pred_p.shape[0]))
        ap = 0
        ar = 0

    if pred_data['num'] == 0 and gt_data['num'] != 0:
        gt_p = gt_data['points']
        fn_gt_index = np.array(range(gt_p.shape[0]))
        sigma_l = gt_data['sigma'][:, 1]
        ap = 0
        ar = 0

    if gt_data['num'] != 0 and pred_data['num'] != 0:
        pred_p = pred_data['points'].astype('int')
        gt_p = gt_data['points']
        sigma_l = gt_data['sigma'][:, 1]
        level = gt_data['level']

        # dist
        dist_matrix = ss.distance_matrix(pred_p, gt_p, p=2)
        match_matrix = np.zeros(dist_matrix.shape, dtype=bool)
        for i_pred_p in range(pred_p.shape[0]):
            pred_dist = dist_matrix[i_pred_p, :]
            match_matrix[i_pred_p, :] = pred_dist <= sigma_l

        # hungarian outputs a match result, which may be not optimal.
        # Nevertheless, the number of tp, fp, tn, fn are same under different match results
        # If you need the optimal result for visualzation,
        # you may treat it as maximum flow problem.
        tp, assign = hungarian(match_matrix)
        fn_gt_index = np.array(np.where(assign.sum(0) == 0))[0]
        tp_pred_index = np.array(np.where(assign.sum(1) == 1))[0]
        tp_gt_index = np.array(np.where(assign.sum(0) == 1))[0]
        fp_pred_index = np.array(np.where(assign.sum(1) == 0))[0]

        pre = tp_pred_index.shape[0] / (tp_pred_index.shape[0] + fp_pred_index.shape[0] + 1e-20)
        rec = tp_pred_index.shape[0] / (tp_pred_index.shape[0] + fn_gt_index.shape[0] + 1e-20)

        img = cv2.imread(ori_img_path)  # bgr
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)

        point_r_value = 5
        thickness = 3
        if gt_data['num'] != 0:
            for i in range(gt_p.shape[0]):
                if i in fn_gt_index:
                    cv2.circle(img, (gt_p[i][0], gt_p[i][1]), point_r_value, (0, 0, 255), -1)  # fn: red
                    cv2.circle(img, (gt_p[i][0], gt_p[i][1]), sigma_l[i], (0, 0, 255), thickness)  #
                else:
                    cv2.circle(img, (gt_p[i][0], gt_p[i][1]), sigma_l[i], (0, 255, 0), thickness)  # gt: green
        if pred_data['num'] != 0:
            for i in range(pred_p.shape[0]):
                if i in tp_pred_index:
                    cv2.circle(img, (pred_p[i][0], pred_p[i][1]), point_r_value, (0, 255, 0), -1)  # tp: green
                else:
                    cv2.circle(img, (pred_p[i][0], pred_p[i][1]), point_r_value * 2, (255, 0, 255), -1)  # fp: Magenta

        cv2.imwrite(saved_img_path + '/' + str(img_id) + '_pre_' + str(pre)[0:6] + '_rec_' + str(rec)[0:6]+
                    '_gt_'+str(gt_data['num'])+'_pred_'+str(pred_data['num']) + '.jpg', img)


if __name__ == '__main__':
    main()




