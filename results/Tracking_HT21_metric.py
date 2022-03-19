import os
from collections import  defaultdict
import  os.path as osp
from train import compute_metrics_all_scenes
import numpy as np
import cv2
from PIL import  Image
def tracking_to_crowdflow():
    method = 'HT21_10'# 'HeadHunter_result' 'fairmot_head''PHDTT' 'PHDTT'

    # Root = os.path.join('D:\Crowd_tracking/HeadHunter',method)
    Root = os.path.join('/media/E/ht/HeadHunter--T-master/results', method)
    # gt_root = 'D:\Crowd_tracking/dataset/HT21/train'
    scenes = sorted(os.listdir(Root))
    print(scenes)
    scenes_pred_dict = []
    scenes_gt_dict = []
    all_sum  = []


    for _, i in enumerate(scenes,0):
        # if _>0:
        #     break
        pred = defaultdict(list)
        gts = defaultdict(list)

        path = os.path.join(Root,i)
        id_list  = []
        with open(path, 'r') as f:
            lines = f.readlines()
            for vi, line in enumerate(lines, 0):
                line = line.strip().split(',')
                img_id = int(line[0])
                tmp_id = int(line[1])
                pred[img_id].append(tmp_id)
                id_list.append(tmp_id)

        # with open(osp.join(gt_root, i.split('.')[0], 'gt', 'gt.txt'), 'r') as f:
        #     lines = f.readlines()
        #     for lin in lines:
        #         lin_list = [float(i) for i in lin.rstrip().split(',')]
        #         ind = int(lin_list[0])
        #         gts[ind].append(int(lin_list[1]))
        # print(id_list)
        id = set(id_list)
        all_sum.append(len(id))
        print(all_sum, sum(all_sum[:5]), sum(all_sum[5:]))


    gt_pre_flow_cnt = torch.cat([torch.tensor([[133., 737., 734., 1040., 321.]]), torch.tensor(all_sum)[None]]).transpose(0, 1)
    print(gt_pre_flow_cnt)
    time = torch.tensor([585.,2080.,1000.,1050.,1008.])
    MAE =  torch.mean(torch.abs(gt_pre_flow_cnt[:,0] - gt_pre_flow_cnt[:,1]))
    MSE = torch.mean((gt_pre_flow_cnt[:, 0] - gt_pre_flow_cnt[:, 1])**2).sqrt()
    WRAE = torch.sum(torch.abs(gt_pre_flow_cnt[:,0] - gt_pre_flow_cnt[:,1])/gt_pre_flow_cnt[:,0]*(time/(time.sum()+1e-10)))*100
    print(MAE, MSE, WRAE)


    #     pred_dict = {'id': i, 'time': len(lines), 'first_frame': 0, 'inflow': [], 'outflow': []}
    #     gt_dict = {'id': i, 'time': len(lines), 'first_frame': 0, 'inflow': [], 'outflow': []}
    #
    #     interval = 75
    #     img_num =len(gts.keys())
    #     print(img_num)
    #     for img_id, ids in gts.items():
    #         if img_id>img_num-interval:
    #             break
    #
    #         img_id_b = img_id+interval
    #
    #         pre_ids,pre_ids_b = pred[img_id],pred[img_id_b]
    #         gt_ids,gt_ids_b = ids, gts[img_id_b]
    #
    #         if img_id == 1:
    #             pred_dict['first_frame'] = len(pre_ids)
    #             gt_dict['first_frame'] = len(gt_ids)
    #         # import pdb
    #         # pdb.set_trace()
    #
    #         # if (img_id-1) % interval ==0 or img_num== 0:
    #         pre_inflow =set(pre_ids_b)-set(pre_ids)
    #         pre_outflow = set(pre_ids)-set(pre_ids_b)
    #
    #         gt_inflow = set(gt_ids_b)-set(gt_ids)
    #         gt_outflow = set(gt_ids)-set(gt_ids_b)
    #         pred_dict['inflow'].append(len(pre_inflow))
    #         pred_dict['outflow'].append(len(pre_outflow))
    #         gt_dict['inflow'].append(len(gt_inflow))
    #         gt_dict['outflow'].append(len(gt_outflow))
    #     # print(pred_dict, gt_dict)
    #     scenes_pred_dict.append(pred_dict)
    #     scenes_gt_dict.append(gt_dict)
    # MAE, MSE, WRAE, MIAE, MOAE, cnt_result = compute_metrics_all_scenes(scenes_pred_dict, scenes_gt_dict, interval)
    # print(MAE, MSE, WRAE, MIAE, MOAE, cnt_result)


def id_counting():
    Root = 'D:/Crowd_tracking/dataset/HT21/train'
    scenes = os.listdir(Root)
    all_sum  = []
    for i in scenes:
        path = os.path.join(Root,i,'gt/gt.txt')
        id_list  = []
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                id_list.append(int(line.strip().split(',')[1]))
            id = set(id_list)
            all_sum.append(len(id))
    print(all_sum, sum(all_sum[:4]), sum(all_sum[4:]))

if __name__ == '__main__':
    import torch
    #PHDTT
    # gt_pre_flow_cnt = torch.tensor([[133.,737.,734.,1040.,321.],[380.,4530.,5528.,1531.,1648.]]).transpose(0,1)
    # #HeadHunter
    gt_pre_flow_cnt = torch.tensor([[133., 737., 734., 1040., 321.], [307.,  2145., 2556., 1531., 888.,]]).transpose(0, 1)
    #
    # #LOI
    # gt_pre_flow_cnt = torch.tensor([[133., 737., 734., 1040., 321.],[72.4 ,493.1 ,275.3 ,409.2,189.8]]).transpose(0, 1)
    # # Hungarian s=10
    # gt_pre_flow_cnt = torch.tensor([[ 129.,  133.],
    #     [ 421.,  737.],
    #     [ 332.,  734.],
    #     [ 331., 1040.],
    #     [ 185.,  321.]])
    #
    #
    # # Hungarian s=12
    # gt_pre_flow_cnt = torch.tensor([[ 188.,  133.],
    #     [ 779.,  737.],
    #     [1069.,  734.],
    #     [ 772., 1040.],
    #     [ 324.,  321.]])
    # # Hungarian s=15
    # gt_pre_flow_cnt = torch.tensor([[ 298.,  133.],
    #     [1833.,  737.],
    #     [1921.,  734.],
    #     [1641., 1040.],
    #     [ 752.,  321.]])
    #
    # #Tracking
    # gt_pre_flow_cnt = torch.tensor([[133., 737., 734., 1040., 321.], [284., 1364., 1435., 1975., 539., ]]).transpose(0, 1)

    ## SSIC sampling
    # gt_pre_flow_cnt = torch.tensor([[133., 737., 734., 1040., 321.], [432.6235237121582, 4244.325263977051, 2307.327682495117, 2219.3844146728516, 1355.9616165161133]]).transpose(0, 1)
    # gt_pre_flow_cnt = torch.tensor([[133., 737., 734., 1040., 321.],[83.13096618652344, 216.19476318359375, 224.47157287597656, 174.38177490234375, 118.87664794921875]]).transpose(0,1)
    #
    time = torch.tensor([585.,2080.,1000.,1050.,1008.])
    MAE =  torch.mean(torch.abs(gt_pre_flow_cnt[:,0] - gt_pre_flow_cnt[:,1]))
    MSE = torch.mean((gt_pre_flow_cnt[:, 0] - gt_pre_flow_cnt[:, 1])**2).sqrt()
    WRAE = torch.sum(torch.abs(gt_pre_flow_cnt[:,0] - gt_pre_flow_cnt[:,1])/gt_pre_flow_cnt[:,0]*(time/(time.sum()+1e-10)))*100

    print(MAE, MSE, WRAE)

    tracking_to_crowdflow()
