import os
from collections import  defaultdict
import  os.path as osp
from train import compute_metrics_all_scenes
def tracking_to_crowdflow():
    method = 'HT21_25'# 'HeadHunter_result' 'fairmot_head''PHDTT''HT21_25 '' '
    Root = os.path.join('D:\Crowd_tracking/HeadHunter',method)
    Root = os.path.join('/media/D/GJY/ht/CVPR2022/tracking_results', method)
    gt_root = '/media/E/ht/dataset/HT21/train'
    scenes = os.listdir(Root)
    scenes.sort()
    print(scenes)
    interval = 1
    scenes_pred_dict = []
    scenes_gt_dict = []
    each_scene_num  = []

    pred = defaultdict(list)
    gts = defaultdict(list)
    for _, sub_scene in enumerate(scenes,0):
        # if _>0:
        #     break
        path = os.path.join(Root,sub_scene)
        id_list  = []
        with open(path, 'r') as f:
            lines = f.readlines()
            for vi, line in enumerate(lines, 0):
                line = line.strip().split(',')
                img_id = int(line[0])
                tmp_id = int(line[1])
                if img_id % interval == 0:
                    pred[img_id].append(tmp_id)
                    id_list.append(tmp_id)
        if os.path.exists(osp.join(gt_root, sub_scene.split('.')[0], 'gt', 'gt.txt')):
            with open(osp.join(gt_root, sub_scene.split('.')[0], 'gt', 'gt.txt'), 'r') as f:
                lines = f.readlines()
                for lin in lines:
                    lin_list = [float(i) for i in lin.rstrip().split(',')]
                    ind = int(lin_list[0])
                    gts[ind].append(int(lin_list[1]))
        id = set(id_list)
        each_scene_num.append(len(id))
        print(each_scene_num)
        print("train:%.1f, test:%.1f"%(sum(each_scene_num[:4]), sum(each_scene_num[4:])))

        pred_dict = {'id': sub_scene, 'time': img_id, 'first_frame': 0, 'inflow': [], 'outflow': []}
        gt_dict = {'id': sub_scene, 'time': img_id, 'first_frame': 0, 'inflow': [], 'outflow': []}


        img_num =len(gts.keys())
        for img_id, ids in gts.items():
            if img_id>img_num-interval:
                break
            img_id_b = img_id+interval
            pre_ids,pre_ids_b = pred[img_id],pred[img_id_b]
            gt_ids,gt_ids_b = ids, gts[img_id_b]

            if img_id == 1:
                pred_dict['first_frame'] = len(pre_ids)
                gt_dict['first_frame'] = len(gt_ids)
            # import pdb
            # pdb.set_trace()

            # if (img_id-1) % interval ==0 or img_num== 0:
            pre_inflow =set(pre_ids_b)-set(pre_ids)
            pre_outflow = set(pre_ids)-set(pre_ids_b)

            gt_inflow = set(gt_ids_b)-set(gt_ids)
            gt_outflow = set(gt_ids)-set(gt_ids_b)
            pred_dict['inflow'].append(len(pre_inflow))
            pred_dict['outflow'].append(len(pre_outflow))
            gt_dict['inflow'].append(len(gt_inflow))
            gt_dict['outflow'].append(len(gt_outflow))
        print(pred_dict)
        print(gt_dict)
        scenes_pred_dict.append(pred_dict)
        scenes_gt_dict.append(gt_dict)
    MAE, MSE, WRAE, MIAE, MOAE, cnt_result = compute_metrics_all_scenes(scenes_pred_dict, scenes_gt_dict, interval)
    print(MAE, MSE, WRAE, MIAE, MOAE, cnt_result)





def id_counting():
    Root = 'D:/Crowd_tracking/dataset/HT21/train'
    scenes = os.listdir(Root)
    each_scene_num  = []
    for i in scenes:
        path = os.path.join(Root,i,'gt/gt.txt')
        id_list  = []
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                id_list.append(int(line.strip().split(',')[1]))
            id = set(id_list)
            each_scene_num.append(len(id))
    print(each_scene_num, sum(each_scene_num[:4]), sum(each_scene_num[4:]))

if __name__ == '__main__':
    import torch
    #PHDTT
    gt_pre_flow_cnt = torch.tensor([[133.,737.,734.,1040.,321.],[380.,4530.,5528.,1531.,1648.]]).transpose(0,1)
    #HeadHunter
    gt_pre_flow_cnt = torch.tensor([[133., 737., 734., 1040., 321.], [307.,  2145., 2556., 1531., 888.,]]).transpose(0, 1)
    #FairMOT
    gt_pre_flow_cnt = torch.tensor([[133., 737., 734., 1040., 321.], [366., 3215., 7011., 2626., 2337., ]]).transpose(0, 1)
    #LOI
    gt_pre_flow_cnt = torch.tensor([[133., 737., 734., 1040., 321.],[72.4 ,493.1 ,275.3 ,409.2,189.8]]).transpose(0, 1)
    # Hungarian s=10
    gt_pre_flow_cnt = torch.tensor([[ 129.,  133.],
        [ 421.,  737.],
        [ 332.,  734.],
        [ 331., 1040.],


        [ 185.,  321.]])


    # Hungarian s=12
    gt_pre_flow_cnt = torch.tensor([[ 188.,  133.],
        [ 779.,  737.],
        [1069.,  734.],
        [ 772., 1040.],
        [ 324.,  321.]])
    # Hungarian s=15
    gt_pre_flow_cnt = torch.tensor([[ 298.,  133.],
        [1833.,  737.],
        [1921.,  734.],
        [1641., 1040.],
        [ 752.,  321.]])
    # Head 10
    gt_pre_flow_cnt = torch.tensor([[133., 737., 734., 1040., 321.], [246, 814, 466, 686, 453]]).transpose(0, 1)

    gt_pre_flow_cnt = torch.tensor([[133., 737., 734., 1040., 321.], [198, 636, 219, 458, 324]]).transpose(0, 1)
    # # Fairmot
    # gt_pre_flow_cnt = torch.tensor([[133., 737., 734., 1040., 321.], [202, 1736, 2871, 1429, 1003]]).transpose(0, 1)
    # #PHDTT
    #
    # gt_pre_flow_cnt = torch.tensor([[133., 737., 734., 1040., 321.], [ 215, 2215, 3252, 1154, 851]]).transpose(0, 1)
    time = torch.tensor([585.,2080.,1000.,1050.,1008.])
    MAE =  torch.mean(torch.abs(gt_pre_flow_cnt[:,0] - gt_pre_flow_cnt[:,1]))
    MSE = torch.mean((gt_pre_flow_cnt[:, 0] - gt_pre_flow_cnt[:, 1])**2).sqrt()
    WRAE = torch.sum(torch.abs(gt_pre_flow_cnt[:,0] - gt_pre_flow_cnt[:,1])/gt_pre_flow_cnt[:,0]*(time/(time.sum()+1e-10)))*100

    print(MAE, MSE, WRAE)

    tracking_to_crowdflow()