import os
import sys
import numpy as np
from scipy import spatial as ss
import pdb

import cv2
from .utils import hungarian,read_pred_and_gt,AverageMeter,AverageCategoryMeter

# gt_file = 'val_gt_loc.txt'
# pred_file = 'TinyFaces_loc_0.8_0.3.txt'

flagError = False
id_std = [i for i in range(3110,3610,1)]
id_std[59] = 3098
num_classes = 6


def compute_metrics(dist_matrix,match_matrix,pred_num,sigma):
    for i_pred_p in range(pred_num):
        pred_dist = dist_matrix[i_pred_p,:]
        match_matrix[i_pred_p,:] = pred_dist<=sigma

    tp, assign = hungarian(match_matrix)
    fn_gt_index = np.array(np.where(assign.sum(0)==0))[0]

    fp_pred_index = np.array(np.where(assign.sum(1)==0))[0]

    # tp_pred_index = np.array(np.where(assign.sum(1)==1))[0]
    # tp_gt_index = np.array(np.where(assign.sum(0)==1))[0]

    tp_pred_index, tp_gt_index = np.where(assign==1)

    tp = tp_pred_index.shape[0]
    fp = fp_pred_index.shape[0]
    fn = fn_gt_index.shape[0]

    #
    # import  pdb
    # pdb.set_trace()
    return tp,fp,fn,tp_pred_index,fp_pred_index ,tp_gt_index, fn_gt_index




def associate_pred2gt_point(pred_data, gt_data):
    # import pdb
    # pdb.set_trace()
    pred_p = pred_data['points'].cpu().numpy()
    gt_p = gt_data['points'].cpu().numpy()
    gt_sigma = gt_data['sigma'].cpu().numpy()
    if gt_p.shape[0]>0:
        gt_data = {'num':gt_p.shape[0], 'points':gt_p,'sigma':gt_sigma}
    else:
        gt_data = {'num':0, 'points':[],'sigma':[]}

    tp_l,fp_l,fn_l = [0,0,0]
    tp_pred_index,tp_gt_index = [],[]
    if gt_data['num'] ==0 and pred_p.shape[0] !=0:
        fp_pred_index = np.array(range(pred_p.shape[0]))
        fp_l = fp_pred_index.shape[0]

    if pred_p.shape[0] ==0 and gt_data['num'] !=0:
        gt_p = gt_data['points']
        fn_gt_index = np.array(range(gt_p.shape[0]))
        fn_l = fn_gt_index.shape[0]


    if gt_data['num'] !=0 and pred_p.shape[0] !=0:
        gt_p = gt_data['points']
        sigma = gt_data['sigma']

        # dist
        dist_matrix = ss.distance_matrix(pred_p,gt_p,p=2)
        match_matrix = np.zeros(dist_matrix.shape,dtype=bool)

        # sigma_s and sigma_l
        tp_l,fp_l,fn_l,tp_pred_index,fp_pred_index ,tp_gt_index, fn_gt_index = compute_metrics(dist_matrix,match_matrix,pred_p.shape[0],sigma)
    return tp_pred_index,tp_gt_index




def associate_pred2gt_point_vis(pred_data, gt_data, gt_diff_idx):
    # import pdb
    # pdb.set_trace()
    pred_p = pred_data.cpu().numpy()
    gt_p = gt_data['points'].cpu().numpy()[gt_diff_idx]
    gt_sigma = gt_data['sigma'].cpu().numpy()[gt_diff_idx]
    if gt_p.shape[0]>0:
        gt_data = {'num':gt_p.shape[0], 'points':gt_p,'sigma':gt_sigma}
    else:
        gt_data = {'num':0, 'points':[],'sigma':[]}

    tp_l,fp_l,fn_l = [0,0,0]
    tp_pred_index,tp_gt_index,fp_pred_index,fn_gt_index = [],[],[],[]
    if gt_data['num'] ==0 and pred_p.shape[0] !=0:
        fp_pred_index = np.array(range(pred_p.shape[0]))
        fp_l = fp_pred_index.shape[0]
        fn_gt_index = np.array([])
    if pred_p.shape[0] ==0 and gt_data['num'] !=0:
        gt_p = gt_data['points']
        fn_gt_index = np.array(range(gt_p.shape[0]))
        fn_l = fn_gt_index.shape[0]
        fp_pred_index = np.array([])

    if gt_data['num'] !=0 and pred_p.shape[0] !=0:
        gt_p = gt_data['points']
        sigma = gt_data['sigma']

        # dist
        dist_matrix = ss.distance_matrix(pred_p,gt_p,p=2)
        match_matrix = np.zeros(dist_matrix.shape,dtype=bool)

        # sigma_s and sigma_l
        tp_l,fp_l,fn_l,tp_pred_index,fp_pred_index ,tp_gt_index, fn_gt_index = compute_metrics(dist_matrix,match_matrix,pred_p.shape[0],sigma)
    return tp_pred_index,fp_pred_index ,tp_gt_index, fn_gt_index

if __name__ == '__main__':
    eval_metrics()
