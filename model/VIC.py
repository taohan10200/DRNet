from model.VGG.VGG16_FPN import VGG16_FPN
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from model.PreciseRoIPooling.pytorch.prroi_pool.functional import prroi_pool2d
from .MatchTool.compute_metric import associate_pred2gt_point
from misc.layer import Gaussianlayer, Point2Mask
from .optimal_transport_layer import Optimal_Transport_Layer
from model.points_from_den import *
from collections import  OrderedDict
import copy
class Video_Individual_Counter(nn.Module):
    def __init__(self, cfg, cfg_data, pretrained=True):
        super(Video_Individual_Counter, self).__init__()
        self.cfg = cfg
        self.dataset_cfg = cfg_data
        self.radius = self.cfg.ROI_RADIUS
        self.feature_scale = 1/4.
        OT_config = {
            'feature_dim': cfg.FEATURE_DIM,
            'sinkhorn_iterations':cfg.sinkhorn_iterations,
            'matched_threshold': 0.2
        }

        if cfg.NET == 'VGG16_FPN':
            self.Extractor = VGG16_FPN()
        else:
            raise  Exception("The backbone is out of setting, Please chose HR_Net or VGG16_FPN")

        # import pdb
        # pdb.set_trace()
        if len(cfg.GPU_ID) >=1:
            self.Extractor = torch.nn.DataParallel(self.Extractor).cuda()

        self.Gaussian = Gaussianlayer().cuda()
        self.gaussian_maximum=self.Gaussian.gaussian.gkernel.weight.max()
        self.Matching_Layer =Optimal_Transport_Layer(OT_config).cuda()

        self.device = torch.cuda.current_device()
        self.get_ROI_and_MatchInfo = get_ROI_and_MatchInfo( self.dataset_cfg .TRAIN_SIZE, self.radius, feature_scale=self.feature_scale)

    @property
    def loss(self):
        return  self.counting_mse_loss, self.batch_match_loss, self.batch_hard_loss, self.batch_norm_loss

    def KPI_cal(self,match_gt,scores,target_pair):
        self.match_pairs_cnt = match_gt['un_a'].size(0) + match_gt['un_b'].size(0) + match_gt['a2b'].size(0)
        scores[-1, -1] = 0
        max0_idx = scores.max(1).indices  # the points in a that have matched in b, return b's index,
        max1_idx = scores.max(0).indices  # the points in b that have matched in b, return a's index
        if match_gt['a2b'].size(0)>0:
            pred_a = max0_idx[match_gt['a2b'][:, 0]]
            pred_b = max1_idx[match_gt['a2b'][:, 1]]
            a_id = torch.cat([target_pair[0]['person_id'],torch.tensor([-1]).to(self.device)])
            b_id = torch.cat([target_pair[1]['person_id'],torch.tensor([-2]).to(self.device)])
            TP = (b_id[pred_a] == a_id[pred_b]).sum()  # correct matched person pairs in two frames
        else:
            TP=0
        TN = (max0_idx[match_gt['un_a']] == scores.size(1) - 1).sum() + (max1_idx[match_gt['un_b']] == scores.size(0) - 1).sum()
        self.correct_pairs_cnt = TP + TN
        return  self.match_pairs_cnt, self.correct_pairs_cnt,TP
    def forward(self, img, target=None):
        for i in range(len(target)):
            for key,data in target[i].items():
                if torch.is_tensor(data):
                    target[i][key]=data.to(self.device)

        img = torch.stack(img, 0)
        img_pair_num = img.size(0)//2
        assert   img.size(0)%2 ==0
        features, pre_map = self.Extractor(img)

        dot_map = torch.zeros_like(pre_map)
        for i, data in enumerate(target):
            points = data['points'].long()
            dot_map[i, 0, points[:, 1], points[:, 0]] = 1
        gt_den = self.Gaussian(dot_map)

        assert pre_map.size() == gt_den.size()
        self.counting_mse_loss = F.mse_loss(pre_map, gt_den * self.dataset_cfg.DEN_FACTOR)
        pre_map = pre_map/self.dataset_cfg.DEN_FACTOR
        #==============================================================

        matched_results = {'matches0': [], 'matches1': [],'matching_scores0': [],'matching_scores1': [], 'gt_matched': [],
                           'gt_count_diff': 0,'pre_count_diff': 0}

        self.batch_match_loss = torch.tensor(0.).to(self.device)
        self.batch_hard_loss = torch.tensor(0.).to(self.device)
        self.batch_norm_loss = torch.tensor(0.).to(self.device)
        match_loss =[]
        hard_loss = []
        norm_loss = []
        match_pairs_cnt=torch.tensor(0.).to(self.device)
        correct_pairs_cnt=torch.tensor(0.).to(self.device)
        TP_cnt = torch.tensor(0.).to(self.device)

        #===================== for the second matching =============================
        pre_data = local_maximum_points(pre_map.detach(),self.gaussian_maximum,radius=self.radius)
        precount_in_batch = [(pre_data['points'][:, 0] == i).sum() for i in range(pre_map.size(0))]
        pre_points = torch.split(pre_data['points'], precount_in_batch)
        # ===================== ======================= =============================

        for pair_idx in range(img_pair_num):
            # ======generate the points of interest for matching========
            count_in_pair=[target[pair_idx * 2]['points'].size(0), target[pair_idx * 2+1]['points'].size(0)]
            if (np.array(count_in_pair) > 0).all() and (np.array(count_in_pair) < 4000).all():
                # In the training phase, they are labeled points
                match_gt,pois = self.get_ROI_and_MatchInfo(target[pair_idx * 2], target[pair_idx * 2+1],noise='ab')
                poi_features = prroi_pool2d(features[pair_idx*2:pair_idx*2+2], pois, 1, 1, self.feature_scale)
                poi_features=  poi_features.squeeze(2).squeeze(2)[None].transpose(1,2) # [batch, dim, num_features]
                mdesc0, mdesc1 = torch.split(poi_features, count_in_pair,dim=2)


                scores, indices0, indices1, mscores0, mscores1 = self.Matching_Layer(mdesc0, mdesc1,match_gt) # batch,dim,num
                match_loss_, hard_loss_ = self.Matching_Layer.loss
                match_loss.append(match_loss_)
                hard_loss.append(hard_loss_)

                tmp_gt_person, tmp_correct_pairs,TP = self.KPI_cal(match_gt, scores.clone(), target[pair_idx * 2:pair_idx * 2+2])
                match_pairs_cnt += tmp_gt_person
                correct_pairs_cnt += tmp_correct_pairs
                TP_cnt += TP

                #=============================================================================
                pre_target_a = {'points': pre_points[pair_idx * 2][:, 2:4]}
                tp_pred_index_a, tp_gt_index_a = associate_pred2gt_point(pre_target_a, target[pair_idx * 2])
                a_ids_ = target[pair_idx * 2]['person_id'][tp_gt_index_a] if len(tp_gt_index_a) else []
                pre_target_a['points'] = pre_target_a['points'][tp_pred_index_a]
                if len(a_ids_)>0:

                    pre_target_a.update({'person_id': a_ids_})

                    match_gt_a, pois = self.get_ROI_and_MatchInfo(pre_target_a, target[pair_idx * 2 + 1], noise='b')

                    pois = pois[pois[:,0]==0].view(-1,5)
                    mdesc0_ = prroi_pool2d(features[pair_idx * 2:pair_idx * 2 + 2], pois, 1, 1, self.feature_scale)
                    mdesc0_ = mdesc0_.squeeze(2).squeeze(2)[None].transpose(1, 2)  # [batch, dim, num_features]

                    scores_, indices0_, indices1_, mscores0_, mscores1_ = \
                        self.Matching_Layer(mdesc0_, mdesc1, match_gt_a,ignore=True)  # batch,dim,num
                    match_loss_, hard_loss_ = self.Matching_Layer.loss
                    match_loss.append(match_loss_)

                pre_target_b = {'points': pre_points[pair_idx * 2 + 1][:, 2:4]}
                tp_pred_index_b, tp_gt_index_b = associate_pred2gt_point(pre_target_b, target[pair_idx * 2+1])
                b_ids_ = target[pair_idx * 2+1]['person_id'][tp_gt_index_b] if len(tp_gt_index_b) else []
                pre_target_b['points'] = pre_target_b['points'][tp_pred_index_b]
                if len(b_ids_)>0:
                    pre_target_b.update({'person_id': b_ids_})
                    match_gt_b, pois = self.get_ROI_and_MatchInfo( target[pair_idx * 2], pre_target_b, noise='a')

                    pois = pois[pois[:,0]==1].view(-1,5)
                    mdesc1_ = prroi_pool2d(features[pair_idx * 2:pair_idx * 2 + 2], pois, 1, 1, self.feature_scale)
                    mdesc1_ = mdesc1_.squeeze(2).squeeze(2)[None].transpose(1, 2)  # [batch, dim, num_features]

                    scores_, indices0_, indices1_, mscores0_, mscores1_ = \
                        self.Matching_Layer(mdesc0, mdesc1_, match_gt_b,ignore=True)  # batch,dim,num
                    match_loss_, hard_loss_ = self.Matching_Layer.loss
                    match_loss.append(match_loss_)
            else:
                indices0 = torch.zeros(count_in_pair[0]).fill_(-1).to(self.device)
                indices1 = torch.zeros(count_in_pair[1]).fill_(-1).to(self.device)
                mscores0 =  torch.zeros(count_in_pair[0]).fill_(0).to(self.device)
                mscores1 = torch.zeros(count_in_pair[1]).fill_(0).to(self.device)
                match_gt = {'a2b':torch.tensor([]).to(self.device), 'un_b':target[pair_idx*2+1]['person_id']}

            matched_results['matches0'].append(indices0)  # use -1 for invalid match
            matched_results['matches1'].append(indices1)  # use -1 for invalid match
            matched_results['matching_scores0'].append(mscores0)
            matched_results['matching_scores1'].append(mscores1)

            matched_results['gt_matched'].append(match_gt['a2b'])
            matched_results['gt_count_diff'] += len(match_gt['un_b'])
            matched_results['pre_count_diff'] += pre_map[pair_idx * 2 + 1].sum() - (indices1 > -1).sum()

        if len(match_loss)>0:
            self.batch_match_loss =  torch.mean(torch.cat(match_loss))
        if len(hard_loss)>0:
            self.batch_hard_loss = torch.mean(torch.cat(hard_loss))
        if len(norm_loss)>0:
            self.batch_norm_loss = torch.mean(torch.stack(norm_loss))

        return pre_map, gt_den,correct_pairs_cnt,match_pairs_cnt, TP_cnt, matched_results

    def val_forward(self,img,target,frame_signal):
        for i in range(len(target)):
            for key,data in target[i].items():
                if torch.is_tensor(data):
                    target[i][key]=data.to(self.device)

        features, pre_map = self.Extractor(img)
        #Start==============density loss=======
        x = torch.zeros(pre_map.size())
        points = target[0]['points'].long()
        x[0, 0, points[:, 1], points[:, 0]] = 1
        gt_den = self.Gaussian(x.cuda())
        assert pre_map.size() == gt_den.size()
        pre_map = pre_map / self.dataset_cfg.DEN_FACTOR

        # =====extract the points of interest from the prediction density map======
        pre_data = local_maximum_points(pre_map,gaussian_maximun=self.gaussian_maximum,radius=self.radius)
        count_in_pair=[(pre_data['points'][:, 0] == i).sum().cpu() for i in range(pre_map.size(0))]
        pre_points = torch.split(pre_data['points'], count_in_pair)
        print('predict_num:',count_in_pair)

        match_gt, pois = self.get_ROI_and_MatchInfo(target[0], target[1],noise=None)

        if (np.array(count_in_pair) > 0).all():
            poi_features = prroi_pool2d(features, pre_data['rois'], 1, 1, self.feature_scale)
            # poi_features = prroi_pool2d(features, pois, 1, 1, self.feature_scale)
            poi_features=  poi_features.squeeze(2).squeeze(2)[None].transpose(1,2) # [batch, dim, num_features]
            mdesc0, mdesc1 = torch.split(poi_features, count_in_pair,dim=2)

            scores, indices0, indices1, mscores0, mscores1 = self.Matching_Layer( mdesc0, mdesc1)

            pre_outflow = scores[:-1, -1].sum().item()
            pre_inflow = scores[-1,:-1][scores[-1,:-1]>0.4].sum().item()

        else:
            indices0 = torch.zeros(1, count_in_pair[0]).fill_(-1)
            indices1 = torch.zeros(1, count_in_pair[1]).fill_(-1)
            mscores0 = torch.zeros(1, count_in_pair[0]).fill_(0)
            mscores1 = torch.zeros(1, count_in_pair[1]).fill_(0)
            pre_outflow = count_in_pair[0].item()
            pre_inflow = count_in_pair[1].item()
            scores =None
        matched_results = {
            'matches0': indices0,  # use -1 for invalid match
            'matches1': indices1,  # use -1 for invalid match
            'matching_scores0': mscores0,
            'matching_scores1': mscores1,
            'gt_matched': match_gt['a2b'],

            'gt_outflow': len(match_gt['un_a']),
            'gt_inflow': len(match_gt['un_b']),

            'pre_outflow':pre_outflow,
            'pre_inflow': pre_inflow,

            'pre_points':pre_points,
            'pre_map' : pre_map,
            'target': target,
            'scores':scores,
            'match_gt': match_gt
        }
        return pre_map, gt_den, matched_results


    def test_forward(self, img, frame_signal):

        features, pre_map = self.Extractor(img)
        #Start==============density loss=======
        pre_map = pre_map / self.dataset_cfg.DEN_FACTOR

        # =====extract the points of interest from the prediction density map======
        pre_data = local_maximum_points(pre_map,gaussian_maximun=self.gaussian_maximum,radius=self.radius)
        count_in_pair=[(pre_data['points'][:, 0] == i).sum() for i in range(pre_map.size(0))]
        pre_points = torch.split(pre_data['points'], count_in_pair)
        print('predict_num:',count_in_pair)

        if (np.array(count_in_pair) > 0).all():
            poi_features = prroi_pool2d(features, pre_data['rois'], 1, 1, self.feature_scale)
            poi_features=  poi_features.squeeze(2).squeeze(2)[None].transpose(1,2) # [batch, dim, num_features]
            mdesc0, mdesc1 = torch.split(poi_features, count_in_pair,dim=2)

            scores, indices0, indices1, mscores0, mscores1 = self.Matching_Layer( mdesc0, mdesc1)

            pre_outflow = scores[:-1, -1].sum().item()
            pre_inflow = scores[-1,:-1].sum().item()
        else:
            indices0 = torch.zeros(1, count_in_pair[0]).fill_(-1)
            indices1 = torch.zeros(1, count_in_pair[1]).fill_(-1)
            mscores0 = torch.zeros(1, count_in_pair[0]).fill_(0)
            mscores1 = torch.zeros(1, count_in_pair[1]).fill_(0)
            pre_outflow = count_in_pair[0].item()
            pre_inflow = count_in_pair[1].item()
        matched_results = {
            'matches0': indices0,  # use -1 for invalid match
            'matches1': indices1,  # use -1 for invalid match
            'matching_scores0': mscores0,
            'matching_scores1': mscores1,
            # 'gt_matched': match_gt['a2b'],
            #
            # 'gt_outflow': len(match_gt['un_a']),
            # 'gt_inflow': len(match_gt['un_b']),

            'pre_outflow':pre_outflow,
            'pre_inflow': pre_inflow,


            'pre_points':pre_points,
            'pre_map' : pre_map,
            'scores': scores,
        }

        return pre_map, matched_results


if __name__ == '__main__':
    import torch
    import torch.nn.functional as F

    g = torch.zeros(3,4)
    p = torch.zeros(3,4)
    p[2,3] = 0

    from model.PreciseRoIPooling.pytorch.prroi_pool.functional import prroi_pool2d
    f = torch.rand(1,1,4,5).cuda()
    count =0
    for i in range(f.size(2)):
        for j in range(f.size(3)):
            count+=1
            f[0,0,i,j] =count
    print(f)
    rois = torch.tensor([[0,3,0,4,1]]).float().cuda()
    print(prroi_pool2d(f,rois , 1, 1, 1))

    # model = Crowd_locator(cfg)
    # s_t = time.time()
    # for i in range(200):
    #     a = torch.rand(2, 3, 512, 1024).cuda()
    #     f, b ,_= model(a, gt_map = None, mode = 'val')
    # e_t = time.time()
    # print(e_t - s_t)
    # # print(model)
    # summary(model, (3, 224, 224))
    # model = Res_FPN().cuda()
    # predict = torch.load('/media/D/ht/Crowd_loc_master/exp/09-17_10-54_JHU_NWPU_Res50_SCAR_1e-05/all_ep_16_mae_3825.0_mse_8343.1_nae_12.509.pth')
    # model.load_state_dict(predict)
    # img = torch.ones(1,3,80,80).cuda()
    # gt =  torch.ones(1,1,80,80).cuda()
    # out = model(img,gt)
    # print(out)
    # input = torch.zeros(2,100)+0.0001
    # target = torch.ones(2,100)
    # loss = F.binary_cross_entropy(input,target)
    # print(loss)
    # model = Res_FPN(pretrained = False).cuda()
    # summary(model,(3,24,24))

    # import torch
    # import torch.nn as nn
    #
    # N, C_in, H, W, C_out = 10, 4, 16, 16, 4
    # x = torch.randn(N, C_in, H, W).float()
    # conv = nn.Sequential(
    #     nn.Conv2d(4, 8, kernel_size=3, stride=3, padding=1, bias=False),
    #     nn.Conv2d(8, 4, kernel_size=3, stride=3, padding=1, bias=False))
    # conv_group = nn.Sequential(
    #     nn.Conv2d(4, 8, kernel_size=3, stride=3, padding=1, groups=4, bias=False),
    #     nn.Conv2d(8, 4, kernel_size=3, stride=3, padding=1, groups=4, bias=False)
    # )
    #
    # y = conv(x)
    # y_group = conv_group(x)
    # conv_1x1 = nn.Conv2d(C_in, C_out, kernel_size=1)
    # print("groups=1时参数大小：%d" % sum(param.numel() for param in conv.parameters()))
    # print("groups=in_channels时参数大小：%d" % sum(param.numel() for param in conv_group.parameters()))
    # print(y_group.size())