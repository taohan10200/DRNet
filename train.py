import numpy as np
import torch
from torch import optim
import datasets
from misc.utils import *
from model.VIC import Video_Individual_Counter
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.cm as cm
from pathlib import Path
from config import cfg
from misc.KPI_pool import Task_KPI_Pool
class Trainer():
    def __init__(self, cfg_data, pwd):
        self.exp_name = cfg.EXP_NAME
        self.exp_path = cfg.EXP_PATH
        self.pwd = pwd
        self.net = Video_Individual_Counter(cfg, cfg_data)
        self.train_loader, self.val_loader, self.restore_transform = datasets.loading_data(cfg.DATASET, cfg.VAL_INTERVALS)

        params = [
            {"params": self.net.Extractor.parameters(), 'lr': cfg.LR_Base, 'weight_decay': cfg.WEIGHT_DECAY},
            {"params": self.net.Matching_Layer.parameters(), "lr": cfg.LR_Thre, 'weight_decay': cfg.WEIGHT_DECAY},
        ]
        self.optimizer = optim.Adam(params)
        self.i_tb = 0
        self.epoch = 1
        self.timer={'iter time': Timer(), 'train time': Timer(), 'val time': Timer()}

        self.num_iters = cfg.MAX_EPOCH * np.int(len(self.train_loader))
        self.train_record = {'best_model_name': '', 'mae': 1e20, 'mse': 1e20, 'seq_MAE':1e20, 'WRAE':1e20, 'MIAE': 1e20, 'MOAE': 1e20}

        if cfg.RESUME:
            latest_state = torch.load(cfg.RESUME_PATH)
            self.net.load_state_dict(latest_state['net'], strict=True)
            self.optimizer.load_state_dict(latest_state['optimizer'])
            self.epoch = latest_state['epoch']
            self.i_tb = latest_state['i_tb']
            self.train_record = latest_state['train_record']
            self.exp_path = latest_state['exp_path']
            self.exp_name = latest_state['exp_name']
            print("Finish loading resume mode")
        self.writer, self.log_txt = logger(self.exp_path, self.exp_name, self.pwd, ['exp','eval','figure','img', 'vis','output'], resume=cfg.RESUME)
        self.task_KPI=Task_KPI_Pool(task_setting={'den': ['gt_cnt', 'mae'], 'match': ['gt_pairs', 'pre_pairs']}, maximum_sample=1000)
    def forward(self):
        for epoch in range(self.epoch, cfg.MAX_EPOCH):
            # self.validate()
            self.epoch = epoch
            self.timer['train time'].tic()
            self.train()
            self.timer['train time'].toc(average=False)
            print( 'train time: {:.2f}s'.format(self.timer['train time'].diff) )
            print( '='*20 )

    def train(self): # training for all datasets
        self.net.train()
        lr1, lr2 = adjust_learning_rate(self.optimizer,
                                   self.epoch,
                                   cfg.LR_Base,
                                   cfg.LR_Thre,
                                   cfg.LR_DECAY)
        batch_loss = {'match':AverageMeter(), 'den':AverageMeter(), 'hard':AverageMeter(), 'norm':AverageMeter()}
        for i, data in enumerate(self.train_loader, 0):
            self.timer['iter time'].tic()
            self.i_tb += 1
            # import pdb
            # pdb.set_trace()
            img, label = data

            pre_map, gt_map, correct_pairs_cnt,match_pairs_cnt, TP, matched_results = self.net(img, label)
            counting_mse_loss,matching_loss,hard_loss, norm_loss  = self.net.loss

            pre_cnt = pre_map.sum()
            gt_cnt = gt_map.sum()

            self.task_KPI.add({'den': {'gt_cnt': gt_map.sum(), 'mae': max(0, gt_cnt-(gt_cnt-pre_cnt).abs())},
                               'match': {'gt_pairs': match_pairs_cnt, 'pre_pairs': correct_pairs_cnt}})
            self.KPI = self.task_KPI.query()

            loss = torch.stack([counting_mse_loss, matching_loss+hard_loss])
            weight = torch.stack([self.KPI['den'],self.KPI['match']]).to(loss.device)
            weight = -(1-weight) * torch.log(weight+1e-8)
            self.weight = weight/weight.sum()

            all_loss = (self.weight*loss).sum() #w_den*matching_loss + w_match*counting_mse_loss# + reg_loss  #0+0.001*norm_loss

            self.optimizer.zero_grad()
            all_loss.backward()
            self.optimizer.step()
            batch_loss['match'].update(matching_loss.item())
            batch_loss['den'].update(counting_mse_loss.item())
            batch_loss['hard'].update(hard_loss.item())
            batch_loss['norm'].update(norm_loss.item())


            if (self.i_tb) % cfg.PRINT_FREQ == 0:
                # self.writer.add_scalar('train_lr1', lr1, self.i_tb)
                # self.writer.add_scalar('train_loss', head_map_loss.item(), self.i_tb)
                self.timer['iter time'].toc(average=False)
                print('[ep %d][it %d][loss_reg %.4f][loss_match %.4f][loss_hard %.4f][lr %.4f][w_d %.3f ][w_m %.3f][acc_d %.2f ][acc_m %.2f][%.2fs]' % \
                      (self.epoch, self.i_tb, batch_loss['den'].avg, batch_loss['match'].avg,batch_loss['hard'].avg,
                       lr1 * 10000,  self.weight[0].item(),self.weight[1].item(),self.KPI['den'].item(),self.KPI['match'].item(),self.timer['iter time'].diff))
                print('       [cnt: gt: %.1f pred: %.1f max_pre: %.1f max_gt: %.1f]  '
                      '[match_gt: %.1f matched_a2b: %.1f ]'
                      '[gt_count_diff: %.1f pre_count_diff: %.1f] '  %
                      (gt_cnt.item(), pre_cnt.item(),pre_map.max().item()*cfg_data.DEN_FACTOR, gt_map.max().item()*cfg_data.DEN_FACTOR,\
                        torch.cat(matched_results['gt_matched']).size(0),TP,
                        matched_results['gt_count_diff'], matched_results['pre_count_diff']  ))
                print(self.net.Matching_Layer.bin_score)
            if (self.i_tb) % 60 == 0:
                kpts0 = label[0]['points'].cpu().numpy()  # h,w-> w,h
                kpts1 = label[1]['points'].cpu().numpy()  # h,w-> w,h
                id0 = label[0]['person_id'].cpu().numpy()
                id1 = label[1]['person_id'].cpu().numpy()
                matches = matched_results['matches0'][0].cpu().detach().numpy()
                confidence = matched_results['matching_scores0'][0].cpu().detach().numpy()
                if kpts0.shape[0] > 0 and kpts1.shape[0] > 0:
                    save_visImg(kpts0, kpts1,  matches, confidence, self.i_tb, img[0].clone(), img[1].clone(), 1,
                                self.exp_path, id0, id1,scene_id='',restore_transform=self.restore_transform)

                save_results_more(self.i_tb, self.exp_path, self.restore_transform, img[1].clone().unsqueeze(0), pre_map[1].detach().cpu().numpy(), \
                                  gt_map[1].detach().cpu().numpy(), pre_map[1].detach().cpu().numpy(),
                                  pre_map[1].detach().cpu().numpy(), pre_map[1].detach().cpu().numpy())
            # import pdb
            # pdb.set_trace()
            if self.i_tb % 2500 == 0:
                self.timer['val time'].tic()
                self.validate()
                self.net.train()
                self.timer['val time'].toc(average=False)
                print('val time: {:.2f}s'.format(self.timer['val time'].diff))
    def validate(self):
        self.net.eval()
        sing_cnt_errors = {'mae': AverageMeter(), 'mse': AverageMeter()}

        scenes_pred_dict = []
        scenes_gt_dict = []
        for scene_id, sub_valset in  enumerate(self.val_loader, 0):
            gen_tqdm = tqdm(sub_valset)
            video_time = len(sub_valset)+cfg.VAL_INTERVALS
            print(video_time)
            pred_dict = {'id': scene_id, 'time':video_time, 'first_frame': 0, 'inflow': [], 'outflow': []}
            gt_dict  = {'id': scene_id, 'time':video_time, 'first_frame': 0, 'inflow': [], 'outflow': []}
            for vi, data in enumerate(gen_tqdm, 0):
                img,target = data
                img,target = img[0], target[0]

                img = torch.stack(img,0)
                with torch.no_grad():
                    b, c, h, w = img.shape
                    if h % 16 != 0: pad_h = 16 - h % 16
                    else: pad_h = 0
                    if w % 16 != 0: pad_w = 16 - w % 16
                    else: pad_w = 0
                    pad_dims = (0, pad_w, 0, pad_h)
                    img = F.pad(img, pad_dims, "constant")

                    if vi % cfg.VAL_INTERVALS== 0 or vi ==len(sub_valset)-1:
                        frame_signal = 'match'
                    else: frame_signal = 'skip'

                    if frame_signal == 'skip':
                        continue
                    else:
                        pred_map, gt_den, matched_results = self.net.val_forward(img, target,frame_signal)

                        #    -----------Counting performance------------------
                        gt_count, pred_cnt = gt_den[0].sum().item(),  pred_map[0].sum().item()

                        s_mae = abs(gt_count - pred_cnt)
                        s_mse = ((gt_count - pred_cnt) * (gt_count - pred_cnt))
                        sing_cnt_errors['mae'].update(s_mae)
                        sing_cnt_errors['mse'].update(s_mse)

                        if vi == 0:
                            pred_dict['first_frame'] = pred_map[0].sum().item()
                            gt_dict['first_frame'] = len(target[0]['person_id'])

                        pred_dict['inflow'].append(matched_results['pre_inflow'])
                        pred_dict['outflow'].append(matched_results['pre_outflow'])
                        gt_dict['inflow'].append(matched_results['gt_inflow'])
                        gt_dict['outflow'].append(matched_results['gt_outflow'])

                        if frame_signal == 'match':
                            pre_crowdflow_cnt, gt_crowdflow_cnt,_,_ =compute_metrics_single_scene(pred_dict, gt_dict,1)# cfg.VAL_INTERVALS)

                            print('den_gt: %.2f den_pre: %.2f mae: %.2f gt_crowd_flow: %.2f, pre_crowd_flow: %.2f gt_inflow: %.2f pre_inflow:%.2f'
                                  %(gt_count,pred_cnt, s_mae,gt_crowdflow_cnt,pre_crowdflow_cnt,matched_results['gt_inflow'],matched_results['pre_inflow']))

                            kpts0 = matched_results['pre_points'][0][:, 2:4].cpu().numpy()
                            kpts1 = matched_results['pre_points'][1][:, 2:4].cpu().numpy()

                            matches = matched_results['matches0'].cpu().numpy()
                            confidence = matched_results['matching_scores0'].cpu().numpy()
                            # if kpts0.shape[0]>0 and kpts1.shape[0]>0:
                            #     save_visImg(kpts0,kpts1,matches, confidence, vi, img[0].clone(),img[1].clone(),
                            #                      cfg.VAL_INTERVALS, cfg.VAL_VIS_PATH,None,None,scene_id,self.restore_transform)

            scenes_pred_dict.append(pred_dict)
            scenes_gt_dict.append(gt_dict)
        # import pdb
        # pdb.set_trace()
        MAE, MSE,WRAE, MIAE, MOAE, cnt_result =compute_metrics_all_scenes(scenes_pred_dict,scenes_gt_dict, 1)#cfg.VAL_INTERVALS)


        print('MAE: %.2f, MSE: %.2f  WRAE: %.2f WIAE: %.2f WOAE: %.2f' % (MAE.data, MSE.data, WRAE.data, MIAE.data, MOAE.data))
        print('Pre vs GT:', cnt_result)
        mae = sing_cnt_errors['mae'].avg
        mse = np.sqrt(sing_cnt_errors['mse'].avg)

        self.train_record = update_model(self,{'mae':mae, 'mse':mse, 'seq_MAE':MAE, 'WRAE':WRAE, 'MIAE': MIAE, 'MOAE': MOAE })

        print_NWPU_summary_det(self,{'mae':mae, 'mse':mse, 'seq_MAE':MAE, 'WRAE':WRAE, 'MIAE': MIAE, 'MOAE': MOAE})

def save_visImg(kpts0,kpts1,matches,confidence,vi, last_frame, cur_frame,intervals,
                save_path,  id0=None, id1=None,scene_id='',restore_transform=None):

    valid = matches > -1
    mkpts0 = kpts0[valid].reshape(-1,2)
    mkpts1 = kpts1[matches[valid]].reshape(-1,2)
    color = cm.jet(confidence[valid])

    text = [
        'VCC',
        'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
        'Matches: {}'.format(len(mkpts1))
    ]
    small_text = [
        'Match Threshold: {:.2f}'.format(0.1),
        'Image Pair: {:06}:{:06}'.format(vi - intervals, vi)
    ]

    out, out_by_point = make_matching_plot_fast(
        last_frame, cur_frame, kpts0, kpts1, mkpts0, mkpts1, color, text,
        path=None, show_keypoints=True, small_text=small_text, restore_transform=restore_transform,
        id0=id0, id1=id1)
    if save_path is not None:
        # print('==> Will write outputs to {}'.format(save_path))
        Path(save_path).mkdir(exist_ok=True)

        stem = '{}_{}_{}_matches'.format(scene_id, vi , vi+ intervals)
        out_file = str(Path(save_path, stem + '.png'))
        print('\nWriting image to {}'.format(out_file))
        cv2.imwrite(out_file, out)
        out_file = str(Path(save_path, stem + '_vis.png'))
        cv2.imwrite(out_file, out_by_point)

def compute_metrics_single_scene(pre_dict, gt_dict, intervals):
    pair_cnt = len(pre_dict['inflow'])
    inflow_cnt, outflow_cnt =torch.zeros(pair_cnt,2), torch.zeros(pair_cnt,2)
    pre_crowdflow_cnt  = pre_dict['first_frame']
    gt_crowdflow_cnt =  gt_dict['first_frame']
    for idx, data in enumerate(zip(pre_dict['inflow'],  pre_dict['outflow'],\
                                   gt_dict['inflow'], gt_dict['outflow']),0):
        inflow_cnt[idx, 0] = data[0]
        inflow_cnt[idx, 1] = data[2]
        outflow_cnt[idx, 0] = data[1]
        outflow_cnt[idx, 1] = data[3]

        if idx %intervals == 0 or  idx== len(pre_dict['inflow'])-1:
            pre_crowdflow_cnt += data[0]
            gt_crowdflow_cnt += data[2]

    return pre_crowdflow_cnt, gt_crowdflow_cnt,  inflow_cnt, outflow_cnt

def compute_metrics_all_scenes(scenes_pred_dict, scene_gt_dict, intervals):
    scene_cnt = len(scenes_pred_dict)
    metrics = {'MAE':torch.zeros(scene_cnt,2), 'WRAE':torch.zeros(scene_cnt,2), 'MIAE':torch.zeros(0), 'MOAE':torch.zeros(0)}
    for i,(pre_dict, gt_dict) in enumerate( zip(scenes_pred_dict, scene_gt_dict),0):
        time = pre_dict['time']
        # import pdb
        # pdb.set_trace()
        pre_crowdflow_cnt, gt_crowdflow_cnt, inflow_cnt, outflow_cnt=\
            compute_metrics_single_scene(pre_dict, gt_dict,intervals)
        mae = np.abs(pre_crowdflow_cnt-gt_crowdflow_cnt)
        metrics['MAE'][i, :] = torch.tensor([pre_crowdflow_cnt, gt_crowdflow_cnt])
        metrics['WRAE'][i,:] = torch.tensor([mae/(gt_crowdflow_cnt+1e-10), time])

        metrics['MIAE'] =  torch.cat([metrics['MIAE'], torch.abs(inflow_cnt[:,0]-inflow_cnt[:,1])])
        metrics['MOAE'] = torch.cat([metrics['MOAE'], torch.abs(outflow_cnt[:, 0] - outflow_cnt[:, 1])])

    MAE = torch.mean(torch.abs(metrics['MAE'][:, 0] - metrics['MAE'][:, 1]))
    MSE = torch.mean((metrics['MAE'][:, 0] - metrics['MAE'][:, 1]) ** 2).sqrt()
    WRAE = torch.sum(metrics['WRAE'][:,0]*(metrics['WRAE'][:,1]/(metrics['WRAE'][:,1].sum()+1e-10)))*100
    MIAE = torch.mean(metrics['MIAE'] )
    MOAE = torch.mean(metrics['MOAE'])

    return MAE,MSE, WRAE,MIAE,MOAE,metrics['MAE']

if __name__=='__main__':
    import os
    import random
    import numpy as np
    import torch
    import datasets
    from config import cfg
    from importlib import import_module
    # ------------prepare enviroment------------
    seed = cfg.SEED
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU_ID
    torch.backends.cudnn.benchmark = True

    # ------------prepare data loader------------
    data_mode = cfg.DATASET
    datasetting = import_module(f'datasets.setting.{data_mode}')
    cfg_data = datasetting.cfg_data

    # ------------Prepare Trainer------------
    # from trainer import Trainer

    # ------------Start Training------------
    pwd = os.path.split(os.path.realpath(__file__))[0]
    cc_trainer = Trainer(cfg_data, pwd)
    cc_trainer.forward()
