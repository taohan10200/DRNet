import datasets
from  config import cfg
import numpy as np
import torch
from torch import optim
import datasets
from misc.utils import *
from model.VCC_glue_Hungarian import Video_Crowd_Counting
from tqdm import tqdm
import torch.nn.functional as F
from pathlib import Path
import argparse
import matplotlib.cm as cm
parser = argparse.ArgumentParser(
    description='VCC test and demo',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--DATASET', type=str, default='HT21',
    help='Directory where to write output frames (If None, no output)')
parser.add_argument(
    '--output_dir', type=str, default='../dataset/demo',
    help='Directory where to write output frames (If None, no output)')
parser.add_argument(
    '--test_intervals', type=int, default=75,
    help='Directory where to write output frames (If None, no output)')
parser.add_argument(
    '--SEED', type=int, default=3035,
    help='Directory where to write output frames (If None, no output)')
parser.add_argument(
    '--GPU_ID', type=str, default='2',
    help='Directory where to write output frames (If None, no output)')
parser.add_argument(
    '--model_path', type=str, default='./exp/HT21/11-07_01-55_HT21_VGG16_FPN_5e-05_(full model)/ep_5_iter_12500_mae_9.797_mse_10.438_seq_MAE_38.768_WRAE_44.561_MIAE_5.209_MOAE_5.535.pth',
    help='pretrained weight path')

opt = parser.parse_args()
opt.output_dir = opt.output_dir+'_'+opt.DATASET

def test(cfg_data):
    net = Video_Crowd_Counting(cfg, cfg_data)
    test_loader, restore_transform = datasets.loading_testset(opt.DATASET, test_interval=opt.test_intervals)
    # latest_state = torch.load(cfg.RESUME_PATH)
    # net.load_state_dict(latest_state['net'], strict=True)
    state_dict = torch.load(opt.model_path)
    net.load_state_dict(state_dict, strict=True)

    net.eval()
    sing_cnt_errors = {'mae': AverageMeter(), 'mse': AverageMeter()}
    gt_flow_cnt = [133,737,734,1040,321]
    scenes_pred_dict = []

    for scene_id, sub_valset in enumerate(test_loader, 0):
        gen_tqdm = tqdm(sub_valset)
        video_time = len(sub_valset) + opt.test_intervals
        print(video_time)
        pred_dict = {'id': scene_id, 'time': video_time, 'first_frame': 0, 'inflow': [], 'outflow': []}
        for vi, data in enumerate(gen_tqdm, 0):
            img,__ = data
            img = img[0]
            # import  pdb
            # pdb.set_trace()
            # if vi>=21:break
            img = torch.stack(img, 0)
            with torch.no_grad():
                b, c, h, w = img.shape
                if h % 16 != 0:
                    pad_h = 16 - h % 16
                else:
                    pad_h = 0
                if w % 16 != 0:
                    pad_w = 16 - w % 16
                else:
                    pad_w = 0
                pad_dims = (0, pad_w, 0, pad_h)
                img = F.pad(img, pad_dims, "constant")

                if vi % opt.test_intervals == 0 or vi == len(sub_valset) - 1:
                    frame_signal = 'match'
                else:
                    frame_signal = 'skip'

                if frame_signal == 'skip':
                    continue
                else:
                    pred_map, matched_results = net.test_forward(img, frame_signal)

                    #    -----------Counting performance------------------
                    pred_cnt = pred_map[0].sum().item()

                    if vi == 0:
                        pred_dict['first_frame'] = matched_results['pre_points'][0].size(0)#pred_map[0].sum().item()

                    pred_dict['inflow'].append(matched_results['pre_inflow'])
                    pred_dict['outflow'].append(matched_results['pre_outflow'])

                if frame_signal == 'match':
                    pre_crowdflow_cnt, _, _ = compute_metrics_single_scene(pred_dict, opt.test_intervals)

                    print(' den_pre:  %.2f pre_crowd_flow: %.2f pre_inflow:%.2f'
                          %  (pred_cnt, pre_crowdflow_cnt,matched_results['pre_inflow']))

                    kpts0 = matched_results['pre_points'][0][:, 2:4].cpu().numpy()
                    kpts1 = matched_results['pre_points'][1][:, 2:4].cpu().numpy()

                    # matches = matched_results['matches0'].cpu().numpy()
                    # confidence = matched_results['matching_scores0'].cpu().numpy()
                    # if kpts0.shape[0] > 0 and kpts1.shape[0] > 0:
                    #     save_visImg(kpts0, kpts1, matches, confidence, vi, img[0].clone(), img[1].clone(),
                    #                 opt.test_intervals, opt.output_dir, None, None, scene_id, restore_transform)

        scenes_pred_dict.append(pred_dict)

    # import pdb
    # pdb.set_trace()
    MAE,MSE, WRAE, crowdflow_cnt = compute_metrics_all_scenes(scenes_pred_dict, gt_flow_cnt, opt.test_intervals)


    print(MAE, MSE, WRAE, crowdflow_cnt)

    # print(scene_cnt)
    # np.save('scene_cnt.py',scene_cnt)


def compute_metrics_single_scene(pre_dict, intervals):
    pair_cnt = len(pre_dict['inflow'])
    inflow_cnt, outflow_cnt =torch.zeros(pair_cnt,2), torch.zeros(pair_cnt,2)
    pre_crowdflow_cnt  = pre_dict['first_frame']

    for idx, data in enumerate(zip(pre_dict['inflow'],  pre_dict['outflow']),0):
        inflow_cnt[idx, 0] = data[0]
        outflow_cnt[idx, 0] = data[1]


        # if idx %intervals == 0 or  idx== len(pre_dict['inflow'])-1:
        pre_crowdflow_cnt += data[0]


    return pre_crowdflow_cnt,  inflow_cnt, outflow_cnt

def compute_metrics_all_scenes(scenes_pred_dict, scene_gt_dict, intervals):
    scene_cnt = len(scenes_pred_dict)
    metrics = {'MAE':torch.zeros(scene_cnt,2), 'WRAE':torch.zeros(scene_cnt,2)}
    for i,(pre_dict, gt_dict) in enumerate( zip(scenes_pred_dict, scene_gt_dict),0):
        time = pre_dict['time']
        gt_crowdflow_cnt = gt_dict
        # import pdb
        # pdb.set_trace()
        pre_crowdflow_cnt, inflow_cnt, outflow_cnt=\
            compute_metrics_single_scene(pre_dict,intervals)
        mae = np.abs(pre_crowdflow_cnt-gt_crowdflow_cnt)

        metrics['MAE'][i,:] = torch.tensor([pre_crowdflow_cnt, gt_crowdflow_cnt])
        metrics['WRAE'][i,:] = torch.tensor([mae/(gt_crowdflow_cnt+1e-10), time])

    MAE =  torch.mean(torch.abs(metrics['MAE'][:,0] - metrics['MAE'][:,1]))
    MSE = torch.mean((metrics['MAE'][:, 0] - metrics['MAE'][:, 1]) ** 2).sqrt()
    WRAE = torch.sum(metrics['WRAE'][:,0]*(metrics['WRAE'][:,1]/(metrics['WRAE'][:,1].sum()+1e-10)))*100



    return MAE, MSE, WRAE, metrics['MAE']

def save_visImg( kpts0, kpts1, matches, confidence, vi, last_frame, cur_frame, intervals,
                save_path, id0=None, id1=None, scene_id='',restore_transform=None):
    valid = matches > -1
    mkpts0 = kpts0[valid].reshape(-1, 2)
    mkpts1 = kpts1[matches[valid]].reshape(-1, 2)
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

        stem = '{}_{}_{}_matches'.format(scene_id, vi, vi + intervals)
        out_file = str(Path(save_path, stem + '.png'))
        print('\nWriting image to {}'.format(out_file))
        cv2.imwrite(out_file, out)
        out_file = str(Path(save_path, stem + '_vis.png'))
        cv2.imwrite(out_file, out_by_point)

        #     vis_results_more(self.exp_name, self.epoch, self.writer, self.restore_transform, img,
                #                 pred_map.numpy(), dot_map.numpy(),binar_map,
                #                 pred_threshold.numpy(),pred_data['new_boxes'], gt_data['boxes'].squeeze(0))
        # seq_cnt_errors['seq_mae'].update(time_count['time_count_gt'].sum - time_count['time_count_pre'].sum)
        # seq_cnt_errors['seq_mse'].update((time_count['time_count_gt'].sum - time_count['time_count_pre'].sum) ** 2)
    #
    # loss = losses.avg
    # mae = sing_cnt_errors['mae'].avg
    # mse = np.sqrt(sing_cnt_errors['mse'].avg)
    # nae = sing_cnt_errors['nae'].avg
    #
    # diff_mae = diff_cnt_errors['diff_mae'].avg
    # diff_mse = diff_cnt_errors['diff_mse'].avg
    #
    # seq_mae = seq_cnt_errors['seq_mae'].avg
    # seq_mse = np.sqrt(seq_cnt_errors['seq_mse'].avg)


    # print_NWPU_summary_det(self, {'mae': mae, 'mse': mse, 'nae': nae, 'loss': loss, 'diff_mae': diff_mae, 'diff_mse': diff_mse, 'seq_mae': seq_mae,
    #                               'seq_mse': seq_mse})


if __name__=='__main__':
    import os
    import numpy as np
    import torch
    from config import cfg
    from importlib import import_module


    # ------------prepare enviroment------------
    seed = opt.SEED
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = opt.GPU_ID
    torch.backends.cudnn.benchmark = True

    # ------------prepare data loader------------
    data_mode = opt.DATASET
    datasetting = import_module(f'datasets.setting.{data_mode}')
    cfg_data = datasetting.cfg_data

    # ------------Prepare Trainer------------
    # from trainer import Trainer

    # ------------Start Training------------
    pwd = os.path.split(os.path.realpath(__file__))[0]
    test(cfg_data)
