import datasets
from  config import cfg
import numpy as np
import torch
import datasets
from misc.utils import *
from model.VIC import Video_Individual_Counter
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
    '--skip_flag', type=bool, default=True,
    help='To caculate the MIAE and MOAE, it should be False')
parser.add_argument(
    '--SEED', type=int, default=3035,
    help='Directory where to write output frames (If None, no output)')
parser.add_argument(
    '--GPU_ID', type=str, default='0',
    help='Directory where to write output frames (If None, no output)')

# parser.add_argument(
#     '--model_path', type=str,
#     default='./model/pretrained_models/HT21.pth',
#     help='pretrained weight path')

parser.add_argument(
    '--model_path', type=str,
    default='./exp/HT21/03-19_12-02_HT21_VGG16_FPN_5e-05/ep_5_iter_12500_mae_27.324_mse_27.654_seq_MAE_40.081_WRAE_45.546_MIAE_3.113_MOAE_2.961.pth',
    help='pretrained weight path')


opt = parser.parse_args()
opt.output_dir = opt.output_dir+'_'+opt.DATASET


def test(cfg_data):
    net = Video_Individual_Counter(cfg, cfg_data)
    test_loader, restore_transform = datasets.loading_testset(opt.DATASET, test_interval=opt.test_intervals, mode='test')
    state_dict = torch.load(opt.model_path)
    net.load_state_dict(state_dict, strict=True)
    net.eval()

    gt_flow_cnt = [133,737,734,1040,321]
    scenes_pred_dict = []
    if opt.skip_flag:
        intervals = 1
    else:
        intervals = opt.test_intervals
    for scene_id, sub_valset in enumerate(test_loader, 0):
        gen_tqdm = tqdm(sub_valset)
        video_time = len(sub_valset) + opt.test_intervals
        print(video_time)
        pred_dict = {'id': scene_id, 'time': video_time, 'first_frame': 0, 'inflow': [], 'outflow': []}
        for vi, data in enumerate(gen_tqdm, 0):
            img,__ = data
            img = img[0]
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

                if frame_signal == 'match' or not opt.skip_flag:

                    pred_map, matched_results = net.test_forward(img, frame_signal)

                    # import pdb
                    # pdb.set_trace()
                    #=========================================================
                    #    -----------Counting performance------------------
                    pred_cnt = pred_map[0].sum().item()


                    ##=====================================================

                    if vi == 0:
                        pred_dict['first_frame'] = pred_map[0].sum().item()

                    pred_dict['inflow'].append(matched_results['pre_inflow'])
                    pred_dict['outflow'].append(matched_results['pre_outflow'])

                if frame_signal == 'match':
                    pre_crowdflow_cnt, _, _ = compute_metrics_single_scene(pred_dict, intervals)

                    print(' den_pre:  %.2f pre_crowd_flow: %.2f pre_inflow:%.2f'
                          %  (pred_cnt, pre_crowdflow_cnt,matched_results['pre_inflow']))

                    kpts0 = matched_results['pre_points'][0][:, 2:4].cpu().numpy()
                    kpts1 = matched_results['pre_points'][1][:, 2:4].cpu().numpy()

                    matches = matched_results['matches0'].cpu().numpy()
                    confidence = matched_results['matching_scores0'].cpu().numpy()
                    # if kpts0.shape[0] > 0 and kpts1.shape[0] > 0:
                    #     save_visImg(kpts0, kpts1, matches, confidence, vi, img[0].clone(), img[1].clone(),
                    #                 opt.test_intervals, opt.output_dir, None, None, scene_id, restore_transform)

        scenes_pred_dict.append(pred_dict)

    # import pdb
    # pdb.set_trace()
    MAE,MSE, WRAE, crowdflow_cnt  = compute_metrics_all_scenes(scenes_pred_dict, gt_flow_cnt, intervals)


    print('MAE: %.2f, MSE: %.2f  WRAE: %.2f' % (MAE.data, MSE.data, WRAE.data))
    print(crowdflow_cnt)

    return  MAE,MSE, WRAE
    # np.save('scene_cnt.py',scene_cnt)


def compute_metrics_single_scene(pre_dict, intervals):
    pair_cnt = len(pre_dict['inflow'])
    inflow_cnt, outflow_cnt =torch.zeros(pair_cnt,2), torch.zeros(pair_cnt,2)
    pre_crowdflow_cnt  = pre_dict['first_frame']

    for idx, data in enumerate(zip(pre_dict['inflow'],  pre_dict['outflow']),0):
        inflow_cnt[idx, 0] = data[0]
        outflow_cnt[idx, 0] = data[1]
        if idx %intervals == 0 or  idx== len(pre_dict['inflow'])-1:
            pre_crowdflow_cnt += data[0]


    return pre_crowdflow_cnt,  inflow_cnt, outflow_cnt

def compute_metrics_all_scenes(scenes_pred_dict, scene_gt_dict, intervals):
    scene_cnt = len(scenes_pred_dict)
    metrics = {'MAE':torch.zeros(scene_cnt,2), 'WRAE':torch.zeros(scene_cnt,2)}
    for i,(pre_dict, gt_dict) in enumerate( zip(scenes_pred_dict, scene_gt_dict),0):
        time = pre_dict['time']
        gt_crowdflow_cnt = gt_dict
        pre_crowdflow_cnt, inflow_cnt, outflow_cnt=\
            compute_metrics_single_scene(pre_dict,intervals)
        mae = np.abs(pre_crowdflow_cnt-gt_crowdflow_cnt)

        metrics['MAE'][i,:] = torch.tensor([pre_crowdflow_cnt, gt_crowdflow_cnt])
        metrics['WRAE'][i,:] = torch.tensor([mae/(gt_crowdflow_cnt+1e-10), time])

    MAE =  torch.mean(torch.abs(metrics['MAE'][:,0] - metrics['MAE'][:,1]))
    MSE = torch.mean((metrics['MAE'][:, 0] - metrics['MAE'][:, 1])**2).sqrt()
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

    pwd = os.path.split(os.path.realpath(__file__))[0]
    mae, mse, wrae=[],[],[]

    MAE, MSE, WRAE= test(cfg_data,)
    mae.append(MAE.item())
    mse.append(MSE.item())
    wrae.append(WRAE.item())
    print(mae)
    print(mse)
    print(wrae)
# [816.3448486328125, 640.3782958984375, 507.827392578125, 406.81219482421875, 320.8839416503906, 244.4170379638672,
# 213.17477416992188, 179.2323455810547, 161.1795654296875, 158.57400512695312, 141.13636779785156, 136.9983367919922, 136.33468627929688, 140.97805786132812, 131.7980499267578, 158.04241943359375, 145.49990844726562, 142.96688842773438, 150.99642944335938, 161.00091552734375, 164.7197723388672, 156.69297790527344, 165.933837890625, 170.65945434570312, 185.41761779785156]
# [943.0086059570312, 743.6002807617188, 602.62255859375, 489.0594787597656, 405.30120849609375, 338.2956848144531,
# 295.9758605957031, 237.63165283203125, 214.13047790527344, 205.19427490234375, 192.3365020751953, 177.90206909179688, 182.60838317871094, 191.5061492919922, 176.87466430664062, 212.279541015625, 209.7168426513672, 207.7462921142578, 213.74208068847656, 235.60743713378906, 240.92457580566406, 233.2116241455078, 255.526123046875, 260.2579345703125, 266.4524230957031]
# [154.50405883789062, 124.56787872314453, 101.60711669921875, 84.25566864013672, 67.91519165039062, 55.78730010986328,
# 49.020721435546875, 39.328514099121094, 34.659854888916016, 32.26956558227539, 27.437755584716797, 25.305044174194336, 22.083044052124023, 22.094196319580078, 19.489850997924805, 22.019485473632812, 18.96535873413086, 18.027475357055664, 19.00623321533203, 19.262075424194336, 19.47953987121582, 18.248493194580078, 18.887287139892578, 19.31798553466797, 22.53875160217285]

#
# [194.53955078125, 230.01107788085938, 246.7897491455078, 267.2936096191406, 279.252685546875, 294.0309143066406]
# [283.6376037597656, 313.34832763671875, 330.8506164550781, 346.0884704589844, 357.9461364746094, 373.9583740234375]
# [23.79248046875, 30.38051986694336, 32.901893615722656, 37.1819953918457, 39.26314163208008, 42.06587219238281]
