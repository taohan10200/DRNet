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
from train import compute_metrics_single_scene,compute_metrics_all_scenes
import  os.path as osp
from model.MatchTool.compute_metric import associate_pred2gt_point_vis

parser = argparse.ArgumentParser(
    description='VIC test and demo',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--DATASET', type=str, default='SENSE',
    help='Directory where to write output frames (If None, no output)')
parser.add_argument(
    '--output_dir', type=str, default='../dataset/demo_den_test',
    help='Directory where to write output frames (If None, no output)')
parser.add_argument(
    '--test_intervals', type=int, default=15,
    help='Directory where to write output frames (If None, no output)')
parser.add_argument(
    '--skip_flag', type=bool, default=False,
    help='if you need to caculate the MIAE and MOAE, it should be False')
parser.add_argument(
    '--SEED', type=int, default=3035,
    help='Directory where to write output frames (If None, no output)')
parser.add_argument(
    '--GPU_ID', type=str, default='2',
    help='Directory where to write output frames (If None, no output)')

parser.add_argument(
    '--model_path', type=str, default='./model/pretrained_models/SenseCrowd.pth',
    help='pretrained weight path')

# parser.add_argument(
#     '--model_path', type=str, default='./exp/SENSE/03-22_17-33_SENSE_VGG16_FPN_5e-05/ep_15_iter_115000_mae_2.211_mse_3.677_seq_MAE_6.439_WRAE_9.506_MIAE_1.447_MOAE_1.474.pth',
#     help='pretrained weight path')



opt = parser.parse_args()
opt.output_dir = opt.output_dir+'_'+opt.DATASET


def test(cfg_data):
    net = Video_Individual_Counter(cfg, cfg_data)
    with open(osp.join(cfg_data.DATA_PATH, 'scene_label.txt'), 'r') as f:
        lines = f.readlines()
    scene_label = {}
    for line in lines:
        line = line.rstrip().split(' ')
        scene_label.update({line[0]: [int(i) for i in line[1:]] })

    test_loader, restore_transform = datasets.loading_testset(opt.DATASET, test_interval=opt.test_intervals,mode='test')

    state_dict = torch.load(opt.model_path)
    net.load_state_dict(state_dict, strict=True)
    net.eval()

    scenes_pred_dict = {'all':[], 'in':[], 'out':[], 'day':[],'night':[], 'scenic0':[], 'scenic1':[],'scenic2':[],
                      'scenic3':[],'scenic4':[],'scenic5':[], 'density0':[],'density1':[],'density2':[], 'density3':[],'density4':[] }
    scenes_gt_dict =  {'all':[], 'in':[], 'out':[], 'day':[],'night':[], 'scenic0':[], 'scenic1':[],'scenic2':[],
                      'scenic3':[],'scenic4':[],'scenic5':[], 'density0':[],'density1':[],'density2':[], 'density3':[],'density4':[] }

    if opt.skip_flag:
        intervals = 1
    else:
        intervals = opt.test_intervals

    for scene_id, sub_valset in enumerate(test_loader, 0):
        # if scene_id>2:
        #     break
        gen_tqdm = tqdm(sub_valset)
        video_time = len(sub_valset) + opt.test_intervals
        print(video_time)

        scene_name = ''
        pred_dict = {'id': scene_id, 'time': video_time, 'first_frame': 0, 'inflow': [], 'outflow': []}
        gt_dict = {'id': scene_id, 'time': video_time, 'first_frame': 0, 'inflow': [], 'outflow': []}

        for vi, data in enumerate(gen_tqdm, 0):
            img, target = data
            # import pdb
            # pdb.set_trace()
            img, target = img[0], target[0]
            scene_name = target[0]['scene_name']
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

                    pred_map, gt_den, matched_results = net.val_forward(img, target, frame_signal)


                    # save_inflow_outflow_density(img, matched_results['scores'], matched_results['pre_points'],
                    #                             matched_results['target'], matched_results['match_gt'],
                    #                             osp.join(opt.output_dir, scene_name), scene_name, vi, opt.test_intervals)

                    #    -----------Counting performance------------------
                    gt_count, pred_cnt = gt_den[0].sum().item(), pred_map[0].sum().item()

                    # ============================================================
                    pred_cnt = pred_map[0].sum().item()

                    #===================================================================
                    if vi == 0:
                        pred_dict['first_frame'] = pred_map[0].sum().item()
                        gt_dict['first_frame'] = len(target[0]['person_id'])


                    pred_dict['inflow'].append(matched_results['pre_inflow'])
                    pred_dict['outflow'].append(matched_results['pre_outflow'])
                    gt_dict['inflow'].append(matched_results['gt_inflow'])
                    gt_dict['outflow'].append(matched_results['gt_outflow'])

                if frame_signal == 'match':
                    pre_crowdflow_cnt, gt_crowdflow_cnt, _, _ = compute_metrics_single_scene(pred_dict, gt_dict, intervals)

                    print('den_gt: %.2f den_pre: %.2f gt_crowd_flow: %.2f, pre_crowd_flow: %.2f gt_inflow: %.2f pre_inflow:%.2f'
                          % (gt_count, pred_cnt, gt_crowdflow_cnt, pre_crowdflow_cnt, matched_results['gt_inflow'],
                             matched_results['pre_inflow']))

                    kpts0 = matched_results['pre_points'][0][:, 2:4].cpu().numpy()
                    kpts1 = matched_results['pre_points'][1][:, 2:4].cpu().numpy()

                    matches = matched_results['matches0'].cpu().numpy()
                    confidence = matched_results['matching_scores0'].cpu().numpy()
                    if kpts0.shape[0] > 0 and kpts1.shape[0] > 0:
                        save_visImg(kpts0, kpts1, matches, confidence, vi, img[0].clone(), img[1].clone(),
                                    opt.test_intervals, osp.join(opt.output_dir,scene_name), None, None, scene_name, restore_transform)

                        save_inflow_outflow_density(img, matched_results['scores'], matched_results['pre_points'],
                                                    matched_results['target'], matched_results['match_gt'],
                                                    osp.join(opt.output_dir,scene_name), scene_name, vi, opt.test_intervals)

        scenes_pred_dict['all'].append(pred_dict)
        scenes_gt_dict['all'].append(gt_dict)

        scene_l = scene_label[scene_name]
        if scene_l[0] == 0: scenes_pred_dict['in'].append(pred_dict);  scenes_gt_dict['in'].append(gt_dict)
        if scene_l[0] == 1: scenes_pred_dict['out'].append(pred_dict);  scenes_gt_dict['out'].append(gt_dict)
        if scene_l[1] == 0: scenes_pred_dict['day'].append(pred_dict);  scenes_gt_dict['day'].append(gt_dict)
        if scene_l[1] == 1: scenes_pred_dict['night'].append(pred_dict);  scenes_gt_dict['night'].append(gt_dict)
        if scene_l[2] == 0: scenes_pred_dict['scenic0'].append(pred_dict);  scenes_gt_dict['scenic0'].append(gt_dict)
        if scene_l[2] == 1: scenes_pred_dict['scenic1'].append(pred_dict);  scenes_gt_dict['scenic1'].append(gt_dict)
        if scene_l[2] == 2: scenes_pred_dict['scenic2'].append(pred_dict);  scenes_gt_dict['scenic2'].append(gt_dict)
        if scene_l[2] == 3: scenes_pred_dict['scenic3'].append(pred_dict);  scenes_gt_dict['scenic3'].append(gt_dict)
        if scene_l[2] == 4: scenes_pred_dict['scenic4'].append(pred_dict);  scenes_gt_dict['scenic4'].append(gt_dict)
        if scene_l[2] == 5: scenes_pred_dict['scenic5'].append(pred_dict);  scenes_gt_dict['scenic5'].append(gt_dict)
        if scene_l[3] == 0: scenes_pred_dict['density0'].append(pred_dict);  scenes_gt_dict['density0'].append(gt_dict)
        if scene_l[3] == 1: scenes_pred_dict['density1'].append(pred_dict);  scenes_gt_dict['density1'].append(gt_dict)
        if scene_l[3] == 2: scenes_pred_dict['density2'].append(pred_dict);  scenes_gt_dict['density2'].append(gt_dict)
        if scene_l[3] == 3: scenes_pred_dict['density3'].append(pred_dict);  scenes_gt_dict['density3'].append(gt_dict)
        if scene_l[3] == 4: scenes_pred_dict['density4'].append(pred_dict);  scenes_gt_dict['density4'].append(gt_dict)

    for key in scenes_pred_dict.keys():
        s_pred_dict = scenes_pred_dict[key]
        s_gt_dict = scenes_gt_dict[key]
        MAE, MSE, WRAE, MIAE, MOAE, cnt_result = compute_metrics_all_scenes(s_pred_dict, s_gt_dict, intervals)
        if key == 'all':save_cnt_result = cnt_result

        print('='*20, key, '='*20)
        print('MAE: %.2f, MSE: %.2f  WRAE: %.2f WIAE: %.2f WOAE: %.2f' % (MAE.data, MSE.data, WRAE.data, MIAE.data, MOAE.data))
    print(save_cnt_result)

    os.makedirs('./results',mode=0o777, exist_ok=True)
    np.save(os.path.join('./results',str(opt.test_intervals)+'_SENSE_cnt.py'),save_cnt_result.numpy())


def save_visImg( kpts0, kpts1, matches, confidence, vi, last_frame, cur_frame, intervals,
                save_path, id0=None, id1=None, scene_id='',restore_transform=None):
    valid = matches > -1
    mkpts0 = kpts0[valid].reshape(-1, 2)
    mkpts1 = kpts1[matches[valid]].reshape(-1, 2)
    color = cm.jet(confidence[valid])

    text = [
        'VIC',
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
        os.makedirs(save_path,mode =0o777, exist_ok=True)

        stem = '{}_{}_{}_matches'.format(scene_id, vi, vi + intervals)
        out_file = str(Path(save_path, stem + '.png'))
        print('\nWriting image to {}'.format(out_file))
        cv2.imwrite(out_file, out)
        out_file = str(Path(save_path, stem + '_vis.png'))
        cv2.imwrite(out_file, out_by_point)


def generate_cycle_mask(height, width, back_color, fore_color):
    x, y = np.ogrid[-height:height + 1, -width:width + 1]
    # ellipse mask
    cir_idx = ((x) ** 2 / (height ** 2) + (y) ** 2 / (width ** 2) <= 1)
    mask = np.zeros((2 * height + 1, 2 * width + 1, 3)).astype(np.uint8)
    mask[cir_idx == 0] = back_color
    mask[cir_idx == 1] = fore_color
    return mask


def save_inflow_outflow_density(img, scores, pre_points, target, match_gt, save_path, scene_id, vi, intervals):
    scores = scores.cpu().numpy()
    _, __, img_h, img_w = img.size()
    gt_inflow = np.zeros((img_h, img_w, 3)).astype(np.uint8)
    gt_outflow = np.zeros((img_h, img_w, 3)).astype(np.uint8)
    pre_inflow = np.zeros((img_h, img_w, 3)).astype(np.uint8)
    pre_outflow = np.zeros((img_h, img_w, 3)).astype(np.uint8)

    RoyalBlue1 = np.array([255, 118, 72])  # np.array([205,82,180])
    red = [0, 0, 255]
    green = [0, 255, 0]
    blue = [255, 0, 0]
    gt_inflow[:, :, 0:3] = RoyalBlue1
    gt_outflow[:, :, 0:3] = RoyalBlue1
    pre_inflow[:, :, 0:3] = RoyalBlue1
    pre_outflow[:, :, 0:3] = RoyalBlue1
    # matched_mask = np.zeros(scores.shape)
    # matched_mask[match_gt['a2b'][:, 0], match_gt['a2b'][:, 1]] = 1
    # matched_mask[match_gt['un_a'], -1] = 1
    # matched_mask[-1, match_gt['un_b']] = 1
    kernel = 8
    wide = 2 * kernel + 1

    pre_outflow_p = pre_points[0][scores[:-1, -1] > 0.4][:, 2:4]
    tp_pred_index, fp_pred_index, tp_gt_index, fn_gt_index = associate_pred2gt_point_vis(pre_outflow_p, target[0], match_gt['un_a'].cpu().numpy())

    # import pdb
    # pdb.set_trace()
    for row_id, pos in enumerate(pre_outflow_p, 0):
        w, h = pos.cpu().numpy().astype(np.int64)
        h_min, h_max = max(0, h - kernel), min(img_h, h + kernel + 1)
        w_min, w_max = max(0, w - kernel), min(img_w, w + kernel + 1)
        if row_id in tp_pred_index:
            mask = generate_cycle_mask(kernel, kernel, RoyalBlue1, red)
        if row_id not in tp_pred_index:
            mask = generate_cycle_mask(kernel, kernel, RoyalBlue1, green)
        pre_outflow[h_min:h_max, w_min:w_max] = mask[max(kernel - h, 0):wide - max(0, kernel + 1 + h - img_h),
                                                max(kernel - w, 0):wide - max(0, kernel + 1 + w - img_w)]
    for pos in (target[0]['points'][match_gt['un_a']][fn_gt_index]):
        w, h = pos.cpu().numpy().astype(np.int64)
        h_min, h_max = max(0, h - kernel), min(img_h, h + kernel + 1)
        w_min, w_max = max(0, w - kernel), min(img_w, w + kernel + 1)

        mask = generate_cycle_mask(kernel, kernel, RoyalBlue1, blue)
        pre_outflow[h_min:h_max, w_min:w_max] = mask[max(kernel - h, 0):wide - max(0, kernel + 1 + h - img_h),
                                                max(kernel - w, 0):wide - max(0, kernel + 1 + w - img_w)]

    # ================================pre_inflow========================
    pre_inflow_p = pre_points[1][scores[-1, :-1] > 0.4][:, 2:4]
    tp_pred_index, fp_pred_index, tp_gt_index, fn_gt_index = associate_pred2gt_point_vis(pre_inflow_p, target[1], match_gt['un_b'].cpu().numpy())

    for column_id, pos in enumerate(pre_inflow_p, 0):
        w, h = pos.cpu().numpy().astype(np.int64)
        h_min, h_max = max(0, h - kernel), min(img_h, h + kernel + 1)
        w_min, w_max = max(0, w - kernel), min(img_w, w + kernel + 1)
        if column_id in tp_pred_index:
            mask = generate_cycle_mask(kernel, kernel, RoyalBlue1, red)
        if column_id not in tp_pred_index:
            mask = generate_cycle_mask(kernel, kernel, RoyalBlue1, green)
        pre_inflow[h_min:h_max, w_min:w_max] = mask[max(kernel - h, 0):wide - max(0, kernel + 1 + h - img_h),
                                               max(kernel - w, 0):wide - max(0, kernel + 1 + w - img_w)]
    for pos in (target[1]['points'][match_gt['un_b']][fn_gt_index]):
        w, h = pos.cpu().numpy().astype(np.int64)
        h_min, h_max = max(0, h - kernel), min(img_h, h + kernel + 1)
        w_min, w_max = max(0, w - kernel), min(img_w, w + kernel + 1)

        mask = generate_cycle_mask(kernel, kernel, RoyalBlue1, blue)
        pre_inflow[h_min:h_max, w_min:w_max] = mask[max(kernel - h, 0):wide - max(0, kernel + 1 + h - img_h),
                                               max(kernel - w, 0):wide - max(0, kernel + 1 + w - img_w)]
    # pred_inflow_map = cv2.applyColorMap((255 * pre_inflow / (pre_inflow.max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)

    for row_id in match_gt['un_a'].cpu().numpy():
        # import pdb
        # pdb.set_trace()
        w, h = target[0]['points'][row_id].cpu().numpy().astype(np.int64)
        h_min, h_max = max(0, h - kernel), min(img_h, h + kernel + 1)
        w_min, w_max = max(0, w - kernel), min(img_w, w + kernel + 1)
        mask = generate_cycle_mask(kernel, kernel, RoyalBlue1, [0, 0, 255])
        gt_outflow[h_min:h_max, w_min:w_max] = mask[max(kernel - h, 0):wide - max(0, kernel + 1 + h - img_h),
                                               max(kernel - w, 0):wide - max(0, kernel + 1 + w - img_w)]

        print(w, h)
    # gt_outflow_map = cv2.applyColorMap((255 * gt_outflow / (gt_outflow.max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)

    for column_id in match_gt['un_b'].cpu().numpy():
        w, h = target[1]['points'][column_id].cpu().numpy().astype(np.int64)
        h_min, h_max = max(0, h - kernel), min(img_h, h + kernel + 1)
        w_min, w_max = max(0, w - kernel), min(img_w, w + kernel + 1)
        mask = generate_cycle_mask(kernel, kernel, RoyalBlue1, [0, 0, 255])
        gt_inflow[h_min:h_max, w_min:w_max] = mask[max(kernel - h, 0):wide - max(0, kernel + 1 + h - img_h),
                                              max(kernel - w, 0):wide - max(0, kernel + 1 + w - img_w)]
    # gt_inflow_map = cv2.applyColorMap((255 * gt_inflow / (gt_inflow.max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)

    os.makedirs(save_path, mode=0o777, exist_ok=True)
    stem = '{}_{}_{}_matches_outflow_pre_{}'.format(scene_id, vi, vi + intervals, np.round(scores[:-1, -1].sum(), 2))
    out_file = str(Path(save_path, stem + '.png'))
    print('\n Writing image to {}'.format(out_file))
    cv2.imwrite(out_file, pre_outflow)

    stem = '{}_{}_{}_matches_inflow_pre_{}'.format(scene_id, vi, vi + intervals, np.round(scores[-1, :-1].sum(), 2))
    out_file = str(Path(save_path, stem + '.png'))
    print('\n Writing image to {}'.format(out_file))
    cv2.imwrite(out_file, pre_inflow)

    stem = '{}_{}_{}_matches_outflow_gt_{}'.format(scene_id, vi, vi + intervals, match_gt['un_a'].size(0))
    out_file = str(Path(save_path, stem + '.png'))
    print('\n Writing image to {}'.format(out_file))
    cv2.imwrite(out_file, gt_outflow)

    stem = '{}_{}_{}_matches_inflow_gt_{}'.format(scene_id, vi, vi + intervals, match_gt['un_b'].size(0))
    out_file = str(Path(save_path, stem + '.png'))
    print('\n Writing image to {}'.format(out_file))
    cv2.imwrite(out_file, gt_inflow)
    # import pdb
    # pdb.set_trace()
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

    # ------------Start Training------------
    pwd = os.path.split(os.path.realpath(__file__))[0]
    test(cfg_data)

# ==================== all ====================
# MAE: 12.29, MSE: 24.71  WRAE: 12.73 WIAE: 1.98 WOAE: 2.01
# ==================== in ====================
# MAE: 12.61, MSE: 23.44  WRAE: 16.94 WIAE: 1.98 WOAE: 1.94
# ==================== out ====================
# MAE: 12.19, MSE: 25.09  WRAE: 11.44 WIAE: 1.97 WOAE: 2.04
# ==================== day ====================
# MAE: 11.78, MSE: 22.94  WRAE: 12.50 WIAE: 2.05 WOAE: 2.02
# ==================== night ====================
# MAE: 14.06, MSE: 30.04  WRAE: 13.53 WIAE: 1.72 WOAE: 2.00
# ==================== scenic0 ====================
# MAE: 8.35, MSE: 17.04  WRAE: 9.61 WIAE: 1.55 WOAE: 1.80
# ==================== scenic1 ====================
# MAE: 11.15, MSE: 20.78  WRAE: 15.61 WIAE: 1.86 WOAE: 1.87
# ==================== scenic2 ====================
# MAE: 18.08, MSE: 29.65  WRAE: 17.41 WIAE: 2.29 WOAE: 2.13
# ==================== scenic3 ====================
# MAE: 33.56, MSE: 50.17  WRAE: 18.21 WIAE: 4.82 WOAE: 3.76
# ==================== scenic4 ====================
# MAE: 11.21, MSE: 21.40  WRAE: 13.06 WIAE: 1.97 WOAE: 1.69
# ==================== scenic5 ====================
# MAE: 29.91, MSE: 50.06  WRAE: 21.60 WIAE: 3.96 WOAE: 3.77
# ==================== density0 ====================
# MAE: 4.08, MSE: 5.75  WRAE: 12.58 WIAE: 1.08 WOAE: 1.21
# ==================== density1 ====================
# MAE: 7.99, MSE: 11.13  WRAE: 10.47 WIAE: 1.68 WOAE: 1.77
# ==================== density2 ====================
# MAE: 23.25, MSE: 32.89  WRAE: 14.51 WIAE: 3.14 WOAE: 3.07
# ==================== density3 ====================
# MAE: 50.03, MSE: 64.07  WRAE: 19.93 WIAE: 6.00 WOAE: 5.10
# ==================== density4 ====================
# MAE: 76.95, MSE: 83.93  WRAE: 24.46 WIAE: 6.32 WOAE: 5.99
# tensor([[ 89.4147,  93.0000],
#         [148.2149, 259.0000],
#         [ 29.7278,  37.0000],
#         [ 71.6076,  87.0000],
#         [ 38.6680,  43.0000],
#         [ 61.2983,  65.0000],
#         [ 70.3064,  77.0000],
#         [154.6082, 227.0000],
#         [ 48.5384,  50.0000],
#         [ 40.7733,  44.0000],
#         [172.4883, 284.0000],
#         [ 65.9117,  77.0000],
#         [164.5734, 232.0000],
#         [ 11.7632,  12.0000],
#         [ 41.9911,  39.0000],
#         [ 73.3622,  81.0000],
#         [ 15.7980,  22.0000],
#         [ 58.1445,  76.0000],
#         [ 78.0700,  81.0000],
#         [ 96.6829, 100.0000],
#         [123.0839, 127.0000],
#         [ 41.0895,  40.0000],
#         [ 46.0548,  43.0000],
#         [ 50.4754,  50.0000],
#         [ 54.0197,  53.0000],
#         [ 23.2172,  22.0000],
#         [ 75.5792,  69.0000],
#         [ 84.4526,  89.0000],
#         [121.0936, 115.0000],
#         [243.7008, 306.0000],
#         [ 47.3643,  45.0000],
#         [ 43.6313,  62.0000],
#         [ 77.0213,  71.0000],
#         [ 62.3260,  71.0000],
#         [ 55.6199,  54.0000],
#         [ 58.6624,  55.0000],
#         [ 37.3744,  35.0000],
#         [ 27.4079,  30.0000],
#         [ 38.2823,  41.0000],
#         [ 83.2303,  78.0000],
#         [ 26.2175,  31.0000],
#         [147.9063, 164.0000],
#         [116.8620, 160.0000],
#         [ 29.7327,  35.0000],
#         [ 27.4655,  26.0000],
#         [ 75.3924,  86.0000],
#         [157.0550, 180.0000],
#         [111.5766, 191.0000],
#         [163.5789, 171.0000],
#         [ 80.2234,  75.0000],
#         [ 88.8087, 103.0000],
#         [ 35.8673,  34.0000],
#         [107.7521, 104.0000],
#         [ 77.2214,  97.0000],
#         [ 73.8145,  70.0000],
#         [ 55.1942,  61.0000],
#         [221.1072, 237.0000],
#         [ 52.5339,  60.0000],
#         [ 64.5467,  71.0000],
#         [ 68.5728,  71.0000],
#         [ 53.1565,  47.0000],
#         [ 94.2252, 100.0000],
#         [ 57.4984,  49.0000],
#         [ 37.9518,  29.0000],
#         [ 40.8593,  33.0000],
#         [ 35.7516,  35.0000],
#         [ 80.8944,  98.0000],
#         [111.9066, 135.0000],
#         [ 30.5592,  30.0000],
#         [ 33.7159,  37.0000],
#         [ 99.1224, 101.0000],
#         [136.8730, 232.0000],
#         [ 71.0780,  69.0000],
#         [ 21.6167,  28.0000],
#         [ 46.0675,  45.0000],
#         [ 22.1183,  37.0000],
#         [ 52.8302,  54.0000],
#         [ 19.6272,  23.0000],
#         [ 35.3046,  34.0000],
#         [ 39.5665,  57.0000],
#         [ 28.2371,  16.0000],
#         [ 75.8426,  77.0000],
#         [ 26.7141,  29.0000],
#         [114.5060, 134.0000],
#         [ 72.5601,  88.0000],
#         [ 68.5220,  67.0000],
#         [112.4594, 130.0000],
#         [ 84.0589,  62.0000],
#         [227.6907, 329.0000],
#         [ 33.1534,  35.0000],
#         [ 48.8285,  47.0000],
#         [ 58.0883,  52.0000],
#         [ 57.2759,  52.0000],
#         [ 45.3223,  46.0000],
#         [ 68.1864,  64.0000],
#         [120.8597, 169.0000],
#         [ 59.3080,  64.0000],
#         [191.3976, 250.0000],
#         [175.0085, 233.0000],
#         [ 10.6638,  11.0000],
#         [ 80.5845,  97.0000],
#         [ 81.8619,  96.0000],
#         [ 35.5707,  31.0000],
#         [232.4688, 316.0000],
#         [ 70.8171,  81.0000],
#         [ 39.7070,  41.0000],
#         [ 55.2743,  57.0000],
#         [ 87.0791, 108.0000],
#         [177.9434, 190.0000],
#         [ 91.6090,  83.0000],
#         [ 51.4202,  48.0000],
#         [ 73.2659,  74.0000],
#         [ 47.3627,  70.0000],
#         [144.2767, 175.0000],
#         [ 54.6253,  63.0000],
#         [ 98.3246,  97.0000],
#         [ 56.5126,  61.0000],
#         [ 38.2883,  23.0000],
#         [ 70.7073,  69.0000],
#         [ 38.0360,  22.0000],
#         [ 53.3014,  57.0000],
#         [ 65.6217,  69.0000],
#         [ 45.7053,  53.0000],
#         [ 49.8301,  44.0000],
#         [ 23.4349,  22.0000],
#         [ 75.9271,  76.0000],
#         [ 76.7114,  76.0000],
#         [ 24.8107,  26.0000],
#         [105.8516, 116.0000],
#         [ 72.5226,  73.0000],
#         [ 68.9267,  68.0000],
#         [ 66.7460, 152.0000],
#         [ 48.6040,  56.0000],
#         [ 54.7598,  62.0000],
#         [ 65.3288,  75.0000],
#         [ 44.4746,  50.0000],
#         [158.6062, 230.0000],
#         [ 67.7573,  73.0000],
#         [ 49.4845,  37.0000],
#         [ 87.4544,  71.0000],
#         [ 86.9877,  86.0000],
#         [ 85.2368,  80.0000],
#         [ 77.9463,  75.0000],
#         [ 58.9658,  77.0000],
#         [ 51.8347,  53.0000],
#         [123.8188, 142.0000],
#         [112.5238, 164.0000],
#         [ 69.3425,  74.0000],
#         [ 62.7756,  67.0000],
#         [110.3728, 108.0000],
#         [ 50.3846,  58.0000],
#         [164.9739, 202.0000],
#         [ 26.0592,  23.0000],
#         [ 98.3755,  97.0000],
#         [ 64.1170,  62.0000],
#         [ 23.0675,  24.0000],
#         [ 48.8955,  59.0000],
#         [ 53.0321,  52.0000],
#         [ 45.3135,  43.0000],
#         [ 95.5922,  98.0000],
#         [ 46.4823,  43.0000],
#         [ 56.9812,  84.0000],
#         [172.6967, 179.0000],
#         [ 80.6760,  73.0000],
#         [ 25.7448,  25.0000],
#         [ 16.1685,  26.0000],
#         [ 62.0806,  55.0000],
#         [ 93.0958, 102.0000],
#         [ 37.5434,  51.0000],
#         [ 20.9470,  21.0000],
#         [ 42.8672,  57.0000],
#         [ 37.6022,  22.0000],
#         [ 26.5846,  26.0000],
#         [ 34.8923,  37.0000],
#         [ 42.0879,  46.0000],
#         [ 41.3744,  45.0000],
#         [116.4209, 131.0000],
#         [143.4008, 142.0000],
#         [ 28.7674,  29.0000],
#         [ 71.7286,  66.0000],
#         [ 42.6456,  69.0000],
#         [ 47.6907,  44.0000],
#         [ 26.2184,  22.0000],
#         [ 20.1989,  23.0000],
#         [ 36.0446,  36.0000],
#         [ 58.5359,  58.0000],
#         [ 17.0322,  19.0000],
#         [ 55.8627,  75.0000],
#         [126.8426, 124.0000],
#         [159.0815, 184.0000],
#         [ 37.2157,  38.0000],
#         [ 76.2001,  92.0000],
#         [106.7688, 105.0000],
#         [ 63.7852,  64.0000],
#         [247.1487, 312.0000],
#         [ 31.1691,  32.0000],
#         [ 68.5482,  67.0000],
#         [ 84.2573,  76.0000],
#         [ 64.7561,  67.0000],
#         [110.7983, 120.0000],
#         [ 51.1362,  50.0000],
#         [ 14.2311,  19.0000],
#         [ 89.2847,  87.0000],
#         [122.7155, 142.0000],
#         [ 53.4915,  54.0000],
#         [ 80.5151,  89.0000],
#         [ 16.3571,  18.0000],
#         [ 44.3656,  37.0000],
#         [118.4887, 146.0000],
#         [ 30.9687,  30.0000],
#         [ 99.9197, 108.0000],
#         [ 83.3560,  93.0000],
#         [ 40.8250,  50.0000],
#         [ 49.4679,  41.0000],
#         [264.1026, 390.0000],
#         [ 14.9821,  13.0000],
#         [ 86.6475,  91.0000],
#         [123.1816, 122.0000],
#         [ 32.7715,  36.0000],
#         [ 26.4360,  26.0000],
#         [ 49.7715,  52.0000],
#         [ 20.3532,  18.0000],
#         [ 20.4764,  18.0000],
#         [ 35.3715,  33.0000],
#         [ 75.1884,  77.0000],
#         [119.4564, 129.0000],
#         [ 47.2260,  46.0000],
#         [ 58.8956,  67.0000],
#         [ 58.2124,  69.0000],
#         [ 26.4284,  27.0000],
#         [133.6892, 137.0000],
#         [ 40.5817,  42.0000],
#         [ 32.0720,  35.0000],
#         [ 90.3207,  98.0000],
#         [ 68.1723,  77.0000],
#         [ 57.4718,  68.0000],
#         [ 21.4721,  25.0000],
#         [129.9447, 255.0000],
#         [137.8681, 112.0000],
#         [ 60.6594,  61.0000],
#         [ 26.3088,  43.0000],
#         [109.1689, 110.0000],
#         [ 77.0445,  84.0000],
#         [ 48.9617,  50.0000],
#         [ 84.1961,  81.0000],
#         [ 96.4772,  87.0000],
#         [ 56.4531,  61.0000],
#         [ 38.2997,  45.0000],
#         [ 64.8246,  63.0000],
#         [ 50.9867,  54.0000],
#         [ 26.2117,  25.0000],
#         [ 48.4944,  49.0000],
#         [ 45.4879,  79.0000],
#         [ 36.5422,  51.0000]])