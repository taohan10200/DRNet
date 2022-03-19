import datasets
from  config import cfg
import numpy as np
import torch
from torch import optim
import datasets
from misc.utils import *
from model.VIC import Video_Crowd_Counting
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
    '--DATA_PATH', type=str, default='/data/GJY/ht/CVPR2022/dataset/HT21/train',
    help='Directory where to write output frames (If None, no output)')
parser.add_argument(
    '--output_dir', type=str, default='/data/GJY/ht/CVPR2022/dataset/demo_video_HT/pred',
    help='Directory where to write output frames (If None, no output)')
parser.add_argument(
    '--test_intervals', type=int, default=75,
    help='Directory where to write output frames (If None, no output)')
parser.add_argument(
    '--skip_flag', type=bool, default=False,
    help='To caculate the MIAE and MOAE, it should be False')
parser.add_argument(
    '--SEED', type=int, default=3035,
    help='Directory where to write output frames (If None, no output)')
parser.add_argument(
    '--GPU_ID', type=str, default='0',
    help='Directory where to write output frames (If None, no output)')
# parser.add_argument(
#     '--model_path', type=str,
#     default='../exp/HT21/11-07_01-55_HT21_VGG16_FPN_5e-05_(full model)/ep_5_iter_12500_mae_9.797_mse_10.438_seq_MAE_38.768_WRAE_44.561_MIAE_5.209_MOAE_5.535.pth',
#     help='pretrained weight path')

# parser.add_argument(
#     '--model_path', type=str,
#     default='/data/GJY/ht/CVPR2022/VCC/exp/SENSE/11-06_02-04_SENSE_VGG16_FPN_5e-05(full mode)/ep_17_iter_262500_mae_2.141_mse_3.491_seq_MAE_5.763_WRAE_10.162_MIAE_1.623_MOAE_1.702.pth',
#     help='pretrained weight path')
#

parser.add_argument(
    '--model_path', type=str,
    default='/data/GJY/ht/CVPR2022/VCC/exp/HT21/11-07_01-55_HT21_VGG16_FPN_5e-05_(full model)/ep_15_iter_40000_mae_4.535_mse_5.191_seq_MAE_75.099_WRAE_86.321_MIAE_8.198_MOAE_9.289.pth',
    help='pretrained weight path')

opt = parser.parse_args()
from datasets.dataset import TestDataset,Dataset
from torch.utils.data import  DataLoader
import torchvision.transforms as standard_transforms
import misc.transforms as own_transforms
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return tuple(zip(*batch))
def createRestore(mean_std):
    return standard_transforms.Compose([
        own_transforms.DeNormalize(*mean_std),
        standard_transforms.ToPILImage()
    ])

def test(cfg_data):
    img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*cfg_data.MEAN_STD)
    ])
    sub_dataset = TestDataset(scene_name='HT21-02',
                          base_path=opt.DATA_PATH,
                          main_transform=None,
                          img_transform=img_transform,
                          interval=opt.test_intervals,
                          target=True,
                          datasetname=opt.DATASET)
    test_loader = DataLoader(sub_dataset, batch_size=cfg_data.VAL_BATCH_SIZE,
                            collate_fn=collate_fn, num_workers=0, pin_memory=True)
    restore_transform = createRestore(cfg_data.MEAN_STD)

    net = Video_Crowd_Counting(cfg, cfg_data)

    # latest_state = torch.load(cfg.RESUME_PATH)
    # net.load_state_dict(latest_state['net'], strict=True)
    state_dict = torch.load(opt.model_path)
    net.load_state_dict(state_dict, strict=True)
    net.eval()

    scenes_pred_dict = []
    if opt.skip_flag:
        intervals = 1
    else:
        intervals = opt.test_intervals
    for scene_id, sub_valset in enumerate([test_loader], 0):
        gen_tqdm = tqdm(sub_valset)
        video_time = len(sub_valset) + opt.test_intervals
        print(video_time)
        pred_dict = {'id': scene_id, 'time': video_time, 'first_frame': 0, 'inflow': [], 'outflow': []}
        gt_dict = {'id': scene_id, 'time': video_time, 'first_frame': 0, 'inflow': [], 'outflow': []}
        time = []
        cnt  = []
        gt_cnt = []
        for vi, data in enumerate(gen_tqdm, 0):
            img,target = data
            img,target = img[0],target[0]
            # if vi>=21:break
            # import  pdb
            # pdb.set_trace()
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

                    pred_map, gt_den, matched_results = net.val_forward(img,target, frame_signal)
                    # import pdb
                    # pdb.set_trace()
                    #    -----------Counting performance------------------
                    pred_cnt = pred_map[0].sum().item()
                    ##=====================================================

                    if vi == 0:
                        pred_dict['first_frame'] = pred_map[0].sum().item()
                        gt_dict['first_frame'] = len(target[0]['person_id'])

                    pred_dict['inflow'].append(matched_results['pre_inflow'])
                    pred_dict['outflow'].append(matched_results['pre_outflow'])
                    gt_dict['inflow'].append(matched_results['gt_inflow'])
                    gt_dict['outflow'].append(matched_results['gt_outflow'])
                time.append(round(vi/intervals*3., 2))
                if frame_signal == 'match':
                    pre_crowdflow_cnt, gt_crowdflow_cnt, _, _ = compute_metrics_single_scene(pred_dict, gt_dict, intervals = intervals)
                    cnt.append(pre_crowdflow_cnt)
                    gt_cnt.append(gt_crowdflow_cnt)

                    print(' den_pre:  %.2f pre_crowd_flow: %.2f pre_inflow:%.2f'
                          %  (pred_cnt, pre_crowdflow_cnt,matched_results['pre_inflow']))
                else:
                    cnt.append(cnt[-1])
                    gt_cnt.append(gt_cnt[-1])
                kpts0 = matched_results['pre_points'][0][:, 2:4].cpu().numpy()
                kpts1 = matched_results['pre_points'][1][:, 2:4].cpu().numpy()

                matches = matched_results['matches0'].cpu().numpy()
                confidence = matched_results['matching_scores0'].cpu().numpy()
                scores = matched_results['scores'].cpu().numpy()
                if kpts0.shape[0] > 0 and kpts1.shape[0] > 0:
                    save_visImg(kpts0, kpts1, matches, confidence, vi, img[0].clone(), img[1].clone(),
                                opt.test_intervals, opt.output_dir, time, cnt,gt_cnt, scene_id, scores,pred_dict,restore_transform)

        scenes_pred_dict.append(pred_dict)
    # import pdb
    # pdb.set_trace()

    # np.save('scene_cnt.py',scene_cnt)

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
# def compute_metrics_single_scene(pre_dict, intervals):
#     pair_cnt = len(pre_dict['inflow'])
#     inflow_cnt, outflow_cnt =torch.zeros(pair_cnt,2), torch.zeros(pair_cnt,2)
#     pre_crowdflow_cnt  = pre_dict['first_frame']
#
#     for idx, data in enumerate(zip(pre_dict['inflow'],  pre_dict['outflow']),0):
#         inflow_cnt[idx, 0] = data[0]
#         outflow_cnt[idx, 0] = data[1]
#         if idx %intervals == 0 or  idx== len(pre_dict['inflow'])-1:
#             pre_crowdflow_cnt += data[0]
#
#
#     return pre_crowdflow_cnt,  inflow_cnt, outflow_cnt

import matplotlib.pyplot as plt
def make_matching_plot_fast(image0, image1, kpts0, kpts1, mkpts0,
                            mkpts1, color, text, path=None,
                            show_keypoints=False, margin=10,
                            opencv_display=False, opencv_title='',
                            small_text = [], restore_transform=None,
                            id0=None,id1=None, scores=None
                            ):
    image0 = np.array(restore_transform(image0))
    image1 = np.array(restore_transform(image1))
    image0 = cv2.cvtColor(image0, cv2.COLOR_RGB2BGR)
    image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)
    H0, W0, C = image0.shape
    H1, W1, C = image1.shape
    pre_inflow = np.zeros((H0, W0, 3)).astype(np.uint8)
    pre_outflow = np.zeros((H1, W1, 3)).astype(np.uint8)

    H, W = max(H0, H1)+50, W0 + 1600 + margin

    out = 255*np.ones((H, W, C), np.uint8)
    out[:H0, :W0,:] = image1
    # out[:H1, W0+margin:,:] = image1
    # out = np.stack([out]*3, -1)
    # import pdb
    # pdb.set_trace()
    out_by_point = out.copy()
    point_r_value = 10
    thickness = 3
    white = (255, 255, 255)
    RoyalBlue1 = np.array([255, 118, 72])  # np.array([205,82,180])
    red = [0, 0, 255]
    green = [0, 255, 0]
    blue = [255, 0, 0]
    pre_inflow[:, :, 0:3] = RoyalBlue1
    pre_outflow[:, :, 0:3] = RoyalBlue1

    kernel = 8
    wide = 2 * kernel + 1

    # ===================begin: inflow outflow map ================
    # pre_outflow_p = kpts0[scores[:-1, -1] > 0.2]
    # scores_= scores[:-1, -1][scores[:-1, -1] > 0.2]
    # for row_id, (pos,s) in enumerate(zip(pre_outflow_p, scores_), 0):
    #     w, h = pos.astype(np.int64)
    #     h_min, h_max = max(0, h - kernel), min(H0, h + kernel + 1)
    #     w_min, w_max = max(0, w - kernel), min(W0, w + kernel + 1)
    #     red_ = [0,0,int(255*s)]
    #     mask = generate_cycle_mask(kernel, kernel, RoyalBlue1, red_)
    #
    #     pre_outflow[h_min:h_max, w_min:w_max] = mask[max(kernel - h, 0):wide - max(0, kernel + 1 + h - H0),
    #                                             max(kernel - w, 0):wide - max(0, kernel + 1 + w - W0)]
    # # ================================pre_inflow========================
    # pre_inflow_p = kpts1[scores[-1, :-1] > 0.2]
    # scores_ = scores[-1, :-1][scores[-1, :-1] > 0.2]
    # for column_id, (pos,s) in enumerate(zip(pre_inflow_p,scores_), 0):
    #     w, h = pos.astype(np.int64)
    #     h_min, h_max = max(0, h - kernel), min(H0, h + kernel + 1)
    #     w_min, w_max = max(0, w - kernel), min(W0, w + kernel + 1)
    #     red_ = [0, 0, int(255 * s)]
    #     mask = generate_cycle_mask(kernel, kernel, RoyalBlue1, red_)
    #     pre_inflow[h_min:h_max, w_min:w_max] = mask[max(kernel - h, 0):wide - max(0, kernel + 1 + h - H0),
    #                                            max(kernel - w, 0):wide - max(0, kernel + 1 + w - W0)]
    # out[H0:, :W0, :] = pre_outflow
    # out[H1:, W0 + margin:, :] = pre_inflow
    #===================end: inflow outflow map ================

    if show_keypoints:
        kpts0, kpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)
        for x, y in kpts1:
            cv2.circle(out, (x, y), point_r_value, red, thickness, lineType=cv2.LINE_AA)
            cv2.circle(out, (x, y), 3, white, -1, lineType=cv2.LINE_AA)

            # cv2.circle(out_by_point, (x, y), point_r_value, red, thickness, lineType=cv2.LINE_AA)

        # for x, y in kpts1:
        #     cv2.circle(out, (x + margin + W0, y), point_r_value, red, thickness,
        #                lineType=cv2.LINE_AA)
        #     cv2.circle(out, (x + margin + W0, y), 3, white, -1, lineType=cv2.LINE_AA)
        #
        #     cv2.circle(out_by_point, (x + margin + W0, y), point_r_value, blue, thickness,
        #                lineType=cv2.LINE_AA)


    mkpts0, mkpts1 = np.round(mkpts0).astype(int), np.round(mkpts1).astype(int)
    color = (np.array(color[:, :3])*255).astype(int)[:, ::-1]

    for (x0, y0), (x1, y1), c in zip(mkpts0, mkpts1, color):
        c = c.tolist()
        # cv2.line(out, (x0, y0), (x1 + margin + W0, y1),
        #          color=c, thickness=1, lineType=cv2.LINE_AA)
        # display line end-points as circles
        cv2.circle(out, (x1, y1), point_r_value, green, thickness, lineType=cv2.LINE_AA)
        # cv2.circle(out, (x1 + margin + W0, y1), point_r_value, green, thickness,
        #            lineType=cv2.LINE_AA)
        #
        # cv2.circle(out_by_point, (x0, y0), point_r_value, green, thickness, lineType=cv2.LINE_AA)
        # cv2.circle(out_by_point, (x1 + margin + W0, y1), point_r_value, green, thickness,
        #            lineType=cv2.LINE_AA)

    # Ht = int(H*10 / 480)  # text height
    # txt_color_fg = (255, 255, 255)
    # txt_color_bg = (0, 0, 0)
    # text_S_H = 65
    #
    # for i, t in enumerate(text):
    #     if i == 0:
    #         cv2.putText(out, t, (W0-600, H0+text_S_H), cv2.FONT_HERSHEY_DUPLEX,
    #                     H*1.2/1080, txt_color_fg, 2, cv2.LINE_AA)
    #     if i == 1:
    #         cv2.putText(out, t, (30, H0 + text_S_H), cv2.FONT_HERSHEY_DUPLEX,
    #                     H * 1.2 / 1080, txt_color_fg, 2, cv2.LINE_AA)
    #     if i == 2:
    #         cv2.putText(out, t, (2*W0-800, H0 + text_S_H), cv2.FONT_HERSHEY_DUPLEX,
    #                     H * 1.2 / 1080, txt_color_fg, 2, cv2.LINE_AA)
    #     if i == 3:
    #         cv2.putText(out, t, (30, H0 + int(text_S_H*2.5)), cv2.FONT_HERSHEY_DUPLEX,
    #                     H * 1.2 / 1080, txt_color_fg, 2, cv2.LINE_AA)
    #
    #     if i == 4:
    #         cv2.putText(out, t, (2*W0-800, H0 + int(text_S_H*2.5)), cv2.FONT_HERSHEY_DUPLEX,
    #                     H * 1.2 / 1080, txt_color_fg, 2, cv2.LINE_AA)
        # cv2.putText(out, t, (10, Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
        #             H*1.0/480, txt_color_fg, 1, cv2.LINE_AA)
        # cv2.putText(out_by_point, t, (10, Ht * (i + 1)), cv2.FONT_HERSHEY_DUPLEX,
        #         H * 1.0 / 480, txt_color_fg, 1, cv2.LINE_AA)
    if path is not None:
        cv2.imwrite(str(path), out)
        cv2.imwrite(str('point_'+path), out_by_point)
    if opencv_display:
        cv2.imshow(opencv_title, out)
        cv2.waitKey(1)

    return out,out_by_point

from matplotlib.backends.backend_agg import FigureCanvasAgg
import PIL.Image as Image
def save_visImg( kpts0, kpts1, matches, confidence, vi, last_frame, cur_frame, intervals,
                save_path, time=None, cnt=None, gt_cnt=None, scene_id='', scores = None, pred_dict =None, restore_transform=None):
    valid = matches > -1
    mkpts0 = kpts0[valid].reshape(-1, 2)
    mkpts1 = kpts1[matches[valid]].reshape(-1, 2)
    color = cm.jet(confidence[valid])


    # for  i in range(number - number%intervals, number):
    #     inflow_cnt +=(pred_dict['inflow'][i] - pred_dict['inflow'][i-1])
    #     total_cnt  += (pred_dict['inflow'][i] - pred_dict['inflow'][i-1])
    text = [
        'pedestrian_cnt: {}'.format(np.around(cnt[-1],2)),
        'previous_frame: {}s'.format(vi/intervals, 2),
        'inflow_tmp: {}'.format(np.around(pred_dict['inflow'][-1], 2)),
        'outflow_tmp: {}'.format(np.around(pred_dict['outflow'][-1], 2)),
        'inflow_tmp: {}'.format(np.around(pred_dict['inflow'][-1], 2)),

    ]

    def fig2data(fig):
        fig.canvas.draw()
        # 获取图像尺寸
        w, h = fig.canvas.get_width_height()
        print(fig.canvas.get_width_height())
        # 获取 argb 图像
        buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
        buf.shape = (w, h, 4)

        # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
        buf = np.roll(buf, 3, axis=2)
        image = Image.frombytes("RGBA", (w, h), buf.tostring())
        image = np.asarray(image)
        rgb_image = image[:, :, :3]
        return  rgb_image
    fig, ax =plt.subplots(figsize = (16,10))
    ax.cla()  # clear plot
    plt.tick_params(labelsize=28) #设置刻度字体
    plt.xlim(0, 130)
    plt.ylim(0, 1400)


    ax.set_xlabel('Time (s)', fontsize=32, fontfamily='Times New Roman')
    ax.set_ylabel('pedestrian number',fontsize=32, fontfamily='Times New Roman')
    # print(time, cnt)
    ax.plot(time, cnt, 'b', lw=4)  # draw line chart
    ax.plot(time, gt_cnt, 'r', lw=4)  # draw line chart
    plt.legend(['Predicted: ' + str(np.around(cnt[-1],2)), 'Ground Truth: '+str(np.around(gt_cnt[-1],2))],
               loc = 'upper left', fontsize=32)
    # plt.text(3, 950, r"Pedestrian number: $N= N_{0} + \sum N_{in}$", fontsize=32, fontfamily='Times New Roman')
    plt.show()
    curve = fig2data(fig)

    out, out_by_point = make_matching_plot_fast(
        last_frame, cur_frame, kpts0, kpts1, mkpts0, mkpts1, color, text,
        path=None, show_keypoints=True, small_text=None, restore_transform=restore_transform,
        id0=None, id1=None,scores=scores)
    #=== cancat========
    H0, W0, _  = out.shape
    H1, W1, _  = curve.shape
    H_s = int((H0-H1)/2)
    out[H_s:H_s+H1, W0-W1:,:] = curve
    txt_color_fg = (255, 255, 255)
    red = (0, 0, 255)
    cv2.circle(out, (20, 1080+27), 10, red, thickness=4, lineType=cv2.LINE_AA)
    cv2.putText(out, 'People who come into the scene during time: [' +str(round(time[-1]-3,1)) +'s~'+ str(round(time[-1],1)) +'s]',
                (40, 1080+42), cv2.FONT_HERSHEY_DUPLEX, 1.5, [0,0,0], 2, cv2.LINE_AA)

    cv2.circle(out, (1700, 1080+27), 10, [0,255,0], thickness=4, lineType=cv2.LINE_AA)
    cv2.putText(out, 'People who still stay in the scene compared with time: ' +str(round(time[-1]-3,1)) +'s',
                (1740, 1080+42), cv2.FONT_HERSHEY_DUPLEX, 1.5, [0,0,0], 2, cv2.LINE_AA)

    # for i, t in enumerate(text):
    #     if i == 0:
    #         cv2.putText(out, t, (600, 70), cv2.FONT_HERSHEY_DUPLEX,
    #                     2, txt_color_fg, 2, cv2.LINE_AA)
    if save_path is not None:
        # print('==> Will write outputs to {}'.format(save_path))

        os.makedirs(save_path, mode=0o777, exist_ok=True)
        stem = '{}_{}_matches'.format(vi, vi + intervals)
        out_file = str(Path(save_path, stem + '.jpg'))
        print('\nWriting image to {}'.format(out_file))
        cv2.imwrite(out_file, out)
        # out_file = str(Path(save_path, stem + '_vis.jpg'))
        # cv2.imwrite(out_file, curve)

def generate_cycle_mask(height, width, back_color, fore_color):
    x, y = np.ogrid[-height:height + 1, -width:width + 1]
    # ellipse mask
    cir_idx = ((x) ** 2 / (height ** 2) + (y) ** 2 / (width ** 2) <= 1)
    mask = np.zeros((2 * height + 1, 2 * width + 1, 3)).astype(np.uint8)
    mask[cir_idx == 0] = back_color
    mask[cir_idx == 1] = fore_color
    return mask
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
