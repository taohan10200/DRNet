import os
import math
import numpy as np
import time
import random
import shutil
import cv2
from PIL import Image
import pdb
import torch
from torch import nn
import torchvision.utils as vutils
import torchvision.transforms as standard_transforms

def adjust_learning_rate(optimizer, epoch,base_lr1=0, base_lr2=0, power=0.9):
    lr1 =  base_lr1 * power ** ((epoch-1))
    lr2 =  base_lr2 * power ** ((epoch - 1))
    optimizer.param_groups[0]['lr'] = lr1
    optimizer.param_groups[1]['lr'] = lr2
    return lr1 , lr2


def weights_normal_init(*models):
    for model in models:
        dev=0.01
        if isinstance(model, list):
            for m in model:
                weights_normal_init(m, dev)
        else:
            for m in model.modules():            
                if isinstance(m, nn.Conv2d):        
                    m.weight.data.normal_(0.0, dev)
                    if m.bias is not None:
                        m.bias.data.fill_(0.0)
                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0.0, dev)


def logger(exp_path, exp_name, work_dir, exception, resume=False):

    from tensorboardX import SummaryWriter
    
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    writer = SummaryWriter(exp_path+ '/' + exp_name)
    log_file = exp_path + '/' + exp_name + '/' + exp_name + '.txt'
    
    cfg_file = open('./config.py',"r")  
    cfg_lines = cfg_file.readlines()
    
    with open(log_file, 'a') as f:
        f.write(''.join(cfg_lines) + '\n\n\n\n')

    if not resume:
        copy_cur_env(work_dir, exp_path+ '/' + exp_name + '/code', exception)

    return writer, log_file


def logger_txt(log_file,epoch,scores):
    snapshot_name = 'ep_%d' % epoch
    for key, data in scores.items():
        snapshot_name+= ('_'+ key+'_%3f'%data)
    with open(log_file, 'a') as f:
        f.write('='*15 + '+'*15 + '='*15 + '\n\n')
        f.write(snapshot_name + '\n')
        f.write('[')
        for key, data in scores.items():
            f.write(' '+ key+' %.2f'% data)
        f.write('\n')
        f.write('='*15 + '+'*15 + '='*15 + '\n\n')


def save_results_more(iter, exp_path, restore, img, pred_map, gt_map, binar_map,threshold_matrix,Instance_weights, boxes=None):  # , flow):

    pil_to_tensor = standard_transforms.ToTensor()

    UNIT_H , UNIT_W = img.size(2), img.size(3)
    for idx, tensor in enumerate(zip(img.cpu().data, pred_map, gt_map, binar_map, threshold_matrix,Instance_weights)):
        if idx > 1:  # show only one group
            break
        pil_input = restore(tensor[0])
        pred_color_map = cv2.applyColorMap((255 * tensor[1] / (tensor[2].max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)
        gt_color_map = cv2.applyColorMap((255 * tensor[2] / (tensor[2].max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)
        binar_color_map = cv2.applyColorMap((255 * tensor[3] / (tensor[4].max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)
        gt_matched_color_map = cv2.applyColorMap((255 * tensor[4]/ (tensor[4].max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)
        weights_color_map = cv2.applyColorMap((255 * tensor[5] / (tensor[5].max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)

        point_color = (0, 255, 0)  # BGR
        thickness = 1
        lineType = 4
        pil_input = np.array(pil_input)

        # for i, box in enumerate(boxes, 0):
        #     wh_LeftTop = (box[0], box[1])
        #     wh_RightBottom = (box[2], box[3])
        #     cv2.rectangle(binar_color_map, wh_LeftTop, wh_RightBottom, point_color, thickness, lineType)
        #     cv2.rectangle(pil_input, wh_LeftTop, wh_RightBottom, point_color, thickness, lineType)

        pil_input = Image.fromarray(pil_input)
        pil_label = Image.fromarray(cv2.cvtColor(gt_color_map, cv2.COLOR_BGR2RGB))
        pil_output = Image.fromarray(cv2.cvtColor(pred_color_map, cv2.COLOR_BGR2RGB))
        pil_binar = Image.fromarray(cv2.cvtColor(binar_color_map, cv2.COLOR_BGR2RGB))
        pil_gt_matched = Image.fromarray(cv2.cvtColor(gt_matched_color_map, cv2.COLOR_BGR2RGB))
        pil_weights = Image.fromarray(cv2.cvtColor(weights_color_map, cv2.COLOR_BGR2RGB))

        imgs = [pil_input, pil_label, pil_output, pil_binar, pil_gt_matched,pil_weights]

        w_num , h_num=3, 2

        target_shape = (w_num * (UNIT_W + 10), h_num * (UNIT_H + 10))
        target = Image.new('RGB', target_shape)
        count = 0
        for img in imgs:
            x, y = int(count%w_num) * (UNIT_W + 10), int(count // w_num) * (UNIT_H + 10)  # 左上角坐标，从左到右递增
            target.paste(img, (x, y, x + UNIT_W, y + UNIT_H))
            count+=1

        target.save(os.path.join(exp_path,'{}_den.jpg'.format(iter)))
        # cv2.imwrite('./exp/{}_vis_.png'.format(iter), img)


def vis_results_more(exp_name, epoch, writer, restore, img, pred_map, gt_map, binar_map,threshold_matrix, pred_boxes, gt_boxes):

    pil_to_tensor = standard_transforms.ToTensor()

    x = []
    y = []

    for idx, tensor in enumerate(zip(img.cpu().data, pred_map, gt_map, binar_map, threshold_matrix)):
        if idx > 1:  # show only one group
            break

        pil_input = restore(tensor[0])
        pred_color_map = cv2.applyColorMap((255 * tensor[1] / (tensor[2].max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)
        gt_color_map = cv2.applyColorMap((255 * tensor[2] / (tensor[2].max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)
        binar_color_map = cv2.applyColorMap((255 * tensor[3] ).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)
        threshold_color_map = cv2.applyColorMap((255 * tensor[4] / (tensor[2].max()  + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)

        point_color = (0, 255, 0)  # BGR
        thickness = 1
        lineType = 4
        pil_input = np.array(pil_input)

        for i, box in enumerate(pred_boxes, 0):
            wh_LeftTop = (box[0], box[1])
            wh_RightBottom = (box[2], box[3])
            # print(wh_LeftTop, wh_RightBottom)
            cv2.rectangle(binar_color_map, wh_LeftTop, wh_RightBottom, point_color, thickness, lineType)
            cv2.rectangle(pil_input, wh_LeftTop, wh_RightBottom, point_color, thickness, lineType)
        point_color = (255, 0, 0)  # BGR

        for i, box in enumerate(gt_boxes, 0):
            wh_LeftTop = (box[0], box[1])
            wh_RightBottom = (box[2], box[3])
            cv2.rectangle(pil_input, wh_LeftTop, wh_RightBottom, point_color, thickness, lineType)

        pil_input = Image.fromarray(pil_input)
        pil_label = Image.fromarray(cv2.cvtColor(gt_color_map, cv2.COLOR_BGR2RGB))
        pil_output = Image.fromarray(cv2.cvtColor(pred_color_map, cv2.COLOR_BGR2RGB))
        pil_binar = Image.fromarray(cv2.cvtColor(binar_color_map, cv2.COLOR_BGR2RGB))

        pil_threshold = Image.fromarray(cv2.cvtColor(threshold_color_map, cv2.COLOR_BGR2RGB))


        x.extend([pil_to_tensor(pil_input.convert('RGB')), pil_to_tensor(pil_label.convert('RGB')),
                  pil_to_tensor(pil_output.convert('RGB')), pil_to_tensor(pil_binar.convert('RGB')),
                  pil_to_tensor(pil_threshold.convert('RGB'))])

    x = torch.stack(x, 0)
    x = vutils.make_grid(x, nrow=3, padding=5)
    x = (x.numpy() * 255).astype(np.uint8)

    writer.add_image(exp_name + '_epoch_' + str(epoch + 1), x)

def vis_results(exp_name, epoch, writer, restore, img, pred_map, gt_map,binar_map,boxes):#, flow):

    pil_to_tensor = standard_transforms.ToTensor()

    x = []
    y = []
    
    for idx, tensor in enumerate(zip(img.cpu().data, pred_map, gt_map,binar_map)):
        if idx>1:# show only one group
            break

        pil_input = restore(tensor[0])
        pred_color_map = cv2.applyColorMap((255*tensor[1]/(tensor[2].max()+1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)
        gt_color_map = cv2.applyColorMap((255*tensor[2]/(tensor[2].max()+1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)
        binar_color_map = cv2.applyColorMap((255*tensor[3]/(tensor[2].max()+1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)

        point_color = (0, 255, 0)  # BGR
        thickness = 1
        lineType = 4
        pil_input = np.array(pil_input)
        # print(pil_input, binar_color_map)
        for i, box in enumerate(boxes, 0):
            wh_LeftTop = (box[0], box[1])
            wh_RightBottom = (box[0] + box[2], box[1] + box[3])
            # print(wh_LeftTop, wh_RightBottom)
            cv2.rectangle(binar_color_map, wh_LeftTop, wh_RightBottom, point_color, thickness, lineType)
            cv2.rectangle(pil_input,       wh_LeftTop, wh_RightBottom, point_color, thickness, lineType)

        pil_input =Image.fromarray(pil_input)
        pil_label = Image.fromarray(cv2.cvtColor(gt_color_map,cv2.COLOR_BGR2RGB))
        pil_output = Image.fromarray(cv2.cvtColor(pred_color_map,cv2.COLOR_BGR2RGB))
        pil_binar = Image.fromarray(cv2.cvtColor(binar_color_map, cv2.COLOR_BGR2RGB))
        x.extend([pil_to_tensor(pil_input.convert('RGB')), pil_to_tensor(pil_label.convert('RGB')),
                  pil_to_tensor(pil_output.convert('RGB')),pil_to_tensor(pil_binar.convert('RGB'))])
        # pdb.set_trace()  sum(sum(flow[0].cpu().data.numpy().transpose((1,2,0))[:,:,0]))
        # flow = flow[0].cpu().data.numpy().transpose((1,2,0))
        # flow0 = cv2.applyColorMap((255*flow[:,:,0]/(flow[:,:,0].max()+1e-10)).astype(np.uint8).squeeze(),cv2.COLORMAP_JET)
        # flow1 = cv2.applyColorMap((255*flow[:,:,1]/(flow[:,:,1].max()+1e-10)).astype(np.uint8).squeeze(),cv2.COLORMAP_JET)
        # flow2 = cv2.applyColorMap((255*flow[:,:,2]/(flow[:,:,2].max()+1e-10)).astype(np.uint8).squeeze(),cv2.COLORMAP_JET)
        # flow0 = Image.fromarray(cv2.cvtColor(flow0,cv2.COLOR_BGR2RGB))
        # flow1 = Image.fromarray(cv2.cvtColor(flow1,cv2.COLOR_BGR2RGB))
        # flow2 = Image.fromarray(cv2.cvtColor(flow2,cv2.COLOR_BGR2RGB))
        # y.extend([pil_to_tensor(flow0.convert('RGB')), pil_to_tensor(flow1.convert('RGB')), pil_to_tensor(flow2.convert('RGB'))])


    x = torch.stack(x, 0)
    x = vutils.make_grid(x, nrow=4, padding=5)
    x = (x.numpy()*255).astype(np.uint8)

    # y = torch.stack(y,0)
    # y = vutils.make_grid(y,nrow=3,padding=5)
    # y = (y.numpy()*255).astype(np.uint8)

    # x = np.concatenate((x,y),axis=1)
    writer.add_image(exp_name + '_epoch_' + str(epoch+1), x)


def print_NWPU_summary(trainer, scores):
    f1m_l, ap_l, ar_l, mae, mse, nae, loss = scores
    train_record = trainer.train_record
    with open(trainer.log_txt, 'a') as f:
        f.write('='*15 + '+'*15 + '='*15 + '\n')
        f.write(str(trainer.epoch) + '\n\n')

        f.write('  [F1 %.4f Pre %.4f Rec %.4f ] [mae %.4f mse %.4f nae %.4f] [val loss %.4f]\n\n' % (f1m_l, ap_l, ar_l,mae, mse, nae,loss))

        f.write('='*15 + '+'*15 + '='*15 + '\n\n')

    print( '='*50 )
    print( trainer.exp_name )
    print( '    '+ '-'*20 )
    print( '  [F1 %.4f Pre %.4f Rec %.4f] [mae %.2f mse %.2f], [val loss %.4f]'\
            % (f1m_l, ap_l, ar_l, mae, mse, loss) )
    print( '    '+ '-'*20 )
    print( '[best] [model: %s] , [F1 %.4f Pre %.4f Rec %.4f] [mae %.2f], [mse %.2f], [nae %.4f]' % (train_record['best_model_name'], \
                                                        train_record['best_F1'], \
                                                        train_record['best_Pre'], \
                                                        train_record['best_Rec'],\
                                                        train_record['best_mae'],\
                                                        train_record['best_mse'],\
                                                        train_record['best_nae']) )
    print( '='*50 )  

def print_NWPU_summary_det(trainer, scores):
    train_record = trainer.train_record
    with open(trainer.log_txt, 'a') as f:
        f.write('='*15 + '+'*15 + '='*15 + '\n')
        f.write(str(trainer.epoch) + '\n\n')
        f.write('  [')
        for key, data in scores.items():
            f.write(' ' +key+  ' %.3f'% data)
        f.write('\n\n')
        f.write('='*15 + '+'*15 + '='*15 + '\n\n')

    print( '='*50 )
    print( trainer.exp_name )
    print( '    '+ '-'*20 )
    content = '  ['
    for key, data in scores.items():
        if isinstance(data,str):
            content +=(' ' + key + ' %s' % data)
        else:
            content += (' ' + key + ' %.3f' % data)
    content += ']'
    print( content)
    print( '    '+ '-'*20 )
    best_str = '[best]'
    for key, data in train_record.items():
        best_str += (' [' + key +' %s'% data + ']')
    print( best_str)
    print( '='*50 )

def update_model(trainer, scores):
    train_record = trainer.train_record
    log_file = trainer.log_txt
    epoch = trainer.epoch
    snapshot_name = 'ep_%d_iter_%d'% (epoch,trainer.i_tb)
    for key, data in scores.items():
        snapshot_name+= ('_'+ key+'_%.3f'%data)
    # snapshot_name = 'ep_%d_F1_%.3f_Pre_%.3f_Rec_%.3f_mae_%.1f_mse_%.1f' % (epoch + 1, F1, Pre, Rec, mae, mse)

    for key, data in  scores.items():
        print(key,data)
        if data<train_record[key] :
            train_record['best_model_name'] = snapshot_name
            if log_file is not None:
                logger_txt(log_file,epoch,scores)
            to_saved_weight = trainer.net.state_dict()

            torch.save(to_saved_weight, os.path.join(trainer.exp_path, trainer.exp_name, snapshot_name + '.pth'))

        if data < train_record[key]:
            train_record[key] = data
    latest_state = {'train_record':train_record, 'net':trainer.net.state_dict(), 'optimizer':trainer.optimizer.state_dict(),
                    'epoch': trainer.epoch, 'i_tb':trainer.i_tb, 'num_iters':trainer.num_iters,\
                    'exp_path':trainer.exp_path, 'exp_name':trainer.exp_name}
    torch.save(latest_state,os.path.join(trainer.exp_path, trainer.exp_name, 'latest_state.pth'))

    return train_record


def copy_cur_env(work_dir, dst_dir, exception):

    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    for filename in os.listdir(work_dir):

        file = os.path.join(work_dir,filename)
        dst_file = os.path.join(dst_dir,filename)

        if os.path.isdir(file) and filename not in exception:
            shutil.copytree(file, dst_file)
        elif os.path.isfile(file):
            shutil.copyfile(file,dst_file)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.cur_val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, cur_val):
        self.cur_val = cur_val
        self.sum += cur_val
        self.count += 1
        self.avg = self.sum / self.count


class AverageCategoryMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self,num_class):
        self.num_class = num_class
        self.reset()

    def reset(self):
        self.cur_val = np.zeros(self.num_class)
        self.sum = np.zeros(self.num_class)


    def update(self, cur_val):
        self.cur_val = cur_val
        self.sum += cur_val


# class AverageCategoryMeter(object):
#     """Computes and stores the average and current value"""
#
#     def __init__(self,num_class):
#         self.num_class = num_class
#         self.reset()
#
#     def reset(self):
#         self.cur_val = np.zeros(self.num_class)
#         self.avg = np.zeros(self.num_class)
#         self.sum = np.zeros(self.num_class)
#         self.count = np.zeros(self.num_class)
#
#     def update(self, cur_val, class_id):
#         self.cur_val[class_id] = cur_val
#         self.sum[class_id] += cur_val
#         self.count[class_id] += 1
#         self.avg[class_id] = self.sum[class_id] / self.count[class_id]


class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


def vis_results_img(img, pred, restore):
    # pdb.set_trace()
    img = img.cpu()
    pred = pred.cpu().numpy()
    pil_input = restore(img)
    pred_color_map = cv2.applyColorMap(
        (255*pred / (pred.max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)
    pil_output = Image.fromarray(
        cv2.cvtColor(pred_color_map, cv2.COLOR_BGR2RGB))
    x = []
    pil_to_tensor = standard_transforms.ToTensor()
    x.extend([pil_to_tensor(pil_input.convert('RGB')),
              pil_to_tensor(pil_output.convert('RGB'))])
    x = torch.stack(x, 0)
    x = vutils.make_grid(x, nrow=3, padding=5)
    x = (x.numpy() * 255).astype(np.uint8)

    # pdb.set_trace()
    return x

def make_matching_plot_fast(image0, image1, kpts0, kpts1, mkpts0,
                            mkpts1, color, text, path=None,
                            show_keypoints=False, margin=10,
                            opencv_display=False, opencv_title='',
                            small_text = [], restore_transform=None,
                            id0=None,id1=None
                            ):

    image0 = np.array(restore_transform(image0))
    image1 = np.array(restore_transform(image1))
    image0 = cv2.cvtColor(image0, cv2.COLOR_RGB2BGR)
    image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)
    H0, W0, C = image0.shape
    H1, W1, C = image1.shape
    H, W = max(H0, H1), W0 + W1 + margin

    out = 255*np.ones((H, W, C), np.uint8)
    out[:H0, :W0,:] = image0
    out[:H1, W0+margin:,:] = image1
    # out = np.stack([out]*3, -1)
    # import pdb
    # pdb.set_trace()
    out_by_point = out.copy()
    point_r_value = 15
    thickness = 3
    white = (255, 255, 255)
    green = (0, 255, 0)
    red = (0, 0, 255)
    blue = (255, 0, 0)
    if show_keypoints:
        kpts0, kpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)
        for x, y in kpts0:
            cv2.circle(out, (x, y), point_r_value, red, thickness, lineType=cv2.LINE_AA)
            cv2.circle(out, (x, y), 3, white, -1, lineType=cv2.LINE_AA)

            cv2.circle(out_by_point, (x, y), point_r_value, red, thickness, lineType=cv2.LINE_AA)

        for x, y in kpts1:
            cv2.circle(out, (x + margin + W0, y), point_r_value, red, thickness,
                       lineType=cv2.LINE_AA)
            cv2.circle(out, (x + margin + W0, y), 3, white, -1, lineType=cv2.LINE_AA)

            cv2.circle(out_by_point, (x + margin + W0, y), point_r_value, blue, thickness,
                       lineType=cv2.LINE_AA)

        if id0 is not  None:
            for i, (id, centroid) in enumerate(zip(id0, kpts0)):
                cv2.putText(out, str(id), (centroid[0],centroid[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        if id1 is not None:
            for i, (id, centroid) in enumerate(zip(id1, kpts1)):
                cv2.putText(out, str(id), (centroid[0]+margin+W0, centroid[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    mkpts0, mkpts1 = np.round(mkpts0).astype(int), np.round(mkpts1).astype(int)
    color = (np.array(color[:, :3])*255).astype(int)[:, ::-1]

    for (x0, y0), (x1, y1), c in zip(mkpts0, mkpts1, color):
        c = c.tolist()
        cv2.line(out, (x0, y0), (x1 + margin + W0, y1),
                 color=c, thickness=1, lineType=cv2.LINE_AA)
        # display line end-points as circles
        cv2.circle(out, (x0, y0), point_r_value, green, thickness, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + W0, y1), point_r_value, green, thickness,
                   lineType=cv2.LINE_AA)

        cv2.circle(out_by_point, (x0, y0), point_r_value, green, thickness, lineType=cv2.LINE_AA)
        cv2.circle(out_by_point, (x1 + margin + W0, y1), point_r_value, green, thickness,
                   lineType=cv2.LINE_AA)

    # Ht = int(H*30 / 480)  # text height
    # txt_color_fg = (255, 255, 255)
    # txt_color_bg = (0, 0, 0)
    # for i, t in enumerate(text):
    #     cv2.putText(out, t, (10, Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
    #                 H*1.0/480, txt_color_bg, 2, cv2.LINE_AA)
    #     cv2.putText(out, t, (10, Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
    #                 H*1.0/480, txt_color_fg, 1, cv2.LINE_AA)
    #     cv2.putText(out_by_point, t, (10, Ht * (i + 1)), cv2.FONT_HERSHEY_DUPLEX,
    #             H * 1.0 / 480, txt_color_fg, 1, cv2.LINE_AA)
    if path is not None:
        cv2.imwrite(str(path), out)
        cv2.imwrite(str('point_'+path), out_by_point)
    if opencv_display:
        cv2.imshow(opencv_title, out)
        cv2.waitKey(1)

    return out,out_by_point