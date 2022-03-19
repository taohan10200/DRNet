import cv2 as cv
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from  misc.utils import  AverageMeter

class Instance_loss(nn.Module):
    def __init__(self, factor = 100):
        super(Instance_loss, self).__init__()
        self.mean_weight = AverageMeter()
        self.factor = factor
        for i in range(10):
            self.mean_weight.update(1./self.factor)

    def forward(self,binar_map, gt_mask):
        b, c, h, w = gt_mask.size()
        Instance_weight = np.zeros((b, c, h, w))
        back_weight = torch.zeros(b, c, h, w).cuda()
        back_weight[gt_mask == 0] =1

        dilation_mask = F.max_pool2d(gt_mask, kernel_size=(9,9),stride = 2, padding = 4)
        dilation_mask = F.interpolate(dilation_mask, scale_factor=2)
        outer_mask = dilation_mask - gt_mask


        num_instance= 0
        for it in range(b):
            boxes, cnt = get_boxInfo_from_Binar_map(gt_mask[it,0,:,:].cpu().numpy())
            num_instance +=cnt
            for box in boxes:
                s_w, s_h, box_w, box_h,  = box[0], box[1],box[2], box[3]
                kernel = fspecial(box_w, box_h)
                # self.mean_weight.update(1./(box_w*box_h))
                Instance_weight[it, 0, s_h:s_h+box_h, s_w:s_w+box_w] = kernel

        Instance_weight = torch.from_numpy(Instance_weight).float().cuda()

        outer_weight = F.max_pool2d(Instance_weight,kernel_size=(9,9),stride = 2, padding = 4)
        outer_weight = F.interpolate(outer_weight, scale_factor=2)
        outer_weight = outer_weight*outer_mask


        threshold_gt=outer_mask*0.7+gt_mask*0.2
        # cv.imwrite('0008_outer_mask.png', outer_mask.cpu().squeeze().numpy() * 255)
        # cv.imwrite('0008_outer_weight.png', (outer_weight/outer_weight.max()).cpu().squeeze().numpy() * 255)

        # slice_h, slice_w = 15,15
        # pool = back_weight+Instance_weight
        # pool = F.avg_pool2d(pool, kernel_size=(slice_h,slice_w),stride =4, padding = 7)
        # # pool[pool<1] = 1
        # pool =F.interpolate(pool,scale_factor=4)
        # back_weight = pool*back_weight
            # weight = weight/np.max(weight)*255


            #
            # for i in range(0, h, slice_h):
            #     h_s, h_e = max(min(h - slice_h, i), 0), min(h, i + slice_h)
            #     for j in range(0, w, slice_w):
            #         w_s, w_e = max(min(w - slice_w, j), 0), min(w, j + slice_w)
            #         tmp_weight=max(1, Instance_weight[it, 0, h_s:h_e, w_s:w_e].mean())
            #         back_weight[it, 0, h_s:h_e, w_s:w_e] *= tmp_weight

        all_weight = (outer_weight+Instance_weight)
            # print(all_weight.sum()/self.factor, Instance_weight.sum(), gt_mask.sum()/255, all_weight.max())

            # all_weight = all_weight*255
            # print(all_weight.min(),self.mean_weight.avg)
        # cv.imwrite('0008_Instance_weight.png', (Instance_weight/Instance_weight.max()).cpu().squeeze().numpy()*255)
        # cv.imwrite('0008_all_weight.png', (all_weight/all_weight.max()).cpu().squeeze().numpy()*255)

        loss1 = torch.abs(Instance_weight*(binar_map - gt_mask)).sum()/(1e-10+num_instance)
        # loss2 = torch.abs(back_weight*(binar_map - gt_mask)).mean()
        # print(loss1, loss2)

        return loss1,  dilation_mask, threshold_gt, all_weight+1
def fspecial( wide,height):
    sigma = max(wide,height)//2
    wide, height = (wide - 1.0) / 2.0 , (height - 1.0) / 2.0
    x, y = np.ogrid[-height : height + 1, -wide : wide+ 1]
    h = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sum_h = h.sum()
    if sum_h != 0:
        h /= sum_h
    return h

def get_boxInfo_from_Binar_map( Binar_numpy, min_area=2):
    Binar_numpy = Binar_numpy.squeeze().astype(np.uint8)
    assert Binar_numpy.ndim == 2
    cnt, labels, stats, centroids = cv.connectedComponentsWithStats(Binar_numpy, connectivity=4)  # centriod (w,h)
    # print(stats, cnt)
    boxes = stats[1:, :]
    index = (boxes[:, 4] >= min_area)
    boxes = boxes[index]

    return boxes, cnt-1

if __name__=="__main__":
    import time
    # print(fspecial(wide=1, height=15))
    s = time.time()
    gt_mask = cv.imread('0008.png',cv.IMREAD_GRAYSCALE)/255.

    h, w = gt_mask.shape
    a = torch.rand(1,1, h, w ).cuda()
    gt_mask = torch.from_numpy(gt_mask).float().cuda()
    gt_mask = gt_mask[None,None, :,:]
    print(gt_mask.size())
    print(a.size())
    loss= Instance_loss(factor=50)
    L, w=loss(a, gt_mask)
    print(L)
    print('time:{}'.format(time.time()-s))