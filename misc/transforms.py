import numbers
import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from config import cfg
import torch
import numpy
import pdb
import cv2
from torchvision.transforms import functional as TrF
from misc import inflation


class ProcessSub(object):
    def __init__(self, T=0.1, K=51):
        self.T = T
        self.inf = inflation.inflation(K=K)

    def getHS(self, flow):
        # h direction  s or v magnitude
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        h = ang * 180 / np.pi / 2  # angle
        s = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # magnitude
        return h, s

    def __call__(self, flow):
        h, s = self.getHS(flow[:, :, 0:2])
        flow[:, :, 0] = h.astype(np.float32) / 255
        flow[:, :, 1] = s.astype(np.float32) / 255
        # Threshold
        temp = np.ones(flow[:, :, 2].shape)
        temp[abs(flow[:, :, 2]) < self.T] = 0
        flow[:, :, 2] = flow[:, :, 2] * temp
        # inflation
        return flow


class RandomEmptyFlow(object):
    def __call__(self, flow):
        if random.random() < 0.04:
            flow = numpy.zeros((flow.shape[0], flow.shape[1], flow.shape[2])).astype(numpy.float32)
        return flow


# ===============================img tranforms============================

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask, bbx=None):
        if bbx is None:
            for t in self.transforms:
                img, mask = t(img, mask)
            return img, mask
        for t in self.transforms:
            img, mask, bbx = t(img, mask, bbx)
        return img, mask, bbx


class RandomHorizontallyFlip(object):
    def __init__(self, task=None):
        self.task = task

    def __call__(self, img, gt, flip_flag=0, bbx=None):

        if flip_flag :
            w, h = img.size
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            gt['points'][:,0] =w -  gt['points'][:,0]

            # if bbx is not None:
            #     xmin = w - bbx[:, 3]
            #     xmax = w - bbx[:, 1]
            #     bbx[:, 1] = xmin
            #     bbx[:, 3] = xmax
            #     return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT), bbx
            # if bbx is None:
            #     return img, gt  # flow
        return img, gt


class RandomVerticallyFlip(object):
    def __call__(self, img, mask, flow=None, bbx=None):
        if random.random() < 0.5:
            if bbx is None:

                return img.transpose(Image.FLIP_TOP_BOTTOM), mask.transpose(Image.FLIP_TOP_BOTTOM)
            w, h = img.size
            ymin = w - bbx[:, 2]
            ymax = w - bbx[:, 0]
            bbx[:, 0] = ymin
            bbx[:, 2] = ymax
            return img.transpose(Image.FLIP_TOP_BOTTOM), mask.transpose(Image.FLIP_TOP_BOTTOM), bbx
        if bbx is None:
            return img, mask  # , flow
        return img, mask, bbx


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size),)
        else:
            self.size = size

    def __call__(self, img, mask):
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))


class ScalebyRate(object):
    def __init__(self, rateRange):
        self.rateRange = rateRange

    def __call__(self, img, den):
        img_w, img_h = img.size
        den_w, den_h = den.size

        # init_random_rate = self.rateRange[0] + random.random()*(self.rateRange[1]-self.rateRange[0])

        init_random_rate = random.uniform(self.rateRange[0], self.rateRange[1])

        dst_img_w = int(img_w * init_random_rate) // 32 * 32
        dst_img_h = int(img_h * init_random_rate) // 32 * 32

        real_rate_w = dst_img_w / img_w
        real_rate_h = dst_img_h / img_h

        dst_den_w = int(den_w * init_random_rate) // 32 * 32
        dst_den_h = int(den_h * init_random_rate) // 32 * 32

        den = np.array(den.resize((dst_den_w, dst_den_h), Image.BILINEAR)) / real_rate_w / real_rate_h
        den = Image.fromarray(den)

        return img.resize((dst_img_w, dst_img_h), Image.BILINEAR), den


class ScaleByRateWithMin(object):
    def __init__(self, min_w, min_h, task=None):
        self.min_w = min_w
        self.min_h = min_h
        self.task = task

    def __call__(self, img, gt):  # dot, flow):
        w, h = img.size

        new_w = self.min_w
        new_h = self.min_h
        img = img.resize((new_w, new_h), Image.ANTIALIAS)

        rate = new_w / w
        gt['points'] =  gt['points']  * rate #cv2.resize(gt[1], (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        gt['sigma']  =  gt['sigma']  * rate
        return img, gt

def check_image(img,target, crop_size):
    w, h = img.size
    c_h, c_w = crop_size
    if w < c_w or h < c_h:
        delta_w = max(c_w - w, 0)
        delta_h = max(c_h - h, 0)
        padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
        img = ImageOps.expand(img, padding)
        target['points'] =target['points']+torch.tensor([delta_w // 2, delta_h // 2], dtype = torch.float32)

    return  img, target
class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img, gt, crop_left, crop_size ):

        th, tw = crop_size[0], crop_size[1]
        x1,y1 = crop_left
        img = img.crop((x1, y1, x1 + tw, y1 + th))
        index = (gt['points'][:,0]>x1+1) & (gt['points'][:,0]<x1 + tw-1) & (gt['points'][:,1]>y1+1) & (gt['points'][:,1]<y1 + th-1)

        gt['points'] = gt['points'][index].view(-1,2).contiguous()
        gt['points'] -= torch.tensor([x1, y1], dtype = torch.float32)

        gt['person_id'] =  gt['person_id'][index] #gt[1][y1:y1 + th, x1:x1 + tw]
        gt['sigma'] = gt['sigma'][index]

        return img, gt

class FreeScale(object):
    def __init__(self, size):
        self.size = size  # (h, w)

    def __call__(self, img, mask):
        return img.resize((self.size[1], self.size[0]), Image.BILINEAR), mask.resize((self.size[1], self.size[0]), Image.NEAREST)


class ScaleDown(object):
    def __init__(self, size):
        self.size = size  # (h, w)

    def __call__(self, mask):
        return mask.resize((self.size[1] / cfg.TRAIN.DOWNRATE, self.size[0] / cfg.TRAIN.DOWNRATE), Image.NEAREST)


class Scale(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        if img.size != mask.size:
            print(img.size)
            print(mask.size)
        assert img.size == mask.size
        w, h = img.size
        if (w <= h and w == self.size) or (h <= w and h == self.size):
            return img, mask
        if w < h:
            ow = self.size
            oh = int(self.size * h / w)
            return img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST)
        else:
            oh = self.size
            ow = int(self.size * w / h)
            return img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST)


# ===============================image tranforms============================

class RGB2Gray(object):
    def __init__(self, ratio):
        self.ratio = ratio  # [0-1]

    def __call__(self, img):
        if random.random() < 0.1:
            return TrF.to_grayscale(img, num_output_channels=3)
        else:
            return img


class GammaCorrection(object):
    def __init__(self, gamma_range=[0.4, 2]):
        self.gamma_range = gamma_range

    def __call__(self, img):
        if random.random() < 0.5:
            gamma = random.uniform(self.gamma_range[0], self.gamma_range[1])
            return TrF.adjust_gamma(img, gamma)
        else:
            return img


# ===============================label tranforms============================

class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()


class LabelNormalize(object):
    def __init__(self, para):
        self.para = para

    def __call__(self, tensor):
        # tensor = 1./(tensor+self.para).log()
        tensor = torch.from_numpy(np.array(tensor))
        tensor = tensor * self.para
        return tensor


class GTScaleDown(object):
    def __init__(self, factor=8):
        self.factor = factor

    def __call__(self, img):
        w, h = img.size
        if self.factor == 1:
            return img
        tmp = np.array(img.resize((w // self.factor, h // self.factor), Image.BICUBIC)) * self.factor * self.factor
        img = Image.fromarray(tmp)
        return img


class tensormul(object):
    def __init__(self, mu=255.0):
        self.mu = 255.0

    def __call__(self, _tensor):
        _tensor.mul_(self.mu)
        return _tensor


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    main_transform = Compose([
        ScaleByRateWithMin([0.8, 1.2], 1024, 512)])
    img = Image.open('T:\ProcessedData/NWPU/images/4868.jpg')
    mask = Image.open('T:\ProcessedData/NWPU/mask_50_60/4868.png')
    new_img, new_mask = main_transform(img, mask)
    plt.imshow(img)
    plt.show()
    plt.imshow(mask)
    plt.show()
    plt.imshow(new_img)
    plt.show()
    plt.imshow(new_mask)
    plt.show()