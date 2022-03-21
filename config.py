import os
from easydict import EasyDict as edict
import time
import torch

# init
__C = edict()
cfg = __C

#------------------------------TRAIN------------------------
__C.SEED = 3035  # random seed,  for reproduction
__C.DATASET = 'HT21'       # dataset selection:  HT21, SENSE
__C.NET = 'VGG16_FPN'     # 'VGG16_FPN'

__C.PRE_VGG_WEIGHTS = './exp/pretrain_counter/ep_19_mae_3.413474_mse_4.425373_nae_0.019419.pth'
__C.PRE_VGG_MATCH_WEIGHTS = './exp/HT21/11-07_01-55_HT21_VGG16_FPN_5e-05_(full model)/ep_5_iter_12500_mae_9.797_mse_10.438_seq_MAE_38.768_WRAE_44.561_MIAE_5.209_MOAE_5.535.pth'#'./exp/NWPU/pretrained1/ep_94_iter_125146_mae_0.266826_match_0.145812.pth'# ep_7_iter_8939_mae_0.474924_match_0.233859.pth'#'./exp/HT21/10-11_04-31_HT21_VGG16_FPN_2e-05/ep_30_mae_1.304089_mse_1.726433_nae_0.025763_seq_mae_-11.999855_seq_mse_11.999855.pth'
__C.PRE_VGG_MATCH_WEIGHTS = './exp/SENSE/11-06_02-04_SENSE_VGG16_FPN_5e-05/ep_8_iter_132500_mae_2.796_mse_4.887_seq_MAE_5.873_WRAE_10.335_MIAE_1.789_MOAE_1.809.pth'
__C.RESUME = False # continue training
__C.RESUME_PATH = './exp/SENSE/11-23_04-55_SENSE_Res50_FPN_5e-05/latest_state.pth'
__C.GPU_ID = '0'  # sigle gpu: '0'; multi gpus: '0,1'

__C.sinkhorn_iterations = 100
__C.FEATURE_DIM = 256
__C.ROI_RADIUS = 4.
if __C.DATASET == 'SENSE':
    __C.VAL_INTERVALS =15
else:
    __C.VAL_INTERVALS = 50
# learning rate settings
__C.LR_Base = 5e-5  # learning rate
__C.LR_Thre = 1e-2

__C.LR_DECAY = 0.95
__C.WEIGHT_DECAY = 1e-5  # decay rate
# when training epoch is more than it, the learning rate will be begin to decay

__C.MAX_EPOCH = 20

# print
__C.PRINT_FREQ = 20

now = time.strftime("%m-%d_%H-%M", time.localtime())

__C.EXP_NAME = now \
    + '_' + __C.DATASET \
    + '_' + __C.NET \
    + '_' + str(__C.LR_Base)

__C.VAL_VIS_PATH = './exp/'+__C.DATASET+'_val'
__C.EXP_PATH = os.path.join('./exp', __C.DATASET)  # the path of logs, checkpoints, and current codes
if not os.path.exists(__C.EXP_PATH ):
    os.makedirs(__C.EXP_PATH )
#------------------------------VAL------------------------

if __C.DATASET == 'HT21':
    __C.VAL_FREQ = 1  # Before __C.VAL_DENSE_START epoches, the freq is set as __C.VAL_FREQ
    __C.VAL_DENSE_START = 2
else:
    __C.VAL_FREQ = 1
    __C.VAL_DENSE_START = 0
#------------------------------VIS------------------------

#================================================================================
