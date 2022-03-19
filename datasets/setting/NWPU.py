from easydict import EasyDict as edict


# init
__C_NWPU = edict()

cfg_data = __C_NWPU

__C_NWPU.TRAIN_SIZE = (512,1024)
__C_NWPU.DATA_PATH = '/data/GJY/ht/ProcessedData/NWPU_2048/'
__C_NWPU.TRAIN_LST = 'pre_train.txt'
__C_NWPU.VAL_LST =  'val.txt'
__C_NWPU.VAL4EVAL = 'val_gt_loc.txt'

__C_NWPU.MEAN_STD = ([0.446139603853, 0.409515678883, 0.395083993673], [0.288205742836, 0.278144598007, 0.283502370119])

__C_NWPU.DEN_FACTOR = 200
__C_NWPU.LOG_PARA = 1.
__C_NWPU.RESUME_MODEL = ''#model path
__C_NWPU.TRAIN_BATCH_SIZE = 4 #imgs

__C_NWPU.VAL_BATCH_SIZE = 1 # must be 1

__C_NWPU.TRAIN_FRAME_INTERVALS=(5,25)  # 2s-8s


