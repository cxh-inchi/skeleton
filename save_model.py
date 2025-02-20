import datetime
import itertools
# import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
# from functions.functions_idx import *
from model.models import *
import dill       
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts,StepLR
import os


os.environ['OMP_NUM_THREADS'] = '2'

"""
Checkpoint resume
"""
Nframes = 8
ch=5
DEVICE = torch.device('cpu')
posenet = KPTSNet_singlegru_().to(DEVICE)
path_checkpoint = "/datahdd/smartradar/cxh/mmwhumanpose_training/checkpoint/3G_che_sgru8_010816/checkpoint_29_epoch.pkl"
checkpoint = torch.load(path_checkpoint, map_location=DEVICE)
# posenet.load_state_dict(checkpoint['model_state_dict'])
torch.save(checkpoint['model_state_dict'], '/datahdd/smartradar/cxh/mmwhumanpose_training/data/3G_ck_sgru8_010816.pth')
