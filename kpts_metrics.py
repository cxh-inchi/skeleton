import pickle

from tqdm import tqdm
from functions import *
from model.models import *
from torch.utils.data import DataLoader,SequentialSampler
import torch

if __name__ == '__main__':
    
    DEVICE = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    model_path = '/datahdd/smartradar/cxh/mmwhumanpose_training/data/posenet_0214_1336.pth'
    # model_path = 'data/posenet_0301_simp.pth'
    # model_path = 'data/posenet_0109_0918.pth' ## pointnext
    # test_root_paths = [
    #     r"D:\1-humanpose\2-traindata\testset_0720_match",
    #     r"D:\1-humanpose\2-traindata\testset_0728_match",
    #     r"D:\1-humanpose\2-traindata\testset_0825_k",
    #     r"D:\1-humanpose\2-traindata\testset_0825_g",
    # ]
    # root_path = get_full_paths(test_root_paths)
    # root_path = r'D:\1-humanpose\2-traindata\data_1frame\dataset_person_0130_stand\test_singlepeople'

    # test_data = PCDataSet(root_path, Nframes=15, dimpcloud=5, npoint=256)
    with open('/datahdd/smartradar/cxh/mmwhumanpose_training/test_dataset_kpts_1113_nframe8.pkl', 'rb') as file:
        test_data = pickle.load(file)
    test_data_size = len(test_data)
    print("Length of test data: {}".format(test_data_size))

    test_dataloader = DataLoader(test_data, batch_size=8,drop_last=True,sampler=SequentialSampler(test_data))
    model_weights = torch.load(model_path)
    # posenet = KPTSNet_S4().to(DEVICE)
    # posenet = KPTSNet_S3().to(DEVICE)
    # posenet = KPTSNext().to(DEVICE)
    posenet = KPTSNet_singlegru().to(DEVICE)
    # posenet = KPTSNet().to(DEVICE)
    posenet.load_state_dict(model_weights)
    posenet.eval()
    pred_list = []
    label_list = []
    i=0
    with torch.no_grad():
        for pclouds, labels in tqdm(test_dataloader):
            pclouds, labels = pclouds.to(DEVICE), labels.to(DEVICE)
            predictions,_= posenet(pclouds)
            pred_list.append(predictions.cpu().detach().numpy())
            label_list.append(labels.cpu().detach().numpy())
            # i+=1
            # if i==50:
            #     break
        mpjpe = MPJPE(pred_list, label_list)
        PCKm = PCKmean(pred_list, label_list, delta=0.15)
        print('MPJPE = {}, PCKm = {}'.format(mpjpe, PCKm))
        
        mpjpe, pa_mpjpe = compute_error_3d(label_list, pred_list)
        print('MPJPE = {}, PA-MPJPE = {}'.format(mpjpe, pa_mpjpe))

        Accel = compute_error_accel(label_list, pred_list)
        print('Accel = {}'.format(Accel))
 