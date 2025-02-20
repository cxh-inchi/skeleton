import datetime
import pickle
import torch
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from tqdm import tqdm
import time
import os
import numpy as np
from functions import *
from model.models import *
from torch.utils.data import DataLoader


for idx in [1]:
    # torch.set_num_threads(2)
    Ind_checkpoint = False
    Ind_pretrain = True
    Teacher_md_path = '/datahdd/xhjia/code/mmwhumanpose_training/data/posenet_0321_gru.pth'
    pre_md_pth = '/datahdd/xhjia/code/mmwhumanpose_training/data/posenet_0321_gru.pth'
    chkpnt_path = '/datahdd/xhjia/code/mmwhumanpose_training/checkpoint/ck_2713/checkpoint_1_epoch.pkl'
    lr = 0.0002
    bs = 80
    MAX_EPOCH = 30
    checkpoint_interval = 2
    start_epoch = 0
    ch = 5
    Nframes = 15                                                                   
    npoint = 256
    dt = datetime.datetime.now().strftime('%d%H')
    file_path_ck = os.path.join("checkpoint", "ck_" + dt)
    if not os.path.exists(file_path_ck):
        os.makedirs(file_path_ck)
    DEVICE = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
    savedatet = datetime.datetime.now().strftime('%d_%H%M')
    SW = SummaryWriter(
        os.path.join("logs_train",
                     "train_" + savedatet + 'bs_' + str(bs) + 'lr_' + str(lr)))

    # train_root_paths = [
    #     "/datahdd/xhjia/data/traindata/trainset_0720_match",
    #     "/datahdd/xhjia/data/traindata/trainset_0728_match",
    #     "/datahdd/xhjia/data/traindata/trainset_0825_g",
    #     "/datahdd/xhjia/data/traindata/trainset_0825_k",
    #     "/datahdd/xhjia/data/traindata/trainset_1212",
    #     "/datahdd/xhjia/data/traindata/trainset_1116_k",
    # ]
    # test_root_paths = [
    #     "/datahdd/xhjia/data/testdata/testset_0720_match",
    #     "/datahdd/xhjia/data/testdata/testset_0728_match",
    #     "/datahdd/xhjia/data/testdata/testset_0825_k",
    #     "/datahdd/xhjia/data/testdata/testset_0825_g",
    # ]

    
    # train_full_paths = [os.path.join(rp_item, fn_item) for rp_item in train_root_paths for fn_item in
    #                     os.listdir(rp_item)]
    # test_full_paths = [os.path.join(rp_item, fn_item) for rp_item in test_root_paths for fn_item in os.listdir(rp_item)]
    
    # train_data = PCDataSet(train_full_paths, ind_jitter=True, ind_rnd_gap=True, ind_cut=True, ind_rot=True,
    #                        ind_tl=False, ind_shuffle=True,
    #                        ind_scale=True, ind_reduce=True, Nframes=Nframes, dimpcloud=ch, npoint=npoint)
    # test_data = PCDataSet(test_full_paths, ind_jitter=False, ind_rnd_gap=True, Nframes=Nframes, dimpcloud=ch,
    #                       npoint=npoint)
    # with open('train_dataset_kpts_0305.pkl', 'wb') as file:
    #     pickle.dump(train_data, file)
    # with open('test_dataset_kpts_0305.pkl', 'wb') as file:
    #     pickle.dump(test_data, file)
    with open('/datahdd/xhjia/code/mmwhumanpose_training/train_dataset_kpts_0410.pkl', 'rb') as file:
        train_data = pickle.load(file)
    with open('/datahdd/xhjia/code/mmwhumanpose_training/test_dataset_kpts_0410.pkl', 'rb') as file:
        test_data = pickle.load(file)
    train_data_size = len(train_data)
    test_data_size = len(test_data)
    print("Train set length: {}, test set length: {}".format(train_data_size, test_data_size))

    train_dataloader = DataLoader(train_data, batch_size=bs, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=bs)
    posenet = KPTSNet_S3(nkpoint=17, dimkpoint=3, ch=ch).to(DEVICE)
    teachernet = KPTSNet_T(nkpoint=17, dimkpoint=3, ch=ch).to(DEVICE)
    model_path = os.path.join(Teacher_md_path)
    preweights = torch.load(model_path)
    teachernet.load_state_dict(preweights, strict=False)
    teachernet.eval()
    if Ind_pretrain:
        model_path = os.path.join(pre_md_pth)
        preweights = torch.load(model_path)
        posenet.load_state_dict(preweights, strict=False)

    loss_fc = KSLossFunc(nn.MSELoss(reduction='sum').to(DEVICE), 
                         nn.MSELoss(reduction='sum').to(DEVICE),
                         nn.MSELoss(reduction='sum').to(DEVICE), bs)
    max_iters = len(train_dataloader) * MAX_EPOCH

    # optimizer = torch.optim.AdamW(params=posenet.parameters(),
    #                               lr=lr,
    #                               betas=(0.95, 0.99),
    #                               weight_decay=0.01)
    # scheduler_mul = torch.optim.lr_scheduler.OneCycleLR(optimizer,
    #                                     max_lr=lr*10,
    #                                     total_steps=max_iters,
    #                                     pct_start=0.4,
    #                                     anneal_strategy='cos',
    #                                     cycle_momentum=True,
    #                                     base_momentum=0.95*0.895,
    #                                     max_momentum=0.95,
    #                                     div_factor=10)
    optimizer = torch.optim.Adam(posenet.parameters(), lr=lr, weight_decay=1e-4)
    # scheduler_mul = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.6)
    scheduler_mul = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99)
    tot_train_loss = []
    tot_train_loss1 = []
    tot_train_loss2 = []
    tot_train_loss3 = []
    tot_test_loss = []
    tot_test_loss1 = []
    tot_test_loss2 = []
    tot_test_loss3 = []
    """
    Checkpoint resume
    """
    if Ind_checkpoint:
        path_checkpoint = os.path.join(chkpnt_path)
        checkpoint = torch.load(path_checkpoint, map_location= 'cpu' )
        posenet.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict((checkpoint['optimizer_state_dict']))
        start_epoch = checkpoint['epoch'] + 1
        scheduler_mul.last_epoch = start_epoch - 1
        tot_train_loss = checkpoint['tot_train_loss']
        tot_train_loss1 = checkpoint['tot_train_loss1']
        tot_train_loss2 = checkpoint['tot_train_loss2']
        tot_train_loss3 = checkpoint['tot_train_loss3']
        tot_test_loss = checkpoint['tot_test_loss']
        tot_test_loss1 = checkpoint['tot_test_loss1']
        tot_test_loss2 = checkpoint['tot_test_loss2']
        tot_test_loss3 = checkpoint['tot_test_loss3']
    since_time = time.time()

    epoch_tqdm = tqdm(range(start_epoch, MAX_EPOCH), dynamic_ncols=True, leave=False)
    for epoch in range(start_epoch, MAX_EPOCH):
        """
        Training phase
        """
        posenet.train()
        train_loss_epoch = 0
        train_loss1_epoch = 0
        train_loss2_epoch = 0
        train_loss3_epoch = 0
        train_tqdm = tqdm(train_dataloader, dynamic_ncols=True, leave=False)
        for pcloud, labels in train_dataloader:
            pcloud, labels = pcloud.to(DEVICE), labels.to(DEVICE)
            T_preds, T_feas = teachernet(pcloud)
            S_preds, S_feas = posenet(pcloud)
            loss, loss1, loss2, loss3= loss_fc(S_preds, T_preds.detach(), labels, S_feas, T_feas.detach()) #tht
            train_loss_epoch += loss.item()
            train_loss1_epoch += loss1.item()
            train_loss2_epoch += loss2.item()
            train_loss3_epoch += loss3.item()
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            train_tqdm.set_description("Train loss: {}".format(round(loss.item(), 5)))
            train_tqdm.update()
        scheduler_mul.step()
        print(optimizer.state_dict()['param_groups'][0]['lr'])
        tot_train_loss.append(train_loss_epoch/np.ceil(train_data_size/bs))
        tot_train_loss1.append(train_loss1_epoch/np.ceil(train_data_size/bs))
        tot_train_loss2.append(train_loss2_epoch/np.ceil(train_data_size/bs))
        tot_train_loss3.append(train_loss3_epoch/np.ceil(train_data_size/bs))
        """
        Test phase
        """
        posenet.eval()
        with torch.no_grad():
            test_loss_epoch = 0
            test_loss1_epoch = 0
            test_loss2_epoch = 0
            test_loss3_epoch = 0
            test_tqdm = tqdm(test_dataloader, dynamic_ncols=True, leave=False)
            for pcloud, labels in test_dataloader:
                pcloud, labels = pcloud.to(DEVICE), labels.to(DEVICE)
                T_preds, T_feas = teachernet(pcloud)
                S_preds, S_feas = posenet(pcloud)
                loss, loss1, loss2, loss3 = loss_fc(S_preds, T_preds, labels, S_feas, T_feas)
                test_loss_epoch += loss.item()
                test_loss1_epoch += loss1.item()
                test_loss2_epoch += loss2.item()
                test_loss3_epoch += loss3.item()
                test_tqdm.set_description("Test loss: {}".format(round(loss.item(), 5)))
                test_tqdm.update()
            tot_test_loss.append(test_loss_epoch/np.ceil(test_data_size/bs))
            tot_test_loss1.append(test_loss1_epoch/np.ceil(test_data_size/bs))
            tot_test_loss2.append(test_loss2_epoch/np.ceil(test_data_size/bs))
            tot_test_loss3.append(test_loss3_epoch/np.ceil(test_data_size/bs))
        """
        Checkpoint save
        """
        if (epoch+1) % checkpoint_interval == 0:
            checkpoint = {
                "model_state_dict": posenet.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch * len(train_dataloader),
                "tot_train_loss": tot_train_loss,
                "tot_train_loss1": tot_train_loss1,
                "tot_train_loss2": tot_train_loss2,
                "tot_train_loss3": tot_train_loss3,
                "tot_test_loss": tot_test_loss,
                "tot_test_loss1": tot_test_loss1,
                "tot_test_loss2": tot_test_loss2,
                "tot_test_loss3": tot_test_loss3
            }
            path_checkpoint = os.path.join(file_path_ck, "checkpoint_{}_epoch.pkl".format(epoch))
            torch.save(checkpoint, path_checkpoint)

        SW.add_scalars('paras/loss', {'train loss': tot_train_loss[epoch],
                                      'train loss1': tot_train_loss1[epoch],
                                      'train loss2': tot_train_loss2[epoch],
                                      'train loss3': tot_train_loss3[epoch],
                                        'test loss': tot_test_loss[epoch],
                                        'test loss1': tot_test_loss1[epoch],
                                        'test loss2': tot_test_loss2[epoch],
                                        'test loss3': tot_test_loss3[epoch]}, 
                                        epoch)

        epoch_tqdm.set_description("Train loss: {}, Test loss: {}"
                            .format(round(tot_train_loss[epoch], 5), round(tot_test_loss[epoch], 5)))
        epoch_tqdm.update()

    datet = datetime.datetime.now().strftime('%m%d_%H%M')
    torch.save(posenet.cpu().state_dict(),
               os.path.join("data", "posenet_" + datet + ".pth"))
    print("model saved")

    tot_time = time.time() - since_time
    np.savez(os.path.join('data', 'tot_metric_' + datet), tot_train_loss=tot_train_loss,
             tot_test_loss=tot_test_loss, tot_time=tot_time)
    SW.close()