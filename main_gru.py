import datetime
import itertools
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from functions.functions_idx import *
from model.models import *
import dill
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts,StepLR
from calc_rate import calculate_rates
from torch.utils.data import Subset
from functions.focal_loss import FocalLoss
os.environ['OMP_NUM_THREADS'] = '2'

if __name__ == '__main__':
    # lr_list = [0.001,0.0005,0.0001]
    # for idx in lr_list:
    for idx in [1]: 
        torch.set_num_threads(2)
        Ind_checkpoint = False
        Ind_pretrained=True
        # 学习率
        lr = 0.0001
        # lr = idx
        bs = 120 #120
        MAX_EPOCH = 50#200
        checkpoint_interval = 2
        test_interval = 1
        start_epoch = 0
        ch = 5
        Nframes = 15
        npoint = 256
        name = "sgru_kpt_action_3BN_pntgru"
        dt = datetime.datetime.now().strftime('%Y_%m%d_%H%M')+name
        file_path_ck = "./pointnet_based/checkpoint/ck_"+dt
        if not os.path.exists(file_path_ck):
            os.makedirs(file_path_ck)
        file_path_matrix = "./pointnet_based/conf_matrix/ck_"+dt
        if not os.path.exists(file_path_matrix):
            os.makedirs(file_path_matrix)

        # DEVICE = torch.device('cpu')
        DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        # savedatet = datetime.datetime.now().strftime('%d_
        # %H%M')
        SW = SummaryWriter("./pointnet_based/logs_train/train_"+dt+'bs_'+str(bs)+'lr_'+str(lr))

        # root_path0 = '/datahdd/wbl/data/data/03144'
        # file_list = os.listdir(root_path0)
        # root_path1 = [os.path.join(root_path0, item) for item in file_list]
        # root_path2 = '/datahdd/wbl/data/data/testset_0825_k'
        # file_list_t = os.listdir(root_path2)
        # root_path3 = [os.path.join(root_path2, item) for item in file_list_t]  



        train_root_paths = [
        #0328tht
        # r"data/new_data/0328tht/data_1116_0327",
        # r"/datahdd/smartradar/wsy/data/new_data/0328tht/trainset_0720_g",
        r"/datahdd/smartradar/wsy/data/new_data/0328tht/trainset_0728_match",
        # r"/datahdd/smartradar/wsy/data/new_data/0328tht/trainset_0825_g",
        r"/datahdd/smartradar/wsy/data/new_data/0328tht/trainset_0825_k",
        # r"/datahdd/smartradar/wsy/data/new_data/0328tht/train_0801",
        # r"data/new_data/0328tht/train_0809",
        ]
        test_root_paths = [
        # 0328tht
        r"/datahdd/smartradar/wsy/data/new_data/0328tht/testset_0720_g",
        r"/datahdd/smartradar/wsy/data/new_data/0328tht/testset_0728_k",
        r"/datahdd/smartradar/wsy/data/new_data/0328tht/testset_0825_g",
        r"/datahdd/smartradar/wsy/data/new_data/0328tht/testset_0825_k",
        ]

        # train_full_paths = [os.path.join(rp_item, fn_item) for rp_item in train_root_paths for fn_item in
        #                 os.listdir(rp_item)]
        # test_full_paths = [os.path.join(rp_item, fn_item) for rp_item in test_root_paths for fn_item in os.listdir(rp_item)]
        



        # train_data = PCDataSet(train_full_paths, dfilename='points', ind_jitter=True, ind_rnd_gap=True, ind_cut=True, ind_rot=True, ind_shuffle=True,
        #                                ind_scale=True, ind_tl=False, balance = True, Nframes=Nframes, dimpcloud=ch, npoint=npoint)
        # with open('/datahdd/smartradar/wsy/data/pkl/train_0829_15f_debug1.pkl','wb') as f:
        #     dill.dump(train_data, f)

        # test_data = PCDataSet(test_full_paths, dfilename='points', ind_jitter=False, ind_rnd_gap=False, balance = True, Nframes=Nframes, dimpcloud=ch, npoint=npoint)
        # with open('/datahdd/smartradar/wsy/data/pkl/test_0829_15f_debug1.pkl','wb') as f:
        #     dill.dump(test_data, f)
        #     continue
        with open('/datahdd/smartradar/wsy/data/pkl/train1129_15f.pkl','rb') as f:
            train_data = dill.load(f)            
    
        with open('/datahdd/smartradar/wsy/data/pkl/test1119_15f.pkl','rb') as f:
            test_data = dill.load(f)
           
            


        
        train_data_size = len(train_data)
        test_data_size = len(test_data)
        print("Train set length: {}, test set length: {}".format(train_data_size, test_data_size))
        
        # num_samples = train_data_size // 10
        # indices = np.random.choice(train_data_size, size=num_samples, replace=False)
        # train_subset = Subset(train_data, indices)
        
        train_dataloader = DataLoader(train_data, batch_size=bs, shuffle=True, num_workers=8, drop_last=True)
        test_dataloader = DataLoader(test_data, batch_size=bs, drop_last=True)
        
        posenet = KPTSNet_action(nkpoint=17, dimkpoint=3, ch=ch, nframes=Nframes).to(DEVICE)
        pretrained_path = '/datahdd/smartradar/cxh/mmwhumanpose_training/data/sgru_kpts_action_3BN_newtcn.pth'#posenet_0424_8frames
        # pre_model = torch.load(pretrained_path).to(DEVICE)
        # pretrained_weights = pre_model.state_dict()
        if Ind_pretrained:
            pretrained_weights = torch.load(pretrained_path)
            posenet.load_state_dict(pretrained_weights, strict=True)
        

         
        # posenet.pointnet2.load_state_dict(pre_model.pointnet2.state_dict(), strict=False)
        # posenet.gru.load_state_dict(pre_model.gru.state_dict(), strict=False)
        # posenet.mlp.load_state_dict(pre_model.mlp.state_dict(), strict=False)
        stgcn_para = [
            # {'params': posenet.pointnet2.parameters(), 'lr': lr},
            # {'params': posenet.gru.parameters(), 'lr': lr},
            {'params': posenet.mlp.parameters(), 'lr': lr},
            
        ]
        # summary(posenet, (6, 3, 512), batch_size=1, device=DEVICE.type)
        #loss_fc = LossFunc(nn.MSELoss(reduction='sum').to(DEVICE), bs)
        # loss_fc = nn.CrossEntropyLoss()
        # weights = torch.tensor([1, 1.5, 1, 1, 1.2, 1, 1.2, 1.5, 1, 0]).to(DEVICE)
        # loss_fc = nn.CrossEntropyLoss(reduction='none',weight=weights)
        alpha = torch.tensor([0, 0, 0, 0, 0, 0, 0, 10, 0]).to(DEVICE)
        loss_fc = FocalLoss(gamma=2, alpha = alpha)

    
        optimizer = torch.optim.Adam(stgcn_para, lr=lr, weight_decay=1e-3)
        # optimizer = torch.optim.Adam(posenet.parameters(), lr=lr, weight_decay=1e-2)

        scheduler_mul = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)#0.99
        # 定义余弦退火调度器
        # scheduler_cosine = CosineAnnealingLR(optimizer, T_max=10)  # T_max 为一个周期的迭代次数
        # scheduler_cosine = CosineAnnealingWarmRestarts(optimizer,T_0=5,T_mult=1)




        tot_train_loss = []
        tot_test_loss = []
        tot_train_accuracy = []
        tot_test_accuracy = []
        best_tot_loss = 10
        """
        Checkpoint resume
        """
        if Ind_checkpoint:
            path_checkpoint = "/datahdd/smartradar/cxh/mmwhumanpose_training/pointnet_based/checkpoint/ck_2024_1209_1142sgru_kpt_action_3BN_pntgru/checkpoint_3_epoch.pkl"
            checkpoint = torch.load(path_checkpoint)
            posenet.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict((checkpoint['optimizer_state_dict']))
            start_epoch = checkpoint['epoch'] + 1
            scheduler_mul.last_epoch = start_epoch - 1
            tot_train_loss = checkpoint['tot_train_loss']
            tot_test_loss = checkpoint['tot_test_loss']
            tot_train_accuracy = checkpoint['tot_train_accuracy']
            tot_test_accuracy = checkpoint['tot_test_accuracy']
        since_time = time.time()
        epoch_tqdm = tqdm(range(start_epoch, MAX_EPOCH), dynamic_ncols=True, leave=False)
        
        for epoch in range(start_epoch, MAX_EPOCH):
            """
            Training phase
            """
            # epoch_idx = 0
            y_true_train = []
            y_pred_train = []
            accuracy_train = []
            posenet.train()
            train_loss_epoch = 0
            train_tqdm = tqdm(train_dataloader, dynamic_ncols=True, leave=False)
            

            for pcloud, labels  in train_dataloader:
                batch_idx = 0
                batch_idx += 1
                pcloud, labels = pcloud.to(DEVICE), labels.to(DEVICE)
                # class_weights = torch.tensor(class_weights, dtype=torch.float32).cuda()  # 如果在使用 GPU
                
                _, outputs = posenet(pcloud)
                # get_weights(class_weights)
                # labels= change(labels)
                # labels, ignore_list = change(labels)
                kpt_mask = torch.ones_like(labels)
                # for i in ignore_list:
                #     mask[i] = 0
                # kpt_mask = mask != 0

                labels_mask = labels
                loss = loss_fc(outputs, labels_mask.long())  * kpt_mask
                loss = loss.sum() / kpt_mask.sum()
                loss = loss*10
                # loss = loss_fc(outputs, labels)
                train_loss_epoch += loss.item()
                
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
                # 获取当前迭代次数
                iteration = epoch * len(train_dataloader) + batch_idx
                _, predicted = torch.max(outputs, 1)
                y_true_train.extend(labels.cpu().numpy())  # 将真实标签添加到列表中
                y_pred_train.extend(predicted.cpu().numpy())  # 将预测标签添加到列表中
                
                train_tqdm.set_description("Train loss: {}".format(round(loss.item(), 5)))
                train_tqdm.update()

            conf_matrix_train = confusion_matrix(y_true_train, y_pred_train)
            correct_samples_train = conf_matrix_train.diagonal().sum()
            total_samples_train = conf_matrix_train.sum()
            accuracy_train = correct_samples_train / total_samples_train
            tot_train_accuracy.append(accuracy_train)
            scheduler_mul.step()

            # scheduler_cosine.step()
            # # 将学习率记录到TensorBoard
            current_lr = optimizer.param_groups[0]['lr']
            SW.add_scalar('Learning Rate', current_lr, epoch)
            tot_train_loss.append(train_loss_epoch/np.ceil(train_data_size/bs))
            """
            Test phase
            """
            y_true_test = []
            y_pred_test = []
            accuracy_test = []
            if (epoch+1) % test_interval == 0:
                posenet.eval()
                test_loss_epoch = 0
                test_tqdm = tqdm(test_dataloader, dynamic_ncols=True, leave=False)
                for pcmaps, labels in test_tqdm:
                    pcmaps, labels = pcmaps.to(DEVICE), labels.to(DEVICE)
                    _, outputs = posenet(pcmaps)

                    # labels= change(labels)
                    # labels, ignore_list = change(labels)
                    kpt_mask = torch.ones_like(labels)
                    # for i in ignore_list:
                    #     mask[i] = 0
                    # kpt_mask = mask != 0

                    labels_mask = labels
                
                    # expanded_kpt_mask = kpt_mask.unsqueeze(1)
                    # outputs_mask = outputs * expanded_kpt_mask

                    loss = loss_fc(outputs, labels_mask.long())  * kpt_mask
                    loss = loss.sum() / kpt_mask.sum()
                    loss = loss*10
                    # loss = loss_fc(outputs, labels)
                    test_loss_epoch += loss.item()
                    
                        
                    _, predicted = torch.max(outputs, 1)
                    # labels[labels == 2] = 0
                    # predicted[labels == 2] = 0
                    y_true_test.extend(labels.cpu().numpy())  # 将真实标签添加到列表中
                    y_pred_test.extend(predicted.cpu().numpy())  # 将预测标签添加到列表中
                    test_tqdm.set_description("Test loss: {}".format(round(loss.item(), 5)))
                    test_tqdm.update()
                conf_matrix_test = confusion_matrix(y_true_test, y_pred_test)
                correct_samples_test = conf_matrix_test.diagonal().sum()
                total_samples_test = conf_matrix_test.sum()
                accuracy_test = correct_samples_test / total_samples_test
                tot_test_accuracy.append(accuracy_test)
                tot_test_loss.append(test_loss_epoch/np.ceil(test_data_size/bs))
            
            if test_loss_epoch/np.ceil(test_data_size/bs) < best_tot_loss:
                best_tot_loss = test_loss_epoch/np.ceil(test_data_size/bs)
                best_model = posenet.cpu().state_dict()
                posenet.to(DEVICE)
                
            


            """
            Checkpoint save
            """
            if (epoch+1) % checkpoint_interval == 0:
                checkpoint = {
                    "model_state_dict": posenet.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "tot_train_loss": tot_train_loss,
                    "tot_test_loss": tot_test_loss,
                    'tot_train_accuracy':tot_train_accuracy,
                    'tot_test_accuracy':tot_test_accuracy
                }
                path_checkpoint = os.path.join(file_path_ck, "checkpoint_{}_epoch.pkl".format(epoch))
                torch.save(checkpoint, path_checkpoint)

            SW.add_scalars('paras/loss', {'train loss': tot_train_loss[epoch],
                                            'test loss': tot_test_loss[epoch]}, epoch)
            SW.add_scalars('Accuracy', {'train accuracy': tot_train_accuracy[epoch],
                                        'test accuracy': tot_test_accuracy[epoch]}, epoch)
            epoch_tqdm.set_description("Train loss: {}, Test loss: {},Train accuracy: {},Test accuracy: {}"
                                 .format(round(tot_train_loss[epoch], 5), round(tot_test_loss[epoch], 5),
                                         round(tot_train_accuracy[epoch], 5), round(tot_test_accuracy[epoch], 5)))
            epoch_tqdm.update()
            

            if (epoch + 1) % 10 == 0:
                # 定义类标签
                class_names = ['walking', 'run' , 'stand' , 'wave', 'sit', 'up', 'squat' #]
                            , 'fall' , 'lying']
            
                # 绘制混淆矩阵图
                plt.figure(figsize=(8, 6))
                plt.imshow(conf_matrix_test, interpolation='nearest', cmap=plt.cm.Blues)
                plt.title(f"Confusion Matrix Test at Epoch {epoch + 1}")
                plt.colorbar()
                tick_marks = np.arange(len(class_names))
                
                plt.xticks(tick_marks, class_names, rotation=45)
                plt.yticks(tick_marks, class_names)

                # 在每个格子中显示对应的数值
                thresh = conf_matrix_test.max() / 2.
                for i, j in itertools.product(range(conf_matrix_test.shape[0]), range(conf_matrix_test.shape[1])):
                    plt.text(j, i, format(conf_matrix_test[i, j], 'd'), horizontalalignment="center",
                            color="white" if conf_matrix_test[i, j] > thresh else "black")

                plt.ylabel('True label')
                plt.xlabel('Predicted label')
                plt.tight_layout()
                
                plt.savefig(file_path_matrix + f'/conf_matrix_test_epoch_{epoch + 1}.png', format='png')#保存图像为png格式
                
                # 清空当前图表，准备绘制下一张图
                plt.clf()

                # 绘制混淆矩阵图
                plt.figure(figsize=(8, 6))
                plt.imshow(conf_matrix_train, interpolation='nearest', cmap=plt.cm.Blues)
                plt.title(f"Confusion Matrix Train at Epoch {epoch + 1}")
                plt.colorbar()
                tick_marks = np.arange(len(class_names))
                plt.xticks(tick_marks, class_names, rotation=45)
                plt.yticks(tick_marks, class_names)

                # 在每个格子中显示对应的数值
                thresh = conf_matrix_train.max() / 2.
                for i, j in itertools.product(range(conf_matrix_train.shape[0]), range(conf_matrix_train.shape[1])):
                    plt.text(j, i, format(conf_matrix_train[i, j], 'd'), horizontalalignment="center",
                            color="white" if conf_matrix_train[i, j] > thresh else "black")
                    
                
                plt.ylabel('True label')
                plt.xlabel('Predicted label')
                plt.tight_layout()
                
                plt.savefig(file_path_matrix + f'/conf_matrix_train_epoch_{epoch + 1}.png', format='png')  # 保存图像为png格式
                
        torch.save(posenet.cpu().state_dict(), "./pointnet_based/data/action_" + dt + ".pth")
        torch.save(best_model, "./pointnet_based/data/action_bestmodel_" + dt + ".pth")
        # 只保存权重时， logs_train有一定问题
        # torch.save(posenet.state_dict(), "./data/posenet_" + datet + ".pth")
        print("model saved")

        tot_time = time.time() - since_time
        np.savez('./pointnet_based/data/tot_metric_' + dt, tot_train_loss=tot_train_loss,
                    tot_test_loss=tot_test_loss,  tot_train_accuracy=tot_train_accuracy, tot_test_accuracy=tot_test_accuracy, tot_time=tot_time)
        print("tot_train_accuracy:",max(tot_train_accuracy),"/ntot_test_accuracy:",max(tot_test_accuracy))
        SW.close()
        print(dt)
        print("Training complete in {:.0f}m {:.0f}s".format(tot_time // 60, tot_time % 60))

  # 计算测试集的误报率和漏报率
    fpr_test, fnr_test = calculate_rates(conf_matrix_test)
    fpr_train, fnr_train = calculate_rates(conf_matrix_train)  # 计算训练集的 FPR 和 FNR

    # 创建或打开 txt 文件（'a' 表示追加模式）
    with open('./pointnet_based/accuracy_results.txt', 'a') as f:
        f.write(f"Results for run on {dt}:\n")

        # 写入测试集每个类别的误报率和漏报率
        f.write("Test Set Class-wise False Positive Rate and False Negative Rate:\n")
        for i, class_name in enumerate(class_names):
            f.write(f"类别: {class_name}, 测试集 误报率: {fpr_test[i] * 100:.2f}%, 测试集 漏报率: {fnr_test[i] * 100:.2f}%\n")
        
        # 写入训练集每个类别的误报率和漏报率
        f.write("Train Set Class-wise False Positive Rate and False Negative Rate:\n")
        for i, class_name in enumerate(class_names):
            f.write(f"类别: {class_name}, 训练集 误报率: {fpr_train[i] * 100:.2f}%, 训练集 漏报率: {fnr_train[i] * 100:.2f}%\n")

        f.write("\n")


