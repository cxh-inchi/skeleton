from model.models import *
import datetime
import time
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from tqdm import tqdm
import torch.multiprocessing as mp

from functions import *
import os
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import argparse

os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29512'

parser = argparse.ArgumentParser(description='mmW Radar Pose Training')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--decay', type=float, default=0.99, help='weight decay for ExponentialLR')
parser.add_argument('--bs', type=int, default=128, help='batch_size per node ')
parser.add_argument('--MAX_EPOCH', type=int, default=45, help='Maximum training epoch number')
parser.add_argument('--checkpoint_interval', type=int, default=5, help='the epoch interval between storing checkpoints')
parser.add_argument('--Nframes', type=int, default=15, help='number of successive time frames')
parser.add_argument('--ch', type=int, default=5, help='point cloud dim')
parser.add_argument('--seed', type=int, default=1, help='seed for initializing training. ')
parser.add_argument('--npoint', type=int, default=256, help='fixed point number for infer input. ')
parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--Ind_pretrain', type=bool, default=True, help='indicator to use pretrain model')
parser.add_argument('--pre_md_pth', default='data\\posenet_1215_256.pth', type=str, help='pretrained model path')
parser.add_argument('--Ind_checkpoint', type=bool, default=True, help='indicator to checkpoint')
parser.add_argument('--chkpnt_path', default='checkpoint\\ck_1519\\checkpoint_44_epoch.pkl', type=str,
                    help='check point model path')
parser.add_argument('--file_root_path', default='D:\\1-humanpose\\1-traincode\\humanpose_traincode', type=str,
                    help='root path of the code')


def main():
    args = parser.parse_args()
    args.nprocs = torch.cuda.device_count()
    mp.spawn(main_worker, nprocs=args.nprocs, args=(args.nprocs, args))
    # local_rank = int(os.environ["LOCAL_RANK"])
    # main_worker(local_rank,args.nprocs, args)


def main_worker(local_rank, nprocs, args):
    torch.set_num_threads(2)
    args.local_rank = local_rank
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    start_epoch = 0

    train_root_paths = [
        r"D:\1-humanpose\2-traindata\trainset_0720_match",
        r"D:\1-humanpose\2-traindata\trainset_0728_match",
        r"D:\1-humanpose\2-traindata\trainset_0825_g",
        r"D:\1-humanpose\2-traindata\trainset_0825_k",
        r"D:\1-humanpose\2-traindata\trainset_1212",
    ]
    test_root_paths = [
        r"D:\1-humanpose\2-traindata\testset_0720_match",
        r"D:\1-humanpose\2-traindata\testset_0728_match",
        r"D:\1-humanpose\2-traindata\testset_0825_k",
        r"D:\1-humanpose\2-traindata\testset_0825_g",
    ]

    train_full_paths = get_full_paths(train_root_paths)
    test_full_paths = get_full_paths(test_root_paths)

    torch.distributed.init_process_group("nccl", init_method='tcp://127.0.0.1:29512', world_size=args.nprocs,
                                         rank=local_rank)
    torch.cuda.set_device(local_rank)

    if local_rank == 0:
        since_time = time.time()
        dt = datetime.datetime.now().strftime('%m%d_%H%M')
        file_path_ck = os.path.join(args.file_root_path, "checkpoint", "ck_" + dt)
        if not os.path.exists(file_path_ck):
            os.makedirs(file_path_ck)

        savedatet = datetime.datetime.now().strftime('%d_%H%M')

        SW = SummaryWriter(
            os.path.join(args.file_root_path, "logs_train",
                         "train_" + savedatet + 'bs_' + str(args.bs) + 'lr_' + str(args.lr))
        )

    train_data = PCDataSet(train_full_paths, ind_jitter=True, ind_rnd_gap=True, ind_cut=True, ind_rot=True,
                           ind_tl=False,
                           ind_scale=True, ind_reduce=True, Nframes=args.Nframes, dimpcloud=args.ch, npoint=args.npoint)

    train_data_size = len(train_data)
    test_data = PCDataSet(test_full_paths, ind_rnd_gap=True, Nframes=args.Nframes, dimpcloud=args.ch,
                          npoint=args.npoint)
    test_data_size = len(test_data)
    print("Train set length: {}, test set length: {}".format(train_data_size, test_data_size))
    train_sampler = DistributedSampler(train_data)
    train_dataloader = DataLoader(train_data, batch_size=args.bs, pin_memory=True, sampler=train_sampler)
    test_dataloader = DataLoader(test_data, batch_size=args.bs)

    posenet = KPTSNet(nkpoint=17, dimkpoint=3, ch=args.ch).cuda(local_rank)

    if args.Ind_pretrain:
        model_path = os.path.join(args.file_root_path, args.pre_md_pth)
        preweights = torch.load(model_path)
        posenet.load_state_dict(preweights, strict=False)

    posenet = DistributedDataParallel(posenet, device_ids=[local_rank], output_device=local_rank)
    loss_fc = LossFunc(nn.MSELoss(reduction='sum').cuda(local_rank), args.bs)
    optimizer = torch.optim.Adam(posenet.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler_mul = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.decay)

    tot_train_loss = []
    tot_test_loss = []
    """
    Checkpoint resume
    """
    if args.Ind_checkpoint:
        path_checkpoint = os.path.join(args.file_root_path, args.chkpnt_path)
        checkpoint = torch.load(path_checkpoint, map_location='cpu')
        posenet = KPTSNet(nkpoint=17, dimkpoint=3, ch=args.ch)
        posenet.load_state_dict(checkpoint['model_state_dict'])
        posenet = DistributedDataParallel(posenet.cuda(local_rank), device_ids=[local_rank], output_device=local_rank)
        optimizer.load_state_dict((checkpoint['optimizer_state_dict']))
        start_epoch = checkpoint['epoch'] + 1
        scheduler_mul.last_epoch = start_epoch - 1
        tot_train_loss = checkpoint['tot_train_loss']
        tot_test_loss = checkpoint['tot_test_loss']

    if local_rank == 0:
        epoch_tqdm = tqdm(range(start_epoch, args.MAX_EPOCH), dynamic_ncols=True, leave=False)

    for epoch in range(start_epoch, args.MAX_EPOCH):
        train_sampler.set_epoch(epoch)

        """
        Training phase
        """
        posenet.train()
        train_loss_epoch = 0
        if local_rank == 0:
            train_tqdm = tqdm(train_dataloader, dynamic_ncols=True, leave=False)
        for pcloud, labels in train_dataloader:
            pcloud, labels = pcloud.cuda(local_rank, non_blocking=True), labels.cuda(local_rank, non_blocking=True)
            outputs = posenet(pcloud)
            loss = loss_fc(outputs, labels)
            torch.distributed.barrier()
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            train_loss_epoch += reduce_mean(loss).item()
            if local_rank == 0:
                train_tqdm.set_description("Train loss: {}".format(round(loss.item(), 5)))
                train_tqdm.update()
        if local_rank == 0:
            tot_train_loss.append(train_loss_epoch / len(train_dataloader))
        scheduler_mul.step()
        """
        Test phase
        """
        if local_rank == 0:
            posenet.eval()
            with torch.no_grad():
                test_loss_epoch = 0

                test_tqdm = tqdm(test_dataloader, dynamic_ncols=True, leave=False)
                for pcloud, labels in test_dataloader:
                    pcloud, labels = pcloud.cuda(non_blocking=True), labels.cuda(non_blocking=True)
                    outputs = posenet(pcloud)
                    loss = loss_fc(outputs, labels)
                    test_loss_epoch += loss.item()
                    test_tqdm.set_description("Test loss: {}".format(round(loss.item(), 5)))
                    test_tqdm.update()
                tot_test_loss.append(test_loss_epoch / len(test_dataloader))

        if local_rank == 0:
            """
            Checkpoint save
            """
            if (epoch + 1) % args.checkpoint_interval == 0:
                checkpoint = {
                    "model_state_dict": posenet.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "tot_train_loss": tot_train_loss,
                    "tot_test_loss": tot_test_loss
                }
                path_checkpoint = os.path.join(file_path_ck, "checkpoint_{}_epoch.pkl".format(epoch))
                torch.save(checkpoint, path_checkpoint)

            SW.add_scalars('paras/loss', {'train loss': tot_train_loss[epoch],
                                          'test loss': tot_test_loss[epoch]}, epoch)

            epoch_tqdm.set_description("Train loss: {}, Test loss: {}"
                                       .format(round(tot_train_loss[epoch], 5), round(tot_test_loss[epoch], 5)))
            epoch_tqdm.update()

    if local_rank == 0:
        datet = datetime.datetime.now().strftime('%m%d_%H%M')
        torch.save(posenet.module.cpu().state_dict(),
                   os.path.join(args.file_root_path, "data", "posenet_" + datet + ".pth"))
        print("model saved")

        tot_time = time.time() - since_time
        np.savez(os.path.join('data', 'tot_metric_' + datet), tot_train_loss=tot_train_loss,
                 tot_test_loss=tot_test_loss, tot_time=tot_time)
        SW.close()


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


def reduce_sum(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt


if __name__ == '__main__':
    main()
