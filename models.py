import torch
from torch import nn

from model.pointnet_util import PPointNet2
from model.stgcn_util import stgcnModel


class KPTSNet(nn.Module):
    def __init__(self, nkpoint=17, dimkpoint=3, ch=5, nframes=15):
        super().__init__()
        self.nkpoint = nkpoint
        self.nframes = nframes
        self.dimkpoint = dimkpoint
        self.pointnet2 = PPointNet2(ch)
        self.gru = nn.GRU(input_size=1024, hidden_size=512, num_layers=2,
                          batch_first=True, dropout=0.5, bidirectional=True
                          )
        self.mlp = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, self.nkpoint * self.dimkpoint),
        )
        self.tcn = nn.Sequential(
            nn.BatchNorm1d(self.nframes),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.nframes, 32, 5, 3, ),
            nn.MaxPool1d(3),
            nn.Dropout(0.5, inplace=True),#0.2
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, 3, 2, ),
            nn.MaxPool1d(2),
            nn.Dropout(0.5, inplace=True),#0.2
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 3, 2, ),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(1664, 10)
        )
        # self.tcn = nn.Sequential(
        #     nn.BatchNorm2d(3),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(3, 16, 4, (1, 2), ),
        #     nn.Dropout(0.2, inplace=True),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(16, 32, 3, 2, ),
        #     nn.Dropout(0.2, inplace=True),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(32, 64, 3, 1, ),
        #     nn.ReLU(inplace=True),
        #     nn.Flatten(),
        #     nn.Linear(64, 9)
        # )


        # self.stgcn = stgcnModel(in_channels=3, num_class=9, graph_args={'strategy': 'spatial'}, edge_importance_weighting=True)

    def forward(self, pc_input):
        B, T, C, N = pc_input.shape
        comb_in = torch.reshape(pc_input, (-1, C, N))
        features = None if C == 3 else comb_in[:, 3:, :]
        pn_out = self.pointnet2(comb_in[:, :3, :], features)
        gru_in = torch.reshape(pn_out, (B, T, -1))
        gru_out, _ = self.gru(gru_in)
        mlp_in = torch.reshape(gru_out, (B * T, -1))
        mlp_out = self.mlp(mlp_in)
        mlp_trans = torch.reshape(mlp_out, (B, T, -1))
        # stgcn_in = torch.reshape(mlp_trans, (B, T, self.nkpoint, self.dimkpoint,))
        # stgcn_out = self.stgcn(stgcn_in)
        tcn_out = self.tcn(gru_out)
        # tcn_out = torch.sigmoid(tcn_out)
        # tcn_in = torch.reshape(mlp_trans, (-1, self.nframes, 17, 3)).permute(0, 3, 1, 2)
        # tcn_out = self.tcn(tcn_in)
        return mlp_trans[:, -1, :], tcn_out


if __name__ == '__main__':
    import os

    import torch

    N = 15
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    input = torch.randn((4, N, 5, 256))
    model = KPTSNet(nframes=N)
    output_keypoint, output_pose = model(input)
    print(output_keypoint.size())
    print(output_pose.size())
    
    
    
    
    
class KPTSNet_S2(nn.Module):
    def __init__(self, nkpoint=17, dimkpoint=3, ch=5, nframes=15):
        super().__init__()
        self.nkpoint = nkpoint
        self.nframes = nframes
        self.dimkpoint = dimkpoint
        self.pointnet2 = PPointNet2(ch)
        self.CNN = cnn()

        self.mlp = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, self.nkpoint * self.dimkpoint),
        )
        self.tcn = nn.Sequential(
                    nn.BatchNorm1d(self.nframes),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(self.nframes, 32, 5, 3, ),
                    nn.MaxPool1d(3),
                    nn.Dropout(0.5, inplace=True),#0.2
                    nn.ReLU(inplace=True),
                    nn.Conv1d(32, 64, 3, 2, ),
                    nn.MaxPool1d(2),
                    nn.Dropout(0.5, inplace=True),#0.2
                    nn.ReLU(inplace=True),
                    nn.Conv1d(64, 128, 3, 2, ),
                    nn.ReLU(inplace=True),
                    nn.Flatten(),
                    nn.Linear(1664, 9)
            )
    def forward(self, pc_input):
        B, T, C, N = pc_input.shape
        comb_in = torch.reshape(pc_input, (-1, C, N))
        features = None if C == 3 else comb_in[:, 3:, :]
        pn_out = self.pointnet2(comb_in[:, :3, :], features)
        cnn_in = torch.reshape(pn_out, (B, T, -1))#tcn_in = torch.reshape(pn_out, (B, T, -1))
        cnn_out = self.CNN(cnn_in)
        mlp_in = torch.reshape(cnn_out, (B * T, -1))
        mlp_out = self.mlp(mlp_in)
        mlp_trans = torch.reshape(mlp_out, (B, T, -1))
        tcn_out = self.tcn(cnn_out)
        return mlp_trans, tcn_out
    
    
    
class cnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
        nn.Conv1d(15, 64, 5, 2, 0, 1),#nn.Conv1d(10, 64, 5, 2, 0, 1)
        nn.ReLU(),
        nn.BatchNorm1d(64),
        nn.Conv1d(64, 256, 5, 3, 0, 2),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Conv1d(256, 512, 5, 2, 0, 4),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Conv1d(512, 1024, 5, 2, 0, 8),
        )

    def forward(self, x):
        out = self.cnn(x)
        return out[..., -15:].transpose(1, 2) 
    
    
    
    
    
    
class KPTSNet_S2_Action_GRU(nn.Module):
    def __init__(self, nkpoint=17, dimkpoint=3, ch=5,nframes=15):
        super().__init__()
        self.nkpoint = nkpoint
        self.dimkpoint = dimkpoint
        self.pointnet2 = PPointNet2(ch)
        self.nframe=nframes
        self.gru = nn.GRU(1024, 512, 2, bidirectional=True, batch_first=True,dropout=0.5)
        self.mlp = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, self.nkpoint * self.dimkpoint),
        )
        # self.action_layernorm = nn.LayerNorm([self.nframe,51])
        self.gru_action = nn.GRU(51, 153, 1, bidirectional=False, batch_first=False)
        self.mlp_action = nn.Linear(153, 102)
        self.mlp_action_2 = nn.Linear(102, 9)

    def forward(self, pc_input):
        B, T, C, N = pc_input.shape
        comb_in = torch.reshape(pc_input, (-1, C, N))
        features = None if C == 3 else comb_in[:, 3:, :]
        pn_out = self.pointnet2(comb_in[:, :3, :], features)
        gru_in = torch.reshape(pn_out, (B, T, -1))
        gru_out,_ = self.gru(gru_in)
        gru_out = gru_out
        mlp_in = gru_out
        mlp_out = self.mlp(mlp_in)
        # action_in = self.action_layernorm(mlp_out)
        action_out,_=self.gru_action(mlp_out.transpose(0,1))
        action_out=self.mlp_action(action_out[-1])
        action_out=self.mlp_action_2(action_out)
        return mlp_out[:,-1], action_out