
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, MaxPool1d, Linear, GRU


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from model.Encoder import Encoder


class Embed2GraphByProduct(nn.Module):

    def __init__(self, input_dim, roi_num=264):
        super().__init__()

    def forward(self, x):

        m = torch.einsum('ijk,ipk->ijp', x, x)

        m = torch.unsqueeze(m, -1)

        return m

class GCNPredictor(nn.Module):

    def __init__(self, node_input_dim, roi_num=360):
        super().__init__()
        inner_dim = roi_num
        self.roi_num = roi_num
        self.gcn = nn.Sequential(
            nn.Linear(node_input_dim, inner_dim),
            nn.LeakyReLU(negative_slope=0.2),
            Linear(inner_dim, inner_dim)
        )
        self.bn1 = torch.nn.BatchNorm1d(inner_dim)

        self.gcn1 = nn.Sequential(
            nn.Linear(inner_dim, inner_dim),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.bn2 = torch.nn.BatchNorm1d(inner_dim)
        self.gcn2 = nn.Sequential(
            nn.Linear(inner_dim, 64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(64, 8),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.bn3 = torch.nn.BatchNorm1d(inner_dim)


        self.fcn = nn.Sequential(
            nn.Linear(int(8 * int(roi_num * 0.7)), 256),
            # nn.Linear(8 * roi_num, 256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(256, 32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(32, 2)
        )
        self.norm = torch.nn.LayerNorm(normalized_shape=roi_num, elementwise_affine=True)
        self.weight = torch.nn.Parameter(torch.Tensor(1, 8))

        self.softmax = nn.Sigmoid()


    def forward(self, m, node_feature):
        bz = m.shape[0]

        x = torch.einsum('ijk,ijp->ijp', m, node_feature)

        x = self.gcn(x)

        x = x.reshape((bz*self.roi_num, -1))
        x = self.bn1(x)
        x = x.reshape((bz, self.roi_num, -1))

        x = torch.einsum('ijk,ijp->ijp', m, x)

        x = self.gcn1(x)

        x = x.reshape((bz*self.roi_num, -1))
        x = self.bn2(x)
        x = x.reshape((bz, self.roi_num, -1))

        x = torch.einsum('ijk,ijp->ijp', m, x)

        x = self.gcn2(x)

        x = self.bn3(x)

        score = (x * self.weight).sum(dim=-1)

        # score = self.norm(score)
        score = self.softmax(score)
        sc = score

        _, idx = score.sort(dim=-1)
        _, rank = idx.sort(dim=-1)

        l = int(m.shape[1] * 0.7)
        x_p = torch.empty(bz, l, 8)

        for i in range(x.shape[0]):
            x_p[i] = x[i, rank[i, :l], :]


        x = x_p.view(bz,-1).to(device)
        # x = x.view(bz,-1).to(device)


        # return self.fcn(x), x
        return self.fcn(x), sc


class PLSNet(nn.Module):

    def __init__(self, model_config, roi_num=360, node_feature_dim=360, time_series=512):
        super().__init__()

        self.extract = Encoder(input_dim=time_series, num_head=4, embed_dim=model_config['embedding_size'])




        self.emb2graph = Embed2GraphByProduct(
                model_config['embedding_size'], roi_num=roi_num)

        self.predictor = GCNPredictor(node_feature_dim, roi_num=roi_num)
        self.fc_q = nn.Sequential(nn.Linear(in_features=model_config['embedding_size'], out_features=roi_num),
                                  nn.LeakyReLU(negative_slope=0.2))


        self.fc_p = nn.Sequential(nn.Linear(in_features=roi_num, out_features=roi_num),
                                  nn.LeakyReLU(negative_slope=0.2))

    def forward(self, t, nodes, pseudo):
        x = self.extract(t)
        m = F.softmax(x, dim=-1)
        m = self.emb2graph(m)

        m = m[:, :, :, 0]

        bz, _, _ = m.shape

        edge_variance = torch.mean(torch.var(m.reshape((bz, -1)), dim=1))

        pseudo = self.fc_p(pseudo)
        nodes = nodes + pseudo
        return self.predictor(m, nodes), m, edge_variance

