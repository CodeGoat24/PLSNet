import csv
import os
from typing import overload
import torch
from nilearn.connectome import ConnectivityMeasure
from nilearn import plotting
from numpy.lib import save
from torch.optim import lr_scheduler
from pathlib import Path
import argparse
import yaml
import torch
import os


from model.model import PLSNet

from dataloader import init_dataloader
from util import Logger, accuracy, TotalMeter
import numpy as np

import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
# device = torch.device("cpu")

count = {}
matrix_pearson = []
matrix_attention = []

def test_score(dataloader, model):
    labels = []
    result = []
    ranks = []


    for data_in, pearson, label, pseudo in dataloader:
        label = label.long()
        data_in, pearson, label, pseudo = data_in.to(
            device), pearson.to(device), label.to(device), pseudo.to(device)
        [output, score], matrix, _ = model(data_in, pearson, pseudo)

        result += F.softmax(output, dim=1)[:, 1].tolist()
        labels += label.tolist()
        _, idx = score.sort(dim=-1)
        _, rank = idx.sort(dim=-1)
        ranks += rank.detach().cpu().numpy().tolist()

        global matrix_pearson
        matrix_pearson += data_in.detach().cpu().numpy().tolist()
        global matrix_attention
        matrix_attention += matrix.detach().cpu().numpy().tolist()


    result = np.array(result)
    result[result > 0.5] = 1
    result[result <= 0.5] = 0
    ranks = np.array(ranks)
    ranks = ranks[result == 1]

    for r in ranks[:, :10]:
        for x in r:
            if x not in count:
                count[x] = 1
            else:
                count[x] = count[x] + 1





with open('setting/abide_PLSNet.yaml') as f:
    config = yaml.load(f, Loader=yaml.Loader)
    (train_dataloader, val_dataloader, test_dataloader), node_size, node_feature_size, timeseries_size = init_dataloader(config['data'])
    # print(config['data']['batch_size'])
    model = PLSNet(config['model'], node_size,
                 node_feature_size, timeseries_size).to(device)
    model.load_state_dict(torch.load('/home/star/CodeGoat24/FBNETGEN/result/ABIDE_AAL_71.43%/model_71.42857139098821%.pt'))

    model.eval()
    test_score(dataloader=train_dataloader, model=model)
    test_score(dataloader=val_dataloader, model=model)
    test_score(dataloader=test_dataloader, model=model)
    # count = sorted(count.items(), key=lambda d: d[1], reverse=True)
    # print(count)
    # print(sorted(count[:10]))
    csv_reader = csv.reader(open('aal_labels2.csv', encoding='utf-8'))
    label = [row[1] for row in csv_reader]
    # 画Pearson邻接矩阵
    # matrix_pearson = np.array(matrix_pearson)
    #
    # connectivity = ConnectivityMeasure(kind='correlation')
    # connectivity_matrices = connectivity.fit_transform(matrix_pearson.swapaxes(1, 2))
    # mean_connectivity_matrices = connectivity_matrices.mean(axis=0)
    # plotting.plot_matrix(mean_connectivity_matrices, figure=(10, 8), labels=label[2:], vmax=1.2, vmin=-0.6, colorbar=True, reorder=False, title="")
    # plotting.show()

    # 画attention邻接矩阵
    matrix_attention = np.array(matrix_attention)

    connectivity = ConnectivityMeasure(kind='correlation')
    connectivity_matrices = connectivity.fit_transform(matrix_attention.swapaxes(1, 2))
    mean_connectivity_matrices = connectivity_matrices.mean(axis=0)
    plotting.plot_matrix(mean_connectivity_matrices, figure=(10, 8), labels=label[2:], vmax=1.065, vmin=0.812, colorbar=True,
                         reorder=False, title="")
    plotting.show()

# vmax=1.065, vmin=0.815