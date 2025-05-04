import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from methods.standardModel import GNNModel, EarlyStopping
from methods.cluster import Cluster
from PreProcess import NormalOrder, ClusterOrder

lr = 0.01
epochs = 100
model_save_path = "output"
dropout = 0.5
patience = 20
file_path = 'DataSets'  # 替换为实际的文件路径
output_path = 'output/dataSet.csv'
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
dimension =50
nhead = 1
batch_size = 64



def multiModel():
    loss_record = []
    n_cluster = 1
    nodes =10
    # cluster_center聚类中心，n_clusters类别数，num每个类的样本个数,maxerror簇中最多的异常值个数
    cluster_center, n_clusters, num, special_cluster = Cluster(file_path, n_clusters=n_cluster)
    cluster_center = torch.tensor(cluster_center).to(device).float()
    input_train_set, input_distance_set, input_order_set, train_value_set, train_order_set, train_distance_set, train_label_set, u_in, nodes = NormalOrder(
        device, n_clusters, dimension, cluster_center, True,nodes,False)
    # input_train_set, input_distance_set,input_order_set, train_value_set, train_order_set, train_distance_set, train_label_set, u_in, nodes=ClusterOrder(device,n_clusters,dimension,cluster_center,True)

    early_stopping = EarlyStopping(patience=patience, path=os.path.join(model_save_path, 'GNN_model.pth'))

    # models训练
    models = GNNModel(nfeat=dimension, nodes=nodes, nhead=nhead).to(device)
    models_optimizer = torch.optim.Adam(models.parameters(), lr=lr)
    start = 0  # 迭代器起始点
    criterion = nn.MSELoss()
    print("Start training GNN model")

    for e in range(start, epochs):
        epochs_loss = 0
        for b in range(0, input_train_set.size(0), batch_size):
            input_train_set_batch = input_train_set[b:b + batch_size]
            input_distance_set_batch = input_distance_set[b:b + batch_size]
            input_order_set_batch = input_order_set[b:b + batch_size]
            input_train_set_batch = input_train_set_batch.to(device)
            input_distance_set_batch = input_distance_set_batch.to(device)
            input_order_set_batch = input_order_set_batch.to(device)
            input_train_set_batch.requires_grad = True
            input_distance_set_batch.requires_grad = True
            input_order_set_batch.requires_grad = True
            input_train_set_batch = input_train_set_batch.float()
            input_distance_set_batch = input_distance_set_batch.float()
            models_optimizer.zero_grad()
            outputs = models(input_train_set_batch,input_distance_set_batch)
            # 计算损失
            loss = criterion(outputs, input_train_set_batch)
            epochs_loss += loss.item()
            loss.backward()
            models_optimizer.step()
            loss_record.append(loss.item())
        ave_loss = epochs_loss / (input_train_set.size(0) / batch_size)
        print("epoch:", e, "loss:", ave_loss)
        early_stopping(ave_loss, models)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    torch.save(models.state_dict(), os.path.join(model_save_path, 'GNN_model1.pth'))
    models.eval()  # 将模型设置为评估模式

    nodes =10
    input_train_set, input_distance_set, input_order_set, train_value_set, train_order_set, train_distance_set, train_label_set, u_in, nodes = NormalOrder(
        device, n_clusters, dimension, cluster_center, True,nodes,True)
    # 训练集上效果
    train_outputs, e_in, e_out = models.encoder(train_value_set,train_distance_set)
    e_in = e_in.view(-1, dimension).cpu().detach().numpy()

    # 计算特征距离
    reshaped_tensor = e_out.view(-1, dimension).cpu().detach().numpy()
    labels = train_label_set.cpu().detach().numpy()

    cluster_labels = np.empty(0)
    orders = train_order_set.cpu().detach().numpy().reshape(-1)
    for i in range(len(orders)):
        cluster_labels = np.append(cluster_labels, special_cluster[int(orders[i])])

    # 输出原数据与处理数据
    df = pd.DataFrame(e_in)
    df.to_csv('output/train_origin.csv', header=False, index=False)

    array_all_col = labels.reshape(-1, 1)
    array_all_col = np.concatenate((array_all_col, e_in), axis=1)
    df = pd.DataFrame(array_all_col)
    df.to_csv('output/IMM_o.csv', header=False, index=True)

    df = pd.DataFrame(reshaped_tensor)
    df.to_csv('output/train_reshaped.csv', header=False, index=False)

    array_all_col = labels.reshape(-1, 1)
    array_all_col = np.concatenate((array_all_col, reshaped_tensor), axis=1)
    df = pd.DataFrame(array_all_col)
    df.to_csv('output/IMM_r.csv', header=False, index=True)

    df = pd.DataFrame(labels)
    df.to_csv('output/train_labels.csv', header=False, index=False)

    df = pd.DataFrame(loss_record)
    df.to_csv('output/loss_record.csv', header=False, index=False)

    df = pd.DataFrame(cluster_labels)
    df.to_csv('output/cluster_labels.csv', header=False, index=False)


multiModel()
