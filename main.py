import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sympy.codegen.fnodes import dimension

from methods.standardModel import GNNModel, EarlyStopping
from methods.cluster import Cluster
from PreProcess import NormalOrder, ClusterOrder

lr = 0.01
epochs = 100
model_save_path = "output"
os.makedirs("output", exist_ok=True)
dropout = 0.5
patience = 20
file_path = 'DataSets/Annthyroid.csv'  # 替换为实际的文件路径
output_path = 'output/dataSet.csv'
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
nhead = 1
batch_size = 64


def multiModel():
    # 结果存储列表
    all_e_in = []
    all_e_out = []
    all_labels = []
    all_cluster_labels = []

    # 聚类处理并保存簇文件
    cluster_center, n_clusters, num, special_cluster = Cluster(file_path, n_clusters=7)
    dimension = cluster_center.shape[1]
    cluster_center = torch.tensor(cluster_center).to(device).float()

    # 为每个簇单独训练模型
    for cluster_id in range(n_clusters):
        print(f"\nProcessing Cluster {cluster_id}/{n_clusters - 1}")

        # 1. 处理当前簇数据
        input_train_set, input_distance_set, input_order_set, train_value_set, train_order_set, train_distance_set, train_label_set, u_in, nodes =\
            NormalOrder(device, cluster_id, dimension, cluster_center, True, 10)

        model = GNNModel(nfeat=dimension, nodes=nodes, nhead=nhead).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        early_stopping = EarlyStopping(
            patience=patience,
            path=os.path.join(model_save_path, f'GNN_model_cluster{cluster_id}.pth')
        )

        # 3. 训练模型
        criterion = nn.MSELoss()
        best_loss = float('inf')

        for epoch in range(epochs):
            model.train()
            total_loss = 0

            # 批量训练
            for b in range(0, input_train_set.size(0), batch_size):
                batch_data = input_train_set[b:b + batch_size].to(device).float()
                batch_dist = input_distance_set[b:b + batch_size].to(device).float()

                outputs = model(batch_data, batch_dist)

                loss = criterion(outputs, batch_data)
                total_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            avg_loss = total_loss / (input_train_set.size(0) // batch_size)
            print(f"Epoch {epoch} Loss: {avg_loss:.8f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), early_stopping.path)
            early_stopping(avg_loss, model)

            if early_stopping.early_stop:
                print("Early stopping triggered")
                break

        # 4. 评估模型
        model.load_state_dict(torch.load(early_stopping.path))
        model.eval()

        with torch.no_grad():
            train_outputs, e_in, e_out = model.encoder(train_value_set.to(device).float(),train_distance_set.to(device).float())

            # 转换为numpy数组
            e_in_np = e_in.view(-1, dimension).cpu().numpy()
            e_out_np = e_out.view(-1, dimension).cpu().numpy()
            labels_np = train_label_set.cpu().numpy()

            # 收集结果
            all_e_in.extend(e_in_np)
            all_e_out.extend(e_out_np)
            all_labels.extend(labels_np)

            # 生成簇标签
            cluster_labels = np.full(len(labels_np), cluster_id)
            all_cluster_labels.extend(cluster_labels)

    # 5. 最终结果聚合
    save_results(all_labels, all_e_in, all_e_out,all_cluster_labels, special_cluster)


def save_results(labels, e_in, e_out, cluster_labels, special_cluster):
    """保存最终结果到CSV文件"""
    labels = np.array(labels)
    e_in = np.array(e_in)
    e_out = np.array(e_out)
    cluster_labels = np.array(cluster_labels)


    pd.DataFrame(e_in).to_csv( 'output/train_origin.csv',header=False, index=False)


    reshaped_tensor = e_out.reshape(-1, e_out.shape[-1])
    pd.DataFrame(reshaped_tensor).to_csv('output/train_reshaped.csv',header=False, index=False)


    def save_with_labels(data, labels, filename):
        labeled_data = np.hstack((labels.reshape(-1, 1),data.reshape(-1, data.shape[-1])))
        pd.DataFrame(labeled_data).to_csv(filename,header=False, index=False)

    save_with_labels(e_in, labels, 'output/IMM_o.csv')
    save_with_labels(reshaped_tensor, labels, 'output/IMM_r.csv')

    # 4. 保存其他元数据
    pd.DataFrame(labels).to_csv('output/train_labels.csv',header=False, index=False)

    pd.DataFrame(cluster_labels).to_csv('output/cluster_labels.csv',header=False, index=False)


multiModel()
