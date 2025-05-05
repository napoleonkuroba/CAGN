import os
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split


from sklearn.cluster import MiniBatchKMeans

def NormalOrder(device, cluster_id, dimension, cluster_center, reorder, nodes):
    # 读取簇数据
    cluster_file_path = f"temp/cluster_{cluster_id}.csv"
    cluster_data = pd.read_csv(cluster_file_path, header=None)

    # 转换为tensor
    cluster_data_tensor = torch.tensor(
        cluster_data.iloc[:, 0:].values,
        dtype=torch.float32
    ).to(device)

    # 提取标签和特征
    labels = cluster_data_tensor[:, 0]
    features = cluster_data_tensor[:, 1:-1]

    # 子聚类处理
    safe_n_clusters = min(nodes, features.shape[0])
    if safe_n_clusters < 1:
        return torch.empty(0), torch.empty(0), torch.empty(0), *([torch.empty(0)] * 4)

    kmeans = MiniBatchKMeans(n_clusters=safe_n_clusters, random_state=42)
    kmeans.fit(features.cpu().numpy())
    sub_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(device)

    # 构建图数据
    u_in = torch.empty(0).to(device)
    train_value_set = torch.empty(0).to(device)
    train_distance_set = torch.empty(0).to(device)
    train_order_set = torch.empty(0).to(device)
    train_label_set = torch.empty(0).to(device)

    for j in range(len(cluster_data_tensor)):
        cur_data = cluster_data_tensor[j, 1:-1].reshape(-1, dimension)
        input_sample = torch.cat([cur_data, sub_centers], dim=0)
        input_data = input_sample.reshape(-1, dimension)

        # 计算距离矩阵
        input_instances = input_data.cpu().numpy()
        distances = cdist(input_instances, input_instances, metric='euclidean')
        weight = distance_to_weight(distances)
        adj = torch.tensor(weight, dtype=torch.float32).to(device)

        # 构建数据集
        u_in = torch.cat((u_in, cur_data), 0)
        train_value_set = torch.cat((train_value_set, input_data), 0)
        train_distance_set = torch.cat((train_distance_set, adj), 0)
        train_order_set = torch.cat((train_order_set, torch.tensor([cluster_id]).to(device)), 0)
        train_label_set = torch.cat((train_label_set, labels[j].unsqueeze(0)), dim=0)

    # 数据重塑和打乱
    u_in = u_in.reshape(-1, dimension).float()
    train_value_set = train_value_set.reshape(-1, nodes + 1, dimension).float()
    train_distance_set = train_distance_set.reshape(-1, nodes + 1, nodes + 1).float()
    train_label_set = torch.squeeze(train_label_set)

    if reorder:
        shuffled_indices = torch.randperm(train_label_set.size(0))
        u_in = u_in[shuffled_indices]
        train_value_set = train_value_set[shuffled_indices]
        train_distance_set = train_distance_set[shuffled_indices]
        train_label_set = train_label_set[shuffled_indices]
        train_order_set = train_order_set[shuffled_indices]

    # 划分训练集
    input_train_set, _, input_distance_set, _, input_order_set, _ = train_test_split(
        train_value_set, train_distance_set, train_order_set,
        test_size=0.4, random_state=40
    )

    return (
        input_train_set, input_distance_set, input_order_set,
        train_value_set, train_order_set, train_distance_set,
        train_label_set, u_in, nodes + 1
    )
def ClusterOrder(device, n_clusters, dimension, cluster_center, reorder,nodes):
    u_in = torch.empty(0).to(device)
    train_order_set = torch.empty(0).to(device)
    train_value_set = torch.empty(0).to(device)
    train_distance_set = torch.empty(0).to(device)
    train_label_set = torch.empty(0).to(device)
    input_train_set = torch.empty(0).to(device)
    input_distance_set = torch.empty(0).to(device)
    input_order_set = torch.empty(0).to(device)

    for id in range(n_clusters):  # 遍历所有类
        cluster_file_path = f"temp/cluster_{id}.csv"
        cluster_data = pd.read_csv(cluster_file_path, header=None)
        # 确保数据已经转换为PyTorch Tensor，并移到设备上
        cluster_data = torch.tensor(cluster_data.iloc[:, 0:].values, dtype=torch.float32).to(device)
        labels = cluster_data[:, 0]

        for j in range(len(cluster_data)):  # 遍历当前簇
            input_sample = None
            cur_data = cluster_data[j, 1:-1].reshape(-1, dimension)
            for index in range(n_clusters):
                # 当前数据与遍历簇相同
                if id == index:
                    if input_sample is None:
                        input_sample = cur_data
                    else:
                        input_sample = torch.cat([input_sample, cur_data], dim=0)
                else:
                    center_data = cluster_center[index, :].reshape(-1, dimension)
                    if input_sample is None:
                        input_sample = center_data
                    else:
                        input_sample = torch.cat([input_sample, center_data], dim=0)

            input_data = input_sample.reshape(-1, dimension)
            input_instances = input_data.cpu().detach().numpy()
            distances = cdist(input_instances, input_instances, metric='euclidean')
            weight = distance_to_weight(distances)
            adj = torch.tensor(weight, dtype=torch.float32).to(device)
            # 加入数据集
            u_in = torch.cat((u_in, cur_data), 0)
            train_value_set = torch.cat((train_value_set, input_data), 0)
            train_distance_set = torch.cat((train_distance_set, adj), 0)
            torch_id = torch.tensor(id).unsqueeze(0).to(device)
            train_order_set = torch.cat((train_order_set, torch_id), 0)
            train_label_set = torch.cat((train_label_set, labels[j].unsqueeze(0)), dim=0)
            # 异常数据不加入
            if labels[j] == 0 or labels[j] == 1:
                input_train_set = torch.cat((input_train_set, input_data), 0)
                input_distance_set = torch.cat((input_distance_set, adj), 0)
                input_order_set = torch.cat((input_order_set, torch_id), 0)
    # 数据集形状变化
    u_in = u_in.reshape(-1, dimension).float()
    input_train_set = input_train_set.reshape(-1, nodes, dimension).float()
    input_distance_set = input_distance_set.reshape(-1, nodes, nodes).float()
    input_order_set= input_order_set.reshape(-1, 1)
    train_value_set = train_value_set.reshape(-1, nodes, dimension).float()
    train_order_set = train_order_set.reshape(-1, 1)
    train_distance_set = train_distance_set.reshape(-1, nodes, nodes).float()
    train_label_set = torch.squeeze(train_label_set)
    if reorder:
        shuffled_indices = torch.randperm(train_label_set.size(0))
        u_in = u_in[shuffled_indices]
        train_value_set = train_value_set[shuffled_indices]
        train_distance_set = train_distance_set[shuffled_indices]
        train_label_set = train_label_set[shuffled_indices]
        train_order_set = train_order_set[shuffled_indices]
    input_train_set, _, input_distance_set, _, input_order_set, _ = train_test_split(input_train_set,input_distance_set,input_order_set, test_size=0.4,
                                                                             random_state=40)
    return input_train_set, input_distance_set,input_order_set, train_value_set, train_order_set, train_distance_set, train_label_set, u_in, nodes


def distance_to_weight(distance_matrix):
    size = distance_matrix.shape[0]
    weight_matrix = np.zeros((size, size), dtype=float)
    # 展平矩阵
    matrix = distance_matrix.ravel()
    matrix_nonzero = matrix[matrix != 0]
    order = np.sort(matrix_nonzero.flatten())[::-1]
    max_value=max(order)

    for i in range(size):
        for j in range(size):
            if distance_matrix[i, j] != 0:  # 仅处理非零元素
                # 从order中找到其序列，并用倒序列交换
                value = distance_matrix[i, j]
                for order_index in range(len(order)):
                    if value == order[order_index]:
                        weight_matrix[i, j] = (order[len(order) - order_index - 1]/max_value)*0.9
                        break
            else:
                weight_matrix[i, j] = 1.0

    return weight_matrix
