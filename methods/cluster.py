import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import joblib
import os


def Cluster(train_data_path, n_clusters):
    # 数据聚类
    special_cluster = np.empty(0)
    abandon_cluster = []

    # 读取数据
    data = pd.read_csv(train_data_path, header=None)
    X = data.iloc[:, 1:].values

    # 聚类模型
    cluster = KMeans(n_clusters=n_clusters, init='k-means++')
    cluster.fit(X)
    clusters_center = cluster.cluster_centers_
    labels = cluster.fit_predict(X)

    # 保存模型
    joblib.dump(cluster, "cluster.joblib")

    data['Cluster'] = labels
    print("cluster_id", "instances", "error_num")

    # 初始化变量
    valid_clusters = 0
    all_centers = []
    cluster_info = []

    # 确保临时目录存在
    os.makedirs("temp", exist_ok=True)

    # 处理每个簇
    for cluster_id in range(n_clusters):
        cluster_data = data[data['Cluster'] == cluster_id]
        error_num = (cluster_data.iloc[:, 0] == 1.0).sum()  # 统计异常样本数

        # 记录簇信息
        print(cluster_id, len(cluster_data), error_num)

        # 处理有效簇(至少2个样本)
        if len(cluster_data) > 1:
            # 保存簇数据
            cluster_file_path = f"temp/cluster_{cluster_id}.csv"
            cluster_data.to_csv(cluster_file_path, index=False, header=False)

            # 获取正常样本的中心(均值)
            normal_samples = cluster_data[cluster_data.iloc[:, 0] == 0]
            if len(normal_samples) > 0:
                center = normal_samples.iloc[:, 1:-1].mean(axis=0).values
            else:
                center = cluster_data.iloc[:, 1:-1].mean(axis=0).values

            all_centers.append(center)
            valid_clusters += 1

            # 标记特殊簇
            if error_num == 0:
                special_cluster = np.append(special_cluster, 1)  # 全正常簇
            elif error_num == len(cluster_data):
                special_cluster = np.append(special_cluster, -1)  # 全异常簇
                abandon_cluster.append(cluster_id)  # 记录全异常簇
            else:
                special_cluster = np.append(special_cluster, 0)  # 混合簇
        else:
            special_cluster = np.append(special_cluster, -2)  # 标记为无效簇

    # 打乱每个有效簇中的数据顺序
    for cluster_id in range(n_clusters):
        cluster_file_path = f"temp/cluster_{cluster_id}.csv"
        if os.path.exists(cluster_file_path):
            data_cluster = pd.read_csv(cluster_file_path, header=None)
            # 先按标签排序(异常在前)，再打乱顺序
            data_sorted = data_cluster.sort_values(by=data_cluster.columns[0], ascending=False, ignore_index=True)
            shuffled_data = data_sorted.sample(frac=1, random_state=42)
            shuffled_data.to_csv(cluster_file_path, index=False, header=False)

    # 转换中心点数组
    all_centers = np.array(all_centers)

    return all_centers, valid_clusters, len(abandon_cluster), special_cluster