import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import joblib
def Cluster(train_data_path,n_clusters):
    # 数据聚类
    special_cluster = np.empty(0)
    abandon_cluster = []
    file_path = train_data_path
    data = pd.read_csv(file_path, header=None)
    X = data.iloc[:, 1:].values
    cluster = KMeans(n_clusters=n_clusters,init='k-means++')
    # cluster = KMedoids(n_clusters=n_clusters, random_state=42)
    cluster.fit(X)
    clusters_center = cluster.cluster_centers_
    print(clusters_center.shape)
    labels = cluster.fit_predict(X)
    model_save_path="cluster.joblib"
    joblib.dump(cluster, model_save_path)
    data['Cluster'] = labels
    print("cluster_id", "instances", "error_num")
    low_num = 0
    centers = np.empty(0)
    k=0
    for cluster_id in range(n_clusters):
        cluster_data = data[data['Cluster'] == cluster_id]
        error_num = (cluster_data.iloc[:, 0] == 1.0).sum()
        if len(cluster_data)>1:
            k=k+1
            print(cluster_id, len(cluster_data), error_num)
            cluster_file_path = f"temp/cluster_{cluster_id}.csv"
            cluster_data.to_csv(cluster_file_path, index=False, header=False)
            centers = cluster_data[cluster_data.iloc[:, 0] == 0]
            centers = centers.drop(centers.columns[0], axis=1)
            centers = centers.drop(centers.columns[-1], axis=1).values
        if error_num == 0:
            special_cluster = np.append(special_cluster, 1)
            low_num = low_num +1
        elif error_num == len(cluster_data):
            special_cluster = np.append(special_cluster, -1)
        else:
            special_cluster = np.append(special_cluster, 0)
    for cluster_id in range(k):
        cluster_file_path = f"temp/cluster_{cluster_id}.csv"
        data_cluster = pd.read_csv(cluster_file_path, header=None)
        data_sorted = data_cluster.sort_values(by=data_cluster.columns[0], ascending=False, ignore_index=True)#从大到小排序
        shuffled_data = data_sorted.sample(frac=1, random_state=42)#将簇打乱顺序
        shuffled_data.to_csv(cluster_file_path, index=False, header=False)
    print(clusters_center.shape)
    return centers,k,k,special_cluster