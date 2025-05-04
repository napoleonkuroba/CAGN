import pandas as pd
import os


class ClusterLoader:
    def __init__(self, cluster_dir="temp"):
        self.cluster_dir = cluster_dir
        self.valid_clusters = self._find_valid_clusters()

    def _find_valid_clusters(self):
        """预扫描有效簇并建立索引"""
        valid_ids = []
        i = 0
        while True:
            path = os.path.join(self.cluster_dir, f"cluster_{i}.csv")
            if not os.path.exists(path):
                break

            df = pd.read_csv(path, header=None)
            if self._has_anomaly(df):
                valid_ids.append(i)

            i += 1

        return valid_ids

    def _has_anomaly(self, df):
        """检查是否存在异常样本"""
        return (df.iloc[:, 0] == 1).sum() > 0

    def get_cluster(self, index):
        """
        获取指定索引的有效簇数据
        返回: (normal_data, all_data) 或 None
        """
        if index >= len(self.valid_clusters):
            return None

        original_id = self.valid_clusters[index]
        path = os.path.join(self.cluster_dir, f"cluster_{original_id}.csv")
        df = pd.read_csv(path, header=None)

        # 分离正常数据和全部数据
        normal_data = df[df.iloc[:, 0] == 0].iloc[:, 0:].values
        all_data = df.iloc[:, 0:].values

        return normal_data, all_data

    @property
    def cluster_count(self):
        """获取有效簇数量"""
        return len(self.valid_clusters)

    def save_clusters_to_csv(loader, output_dir="saved_clusters", prefix="cluster"):
        """
        将有效簇保存为CSV文件
        参数:
            loader: ClusterLoader实例
            output_dir: 输出目录 (默认"saved_clusters")
            prefix: 文件名前缀 (默认"cluster")
        """
        os.makedirs(output_dir, exist_ok=True)

        for i in range(loader.cluster_count):
            cluster_data = loader.get_cluster(i)
            if cluster_data is None:
                continue

            normal, all_data = cluster_data

            # 保存正常样本
            pd.DataFrame(normal).to_csv(
                os.path.join(output_dir, f"{prefix}_{i}_normal.csv"),
                header=False,
                index=False
            )

            # 保存全部样本
            pd.DataFrame(all_data).to_csv(
                os.path.join(output_dir, f"{prefix}_{i}_all.csv"),
                header=False,
                index=False
            )


# 使用示例
if __name__ == "__main__":
    loader = ClusterLoader()
    loader.save_clusters_to_csv()

    print(f"找到 {loader.cluster_count} 个有效簇")

    if loader.cluster_count > 0:
        for i in range(loader.cluster_count):
            print(f"获取第 {i} 个有效簇...")
            normal, all_data = loader.get_cluster(i)
            print(f"- 正常样本数：{len(normal)}")
            print(f"- 总样本数：{len(all_data)}")