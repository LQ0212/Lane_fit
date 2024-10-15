from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler  # 进行数据标准化处理

# DBSCAN参数，x方向和y方向可能需要不同的聚类参数
EPSILON_X = 0.05  # x方向的邻域半径
EPSILON_Y = 0.07  # y方向的邻域半径

def grid_map_callback(msg):
    global center_points, filtered_points, curve_params, filtered_heights, yaw_received, centers

    # 你的初始聚类代码...
    
    if temp_filtered_points:
        # 对所有符合条件的点进行密度聚类
        clustering = DBSCAN(eps=EPSILON, min_samples=MIN_SAMPLES).fit(temp_filtered_points)
        labels = clustering.labels_
        
        # 遍历所有的簇标签（忽略噪声点 -1）
        cluster_center = []
        unique_labels = set(labels)
        for label in unique_labels:
            if label == -1:
                continue  # 跳过噪声点

            # 提取属于当前簇的所有点
            points_in_cluster = np.array(temp_filtered_points)[labels == label]

            # 计算簇的质心
            centroid = points_in_cluster.mean(axis=0)
            
            # 将质心加入簇中心列表
            cluster_center.append(centroid)
        cluster_center = np.array(cluster_center)

        # 对簇中心的 x 坐标进行额外的 DBSCAN 聚类
        x_coords = cluster_center[:, 0].reshape(-1, 1)  # 提取 x 坐标
        x_clustering = DBSCAN(eps=EPSILON_X, min_samples=MIN_SAMPLES).fit(x_coords)
        x_labels = x_clustering.labels_

        # 对簇中心的 y 坐标进行额外的 DBSCAN 聚类
        y_coords = cluster_center[:, 1].reshape(-1, 1)  # 提取 y 坐标
        y_clustering = DBSCAN(eps=EPSILON_Y, min_samples=MIN_SAMPLES).fit(y_coords)
        y_labels = y_clustering.labels_

        # 为每个簇中心点组合三个标签，形成一个三维向量标签
        three_dimensional_labels = np.array([labels, x_labels, y_labels]).T
        
        # 打印或存储标签
        print("三维标签：", three_dimensional_labels)

        # 使用三维标签进一步处理...

        # 其余代码...

