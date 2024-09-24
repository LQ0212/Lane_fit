import rospy
import tf
import numpy as np
import matplotlib.pyplot as plt
from grid_map_msgs.msg import GridMap
from scipy.optimize import curve_fit
from sklearn.cluster import DBSCAN  # 导入DBSCAN
import threading
import math

# 定义高度阈值
HEIGHT_THRESHOLD = 0.02  # 根据实际需求调整

# 定义采样间隔
INTERVAL = 0.02

# DBSCAN参数
EPSILON = 0.02  # 邻域半径
MIN_SAMPLES = 3  # 最小样本数

# 圆心位置和圆环的内外半径
center_x = 0.0  # 圆心的X坐标
center_y = -7.55  # 圆心的Y坐标
r1 = 7.0  # 内半径
r2 = 8.0  # 外半径

# 用于存储数据的全局变量
center_points = []
filtered_points = np.array([])  # 初始化为空的 NumPy 数组
curve_params = None

# 互斥锁以保证线程安全
lock = threading.Lock()

def grid_map_callback(msg):
    global center_points, filtered_points, curve_params

    try:
        # 找到'elevation'层的索引
        elevation_index = msg.layers.index('elevation')
    except ValueError:
        rospy.logwarn("没有找到 'elevation' 层")
        return

    # 获取'elevation'层的数据
    elevation_data = msg.data[elevation_index]

    # 将数据转换为numpy数组，并根据正确的行列顺序重塑
    cols = elevation_data.layout.dim[0].size
    rows = elevation_data.layout.dim[1].size
    elevation_array = np.array(elevation_data.data).reshape(cols, rows).T  # 注意这里是cols, rows并转置

    # 获取分辨率和位置
    resolution = msg.info.resolution
    origin_x = msg.info.pose.position.x
    origin_y = msg.info.pose.position.y
    length_x = msg.info.length_x
    length_y = msg.info.length_y
    # 初始化临时变量
    temp_filtered_points = []
    temp_filtered_heights = []

    for i in range(rows):
        for j in range(cols):
            # 计算当前点的实际位置
            x = (j - cols // 2) * resolution + origin_x
            y = (i - rows // 2) * resolution + origin_y

            # 计算点到圆心的距离
            distance_to_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

            # 筛选在圆环范围内且高度超过阈值的点
            if r1 <= distance_to_center <= r2 and elevation_array[i, j] > HEIGHT_THRESHOLD:
                temp_filtered_points.append((x, y))
                temp_filtered_heights.append(elevation_array[i, j])

    if temp_filtered_points:
        # 对所有符合条件的点进行密度聚类
        clustering = DBSCAN(eps=EPSILON, min_samples=MIN_SAMPLES).fit(temp_filtered_points)
        labels = clustering.labels_

        # 过滤掉噪声点
        noise_mask = labels != -1
        filtered_points = np.array(temp_filtered_points)[noise_mask]
        filtered_heights = np.array(temp_filtered_heights)[noise_mask]

        # 计算每个距离的中心点
        temp_center_points = []
        for i in range(int(r1 / INTERVAL), int(r2 / INTERVAL)):
            distance = i * INTERVAL
            points_at_distance = filtered_points[(filtered_points[:, 0] >= distance - INTERVAL / 2) &
                                                (filtered_points[:, 0] < distance + INTERVAL / 2)]
            if points_at_distance.size > 0:
                center_x = np.mean(points_at_distance[:, 0])
                center_y = np.mean(points_at_distance[:, 1])
                temp_center_points.append((center_x, center_y))

        if temp_center_points:
            # 二次曲线拟合
            temp_center_points = np.array(temp_center_points)
            x_data = temp_center_points[:, 0]
            y_data = temp_center_points[:, 1]

            def cubic_func(x, a, b, c):
                return a * x ** 2 + b * x + c

            popt, _ = curve_fit(cubic_func, x_data, y_data)

            # 使用互斥锁更新全局变量
            with lock:
                center_points = temp_center_points
                curve_params = popt
    else:
        print("未找到符合条件的点")

def plot_data(event):
    global center_points, filtered_points, curve_params, filtered_heights

    with lock:
        if center_points is None or len(center_points) == 0 or filtered_points.size == 0 or curve_params is None:
            return

        plt.clf()

        # 绘制过滤后的点
        plt.scatter(filtered_points[:, 0], filtered_points[:, 1], c='blue', label='Filtered Points')

        # 在点上方显示高度
        for i, point in enumerate(filtered_points):
            plt.text(point[0], point[1], f'{filtered_heights[i]:.2f}', fontsize=9, ha='center')

        # 绘制中心点
        plt.scatter(center_points[:, 0], center_points[:, 1], c='red', label='Center Points')

        # 绘制拟合曲线
        x_data = center_points[:, 0]
        a, b, c = curve_params
        x_fit = np.linspace(min(x_data), max(x_data), 100)
        y_fit = a * x_fit ** 2 + b * x_fit + c
        plt.plot(x_fit, y_fit, c='green', label='Fitted Curve')

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.title('Filtered Points, Center Points and Fitted Curve')
        plt.draw()

def main():
    rospy.init_node('grid_map_subscriber')

    # 订阅/grid_map主题
    rospy.Subscriber('/elevation_mapping/elevation_map_raw', GridMap, grid_map_callback)

    # 在主线程中设置定时器以更新图形
    fig = plt.figure()
    timer = fig.canvas.new_timer(interval=1000)  # 每隔1秒更新一次图形
    timer.add_callback(plot_data, None)
    timer.start()

    plt.show(block=True)
    rospy.spin()

if __name__ == '__main__':
    main()
