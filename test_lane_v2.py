"""
/*  A script of lane fit version 2 (with twice DBSCAN)
    *
    * Author： li qi
    * Date: 2024-9-9
    * Email: liqi0037@e.ntu.edu.sg
    *
*/
"""
import sys
import rospy
import tf
import numpy as np
import matplotlib.pyplot as plt
from grid_map_msgs.msg import GridMap
from std_msgs.msg import Float32MultiArray
from nav_msgs.msg import Odometry
from scipy.optimize import curve_fit
from sklearn.cluster import DBSCAN  # 导入DBSCAN
import threading
import math
from scipy.optimize import minimize
# 定义高度阈值
HEIGHT_THRESHOLD = 0.05# 根据实际需求调整
# 定义前方范围宽度
WIDTH = 0.14
# 定义采样间隔
INTERVAL = np.radians(2.1)
# 定义距离阈值
DISTANCE_THRESHOLD = 0.5
# DBSCAN参数
EPSILON = 0.03  # 邻域半径
MIN_SAMPLES = 5  # 最小样本数
r1 = 7.40  # 内半径
r_center = 7.75 # 中心线圆半径
r2 = 8.10  # 外半径
EPSILON_X = 0.05  # x方向的邻域半径
EPSILON_Y = 0.07  # y方向的邻域半径
MIN_SAMPLES_SEC = 3   # 最小样本数
# 用于存储数据的全局变量
centers = []
center_points = []
filtered_points = np.array([])  # 初始化为空的 NumPy 数组
curve_params = None
current_yaw = 0.0
world_point = []
yaw_received = False
# 互斥锁以保证线程安全
lock = threading.Lock()
yaw_lock = threading.Lock()
last_target = 0.0
#初始化滤波值设置滤波系数
filtered_y_fit = 0.0
filtered_slope = 0.0
alpha = 0.1

def transform_to_body_frame(x, y, robot_x, robot_y, yaw):
    """
    将栅格位置从世界坐标系转换为机身坐标系
    """
    # 先计算相对于机器人的相对位置
    x_rel = x - robot_x
    y_rel = y - robot_y
    
    # 然后进行坐标变换
    x_body = x_rel * np.cos(yaw) + y_rel * np.sin(yaw)
    y_body = -x_rel * np.sin(yaw) + y_rel * np.cos(yaw)
    
    return x_body, y_body

def transform_to_world_frame(x_body, y_body, robot_x, robot_y, yaw):
    """
    将机身坐标系中的点 (x_body, y_body) 转换到世界坐标系，考虑机器人在世界坐标系中的位置 (robot_x, robot_y) 和朝向 yaw。
    """
    # 先进行坐标变换
    x_world = x_body * np.cos(yaw) - y_body * np.sin(yaw)
    y_world = x_body * np.sin(yaw) + y_body * np.cos(yaw)
    
    # 然后加上机器人在世界坐标系中的位置
    x_world += robot_x
    y_world += robot_y
    
    return x_world, y_world

def grid_map_callback(msg):
    global center_points, filtered_points, curve_params, filtered_heights, yaw_received, centers, world_point, last_target

    try:
        # 找到'elevation'层的索引
        elevation_index = msg.layers.index('elevation')
    except ValueError:
        rospy.logwarn("没有找到 'elevation' 层")
        return

    # 获取'elevation'层的数据
    elevation_data = msg.data[elevation_index]

    # 将数据转换为numpy数组
    rows = elevation_data.layout.dim[0].size
    cols = elevation_data.layout.dim[1].size
    # 提取前方0.5米宽范围内高于一定高度的点
    elevation_array = np.array(elevation_data.data).reshape(rows, cols)
    # height_mask = elevation_array > HEIGHT_THRESHOLD
    
    # 获取分辨率和位置
    resolution = msg.info.resolution
    origin_x = msg.info.pose.position.x
    origin_y = msg.info.pose.position.y
    length_x = msg.info.length_x
    length_y = msg.info.length_y

    # 初始化临时变量
    temp_center_points = []
    temp_filtered_points = []
    temp_filtered_heights = []
    # 获取当前的yaw值
    if yaw_received:
        with yaw_lock:
            yaw = current_yaw
    else:
        yaw = 0
    for i in range(rows):
        for j in range(cols):
            # 计算当前点的实际位置
            world_y = (rows-i) * resolution + origin_y - length_y/2
            world_x = (cols-j) * resolution + origin_x - length_x/2
            body_x, body_y = transform_to_body_frame(world_x, world_y, origin_x, origin_y, yaw)
            # cc_x = origin_x - r_center * math.sin(yaw)
            # cc_y = origin_y - r_center * math.cos(yaw)
            # 计算点到圆心的距离
            if body_x < 0.3:
                break
            distance_to_center = np.sqrt((body_x) ** 2 + (body_y - r_center) ** 2)
            # 筛选在圆环范围内且高度超过阈值的点
            if r1 <= distance_to_center <= r2 and elevation_array[i, j] > HEIGHT_THRESHOLD:
                angle_to_center = np.arctan2(world_y - origin_y, world_x - origin_x)

                # 将角度转换到 [0, 2π] 范围内
                angle_to_center = angle_to_center if angle_to_center >= 0 else angle_to_center + 2 * np.pi

                # 计算yaw的范围
                yaw_min = yaw - np.pi / 2
                yaw_max = yaw + np.pi / 2

                # 将yaw的范围转换到 [0, 2π] 范围内
                yaw_min = yaw_min if yaw_min >= 0 else yaw_min + 2 * np.pi
                yaw_max = yaw_max if yaw_max >= 0 else yaw_max + 2 * np.pi
                # 检查点的角度是否在yaw±90°的范围内
                if (yaw_min <= angle_to_center <= yaw_max) or (yaw_min > yaw_max and (angle_to_center <= yaw_max or angle_to_center >= yaw_min)):
                    temp_filtered_points.append((body_x, body_y))
                    temp_filtered_heights.append(elevation_array[i, j]) 

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
        # 过滤掉噪声点
        noise_mask = labels != -1
        filtered_points = np.array(temp_filtered_points)[noise_mask]
        filtered_heights = np.array(temp_filtered_heights)[noise_mask]
        # 对簇中心的 x 坐标进行额外的 DBSCAN 聚类
        x_coords = cluster_center[:, 0].reshape(-1, 1)  # 提取 x 坐标
        x_clustering = DBSCAN(eps=EPSILON_X, min_samples=MIN_SAMPLES_SEC).fit(x_coords)
        x_labels = x_clustering.labels_

        # # 对簇中心的 y 坐标进行额外的 DBSCAN 聚类
        # y_coords = cluster_center[:, 1].reshape(-1, 1)  # 提取 y 坐标
        # y_clustering = DBSCAN(eps=EPSILON_Y, min_samples=MIN_SAMPLES_SEC).fit(y_coords)
        # y_labels = y_clustering.labels_

        # 为每个簇中心点组合三个标签，形成一个三维向量标签
        two_dimensional_labels = np.array([[label, x_label] for label, x_label in zip(labels, x_labels)])
        middle_points = []
        unique_x_labels = set(x_labels)
        for x_label in unique_x_labels:
            if x_label == -1:
                continue
            points_with_same_x_label = cluster_center[x_labels == x_label]
            if len(points_with_same_x_label) != 3:
                continue
            x_data = points_with_same_x_label[:, 0]
            y_data = points_with_same_x_label[:, 1]
            sorted_point = sorted(points_with_same_x_label, key=lambda point:point[1])
            middle_points.append(sorted_point[1])
            # def linear_func(x, m, b):
            #     return m * x + b
            # popt, _ = curve_fit(linear_func, x_data, y_data)
            # m, b =popt
            # x_middle = (x_data.min() + x_data.max()) / 2
            # y_middle = linear_func(x_middle, m, b)
            # middle_points.append((x_middle, y_middle))
    if middle_points:
        middle_points = np.array(middle_points)
        sorted_middle_point = sorted(middle_points, key=lambda point:point[0])
        sorted_middle_point = np.array(sorted_middle_point)
        if len(sorted_middle_point) >= 2:
            if sorted_middle_point[0, 0] <= 0.8:
                slope = (sorted_middle_point[1, 1] - sorted_middle_point[0, 1]) / (sorted_middle_point[1, 0] - sorted_middle_point[0, 0])
                y_at_x_07 = slope * (0.8 - sorted_middle_point[0, 0]) + sorted_middle_point[0 , 1] - 1.0
                last_target = y_at_x_07
            else:
                y_at_x_07 = last_target
        else:
            y_at_x_07 = last_target
        world_point_x, world_point_y = transform_to_world_frame(0.8, y_at_x_07,
                                                origin_x, origin_y, yaw)
        world_point = world_point_x, world_point_y
        # min_x_point = min(middle_points, key=lambda point:point[0])
        # world_point_x, world_point_y = transform_to_world_frame(min_x_point[0], min_x_point[1],
        #                                                         origin_x, origin_y, yaw)
        # world_point = world_point_x, world_point_y
        with lock:
            center_points = middle_points
            centers = cluster_center
    else:
        world_point = transform_to_world_frame(0.6, 0, origin_x, origin_y, yaw)
        print("未找到符合条件的点")

def odom_callback(msg):
    global world_point, current_yaw, yaw_received, body_x, body_y
    position = msg.pose.pose.position
    orientation = msg.pose.pose.orientation

    # 提取四元数转换为欧拉角
    quaternion = (orientation.x, orientation.y, orientation.z, orientation.w)
    euler = tf.transformations.euler_from_quaternion(quaternion)
    with yaw_lock:
        current_yaw = euler[2]  # 获取yaw角度
        yaw_received = True
    # 计算点相对机身坐标系的位置
    world_x, world_y = world_point
    body_x, body_y = transform_to_body_frame(world_x, world_y, position.x, position.y, current_yaw)
    def expect_func(x, r):
        return r - np.sqrt(r**2 - x**2)
        
    y_fit = expect_func(body_x, r_center)
    y_error = body_y - y_fit
    # 创建 Float32MultiArray 消息
    target_msg = Float32MultiArray()
    # print(y_error)
    data_list = []
    data_list.append(y_error)
    error_2 = 0.0
    data_list.append(error_2)
    target_msg.data = data_list
    # # 创建 Float32MultiArray 消息
    # target_msg = Float32MultiArray()
    # target_msg.data = [body_x, body_y]

    # 发布消息
    target_point_pub.publish(target_msg)
    # rospy.loginfo(f"Target point in body frame: x = {body_x}, y = {body_y}")


def plot_data(event):
    global center_points, filtered_points, curve_params, filtered_heights, centers, body_x, body_y, world_point

    with lock:
        if center_points is None or len(center_points) == 0 or filtered_points.size == 0:
            return


        plt.clf()
        plt.grid(True)
        # 绘制过滤后的点
        plt.plot(filtered_points[:, 0], filtered_points[:, 1], 'b+', label='Filtered Points')
        
        # # 在点上方显示高度
        # for i, point in enumerate(filtered_points):
        #     plt.text(point[0], point[1], f'{filtered_heights[i]:.2f}', fontsize=9, ha='center')
        
        # 绘制中心点
        plt.plot(center_points[:, 0], center_points[:, 1], 'ro', label='Center Points')
        # plt.plot(body_x, body_y, "ro", label = 'Target Point')
        # plt.plot(world_point[0], world_point[1], "ro", label = 'Target Point')
        # 绘制簇的中心
        plt.plot(centers[:, 0], centers[:, 1], 'g*', label='Cluster Center Points')
        
        # # 在点上方显示高度
        # for i, point in enumerate(filtered_points):
        #     plt.text(point[0], point[1], f'{filtered_heights[i]:.2f}', fontsize=9, ha='center')
        
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.title('Filtered Points, Center Points and Fitted Curve')
        plt.draw()

def on_key(event):
    """
    处理按键事件，当按下 'q' 键时退出程序
    """
    if event.key == 'q':
        plt.close()
        rospy.signal_shutdown("User exit")
        sys.exit(0)

def main():
    rospy.init_node('grid_map_subscriber')
    
    # 订阅/grid_map主题
    rospy.Subscriber('/elevation_mapping/elevation_map_raw', GridMap, grid_map_callback)
    rospy.Subscriber('/odom', Odometry, odom_callback)
    # 创建发布器
    global target_point_pub
    target_point_pub = rospy.Publisher('/target_point', Float32MultiArray, queue_size=10)    
    # 在主线程中设置定时器以更新图形
    # fig = plt.figure()
    # fig.canvas.mpl_connect('key_press_event', on_key)  # 监听键盘事件
    # timer = fig.canvas.new_timer(interval=1000)  # 每隔1秒更新一次图形
    # timer.add_callback(plot_data, None)
    # timer.start()
    # plt.show(block=True)
    rospy.spin()

if __name__ == '__main__':
    main()
