# /*  A script of lane fit
#     *
#     * Author： li qi
#     * Date: 2024-8-27
#     * Email: liqi0037@e.ntu.edu.sg
#     *
# */
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
import sys
# 定义高度阈值
HEIGHT_THRESHOLD = 0.04  # 根据实际需求调整
# 定义前方范围宽度
WIDTH = 0.14
# 定义采样间隔
INTERVAL = np.radians(2.1)
# 定义距离阈值
DISTANCE_THRESHOLD = 0.5
# DBSCAN参数
EPSILON = 0.06  # 邻域半径
MIN_SAMPLES = 6  # 最小样本数
r1 = 7.25  # 内半径
r_center = 7.75 # 中心线圆半径
r2 = 8.25  # 外半径
body_x = 0
body_y = 0
# 用于存储数据的全局变量
centers = []
center_points = []
filtered_points = np.array([])  # 初始化为空的 NumPy 数组
curve_params = None
current_yaw = 0.0
yaw_received = False
# 互斥锁以保证线程安全
lock = threading.Lock()
yaw_lock = threading.Lock()
world_point = [(0, 0)]
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
    global center_points, filtered_points, curve_params, filtered_heights, yaw_received, centers, world_point

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

        # 找到最近的簇中心点
        angles = np.arctan2(cluster_center[:, 0], abs(cluster_center[:,1] - r_center))
        current_angle = angles.min()
        while current_angle < angles.min() + 2.5 * INTERVAL:
            points_in_sector = cluster_center[(angles >= current_angle - INTERVAL / 2) &
                                            ( angles <= current_angle + INTERVAL / 2)]
            if len(points_in_sector) ==3:
                if len(points_in_sector) % 2 == 1:
                    sorted_points = points_in_sector[np.argsort(points_in_sector[:, 1])]
                    median_point = sorted_points[len(sorted_points) // 2]
                    center_x, center_y = median_point[0], median_point[1]
                else:
                    center_x = np.mean(points_in_sector[:, 0])
                    center_y = np.mean(points_in_sector[:, 1])
                temp_center_points.append((center_x, center_y))
            # 更新角度
            current_angle += INTERVAL
        # 初始化当前距离
        # current_distance = nearest_center[0]
        
        # while current_distance <= 1.5:
        #     points_at_distance = cluster_center[(cluster_center[:, 0] >= current_distance - INTERVAL / 2) &
        #                                         (cluster_center[:, 0] <= current_distance + INTERVAL / 2)]
        #     if points_at_distance.size > 0:
        #         center_x = np.mean(points_at_distance[:, 0])
        #         center_y = np.mean(points_at_distance[:, 1])
        #         temp_center_points.append((center_x, center_y))
        #     current_distance += INTERVAL
    if temp_center_points:
        # 二次曲线拟合
        temp_center_points = np.array(temp_center_points)
        x_data = temp_center_points[:, 0]
        y_data = temp_center_points[:, 1]
        # length = len(temp_center_points)
        # if temp_center_points[0, 0] >= 0.5:
        #     target_point = temp_center_points[0, :]
        # else:
        #     target_point = temp_center_points[1, :]
        # world_point_x, world_point_y = transform_to_world_frame(target_point[0], target_point[1],
        #                                         origin_x, origin_y, yaw)
        if len(temp_center_points) >= 2:
            if temp_center_points[0, 0] <= 0.7:
                slope = (temp_center_points[1, 1] - temp_center_points[0, 1]) / (temp_center_points[1, 0] - temp_center_points[0, 0])
                y_at_x_07 = slope * (0.7 - temp_center_points[0, 0]) + temp_center_points[0 , 1] -0.025
                last_target = y_at_x_07
            else:
                y_at_x_07 = last_target
        else:
            y_at_x_07 = last_target
        world_point_x, world_point_y = transform_to_world_frame(0.7, y_at_x_07,
                                                origin_x, origin_y, yaw)
        world_point = world_point_x, world_point_y

    #     def cubic_func(x, a, b, c):
    #         return a * x**2 + b * x + c
        
    #     popt, _ = curve_fit(cubic_func, x_data, y_data)

    #     # 获取拟合参数
    #     a, b, c = popt
    #     # def plan_func(x, r):
    #     #     return np.sqrt(r**2 - x**2) - r
    #     # 计算横向距离的平均值
    #     y_fit = cubic_func(0, a, b, c)
    #     # filtered_y_fit = alpha * y_fit + (1 - alpha) * filtered_y_fit
    #     # filtered_slope = alpha * b + (1 - alpha) * filtered_slope
        
    #     print("拟合参数: a = {}, b = {}, c = {}".format(a, b, c))
    #     print("横向距离: ", temp_center_points[0,1])
    #     print("切线角度: ", math.atan2(b,1))
    #     # 使用互斥锁更新全局变量
        with lock:
            center_points = temp_center_points
            # curve_params = popt
            centers = cluster_center
    else:
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

    # 创建 Float32MultiArray 消息
    target_msg = Float32MultiArray()
    target_msg.data.append(body_x)
    target_msg.data.append(body_y)  

    # 发布消息
    target_point_pub.publish(target_msg)
    # rospy.loginfo(f"Target point in body frame: x = {body_x}, y = {body_y}")

def plot_data(event):
    global center_points, filtered_points, filtered_heights, centers, world_point, body_x, body_y

    with lock:
        if center_points is None or len(center_points) == 0 or filtered_points.size == 0 is None:
            return


        plt.clf()
        plt.grid(True)
        # 绘制过滤后的点
        plt.plot(filtered_points[:, 0], filtered_points[:, 1], 'b+', label='Filtered Points')
        
        # # 在点上方显示高度
        # for i, point in enumerate(filtered_points):
        #     plt.text(point[0], point[1], f'{filtered_heights[i]:.2f}', fontsize=9, ha='center')
        
        # 绘制中心点
        # plt.scatter(center_points[:, 0], center_points[:, 1], c='green', label='Center Points')
        plt.plot(body_x, body_y, "ro", label = 'Target Point')
        # 绘制簇的中心
        plt.plot(centers[:, 0], centers[:, 1], 'g*', label='Cluster Center Points')
        # # 绘制拟合曲线
        # x_data = center_points[:, 0]
        # a, b, c = curve_params
        # x_fit = np.linspace(min(x_data), max(x_data), 100)
        # y_fit = a * x_fit**2 + b * x_fit + c
        # plt.plot(x_fit, y_fit, c='green', label='Fitted Curve')
        
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
    fig = plt.figure()
    fig.canvas.mpl_connect('key_press_event', on_key)  # 监听键盘事件
    timer = fig.canvas.new_timer(interval=1000)  # 每隔1秒更新一次图形
    timer.add_callback(plot_data, None)
    timer.start()
    plt.show(block=True)
    rospy.spin()

if __name__ == '__main__':
    main()
