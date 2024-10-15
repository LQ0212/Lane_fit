#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PointStamped
from grid_map_msgs.msg import GridMap
#import grid_map_ros
import numpy as np

class HeightEstimator:
    def __init__(self):
        self.grid_map = None
        
        rospy.Subscriber('/elevation_mapping/elevation_map_raw', GridMap, self.grid_map_callback)
        rospy.Subscriber('/clicked_point', PointStamped, self.clicked_point_callback)

    def grid_map_callback(self, msg):
        self.grid_map = msg

    def clicked_point_callback(self, msg):
        if self.grid_map is None:
            rospy.logwarn("No grid map data available.")
            return

        point = msg.point
        height = self.get_height_at_point(point)
        if height is not None:
            rospy.loginfo(f"Height at clicked point ({point.x}, {point.y}): {height}")
        else:
            rospy.logwarn("Clicked point is out of grid map bounds or no elevation data available.")

    def get_height_at_point(self, point):
        x = point.x
        y = point.y

        resolution = self.grid_map.info.resolution
        half_size_x = self.grid_map.info.length_x / 2.0
        half_size_y = self.grid_map.info.length_y / 2.0

        grid_x = int((x + half_size_x) / resolution)
        grid_y = int((y + half_size_y) / resolution)

        if 0 <= grid_x < self.grid_map.info.length_x/resolution and 0 <= grid_y < self.grid_map.info.length_y/resolution:
            index = grid_y * int(self.grid_map.info.length_x/resolution) + (200-grid_x)
            layer_names = self.grid_map.layers

            if 'elevation' in layer_names:
                elevation_index = layer_names.index('elevation')
                elevation_data = self.grid_map.data[elevation_index]
                height = elevation_data.data[index]

                if np.isnan(height):
                    return None
                return height
        return None

if __name__ == '__main__':
    rospy.init_node('height_estimator')
    HeightEstimator()
    rospy.spin()
