#!/usr/bin/env python
import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import PointStamped
from rospy import Publisher, Subscriber
from sensor_msgs.msg import PointCloud2
from sklearn.cluster import DBSCAN

if __name__ == "__main__":
    pub = Publisher("/yellow_tape/center", PointStamped, queue_size=10)

    def callback(msg: PointCloud2):
        gen = pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z"))
        xyz = np.array(list(gen))
        if len(xyz) == 0:
            rospy.logwarn("No yellow point cloud found")
            return

        dbscan = DBSCAN(eps=0.005, min_samples=3)
        clusters = dbscan.fit_predict(xyz)
        if len(clusters) == 0:
            rospy.logwarn("No yellow point cloud found")
            return
        rospy.logdebug(f"Number of clusters: {np.max(clusters) + 1}")
        n_label = np.max(clusters) + 1
        cluster_sizes = [np.sum(clusters == i) for i in range(n_label)]
        largest_cluster_idx = np.argmax(cluster_sizes)
        points_clustered = xyz[clusters == largest_cluster_idx]

        mean = np.mean(points_clustered, axis=0)
        point = PointStamped()
        point.header = msg.header
        point.point.x = mean[0]
        point.point.y = mean[1]
        point.point.z = mean[2]
        pub.publish(point)

    rospy.init_node("dummy")
    topic_name = "/yellow_tape/tf_transform_cloud/output"
    sub = Subscriber(topic_name, PointCloud2, callback)
    rospy.spin()
