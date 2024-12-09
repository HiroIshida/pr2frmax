import time

import rospy
import tf2_geometry_msgs
import tf2_ros
from geometry_msgs.msg import PoseStamped
from posedetection_msgs.msg import ObjectDetection
from rospy import Subscriber

from pr2dmp.utils import RichTrasnform


class FridgeDetector:
    def __init__(self):
        topic_name = "/kinect_head/rgb/ObjectDetection"
        self.sub = Subscriber(topic_name, ObjectDetection, self.callback_object_detection)
        self.pose = None
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        rospy.loginfo("FridgeDetector initialized")

    def get_current_transform(self) -> RichTrasnform:
        self.reset()
        timeout = 5
        ts = time.time()
        while self.pose is None:
            rospy.sleep(0.01)
            elapsed_time = time.time() - ts
            if elapsed_time > timeout:
                assert False
        transform = self.tf_buffer.lookup_transform(
            "base_footprint", self.pose.header.frame_id, rospy.Time(0), rospy.Duration(1.0)
        )
        transformed_pose = tf2_geometry_msgs.do_transform_pose(self.pose, transform)
        return RichTrasnform.from_ros_pose_stamped(transformed_pose, "fridge")

    def reset(self):
        self.pose = None

    def callback_object_detection(self, msg: ObjectDetection):
        if self.pose is None:
            transform = msg.objects[0].pose
            pose = PoseStamped(header=msg.header, pose=transform)
            self.pose = pose
            rospy.loginfo("object detected")


if __name__ == "__main__":
    topic_name = "/kinect_head/rgb/ObjectDetection"
    rospy.init_node("detector")
    FridgeDetector()
    rospy.spin()
