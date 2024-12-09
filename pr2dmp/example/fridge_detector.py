import time

import rospy
from geometry_msgs.msg import PoseStamped
from posedetection_msgs.msg import ObjectDetection
from rospy import Subscriber

from pr2dmp.utils import RichTrasnform


class PoseQueue:
    def __init__(self, max_size=10):
        self.queue = []
        self.max_size = max_size

    def append(self, pose: PoseStamped):
        if len(self.queue) > self.max_size:
            self.queue.pop(0)
        self.queue.append(pose)

    def get_average_pose(self) -> PoseStamped:
        if len(self.queue) == 0:
            return None

        x = sum([pose.pose.position.x for pose in self.queue]) / len(self.queue)
        y = sum([pose.pose.position.y for pose in self.queue]) / len(self.queue)
        z = sum([pose.pose.position.z for pose in self.queue]) / len(self.queue)

        pose = PoseStamped()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        return pose


class FridgeDetector:
    def __init__(self):
        topic_name = "/kinect_head/rgb/ObjectDetection"
        self.sub = Subscriber(topic_name, ObjectDetection, self.callback_object_detection)
        self.pose = None
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
        return RichTrasnform.from_ros_pose_stamped(self.pose, "fridge")

    def reset(self):
        self.pose = None

    def callback_object_detection(self, msg: ObjectDetection):
        if self.pose is None:
            transform = msg.objects[0].pose
            pose = PoseStamped(header=msg.header, pose=transform)
            self.pose = pose
            rospy.loginfo(f"object detected")


if __name__ == "__main__":
    topic_name = "/kinect_head/rgb/ObjectDetection"
    rospy.init_node("detector")
    FridgeDetector()
    rospy.spin()
