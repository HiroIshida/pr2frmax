import rospy
from geometry_msgs.msg import PoseStamped
from jsk_recognition_msgs.msg import BoundingBox

from pr2dmp.utils import RichTrasnform


class FridgePoseProvider:
    def __init__(self):
        self.sub = rospy.Subscriber(
            "/local/fridge_pose_estimator/fridge_box", BoundingBox, self.callback
        )
        self.pose = None

    def callback(self, msg: BoundingBox):
        if self.pose is None:
            pose_stamped = PoseStamped()
            pose_stamped.header = msg.header
            pose_stamped.pose = msg.pose
            self.pose = pose_stamped

    def get_transform(self) -> RichTrasnform:
        while self.pose is None:
            rospy.sleep(0.1)
            rospy.loginfo("waiting for fridge pose")
        return RichTrasnform.from_ros_pose_stamped(self.pose, "fridge")

    def reset(self):
        self.pose = None


if __name__ == "__main__":
    rospy.init_node("fridge_pose_provider")
    FridgePoseProvider()
    rospy.spin()
