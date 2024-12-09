from geometry_msgs.msg import Pose, PoseStamped
from skrobot.coordinates import Coordinates, Transform
from skrobot.coordinates.math import quaternion2matrix


class RichTrasnform(Transform):
    frame_from: str
    frame_to: str

    def __init__(self, translation, rotation, frame_from, frame_to):
        super().__init__(translation, rotation)
        self.frame_from = frame_from
        self.frame_to = frame_to

    @classmethod
    def from_ros_pose_stamped(cls, pose_stamped: PoseStamped, frame_from: str):
        return cls.from_ros_pose(pose_stamped.pose, frame_from, pose_stamped.header.frame_id)

    @classmethod
    def from_ros_pose(cls, pose: Pose, frame_from: str, frame_to: str):
        translation = [pose.position.x, pose.position.y, pose.position.z]
        q = [pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z]
        rotation = quaternion2matrix(q)
        return cls(translation, rotation, frame_from, frame_to)

    @classmethod
    def from_co(cls, co: Coordinates, frame_from: str, frame_to: str):
        return cls(co.worldpos(), co.worldrot(), frame_from, frame_to)

    def inv(self) -> "RichTrasnform":
        tf = super().inverse_transformation()
        return RichTrasnform(tf.translation, tf.rotation, self.frame_to, self.frame_from)

    def inverse_transformation(self):
        raise NotImplementedError("Use inv() method instead")

    def __mul__(self, tf_23) -> "RichTrasnform":
        # check if the frame is correct
        if self.frame_to != tf_23.frame_from:
            raise ValueError(f"Frame mismatch: {self.frame_to} != {tf_23.frame_from}")
        ret = super().__mul__(tf_23)
        return RichTrasnform(ret.translation, ret.rotation, self.frame_from, tf_23.frame_to)
