import time
from typing import List, Optional, Tuple

import numpy as np
import rospy
import tf2_ros
from skrobot.coordinates.math import quaternion2rpy, rpy_matrix

from pr2dmp.utils import RichTrasnform


class PoseQueue:
    queue: List[Tuple[np.ndarray, rospy.Time]]
    window: float = 5.0

    def __init__(self, max_size: int, position_only: bool):
        self.queue = []
        self.max_size = max_size
        self.position_only = position_only  # because rotation is super noisy

    def push(self, pose: np.ndarray, timestamp: rospy.Time):
        if self.position_only:
            assert len(pose) == 3
        if len(self.queue) >= self.max_size:
            self.queue.pop(0)
        self.queue.append((pose, timestamp))

    def get_mean(
        self, timestamp: rospy.Time, outlier_th_sigma: float = 1.0
    ) -> Optional[np.ndarray]:
        if len(self.queue) < self.max_size:
            return None

        oldest_timestamp = self.queue[0][1]
        elapsed_from_oldest = (timestamp - oldest_timestamp).to_sec()
        is_too_old = elapsed_from_oldest > self.window
        if is_too_old:
            rospy.logwarn(f"(get_mean) data is too old. elapsed: {elapsed_from_oldest}")
            return None

        poses = np.array([pose for pose, _ in self.queue])

        # Because, calibration error is not so big, we can remove
        # outlier by simple thresholding.
        if self.position_only:
            is_outlier = np.any(np.abs(poses) > 0.1, axis=1)
        else:
            is_pos_outlier = np.any(np.abs(poses[:, :3]) > 0.1, axis=1)
            is_ypr_outlier = np.any(np.abs(poses[:, 3:]) > np.deg2rad(30), axis=1)
            is_outlier = np.logical_or(is_pos_outlier, is_ypr_outlier)

        rate_of_outlier = np.sum(is_outlier) / len(is_outlier)
        if rate_of_outlier > 0.6:
            rospy.logwarn(f"rate of outlier is too high: {rate_of_outlier}")
            return None

        poses = poses[~is_outlier]
        std = np.std(poses, axis=0)
        if self.position_only:
            is_stable = np.all(std < 0.003)
        else:
            position_std, ypr_std = std[:3], std[3:]
            is_stable = np.all(position_std < 0.003) and np.all(ypr_std < np.deg2rad(3))

        if not is_stable:
            rospy.logwarn(
                f"(get_mean) data is not stable. position_std: {position_std}, ypr_std: {ypr_std}"
            )
            return None

        mean = np.mean(poses, axis=0)
        return mean


class AprilOffsetDetector:
    def __init__(
        self,
        tf_lb: Optional[Tuple[tf2_ros.Buffer, tf2_ros.TransformListener]] = None,
        position_only: bool = False,
        debug: bool = False,
    ):
        if tf_lb is None:
            buffer = tf2_ros.Buffer()
            listener = tf2_ros.TransformListener(buffer)
            self.tf_buffer = buffer
            self.tf_listener = listener
        else:
            self.tf_buffer = tf_lb[0]
            self.tf_listener = tf_lb[1]
        self.tf_listener.unregister
        self.pose_queue = PoseQueue(30, position_only=position_only)
        self.timer = rospy.Timer(rospy.Duration(0.1), self.timer_callback)
        self.position_only = position_only

        if debug:
            self.log_func = rospy.loginfo
        else:
            self.log_func = lambda x: None

    def timer_callback(self, event):
        transform = self.tf_buffer.lookup_transform(
            "apriltag_fk", "apriltag", rospy.Time(0), rospy.Duration(1.0)
        )
        pose = transform.transform
        translation = np.array([pose.translation.x, pose.translation.y, pose.translation.z])
        if self.position_only:
            self.pose_queue.push(translation, transform.header.stamp)
        else:
            quaternion = np.array(
                [
                    pose.rotation.w,
                    pose.rotation.x,
                    pose.rotation.y,
                    pose.rotation.z,
                ]
            )
            ypr = quaternion2rpy(quaternion)[0]
            vec = np.hstack([translation, ypr])
            self.pose_queue.push(vec, transform.header.stamp)

    def get_gripper_offset(self) -> Tuple[np.ndarray, np.ndarray]:
        self.log_func("(get_gripper_offset) waiting for stable data")
        timeout = 5.0
        ts = time.time()
        while True:
            mean = self.pose_queue.get_mean(rospy.Time.now())
            if mean is not None:
                break
            rospy.sleep(0.1)
            if time.time() - ts > timeout:
                rospy.logerr("Timeout")
                raise TimeoutError
        self.log_func(f"(get_gripper_offset) detected stable data. elapsed: {time.time() - ts}")
        self.log_func(f"(get_gripper_offset) mean: {mean}")
        if self.position_only:
            position = mean
            rpy = np.zeros(3)
        else:
            position = mean[:3]
            ypr = mean[3:]
            rpy = ypr[::-1]

        matrix = rpy_matrix(*rpy)
        return RichTrasnform(position, matrix, "apriltag", "apriltag_hat")


if __name__ == "__main__":
    rospy.init_node("gripper_offset_detector")
    detector = AprilOffsetDetector(position_only=False)
    time.sleep(2)
    print("call")
    ts = time.time()
    position, rpy = detector.get_gripper_offset()
    print(f"time: {time.time() - ts}")
    print(position)
