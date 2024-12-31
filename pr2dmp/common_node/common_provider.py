import pickle
import time
from pathlib import Path
from typing import Generic, Optional, Tuple, TypeVar

import numpy as np
import rospy
from cv_bridge import CvBridge
from ffridge.ros_numpy.point_cloud2 import pointcloud2_to_xyz_array
from sensor_msgs.msg import Image, PointCloud2

T = TypeVar("T", np.ndarray, np.ndarray)


class DataProvider(Generic[T]):
    data: Optional[T]
    sec: Optional[float]
    stop_flag: bool
    is_dummy: bool

    def __init__(self, topic: str, msg_type, callback_queue_size: int = 1, is_dummy: bool = False):
        self.data = None
        self.sec = None
        self.stop_flag = True
        self.is_dummy = is_dummy
        if is_dummy:
            with open(self.cache_path(), "rb") as f:
                self.data, self.sec = pickle.load(f)
        else:
            rospy.Subscriber(topic, msg_type, self.callback, queue_size=callback_queue_size)

    def reset(self):
        if not self.is_dummy:
            self.data = None
            self.sec = None

    def callback(self, msg):
        if self.stop_flag:
            return
        self._process_message(msg)

    def _process_message(self, msg):
        raise NotImplementedError

    def get(self, timeout: float = 10) -> Optional[Tuple[T, float]]:
        ts = time.time()
        while self.data is None:
            if time.time() - ts > timeout:
                return None
        return self.data, self.sec

    def stop(self):
        self.stop_flag = True

    def start(self):
        self.stop_flag = False

    @classmethod
    def cache_directory(cls):
        p = Path("~/.cache/ffridge").expanduser()
        p.mkdir(parents=True, exist_ok=True)
        return p

    @classmethod
    def cache_path(cls):
        return cls.cache_directory() / f"{cls.__name__}.pkl"

    def save_debug_data(self):
        file_path = self.cache_path()
        rospy.loginfo(f"save debug data to {file_path}")
        with open(file_path, "wb") as f:
            pickle.dump((self.data, self.sec), f)


class PointCloudProvider(DataProvider[np.ndarray]):
    def __init__(self, is_dummy: bool = False):
        super().__init__(
            topic="/local/tf_transform/output", msg_type=PointCloud2, is_dummy=is_dummy
        )

    def _process_message(self, msg: PointCloud2):
        points_tmp = pointcloud2_to_xyz_array(msg, remove_nans=False)
        self.data = points_tmp.reshape(-1, 3)
        self.sec = msg.header.stamp.to_sec()


class FridgeSegmProvider(DataProvider[np.ndarray]):
    def __init__(self, is_dummy: bool = False):
        super().__init__(
            topic="/docker/fridge/detic_segmentor/segmentation", msg_type=Image, is_dummy=is_dummy
        )

    def _process_message(self, img: Image):
        bridge = CvBridge()
        seg_image = bridge.imgmsg_to_cv2(img, desired_encoding="passthrough")
        n_label = np.max(seg_image)

        if n_label != 1:
            rospy.logwarn(f"fridge mask has {n_label} labels")
            return

        self.data = seg_image == 1
        self.sec = img.header.stamp.to_sec()
