import time
from typing import ClassVar, Optional

import numpy as np
import rospy
from jsk_recognition_msgs.msg import BoundingBox
from plainmp.psdf import BoxSDF, Pose
from scipy.optimize import minimize
from sklearn.cluster import DBSCAN
from skrobot.coordinates import Coordinates
from skrobot.coordinates.math import rpy_matrix
from skrobot.model.primitives import Box

from pr2dmp.common_node.common_provider import FridgeSegmProvider, PointCloudProvider
from pr2dmp.utils import RichTrasnform


class FridgeEnv:
    fridge_size: ClassVar[np.ndarray] = np.array([0.59, 0.54, 1.48])

    def __init__(self):
        self.fridge_model = Box(self.fridge_size)

        left_redzone = Box([0.59, 2.0, 1.48], face_colors=[255, 0, 0, 200])
        left_redzone.translate([0.0, 1.3, 0])
        right_redzone = Box([0.59, 2.0, 1.48], face_colors=[255, 0, 0, 200])
        right_redzone.translate([-0.15, -1.0 - 0.54 * 0.5 - 0.2, 0])
        table_redzone = Box([3.0, 2.0, 1.0], face_colors=[255, 0, 0, 200])
        table_redzone.translate([-2.5, 2.0, 0.0])

        self.fridge_model.assoc(left_redzone, "local")
        self.fridge_model.assoc(right_redzone, "local")
        self.fridge_model.assoc(table_redzone)
        self.obstacles = [left_redzone, right_redzone, table_redzone, self.fridge_model]

    @property
    def fridge_shape(self) -> np.ndarray:
        return self.fridge_model._extents

    def set_fridge_pose(self, pose3d: np.ndarray):
        fx, fy, fyaw = pose3d
        height = 0.5 * self.fridge_model._extents[2]
        co = Coordinates([fx, fy, height], [fyaw, 0, 0])
        self.fridge_model.newcoords(co)


class FridgePoseProvider:
    debug_selected_points: Optional[np.ndarray]
    stop_flag: bool

    def __init__(self, verbose: bool = False, is_dummy: bool = False):
        self.cloud_prov = PointCloudProvider(is_dummy=is_dummy)
        self.segm_prov = FridgeSegmProvider(is_dummy=is_dummy)
        self.est_fridge_pub = rospy.Publisher(
            "est_fridge_box", BoundingBox, latch=True, queue_size=1
        )
        self.verbose = verbose
        self.debug_selected_points = None
        self.stop_flag = True

    def reset(self):
        self.cloud_prov.reset()
        self.segm_prov.reset

    def stop(self):
        self.cloud_prov.stop()
        self.segm_prov.stop()

    def start(self):
        self.cloud_prov.start()
        self.segm_prov.start()

    def _get_relevant_points(self):
        cloud, t_cloud = self.cloud_prov.get()
        segm, t_segm = self.segm_prov.get()
        fridge_cloud = cloud[segm.flatten()]
        fridge_cloud = fridge_cloud[np.all(np.isfinite(fridge_cloud), axis=1)]
        if self.verbose:
            t_now = time.time()
            t_cloud_diff = t_now - t_cloud
            t_segm_diff = t_now - t_segm
            rospy.loginfo(f"obtained cloud and segm with {t_cloud_diff}, {t_segm_diff} [s] delay")
        down = fridge_cloud[np.arange(0, fridge_cloud.shape[0], 12)]
        down = down[down[:, 2] < 1.4]

        dbscan = DBSCAN(eps=0.05, min_samples=3, n_jobs=1, leaf_size=20)
        clusters = dbscan.fit_predict(down)
        n_label = np.max(clusters) + 1
        largest_indices = None
        largest_cluster_size = 0
        for i in range(n_label):
            indices = np.where(clusters == i)
            if len(indices[0]) > largest_cluster_size:
                largest_cluster_size = len(indices[0])
                largest_indices = indices
        selected_points = down[largest_indices]
        self.debug_selected_points = selected_points
        return selected_points

    def _fit_fridge_position(self, selected_points: np.ndarray):
        fridge_d, fridge_w, fridge_h = FridgeEnv.fridge_size
        size = np.array([fridge_d, fridge_w, fridge_h])

        def yaw_matrix(yaw):
            return np.array(
                [[np.cos(yaw), -np.sin(yaw), 0], [+np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
            )

        def cost(var: np.ndarray):
            x, y, yaw = var
            pose = Pose(np.array([x, y, 0.5 * fridge_h]), yaw_matrix(yaw))
            box = BoxSDF(size, pose)
            dist = box.evaluate_batch(selected_points.T)
            return np.mean(dist**2)

        mean = np.mean(selected_points, axis=0)
        guess_center = [mean[0] + 0.3, mean[1], 0.0]
        lb = guess_center - np.array([0.5, 0.5, 0.4 * np.pi])
        ub = guess_center + np.array([0.5, 0.5, 0.2 * np.pi])
        var_min = None
        min_cost = np.inf
        for _ in range(300):
            initial_guess = np.random.uniform(lb, ub)
            f_eval = cost(initial_guess)
            if f_eval < min_cost:
                min_cost = f_eval
                var_min = initial_guess

        res = minimize(cost, var_min, method="Nelder-Mead")
        return res.x

    def get(self):
        self.reset()
        self.start()
        selected_points = self._get_relevant_points()
        x, y, yaw = self._fit_fridge_position(selected_points)

        # self.est_fridge_pub.publish(
        return np.array([x, y, yaw])

    def get_transform(self) -> RichTrasnform:
        x, y, yaw = self.get()
        mat = rpy_matrix(yaw, 0, 0)
        tf_fridge = RichTrasnform([x, y, 0], mat, "fridge", "base_footprint")

        # debug publish bounding box
        bbox = BoundingBox()
        bbox.header.stamp = rospy.Time.now()
        bbox.header.frame_id = "base_footprint"
        bbox.pose = tf_fridge.to_ros_pose()
        bbox.pose.position.z += 0.5 * FridgeEnv.fridge_size[2]
        bbox.dimensions.x = FridgeEnv.fridge_size[0]
        bbox.dimensions.y = FridgeEnv.fridge_size[1]
        bbox.dimensions.z = FridgeEnv.fridge_size[2]
        self.est_fridge_pub.publish(bbox)
        return tf_fridge

    def save_debug_data(self):
        self.segm_prov.save_debug_data()
        self.cloud_prov.save_debug_data()


if __name__ == "__main__":
    rospy.init_node("fridge_pose_provider")
    # prov = FridgePoseProvider(verbose=True)
    cloud_prov = PointCloudProvider()
    cloud_prov.start()
    pts = cloud_prov.get()
    print(pts)
