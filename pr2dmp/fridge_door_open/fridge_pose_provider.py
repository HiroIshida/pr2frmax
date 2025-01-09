import time
from typing import ClassVar

import cv2
import numpy as np
import rospy
from jsk_recognition_msgs.msg import BoundingBox
from sklearn.cluster import DBSCAN
from skrobot.coordinates import Coordinates
from skrobot.model.primitives import Axis, Box

from pr2dmp.common_node.common_provider import FridgeSegmProvider, PointCloudProvider
from pr2dmp.ransac import fit_plane_axis_aligned


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

    def get(self):
        self.reset()
        self.start()
        selected_points = self._get_relevant_points()
        ts = time.time()
        coeffs, indices_inliers = fit_plane_axis_aligned(selected_points, 0.01)
        print(f"fitting time: {time.time() - ts}")
        self.inliers = selected_points[indices_inliers]

        # chekc if inliers are on the same plane
        A, B, C, D = coeffs
        assert abs(A**2 + B**2 + C**2 - 1) < 1e-5
        norm_abc = np.linalg.norm([A, B, C])
        normal = -np.array([A, B, C]) / norm_abc
        dist = D
        if normal[0] < 0:
            normal = -normal
            dist = -dist

        ex = normal
        ez_tmp = np.array([0, 0, 1])
        ey = np.cross(ez_tmp, ex) / np.linalg.norm(np.cross(ez_tmp, ex))
        ez = np.cross(ex, ey)

        # take dot produce of ey and inliner points to get min and max
        y_coords = np.dot(ey, self.inliers.T)
        y_min, y_max = np.min(y_coords), np.max(y_coords)

        pos_global = (dist + 0.5 * FridgeEnv.fridge_size[0]) * ex + 0.5 * (y_min + y_max) * ey
        x_global, y_global = pos_global[:2]
        yaw = np.arctan2(ex[1], ex[0])
        rotmat = np.array([ex, ey, ez])
        self.debug_coords = Coordinates(pos_global, rotmat.T)
        return np.array([x_global, y_global, yaw])

    def _get_relevant_points(self):
        cloud, t_cloud = self.cloud_prov.get()
        segm, t_segm = self.segm_prov.get()
        segm = cv2.dilate(segm.astype(np.uint8), np.ones((30, 30), np.uint8), iterations=1).astype(
            bool
        )
        fridge_cloud = cloud[segm.flatten()]
        fridge_cloud = fridge_cloud[np.all(np.isfinite(fridge_cloud), axis=1)]

        if self.verbose:
            t_now = time.time()
            t_cloud_diff = t_now - t_cloud
            t_segm_diff = t_now - t_segm
            rospy.loginfo(f"obtained cloud and segm with {t_cloud_diff}, {t_segm_diff} [s] delay")
        down = fridge_cloud[np.arange(0, fridge_cloud.shape[0], 12)]
        down = down[down[:, 2] < 1.4]
        down = down[down[:, 2] > 0.2]

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

    def save_debug_data(self):
        self.segm_prov.save_debug_data()
        self.cloud_prov.save_debug_data()


if __name__ == "__main__":
    rospy.init_node("fridge_pose_provider")
    prov = FridgePoseProvider(verbose=False, is_dummy=True)
    ret = prov.get()

    from skrobot.model.primitives import Axis, PointCloudLink
    from skrobot.models.pr2 import PR2
    from skrobot.viewers import PyrenderViewer

    v = PyrenderViewer()
    pr2 = PR2()
    pr2.reset_manip_pose()
    env = FridgeEnv()
    env.set_fridge_pose(ret)
    v.add(env.fridge_model)
    v.add(pr2)
    v.add(Axis.from_coords(prov.debug_coords))
    v.add(PointCloudLink(prov.inliers))
    v.show()
    import time

    time.sleep(1000)
