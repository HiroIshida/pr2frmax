import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
from movement_primitives.dmp import DMP, CartesianDMP
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from skrobot.coordinates import Transform
from skrobot.coordinates.math import matrix2quaternion, xyzw2wxyz


def root_path() -> Path:
    root = Path("~/.pr2dmp").expanduser()
    root.mkdir(exist_ok=True)
    return root


def project_root_path(project_name: str) -> Path:
    root = root_path() / project_name
    root.mkdir(exist_ok=True)
    return root


@dataclass
class Demonstration:
    ef_frame: str
    tf_ef_to_ref_list: List[Transform]
    q_list: Optional[List[np.ndarray]]
    joint_names: Optional[List[str]]

    def __post_init__(self) -> None:
        assert len(self.tf_ef_to_ref_list) == len(self.q_list)

    def __len__(self) -> int:
        return len(self.tf_ef_to_ref_list)

    def save(self, project_name: str) -> None:
        def transform_to_vector(t: Transform) -> np.ndarray:
            rot = t.rotation
            return np.hstack([t.translation, rot.flatten()])

        dic = {}
        dic["ef_frame"] = self.ef_frame
        dic["trajectory"] = [transform_to_vector(t).tolist() for t in self.tf_ef_to_ref_list]
        dic["q_list"] = [q.tolist() for q in self.q_list]
        dic["joint_names"] = self.joint_names
        path = project_root_path(project_name) / "demonstration.json"
        with open(path, "w") as f:
            json.dump(dic, f, indent=4)

    @classmethod
    def load(cls, project_name: str) -> "Demonstration":
        path = project_root_path(project_name) / "demonstration.json"
        with open(path, "r") as f:
            dic = json.load(f)
        ef_frame = dic["ef_frame"]
        trajectory = []
        for t in dic["trajectory"]:
            translation = t[:3]
            rotation = t[3:]
            rot = np.array(rotation).reshape(3, 3)
            t = Transform(translation, rot)
            trajectory.append(t)
        q_list = [np.array(q) for q in dic["q_list"]]
        joint_names = dic["joint_names"]
        return cls(ef_frame, trajectory, q_list, joint_names)

    def get_dmp(self, param: Optional[np.ndarray] = None) -> DMP:
        # resample
        n_wp_resample = 100  # except the start point
        n_wp_orignal = len(self)
        n_segment_original = n_wp_orignal - 1
        m_assign_base = n_wp_resample // n_segment_original
        n_wp_arr = np.array([m_assign_base] * n_segment_original)
        rem = n_wp_resample % n_segment_original
        for i in range(rem):
            n_wp_arr[i] += 1
        assert sum(n_wp_arr) == n_wp_resample

        vec_list = []

        # the first point
        tf = self.tf_ef_to_ref_list[0]
        pos = tf.translation
        rot = tf.rotation
        wxyz = matrix2quaternion(rot)
        vec = np.hstack([pos, wxyz])
        vec_list.append(vec)

        for i_segment in range(n_segment_original):
            tf_start = self.tf_ef_to_ref_list[i_segment]
            tf_end = self.tf_ef_to_ref_list[i_segment + 1]
            pos_start, rot_start = tf_start.translation, tf_start.rotation
            pos_end, rot_end = tf_end.translation, tf_end.rotation

            tlin = np.linspace(0, 1, n_wp_arr[i_segment] + 1)

            # position
            pos_list = []
            for t in tlin[1:]:
                pos = pos_start * (1 - t) + pos_end * t
                pos_list.append(pos)

            # rotation
            rotations = R.from_matrix(np.array([rot_start, rot_end]))
            slerp = Slerp(np.array([0, 1]), rotations)
            interp_rots = slerp(tlin[1:])
            quat_list = []
            for rot in interp_rots:
                xyzw = rot.as_quat()
                quat_list.append(xyzw2wxyz(xyzw))

            assert len(pos_list) == len(quat_list)
            for pos, quat in zip(pos_list, quat_list):
                vec = np.hstack([pos, quat])
                vec_list.append(vec)

        dmp = CartesianDMP(1.0, dt=0.01, n_weights_per_dim=10, int_dt=0.0001)
        Y = np.array(vec_list)
        T = np.linspace(0, 1, 101)
        dmp.imitate(T, Y)
        dmp.configure(start_y=Y[0], goal_y=Y[-1])
        return dmp