import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
from movement_primitives.dmp import DMP
from skrobot.coordinates import Transform
from skrobot.coordinates.math import matrix2quaternion, wxyz2xyzw

from pr2dmp.trajectory import Trajectory


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
        path = project_root_path(project_name) / f"demonstration.json"
        with open(path, "w") as f:
            json.dump(dic, f, indent=4)

    @classmethod
    def load(cls, project_name: str) -> "Demonstration":
        path = project_root_path(project_name) / f"demonstration.json"
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

    def fit_dmp(self) -> DMP:
        # https://github.com/dfki-ric/movement_primitives/blob/031f4f8840f5fc87179ca5ee71f485cc4b74e2d5/examples/sim_cartesian_dmp.py#L7
        dmp = DMP(n_dims=7, execution_time=1.0, n_weight_per_dim=8, dt=0.1)
        len(self.tf_ef_to_ref_list)

        vec_list = []
        for tf_ef_to_ref in self.tf_ef_to_ref_list:
            pos = tf_ef_to_ref.translation
            rot = tf_ef_to_ref.rotation
            wxyz = matrix2quaternion(rot)
            xyzw = wxyz2xyzw(wxyz)
            vec = np.hstack([pos, xyzw])
            vec_list.append(vec)
        traj = Trajectory(np.array(vec_list))
        interped = traj.resample(100).numpy()
        ts = np.linspace(0, 1, 100)
        dmp.imitate(ts, interped)
        dmp.configure(start_y=interped[0], goal_y=interped[-1])
