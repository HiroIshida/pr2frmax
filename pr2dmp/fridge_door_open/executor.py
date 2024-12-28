import pickle
import time
from typing import Optional

import numpy as np
from skrobot.interfaces.ros import PR2ROSRobotInterface

from pr2dmp.common_node.gripper_offset_detector import AprilOffsetDetector
from pr2dmp.demonstration import (
    Demonstration,
    DMPParameter,
    RawDemonstration,
    project_root_path,
)
from pr2dmp.fridge_door_open import fridge_detector


class Executor:
    def __init__(self, ri: PR2ROSRobotInterface):
        sampler_cache_dir = project_root_path("fridge_door_open") / "sampler_cache"
        idx_cache = 280
        cache_path = sampler_cache_dir / f"opt_param_{idx_cache}.pkl"
        with open(cache_path, "rb") as f:
            opt_param_vec = pickle.load(f)

        param = DMPParameter()
        param.forcing_term_pos = opt_param_vec[:30].reshape(3, 10)
        param.gripper_forcing_term = opt_param_vec[30:].reshape(1, 10)

        demo = Demonstration.load("fridge_door_open", "open")
        demo.get_dmp_trajectory(param)  # cache for next time
        self.demo = demo
        self.fridge_detector = fridge_detector.FridgeDetector()
        self.april_offset_detector = AprilOffsetDetector()
        self.ri = ri
        self.param = param

    def execute(self, q_whole_init: Optional[np.ndarray] = None):
        if q_whole_init is None:
            q_whole_init = self.ri.angle_vector()
        tf_ref_to_base = self.fridge_detector.get_current_transform()
        ts = time.time()
        qs, gs = self.demo.get_dmp_trajectory_pr2(
            tf_ref_to_base, None, q_whole_init, param=self.param
        )
        print(f"get_dmp_trajectory_pr2: {time.time() - ts}")
        self.ri.angle_vector(qs[0])
        self.ri.wait_interpolation()
        tf_ap_to_aphat = self.april_offset_detector.get_gripper_offset()
        qs, gs = self.demo.get_dmp_trajectory_pr2(
            tf_ref_to_base, tf_ap_to_aphat, q_whole_init, param=self.param
        )

        slow = False
        for q, g in zip(qs, gs):
            self.ri.move_gripper("larm", g - 0.01, effort=100)
            av_time = 0.7 if slow else 0.4
            sleep_time = 0.45 if slow else 0.2
            self.ri.angle_vector(q, time=av_time)
            time.sleep(sleep_time)


if __name__ == "__main__":
    from skrobot.models.pr2 import PR2

    pr2 = PR2()
    ri = PR2ROSRobotInterface(pr2)
    rdemo = RawDemonstration.load("fridge_door_open", "init")
    ri.angle_vector(rdemo.q_list[0])
    ri.wait_interpolation()
    executor = Executor(ri)
    executor.execute()
