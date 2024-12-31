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
from pr2dmp.fridge_door_open.fridge_pose_provider import FridgePoseProvider
from pr2dmp.pr2_controller_utils import (
    set_arm_controller_mode,
    set_gripper_controller_mode,
    set_head_controller_mode,
)


class Executor:
    def __init__(self, ri: PR2ROSRobotInterface):
        project_root_path("fridge_door_open") / "sampler_cache"
        # idx_cache = 280
        # cache_path = sampler_cache_dir / f"opt_param_{idx_cache}.pkl"
        # with open(cache_path, "rb") as f:
        #     opt_param_vec = pickle.load(f)

        param = DMPParameter()
        # param.forcing_term_pos = opt_param_vec[:30].reshape(3, 10)
        # param.gripper_forcing_term = opt_param_vec[30:].reshape(1, 10)

        demo = Demonstration.load("fridge_door_open", "open")
        demo.get_dmp_trajectory(param)  # cache for next time
        self.demo = demo
        self.pose_provider = FridgePoseProvider()
        self.pose_provider.start()
        self.april_offset_detector = AprilOffsetDetector()
        self.ri = ri
        self.param = param

    def execute(self, q_whole_init: Optional[np.ndarray] = None):
        if q_whole_init is None:
            q_whole_init = self.ri.angle_vector()
        tf_ref_to_base = self.pose_provider.get_transform()
        ts = time.time()
        qs, gs = self.demo.get_dmp_trajectory_pr2(
            tf_ref_to_base, None, q_whole_init, param=self.param
        )
        print(f"get_dmp_trajectory_pr2: {time.time() - ts}")
        self.ri.angle_vector(qs[0])
        self.ri.wait_interpolation()
        tf_ap_to_aphat = self.april_offset_detector.get_gripper_offset()
        qs, gs = self.demo.get_dmp_trajectory_pr2(
            tf_ref_to_base, tf_ap_to_aphat, q_whole_init, param=self.param, n_sample=20
        )

        init_gripper_pos = 0.05
        self.ri.move_gripper("larm", init_gripper_pos)
        self.ri.wait_interpolation()
        scale = 1.0
        self.ri.angle_vector_sequence(list(qs), times=[0.3 * scale] * len(qs))
        time.sleep(2 * scale)
        self.ri.move_gripper("larm", 0.01, effort=100)
        time.sleep(2 * scale)
        self.ri.move_gripper("larm", 0.03, effort=100)


if __name__ == "__main__":
    from skrobot.models.pr2 import PR2

    set_arm_controller_mode("rarm", "tight")
    set_arm_controller_mode("larm", "tight")
    set_gripper_controller_mode("rarm", "tight")
    set_gripper_controller_mode("larm", "tight")
    set_head_controller_mode("tight")

    pr2 = PR2(use_tight_joint_limit=False)
    ri = PR2ROSRobotInterface(pr2)
    rdemo = RawDemonstration.load("fridge_door_open", "init")
    ri.angle_vector(rdemo.q_list[0])
    ri.wait_interpolation()
    executor = Executor(ri)
    executor.execute()
