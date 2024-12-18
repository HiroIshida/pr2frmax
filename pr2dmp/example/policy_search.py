# flake8: noqa F402
import site
import sys

site.addsitedir("/usr/lib/python3/dist-packages")
import argparse
import pickle
import subprocess
import time
from typing import Optional

import numpy as np
import rospy
from frmax2.core import (
    CompositeMetric,
    DGSamplerConfig,
    DistributionGuidedSampler,
    UniformSituationSampler,
)
from plainmp.robot_spec import PR2RarmSpec
from skrobot.coordinates.math import rpy_matrix
from skrobot.interfaces.ros import PR2ROSRobotInterface
from skrobot.model.primitives import Axis
from skrobot.models.pr2 import PR2
from skrobot.viewers import PyrenderViewer

from pr2dmp.common_node.gripper_offset_detector import AprilOffsetDetector
from pr2dmp.demonstration import (
    Demonstration,
    DMPParameter,
    RawDemonstration,
    project_root_path,
)
from pr2dmp.example.fridge_detector import FridgeDetector
from pr2dmp.pr2_controller_utils import (
    set_arm_controller_mode,
    set_gripper_controller_mode,
    set_head_controller_mode,
)
from pr2dmp.utils import RichTrasnform


def speak(message: str) -> None:
    rospy.loginfo(message)
    subprocess.call('echo "{}" | festival --tts'.format(message), shell=True)


class RolloutExecutor:
    def __init__(self, demo: Demonstration):
        # use tight controller
        set_arm_controller_mode("rarm", "tight")
        set_arm_controller_mode("larm", "tight")
        set_gripper_controller_mode("rarm", "tight")
        set_gripper_controller_mode("larm", "tight")
        set_head_controller_mode("tight")

        robot = PR2(use_tight_joint_limit=False)
        ri = PR2ROSRobotInterface(robot)
        robot.angle_vector(ri.angle_vector())
        torso_current_height = robot.torso_lift_joint.joint_angle()

        torso_index = robot.joint_names.index("torso_lift_joint")
        av_init = demo.q_list[0]
        av_init[torso_index] = torso_current_height

        detector = FridgeDetector()
        april_detector = AprilOffsetDetector(debug=True)

        self.cleaup_motion = RawDemonstration.load("fridge_door_open", "close")
        q_init = self.cleaup_motion.resolved_trajectory(av_init, PR2RarmSpec())[0]

        self.q_full_init = q_init
        self.demo = demo
        self.ri = ri
        self.detector = detector
        self.april_detector = april_detector

    def get_manual_annotation(self) -> Optional[bool]:
        while True:
            speak("manual annotation required")
            user_input = input("Add label: Enter 'y' for True or 'n' for False, r for retry")
            if user_input.lower() == "y":
                return True
            elif user_input.lower() == "n":
                return False
            elif user_input.lower() == "r":
                return None

    def cleanup(self):
        q_list = self.cleaup_motion.resolved_trajectory(self.ri.angle_vector(), PR2RarmSpec())
        self.ri.angle_vector_sequence(q_list, [1.5] * len(q_list))
        self.ri.wait_interpolation()

    def rollout(
        self, param_vec: Optional[np.ndarray], error: Optional[np.ndarray], slow: bool = False
    ) -> Optional[bool]:
        if param_vec is None:
            param_vec = np.zeros(40)
        if error is None:
            error = np.zeros(3)
        rospy.loginfo(f"executing rollout with param {...} and error {error}")

        # reset robot to home position
        self.ri.angle_vector(self.q_full_init)
        self.ri.wait_interpolation()
        time.sleep(1)

        # observe the fridge and apriltag
        tf_ref_to_base = self.detector.get_current_transform()
        tf_ap_to_aphat = self.april_detector.get_gripper_offset()

        param = DMPParameter()
        param.forcing_term_pos = param_vec[:30].reshape(3, 10)
        param.gripper_forcing_term = param_vec[30:].reshape(1, 10)

        x_err, y_err, yaw_err = error
        rotmat = rpy_matrix(yaw_err, 0, 0)
        tf_obsref_to_ref = RichTrasnform(np.array([x_err, y_err, 0.0]), rotmat, "fridge", "fridge")
        qs, gs = demo.get_dmp_trajectory_pr2(
            tf_ref_to_base,
            tf_ap_to_aphat,
            self.ri.angle_vector(),
            n_sample=15,
            param=param,
            tf_obsref_to_ref=tf_obsref_to_ref,
        )

        for q, g in zip(qs, gs):
            self.ri.move_gripper("larm", g - 0.01, effort=100)
            av_time = 1.0 if slow else 0.4
            sleep_time = 0.6 if slow else 0.2
            self.ri.angle_vector(q, time=av_time)
            time.sleep(sleep_time)

        label = self.get_manual_annotation()
        if label:
            self.cleanup()
        return label


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true", help="on real robot")
    parser.add_argument("--mode", type=str, default="simu", choices=["simu", "dry", "train"])
    parser.add_argument("--slow", action="store_true", help="slow")
    args = parser.parse_args()

    demo = Demonstration.load("fridge_door_open")
    spec = PR2RarmSpec()
    robot = PR2(use_tight_joint_limit=False)

    if args.mode in ("dry", "train"):
        executor = RolloutExecutor(demo)
        if args.mode == "dry":
            executor.rollout(None, None, args.slow)
        else:
            n_param = 30 + 10
            ls_param = np.ones(n_param) * 10
            ls_err = np.array([0.015, 0.015, np.deg2rad(5.0)])

            situ_sampler = UniformSituationSampler(-ls_err * 3, ls_err * 3)

            param_init = np.zeros(n_param)
            X = [np.hstack([param_init, np.zeros(ls_err.size)])]
            Y = [True]
            for i in range(1):
                speak(f"initial sampling number {i}")
                e = situ_sampler()
                label = executor.rollout(param_init, e, args.slow)
                if label is None:
                    sys.exit()
                x = np.hstack([param_init, e])
                X.append(x)
                Y.append(label)

            config = DGSamplerConfig(
                param_ls_reduction_rate=0.999,
                n_mc_param_search=30,
                c_svm=10000,
                integration_method="mc",
                n_mc_integral=300,
                r_exploration=1.0,
            )
            X = np.array(X)
            Y = np.array(Y, dtype=bool)
            metric = CompositeMetric.from_ls_list([ls_param, ls_err])
            sampler = DistributionGuidedSampler(
                X,
                Y,
                metric,
                param_init,
                config,
                situation_sampler=situ_sampler,
                use_prefacto_branched_ask=False,
            )
            sampler_cache_dir = project_root_path("fridge_door_open") / "sampler_cache"
            sampler_cache_dir.mkdir(exist_ok=True)
            for i in range(1000):
                speak(f"active sampling iteration {i}")
                x = sampler.ask()
                param, error = x[:-3], x[-3:]
                label = executor.rollout(param, error, args.slow)
                sampler.tell(x, label)
                file_path = sampler_cache_dir / f"cache_{i}.pkl"
                with open(file_path, "wb") as f:
                    pickle.dump(sampler, f)
                rospy.loginfo(f"saved sampler to {file_path}")
    else:
        # here we use the recorded ref_to_base pose
        param = DMPParameter()
        param.forcing_term_pos = np.random.uniform(-30, 30, (3, 10))
        tf_obsref_to_ref = RichTrasnform.from_xytheta(-0.0, +0.0, 0.0, "fridge", "fridge")
        qs, gs = demo.get_dmp_trajectory_pr2(
            tf_obsref_to_ref=tf_obsref_to_ref, n_sample=15, param=param
        )
        viewer = PyrenderViewer()
        viewer.add(robot)
        axis = Axis.from_coords(demo.tf_ref_to_base.to_coordinates())
        viewer.add(axis)
        viewer.show()
        time.sleep(2)
        for q in qs:
            robot.angle_vector(q)
            time.sleep(0.5)
            viewer.redraw()
        time.sleep(1000)
