import argparse
import time

from plainmp.robot_spec import PR2RarmSpec
from skrobot.interfaces.ros import PR2ROSRobotInterface
from skrobot.model.primitives import Axis
from skrobot.models.pr2 import PR2
from skrobot.viewers import PyrenderViewer

from pr2dmp.demonstration import Demonstration
from pr2dmp.example.fridge_detector import FridgeDetector
from pr2dmp.pr2_controller_utils import (
    set_arm_controller_mode,
    set_gripper_controller_mode,
)
from pr2dmp.utils import RichTrasnform

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true", help="on real robot")
    args = parser.parse_args()

    demo = Demonstration.load("fridge_door_open")
    spec = PR2RarmSpec()
    robot = PR2(use_tight_joint_limit=False)

    if args.real:
        set_arm_controller_mode("rarm", "tight")
        set_arm_controller_mode("larm", "tight")
        set_gripper_controller_mode("rarm", "tight")
        set_gripper_controller_mode("larm", "tight")
        ri = PR2ROSRobotInterface(robot)
        ri.angle_vector(demo.q_list[0])
        ri.wait_interpolation()
        detector = FridgeDetector()
        tf_ref_to_base = detector.get_current_transform()
        qs, gs = demo.get_dmp_trajectory_pr2(tf_ref_to_base, ri.angle_vector(), n_sample=15)

        for q, g in zip(qs, gs):
            ri.move_gripper("rarm", g - 0.012, effort=100)
            ri.angle_vector(q, time=0.3)
            time.sleep(0.2)
    else:
        # here we use the recorded ref_to_base pose
        tf_obsref_to_ref = RichTrasnform.from_xytheta(-0.05, +0.05, 0.3, "fridge", "fridge")
        qs, gs = demo.get_dmp_trajectory_pr2(tf_obsref_to_ref=tf_obsref_to_ref)
        viewer = PyrenderViewer()
        viewer.add(robot)
        axis = Axis.from_coords(demo.tf_ref_to_base.to_coordinates())
        viewer.add(axis)
        viewer.show()
        for q in qs:
            robot.angle_vector(q)
            time.sleep(0.1)
            viewer.redraw()
        time.sleep(1000)
