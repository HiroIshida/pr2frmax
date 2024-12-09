import argparse
import time

import numpy as np
from plainmp.ik import IKConfig, solve_ik
from plainmp.robot_spec import PR2RarmSpec
from plainmp.utils import set_robot_state
from skrobot.coordinates import Coordinates
from skrobot.coordinates.math import matrix2quaternion, wxyz2xyzw
from skrobot.interfaces.ros.pr2 import PR2ROSRobotInterface
from skrobot.model.primitives import Axis
from skrobot.models.pr2 import PR2
from skrobot.viewers import PyrenderViewer

from pr2dmp.demonstration import Demonstration
from pr2dmp.example.fridge_detector import FridgeDetector

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true", help="move real robot")
    args = parser.parse_args()

    demo = Demonstration.load("fridge_door_open")
    pr2 = PR2()
    ri = PR2ROSRobotInterface(pr2)
    detector = FridgeDetector()
    tf_fridge_to_basefootprint = detector.get_current_transform()
    spec = PR2RarmSpec()
    lb, ub = spec.angle_bounds()
    dic = {name: angle for name, angle in zip(demo.joint_names, demo.q_list[0])}
    q_init = np.array([dic[name] for name in spec.control_joint_names])

    ikconfig = IKConfig(ftol=1e-7, acceptable_error=1e-4)

    q_control_list = []
    for tf_ef_to_fridge in demo.tf_ef_to_ref_list:
        tf_ef_to_basefootprint = tf_ef_to_fridge * tf_fridge_to_basefootprint
        pos = tf_ef_to_basefootprint.translation
        quat = wxyz2xyzw(matrix2quaternion(tf_ef_to_basefootprint.rotation))
        cst = spec.create_gripper_pose_const(np.hstack([pos, quat]))
        ret = solve_ik(cst, None, lb, ub, q_seed=q_init, config=ikconfig)
        q_init = ret.q
        assert ret.success
        q_control_list.append(ret.q)

    if args.real:
        pass
    else:
        v = PyrenderViewer()
        co = Coordinates(
            pos=tf_ef_to_basefootprint.translation, rot=tf_fridge_to_basefootprint.rotation
        )
        axis = Axis.from_coords(co)
        v.add(pr2)
        v.add(axis)
        set_robot_state(pr2, spec.control_joint_names, q_control_list[0])
        v.show()
        time.sleep(2)
        for q_control in q_control_list:
            set_robot_state(pr2, spec.control_joint_names, q_control)
            v.redraw()
            time.sleep(3)
        time.sleep(100)
