import argparse

import numpy as np
from skrobot.interfaces.ros import PR2ROSRobotInterface
from skrobot.models.pr2 import PR2

from pr2dmp.demonstration import RawDemonstration
from pr2dmp.pr2_controller_utils import set_arm_controller_mode

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pn", type=str, help="project name")
    parser.add_argument("-dn", type=str, help="demonstration name")
    args = parser.parse_args()
    assert args.pn is not None
    assert args.dn is not None

    set_arm_controller_mode("rarm", "loose")
    set_arm_controller_mode("larm", "loose")
    pr2 = PR2()
    ri = PR2ROSRobotInterface(pr2)

    q_list = []
    while True:
        print("[q] to quit")
        print("any key to record")
        key = input()
        if key == "q":
            break
        q_list.append(ri.angle_vector())

    demo = RawDemonstration(np.array(q_list))
    demo.save(args.pn, args.dn)

    set_arm_controller_mode("rarm", "tight")
    set_arm_controller_mode("larm", "tight")
