import argparse

import numpy as np
from control_msgs.msg import JointControllerState
from skrobot.interfaces.ros.pr2 import PR2ROSRobotInterface
from skrobot.models.pr2 import PR2

from pr2dmp.common_node.gripper_offset_detector import AprilOffsetDetector
from pr2dmp.demonstration import Demonstration
from pr2dmp.fridge_door_open.fridge_detector import FridgeDetector
from pr2dmp.pr2_controller_utils import (
    set_arm_controller_mode,
    set_gripper_controller_mode,
    set_head_controller_mode,
)
from pr2dmp.utils import RichTrasnform

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", type=str, choices=["larm", "rarm"], default="larm")
    parser.add_argument("--nocalib", action="store_true", help="Do not calibrate gripper")
    args = parser.parse_args()

    pr2 = PR2()
    ri = PR2ROSRobotInterface(pr2)
    for arm in ["larm", "rarm"]:
        set_arm_controller_mode(arm, "loose")
        set_gripper_controller_mode(arm, "loose")
        set_gripper_controller_mode(arm, "tight")
    set_head_controller_mode("loose")
    # HACK: gripper must be first set to loose mode then to tight mode
    # loose => for the gripper to be able to move by human
    # tight => to measure the gripper width

    arm = args.arm
    do_claib = not args.nocalib
    pr2.angle_vector(ri.angle_vector())
    q_list = []
    gw_list = []
    tf_ap_to_aphat = None
    tf_fridge_to_basefootprint = None
    while True:
        print("[q] to quit")
        print("any key to record")
        key = input()
        if key == "q":
            break
        if len(q_list) == 0:
            fridge_detector = FridgeDetector()
            tf_fridge_to_basefootprint = fridge_detector.get_current_transform()

            if do_claib:
                offset_detector = AprilOffsetDetector(debug=True)
                tf_ap_to_aphat = offset_detector.get_gripper_offset()
            else:
                tf_ap_to_aphat = RichTrasnform(np.zeros(3), np.eye(3), "apriltag", "apriltag_hat")

        gripper_state: JointControllerState = ri.gripper_states[arm]
        gripper_width = gripper_state.process_value

        pr2.angle_vector(ri.angle_vector())
        q_list.append(pr2.angle_vector())
        gw_list.append(gripper_width)
    gripper_frame_name = "l_gripper_tool_frame" if arm == "larm" else "r_gripper_tool_frame"
    demo = Demonstration(
        gripper_frame_name,
        "fridge",
        tf_ap_to_aphat,
        q_list,
        pr2.joint_names,
        gw_list,
        tf_ref_to_base=tf_fridge_to_basefootprint,
    )
    demo_name = input("demo name: (e.g. open): ")
    demo.save("fridge_door_open", demo_name)
