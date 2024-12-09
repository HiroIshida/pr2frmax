from control_msgs.msg import JointControllerState
from skrobot.interfaces.ros.pr2 import PR2ROSRobotInterface
from skrobot.model.link import Link
from skrobot.models.pr2 import PR2

from pr2dmp.demonstration import Demonstration
from pr2dmp.example.fridge_detector import FridgeDetector
from pr2dmp.pr2_controller_utils import (
    set_arm_controller_mode,
    set_gripper_controller_mode,
)
from pr2dmp.utils import RichTrasnform

if __name__ == "__main__":
    pr2 = PR2()
    ri = PR2ROSRobotInterface(pr2)
    set_arm_controller_mode("rarm", "loose")
    set_gripper_controller_mode("rarm", "loose")
    set_gripper_controller_mode("rarm", "tight")
    # HACK: gripper must be first set to loose mode then to tight mode
    # loose => for the gripper to be able to move by human
    # tight => to measure the gripper width

    detector = FridgeDetector()
    pr2.angle_vector(ri.angle_vector())
    tf_fridge_to_basefootprint = detector.get_current_transform()
    q_list = []
    tf_list = []
    gw_list = []
    while True:
        print("[q] to quit")
        print("any key to record")
        key = input()
        if key == "q":
            break
        pr2.angle_vector(ri.angle_vector())
        gripper_state: JointControllerState = ri.gripper_states["rarm"]
        gripper_width = gripper_state.process_value
        gripper_link: Link = pr2.r_gripper_tool_frame
        tf_ef_to_basefootprint = RichTrasnform.from_co(
            gripper_link.copy_worldcoords(), gripper_link.name, "base_footprint"
        )
        tf_ef_to_fridge = tf_ef_to_basefootprint * tf_fridge_to_basefootprint.inv()
        q_list.append(pr2.angle_vector())
        tf_list.append(tf_ef_to_fridge)
        gw_list.append(gripper_width)
    demo = Demonstration(
        gripper_link.name,
        "fridge",
        tf_list,
        q_list,
        pr2.joint_names,
        gw_list,
        tf_ref_to_base=tf_fridge_to_basefootprint,
    )
    demo.save("fridge_door_open")
