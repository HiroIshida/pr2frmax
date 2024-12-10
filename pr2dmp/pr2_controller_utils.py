import argparse
from typing import Dict, Literal

import rospy
from pr2_mechanism_msgs.srv import (
    ListControllers,
    ListControllersResponse,
    SwitchController,
)


def get_controller_states() -> Dict[str, bool]:
    sp = rospy.ServiceProxy("/pr2_controller_manager/list_controllers", ListControllers)
    resp: ListControllersResponse = sp()
    return {cont: (state == "running") for (cont, state) in zip(resp.controllers, resp.state)}


def set_gripper_controller_mode(
    arm: Literal["rarm", "larm"], mode: Literal["tight", "loose"]
) -> None:
    if arm == "rarm":
        controller_name = "r_gripper_controller"
    else:
        controller_name = "l_gripper_controller"
    state = get_controller_states()
    if mode == "tight":
        if state[controller_name]:
            rospy.loginfo(f"{controller_name} is already active")
            return
        sp = rospy.ServiceProxy("/pr2_controller_manager/switch_controller", SwitchController)
        resp = sp(start_controllers=[controller_name], stop_controllers=[])
        rospy.loginfo("controller service response: {}".format(resp))
        state = get_controller_states()
        assert not state[controller_name]
    elif mode == "loose":
        if not state[controller_name]:
            rospy.loginfo(f"{controller_name} is already inactive")
            return
        sp = rospy.ServiceProxy("/pr2_controller_manager/switch_controller", SwitchController)
        resp = sp(start_controllers=[], stop_controllers=[controller_name])
        rospy.loginfo("controller service response: {}".format(resp))
        state = get_controller_states()
        assert not state[controller_name]
    else:
        raise ValueError("mode must be 'tight' or 'loose")


def set_arm_controller_mode(arm: Literal["rarm", "larm"], mode: Literal["tight", "loose"]) -> None:
    """Set the controller mode for a specified arm.

    Args:
        arm: Which arm to control ("rarm" or "larm")
        mode: Which mode to set ("tight" or "loose")
    """
    if arm == "rarm":
        controller_name = "r_arm_controller"
    elif arm == "larm":
        controller_name = "l_arm_controller"
    else:
        raise ValueError("arm must be 'rarm' or 'larm'")

    loose_controller_name = controller_name + "_loose"
    state = get_controller_states()
    is_controller_active = state[controller_name]
    is_loose_controller_active = state[loose_controller_name]
    is_xor = is_controller_active ^ is_loose_controller_active
    assert is_xor, "controller state is strange"

    if mode == "tight":
        target_controller = controller_name
        other_controller = loose_controller_name
        check_active = is_controller_active
    elif mode == "loose":
        target_controller = loose_controller_name
        other_controller = controller_name
        check_active = is_loose_controller_active
    else:
        raise ValueError("mode must be 'tight' or 'loose'")

    if check_active:
        rospy.loginfo(f"{target_controller} is already active")
        return

    sp = rospy.ServiceProxy("/pr2_controller_manager/switch_controller", SwitchController)
    resp = sp(start_controllers=[target_controller], stop_controllers=[other_controller])
    rospy.loginfo("controller service response: {}".format(resp))

    state = get_controller_states()
    assert state[target_controller]
    assert state[other_controller]


if __name__ == "__main__":
    rospy.init_node("pr2_controller_switcher", log_level=rospy.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--loose", action="store_true", help="Switch to loose controller")
    parser.add_argument("--tight", action="store_true", help="Switch to tight controller")
    parser.add_argument("--arm", type=str, choices=["rarm", "larm"])
    args = parser.parse_args()

    if args.arm is None:
        arms = ["rarm", "larm"]
    else:
        arms = [args.arm]

    if args.loose:
        for arm in arms:
            set_controller_mode(arm, "loose")
    if args.tight:
        for arm in arms:
            set_controller_mode(arm, "tight")
