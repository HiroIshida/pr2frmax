import numpy as np
from skrobot.coordinates import Transform
from skrobot.interfaces.ros.pr2 import PR2ROSRobotInterface
from skrobot.models.pr2 import PR2

from pr2dmp.demonstration import Demonstration

if __name__ == "__main__":
    robot = PR2()
    ri = PR2ROSRobotInterface(robot)

    ef_frame = "l_gripper_tool_frame"
    tf_ref_to_world = Transform([0.0, 0.0, 0.0], np.eye(3))
    joint_names = robot.joint_names
    q_list = []
    tf_list = []

    while True:
        print("[q] to quit")
        print("any key to record")
        key = input()
        if key == "q":
            break
        q_now = ri.angle_vector()
        co = ri.robot.__dict__[ef_frame].copy_worldcoords()
        tf_ef_to_world = Transform(co.worldpos(), co.worldrot())
        tf_ef_to_ref = tf_ef_to_world * tf_ref_to_world.inverse_transformation()
        q_list.append(q_now)
        tf_list.append(tf_ef_to_ref)
        print("recorded")
    demo = Demonstration(ef_frame, tf_list, q_list, joint_names)
    demo.save("test")
