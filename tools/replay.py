import time

import matplotlib.pyplot as plt
import numpy as np
from plainmp.ik import IKConfig, solve_ik
from plainmp.robot_spec import PR2LarmSpec
from plainmp.utils import set_robot_state
from skrobot.coordinates.math import wxyz2xyzw
from skrobot.models.pr2 import PR2
from skrobot.viewers import PyrenderViewer

from pr2dmp.demonstration import Demonstration, DMPParameter

np.random.seed(0)

if __name__ == "__main__":
    robot = PR2(use_tight_joint_limit=False)
    demo = Demonstration.load("test")
    param = DMPParameter(goal_pos_diff=np.array([0.0, -0.0, -0.05]))
    dmp = demo.get_dmp_trajectory(param)
    _, traj = dmp.open_loop()
    plt.plot(traj[:, :3])
    plt.show()

    spec = PR2LarmSpec()
    lb, ub = spec.angle_bounds()

    q_init = []
    for joint_name in spec.control_joint_names:
        index = demo.joint_names.index(joint_name)
        angle = demo.q_list[0][index]
        q_init.append(angle)

    q_list = [np.array(q_init)]
    for t, tf_cmd in enumerate(traj):
        pos, q_wxyz = tf_cmd[:3], tf_cmd[3:]
        q_wxyz = wxyz2xyzw(q_wxyz)
        cst = spec.create_gripper_pose_const(np.hstack([pos, q_wxyz]))
        ret = solve_ik(
            cst,
            None,
            lb,
            ub,
            q_seed=q_list[-1],
            config=IKConfig(ftol=1e-7, acceptable_error=1e-4),
            max_trial=100 if t == 0 else 1,
        )
        assert ret.success
        print("solved")
        q_list.append(ret.q)

    v = PyrenderViewer()
    v.add(robot)
    v.show()
    for q in q_list:
        set_robot_state(robot, spec.control_joint_names, q)
        v.redraw()
        time.sleep(0.1)
    time.sleep(100)
