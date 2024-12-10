import argparse
import time

from plainmp.robot_spec import PR2RarmSpec
from skrobot.interfaces.ros import PR2ROSRobotInterface
from skrobot.models.pr2 import PR2
from skrobot.viewers import PyrenderViewer

from pr2dmp.demonstration import Demonstration

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true", help="on real robot")
    args = parser.parse_args()

    demo = Demonstration.load("fridge_door_open")
    spec = PR2RarmSpec()
    robot = PR2(use_tight_joint_limit=False)

    if args.real:
        ri = PR2ROSRobotInterface(robot)
    else:
        # here we use the recorded ref_to_base pose
        qs, gs = demo.get_dmp_trajectory_pr2()
        viewer = PyrenderViewer()
        viewer.add(robot)
        viewer.show()
        for q in qs:
            robot.angle_vector(q)
            time.sleep(0.1)
            viewer.redraw()
        time.sleep(1000)
