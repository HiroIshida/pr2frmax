import time

from skrobot.coordinates import Coordinates
from skrobot.model.primitives import Axis
from skrobot.models.pr2 import PR2
from skrobot.viewers import PyrenderViewer

from pr2dmp.demonstration import Demonstration

if __name__ == "__main__":
    demo = Demonstration.load("fridge_door_open")
    co = Coordinates(demo.tf_ref_to_base.translation, demo.tf_ref_to_base.rotation)
    axis = Axis.from_coords(co)
    pr2 = PR2()

    for q, tf_ef_to_ref in zip(demo.q_list, demo.tf_ef_to_ref_list):
        tf_ef_to_base = tf_ef_to_ref * demo.tf_ref_to_base
        pr2.angle_vector(q)
        print(tf_ef_to_base.translation)
        print(pr2.__dict__["r_gripper_tool_frame"].worldpos())

    v = PyrenderViewer()
    v.add(axis)
    v.add(pr2)
    v.show()
    pr2.angle_vector(demo.q_list[0])
    for q in demo.q_list:
        pr2.angle_vector(q)
        time.sleep(1.0)
        v.redraw()
    import time

    time.sleep(1000)
