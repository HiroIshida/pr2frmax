from pr2dmp.demonstration import Demonstration

if __name__ == "__main__":
    demo = Demonstration.load("fridge_door_open")
    dmp = demo.get_dmp_trajectory_pr2()
