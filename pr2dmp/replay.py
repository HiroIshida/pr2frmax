from pr2dmp.demonstration import Demonstration

if __name__ == "__main__":
    project_name = "test"
    demo = Demonstration.load(project_name)

    for t in len(demo):
        tf_ef_to_ref = demo.tf_ef_to_ref_list[t]
