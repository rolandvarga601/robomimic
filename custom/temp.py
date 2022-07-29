from utils import print_hdf5_structure
from utils import get_dataset_xml
from utils import plot_trajectory

if __name__ == '__main__':
    # hdf5_path = '/home/rvarga/implementation/old_robosuite-master/robosuite/models/assets/demonstrations/SawyerPickAndPlace/image.hdf5'
    hdf5_path = '/home/rvarga/implementation/robomimic/custom/data/extended_low_dim.hdf5'

    # print_hdf5_structure(hdf5_path)

    # mujoco_xml = get_dataset_xml(hdf5_path)

    # print(mujoco_xml)

    plot_trajectory(hdf5_path, 199, "obs", "robot0_eef_force")
    # plot_trajectory(hdf5_path, 199, "", "rewards")