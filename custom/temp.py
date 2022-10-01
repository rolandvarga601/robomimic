from utils import print_hdf5_structure
from utils import get_dataset_xml
from utils import plot_trajectory
from utils import get_trajectory
import numpy as np

if __name__ == '__main__':
    # hdf5_path = '/home/rvarga/implementation/old_robosuite-master/robosuite/models/assets/demonstrations/SawyerPickAndPlace/image.hdf5'
    hdf5_path = '/home/rvarga/implementation/robomimic/custom/data/extended_low_dim_shaped.hdf5'

    # print_hdf5_structure(hdf5_path)

    # mujoco_xml = get_dataset_xml(hdf5_path)

    # print(mujoco_xml)

    plot_trajectory(hdf5_path, 196, "obs", "robot0_eef_force")
    # plot_trajectory(hdf5_path, 196, "obs", "robot0_eef_vel_lin")
    # plot_trajectory(hdf5_path, 199, "", "rewards")


    signal = get_trajectory(hdf5_path, 0, "obs", "robot0_eef_force")

    for i in range(1, 200):
        signal = np.vstack((signal, get_trajectory(hdf5_path, i, "obs", "robot0_eef_force")))

    min = signal.min()
    max = signal.max()
    std = np.std(signal, axis=0)

    print(f"The min is {min} and the max is {max} and the standard deviation is {std}")

    