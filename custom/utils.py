import os
import h5py
import matplotlib
import nexusformat.nexus as nx

import torch
from torch.utils.data import DataLoader

from robomimic.utils.dataset import SequenceDataset

import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
    

def get_data_loader(dataset_path, seq_length=1):
    """
    Get a data loader to sample batches of data.
    Args:
        dataset_path (str): path to the dataset hdf5
    """
    dataset = SequenceDataset(
        hdf5_path=dataset_path,
        obs_keys=(                      # observations we want to appear in batches
            "robot0_eef_pos", 
            "robot0_eef_quat", 
            "robot0_gripper_qpos", 
            "object",
        ),
        dataset_keys=(                  # can optionally specify more keys here if they should appear in batches
            # "actions", 
            # "rewards", 
            # "dones",
        ),
        load_next_obs=True,
        frame_stack=1,
        seq_length=seq_length,                  # length-seq_length temporal sequences
        pad_frame_stack=True,
        pad_seq_length=True,            # pad last obs per trajectory to ensure all sequences are sampled
        get_pad_mask=False,
        goal_mode=None,
        hdf5_cache_mode="low_dim",          # cache everything from dataset except images in memory to avoid repeated file i/o
        hdf5_use_swmr=True,
        hdf5_normalize_obs=False,
        filter_by_attribute=None,       # can optionally provide a filter key here
    )
    print("\n============= Created Dataset =============")
    print(dataset)
    print("")

    data_loader = DataLoader(
        dataset=dataset,
        sampler=None,       # no custom sampling logic (uniform sampling)
        batch_size=100,     # batches of size 100
        shuffle=True,
        num_workers=0,
        drop_last=True      # don't provide last batch in dataset pass if it's less than 100 in size
    )
    return data_loader


def print_hdf5_structure(dataset_path):
    # enforce that the dataset exists
    assert os.path.exists(dataset_path)

    with nx.nxload(dataset_path, 'r') as f:
        print(f.tree)

def get_dataset_xml(dataset_path):
    # enforce that the dataset exists
    assert os.path.exists(dataset_path)

    mujoco_xml_file = ""

    with h5py.File(dataset_path, 'r') as f:
        demo_grp = f["data/demo_1"]
        mujoco_xml_file = demo_grp.attrs["model_file"]

    return mujoco_xml_file

def plot_trajectory(dataset_path, demo_num, group_name, signal_name):
    # enforce that the dataset exists
    assert os.path.exists(dataset_path)

    with h5py.File(dataset_path, 'r') as f:
        demos = list(f["data"].keys())

        assert (len(demos) >= demo_num), f"Demo number {demo_num} is higher than the number of demonstrations in the dataset"

        if (group_name == ""):
            signal = f["data/demo_{}/{}".format(demo_num, signal_name)]
        else:
            signal = f["data/demo_{}/{}/{}".format(demo_num, group_name, signal_name)]

        print(f"Found '{signal.name}' in the dataset")

        plt.plot(range(signal.shape[0]), signal[:])
        # plt.plot(range(signal.shape[0]), signal)
        plt.xlabel('Sample index')
        plt.ylabel(signal_name)

        plt.show()



