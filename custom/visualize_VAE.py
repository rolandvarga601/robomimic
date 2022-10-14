from cProfile import label
from logging import exception
import os
from robomimic.utils import tensor_utils as TensorUtils
from visual_utils.training import get_data_loader
from visual_utils.encoding import load_observer
import torch
from robomimic.utils import obs_utils as ObsUtils

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


if __name__ == "__main__":

    encoder = load_observer("/home/rvarga/implementation/robomimic/custom/ckpt/epoch99.pth")
    encoder.set_eval()

    expert_data_path = "/home/rvarga/implementation/robomimic/custom/data/extended_low_dim_shaped.hdf5"
    expert_data_path_mg = "/home/rvarga/implementation/robomimic/custom/data/mg_low_dim_extended_shaped.hdf5"

    assert os.path.exists(expert_data_path)

    data_loader_train = get_data_loader(dataset_path=expert_data_path, seq_length=1, normalize_obs=True, filter_key="train")
    data_loader_valid = get_data_loader(dataset_path=expert_data_path, seq_length=1, normalize_obs=False, filter_key="valid")

    data_loader_iterator = iter(data_loader_valid)

    batch = next(data_loader_iterator)

    obs_norms = []
    for k in range(data_loader_valid.batch_size):
        obs_dict = {key: torch.unsqueeze(batch["obs"][key][k, 0, :], 0) for key in batch["obs"]}
        obs_norms.append(ObsUtils.normalize_obs(obs_dict=obs_dict, obs_normalization_stats=data_loader_train.dataset.get_obs_normalization_stats()))
    
    batch["obs"] = {key: torch.cat(tuple(torch.unsqueeze(o[key], 0) for o in obs_norms), 0) for key in obs_dict}

    batch['obs']['robot0_eef_force'] = batch['obs']['robot0_eef_force']/1.0
    # batch['next_obs']['robot0_eef_force'] = batch['next_obs']['robot0_eef_force']/1.0

    input_batch = encoder.process_batch_for_training(batch)

    info = encoder.train_on_batch(batch=input_batch, epoch=1, validate=True)

    robot_positions = input_batch['obs']['robot0_eef_pos'].cpu().detach().numpy()
    pred_robot_positions = info['predictions']['actions']['robot0_eef_pos'].cpu().detach().numpy()

    robot_force = input_batch['obs']['robot0_eef_force'].cpu().detach().numpy()
    pred_robot_force = info['predictions']['actions']['robot0_eef_force'].cpu().detach().numpy()

    robot_vel = input_batch['obs']['robot0_eef_vel_lin'].cpu().detach().numpy()
    pred_robot_vel = info['predictions']['actions']['robot0_eef_vel_lin'].cpu().detach().numpy()


    timespan = 380

    plt.subplot(331)
    plt.plot(range(timespan), robot_positions[:timespan, 0], label='Original')
    plt.plot(range(timespan), pred_robot_positions[:timespan, 0], label='Reconstructed')
    plt.xlabel('Sample index')
    plt.ylabel('EEF position X')
    plt.legend()


    plt.subplot(334)
    plt.plot(range(timespan), robot_positions[:timespan, 1], label='Original')
    plt.plot(range(timespan), pred_robot_positions[:timespan, 1], label='Reconstructed')
    plt.xlabel('Sample index')
    plt.ylabel('EEF position Y')
    plt.legend()

    plt.subplot(337)
    plt.plot(range(timespan), robot_positions[:timespan, 2], label='Original')
    plt.plot(range(timespan), pred_robot_positions[:timespan, 2], label='Reconstructed')
    plt.xlabel('Sample index')
    plt.ylabel('EEF position Z')
    plt.legend()

    plt.subplot(332)
    plt.plot(range(timespan), robot_force[:timespan, 0], label='Original')
    plt.plot(range(timespan), pred_robot_force[:timespan, 0], label='Reconstructed')
    plt.xlabel('Sample index')
    plt.ylabel('EEF force X')
    plt.legend()

    plt.subplot(335)
    plt.plot(range(timespan), robot_force[:timespan, 1], label='Original')
    plt.plot(range(timespan), pred_robot_force[:timespan, 1], label='Reconstructed')
    plt.xlabel('Sample index')
    plt.ylabel('EEF force Y')
    plt.legend()

    plt.subplot(338)
    plt.plot(range(timespan), robot_force[:timespan, 2], label='Original')
    plt.plot(range(timespan), pred_robot_force[:timespan, 2], label='Reconstructed')
    plt.xlabel('Sample index')
    plt.ylabel('EEF force Z')
    plt.legend()

    plt.subplot(333)
    plt.plot(range(timespan), robot_vel[:timespan, 0], label='Original')
    plt.plot(range(timespan), pred_robot_vel[:timespan, 0], label='Reconstructed')
    plt.xlabel('Sample index')
    plt.ylabel('EEF vel X')
    plt.legend()

    plt.subplot(336)
    plt.plot(range(timespan), robot_vel[:timespan, 1], label='Original')
    plt.plot(range(timespan), pred_robot_vel[:timespan, 1], label='Reconstructed')
    plt.xlabel('Sample index')
    plt.ylabel('EEF vel Y')
    plt.legend()

    plt.subplot(339)
    plt.plot(range(timespan), robot_vel[:timespan, 2], label='Original')
    plt.plot(range(timespan), pred_robot_vel[:timespan, 2], label='Reconstructed')
    plt.xlabel('Sample index')
    plt.ylabel('EEF vel Z')
    plt.legend()

    plt.show()

    print("Finished")
