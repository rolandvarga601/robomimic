from cProfile import label
import os
from robomimic.utils import tensor_utils as TensorUtils
from visual_utils.training import get_data_loader
from visual_utils.encoding import load_observer

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

if __name__ == "__main__":

    encoder = load_observer("/home/rvarga/implementation/robomimic/custom/ckpt/epoch199.pth")
    encoder.set_eval()

    expert_data_path = "/home/rvarga/implementation/robomimic/custom/data/extended_low_dim_shaped.hdf5"

    assert os.path.exists(expert_data_path)

    data_loader = get_data_loader(dataset_path=expert_data_path, seq_length=1)

    data_loader_iterator = iter(data_loader)

    batch = next(data_loader_iterator)

    batch['obs']['robot0_eef_force'] = batch['obs']['robot0_eef_force']/5.0
    batch['next_obs']['robot0_eef_force'] = batch['next_obs']['robot0_eef_force']/5.0

    input_batch = encoder.process_batch_for_training(batch)

    info = encoder.train_on_batch(batch=input_batch, epoch=1, validate=True)

    robot_positions = input_batch['obs']['robot0_eef_pos'].cpu().detach().numpy()
    pred_robot_positions = info['predictions']['actions']['robot0_eef_pos'].cpu().detach().numpy()

    plt.plot(range(100), robot_positions[:100, :], label='Original')
    plt.plot(range(100), pred_robot_positions[:100, :], label='Reconstructed')
    # plt.plot(range(signal.shape[0]), signal)
    plt.xlabel('Sample index')
    plt.ylabel('EEF position')

    plt.legend()

    plt.show()

    print("Finished")
