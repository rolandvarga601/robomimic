from cProfile import label
from math import floor
import numpy as np
from typing import Dict, OrderedDict
import torch
import os

import robomimic.models.vae_nets as VAENets
from robomimic.utils import torch_utils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.tensor_utils as TensorUtils  
import robomimic.utils.torch_utils as TorchUtils
from robomimic.config import config_factory
from robomimic.algo import algo_factory

from utils import get_data_loader

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TKAgg')

# the dataset registry can be found at robomimic/__init__.py
from robomimic import DATASET_REGISTRY

def get_demonstration():
    # set download folder
    download_folder = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(download_folder, exist_ok=True)

    # download the dataset
    task = "can"
    dataset_type = "mh"     # Contains also not proficient human demonstrations
    hdf5_type = "image"

    # If file exists, do not download
    dataset_path = os.path.join(download_folder, "image.hdf5")
    if not os.path.exists(dataset_path):
        FileUtils.download_url(
            url=DATASET_REGISTRY[task][dataset_type][hdf5_type]["url"], 
            download_dir=download_folder,
        )
    else:
        print(f"INFO: File {dataset_path} already exists, it is not overwritten.")

    # enforce that the dataset exists
    assert os.path.exists(dataset_path)

    return dataset_path


def get_model(dataset_path, device):
    """
    Use a default config to construct a VAE representation model.
    """

    # default BC config
    config = config_factory(algo_name="vae_rep")

    # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
    ObsUtils.initialize_obs_utils_with_config(config)

    # read dataset to get some metadata for constructing model
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=dataset_path, 
        all_obs_keys=sorted((
            # "robot0_eef_force",
            # "robot0_eef_pos", 
            # "robot0_eef_quat",
            # "robot0_eef_vel_ang",
            # "robot0_eef_vel_lin",
            "robot0_gripper_qpos",
            # "robot0_gripper_qvel", 
            "object",
        )),
    )

    # make VAE model
    model = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=shape_meta["all_shapes"],
        ac_dim=shape_meta["ac_dim"],
        device=device,
    )
    return model

def print_batch_info(batch):
    print("\n============= Batch Info =============")
    for k in batch:
        if k in ["obs", "next_obs"]:
            print("key {}".format(k))
            for obs_key in batch[k]:
                print("    obs key {} with shape {}".format(obs_key, batch[k][obs_key].shape))
        else:
            print("key {} with shape {}".format(k, batch[k].shape))
    print("")

def run_train_loop(model, data_loader, data_loader_valid=None):
    """
    Note: this is a stripped down version of @TrainUtils.run_epoch and the train loop
    in the train function in train.py. Logging and evaluation rollouts were removed.

    Args:
        model (Algo instance): instance of Algo class to use for training
        data_loader (torch.utils.data.DataLoader instance): torch DataLoader for
            sampling batches
    """
    num_epochs = 100
    gradient_steps_per_epoch = 50
    has_printed_batch_info = False

    # Machine generated dataset parameters
    # num_epochs = 20
    # gradient_steps_per_epoch = 200
    # has_printed_batch_info = False


    # ensure model is in train mode
    model.set_train()

    # set download folder
    ckpt_folder = os.path.join(os.path.dirname(__file__), "ckpt")
    os.makedirs(ckpt_folder, exist_ok=True)

    loss_history = dict()
    loss_history['overall'] = []
    loss_history['reconstruction'] = []
    loss_history['kl'] = []

    if data_loader_valid is not None:
        loss_history['overall_valid'] = []
        loss_history['reconstruction_valid'] = []
        loss_history['kl_valid'] = []


    for epoch in range(1, num_epochs + 1): # epoch numbers start at 1

        # iterator for data_loader - it yields batches
        data_loader_iter = iter(data_loader)

        if data_loader_valid is not None:
            data_loader_valid_iter = iter(data_loader_valid)

        # record losses
        losses = []
        losses_recon = []
        losses_kl = []

        if data_loader_valid is not None:
            # record losses
            valid_losses = []
            valid_losses_recon = []
            valid_losses_kl = []

        for _ in range(gradient_steps_per_epoch):

            # load next batch from data loader
            try:
                batch = next(data_loader_iter)
            except StopIteration:
                # data loader ran out of batches - reset and yield first batch
                data_loader_iter = iter(data_loader)
                batch = next(data_loader_iter)

            if not has_printed_batch_info:
                has_printed_batch_info = True
                print_batch_info(batch)

            # batch['obs']['robot0_eef_force'] = batch['obs']['robot0_eef_force']/1.0
            # batch['next_obs']['robot0_eef_force'] = batch['next_obs']['robot0_eef_force']/1.0

            # process batch for training
            input_batch = model.process_batch_for_training(batch)

            # forward and backward pass
            info = model.train_on_batch(batch=input_batch, epoch=epoch, validate=False)

            # record loss
            step_log = model.log_info(info)
            losses.append(step_log["Loss"])
            losses_recon.append(step_log["Reconstruction_Loss"])
            losses_kl.append(step_log["KL_Loss"])

        if (epoch+1) % 10 == 0:
            # save model
            model_params = model.serialize()
            model_dict = dict(model=model.serialize())
            torch.save(model_dict, os.path.join(ckpt_folder, f"epoch{epoch}" + ".pth"))

        # do anything model needs to after finishing epoch
        model.on_epoch_end(epoch)

        if data_loader_valid is not None:
            for _ in range(5):

                # load next batch from data loader
                try:
                    batch = next(data_loader_valid_iter)
                except StopIteration:
                    # data loader ran out of batches - reset and yield first batch
                    data_loader_valid_iter = iter(data_loader_valid)
                    batch = next(data_loader_valid_iter)

                obs_norms = []
                for k in range(data_loader_valid.batch_size):
                    obs_dict = {key: torch.unsqueeze(batch["obs"][key][k, 0, :], 0) for key in batch["obs"]}
                    obs_norms.append(ObsUtils.normalize_obs(obs_dict=obs_dict, obs_normalization_stats=data_loader.dataset.get_obs_normalization_stats()))
                
                batch["obs"] = {key: torch.cat(tuple(torch.unsqueeze(o[key], 0) for o in obs_norms), 0) for key in obs_dict}

                # process batch for training
                input_batch = model.process_batch_for_training(batch)

                # for k in range(data_loader_valid.batch_size):
                #     obs_dict = dict()
                #     for m in input_batch["obs"]:
                #         obs_dict[m] = input_batch["obs"][m][k,:]
                #     input_batch["obs"] = ObsUtils.normalize_obs(obs_dict=obs_dict, obs_normalization_stats=data_loader.dataset.get_obs_normalization_stats())

                # forward and backward pass
                info = model.train_on_batch(batch=input_batch, epoch=epoch, validate=True)

                # record loss
                step_log = model.log_info(info)
                valid_losses.append(step_log["Loss"])
                valid_losses_recon.append(step_log["Reconstruction_Loss"])
                valid_losses_kl.append(step_log["KL_Loss"])


        loss_history["overall"].append(np.mean(losses))
        loss_history["reconstruction"].append(np.mean(losses_recon))
        loss_history["kl"].append(np.mean(losses_kl))

        if data_loader_valid is not None:
            loss_history["overall_valid"].append(np.mean(valid_losses))
            loss_history["reconstruction_valid"].append(np.mean(valid_losses_recon))
            loss_history["kl_valid"].append(np.mean(valid_losses_kl))

        
        if data_loader_valid is None:
            print(f"Train Epoch {epoch}: Loss {np.mean(losses)} Rec loss: {np.mean(losses_recon)} KL loss: {np.mean(losses_kl)}")
        else:
            print(f"Train Epoch {epoch}: Loss {np.mean(losses)} Rec loss: {np.mean(losses_recon)} KL loss: {np.mean(losses_kl)}" + " "*5 + 
                    f"Valid loss {np.mean(valid_losses)} Valid Rec loss: {np.mean(valid_losses_recon)} Valid KL loss: {np.mean(valid_losses_kl)}")

    return loss_history


if __name__ == "__main__":
    # Download the demonstrations if they are not there
    # dataset_path = get_demonstration()

    # dataset_path = os.path.join(os.path.dirname(__file__), "data", "extended_low_dim_shaped.hdf5")
    dataset_path='/home/rvarga/implementation/robomimic/datasets/lift/mg/low_dim_shaped_donemode0.hdf5'
    # dataset_path = os.path.join(os.path.dirname(__file__), "data", "mg_low_dim_extended_shaped.hdf5")
    assert os.path.exists(dataset_path)

    # set torch device
    device = torch_utils.get_torch_device(try_to_use_cuda=True)

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Setup VAE
    # obs_encoder = setup_encoder(dataset_path=dataset_path)
    VAE_model = get_model(dataset_path=dataset_path, device=device)

    # This helps loading the data for training
    data_loader_train = get_data_loader(dataset_path=dataset_path, seq_length=1, normalize_obs=True, filter_key="train")
    # data_loader_train = get_data_loader(dataset_path=dataset_path, seq_length=1, normalize_obs=True)
    data_loader_valid = get_data_loader(dataset_path=dataset_path, seq_length=1, normalize_obs=False, filter_key="valid")

    sample = next(iter(data_loader_train))

    # print(f"The first element of the batch for eef position (1 sample sequence): {sample['obs']['robot0_eef_pos'][0]}")
    
    # for sensor in sample['obs']:
    #     print(sample['obs']["{}".format(sensor)][0])

    # print(f"Labels batch shape: {train_labels.size()}")

    # Training the encoder network
    # train_encoder(encoder=obs_encoder)

    # run train loop
    loss_history = run_train_loop(model=VAE_model, data_loader=data_loader_train, data_loader_valid=data_loader_valid)

    ax1 = plt.subplot(212)
    ax1.plot(loss_history["overall"], label='training')
    ax1.plot(loss_history["overall_valid"], label='validation')
    ax1.set_xlabel('Epoch')
    ax1.set_title('Loss')
    ax1.set_yscale('log')
    ax1.legend()

    ax2 = plt.subplot(221)
    ax2.plot(loss_history["reconstruction"], label='training')
    ax2.plot(loss_history["reconstruction_valid"], label='validation')
    ax2.set_xlabel('Epoch')
    ax2.set_title('Reconstruction loss')
    ax2.set_yscale('log')
    ax2.legend()
    
    ax3 = plt.subplot(222)
    ax3.plot(loss_history["kl"], label='training')
    ax3.plot(loss_history["kl_valid"], label='validation')
    ax3.set_xlabel('Epoch')
    ax3.set_title('KL loss')
    ax3.set_yscale('log')
    ax3.legend()

    plt.show()

    print("Finished")




