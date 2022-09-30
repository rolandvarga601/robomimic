from math import floor
import numpy as np
from typing import OrderedDict
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



def setup_encoder(dataset_path):

    config = config_factory(algo_name="vae_rep")

    # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
    ObsUtils.initialize_obs_utils_with_config(config)

    # Before constructing the encoder, make sure we register all of our observation keys with corresponding modalities
    # (this will determine how they are processed during training)
    # obs_modality_mapping = {
    #     "low_dim": ["robot0_eef_pos", 
    #         "robot0_eef_quat", 
    #         "robot0_eef_vel_ang",
    #         "robot0_eef_vel_lin",
    #         "robot0_gripper_qpos", 
    #         "robot0_gripper_qvel",
    #         "robot0_eef_force", 
    #         "robot0_joint_pos",
    #         "robot0_joint_pos_cos",
    #         "robot0_joint_pos_sin",
    #         "object", ],
    #     "rgb": ["agentview_image"],
    # }
    # ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping=obs_modality_mapping)

    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=dataset_path,
        all_obs_keys=sorted((
            "robot0_eef_force",
            "robot0_eef_pos", 
            "robot0_eef_quat",
            "robot0_eef_vel_ang",
            "robot0_eef_vel_lin",
            "robot0_gripper_qpos",
            "robot0_gripper_qvel", 
            "object",
        )),
    )

    input_shapes = shape_meta["all_shapes"]

    # encoder_layer_dims = [20, 10, 5]
    # decoder_layer_dims = encoder_layer_dims.reverse()
    
    # latent_dim = encoder_layer_dims[-1]

    # set torch device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)


    vae = VAENets.VAE(
        input_shapes=input_shapes,
        output_shapes=input_shapes,
        device=device,
        condition_shapes=None,
        output_squash=(),
        output_scales=None,
        output_ranges=None,
        goal_shapes=None,
        encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(config["observation"]["encoder"]),
        **VAENets.vae_args_from_config(config["algo"]["vae"]),
        # encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(config["observation"]["encoder"]),
    )
    
    # vae = VAE(
    #     input_shapes=config,
    #     output_shapes=input_shapes,
    #     encoder_layer_dims=encoder_layer_dims,
    #     decoder_layer_dims=decoder_layer_dims,
    #     latent_dim=latent_dim,
    #     device=device,
    #     condition_shapes=None,
    #     decoder_is_conditioned=False,
    #     decoder_reconstruction_sum_across_elements=False,
    #     latent_clip=None,
    #     output_squash=(),
    #     output_scales=None,
    #     output_ranges=None,
    #     prior_learn=False,
    #     prior_is_conditioned=False,
    #     prior_layer_dims=(),
    #     prior_use_gmm=False,
    #     prior_gmm_num_modes=10,
    #     prior_gmm_learn_weights=False,
    #     prior_use_categorical=False,
    #     prior_categorical_dim=10,
    #     prior_categorical_gumbel_softmax_hard=False,
    #     goal_shapes=None,
    #     encoder_kwargs=None,
    #     encoder_kwargs=None,
    # )

    print(vae)

    return vae

    # obs_encoder = ObservationEncoder(feature_activation=torch.nn.ReLU)

    # # Shape of the image coming from the "agentview" camera
    # camera1_shape = [84, 84, 3]

    # # We will use a reconfigurable image processing backbone VisualCore to process the input image observation key
    # net_class = "VisualCore"  # this is defined in models/base_nets.py

    # # kwargs for VisualCore network
    # net_kwargs = {
    #     "input_shape": camera1_shape,
    #     "backbone_class": "ResNet18Conv",  # use ResNet18 as the visualcore backbone
    #     "backbone_kwargs": {"pretrained": False, "input_coord_conv": False},
    #     "pool_class": "SpatialSoftmax",  # use spatial softmax to regularize the model output
    #     "pool_kwargs": {"num_kp": 32}
    # }

    # # register the network for processing the observation key
    # obs_encoder.register_obs_key(
    #     name="agentview_image",
    #     shape=camera1_shape,
    #     net_class=net_class,
    #     net_kwargs=net_kwargs,
    # )

    # # We could mix low-dimensional observation, e.g., endeffector position signal, in the encoder
    # eef_pos_shape = [3]
    # net = MLP(input_dim=17, output_dim=32, layer_dims=(32,64,128,), output_activation=None)
    # obs_encoder.register_obs_key(
    #     name="proprioception",
    #     shape=eef_pos_shape,
    #     net=net,
    # )

    # # Before constructing the encoder, make sure we register all of our observation keys with corresponding modalities
    # # (this will determine how they are processed during training)
    # obs_modality_mapping = {
    #     "low_dim": ["robot0_eef_pos", 
    #         "robot0_eef_quat", 
    #         "robot0_eef_vel_ang",
    #         "robot0_eef_vel_lin",
    #         "robot0_gripper_qpos", 
    #         "robot0_gripper_qvel", 
    #         "object", ],
    #     "rgb": ["agentview_image"],
    # }
    # ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping=obs_modality_mapping)

    # # Finally construct the observation encoder
    # obs_encoder.make()

    # # pretty-print the observation encoder
    # print(obs_encoder)

    # return obs_encoder

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
            "robot0_eef_force",
            "robot0_eef_pos", 
            "robot0_eef_quat",
            "robot0_eef_vel_ang",
            "robot0_eef_vel_lin",
            "robot0_gripper_qpos",
            "robot0_gripper_qvel", 
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

def run_train_loop(model, data_loader):
    """
    Note: this is a stripped down version of @TrainUtils.run_epoch and the train loop
    in the train function in train.py. Logging and evaluation rollouts were removed.

    Args:
        model (Algo instance): instance of Algo class to use for training
        data_loader (torch.utils.data.DataLoader instance): torch DataLoader for
            sampling batches
    """
    num_epochs = 200
    gradient_steps_per_epoch = 100
    has_printed_batch_info = False

    # ensure model is in train mode
    model.set_train()

    # set download folder
    ckpt_folder = os.path.join(os.path.dirname(__file__), "ckpt")
    os.makedirs(ckpt_folder, exist_ok=True)

    for epoch in range(1, num_epochs + 1): # epoch numbers start at 1

        # iterator for data_loader - it yields batches
        data_loader_iter = iter(data_loader)

        # record losses
        losses = []
        losses_recon = []
        losses_kl = []

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

            # process batch for training
            input_batch = model.process_batch_for_training(batch)

            # forward and backward pass
            info = model.train_on_batch(batch=input_batch, epoch=epoch, validate=False)

            # record loss
            step_log = model.log_info(info)
            losses.append(step_log["Loss"])
            losses_recon.append(step_log["Reconstruction_Loss"])
            losses_kl.append(step_log["KL_Loss"])

        # save model
        model_params = model.serialize()
        model_dict = dict(model=model.serialize())
        torch.save(model_dict, os.path.join(ckpt_folder, f"epoch{epoch}" + ".pth"))

        # do anything model needs to after finishing epoch
        model.on_epoch_end(epoch)

        print("Train Epoch {}: Loss {}      Reconstruction loss: {}           KL loss: {}".format(epoch, np.mean(losses), np.mean(losses_recon), np.mean(losses_kl)))


if __name__ == "__main__":
    # Download the demonstrations if they are not there
    # dataset_path = get_demonstration()

    dataset_path = os.path.join(os.path.dirname(__file__), "data", "extended_low_dim.hdf5")
    assert os.path.exists(dataset_path)

    # set torch device
    device = torch_utils.get_torch_device(try_to_use_cuda=True)
    
    # Setup VAE
    # obs_encoder = setup_encoder(dataset_path=dataset_path)
    VAE_model = get_model(dataset_path=dataset_path, device=device)

    # This helps loading the data for training
    data_loader = get_data_loader(dataset_path=dataset_path, seq_length=1)

    sample = next(iter(data_loader))

    print(f"The first element of the batch for eef position (1 sample sequence): {sample['obs']['robot0_eef_pos'][0]}")
    
    # for sensor in sample['obs']:
    #     print(sample['obs']["{}".format(sensor)][0])

    # print(f"Labels batch shape: {train_labels.size()}")

    # Training the encoder network
    # train_encoder(encoder=obs_encoder)

    # run train loop
    run_train_loop(model=VAE_model, data_loader=data_loader)


