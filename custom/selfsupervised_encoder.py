import torch
import os

from robomimic.models.obs_nets import ObservationEncoder, MLP
from robomimic.utils import torch_utils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.tensor_utils as TensorUtils  

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



def setup_encoder():
    obs_encoder = ObservationEncoder(feature_activation=torch.nn.ReLU)

    # Shape of the image coming from the "agentview" camera
    camera1_shape = [84, 84, 3]

    # We will use a reconfigurable image processing backbone VisualCore to process the input image observation key
    net_class = "VisualCore"  # this is defined in models/base_nets.py

    # kwargs for VisualCore network
    net_kwargs = {
        "input_shape": camera1_shape,
        "backbone_class": "ResNet18Conv",  # use ResNet18 as the visualcore backbone
        "backbone_kwargs": {"pretrained": False, "input_coord_conv": False},
        "pool_class": "SpatialSoftmax",  # use spatial softmax to regularize the model output
        "pool_kwargs": {"num_kp": 32}
    }

    # register the network for processing the observation key
    obs_encoder.register_obs_key(
        name="agentview_image",
        shape=camera1_shape,
        net_class=net_class,
        net_kwargs=net_kwargs,
    )

    # We could mix low-dimensional observation, e.g., proprioception signal, in the encoder
    proprio_shape = [3]
    net = MLP(input_dim=3, output_dim=10, layer_dims=(128,), output_activation=None)
    obs_encoder.register_obs_key(
        name="robot0_eef_pos",
        shape=proprio_shape,
        net=net,
    )

    # Before constructing the encoder, make sure we register all of our observation keys with corresponding modalities
    # (this will determine how they are processed during training)
    obs_modality_mapping = {
        "low_dim": ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
        "rgb": ["agentview_image"],
    }
    ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping=obs_modality_mapping)

    # Finally construct the observation encoder
    obs_encoder.make()

    # pretty-print the observation encoder
    print(obs_encoder)

    return obs_encoder

def train_encoder(obs_encoder, inputs):
    # Send to GPU if applicable
    if torch.cuda.is_available():
        inputs = TensorUtils.to_device(inputs, torch.device("cuda:0"))
        obs_encoder.cuda()


if __name__ == "__main__":
    # Download the demonstrations if they are not there
    dataset_path = get_demonstration()

    # set torch device
    device = torch_utils.get_torch_device(try_to_use_cuda=True)
    
    # Setup ResNet18 as visual encoder backbone and MLP for low-dimensional observations
    obs_encoder = setup_encoder()

    # This helps loading the data for training
    data_loader = get_data_loader(dataset_path=dataset_path)

    sample = next(iter(data_loader))

    print(f"The first element of the batch for eef position (10 sample sequence): {sample['obs']['robot0_eef_pos'][0]}")
    # print(f"Labels batch shape: {train_labels.size()}")

    # Training the encoder network
    # train_encoder(encoder=obs_encoder)


