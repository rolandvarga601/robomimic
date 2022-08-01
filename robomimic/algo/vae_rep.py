"""
Implementation of Variational AutoEncoder (VAE) representation learning.
"""
from collections import OrderedDict

import torch
import torch.nn as nn

import robomimic.models.vae_nets as VAENets
import robomimic.utils.obs_utils as ObsUtils
from robomimic.algo import register_algo_factory_func, Algo, PolicyAlgo

@register_algo_factory_func("vae_rep")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the VAE_REP algo class to instantiate, along with additional algo kwargs.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """
    # only one variant of VAE_REP for now
    return VAE_REP, {}

class VAE_REP(Algo):
    """
    BC training with a VAE policy.
    """
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        self.nets = nn.ModuleDict()
        self.nets["policy"] = VAENets.VAE(
            input_shapes=self.obs_shapes,
            output_shapes=self.obs_shapes,
            device=self.device,
            condition_shapes=None,
            output_squash=(),
            output_scales=None,
            output_ranges=None,
            goal_shapes=None,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
            **VAENets.vae_args_from_config(self.algo_config.vae),
            # encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(config["observation"]["encoder"]),
        )
        
        self.nets = self.nets.float().to(self.device)

    def train_on_batch(self, batch, epoch, validate=False):
        """
        Update from superclass to set categorical temperature, for categorical VAEs.
        """
        if self.algo_config.vae.prior.use_categorical:
            temperature = self.algo_config.vae.prior.categorical_init_temp - epoch * self.algo_config.vae.prior.categorical_temp_anneal_step
            temperature = max(temperature, self.algo_config.vae.prior.categorical_min_temp)
            self.nets["policy"].set_gumbel_temperature(temperature)
        return super(VAE_REP, self).train_on_batch(batch, epoch, validate=validate)

    def _forward_training(self, batch):
        """
        Internal helper function for BC algo class. Compute forward pass
        and return network outputs in @predictions dict.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            predictions (dict): dictionary containing network outputs
        """
        vae_inputs = dict(
            actions=batch["actions"],
            obs_dict=batch["obs"],
            goal_dict=batch["goal_obs"],
            freeze_encoder=batch.get("freeze_encoder", False),
        )

        vae_outputs = self.nets["policy"].forward_train(**vae_inputs)
        predictions = OrderedDict(
            actions=vae_outputs["decoder_outputs"],
            kl_loss=vae_outputs["kl_loss"],
            reconstruction_loss=vae_outputs["reconstruction_loss"],
            encoder_z=vae_outputs["encoder_z"],
        )
        if not self.algo_config.vae.prior.use_categorical:
            with torch.no_grad():
                encoder_variance = torch.exp(vae_outputs["encoder_params"]["logvar"])
            predictions["encoder_variance"] = encoder_variance
        return predictions

    def _compute_losses(self, predictions, batch):
        """
        Internal helper function for BC algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.

        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            losses (dict): dictionary of losses computed over the batch
        """

        # total loss is sum of reconstruction and KL, weighted by beta
        kl_loss = predictions["kl_loss"]
        recons_loss = predictions["reconstruction_loss"]
        action_loss = recons_loss + self.algo_config.vae.kl_weight * kl_loss
        return OrderedDict(
            recons_loss=recons_loss,
            kl_loss=kl_loss,
            action_loss=action_loss,
        )

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = PolicyAlgo.log_info(self, info)
        # log["Loss"] = info["losses"]["action_loss"].item()
        # log["KL_Loss"] = info["losses"]["kl_loss"].item()
        # log["Reconstruction_Loss"] = info["losses"]["recons_loss"].item()
        # if self.algo_config.vae.prior.use_categorical:
        #     log["Gumbel_Temperature"] = self.nets["policy"].get_gumbel_temperature()
        # else:
        #     log["Encoder_Variance"] = info["predictions"]["encoder_variance"].mean().item()
        # if "policy_grad_norms" in info:
        #     log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log