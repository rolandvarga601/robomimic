from robomimic.config.base_config import BaseConfig

class VAE_REPConfig(BaseConfig):
    ALGO_NAME = "vae_rep"

    def algo_config(self):

          # optimization parameters
        self.algo.optim_params.policy.learning_rate.initial = 1e-2      # policy learning rate --- Human demo
        # self.algo.optim_params.policy.learning_rate.initial = 1e-3      # policy learning rate --- Machine generated demo
        # self.algo.optim_params.policy.learning_rate.decay_factor = 0.1  # factor to decay LR by (if epoch schedule non-empty)
        # self.algo.optim_params.policy.learning_rate.epoch_schedule = [200, 400, 600, 800] # epochs where LR decay occurs
        self.algo.optim_params.policy.learning_rate.decay_factor = 0.5  # factor to decay LR by (if epoch schedule non-empty)
        self.algo.optim_params.policy.learning_rate.epoch_schedule = range(10, 400, 10) # epochs where LR decay occurs
        self.algo.optim_params.policy.regularization.L2 = 0.00          # L2 regularization strength

        # stochastic VAE policy settings
        self.algo.vae.enabled = True                   # whether to train a VAE policy
        self.algo.vae.latent_dim = 14                    # VAE latent dimension - set to twice the dimensionality of action space
        self.algo.vae.latent_clip = None                # clip latent space when decoding (set to None to disable)
        # self.algo.vae.kl_weight = 1.                    # beta-VAE weight to scale KL loss relative to reconstruction loss in ELBO
        self.algo.vae.kl_weight = 0.001                    # beta-VAE weight to scale KL loss relative to reconstruction loss in ELBO
        # self.algo.vae.kl_weight = 0.04                    # beta-VAE weight to scale KL loss relative to reconstruction loss in ELBO

        # VAE decoder settings
        self.algo.vae.decoder.is_conditioned = True                         # whether decoder should condition on observation
        self.algo.vae.decoder.reconstruction_sum_across_elements = False    # sum instead of mean for reconstruction loss

        # VAE prior settings
        self.algo.vae.prior.learn = False                                   # learn Gaussian / GMM prior instead of N(0, 1)
        self.algo.vae.prior.is_conditioned = False                          # whether to condition prior on observations
        self.algo.vae.prior.use_gmm = False                                 # whether to use GMM prior
        self.algo.vae.prior.gmm_num_modes = 10                              # number of GMM modes
        self.algo.vae.prior.gmm_learn_weights = False                       # whether to learn GMM weights 
        self.algo.vae.prior.use_categorical = False                         # whether to use categorical prior
        self.algo.vae.prior.categorical_dim = 10                            # the number of categorical classes for each latent dimension
        self.algo.vae.prior.categorical_gumbel_softmax_hard = False         # use hard selection in forward pass
        self.algo.vae.prior.categorical_init_temp = 1.0                     # initial gumbel-softmax temp
        self.algo.vae.prior.categorical_temp_anneal_step = 0.001            # linear temp annealing rate
        self.algo.vae.prior.categorical_min_temp = 0.3                      # lowest gumbel-softmax temp

        self.algo.vae.encoder_layer_dims = (34, 22, 16)                          # encoder MLP layer dimensions
        self.algo.vae.decoder_layer_dims = (16, 22, 34)                          # decoder MLP layer dimensions
        self.algo.vae.prior_layer_dims = (34, 22, 16)                            # prior MLP layer dimensions (if learning conditioned prior)

    def observation_config(self):
        """
        This function populates the `config.observation` attribute of the config, and is given 
        to the `Algo` subclass (see `algo/algo.py`) for each algorithm through the `obs_config` 
        argument to the constructor. This portion of the config is used to specify what 
        observation modalities should be used by the networks for training, and how the 
        observation modalities should be encoded by the networks. While this class has a 
        default implementation that usually doesn't need to be overriden, certain algorithm 
        configs may choose to, in order to have seperate configs for different networks 
        in the algorithm. 
        """

        super(VAE_REPConfig, self).observation_config()

        # observation modalities
        self.observation.modalities.obs.low_dim = [             # specify low-dim observations for agent
            # "robot0_eef_force",
            "robot0_eef_pos", 
            "robot0_eef_quat",
            # "robot0_eef_vel_ang",
            # "robot0_eef_vel_lin",
            "robot0_gripper_qpos",
            # "robot0_gripper_qvel", 
            "object",
        ]