import functools
from typing import Optional, Tuple, Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp
from jaxrl.networks.common import HyperDense, HyperEmbedder, HyperLERPBlock, Params, PRNGKey
from flax import core

tfd = tfp.distributions
tfb = tfp.bijectors

def transform_inputs_no_seeds(
    input_mean: jnp.ndarray,
    input_std: jnp.ndarray,
    inputs: jnp.ndarray,
) -> jnp.ndarray:
    mean = input_mean[None, :]
    std = input_std[None, :]
    return (inputs - mean) / (std + 1e-6)

class WorldModel(nn.Module):
    observation_dim: int
    action_dim: int
    reward_dim: int = 1
    hidden_dims: int = 512
    depth: int = 2
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    log_std_min: Optional[float] = -10.0
    log_std_max: Optional[float] = 1
    latent_size: int = 0

    def setup(self):
        self.input_mean = self.param('input_mean', nn.initializers.zeros, (self.observation_dim + self.action_dim,))
        self.input_std = self.param('input_std', nn.initializers.ones, (self.observation_dim + self.action_dim,))

        # SimbaV2 style -- for more details see that paper
        self.scaler_scale = jnp.sqrt(2.0 / self.hidden_dims)
        self.alpha_init=1.0 / (self.depth + 1)
        self.alpha_scale=1.0 / jnp.sqrt(self.hidden_dims)
        self.c_shift = 1.0
        
        self.embedder = HyperEmbedder(
            hidden_dim=self.hidden_dims,
            scaler_init=self.scaler_scale,
            scaler_scale=self.scaler_scale,
            c_shift=self.c_shift,
        )
        self.pre_encoder = HyperDense(self.hidden_dims)
        self.encoder = nn.Sequential(
            [
                HyperLERPBlock(
                    hidden_dim=self.hidden_dims,
                    scaler_init=self.scaler_scale,
                    scaler_scale=self.scaler_scale,
                    alpha_init=self.alpha_init,
                    alpha_scale=self.alpha_scale,
                )
                for _ in range(self.depth)
            ]
        )

        self.means = HyperDense(self.reward_dim + self.observation_dim)
        self.logstds = HyperDense(self.reward_dim + self.observation_dim)

    def __call__(
        self, obs: jnp.ndarray, actions: jnp.ndarray, latents: jnp.ndarray, return_params: bool = False
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = jnp.concatenate((obs, actions), axis=1)
        x = transform_inputs_no_seeds(self.input_mean, self.input_std, x)
        y = jnp.concatenate((x, latents), axis=1)
        y = self.pre_encoder(y)
        z = self.encoder(y)
        means = self.means(z)
        logstds = self.logstds(z)
        logvar = self.log_std_min + (self.log_std_max - self.log_std_min) * 0.5 * (1 + nn.tanh(logstds))

        mean_r = means[..., :self.reward_dim]
        mean_s = means[..., self.reward_dim:]
        
        if return_params is False:
            return mean_r, mean_s
        else:
            return means, logvar

@functools.partial(jax.jit, static_argnames=("model_def", "latent_size", "num_latent_codes"))
@functools.partial(jax.vmap, in_axes=(0, None, 0, 0, 0, None, None))
def _predict_many(
    rng: PRNGKey,
    model_def: nn.Module,
    model_params: Params,
    observations: np.ndarray,
    actions: np.ndarray,
    latent_size: int = 0,
    num_latent_codes: int = 10
) -> Tuple[PRNGKey, jnp.ndarray]:
    rng, key_latent = jax.random.split(rng, 2)
    latents = jax.random.normal(key_latent, shape=(observations.shape[0] * num_latent_codes, latent_size))
    observations_expanded = jnp.repeat(observations[:, None, :], num_latent_codes, axis=1).reshape(-1, observations.shape[-1])
    actions_expanded = jnp.repeat(actions[:, None, :], num_latent_codes, axis=1).reshape(-1, actions.shape[-1])
    
    reward_predictions, next_obs_predictions = model_def.apply({"params": model_params}, observations_expanded, actions_expanded, latents)
    rewards = reward_predictions.reshape(observations.shape[0], num_latent_codes, -1)
    next_obs = next_obs_predictions.reshape(observations.shape[0], num_latent_codes, -1)

    next_obs = next_obs + observations[:, None, :]
    
    return rng, rewards, next_obs

def predict_many(
    rng: PRNGKey,
    model_def: nn.Module,
    model_params: Params,
    observations: np.ndarray,
    actions: np.ndarray,
    num_latent_codes: int = 10
) -> Tuple[PRNGKey, jnp.ndarray]:
    return _predict_many(
        rng, 
        model_def, 
        model_params, 
        observations, 
        actions, 
        model_def.latent_size,
        num_latent_codes
    )

def update_input_scaler(
    model: WorldModel,
    inputs: jnp.ndarray,
) -> WorldModel:
    """Update the input scaler parameters of the model.
    
    Args:
        model: The model to update
        inputs: Input data to compute statistics from [batch_size, input_dim]
    
    Returns:
        Updated model with new scaler parameters
    """
    # Compute mean and std of inputs
    mean = jnp.mean(inputs, axis=1, keepdims=True)
    std = jnp.std(inputs, axis=1, keepdims=True)
    std = jnp.where(std < 1e-12, 1.0, std)
    
    # Update the model parameters
    new_params = dict(model.params)
    new_params['input_mean'] = mean.squeeze()
    new_params['input_std'] = std.squeeze()
    
    return model.replace(params=new_params)

def transform_inputs(
    input_mean: jnp.ndarray,
    input_std: jnp.ndarray,
    inputs: jnp.ndarray,
) -> jnp.ndarray:
    # inputs shape: (num_seeds, num_batch, input_dim)
    # input_mean shape: (num_seeds, input_dim)
    # We need to add a dimension for broadcasting over the batch dimension
    mean = input_mean[:, None, :]  # Shape: (num_seeds, 1, input_dim)
    std = input_std[:, None, :]    # Shape: (num_seeds, 1, input_dim)
    return (inputs - mean) / (std + 1e-6)