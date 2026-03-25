from typing import Optional, Tuple

import jax
import jax.numpy as jnp

import functools

from jaxrl.replay_buffer import Batch
from jaxrl.networks.common import InfoDict, Model
from jaxrl.networks.common import PRNGKey

def huber_replace(td_errors, kappa: float = 1.0):
    return jnp.where(jnp.absolute(td_errors) <= kappa, 0.5 * td_errors ** 2, kappa * (jnp.absolute(td_errors) - 0.5 * kappa))

# Toggle these two lines to switch loss type (Huber vs MSE)
USE_HUBER_WORLD = True
USE_HUBER_NN = True
KAPPA = 1.0

def data_fit_loss(residuals, inv_var, use_huber: bool):
    if use_huber:
        return jnp.mean(jnp.mean(huber_replace(residuals, kappa=KAPPA) * inv_var, axis=-1))
    else:
        return jnp.mean(jnp.mean((residuals ** 2) * inv_var, axis=-1))


@functools.partial(jax.jit, static_argnames=('num_latent_codes', 'latent_size', 'staleness'))
def update_world_model(rng: PRNGKey, inp_world_model: Model, batch: Batch, num_latent_codes: int = 10, latent_size: int = 32, staleness: int = 4) -> Tuple[Model, InfoDict]:
    """Update world model using MSE + variance regularization loss like MBPO."""

    labels = jnp.concatenate([batch.rewards[:, None], batch.next_observations - batch.observations], axis=-1)
    latents = jax.random.normal(rng, shape=(batch.observations.shape[0], num_latent_codes, latent_size))
    zs = find_nearest_neighbors(inp_world_model, batch.observations, batch.actions, labels, latents, num_latent_codes)

    def update_world_model(i, state):
        cur_world_model, info = state
        def world_model_loss_fn(world_model_params):
            # Get model predictions (means and log_stds)
            means, logvar = cur_world_model.apply(
                {'params': world_model_params},
                batch.observations,
                batch.actions,
                zs,
                return_params=True,
            )
            inv_var = jnp.exp(-logvar)
            residuals = means - labels
            
            # The IMLE network outputs a mean and a logvar, one could change this to use the mean but we have found that the variance regularization stabilizes the training
            # Variance regularization term
            var_loss = jnp.mean(jnp.mean(logvar, axis=-1))
            
            # Data-fit losses (both, for logging); choose one for training via toggle
            huber_loss = jnp.mean(jnp.mean(huber_replace(residuals, kappa=KAPPA) * inv_var, axis=-1))
            mse_loss = jnp.mean(jnp.mean((means - labels) ** 2 * inv_var, axis=-1))
            data_loss = huber_loss if USE_HUBER_WORLD else mse_loss
            total_loss = data_loss + var_loss
            
            # Total loss
            var_loss = jnp.mean(jnp.mean(logvar, axis=-1))
            
            return total_loss, {
                'world_model_loss': total_loss,
                'world_model_huber_loss': huber_loss,
                'world_model_mse_loss': mse_loss,
                'world_model_var_loss': var_loss,
                'world_model_unscaled_mse_loss': jnp.mean(jnp.mean((means - labels) ** 2, axis=-1)),
            }
        
        new_world_model, info = cur_world_model.apply_gradient(world_model_loss_fn)
        info['world_model_gnorm'] = info.pop('grad_norm')
        return new_world_model, info

    world_model, info = update_world_model(0, (inp_world_model, {}))
    return jax.lax.fori_loop(1, staleness, update_world_model, (world_model, info))

@functools.partial(jax.vmap, in_axes=(0, 0, 0, None, None))
def update_world_model_vmap(rng: PRNGKey, world_model: Model, batches: Batch, num_latent_codes: int = 10, latent_size: int = 32) -> Tuple[Model, InfoDict]:
    def one_step(i, state):
        world_model, info = state
        new_world_model, new_info = update_world_model(
            rng,
            world_model, 
            jax.tree_map(lambda x: jnp.take(x, i, axis=0), batches),
            num_latent_codes,
            latent_size
        )
        return new_world_model, new_info

    world_model, info = one_step(0, (world_model, {}))
    return jax.lax.fori_loop(1, batches.observations.shape[0], one_step, (world_model, info))

@functools.partial(jax.jit, static_argnames=('num_latent_codes'))
def find_nearest_neighbors(
    model: Model,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    targets: jnp.ndarray,
    latents: jnp.ndarray,
    num_latent_codes: int = 10,
) -> jnp.ndarray:
    """Find nearest neighbor latent codes using the exact model loss."""
    # Use stop_gradient to prevent gradient computation for the entire function
    def no_grad_find_neighbors(model, observations, actions, targets, latents):
        # Sample latent codes
        num_batches = observations.shape[0]
        observations_expanded = jnp.repeat(observations[:, None, :], num_latent_codes, axis=1).reshape(-1, observations.shape[-1])
        actions_expanded = jnp.repeat(actions[:, None, :], num_latent_codes, axis=1).reshape(-1, actions.shape[-1])
        latents_flat = latents.reshape(-1, latents.shape[-1])
        
        # Get model predictions without computing gradients (means and logvar)
        means, logvar = model.apply({'params': model.params}, observations_expanded, actions_expanded, latents_flat, return_params=True)
        means = means.reshape(num_batches, num_latent_codes, -1)
        logvar = logvar.reshape(num_batches, num_latent_codes, -1)
        
        # Compute total loss (MSE + variance regularization) without gradients
        inv_var = jnp.exp(-logvar)
        residuals = means - targets[:, None, :]
        huber_loss = jnp.mean(huber_replace(residuals, kappa=KAPPA) * inv_var, axis=-1)
        mse_loss = jnp.mean((residuals ** 2) * inv_var, axis=-1)
        var_loss = jnp.mean(logvar, axis=-1)
        data_loss = huber_loss if USE_HUBER_NN else mse_loss
        total_loss = data_loss + var_loss
        
        best_indices = jnp.argmin(total_loss, axis=1)
        best_latents = jnp.take_along_axis(latents, best_indices[:, None, None], axis=1)
        best_latents = jnp.squeeze(best_latents, axis=1)
        
        return best_latents
    
    # Apply stop_gradient to the entire function
    return jax.lax.stop_gradient(no_grad_find_neighbors(model, observations, actions, targets, latents))
