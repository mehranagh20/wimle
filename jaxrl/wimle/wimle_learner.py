import functools
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jaxrl.wimle.actor import update as update_actor
from jaxrl.wimle import temperature

from jaxrl.wimle.critic import update_quantile as update_critic_quantile, target_update
from jaxrl.wimle.imle_learner import update_world_model_vmap

from jaxrl.replay_buffer import Batch
from jaxrl.networks import critic_net, policies
from jaxrl.networks.common import InfoDict, Model, PRNGKey
from jaxrl.replay_buffer import ParallelReplayBuffer

from jaxrl.networks.world_model_net import WorldModel, update_input_scaler, predict_many


@functools.partial(jax.vmap, in_axes=(0, 0, 0, 0, 0, 0, None, None, None, None))
def _update(
    rng: PRNGKey, 
    actor: Model, 
    critic: Model, 
    target_critic: Model, 
    temp: Model, 
    batch: Batch, 
    discount: float, 
    tau: float, 
    target_entropy: float, 
    quantile_taus: jnp.ndarray, 
):
    rng, key = jax.random.split(rng)
    
    rng, key = jax.random.split(rng)
    new_critic, critic_info = update_critic_quantile(key, actor, critic, target_critic, temp, batch, discount, taus=quantile_taus)
    new_target_critic = target_update(new_critic, target_critic, tau)
    rng, key = jax.random.split(rng)
    new_actor, actor_info = update_actor(key, actor, new_critic, temp, batch)   
    rng, key = jax.random.split(rng)
    new_temp, alpha_info = temperature.update_temperature(temp, actor_info['entropy'], target_entropy)
    return rng, new_actor, new_critic, new_target_critic, new_temp, {
        **critic_info,
        **actor_info,
        **alpha_info,
    }

@functools.partial(
    jax.jit,
    static_argnames=(
        'discount',
        'tau',
        'target_entropy',
        'num_updates'
    ),
)
def _do_multiple_updates(
    rng: PRNGKey,
    actor: Model,
    critic: Model,
    target_critic: Model,
    temp: Model,
    batches: Batch,
    discount: float,
    tau: float,
    target_entropy: float,
    quantile_taus: jnp.ndarray, 
    step: int,
    num_updates: int
):
    def one_step(i, state):
        step, rng, actor, critic, target_critic, temp, info = state
        step = step + 1
        new_rng, new_actor, new_critic, new_target_critic, new_temp, info = _update(
            rng,
            actor,
            critic,
            target_critic,
            temp,
            jax.tree_map(lambda x: jnp.take(x, i, axis=1), batches),
            discount,
            tau,
            target_entropy,
            quantile_taus
        )
        return step, new_rng, new_actor, new_critic, new_target_critic, new_temp, info

    step, rng, actor, critic, target_critic, temp, info = one_step(
        0, (step, rng, actor, critic, target_critic, temp, {})
    )
    return jax.lax.fori_loop(1, num_updates, one_step, (step, rng, actor, critic, target_critic, temp, info))

class WIMLELearner(object):
    def __init__(
        self,
        seed: int,
        env,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        lr: float = 3e-4,
        world_model_lr: float = 1e-4,
        discount: float = 0.99,
        tau: float = 0.005,
        target_entropy: Optional[float] = None,
        init_temperature: float = 1.0,
        num_seeds: int = 5,
        n_quantiles: int = 100,
        n_models: int = 7,
        model_depth: int = 2,
        model_hidden_dims: int = 256,
        latent_size: int = 32,
        num_latent_codes: int = 10,
        **kwargs
    ) -> None:
        
        self.env = env
        self.n_models = n_models
        self.n_quantiles = n_quantiles
        action_dim = actions.shape[-1]
        observation_dim = observations.shape[-1]
        self.action_dim = float(action_dim)
        quantile_taus = jnp.arange(0, n_quantiles+1) / n_quantiles
        self.quantile_taus = ((quantile_taus[1:] + quantile_taus[:-1]) / 2.0)[None, ...]
        self.seeds = jnp.arange(seed, seed + num_seeds)
        self.target_entropy = -self.action_dim / 2 if target_entropy is None else target_entropy
        self.tau = tau
        self.discount = discount
        self.reset_list = [15001, 50001, 250001]
        self.num_seeds = num_seeds
        self.latent_size = latent_size
        self.num_latent_codes = num_latent_codes

        def _init_models(seed):
            rng = jax.random.PRNGKey(seed)
            rng, actor_key, critic_key, temp_key, world_model_rng = jax.random.split(rng, 5)

            actor_def = policies.NormalTanhPolicy(action_dim)
            critic_def = critic_net.DoubleCritic(output_nodes=self.n_quantiles)
            
            world_model_def = WorldModel(
                observation_dim=observation_dim,
                action_dim=action_dim,
                reward_dim=1,
                hidden_dims=model_hidden_dims,
                depth=model_depth,
                latent_size=latent_size,
            )

            def create_one_world_model(key: PRNGKey) -> Model:
                latents = jax.random.normal(key, shape=(observations.shape[0], latent_size))
                return Model.create(world_model_def, inputs=[key, observations, actions, latents], tx=optax.adamw(learning_rate=world_model_lr))

            world_model_keys = jax.random.split(world_model_rng, self.n_models)
            world_models = jax.vmap(create_one_world_model)(world_model_keys)

            actor = Model.create(actor_def, inputs=[actor_key, observations], tx=optax.adamw(learning_rate=lr))
            critic = Model.create(critic_def, inputs=[critic_key, observations, actions], tx=optax.adamw(learning_rate=lr))
            target_critic = Model.create(critic_def, inputs=[critic_key, observations, actions])
            temp = Model.create(temperature.Temperature(init_temperature), inputs=[temp_key], tx=optax.adam(learning_rate=lr, b1=0.5))
            
            return actor, critic, target_critic, temp, world_models, rng

        self.init_models = jax.jit(jax.vmap(_init_models))
        self.actor, self.critic, self.target_critic, self.temp, self.world_model, self.rng = self.init_models(self.seeds)
        self.step = 1

    def sample_actions(self, observations: np.ndarray, temperature: float = 1.0) -> jnp.ndarray:
        rng, actions = policies.sample_actions(self.rng, self.actor.apply_fn, self.actor.params, observations, temperature)
        self.rng = rng
        actions = np.asarray(actions)
        return np.clip(actions, -1, 1)
    
    def update(self, batch: Batch, num_updates: int, env_step: int) -> InfoDict:
        if env_step in self.reset_list:
            self.reset()

        step, rng, actor, critic, target_critic, temp, info = _do_multiple_updates(
            self.rng,
            self.actor,
            self.critic,
            self.target_critic,
            self.temp,
            batch,
            self.discount,
            self.tau,
            self.target_entropy,
            self.quantile_taus, 
            self.step,
            num_updates
        )
        self.step = step
        self.rng = rng
        self.actor = actor
        self.critic = critic
        self.target_critic = target_critic
        self.temp = temp
        
        return info

    def reset(self):
        self.step = 1
        self.actor, self.critic, self.target_critic, self.temp, self._, self.rng = self.init_models(self.seeds)

    def model_predict_many(self, observations: np.ndarray, actions: np.ndarray, num_latent_codes: int = 10, model: Model = None) -> Tuple[np.ndarray, np.ndarray]:
        if model is None:
            model = self.world_model

        vmapped_predict = jax.vmap(
            predict_many,
            in_axes=(1, None, 1, None, None, None),
            out_axes=(0, 0, 0)
        )
        
        rng = jax.vmap(jax.random.split, in_axes=(0, None))(self.rng, self.n_models)
        _, rewards, next_obs = vmapped_predict(
            rng,
            model.apply_fn,
            jax.lax.stop_gradient(model.params),  # Stop gradients for model params
            observations,
            actions,
            num_latent_codes
        )
        return rewards, next_obs
    
    def train_world_model_ensemble(
        self,
        replay_buffer: ParallelReplayBuffer,
        model_batch_size: int,
        model_num_updates: int = 100,
        num_consider: int = -1,
    ):
        def train_single_model(rng, world_model, num_latent_codes, latent_size, n_updates):
            # Saving the current statistics into the model - 300k is not important, just a large number
            recent = replay_buffer.get_last_window(300000)
            with jax.default_device(jax.devices("cpu")[0]):
                world_model = update_input_scaler(
                    world_model,
                    jnp.concatenate([recent.observations, recent.actions], axis=-1)
                )

            batch = replay_buffer.get_all_last_samples(num_consider, model_batch_size, n_updates)

            model, info = update_world_model_vmap(rng, world_model, batch, num_latent_codes, latent_size)
            return model, info

        vmapped_train = jax.vmap(train_single_model, in_axes=(1, 1, None, None, None), out_axes=(1, 0))
        rng = jax.vmap(jax.random.split, in_axes=(0, None))(self.rng, self.n_models)

        new_world_models, infos = vmapped_train(rng, self.world_model, self.num_latent_codes, self.latent_size, model_num_updates)
        self.world_model = new_world_models
        return infos