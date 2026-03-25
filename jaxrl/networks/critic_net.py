"""Implementations of algorithms for continuous control."""

from typing import Callable, Tuple

import jax.numpy as jnp
from flax import linen as nn

from jaxrl.networks.common import BroNet


class Critic(nn.Module):
    hidden_dims: int = 512
    depth: int = 2
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    output_nodes: int = 1

    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, actions], -1)
        critic = BroNet(hidden_dims=self.hidden_dims, depth=self.depth, activations=self.activations, add_final_layer=True, output_nodes=self.output_nodes)(inputs)
        return critic
        
class DoubleCritic(nn.Module):
    hidden_dims: int = 512
    depth: int = 2
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    output_nodes: int = 1

    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        critic1 = Critic(self.hidden_dims, self.depth, activations=self.activations, output_nodes=self.output_nodes)(observations, actions)
        critic2 = Critic(self.hidden_dims, self.depth, activations=self.activations, output_nodes=self.output_nodes)(observations, actions)
        return critic1, critic2
    
