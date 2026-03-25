import math
import os
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

PRNGKey = Any
Params = flax.core.FrozenDict[str, Any]
Shape = Sequence[int]
Dtype = Any  # this could be a real type?
InfoDict = Dict[str, float]

def tree_norm(tree):
    return jnp.sqrt(sum((x**2).sum() for x in jax.tree_util.tree_leaves(tree)))

# Simba-style layers (used by WorldModel)
_HYPER_EPS = 1e-8


def l2normalize(x: jnp.ndarray, axis: int) -> jnp.ndarray:
    l2norm = jnp.linalg.norm(x, ord=2, axis=axis, keepdims=True)
    return x / jnp.maximum(l2norm, _HYPER_EPS)


class Scaler(nn.Module):
    dim: int
    init: float = 1.0
    scale: float = 1.0

    def setup(self):
        self.scaler = self.param(
            "scaler",
            nn.initializers.constant(1.0 * self.scale),
            self.dim,
        )
        self.forward_scaler = self.init / self.scale

    def __call__(self, x):
        return self.scaler * self.forward_scaler * x


class HyperDense(nn.Module):
    hidden_dim: int

    def setup(self):
        self.w = nn.Dense(
            name="hyper_dense",
            features=self.hidden_dim,
            kernel_init=nn.initializers.orthogonal(scale=1.0, column_axis=0),
            use_bias=False,
        )

    def __call__(self, x):
        return self.w(x)


class HyperMLP(nn.Module):
    hidden_dim: int
    out_dim: int
    scaler_init: float
    scaler_scale: float
    eps: float = 1e-8

    def setup(self):
        self.w1 = HyperDense(self.hidden_dim)
        self.scaler = Scaler(self.hidden_dim, self.scaler_init, self.scaler_scale)
        self.w2 = HyperDense(self.out_dim)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.w1(x)
        x = self.scaler(x)
        x = nn.relu(x) + self.eps
        x = self.w2(x)
        return l2normalize(x, axis=-1)


class HyperEmbedder(nn.Module):
    hidden_dim: int
    scaler_init: float
    scaler_scale: float
    c_shift: float

    def setup(self):
        self.w = HyperDense(self.hidden_dim)
        self.scaler = Scaler(self.hidden_dim, self.scaler_init, self.scaler_scale)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        new_axis = jnp.ones((x.shape[:-1] + (1,))) * self.c_shift
        x = jnp.concatenate([x, new_axis], axis=-1)
        x = l2normalize(x, axis=-1)
        x = self.w(x)
        x = self.scaler(x)
        return l2normalize(x, axis=-1)


class HyperLERPBlock(nn.Module):
    hidden_dim: int
    scaler_init: float
    scaler_scale: float
    alpha_init: float
    alpha_scale: float
    expansion: int = 2

    def setup(self):
        self.mlp = HyperMLP(
            hidden_dim=self.hidden_dim * self.expansion,
            out_dim=self.hidden_dim,
            scaler_init=self.scaler_init / math.sqrt(self.expansion),
            scaler_scale=self.scaler_scale / math.sqrt(self.expansion),
        )
        self.alpha_scaler = Scaler(
            self.hidden_dim,
            init=self.alpha_init,
            scale=self.alpha_scale,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        residual = x
        x = self.mlp(x)
        x = residual + self.alpha_scaler(x - residual)
        return l2normalize(x, axis=-1)


def default_init(scale: Optional[float] = math.sqrt(2.0)):
    return nn.initializers.orthogonal(scale)

class BroNet(nn.Module):
    hidden_dims: int
    depth: int
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    add_final_layer: bool = False
    output_nodes: int = 1

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        if self.depth == 1:
            layer1 = nn.Dense(self.hidden_dims, kernel_init=default_init())
            x = layer1(x)
            x = nn.LayerNorm()(x)
            x = self.activations(x)
            layer2 = nn.Dense(self.hidden_dims, kernel_init=default_init())
            res = layer2(x)
            res = nn.LayerNorm()(res)
            res = self.activations(res)
            layer3 = nn.Dense(self.hidden_dims, kernel_init=default_init())
            res = layer3(res)
            res = nn.LayerNorm()(res)
            x = res + x
            if self.add_final_layer:
                x = nn.Dense(self.output_nodes, kernel_init=default_init())(x)
            return x
        if self.depth == 2:
            layer1 = nn.Dense(self.hidden_dims, kernel_init=default_init())
            x = layer1(x)
            x = nn.LayerNorm()(x)
            x = self.activations(x)
            layer2 = nn.Dense(self.hidden_dims, kernel_init=default_init())
            res = layer2(x)
            res = nn.LayerNorm()(res)
            res = self.activations(res)
            layer3 = nn.Dense(self.hidden_dims, kernel_init=default_init())
            res = layer3(res)
            res = nn.LayerNorm()(res)
            x = res + x
            layer4 = nn.Dense(self.hidden_dims, kernel_init=default_init())
            res = layer4(x)
            res = nn.LayerNorm()(res)
            res = self.activations(res)
            layer5 = nn.Dense(self.hidden_dims, kernel_init=default_init())
            res = layer5(res)
            res = nn.LayerNorm()(res)
            x = res + x
            if self.add_final_layer:
                x = nn.Dense(self.output_nodes, kernel_init=default_init())(x)
            return x
        if self.depth == 3:
            layer1 = nn.Dense(self.hidden_dims, kernel_init=default_init())
            x = layer1(x)
            x = nn.LayerNorm()(x)
            x = self.activations(x)
            layer2 = nn.Dense(self.hidden_dims, kernel_init=default_init())
            res = layer2(x)
            res = nn.LayerNorm()(res)
            res = self.activations(res)
            layer3 = nn.Dense(self.hidden_dims, kernel_init=default_init())
            res = layer3(res)
            res = nn.LayerNorm()(res)
            x = res + x
            layer4 = nn.Dense(self.hidden_dims, kernel_init=default_init())
            res = layer4(x)
            res = nn.LayerNorm()(res)
            res = self.activations(res)
            layer5 = nn.Dense(self.hidden_dims, kernel_init=default_init())
            res = layer5(res)
            res = nn.LayerNorm()(res)
            x = res + x
            layer6 = nn.Dense(self.hidden_dims, kernel_init=default_init())
            res = layer6(x)
            res = nn.LayerNorm()(res)
            res = self.activations(res)
            layer7 = nn.Dense(self.hidden_dims, kernel_init=default_init())
            res = layer7(res)
            res = nn.LayerNorm()(res)
            x = res + x
            
            if self.add_final_layer:
                x = nn.Dense(self.output_nodes, kernel_init=default_init())(x)
            return x

@flax.struct.dataclass
class SaveState:
    params: Params
    opt_state: Optional[optax.OptState] = None


@flax.struct.dataclass
class Model:
    step: int
    apply_fn: nn.Module = flax.struct.field(pytree_node=False)
    params: Params
    tx: Optional[optax.GradientTransformation] = flax.struct.field(
        pytree_node=False)
    opt_state: Optional[optax.OptState] = None

    @classmethod
    def create(cls,
               model_def: nn.Module,
               inputs: Sequence[jnp.ndarray],
               tx: Optional[optax.GradientTransformation] = None) -> 'Model':
        variables = model_def.init(*inputs)

        params = variables.pop('params')

        if tx is not None:
            opt_state = tx.init(params)
        else:
            opt_state = None

        return cls(step=1,
                   apply_fn=model_def,
                   params=params,
                   tx=tx,
                   opt_state=opt_state)

    def __call__(self, *args, **kwargs):
        return self.apply_fn.apply({'params': self.params}, *args, **kwargs)

    def apply(self, *args, **kwargs):
        return self.apply_fn.apply(*args, **kwargs)

    def apply_gradient(self, loss_fn) -> Tuple[Any, 'Model']:
        grad_fn = jax.grad(loss_fn, has_aux=True)
        grads, info = grad_fn(self.params)
        grad_norm = tree_norm(grads)
        info['grad_norm'] = grad_norm

        updates, new_opt_state = self.tx.update(grads, self.opt_state,
                                                self.params)
        new_params = optax.apply_updates(self.params, updates)

        return self.replace(step=self.step + 1,
                            params=new_params,
                            opt_state=new_opt_state), info

    def save(self, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(flax.serialization.to_bytes(SaveState(params=self.params, opt_state=self.opt_state)))

    def load(self, load_path: str) -> 'Model':
        with open(load_path, 'rb') as f:
            contents = f.read()
            saved_state = flax.serialization.from_bytes(
                SaveState(params=self.params, opt_state=self.opt_state), contents
            )
        return self.replace(params=saved_state.params, opt_state=saved_state.opt_state)
