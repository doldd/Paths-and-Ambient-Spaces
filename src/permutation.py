import jax
from jax import numpy as jnp
from jax import jit
from functools import partial
from src.jax_subspace_curve import SubspaceModel

@jit
def up_sorted(b):
    bb = jax.nn.softplus(b)
    bb = bb.at[..., 0].set(b[..., 0])
    # b = b.at[..., 1:].apply(jax.nn.softplus) # because of derivative error of apply softplus (scatter_apply JVP not implemented)
    return jnp.cumsum(bb, axis=-1)
    # return bb @ jnp.tri(bb.shape[-1]).T       # slower


def print_all_biases(p):
    def print_bias(path, p):
        if "bias" in path[-1].key:
            print(f"bias {p}")
    jax.tree_util.tree_map_with_path(print_bias, p)


@jit
def bias_ascending(params):
    def bias_sort(path, p):
        if "bias" in path[-1].key:
            return up_sorted(p)
        else:
            return p
    return jax.tree_util.tree_map_with_path(bias_sort, params)


def initialize_bias(rng_key, params):
    key = rng_key

    def bias_sorted_init(path, p):
        nonlocal key
        # if "bias" in path[-1].key:
        #     key, sample_key = jax.random.split(key)
        #     b = jax.random.normal(sample_key, (p.shape[-1]-1,))-2.68
        #     p = p.at[..., 1:].set(b)
        #     p = p.at[..., 0].set(-0.1*p.shape[1]/2)
        #     return p
        # if "bias" in path[-1].key:
        #     key, sample_key = jax.random.split(key)
        #     b = jax.random.normal(sample_key, (p.shape[-1]-1,))/10-4.6
        #     p = p.at[..., 1:].set(b)
        #     p = p.at[..., 0].set(-0.01*p.shape[1]/2)
        #     return p
        if "bias" in path[-1].key:
            key, sample_key = jax.random.split(key)
            b = jax.random.normal(sample_key, (p.shape[-1]-1,))/10-100
            p = p.at[..., 1:].set(b)
            p = p.at[..., 0].set(-0.0*p.shape[1]/2)
            return p
        else:
            return p
    return jax.tree_util.tree_map_with_path(bias_sorted_init, params)


class SubspaceModelPermFree(SubspaceModel):
    @partial(jit, static_argnums=(0,))
    def __call__(self, params, t, x):
        """
        Computes the output of the model for given parameters, time, and input.

        Parameters:
        - params: The parameters of the model.
        - t: The time parameter.
        - x: The input data.

        Returns:
        - The output of the model.

        """
        # sample Bezier coefficient
        bezier_coeff = self.bezier(t)
        # Compute one parameter set per sample
        sample_param = jax.tree.map(lambda p: jnp.einsum(
            'sk,k...->s...', bezier_coeff, p), params)
        sample_param = dict(params=sample_param)
        # transform biases' into bias which is sorted in ascending order
        sample_param = bias_ascending(sample_param)
        # forward pass per sample
        out = jax.vmap(self.model.apply, in_axes=(0, None))(sample_param, x)
        return out


class UniformTSubspacePermFree(SubspaceModelPermFree):
    pass
