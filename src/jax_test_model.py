from flax import linen as nn
from jax import numpy as jnp
import jax
import numpyro
from numpyro import handlers
from numpyro import distributions as dist
from src.jax_subspace_sampling import init_curve_frame_cp, ortho_at_one_t
from src.jax_subspace_curve import vec_to_pytree, bezier_curve, pytree_to_vec, vec_to_single_pytree
from functools import partial
from typing import Callable
from proba_sandbox.module_sandbox.config.models import LeNettiConfig, LeNetConfig


class MLPModel(nn.Module):
    depth: int = 3
    width: int = 10
    activation: str = "relu"

    def setup(self) -> None:
        self.activation_fn = getattr(nn, self.activation)
        return super().setup()

    @nn.compact
    def __call__(self, x,):
        x = jnp.concat([x, x**2, x**3], axis=-1)
        for _ in range(self.depth):
            x = nn.Dense(self.width)(x)
            x = self.activation_fn(x)
        x = nn.Dense(1)(x)
        return x
    

class MLPModelOld(nn.Module):
    depth: int = 3
    width: int = 10
    activation: str = "relu"

    def setup(self) -> None:
        self.activation_fn = getattr(nn, self.activation)
        return super().setup()

    @nn.compact
    def __call__(self, x,):
        x = jnp.concat([x, x**2], axis=-1)
        for _ in range(self.depth):
            x = nn.Dense(self.width)(x)
            x = self.activation_fn(x)
        x = nn.Dense(1)(x)
        return x
    

class MLPModelUCI(nn.Module):
    depth: int = 3
    width: int = 16
    activation: str = "relu"

    def setup(self) -> None:
        self.activation_fn = getattr(nn, self.activation)
        return super().setup()

    @nn.compact
    def __call__(self, x,):
        for _ in range(self.depth):
            x = nn.Dense(self.width)(x)
            x = self.activation_fn(x)
        x = nn.Dense(1)(x)
        return x
    

class LeNetti(nn.Module):
    """
    A super simple LeNet version.

    Args:
        config (LeNettiConfig): The configuration for the model.
    """

    config: LeNettiConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        """
        Forward pass.

        Args:
            x (jnp.ndarray): The input data of
            shape (batch_size, channels, height, width).
        """
        activation = self.config.activation.flax_activation
        # x = x.transpose((0, 2, 3, 1))
        x = nn.Conv(
            features=1, kernel_size=(3, 3), strides=(1, 1), padding=2, name='conv1'
        )(x)
        x = activation(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=8, use_bias=self.config.use_bias, name='fc1')(x)
        x = activation(x)
        x = nn.Dense(features=8, use_bias=self.config.use_bias, name='fc2')(x)
        x = activation(x)
        x = nn.Dense(features=8, use_bias=self.config.use_bias, name='fc3')(x)
        x = activation(x)
        x = nn.Dense(
            features=self.config.out_dim, use_bias=self.config.use_bias, name='fc4'
        )(x)
        return x
    

class LeNet(nn.Module):
    """
    Implementation of LeNet.

    Args:
        config (LeNetConfig): The configuration for the model.
    """

    config: LeNetConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        """
        Forward pass.

        Args:
            x (jnp.ndarray): The input data of
            shape (batch_size, channels, height, width).
        """
        activation = self.config.activation.flax_activation
        # x = x.transpose((0, 2, 3, 1))
        x = nn.Conv(
            features=6, kernel_size=(5, 5), strides=(1, 1), padding=2, name='conv1'
        )(x)
        x = activation(x)
        x = nn.avg_pool(x, window_shape=(2, 2),
                        strides=(2, 2), padding='VALID')
        x = nn.Conv(
            features=16, kernel_size=(5, 5), strides=(1, 1), padding=0, name='conv2'
        )(x)
        x = activation(x)
        x = nn.avg_pool(x, window_shape=(2, 2),
                        strides=(2, 2), padding='VALID')
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(
            features=120, use_bias=self.config.use_bias, name='fc1')(x)
        x = activation(x)
        x = nn.Dense(features=84, use_bias=self.config.use_bias, name='fc2')(x)
        x = activation(x)
        x = nn.Dense(
            features=self.config.out_dim, use_bias=self.config.use_bias, name='fc3'
        )(x)
        return x


resnet_kernel_init = nn.initializers.variance_scaling(
    2.0, mode='fan_out', distribution='normal')


class ResNetBlock(nn.Module):
    act_fn: Callable  # Activation function
    c_out: int   # Output feature size
    subsample: bool = False  # If True, we apply a stride inside F

    @nn.compact
    def __call__(self, x, train=True):
        # Network representing F
        z = nn.Conv(self.c_out, kernel_size=(3, 3),
                    strides=(1, 1) if not self.subsample else (2, 2),
                    kernel_init=resnet_kernel_init,
                    use_bias=False)(x)
        z = nn.BatchNorm()(z, use_running_average=not train)
        z = self.act_fn(z)
        z = nn.Conv(self.c_out, kernel_size=(3, 3),
                    kernel_init=resnet_kernel_init,
                    use_bias=False)(z)
        z = nn.BatchNorm()(z, use_running_average=not train)

        if self.subsample:
            x = nn.Conv(self.c_out, kernel_size=(1, 1), strides=(
                2, 2), kernel_init=resnet_kernel_init)(x)

        x_out = self.act_fn(z + x)
        return x_out


class PreActResNetBlock(ResNetBlock):

    @nn.compact
    def __call__(self, x, train=True):
        # Network representing F
        z = nn.BatchNorm()(x, use_running_average=not train)
        z = self.act_fn(z)
        z = nn.Conv(self.c_out, kernel_size=(3, 3),
                    strides=(1, 1) if not self.subsample else (2, 2),
                    kernel_init=resnet_kernel_init,
                    use_bias=False)(z)
        z = nn.BatchNorm()(z, use_running_average=not train)
        z = self.act_fn(z)
        z = nn.Conv(self.c_out, kernel_size=(3, 3),
                    kernel_init=resnet_kernel_init,
                    use_bias=False)(z)

        if self.subsample:
            x = nn.BatchNorm()(x, use_running_average=not train)
            x = self.act_fn(x)
            x = nn.Conv(self.c_out,
                        kernel_size=(1, 1),
                        strides=(2, 2),
                        kernel_init=resnet_kernel_init,
                        use_bias=False)(x)

        x_out = z + x
        return x_out


class ResNet(nn.Module):
    num_classes: int
    act_fn: Callable
    block_class: nn.Module
    num_blocks: tuple = (3, 3, 3)
    c_hidden: tuple = (16, 32, 64)

    @nn.compact
    def __call__(self, x, train=True):
        # A first convolution on the original image to scale up the channel size
        x = nn.Conv(self.c_hidden[0], kernel_size=(
            3, 3), kernel_init=resnet_kernel_init, use_bias=False)(x)
        if self.block_class == ResNetBlock:  # If pre-activation block, we do not apply non-linearities yet
            x = nn.BatchNorm()(x, use_running_average=not train)
            x = self.act_fn(x)

        # Creating the ResNet blocks
        for block_idx, block_count in enumerate(self.num_blocks):
            for bc in range(block_count):
                # Subsample the first block of each group, except the very first one.
                subsample = (bc == 0 and block_idx > 0)
                # ResNet block
                x = self.block_class(c_out=self.c_hidden[block_idx],
                                     act_fn=self.act_fn,
                                     subsample=subsample)(x, train=train)

        # Mapping to classification output
        x = x.mean(axis=(-2, -3))
        x = nn.Dense(self.num_classes)(x)
        return x


def init_t_lambda_to_phi(cp_phi, k, epsilon=5., tube_scale=1.):
    curve, d_bezier = bezier_curve(k+1, cp_phi)
    
    # init orthogonal segments for Bezier curve
    t_cut, ortho_at_tcut = init_curve_frame_cp(
        d_bezier, cp_phi, epsilon=epsilon)

    # define fuction to generate orthogonal space at location t
    generate_ortho_space = partial(ortho_at_one_t, 
                                   d_bezier=d_bezier, 
                                   t_cut=t_cut, 
                                   ortho_at_tcut=ortho_at_tcut, 
                                   k=k)
    def clousure(l):
        t = l[0]
        tube = l[1:]*tube_scale
        space_point = curve(t)
        ortho_space = generate_ortho_space(t)
        # first vector is the tangent vector alonge bezier curve
        ortho_space = ortho_space[1:, :]
        space_point += jnp.einsum("ok,o->k", ortho_space, tube)  # t=1
        return space_point
    return jax.jit(clousure), curve, d_bezier


def init_model_tube(model, params, k, t_lambda_to_phi, t_phi_to_weight, prior_correct: bool, prior_scale=0.5, d_bezier: Callable = None, log_norm_d_bezier=jnp.array(0.), dist_scale:str|float=0.05):
    def model_tube(x, y=None, temperature=1.):
        # prior definition
        t = numpyro.sample("t", dist.Uniform(-0.1, 1.1).expand((1,)).to_event(1))
        tube = numpyro.sample("tube", dist.Normal(
            0., prior_scale).expand((k-1,)).to_event(1))
        if dist_scale == 'homo':
            scale = numpyro.sample("scale", dist.LogNormal(
                0, 1.))
        else:
            scale = dist_scale

        lambda_ = jnp.concat([t, tube], axis=-1)
        space_point = t_lambda_to_phi(lambda_)

        if prior_correct:
            # jacobian = jax.jacrev(transform)(jax.lax.stop_gradient(lambda_)).squeeze()
            jacobian = jax.jacrev(t_lambda_to_phi)(lambda_)
            sign, logabsdet = jnp.linalg.slogdet(jacobian)
            numpyro.factor("logabsdet", logabsdet)
        else:
            t_det = jnp.linalg.norm(d_bezier(t), axis=-1)
            numpyro.factor("t_scale", jnp.log(t_det) - log_norm_d_bezier)

        varphi = numpyro.deterministic('varphi', space_point)
        weight_vec = t_phi_to_weight(varphi)

        def apply_model(weight_vec, x):
            weight_pytree = vec_to_pytree(weight_vec, params)
            out = model.apply({'params': weight_pytree}, x)
            return out
        if weight_vec.ndim == 1:
            out = apply_model(weight_vec, x)
        else:
            out = jax.vmap(apply_model, in_axes=(0, None))(weight_vec, x)
        with numpyro.plate("data", x.shape[0], dim=-1):
            with handlers.scale(scale=1/temperature):
                numpyro.sample("obs", dist.Normal(out.squeeze(-1), scale+1e-10), obs=y)
    return model_tube


def init_model_line(model, params, k, t_lambda_to_phi, t_phi_to_weight, dist_scale: str | float = 0.05, d_bezier: Callable = lambda x: x, log_norm_d_bezier=jnp.array(0.)):
    def model_line(x, y=None, temperature=1.):
        # define prior
        t = numpyro.sample("t", dist.Uniform(-0.1, 1.1).expand((1,)).to_event(1))
        if dist_scale == 'homo':
            scale = numpyro.sample("scale", dist.LogNormal(
                0, 1.))
        else:
            scale = dist_scale

        # sample space point
        tube = jnp.zeros(k-1)
        lambda_ = jnp.concat([t, tube], axis=-1)
        space_point = t_lambda_to_phi(lambda_)

        # adjust for different speed
        t_det = jnp.linalg.norm(d_bezier(t), axis=-1)
        numpyro.factor("t_scale", jnp.log(t_det) - log_norm_d_bezier)

        # compute vartheta
        varphi = numpyro.deterministic('varphi', space_point)
        weight_vec = t_phi_to_weight(varphi)

        def apply_model(weight_vec, x):
            weight_pytree = vec_to_pytree(weight_vec, params)
            out = model.apply({'params': weight_pytree}, x)
            return out
        if weight_vec.ndim == 1:
            out = apply_model(weight_vec, x)
        else:
            out = jax.vmap(apply_model, in_axes=(0, None))(weight_vec, x)
        with numpyro.plate("data", x.shape[0], dim=-1):
            with handlers.scale(scale=1/temperature):
                numpyro.sample("obs", dist.Normal(
                    out.squeeze(-1), scale+1e-10), obs=y)
    return model_line


def init_model_phi(model, params, k, t_phi_to_weight, prior_scale:float=1., dist_scale:str|float=0.05):
    def model_phi(x, y=None, temperature=1.):
        varphi = numpyro.sample("varphi", dist.Normal(
            0., prior_scale).expand((k,)).to_event(1))
        if dist_scale == 'homo':
            scale = numpyro.sample("scale", dist.LogNormal(
                0, 1.))
        else:
            scale = dist_scale

        weight_vec = t_phi_to_weight(varphi)

        def apply_model(weight_vec, x):
            weight_pytree = vec_to_pytree(weight_vec, params)
            out = model.apply({'params': weight_pytree}, x)
            return out
        if weight_vec.ndim == 1:
            out = apply_model(weight_vec, x)
        else:
            out = jax.vmap(apply_model, in_axes=(0, None))(weight_vec, x)
        with numpyro.plate("data", x.shape[0], dim=-1):
            with handlers.scale(scale=1/temperature):
                numpyro.sample("obs", dist.Normal(out.squeeze(-1), scale+1e-10), obs=y)
    return model_phi


def init_full_space(model, params, prior_scale: float = 1., dist_scale: str | float = 0.05):
    n_params = pytree_to_vec(params['params']).size

    def model_full(x, y=None, temperature=1.):
        weight_vec = numpyro.sample("weights", dist.Normal(
            0., prior_scale).expand((n_params,)).to_event(1))
        if dist_scale == 'homo':
            scale = numpyro.sample("scale", dist.LogNormal(
                -1, .1))
        else:
            scale = dist_scale

        def apply_model(weight_vec, x):
            weight_pytree = vec_to_single_pytree(weight_vec, params['params'])
            out = model.apply({'params': weight_pytree}, x)
            return out
        out = apply_model(weight_vec, x)
        with numpyro.plate("data", x.shape[0], dim=-1):
            with handlers.scale(scale=1/temperature):
                numpyro.sample("obs", dist.Normal(
                    out.squeeze(-1), scale+1e-8), obs=y)
    return model_full
