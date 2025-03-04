import os
import arviz as az
import jax
import jax.numpy as jnp
from jax import random
import seaborn as sns
from scipy.integrate import cumulative_trapezoid
import jax.numpy as jnp
from jax import random
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from flax import linen as nn
import optax
import jax
import jax.numpy as jnp
from jax import random, jit, grad
import optax
from tqdm import tqdm
import matplotlib.pyplot as plt
# Assuming equivalent JAX implementations
from src.jax_subspace_curve import OrthoSpan, SubspaceModel
import numpyro
import numpyro.distributions as dist
from numpyro import handlers
import arviz as az
from src.jax_subspace_sampling import init_curve_frame_cp, ortho_at_one_t
from src.jax_subspace_curve import bezier_curve, pytree_to_matrix, vec_to_pytree
import pandas as pd
import wandb
import time
from scipy.special import binom
from src.jax_test_model import MLPModel
from src.jax_subspace_curve import bezier_coeff_fn
from jax_tqdm import scan_tqdm


os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=10"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# load data
def load_data():
    data = jnp.load('regression_data.npz')
    x = jnp.array(data['x'])
    y = jnp.array(data['y'])
    x_test = jnp.array(data['xt'])
    y_test = jnp.array(data['yt'])
    return x, y, x_test, y_test 

# Generate data by using the model itself
def gen_data(rng_seed, curve_params, **kwargs):
    rng_key = random.PRNGKey(rng_seed + 1133)
    x = jnp.linspace(-2, 2, 100)
    x = x[(jnp.abs(x) > 0.6)].reshape(-1, 1)

    # define ground trouth model
    rng_key, rng_init = random.split(rng_key)
    model_gen = MLPModel(**curve_params['model_kwargs'])
    params = model_gen.init(rng_init, x)

    # generate train data
    y_gt = model_gen.apply(params, x).squeeze()
    rng_key, rng = random.split(rng_key)
    y = y_gt + jax.random.normal(rng, y_gt.shape) * 0.05

    # generate valid data
    x_val = jnp.linspace(-2, 2, 25)
    x_val = x_val[(jnp.abs(x_val) > 0.6)].reshape(-1, 1)
    y_gt_val = model_gen.apply(params, x_val).squeeze()
    rng_key, rng = random.split(rng_key)
    y_val = y_gt_val + jax.random.normal(rng, y_gt_val.shape) * 0.05

    xt = jnp.linspace(-2.5, 2.5, 33).reshape(-1, 1)
    rng_key, rng = random.split(rng_key)
    y_gtt = model_gen.apply(params, xt).squeeze()
    yt = y_gtt + jax.random.normal(rng, y_gtt.shape) * 0.05

    return x, y, x_val, y_val, xt, yt, y_gtt


def get_curve_ll(model, x, y, t_space):
    def mean_ll(params):
        nll = model.nll(params, t_space, x, y).mean()
        return -nll
    return mean_ll


def main():
    logger = wandb.init()
    config = wandb.config
    single_run(logger, config)

class SubspaceModelGradNorm(SubspaceModel):
    @partial(jit, static_argnums=(0, 6))
    def train_step(self, key, params, x, y, opt_state, optimizer):
        """
        Performs a single training step.

        Parameters:
        - key: The random key for generating random numbers.
        - params: The parameters of the model.
        - x: The input data.
        - y: The target output.
        - opt_state: The optimizer state.
        - optimizer: The optimizer.

        Returns:
        - The loss, updated parameters, and updated optimizer state.

        """
        loss, grads = jax.value_and_grad(
            self.compute_loss, argnums=(1,), )(key, params, x, y, self.n_samples)
        if self.optimize_distparams:
            updates, opt_state = optimizer.update(grads[0], opt_state, params)
            params = optax.apply_updates(params, updates)
        else:
            updates, opt_state = optimizer.update(grads[0]['params'], opt_state, params['params'])
            params['params'] = optax.apply_updates(params['params'], updates)
        return loss, params, opt_state, grads[0]


def single_run(logger, config):
    # x, y, x_test, y_test = load_data()
    x, y, x_val, y_val, x_test, y_test,_ = gen_data(**config)

    # define function to compute bezier curve length
    bezier_coeff_inv = bezier_coeff_fn(config['curve_params']['k'])
    def get_d_bezier(t):
        coeff = jax.vmap(bezier_coeff_inv)(t)

        def clousure(cp):
            n = cp.shape[0] - 1
            cp_diff = cp[1:] - cp[:-1]
            return jnp.einsum('tj,j...->t...', coeff, cp_diff) * n
        return clousure
    
    @jit
    def bezier_length(cp):
        t = jnp.linspace(0, 1, 1000)
        d_bezier = get_d_bezier(t)
        d_t = jnp.linalg.norm(d_bezier(cp), axis=-1)
        length = jax.scipy.integrate.trapezoid(d_t, t)
        return length
    
    @jit
    def lower_bound(cp):
        # get the length of the line between the first and last control
        return jnp.linalg.norm(cp[-1]-cp[0])
    
    @jit
    def upper_bound(cp):
        # get the length of the line between each control point
        return jnp.linalg.norm(jnp.diff(cp, axis=0), axis=1).sum()

    @jit 
    def mass_center(cp):
        return jnp.linalg.norm(cp.mean(axis=0))
    
    @jit
    def gyration(cp):
        mc = cp.mean(0)
        return jnp.linalg.norm(cp - mc, axis=1).mean()
    
    @jit
    def mean_curvature(cp):
        tt= jnp.linspace(0, 1, 1000)
        _, d_bezier= bezier_curve(cp.shape[0], cp)
        curvature= jax.vmap(jax.jacrev(d_bezier))(tt).squeeze()
        curvature_norm= jnp.linalg.norm(curvature, axis=-1)
        return jax.scipy.integrate.trapezoid(curvature_norm, tt)
    
    @jit
    def mean_curvature2(cp):
        tt = jnp.linspace(0, 1, 1000)
        _, d_bezier = bezier_curve(cp.shape[0], cp)

        def single(t):
            grad = d_bezier(t)
            second_dev = jax.jacrev(d_bezier)(t).squeeze()

            norm_r_prime_sq = jnp.dot(grad, grad)  # |r'(t)|^2
            norm_r_double_prime_sq = jnp.dot(
                second_dev, second_dev)  # |r''(t)|^2
            dot_r_prime_double_prime = jnp.dot(
                grad, second_dev)  # r'(t) · r''(t)

            # Gram determinant equivalent for curvature
            numerator = jnp.sqrt(
                norm_r_prime_sq * norm_r_double_prime_sq - dot_r_prime_double_prime**2)
            denominator = norm_r_prime_sq ** (3/2)
            return numerator / denominator
        
        curvature = jax.vmap(single)(tt)
        return jax.scipy.integrate.trapezoid(curvature, tt)
    
    def init_rel_center(cp):
        mean_center_t0 = cp.mean(axis=0)
        @jit
        def rel_center(cp):
            return jnp.linalg.norm(cp.mean(axis=0) - mean_center_t0)
        return rel_center


    # setup
    k = config['curve_params']['k']
    rng_key = random.PRNGKey(config['rng_seed'])
    rng_key, init_key = random.split(rng_key)
    model = MLPModel(**config['curve_params']['model_kwargs'])
    s_model = SubspaceModelGradNorm(
        model, k, n_samples=config['curve_params']['n_samples'], 
        out_scale=0.05,
        optimize_distparams=False)
    params = s_model.init_params(init_key, x)

    # Train
    lr = config['curve_params']['lr']
    if config['curve_params']['optim'] == "adam":
        optimizer = optax.adamw(lr, weight_decay=config['curve_params']['weight_decay'])
    elif config['curve_params']['optim'] == "sgd":
        assert config['curve_params']['weight_decay'] == 0, "SGD currently does not support weight decay"
        optimizer = optax.sgd(lr)
    opt_state = optimizer.init(params['params'])

    best_params = params
    best_metric = -jnp.inf
    curve_performance_fn = get_curve_ll(s_model, x_val, y_val, jnp.linspace(0, 1, 1000))

    # init metric
    cp_w = pytree_to_matrix(params['params'], k)
    rel_center_fn = init_rel_center(cp_w)

    num_epochs = config['curve_params']['num_epochs']
    # train loop
    @scan_tqdm(num_epochs)
    def train(carry, _):
        rng_key, params, opt_state, best_metric, best_params, epoch = carry
        rng_key, subkey = random.split(rng_key)
        loss, params, opt_state, grad = s_model.train_step(
            subkey, params, x, y, opt_state, optimizer)
        # validate
        rng_key, subkey = random.split(rng_key)
        curve_performance = curve_performance_fn(params)
        best_params = jax.lax.cond(
            curve_performance >= best_metric, lambda x: params, lambda x: best_params, None)
        best_metric = jnp.maximum(best_metric, curve_performance)
        
        cp_w = pytree_to_matrix(params['params'], k)
        length = bezier_length(cp_w)
        l_bound = lower_bound(cp_w)
        u_bound = upper_bound(cp_w)
        gyra = gyration(cp_w)
        center = mass_center(cp_w)
        mean_curv = mean_curvature2(cp_w)
        rel_center = rel_center_fn(cp_w)
        grad_norm = jnp.linalg.norm(pytree_to_matrix(grad['params'], k), axis=1).mean()

        return (rng_key, params, opt_state, best_metric, best_params, epoch+1), (loss, curve_performance, length, l_bound, u_bound, gyra, center,  mean_curv,  rel_center, grad_norm, epoch)
    carry = (rng_key, params, opt_state, best_metric, best_params, 0)
    carry, (losses, curve_performance, length, l_bound, u_bound, gyra, center, mean_curv, rel_center, grad_norm, epoch) = jax.lax.scan(train, carry,
                                                    jnp.arange(num_epochs))

    print("Log metrics")
    [logger.log({"train_loss": l, "Curve performance": vl, "bezier_length": leng, "lower bound": lb, "upper bound": ub, "gyration": gym, "center": ce, "Average curvature": a_curve, "Relative Center": rel_c, "Gradient": grad_n, "epoch": e})
     for l, vl, leng, lb, ub, gym, ce, a_curve, rel_c, grad_n, e in zip(losses[::100], curve_performance[::100], length[::100], l_bound[::100], u_bound[::100], gyra[::100], center[::100], mean_curv[::100], rel_center[::100], grad_norm[::100], epoch[::100])]
    print("Save metrics as artifact")
    art = wandb.Artifact(name="length_metrics", type="metric")

    frame = pd.DataFrame(dict(
        loss=losses,
        curve_performance=curve_performance,
        length=length,
        l_bound=l_bound,
        u_bound=u_bound,
        gyra=gyra,
        center=center,
        mean_curve=mean_curv,
        relative_center=rel_center,
        grad_norm=grad_norm,
        epoch=epoch))
    
    frame['epoch_group'] = (
        10**np.log10(frame['epoch']+1).round(4)).astype(int)
    grouped_df = frame.set_index('epoch_group').groupby(level='epoch_group').mean()
    grouped_df.reset_index(level='epoch_group', drop=True, inplace=True)
    grouped_df.to_csv(
        f'tmp_files/{logger.id}_length_metrics.csv', index=True)

    # with open(f'tmp_files/{logger.id}_length_metrics.npz', 'wb') as f:
    #     np.savez(f, loss=losses, curve_performance=curve_performance, length=length, l_bound=l_bound, u_bound=u_bound, gyra=gyra, center=center, mean_curv=mean_curv, relative_center=rel_center, epoch=epoch)
    art.add_file(f'tmp_files/{logger.id}_length_metrics.csv')
    logger.log_artifact(art)

    print("Log performance")
    # logger.log({"train_loss": losses, "Curve performance": curve_performance, "bezier_length": length, "lower bound": l_bound, "upper bound": u_bound, "gyration": gyra, "center": center, "epoch": epoch})
    params = carry[1]  # select best parameters
    best_params = carry[4]  # select best parameters
    rng_key = carry[0]     

    fig, ax = plt.subplots()
    t_space = jnp.linspace(0, 1, 100)
    x_lin = jnp.linspace(-3, 3, 100)[:, None]
    out = s_model(params['params'], t_space, x_lin).squeeze(axis=-1)
    plt.plot(x, y, 'o', label='train')
    # plt.fill_between(x.squeeze(), out.mean(axis=0) - out.std(axis=0), out.mean(axis=0) + out.std(axis=0), alpha=0.5)
    # plot lines using viridis color map
    colors = plt.cm.viridis(t_space)
    for o, c in zip(out, colors):
        plt.plot(x_lin, o, color=c, alpha=0.3)
    plt.plot(x_lin, out.mean(axis=0), label='mean', c='red', linewidth=2, alpha=0.8)
    logger.log({"Curve Predictive_last": wandb.Image(fig)})

    # plot likelihood along curve
    fig, ax = plt.subplots()
    t_space = jnp.linspace(0.0, 1.0, 1000)
    ll = -s_model.nll(params, t_space, x, y).mean(axis=-1)
    mean_ll = ll.mean()
    logger.log({'curve_ll_last': mean_ll})
    ax.plot(t_space, ll)
    ax.set_xlabel("t")
    ax.set_ylabel("log like")
    plt.tight_layout()
    logger.log(
        {'Nll alonge curve_last': wandb.Image(fig)})
    
    fig, ax = plt.subplots()
    t_space = jnp.linspace(0, 1, 100)
    x_lin = jnp.linspace(-3, 3, 100)[:, None]
    out = s_model(best_params['params'], t_space, x_lin).squeeze(axis=-1)
    plt.plot(x, y, 'o', label='train')
    # plt.fill_between(x.squeeze(), out.mean(axis=0) - out.std(axis=0), out.mean(axis=0) + out.std(axis=0), alpha=0.5)
    # plot lines using viridis color map
    colors = plt.cm.viridis(t_space)
    for o, c in zip(out, colors):
        plt.plot(x_lin, o, color=c, alpha=0.3)
    plt.plot(x_lin, out.mean(axis=0), label='mean',
             c='red', linewidth=2, alpha=0.8)
    logger.log({"Curve Predictive_best": wandb.Image(fig)})

    # plot likelihood along curve
    fig, ax = plt.subplots()
    t_space = jnp.linspace(0.0, 1.0, 1000)
    ll = -s_model.nll(best_params, t_space, x, y).mean(axis=-1)
    mean_ll = ll.mean()
    logger.log({'curve_ll_best': mean_ll})
    ax.plot(t_space, ll)
    ax.set_xlabel("t")
    ax.set_ylabel("log like")
    plt.tight_layout()
    logger.log(
        {'Nll alonge curve_best': wandb.Image(fig)})

    # save params
    artifact = wandb.Artifact(name="params", type="pytree")
    jnp.save(f'tmp_files/{logger.id}_params.npy', params)
    jnp.save(f'tmp_files/{logger.id}_best_params.npy', best_params)
    jnp.savez(f'tmp_files/{logger.id}_data.npz', x=x, y=y, x_val=x_val, y_val=y_val, x_test=x_test, y_test=y_test)
    artifact.add_file(f'tmp_files/{logger.id}_params.npy')
    artifact.add_file(f'tmp_files/{logger.id}_best_params.npy')
    artifact.add_file(f'tmp_files/{logger.id}_data.npz')
    artifact.add
    logger.log_artifact(artifact)
    wandb.finish()


if __name__ == "__main__":
    wandb.require("core")
    os.makedirs('tmp_files', exist_ok=True)
    # config = {
    #     'curve_params': {'k': 20,
    #                      'model_kwargs': {'depth': 3, 'width': 10, 'activation': 'relu'},
    #                      'n_samples': 500,
    #                      'lr': 0.01,
    #                      'num_epochs': 10000,
    #                      'weight_decay': 0.01,
    #                      },
    #     'rng_seed': 0
    # }
    # logger = wandb.init(project='subspace_bezier_length_dnn',
    #                     entity='ddold', name='bezier_length_test')
    # logger.config.update(config)
    # single_run(logger, config)
    main()

