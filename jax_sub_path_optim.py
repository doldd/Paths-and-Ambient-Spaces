import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=10"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import jax
from src.jax_test_model import MLPModel, MLPModelUCI, init_t_lambda_to_phi, init_model_tube, init_model_phi, init_model_line
# jax.config.update('jax_platform_name', 'cpu')
from proba_sandbox.module_sandbox.dataset.tabular import TabularLoader
from proba_sandbox.module_sandbox.dataset.base import DataConfig
from copy import deepcopy
from numpyro.diagnostics import hpdi
import arviz as az
import jax.numpy as jnp
from jax import random
import jax.numpy as jnp
from jax import random
import numpy as np
import matplotlib.pyplot as plt
import optax
from jax import random, jit, grad
import optax
from tqdm import tqdm
import matplotlib.pyplot as plt
# Assuming equivalent JAX implementations
from src.jax_subspace_curve import OrthoSpan, SubspaceModel, UniformTSubspace
from numpyro import handlers
import arviz as az
from src.jax_subspace_curve import pytree_to_matrix
import wandb
import time
import blackjax
from numpyro.infer.util import initialize_model
from jax_tqdm import scan_tqdm
from src.jax_subspace_curve import bezier_coeff_fn
from src.permutation import initialize_bias, bias_ascending, UniformTSubspacePermFree, SubspaceModelPermFree


USE_PMAP = False  # set to True if you want to use pmap for parallelization else vmap is used

def load_data():
    data = np.load('regression_data.npz')
    x = jnp.array(data['x'])
    y = jnp.array(data['y'])
    x_test = jnp.array(data['xt'])
    y_test = jnp.array(data['yt'])
    return x, y, x_test, y_test





# def get_curve_ll(model, x, y, t_space):
#     def mean_ll(params):
#         nll = model.nll(params, t_space, x, y).mean()
#         return -nll
#     return mean_ll


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

def load_uci(dataset_name, seed):
    dataconfig = DataConfig(
        path=f'{os.getcwd()}/proba_sandbox/data/{dataset_name}.data',
        source='local',
        data_type='tabular',
        task='regr',
        target_column=None,
        features=None,
        datapoint_limit=None,
        normalize=True,
        train_split=0.7,
        valid_split=0.1,
        test_split=0.2,
    )
    rng_key = random.PRNGKey(seed)
    data = TabularLoader(config=dataconfig, rng=rng_key)
    print(f"""Train shape: x {data.train_x.shape} y{data.train_y.shape}
    Val shape:   x {data.valid_x.shape}  y{data.valid_y.shape}
    Test shape:  x {data.test_x.shape}  y{data.test_y.shape}""")
    return data.train_x, data.train_y, data.valid_x, data.valid_y, data.test_x, data.test_y


def main():
    logger = wandb.init()
    config = wandb.config
    single_run(logger, config)


def single_run(logger, config):
    # load dataset
    if config['dataset'] == 'default':
        x, y, x_test, y_test = load_data()
        x_val, y_val = x, y
    elif config['dataset'] == 'generate':
        x, y, x_val, y_val, x_test, y_test, _ = gen_data(**config)
    else:
        x, y, x_val, y_val, x_test, y_test = load_uci(config['dataset'], config['rng_seed'])

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

    def train_curve(rng_key, k: int, model_kwargs: dict, n_samples: int, lr: float, num_epochs: int, out_scale,
                    optimize_distparams, use_validation: bool = False, uniform_t_optimize: bool = False, bias_ascending_dnn: bool = False):
        rng_key, init_key = random.split(rng_key)
        if (config['dataset'] == 'default') or config['dataset'] == 'generate':
            model = MLPModel(**model_kwargs)
        else:
            print("Using UCI model")
            model = MLPModelUCI(**model_kwargs)

        if uniform_t_optimize:
            if bias_ascending_dnn:
                print("Using s~U(0,S) and bias ascending")
                s_model = UniformTSubspacePermFree(model, k,
                                        n_samples=n_samples,
                                        out_scale=out_scale,
                                        optimize_distparams=optimize_distparams)
            else:
                print("Using s~U(0,1)")
                s_model = UniformTSubspace(model, k,
                                           n_samples=n_samples,
                                           out_scale=out_scale,
                                           optimize_distparams=optimize_distparams)
        else:
            if bias_ascending_dnn:
                print("Using t~U(0,S) and bias ascending")
                s_model = SubspaceModelPermFree(model, k,
                                                n_samples=n_samples,
                                                out_scale=out_scale,
                                                optimize_distparams=optimize_distparams)
            else:
                print("Using t~U(0,1)")
                s_model = SubspaceModel(model, k,
                                        n_samples=n_samples,
                                        out_scale=out_scale,
                                        optimize_distparams=optimize_distparams)
        
        params = s_model.init_params(init_key, x)
        if bias_ascending_dnn:
            # initialize bias for ascending order
            rng_key, init_key = random.split(rng_key)
            params = initialize_bias(init_key, params)

        # Train
        optimizer = optax.adam(lr)
        if optimize_distparams:
            opt_state = optimizer.init(params) # dnn_parameters + dist_parameters
        else:
            opt_state = optimizer.init(params['params'])  # only dnn parameters
        best_loss = np.inf
        best_params = params

        # define metric function. 
        # if use_validation is True, use validation data to select best model parameters
        # else use training data but compute loss trough 1000 curve samples
        def comp_metric_set(x, y):
            return lambda subkey, params: s_model.compute_loss(
                subkey, params, x, y, n_samples=1000)
        if use_validation:
            comp_metric = comp_metric_set(x_val, y_val)
        else:
            comp_metric = comp_metric_set(x, y)

        # train loop
        @scan_tqdm(num_epochs)
        def train(carry, _):
            rng_key, params, opt_state, best_loss, best_params, epoch, epoch_sel = carry
            rng_key, subkey = random.split(rng_key)
            loss, params, opt_state = s_model.train_step(
                subkey, params, x, y, opt_state, optimizer)
            # validate
            rng_key, subkey = random.split(rng_key)
            val_loss = comp_metric(subkey, params)
            best_params, epoch_sel = jax.lax.cond(
                val_loss < best_loss, lambda x: (params, epoch), lambda x: (best_params, epoch_sel), None)
            best_loss = jnp.minimum(best_loss, val_loss)

            cp_w = pytree_to_matrix(params['params'], k)
            length = bezier_length(cp_w)
            return (rng_key, params, opt_state, best_loss, best_params, epoch+1, epoch_sel), (loss, val_loss, length, epoch)
        carry = (rng_key, params, opt_state, best_loss, best_params, 0, -1)
        carry, (losses, valid_losses, length, epochs) = jax.lax.scan(train, carry,
                                             jnp.arange(num_epochs))
        
        # log best loss and epoch wh
        logger.summary.update({'best_loss': carry[3], 'at_epoch': carry[6]})

        # log losses for wandb
        if use_validation:
            [logger.log({"train_loss": l, "Curve valid_loss": vl, "Curve length": leng, 'epoch': e})
             for l, vl, leng, e in zip(losses, valid_losses, length, epochs)]
        else:
            [logger.log({"train_loss": l, "Curve train_loss": vl, "Curve length": leng, 'epoch': e})
             for l, vl, leng, e in zip(losses, valid_losses, length, epochs)]
        params = carry[4] # select best parameters
        rng_key = carry[0]

        # plot predictive for onedimensional regression dataset
        if (config['dataset'] == 'default') or config['dataset'] == 'generate':
            fig, ax = plt.subplots()
            t_space = jnp.linspace(0, 1, 100)
            x_lin = jnp.linspace(-3, 3, 100)[:, None]
            out = s_model(params['params'], t_space, x_lin).squeeze(axis=-1)
            ax.plot(x, y, 'o', label='train')
            # plt.fill_between(x.squeeze(), out.mean(axis=0) - out.std(axis=0), out.mean(axis=0) + out.std(axis=0), alpha=0.5)
            # plot lines using viridis color map
            colors = plt.cm.viridis(t_space)
            for o, c in zip(out, colors):
                ax.plot(x_lin, o, color=c, alpha=0.3)
            ax.plot(x_lin, out.mean(axis=0), label='mean',
                    c='red', linewidth=2, alpha=0.8)
            ax.set_ylabel("y")
            ax.set_xlabel("x")
            plt.tight_layout()
            logger.log(
                {'Curve Predictive': wandb.Image(fig)})

        # plot likelihood along curve
        fig, ax = plt.subplots()
        t_space = jnp.linspace(0.0, 1.0, 1000)
        ll = -s_model.nll(params, t_space, x, y).mean(axis=-1)
        logger.log({'curve_log_likelihood': ll.mean()})
        ax.plot(t_space, ll)
        ax.set_xlabel("t")
        ax.set_ylabel("log like")
        plt.tight_layout()
        logger.log(
            {'Nll alonge curve': wandb.Image(fig)})

        return rng_key, params, model, carry[1]
    
    rng_key = random.PRNGKey(config['rng_seed'])
    rng_key, params, model, last_params = train_curve(rng_key, **config['curve_params'])
    

    # #%% generate transforamtion functions between weight-phi-lambda space
    # # get design matrix from curve parameters stroed as pytree
    # cp_w = pytree_to_matrix(params['params'], k)
    # # define the transformation function between weight space and phi space
    # t_phi_to_weight = OrthoSpan(cp_w)
    # # control points in the varphi space
    # cp_phi = t_phi_to_weight.inv(cp_w)
    # print(f"Control points in phi space: {cp_phi.shape}")
    # # define fuction to generate orthogonal space at location t
    # t_lambda_to_phi, curve, d_bezier = init_t_lambda_to_phi(cp_phi, k,
    #                                                   epsilon=config['sampling']['space_config'].get('epsilon', 5.),
    #                                                   tube_scale=config['sampling']['space_config'].get('tube_scale', 1.)) # for phi space tube_scale is irelevant thus set to default

    artifact = wandb.Artifact(name="params", type="pytree")
    with open(f'tmp_files/{logger.id}_params.npy', 'wb') as f:
        jnp.save(f, params)
    with open(f'tmp_files/{logger.id}_params_last.npy', 'wb') as f:
        jnp.save(f, last_params)
    artifact.add_file(f'tmp_files/{logger.id}_params.npy')
    artifact.add_file(f'tmp_files/{logger.id}_params_last.npy')
    print(f'save tmp_files/{logger.id}_params')
    logger.log_artifact(artifact)

    artifact2 = wandb.Artifact(name="dataset", type="numpy")
    rng_seed = config['rng_seed']
    ds_name = config['dataset']
    file_name = f'tmp_files/{rng_seed}_{ds_name}_data.npz'
    with open(file_name, 'wb') as f:
        jnp.savez(f, x=x, y=y, x_val=x_val, y_val=y_val, x_test=x_test, y_test=y_test)
    artifact2.add_file(file_name)
    print(f"Save {file_name}")
    logger.log_artifact(artifact2)
    logger.finish()


def test_main():
    print(wandb.config)

if __name__ == '__main__':
    # Add requirement for wandb core
    wandb.require("core")
    os.makedirs('tmp_files', exist_ok=True)

    # config = {
    #     'curve_params': {'k': 10,
    #                      'model_kwargs': {'depth': 3, 'width': 10, 'activation': 'relu'},
    #                      'n_samples': 10,
    #                      'lr': 0.01,
    #                      'num_epochs': 1000,
    #                      'use_validation': True,
    #                      'out_scale': 0.05,
    #                      'optimize_distparams': False, 
    #                      'uniform_t_optimize': False,
    #                      'bias_ascending_dnn': False
    #                     },
    #     'rng_seed': 0,
    #     'nuts_sweep_id': '1',
    #     'dataset': 'bikesharing',
    # }
    # logger = wandb.init(project="subspace_test", name="test", entity="ddold", config=config)
    # single_run(logger, config)

    main()
