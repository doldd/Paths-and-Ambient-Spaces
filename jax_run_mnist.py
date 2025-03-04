from src.jax_subspace_curve import OrthoSpan, CategorySubspace, UniformTCategory
import blackjax
from numpyro.infer.util import initialize_model
from jax_tqdm import scan_tqdm
from src.jax_test_model import LeNet, LeNetti, init_t_lambda_to_phi
from proba_sandbox.module_sandbox.config.models import LeNetConfig, LeNettiConfig
import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=10"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import jax
# jax.config.update('jax_platform_name', 'cpu')
from src.jax_subspace_sampling import setup_inference_chain
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
from numpyro import handlers
import arviz as az
from src.jax_subspace_curve import pytree_to_matrix, vec_to_pytree
import wandb
import time
from proba_sandbox.data.dataset_generators.mnist import MNISTGenerator
from torch.utils import data
import numpyro
from numpyro import distributions as dist
from numpyro import handlers


def init_model_tube(model, params, k, t_lambda_to_phi, t_phi_to_weight, prior_correct: bool, prior_scale=0.5, d_bezier=None, log_norm_d_bezier=jnp.array(0.)):
    def model_tube(x, y=None, temperature=1.):
        # prior definition
        t = numpyro.sample(
            "t", dist.Uniform(-0.1, 1.1).expand((1,)).to_event(1))
        tube = numpyro.sample("tube", dist.Normal(
            0., prior_scale).expand((k-1,)).to_event(1))

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
        logit = apply_model(weight_vec, x)
        with numpyro.plate("data", x.shape[0], dim=-1):
            with handlers.scale(scale=1/temperature):
                numpyro.sample("obs", dist.CategoricalLogits(logit), obs=y)
    return model_tube

def load_mnist(seed=0):
    mnistgen = MNISTGenerator(
                batch_size=480,
                data_dir=f'./proba_sandbox/data/generated_data',
                seed=seed,
            )
    train_loader, val_loader, test_loader = mnistgen._create_np_data_loaders()
    ds = train_loader.dataset
    arr = [ds[i] for i in range(len(ds))]
    x, y = data.default_collate(arr)
    x, y = jnp.array(x.numpy().transpose(0, 2, 3, 1)), jnp.array(y.numpy())

    ds = val_loader.dataset
    arr = [ds[i] for i in range(len(ds))]
    x_val, y_val = data.default_collate(arr)
    x_val, y_val = jnp.array(x_val.numpy().transpose(
        0, 2, 3, 1)), jnp.array(y_val.numpy())

    ds = test_loader.dataset
    arr = [ds[i] for i in range(len(ds))]
    x_test, y_test = data.default_collate(arr)
    x_test, y_test = jnp.array(x_test.numpy().transpose(
        0, 2, 3, 1)), jnp.array(y_test.numpy())
    return x, y, x_val, y_val, x_test, y_test


def main():
    logger = wandb.init()
    config = wandb.config
    single_run(logger, config)


def single_run(logger, config, n_samples_nuts=None):
    rng_key = random.PRNGKey(config['rng_seed'])

    # find number of samples for NUTS samples for comparison (only if another sampler as nuts is used)
    def find_n_samples_nuts(config):
        if config['sampling']['sampler'] == 'nuts':
            return None
        if type(config['nuts_sweep_id']) is int:
            return config['nuts_sweep_id']
        config_ = deepcopy(dict(config))
        config_['sampling']['sampler'] = 'nuts'
        api = wandb.Api()
        runs = api.runs("ddold/subspace_sampling_compare")
        n_samples_nuts = None
        for run in runs:
            if (run.state != 'finished') or (run.sweep == None) or (run.sweep.id != config['nuts_sweep_id']):
                continue
            print(f"checking run {run.id}")
            if np.all([run.config.get(k, None) == v for k, v in config_.items()]):
                n_samples_nuts = int(run.summary['num_forward'])
        if n_samples_nuts is None:
            raise ValueError(f"No NUTS run found for config {config}")
        return n_samples_nuts
    n_samples_nuts = find_n_samples_nuts(config) if n_samples_nuts is None else n_samples_nuts
    
    x, y, x_val, y_val, x_test, y_test = load_mnist(config['rng_seed']) # get fold from seed
    k = config['curve_params']['k']

    def train_curve(rng_key, k: int, activation: str, n_samples: int, lr: float, num_epochs: int, small_cnn: bool, batch_size: int, uniform_t_optimize: bool=False):
        rng_key, init_key = random.split(rng_key)

        if small_cnn:
            model_config = LeNettiConfig(activation=activation,
                                         out_dim=10,
                                         use_bias=True)
            model = LeNetti(model_config)
        else:
            model_config = LeNetConfig(activation=activation,
                                    out_dim=10,
                                    use_bias=True)
            model = LeNet(model_config)

        if uniform_t_optimize:
            s_model = UniformTCategory(model, k,
                                       n_samples=n_samples)
        else:
            s_model = CategorySubspace(model, k,
                                    n_samples=n_samples)
        params = s_model.init_params(init_key, x[:2])

        # Train
        optimizer = optax.adam(lr)
        opt_state = optimizer.init(params['params'])  # only dnn parameters
        best_loss = np.inf
        best_params = params

        def train_batch(carry, batch):
            x_, y_ = batch
            rng_key, params, opt_state, best_loss, best_params = carry
            rng_key, subkey = random.split(rng_key)
            loss, params, opt_state = s_model.train_step(
                subkey, params, x_, y_, opt_state, optimizer)
            # validate
            rng_key, subkey = random.split(rng_key)
            val_loss = s_model.compute_loss(
                subkey, params, x_val, y_val, n_samples=10)
            best_params = jax.lax.cond(
                val_loss < best_loss, lambda x: params, lambda x: best_params, None)
            best_loss = jnp.minimum(best_loss, val_loss)
            return (rng_key, params, opt_state, best_loss, best_params), (loss, val_loss)

        @scan_tqdm(num_epochs)
        def train_epoch(carry, _):
            rng_key = carry[0]
            shuffel = jax.random.permutation(rng_key, jnp.arange(y.shape[0]))
            x_ = x[shuffel].reshape(-1, batch_size, 28, 28, 1)
            y_ = y[shuffel].reshape(-1, batch_size)
            carry, (loss_batch, valid_loss_batch) = jax.lax.scan(
                train_batch, carry, (x_, y_))
            return carry, (loss_batch, valid_loss_batch)

        carry = (rng_key, params, opt_state, best_loss, best_params)
        carry, (losses, valid_losses) = jax.lax.scan(
            train_epoch, carry, jnp.arange(num_epochs))

        # log losses for wandb
        n_batches = np.array(losses).shape[-1]
        epochs = np.repeat(np.arange(num_epochs), n_batches)
        losses = jnp.array(losses).flatten()
        valid_losses = jnp.array(valid_losses).flatten()
        [logger.log({"train_loss": l, "Curve valid_loss": vl, 'epoch': e}) for e, l, vl in zip(epochs, losses, valid_losses)]
        params = carry[4] # select best parameters
        rng_key = carry[0]

        def acc_fn(carry, t):
            x, y = carry
            logits = s_model(params['params'], t, x)
            return carry, (logits.argmax(-1) == y).mean(-1)
        # valid accuracy
        # acc_val = acc_fn((x_val, y_val), jnp.linspace(0, 1, 10))[1].mean()
        # logger.log({'acc_val': acc_val})

        # plot likelihood and acc along curve
        fig, ax = plt.subplots(1, 1)
        t_space = jnp.linspace(0.0, 1.0, 500)
        # valid
        def forward(carry, t):
            x_, y_ = carry
            return carry, s_model.nll(params, t, x_, y_).mean(axis=-1)
        # t_spae with shape for (sequential shape and vmapped shape)
        nll = jax.lax.scan(forward, (x_val, y_val),
                           t_space.reshape(-1, 10))[1].flatten()
        acc = jax.lax.scan(acc_fn, (x_val, y_val),
                           t_space.reshape(-1, 10))[1].flatten()
        logger.log({'acc_val_curve': acc.mean()})
        ax.plot(t_space, -nll, label="valid", c=plt.get_cmap('tab10')(1))
        ax2 = ax.twinx()
        ax2.plot(t_space, acc, label="valid acc",
                c=plt.get_cmap('tab10')(1), linestyle="--")
        # train
        nll = jax.lax.scan(forward, (x, y), t_space.reshape(-1, 2))[1].flatten()
        acc = jax.lax.scan(acc_fn, (x, y),
                           t_space.reshape(-1, 2))[1].flatten()
        ax.plot(t_space, -nll, label="train", c=plt.get_cmap('tab10')(0))
        ax2.plot(t_space, acc, label="train acc",
                c=plt.get_cmap('tab10')(0), linestyle="--")
        # test
        nll = jax.lax.scan(forward, (x_test, y_test),
                           t_space.reshape(-1, 10))[1].flatten()
        acc = jax.lax.scan(acc_fn, (x_test, y_test),
                           t_space.reshape(-1, 10))[1].flatten()
        ax.plot(t_space, -nll, label="test", c=plt.get_cmap('tab10')(2))
        ax2.plot(t_space, acc, label="test acc",
         c=plt.get_cmap('tab10')(2), linestyle="--")
        ax.set_xlabel("t")
        ax.set_ylabel("mean log likelihood")
        ax2.set_ylabel("acc")
        ax.legend()
        ax2.legend(loc='lower center')
        logger.log(
            {'Nll alonge curve': wandb.Image(fig)})

        return rng_key, params, model
    
    rng_key, params, model = train_curve(rng_key, **config['curve_params'])
    
    #%% generate transforamtion functions between weight-phi-lambda space
    # get design matrix from curve parameters stroed as pytree
    cp_w = pytree_to_matrix(params['params'], k)
    # define the transformation function between weight space and phi space
    t_phi_to_weight = OrthoSpan(cp_w)
    # control points in the varphi space
    cp_phi = t_phi_to_weight.inv(cp_w)
    del cp_w
    print(f"Control points in phi space: {cp_phi.shape}")
    # define fuction to generate orthogonal space at location t
    t_lambda_to_phi, curve, d_bezier = init_t_lambda_to_phi(cp_phi, k,
                                                      epsilon=config['sampling']['space_config'].get('epsilon', 5.),
                                                      tube_scale=config['sampling']['space_config'].get('tube_scale', 1.)) # for phi space tube_scale is irelevant thus set to default

    sampler = config['sampling']['sampler']
    num_chains = config['sampling']['num_chains']
    temperature = config['sampling']['temperature']
    num_warmup = config['sampling']['num_warmup']
    num_draws = config['sampling']['num_draws']
    
    rng_key = random.PRNGKey(2+config['rng_seed'])
    # define sampling model
    def setup_model(init_key, space: str):
        initial_position = {'t': jax.random.uniform(init_key, (num_chains, 1), minval=0., maxval=1.),
                            'tube': jnp.zeros((num_chains, k-1))}
        if space == 'varphi':
            raise NotImplementedError("Varphi space not implemented yet")
            sampling_model = init_model_phi(
                model, params['params'], k, t_phi_to_weight, 
                prior_scale=config['sampling']['space_config']['prior_scale'],
                dist_scale="homo" if config['curve_params']['optimize_distparams'] else jnp.exp(
                    params['dist_params']['log_scale'].item()))
            # transform same initial position to phi space
            initial_position = jnp.concatenate([initial_position['t'],initial_position['tube']], axis=1)
            initial_position = jax.vmap(t_lambda_to_phi, in_axes=(0,))(
                initial_position)
            initial_position = {'varphi': initial_position}
        elif space == 'lambda':
            tt = jnp.linspace(0., 1., 10_000)
            bezier_grad = jax.vmap(d_bezier)(tt)
            log_normalized_bezier_grad = jnp.log(jnp.trapezoid(
                jnp.linalg.norm(bezier_grad, axis=-1), tt))
            sampling_model = init_model_tube(model,
                                             params['params'],
                                             k,
                                             t_lambda_to_phi,
                                             t_phi_to_weight, 
                                             prior_scale=config['sampling']['space_config']['prior_scale'],
                                             prior_correct=config['sampling']['space_config']['prior_correct'],
                                             d_bezier=d_bezier,
                                             log_norm_d_bezier=log_normalized_bezier_grad,
                                            )
        else:
            raise ValueError(f"Space {space} not supported")
        return sampling_model, initial_position
    
    rng_key, init_key = jax.random.split(rng_key)
    sampling_model, initial_position = setup_model(
        init_key, config['sampling']['space_config']['space'])
    
    # start with sampling -> generate potential_fn for blackjax
    rng_key, init_key = jax.random.split(rng_key)
    init_params, potential_fn, post_proc_fun, _ = initialize_model(
        init_key,
        # model_tube,
        sampling_model,
        model_args=(x, y, temperature),
        # could also set to True => potential_fn_gen(**model_args) instead of potential_fn
        dynamic_args=False,
    )

    def logdensity_fn(params):
        return -potential_fn(params)
        
    def get_sampler_methods(sampler):
        # for sampler specific number of warmup and draws
        num_warmup_ = num_warmup
        num_draws_ = num_draws
        # sampler specific init and kernel functions
        if sampler == 'nuts':
            def init_fn(warmup_key, initial_position, num_warmup):
                adapt = blackjax.window_adaptation(
                    blackjax.nuts, logdensity_fn, target_acceptance_rate=0.8, max_num_doublings=8, is_mass_matrix_diagonal=True, initial_step_size=0.2
                )
                (last_state, parameters), _ = adapt.run(
                    warmup_key, initial_position, num_warmup)
                del parameters['max_num_doublings'] 
                return last_state, parameters

            def get_kernel(parameters):
                sampling_alg = blackjax.nuts(
                    logdensity_fn, **parameters, max_num_doublings=8)
                return sampling_alg
            
        elif sampler == 'rmhmc':
            def init_fn(warmup_key, initial_position, num_warmup):
                adapt = blackjax.window_adaptation(
                    blackjax.rmhmc, logdensity_fn, target_acceptance_rate=0.8,  is_mass_matrix_diagonal=True, initial_step_size=0.2, num_integration_steps=n_samples_nuts
                )
                (last_state, parameters), _ = adapt.run(
                    warmup_key, initial_position, num_warmup)
                return last_state, parameters
            
            def get_kernel(parameters):
                parameters2 = parameters.copy()
                # parameters2['mass_matrix'] = jnp.linalg.inv(parameters2['inverse_mass_matrix'])
                parameters2['mass_matrix'] = parameters2['inverse_mass_matrix']
                del parameters2['inverse_mass_matrix']
                sampling_alg = blackjax.rmhmc(logdensity_fn, **parameters2)
                return sampling_alg

        elif sampler == 'mclmc':
            num_warmup_ = num_warmup*n_samples_nuts//2
            num_draws_ = num_draws*n_samples_nuts//2

            def init_fn(warmup_key, initial_position, num_warmup):
                # create an initial state for the sampler
                initial_state = blackjax.mcmc.mclmc.init(
                    position=initial_position, logdensity_fn=logdensity_fn, rng_key=warmup_key
                )

                # build the kernel
                def kernel(sqrt_diag_cov): 
                    return blackjax.mcmc.mclmc.build_kernel(
                        logdensity_fn=logdensity_fn,
                        integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
                        sqrt_diag_cov=sqrt_diag_cov,
                    )

                # find values for L and step_size
                (
                    blackjax_state_after_tuning,
                    blackjax_mclmc_sampler_params,
                ) = blackjax.mclmc_find_L_and_step_size(
                    mclmc_kernel=kernel,
                    num_steps=num_warmup,
                    state=initial_state,
                    rng_key=warmup_key,
                    diagonal_preconditioning=False,
                )
                return blackjax_state_after_tuning, blackjax_mclmc_sampler_params
            
            def get_kernel(parameters):
                sampling_alg = blackjax.mclmc(
                    logdensity_fn,
                    L=parameters.L,
                    step_size=parameters.step_size
                )
                return sampling_alg
            
        elif sampler == 'random_walk':
            num_warmup_ = 1
            num_draws_ = num_draws*n_samples_nuts

            def init_wrapper(sigma_rw):
                def init_fn(warmup_key, initial_position, num_warmup):
                    kernel = blackjax.additive_step_random_walk(logdensity_fn,
                                                                blackjax.mcmc.random_walk.normal(sigma_rw))
                    last_state = kernel.init(initial_position, warmup_key)
                    parameters = sigma_rw
                    return last_state, parameters
                return init_fn

            sigma_rw = jnp.array([0.15,] + [0.001,]*(k-1))
            if config['sampling']['space_config']['space'] == 'varphi':
                sigma_rw = jnp.full(k, 0.15)
            init_fn = init_wrapper(sigma_rw)
            
            def get_kernel(parameters):
                kernel = blackjax.additive_step_random_walk(logdensity_fn, 
                                                            blackjax.mcmc.random_walk.normal(parameters))
                return kernel
        else:
            raise ValueError(f"Sampler {sampler} not supported")

        return init_fn, get_kernel, num_warmup_, num_draws_
    warmup_fn, run_inference  = setup_inference_chain(mode="SEQUENTIAL", num_chains=num_chains)
    init_fn, get_kernel, num_warmup_, num_draws_ = get_sampler_methods(sampler)
    print(f"Start warmup for {sampler}")
    wall_time, rng_key, last_state, parameters = warmup_fn(rng_key, initial_position, init_fn, num_warmup_)
    print("Warmup Wall time: {:.2f} s".format(wall_time))
    print(f"Start sampling ...")

    wall_time, rng_key, states, infos = run_inference(
        get_kernel, rng_key, last_state, parameters, num_draws_)
    print("Sampling Wall time: {:.2f} s".format(wall_time))
    
    # extract samples for posterior predictive (Subset for mclmc and random_walk)
    selected_states = states.position
    if sampler == 'mclmc':
        selected_states = jax.tree.map(
            lambda x: x[:, ::n_samples_nuts//2], states.position)
        if config['sampling']['space_config']['space'] == 'varphi':
            num_value_grad = states.position['varphi'].shape[1]/num_draws
        else:
            num_value_grad = states.position['t'].shape[1]/num_draws*2
    elif sampler == 'random_walk':
        selected_states = jax.tree.map(
            lambda x: x[:, ::n_samples_nuts], states.position)
        num_value_grad = n_samples_nuts
    else:
        num_value_grad = float(infos.num_integration_steps.sum()/(num_chains*num_draws))
    print(f"transforme samples... e.g. t with shape: {selected_states['t'].shape}")
    posterior = jax.lax.map(lambda x: jax.vmap(post_proc_fun)(x), selected_states)
    # posterior = jax.vmap(jax.vmap(post_proc_fun)), selected_states
    print(f"samples transformed")

    def log_like_fn(x, y):
        def single_log_like_fn(sample):
            with handlers.seed(rng_seed=0):
                cml = handlers.condition(sampling_model, data=sample)
                ex_trace = handlers.trace(cml).get_trace(
                    x, y, config['sampling']['temperature'])
                site = ex_trace['obs']
                log_like = site['fn'].log_prob(
                    site['value'])
                return log_like
        return jax.vmap(lambda sample: jax.lax.map(single_log_like_fn, sample))(posterior)

    def prediction(key, x):
        sample_keys = jax.random.split(key, num_chains)
        def single_prediction_fn(rng_key, sample):
            rng_key, sample_key = random.split(rng_key)
            with handlers.seed(rng_seed=sample_key):
                cml = handlers.condition(sampling_model, data=sample)
                ex_trace = handlers.trace(cml).get_trace(
                    x, None, 1.)
                site = ex_trace['obs']
                return rng_key, site['fn'].sample(sample_key)
        key, preds = jax.vmap(lambda key, sample: jax.lax.scan(
            single_prediction_fn, key, sample))(sample_keys, posterior)
        return key[-1], preds

    samples = az.from_dict(posterior=posterior)
    summary = az.summary(samples)
    print(summary)
    w_tabel = wandb.Table(dataframe=summary)
    logger.log({'summary': w_tabel})


    # posterior predictive valid with data batching
    def compute_metrics(rng_key, x, y):
        # lppd
        log_like = log_like_fn(x, y).reshape(-1, len(y))
        lppd = jnp.mean(jax.scipy.special.logsumexp(
            log_like, axis=0) - jnp.log(log_like.shape[0]))
        del log_like
        # rmse
        rng_key, category = prediction(rng_key, x)
        acc = (category.mean((0, 1)).round() == y).mean(-1)
        del category
        return rng_key, lppd, acc

    rng_key, lppd_valid, acc_valid = compute_metrics(rng_key, x_val, y_val)
    rng_key, lppd_test, acc_test = compute_metrics(rng_key, x_test, y_test)

    # log metrics to wandb
    summary_var = az.summary(samples, var_names=["varphi"])
    res = dict(time_s=wall_time,
            r_hat_max=summary_var['r_hat'].max(),
            r_hat_mean=summary_var['r_hat'].mean(),
            ess_min=summary_var['ess_bulk'].min(),
            ess_mean=summary_var['ess_bulk'].mean(),
            num_forward=num_value_grad,
            acceptance_rate=infos.acceptance_rate.mean() if sampler != 'mclmc' else 1.,
            lppd_test=lppd_test,
            acc_test=acc_test,
            lppd_valid=lppd_valid,
            acc_valid=acc_valid
            )
    logger.summary.update(res)

    # save trace plot
    plt.figure()
    az.plot_trace(samples, compact=False, show=True)
    logger.log({'trace': wandb.Image(plt.gcf())})

    plt.figure()
    az.plot_forest(samples, r_hat=True, ess=True, show=True)
    logger.log({'forest': wandb.Image(plt.gcf())})
    
    print("Save artifacts")
    artifact = wandb.Artifact(name="samples", type="xarray")
    samples.to_netcdf(f"tmp_files/{logger.id}_samples.nc")
    jnp.savez(f"tmp_files/{logger.id}_sampler_state_and_info.npz", dict(states=states, infos=infos))
    jnp.save(f'tmp_files/{logger.id}_params.npy', params)
    print("add files to artifact")
    artifact.add_file(f"tmp_files/{logger.id}_samples.nc")
    artifact.add_file(f"tmp_files/{logger.id}_sampler_state_and_info.npz")
    artifact.add_file(f'tmp_files/{logger.id}_params.npy')
    print("Log artifact")
    time.sleep(5)
    logger.log_artifact(artifact)
    time.sleep(5)
    print("finish run")
    logger.finish()


if __name__ == '__main__':
    # Add requirement for wandb core
    wandb.require("core")
    os.makedirs('tmp_files', exist_ok=True)

    # config = {
    #     'curve_params': {'k': 10,
    #                      'activation': 'relu',
    #                      'n_samples': 1,
    #                      'lr': 0.005,
    #                      'num_epochs': 50,
    #                      'small_cnn': False,
    #                      'batch_size': 480,
    #                      'uniform_t_optimize': True
    #                     },
    #     'sampling': {'space_config': {'space': 'lambda',
    #                                   'prior_scale': .5,
    #                                   'tube_scale': .1,
    #                                   'epsilon': 25.,
    #                                   'prior_correct': False},
    #                  'num_chains': 2,
    #                  'temperature': 10.,
    #                  'num_warmup': 100,
    #                  'num_draws': 100,
    #                  'sampler': 'mclmc'
    #                  },
    #     'rng_seed': 0,
    #     'nuts_sweep_id': '1',
    # }
    # logger = wandb.init(project="subspace_test", name="mnist", entity="ddold", config=config)
    # single_run(logger, config, n_samples_nuts=200)

    main()
