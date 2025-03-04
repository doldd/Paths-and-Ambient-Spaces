from jax_tqdm import scan_tqdm
from numpyro.infer.util import initialize_model
import blackjax
import time
import wandb
from src.jax_subspace_curve import pytree_to_matrix, pytree_to_vec
from numpyro import handlers
from src.jax_subspace_curve import OrthoSpan, SubspaceModel
from tqdm import tqdm
from jax import random, jit, grad
import optax
import matplotlib.pyplot as plt
import numpy as np
from jax import random
import jax.numpy as jnp
import arviz as az
from numpyro.diagnostics import hpdi
from copy import deepcopy
from proba_sandbox.module_sandbox.dataset.base import DataConfig
from proba_sandbox.module_sandbox.dataset.tabular import TabularLoader
from src.jax_test_model import MLPModel, MLPModelUCI, init_t_lambda_to_phi, init_model_tube, init_model_phi, init_model_line, init_full_space
import jax
import os
from src.permutation import bias_ascending
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=10"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# jax.config.update('jax_platform_name', 'cpu')
# Assuming equivalent JAX implementations


USE_PMAP = False  # set to True if you want to use pmap for parallelization else vmap is used


def warmup_fn(rng_key, initial_position, init_fn, num_warmup, num_chains):
    # run warmup adaption
    rng_key, warmup_key = jax.random.split(rng_key)
    warmup_key = jax.random.split(warmup_key, num_chains)

    time_start = time.time()
    if USE_PMAP:
        run = jax.pmap(init_fn, in_axes=(None, 0, 0), static_broadcasted_argnums=(
            0,), devices=jax.devices('cpu'))
    else:
        run = jax.vmap(init_fn, in_axes=(None, 0, 0))
    last_state, parameters = run(num_warmup, warmup_key, initial_position)
    jax.block_until_ready(last_state)
    time_ = time.time() - time_start
    return time_, rng_key, last_state, parameters


def run_inference(kernel, rng_key, last_state, parameters, num_draws, num_chains):
    rng_key, sample_key = jax.random.split(rng_key)
    sample_keys = jax.random.split(sample_key, num_chains)

    def inference_loop(rng_key, parameters, initial_state, num_samples):
        final_state, trace_state, trace_info = blackjax.util.run_inference_algorithm(
            rng_key=rng_key,
            initial_state=initial_state,
            inference_algorithm=kernel(parameters),
            num_steps=num_samples,
            # transform=transform,
            progress_bar=False,
        )
        return trace_state, trace_info

    # run inference loop
    if USE_PMAP:
        inference_loop_multiple_chains = jax.pmap(inference_loop,
                                                  in_axes=(0, 0, 0, None),
                                                  static_broadcasted_argnums=(
                                                      3,),
                                                  devices=jax.devices('cpu'))
    else:
        inference_loop_multiple_chains = jax.vmap(inference_loop, in_axes=(
            0, 0, 0, None))

    # jax.debug.print(f"params: {parameters['max_num_doublings']}")
    time_start = time.time()
    pmap_states, pmap_infos = inference_loop_multiple_chains(
        sample_keys, parameters, last_state, num_draws)
    jax.block_until_ready(pmap_states)
    time_ = time.time() - time_start
    return time_, rng_key, pmap_states, pmap_infos


# def get_curve_ll(model, x, y, t_space):
#     def mean_ll(params):
#         nll = model.nll(params, t_space, x, y).mean()
#         return -nll
#     return mean_ll


def train_de(seed, model, x, y, x_val, y_val, lr, optimize_distparams, num_epochs, use_validation, **kwargs):
    rng_key = random.PRNGKey(seed)
    params = model.init(random.PRNGKey(seed), x)
    params['dist_params'] = {'log_scale': jnp.log(jnp.array(.05))}

    # train loop
    optimizer = optax.adam(learning_rate=lr)

    # dnn_parameters + dist_parameters
    if optimize_distparams:
        opt_state = optimizer.init(params)
    else:
        opt_state = optimizer.init(params['params'])

    @jit
    def loss_fn(params, x, y):
        out = model.apply(params, x)
        nll = -jax.scipy.stats.norm.logpdf(y,
                                           loc=out.squeeze(axis=-1), scale=jnp.exp(params['dist_params']['log_scale'])+1e-8)
        return nll.mean()

    @scan_tqdm(num_epochs)
    def train_loop(carry, _):
        rng_key, params, opt_state, best_loss, best_params, epoch, epoch_sel = carry
        # rng_key, subkey = random.split(rng_key)

        # train
        loss, grads = jax.value_and_grad(loss_fn, argnums=0)(params, x, y)
        if optimize_distparams:
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
        else:
            updates, opt_state = optimizer.update(
                grads['params'], opt_state, params['params'])
            params['params'] = optax.apply_updates(params['params'], updates)

        # validate
        # rng_key, subkey = random.split(rng_key)
        if use_validation:
            val_loss = loss_fn(params, x_val, y_val)
        else:
            val_loss = loss_fn(params, x, y)
        best_params, epoch_sel = jax.lax.cond(
            val_loss < best_loss, lambda x: (params, epoch), lambda x: (best_params, epoch_sel), None)
        best_loss = jnp.minimum(best_loss, val_loss)

        return (rng_key, params, opt_state, best_loss, best_params, epoch+1, epoch_sel), (loss, val_loss, epoch)

    rng_key, sub_key = random.split(rng_key, 2)
    carry = (sub_key, params, opt_state, np.inf, params, 0, -1)
    carry, (losses, valid_losses, epochs) = jax.lax.scan(train_loop, carry,
                                                         jnp.arange(num_epochs))
    return carry, (losses, valid_losses, epochs)

def main():
    logger = wandb.init()
    config = wandb.config
    single_run(logger, config)


def single_run(logger, config):
    rng_key = random.PRNGKey(config['rng_seed']+33)

    def find_params_and_data(config):
        # serach for run with same config
        def find_run():
            try:
                tqdm_run = tqdm(wandb.Api().sweep(config['path_sweep_id']).runs)
            except Exception as e:
                try:
                    run = wandb.Api().run(config['path_sweep_id'])
                    print(
                        f"Using exactly specified run {config['path_sweep_id']}")
                    return run
                except Exception as e:
                    print(f"Error: {e}")
                    print(f"Could not find sweep or run {config['path_sweep_id']}")
                    raise e
            for run in tqdm_run:
                if run.state != 'finished':
                    print(f"Run {run.id} not finished")
                    continue
                same_config = jax.tree.leaves(jax.tree.map(lambda x, y: x == y, config['curve_params'], run.config['curve_params']))
                same_config += [(config['rng_seed'] == run.config['rng_seed']), ( config['dataset'] == run.config['dataset'])]
                if np.all(same_config):
                    print(
                        f"Path optimization Run found; ID: {run.id} Name: {run.name}")
                    return run
            for run in wandb.Api().sweep(config['path_sweep_id']).runs:
                if run.state != 'finished':
                    continue
                same_config = jax.tree.leaves(jax.tree.map(
                    lambda x, y: x == y, config['curve_params'], run.config['curve_params']))
                same_config += [(config['rng_seed'] == run.config['rng_seed']),
                                (config['dataset'] == run.config['dataset'])]
                if np.sum(np.bitwise_not(same_config)) == 1:
                    print(f"No Run found. But this run differs only in one parameter; ID: {run.id} Name: {run.name}")
                    print(f"Config: \n{run.config}")
            raise Exception(f"No run found for config: \n{config}")

        run = find_run()
        for art in run.logged_artifacts():
            # art.download()
            if art.type == 'pytree':
                art_use = logger.use_artifact(art.name)
                file = art_use.get_entry(
                    f"{run.id}_params.npy").download(root="./art_tmp")
                params = jnp.load(file, allow_pickle=True).item()
            elif art.type == 'numpy':
                art_use = logger.use_artifact(art.name)
                rng_seed_id = run.config['rng_seed']
                ds_name = run.config['dataset']
                file_name = f'{rng_seed_id}_{ds_name}_data.npz'
                file = art_use.get_entry(file_name).download(root="./art_tmp")
                data_f = np.load(file)
                x, y, x_val, y_val, x_test, y_test = data_f['x'], data_f['y'], data_f[
                    'x_val'], data_f['y_val'], data_f['x_test'], data_f['y_test']

        return params, (x, y, x_val, y_val, x_test, y_test)
    params, (x, y, x_val, y_val, x_test, y_test) = find_params_and_data(config)
    k = config['curve_params']['k']

    if (config['dataset'] == 'default') or config['dataset'] == 'generate':
        model = MLPModel(**config['curve_params']['model_kwargs'])
    else:
        model = MLPModelUCI(**config['curve_params']['model_kwargs'])

    if config['sampling']['space_config']['space'] != 'fullspace':
        # %% generate transforamtion functions between weight-phi-lambda space
        if config['curve_params']['bias_ascending_dnn']:
            # get transformed bias params
            params = bias_ascending(params)
        # get design matrix from curve parameters stroed as pytree
        cp_w = pytree_to_matrix(params['params'], k)
        # define the transformation function between weight space and phi space
        t_phi_to_weight = OrthoSpan(cp_w)
        # control points in the varphi space
        cp_phi = t_phi_to_weight.inv(cp_w)
        print(f"Control points in phi space: {cp_phi.shape}")
        # define fuction to generate orthogonal space at location t
        t_lambda_to_phi, curve, d_bezier = init_t_lambda_to_phi(cp_phi, k,
                                                                epsilon=config['sampling']['space_config'].get(
                                                                    'epsilon', 5.),
                                                                tube_scale=config['sampling']['space_config'].get('tube_scale', 1.))  # for phi space tube_scale is irelevant thus set to default

    sampler = config['sampling']['sampler']
    num_chains = config['sampling']['num_chains']
    temperature = config['sampling']['temperature']
    num_warmup = config['sampling']['num_warmup']
    num_draws = config['sampling']['num_draws']
    number_forward_passes = config['sampling']['num_forward_passes']

    rng_key = random.PRNGKey(2+config['rng_seed'])
    # define sampling model

    def setup_model(init_key, space: str):
        initial_position = {'t': jax.random.uniform(init_key, (num_chains, 1), minval=0., maxval=1.),
                            'tube': jnp.zeros((num_chains, k-1))}
        if space == 'varphi':
            sampling_model = init_model_phi(
                model, params['params'], k, t_phi_to_weight,
                prior_scale=config['sampling']['space_config']['prior_scale'],
                dist_scale="homo" if config['curve_params']['optimize_distparams'] else np.exp(
                    params['dist_params']['log_scale'].item()))
            # transform same initial position to phi space
            initial_position = jnp.concatenate(
                [initial_position['t'], initial_position['tube']], axis=1)
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
                                             dist_scale="homo" if config['curve_params']['optimize_distparams'] else np.exp(
                                                 params['dist_params']['log_scale'].item())
                                             )
        elif space == 'line':
            tt = jnp.linspace(0., 1., 10_000)
            bezier_grad = jax.vmap(d_bezier)(tt)
            log_normalized_bezier_grad = jnp.log(jnp.trapezoid(
                jnp.linalg.norm(bezier_grad, axis=-1), tt))
            sampling_model = init_model_line(model,
                                             params['params'],
                                             k,
                                             t_lambda_to_phi,
                                             t_phi_to_weight,
                                             d_bezier=d_bezier,
                                             log_norm_d_bezier=log_normalized_bezier_grad,
                                             dist_scale="homo" if config['curve_params']['optimize_distparams'] else np.exp(
                                                 params['dist_params']['log_scale'].item())
                                             )
        elif space == 'fullspace':
            sampling_model = init_full_space(model, params,
                                             dist_scale="homo" if config['curve_params']['optimize_distparams'] else np.exp(
                                                        params['dist_params']['log_scale'].item())
            )
            # train 10 de members for fullspace sampling
            init_params = []
            fig, axes = plt.subplots(5, 2, figsize=(8, 4))
            for i, ax in enumerate(axes.flatten()):
                carry, (losses, valid_losses, epochs) = train_de(i+333+config['rng_seed'],
                                                                 model,
                                                                 x, y, x_val, y_val,
                                                                 **config['curve_params'])
                ax.plot(epochs, losses, label='train')
                ax.plot(epochs, valid_losses, label='valid')
                ax.set_ylim(-2, -1.)
                best_params = carry[4]
                init_params.append(best_params)
                ax.set_title(f"seed={i+333}")
            plt.tight_layout()
            logger.log(
                {'DE Loss': wandb.Image(fig)})
            initial_position = {}
            initial_position['weights'] = jnp.array([pytree_to_vec(p['params']) for p in init_params])
        else:
            raise ValueError(f"Space {space} not supported")
        if config['curve_params']['optimize_distparams']:
            if space == 'fullspace':
                initial_position['scale'] = jnp.array(
                    [jnp.exp(p['dist_params']['log_scale']) for p in init_params])
            else:
                initial_position['scale'] = jnp.exp(
                    jnp.full((num_chains,), params['dist_params']['log_scale']))
            # initial_position['scale'] = jnp.exp(jnp.full((num_chains,), params['dist_params']['log_scale']) + \
            # 0.1 * jax.random.normal(init_key, shape=(num_chains,)))
        return sampling_model, initial_position

    rng_key, init_key = jax.random.split(rng_key)
    sampling_model, initial_position = setup_model(
        init_key, config['sampling']['space_config']['space'])

    # prior predictive
    if (config['dataset'] == 'default') or config['dataset'] == 'generate':
        # generate prior predictive samples for µ
        x_pred = jnp.linspace(-3, 3, 1000).reshape(-1, 1)

        def on_prior_pred_sample(rng_key, i):
            rng_key, subkey = random.split(rng_key)
            traced = handlers.trace(handlers.seed(
                sampling_model, subkey)).get_trace(x_pred)
            prior_pred = traced['obs']['fn'].sample(subkey)
            return rng_key, prior_pred
        rng_key, prior_pred = jax.lax.scan(
            on_prior_pred_sample, rng_key, length=2000)

        # plot prior predictive
        fig, ax = plt.subplots()
        permute = jax.random.permutation(
            rng_key, jnp.arange(prior_pred.shape[0]))
        for i in permute[:20]:
            plt.plot(x_pred, prior_pred[i], alpha=0.5,
                     c='black', linewidth=0.5, label=r'$p(\theta)_i$')
        plt.plot(x_pred, prior_pred.mean(axis=(0)), c='red', label="µ")
        hpdi_ = hpdi(prior_pred, 0.9)
        plt.fill_between(x_pred.squeeze(), hpdi_[0], hpdi_[
            1], alpha=0.3, color='red', label="90% hpdi")
        plt.scatter(x, y, label="Train", s=10)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("prior predictive")
        # extract unique labels to plot the legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.tight_layout()
        logger.log(
            {'Prior Predictive': wandb.Image(fig)})

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
            def init_fn(num_warmup, warmup_key, initial_position):
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
            def init_fn(num_warmup, warmup_key, initial_position):
                adapt = blackjax.window_adaptation(
                    blackjax.rmhmc, logdensity_fn, target_acceptance_rate=0.8,  is_mass_matrix_diagonal=True, initial_step_size=0.2, num_integration_steps=number_forward_passes
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
            num_warmup_ = num_warmup*number_forward_passes//2
            num_draws_ = num_draws*number_forward_passes//2

            def init_fn(num_warmup, warmup_key, initial_position):
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
            num_draws_ = num_draws*number_forward_passes

            def init_wrapper(sigma_rw):
                def init_fn(num_warmup, warmup_key, initial_position):
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

    init_fn, get_kernel, num_warmup_, num_draws_ = get_sampler_methods(sampler)
    print(f"Start warmup for {sampler}")
    wall_time, rng_key, last_state, parameters = warmup_fn(
        rng_key, initial_position, init_fn, num_warmup_, num_chains)
    print("Warmup Wall time: {:.2f} s".format(wall_time))
    print(f"Start sampling ...")

    wall_time, rng_key, states, infos = run_inference(
        get_kernel, rng_key, last_state, parameters, num_draws_, num_chains)
    print("Sampling Wall time: {:.2f} s".format(wall_time))

    # extract samples for posterior predictive (Subset for mclmc and random_walk)
    selected_states = states.position
    if sampler == 'mclmc':
        selected_states = jax.tree.map(
            lambda x: x[:, ::number_forward_passes//2], states.position)
        if config['sampling']['space_config']['space'] == 'varphi':
            num_value_grad = states.position['varphi'].shape[1]/num_draws
        else:
            num_value_grad = states.position['t'].shape[1]/num_draws*2
    elif sampler == 'random_walk':
        selected_states = jax.tree.map(
            lambda x: x[:, ::number_forward_passes], states.position)
        num_value_grad = number_forward_passes
    else:
        num_value_grad = float(
            infos.num_integration_steps.sum()/(num_chains*num_draws))
    # print(f"transforme samples... e.g. t with shape: {selected_states['t'].shape}")
    posterior = jax.lax.map(lambda x: jax.vmap(
        post_proc_fun)(x), selected_states)
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

    if config['dataset'] == 'bikesharing':
        samples = az.from_dict(posterior=posterior)
        summary = az.summary(samples)
        print(summary)
        w_tabel = wandb.Table(dataframe=summary)
        logger.log({'summary': w_tabel})
        performance = None
    else:
        log_likelihood = log_like_fn(x, y)
        samples = az.from_dict(posterior=posterior, log_likelihood={
            'obs': log_likelihood})
        summary = az.summary(samples)
        print(summary)
        w_tabel = wandb.Table(dataframe=summary)
        logger.log({'summary': w_tabel})

        # elpd_loo and p_loo
        performance = az.loo(samples)
        print(performance)

    # posterior predictive valid with data batching

    def compute_metrics(rng_key, x, y):
        # lppd
        log_like = log_like_fn(x, y).reshape(-1, len(y))
        lppd = jnp.mean(jax.scipy.special.logsumexp(
            log_like, axis=0) - jnp.log(log_like.shape[0]))
        del log_like
        # rmse
        rng_key, preds = prediction(rng_key, x)
        rmse = jnp.sqrt(jnp.mean((preds.mean((0, 1)) - y) ** 2))
        del preds
        return rng_key, lppd, rmse

    rng_key, elpd_valid, rmse_valid = compute_metrics(rng_key, x_val, y_val)
    rng_key, elpd_test, rmse_test = compute_metrics(rng_key, x_test, y_test)

    # log metrics to wandb
    summary_var = az.summary(samples, var_names=["varphi"])
    res = dict(time_s=wall_time,
               r_hat_max=summary_var['r_hat'].max(),
               r_hat_mean=summary_var['r_hat'].mean(),
               ess_min=summary_var['ess_bulk'].min(),
               ess_mean=summary_var['ess_bulk'].mean(),
               num_forward=num_value_grad,
               acceptance_rate=infos.acceptance_rate.mean() if sampler != 'mclmc' else 1.,
               p_loo=performance.p_loo if performance is not None else np.inf,
               elpd_loo=performance.elpd_loo if performance is not None else np.inf,
               elpd_test=elpd_test,
               rmse_test=rmse_test,
               elpd_valid=elpd_valid,
               rmse_valid=rmse_valid
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
    print("Save samples")
    samples.to_netcdf(f"tmp_files/{logger.id}_samples.nc")
    print("Save state")
    jnp.savez(f"tmp_files/{logger.id}_sampler_state_and_info.npz",
              dict(states=states, infos=infos))
    artifact.add_file(f"tmp_files/{logger.id}_samples.nc")
    artifact.add_file(f"tmp_files/{logger.id}_sampler_state_and_info.npz")

    print("Log artifact")
    logger.log_artifact(artifact)
    print("finish run")
    logger.finish()


def test_main():
    print(wandb.config)


if __name__ == '__main__':
    # Add requirement for wandb core
    wandb.require("core")
    os.makedirs('tmp_files', exist_ok=True)


    # config = {
    #         'curve_params': {
    #             'k': 2,
    #             'model_kwargs': {
    #                 'depth': 3,
    #                 'width': 16,
    #                 'activation': "relu"
    #             },
    #             'n_samples': 20,
    #             'lr': 0.001,
    #             'num_epochs': 100_000,
    #             'use_validation': True,
    #             'out_scale': 0.05,
    #             'optimize_distparams': True,
    #             'uniform_t_optimize': 'False'
    #             'bias_ascending_dnn': 'False'
    #         },
    #         'sampling': {
    #             'space_config': 
    #                 # {
    #                 #     'space': "lambda",
    #                 #     'prior_scale': 0.5,
    #                 #     'tube_scale': 0.1,
    #                 #     'epsilon': 25.0,
    #                 #     'prior_correct': False
    #                 # },
    #                 {
    #                     'space': "varphi",
    #                     'prior_scale': 1.0
    #                 },
    #             'num_chains': 10,
    #             'temperature': 1.0,
    #             'num_warmup': 1000,
    #             'num_draws': 1000,
    #             'sampler': "mclmc",
    #             'num_forward_passes': 100
    #         },
    #         'rng_seed': 10,
    #         'path_sweep_id': "ddold/subspace_toy_reg/7t2s9d4p",
    #         'dataset': "generate" }
    
    # logger = wandb.init(project="subspace_toy_reg",
    #                     name="test", entity="ddold", config=config)
    # single_run(logger, config)

        # config = {
        #     'curve_params': {'k': 10,
        #                      'model_kwargs': {'depth': 3, 'width': 10, 'activation': 'relu'},
        #                      'n_samples': 10,
        #                      'lr': 0.01,
        #                      'num_epochs': 1000,
        #                      'use_validation': False,
        #                      'out_scale': 0.05,
        #                      'optimize_distparams': False
        #                     },
        #     'sampling': {'space_config': {'space': 'lambda',
        #                                   'prior_scale': .5,
        #                                   'tube_scale': 1.,
        #                                   'epsilon': 5.,
        #                                   'prior_correct': False},
        #                  'num_chains': 10,
        #                  'temperature': 1.,
        #                  'num_warmup': 100,
        #                  'num_draws': 1000,
        #                  'sampler': 'nuts'
        #                  },
        #     'rng_seed': 0,
        #     'nuts_sweep_id': '1',
        #     'dataset': 'bikesharing',
        # }
        # logger = wandb.init(project="subspace_test", name="test", entity="ddold", config=config)
        # single_run(logger, config, n_samples_nuts=370)

        # sweep_config = {
        #     'method': 'grid',
        #     "metric": {"goal": "minimize", "name": "r_hat_max"},
        #     'parameters': {'curve_params': {'parameters': {'k': {'value': 10},
        #                                                    'model_kwargs': {'value': {'depth': 3, 'width': 10, 'activation': 'elu'}},
        #                                                    'n_samples': {'value': 10},
        #                                                    'lr': {'value': 0.001},
        #                                                    'num_epochs': {'value': 10000}}},
        #                    'sampling': {'parameters': {'space_config':
        #                                                {'values':
        #                                                 [{'space': 'varphi', 'prior_scale': 1.},
        #                                                  {'space': 'lambda', 'prior_scale': .5, 'tube_scale':1., 'epsilon': 5.},
        #                                                  {'space': 'lambda', 'prior_scale': .5, 'tube_scale':.1, 'epsilon': 5.},
        #                                                  ]},
        #                                                'num_chains': {'value': 10},
        #                                                'temperature': {'value': 1.},
        #                                                'num_warmup': {'value': 1000},
        #                                                'num_draws': {'value': 1000},
        #                                                'sampler': {'values': ['nuts', 'mclmc']},
        #                                                 }},
        #                     "seed": {"value ": 0}
        #                    }}

        # sweep_config = {
        #     'method': 'grid',
        #     "metric": {"goal": "minimize", "name": "r_hat_max"},
        #     'parameters': {'curve_params': {'parameters': {'k': {'values': [5, 10, 20, 30]},
        #                                                    'model_kwargs': {'parameters':{
        #                                                           'depth': {'value': 3},
        #                                                           'width': {'value': 10},
        #                                                           'activation': {'values': ['relu', 'elu', 'tanh']}
        #                                                    }},
        #                                                    'n_samples': {'value': 10},
        #                                                    'lr': {'value': 0.001},
        #                                                    'num_epochs': {'value': 10000}}},
        #                    'sampling': {'parameters': {'space_config':
        #                                                {'values':
        #                                                 [{'space': 'varphi', 'prior_scale': 1.},
        #                                                  {'space': 'lambda', 'prior_scale': .5,
        #                                                      'tube_scale': 1., 'epsilon': 25.},
        #                                                  {'space': 'lambda', 'prior_scale': .5,
        #                                                      'tube_scale': .1, 'epsilon': 25.},
        #                                                  ]},
        #                                                'num_chains': {'value': 10},
        #                                                'temperature': {'values': [1., 3.]},
        #                                                'num_warmup': {'value': 1000},
        #                                                'num_draws': {'value': 1000},
        #                                                'sampler': {'values': ['nuts', 'random_walk', 'mclmc']},
        #                                                }},
        #                    'rng_seed': {'values': [4,]},
        #                    'nuts_sweep_id': {'value': '1'},
        #                    'prior_correct': {'value': False},
        #                    }}

        # sweep_config = {
        #     'method': 'grid',
        #     "metric": {"goal": "minimize", "name": "r_hat_max"},
        #     'parameters': {'curve_params': {'parameters': {'k': {'value': 10},
        #                                                    'model_kwargs': {'parameters': {
        #                                                        'depth': {'value': 3},
        #                                                        'width': {'value': 10},
        #                                                        'activation': {'values': ['elu']}
        #                                                    }},
        #                                                    'n_samples': {'value': 10},
        #                                                    'lr': {'value': 0.001},
        #                                                    'num_epochs': {'value': 10000}}},
        #                    'sampling': {'parameters': {'space_config':
        #                                                {'values':
        #                                                 [{'space': 'varphi', 'prior_scale': 1.},
        #                                                  {'space': 'lambda', 'prior_scale': .5,
        #                                                     'tube_scale': .1, 'epsilon': 5.},
        #                                                  ]},
        #                                                'num_chains': {'value': 10},
        #                                                'temperature': {'value': 1.},
        #                                                'num_warmup': {'value': 1000},
        #                                                'num_draws': {'value': 1000},
        #                                                'sampler': {'values': ['nuts', 'random_walk', 'mclmc']},
        #                                                }},
        #                    'rng_seed': {'value': 0}
        #                    }}

        # sweep_id = wandb.sweep(sweep=sweep_config, entity="ddold", project="subspace_sampling_compare")
        # wandb.agent(sweep_id, function=main)

    main()
