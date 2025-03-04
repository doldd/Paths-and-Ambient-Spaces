import jax.experimental
import jax.numpy as jnp
from jax import jit, grad
from scipy.special import binom
import numpy as np
import blackjax
import time

__all__ = ['init_curve_frame', 'init_curve_frame_cp',
           'ortho_at_one_t', 'setup_inference_chain']



def init_curve_frame(d_bezier, k, epsilon=1.):
    """
    Initializes the curve frame by finding U-Turns and generating orthogonal frames along the bezier curve.

    Args:
        d_bezier (callable): The first derivative function of the Bézier curve.
        k (int): The subspace dimension. (min of control_points.shape[0]-1 and control_points.shape[1]).
        epsilon (float, optional): The angle threshold for detecting U-Turns. Defaults to 1° => U-Turn > 45°-1°=+-44°. 

    Returns:
        tuple: A tuple containing the cut points along the bezier curve and the corresponding orthogonal frames.

    """

    # find U-Turns
    # find the change in direction of the tangent vector of larger than +-45 degrees
    t_grid = jnp.linspace(0., 1., 100_000)
    tangents = d_bezier(t_grid)
    tangents = tangents / jnp.linalg.norm(tangents, axis=-1, keepdims=True)

    def scan_fn(current_tangent, tangent):
        dot = jnp.dot(current_tangent, tangent)
        lim = np.cos(np.deg2rad(45 - abs(epsilon)))  # +-45 degree
        # if u-turn set current_tangent to the new one
        current_tangent = jnp.where(dot > lim, current_tangent, tangent)
        # 0 if not u-turn, 1 if u-turn
        return current_tangent, jnp.where(dot > lim, 0, 1)

    _, idx = jax.lax.scan(scan_fn, tangents[0], tangents)
    t_cut = t_grid[idx == 1]
    t_cut = jnp.concatenate((jnp.array([0.]), t_cut))

    def frenet_frame_at(t, d_bezier):
        # tangent vector
        d = d_bezier(t)
        v1 = d / jnp.linalg.norm(d, axis=-1, keepdims=True)

        ortho = v1
        dev_fn = d_bezier
        for i in range(1, k):
            dev_fn = jax.jacfwd(dev_fn)  # use second, third, ... derivative

            # gram-schmidt orthogonalization
            proj = jnp.zeros_like(ortho[0])
            # test vectors comes from derivative of bezier curve
            v = dev_fn(t).squeeze()
            for u in ortho:  # ortho vectors
                proj += jnp.dot(u, v) * u
            v_ortho = v - proj  # subtract all orthogonal projections
            v_ortho /= jnp.linalg.norm(v_ortho)  # normalize
            ortho = jnp.concatenate((ortho, v_ortho[None, :]), axis=0)
        return ortho
    init_space_at_t0 = frenet_frame_at(jnp.array([1e-4]), d_bezier)
    print("space_at_t0 shape", init_space_at_t0.shape)
    assert jnp.all(jnp.isnan(init_space_at_t0)) == False, "NaN in init_space_at_t0"

    # at each t_i we have to generate the orthogonal space
    # we can use the previous orthogonal space to generate the new one at t+1
    def ortho_frame_from_previous_frame(ortho_t0, t):
        # tangent vector
        d = d_bezier(t)
        v1 = d / jnp.linalg.norm(d, axis=-1, keepdims=True)

        ortho_t1 = v1

        for i in range(1, k):
            # ortho_t0 is the ortho_t space at time t to generate ortho_{t+1}
            test_v = ortho_t0[i]
            # gram-schmidt orthogonalization
            proj = jnp.zeros_like(ortho_t1[0])
            # test vectors comes from derivative of bezier curve
            for u in ortho_t1:  # ortho vectors
                proj += jnp.dot(u, test_v) * u
            v_ortho = test_v - proj  # subtract all orthogonal projections
            v_ortho /= jnp.linalg.norm(v_ortho)  # normalize
            ortho_t1 = jnp.concatenate((ortho_t1, v_ortho[None, :]), axis=0)
        return ortho_t1, ortho_t1

    last_state, ortho_at_tcut = jax.lax.scan(
        ortho_frame_from_previous_frame, init_space_at_t0, t_cut)
    print(
        f"Stores {len(t_cut)} orthogonal frames  \nwith ortho frame shape {ortho_at_tcut.shape}")
    return t_cut, ortho_at_tcut


def init_curve_frame_cp(d_bezier, control_points, epsilon=1.):
    """
    Initializes the curve frame by finding U-Turns and generating orthogonal frames along the bezier curve.

    Args:
        d_bezier (callable): The first derivative function of the Bézier curve.
        control_points (n+1,k): controlpoints of corresponding bezier curve. (Must be the same as in d_bezier).
        epsilon (float, optional): The angle threshold for detecting U-Turns. Defaults to 1° => U-Turn > 45°-1°=+-44°. 

    Returns:
        tuple: A tuple containing the cut points along the bezier curve and the corresponding orthogonal frames.

    """
    k = min(control_points.shape[0]-1, control_points.shape[1])
    # find U-Turns
    # find the change in direction of the tangent vector of larger than +-45 degrees
    t_grid = jnp.linspace(0., 1., 100_000)
    tangents = jax.vmap(d_bezier)(t_grid)
    tangents = tangents / jnp.linalg.norm(tangents, axis=-1, keepdims=True)

    def scan_fn(current_tangent, tangent):
        dot = jnp.dot(current_tangent, tangent)
        lim = np.cos(np.deg2rad(45 - abs(epsilon)))  # +-45 degree
        # if u-turn set current_tangent to the new one
        current_tangent = jnp.where(dot > lim, current_tangent, tangent)
        # 0 if not u-turn, 1 if u-turn
        return current_tangent, jnp.where(dot > lim, 0, 1)

    _, idx = jax.lax.scan(scan_fn, tangents[0], tangents)
    t_cut = t_grid[idx == 1]
    t_cut = jnp.concatenate((jnp.array([0.]), t_cut))

    def frame_at_t0(control_points):
        # tangent vector
        d = d_bezier(jnp.array(0.))
        v1 = d / jnp.linalg.norm(d, axis=-1, keepdims=True)

        ortho_t1 = v1[None, :]
        for test_v in control_points[1:-1]:  # test vectors
            # gram-schmidt orthogonalization
            proj = jnp.zeros_like(ortho_t1[0])
            # test vectors comes from derivative of bezier curve
            for u in ortho_t1:  # ortho vectors
                proj += jnp.dot(u, test_v) * u
            v_ortho = test_v - proj  # subtract all orthogonal projections
            v_ortho /= jnp.linalg.norm(v_ortho)  # normalize
            ortho_t1 = jnp.concatenate((ortho_t1, v_ortho[None, :]), axis=0)
        return ortho_t1

    init_space_at_t0 = frame_at_t0(control_points)
    print("space_at_t0 shape", init_space_at_t0.shape)
    assert jnp.any(jnp.isnan(init_space_at_t0)
                   ) == False, "NaN in init_space_at_t0"

    # at each t_i we have to generate the orthogonal space
    # we can use the previous orthogonal space to generate the new one at t+1
    def ortho_frame_from_previous_frame(ortho_t0, t):
        # tangent vector
        d = d_bezier(t)
        v1 = d / jnp.linalg.norm(d, axis=-1, keepdims=True)

        ortho_t1 = v1[None, :]

        for i in range(1, k):
            # ortho_t0 is the ortho_t space at time t to generate ortho_{t+1}
            test_v = ortho_t0[i]
            # gram-schmidt orthogonalization
            proj = jnp.zeros_like(ortho_t1[0])
            # test vectors comes from derivative of bezier curve
            for u in ortho_t1:  # ortho vectors
                proj += jnp.dot(u, test_v) * u
            v_ortho = test_v - proj  # subtract all orthogonal projections
            v_ortho /= jnp.linalg.norm(v_ortho)  # normalize
            ortho_t1 = jnp.concatenate((ortho_t1, v_ortho[None, :]), axis=0)
        return ortho_t1, ortho_t1

    last_state, ortho_at_tcut = jax.lax.scan(
        ortho_frame_from_previous_frame, init_space_at_t0, t_cut)
    print(
        f"Stores {len(t_cut)} orthogonal frames  \nwith ortho frame shape {ortho_at_tcut.shape}")
    return t_cut, ortho_at_tcut


def get_bin(t, t_bins):
    """
    Get the index of the bin that a given value falls into.

    Parameters:
    t (float): The value to be assigned to a bin.
    t_bins (ndarray): An array of bin boundaries.

    Returns:
    int: The index of the bin that the value falls into.
    """
    bin_idx = jnp.argmax(t_bins > t) - 1
    bin_idx = jnp.where(t < t_bins[0], 0, bin_idx)
    bin_idx = jnp.where(t > t_bins[-1], len(t_bins)-1, bin_idx)
    return bin_idx


def ortho_at_one_t(t, d_bezier, t_cut, ortho_at_tcut, k):
    """
    Compute the orthogonal vectors at a given time point t.

    Args:
        t (float): The single time point.
        d_bezier (callable): A function that computes the first order derivative of the bezier curve.
        t_cut (ndarray): An array of time points where the bezier curve is cut.
        ortho_at_tcut (ndarray): An array of orthogonal vectors at each cut point.

    Returns:
        ndarray: An array of orthogonal vectors at the given time point t. Shape(k, subspace_dim)
    """
    # tangent vector
    d = d_bezier(t)
    v1 = d / jnp.linalg.norm(d, axis=-1, keepdims=True)

    ortho = jnp.zeros((k, k))
    ortho = ortho.at[0].set(v1)
    bin_idx = get_bin(t, t_cut)

    def outer_loop(ortho, i):
        v = ortho_at_tcut[bin_idx, i]

        def inner_loop(proj, u):
            return proj + jnp.dot(u, v) * u, None
        proj = jnp.zeros_like(v1)
        # normaly loop only over ortho[:i] but this is not possible in jax jit ==> loop over all with zeros doesnt change the result
        proj, _ = jax.lax.scan(inner_loop, proj, ortho)
        # jax.debug.print(f"proj: {proj.mean()}")
        v_ortho = v - proj  # subtract all orthogonal projections
        v_ortho /= jnp.linalg.norm(v_ortho)  # normalize

        ortho = ortho.at[i].set(v_ortho)
        return ortho, None

    ortho, _ = jax.lax.scan(outer_loop, ortho, jnp.arange(1, k))
    return ortho


# def ortho_at_one_t(t, d_bezier, t_cut, ortho_at_tcut, k):
#     """
#     Compute the orthogonal vectors at a given time point t.

#     Args:
#         t (float): The single time point.
#         d_bezier (callable): A function that computes the first order derivative of the bezier curve.
#         t_cut (ndarray): An array of time points where the bezier curve is cut.
#         ortho_at_tcut (ndarray): An array of orthogonal vectors at each cut point.

#     Returns:
#         ndarray: An array of orthogonal vectors at the given time point t. Shape(k, subspace_dim)
#     """
#     # tangent vector
#     d = d_bezier(t)
#     v1 = d / jnp.linalg.norm(d, axis=-1, keepdims=True)

#     ortho = v1[None, :]
#     bin_idx = get_bin(t, t_cut)
#     for i in range(1, k):
#         # gram-schmidt orthogonalization
#         proj = jnp.zeros_like(ortho[0])
#         # test vectors comes from derivative of bezier curve
#         # ortho_t0 is the ortho_t space at time t to generate ortho_{t+1}
#         v = ortho_at_tcut[bin_idx, i]
#         for u in ortho:  # ortho vectors
#             proj += jnp.dot(u, v) * u
#         v_ortho = v - proj  # subtract all orthogonal projections
#         v_ortho /= jnp.linalg.norm(v_ortho)  # normalize
#         ortho = jnp.concatenate((ortho, v_ortho[None, :]), axis=0)
#     return ortho


# usage
# t_cut, ortho_at_tcut = init_curve_frame(d_bezier)
# space = generate_ortho_parallel_transport(t, d_bezier, t_cut, ortho_at_tcut)


def setup_inference_chain(mode, num_chains):
    def warmup_fn(rng_key, initial_position, init_fn, num_warmup):
        # run warmup adaption
        rng_key, warmup_key = jax.random.split(rng_key)
        warmup_key = jax.random.split(warmup_key, num_chains)

        time_start = time.time()

        if mode == "PMAP":
            run = jax.pmap(init_fn, in_axes=(0, 0, None), static_broadcasted_argnums=(
                2,), devices=jax.devices('cpu'))
        elif mode == "VMAP":
            run = jax.vmap(init_fn, in_axes=(0, 0, None))
        elif mode == "SEQUENTIAL":
            def run(rng_key, initial_position, num_warmup):
                def scan_fn(carry, xs):
                    rng_key, init_pos = xs
                    return None, init_fn(rng_key, init_pos, num_warmup)
                carry, (last_state, parameters) = jax.lax.scan(
                    scan_fn, None, (rng_key, initial_position))
                return last_state, parameters

        last_state, parameters = run(warmup_key, initial_position, num_warmup)
        jax.block_until_ready(last_state)
        time_ = time.time() - time_start
        return time_, rng_key, last_state, parameters


    def run_inference(kernel, rng_key, last_state, parameters, num_samples):
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
        if mode == "PMAP":
            inference_loop_multiple_chains = jax.pmap(inference_loop,
                                                    in_axes=(0, 0, 0, None),
                                                    static_broadcasted_argnums=(
                                                        3,),
                                                    devices=jax.devices('cpu'))
        elif mode == "VMAP":
            inference_loop_multiple_chains = jax.vmap(inference_loop, in_axes=(
                0, 0, 0, None))
        elif mode == "SEQUENTIAL":
            def inference_loop_multiple_chains(rng_key, parameters, initial_state, num_samples):
                def scan_fn(carry, xs):
                    rng_key, parameters, initial_state = xs
                    return None, inference_loop(rng_key, parameters, initial_state, num_samples)
                carry, (state, info) = jax.lax.scan(
                    scan_fn, None, (rng_key, parameters, initial_state))
                return state, info

            # inference_loop_multiple_chains = lambda rng_key, parameters, initial_state, num_samples: jax.tree.map(partial(inference_loop, num_samples=num_samples), rng_key, parameters, initial_state)

        # jax.debug.print(f"params: {parameters['max_num_doublings']}")
        time_start = time.time()
        pmap_states, pmap_infos = inference_loop_multiple_chains(
            sample_keys, parameters, last_state, num_samples)
        jax.block_until_ready(pmap_states)
        time_ = time.time() - time_start
        return time_, rng_key, pmap_states, pmap_infos
    
    return warmup_fn, run_inference
