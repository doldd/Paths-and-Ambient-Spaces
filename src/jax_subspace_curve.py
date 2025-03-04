import jax.numpy as jnp
from jax import jit, grad, random
import jax
import optax
from scipy.special import binom
import numpy as np
from functools import partial
from jax import custom_jvp
from typing import Callable
from numpyro import distributions as dist
from jaxopt import Bisection
from jax.scipy.special import gammaln
from flax import traverse_util
# from flax.core import freeze, unfreeze

__all__ = ['OrthoSpan', 'SubspaceModel', 'pytree_to_matrix', 'pytree_to_vec',
           'vec_to_pytree', 'bezier_curve', 'UniformTSubspace', 'UniformTCategory']

@custom_jvp
def pos_pow(x, pow):
    # we expect only positive powers => multiple grad is 0 istead of nan
    return jnp.where(pow < 0, 0, jnp.pow(x, pow))

def pos_pow_jvp(primals, tangents):
    x, pow = primals
    grad = pow * pos_pow(x, pow-1)
    grad *= tangents[0]
    return pos_pow(x, pow), grad
pos_pow.defjvp(pos_pow_jvp)


def comb(N, k):
  return jnp.exp(gammaln(N + 1) - gammaln(k + 1) - gammaln(N - k + 1)).round()

def bezier_coeff_fn(num_bends) -> Callable:
    range = jnp.arange(0, num_bends)
    rev_range = jnp.arange((num_bends - 1), -1, -1)
    binom_ = comb(num_bends - 1, jnp.arange(num_bends))

    def wrapper(t):
        return binom_ * pos_pow(t, range) * pos_pow((1.0 - t), rev_range)
    return wrapper

def bezier_curve(num_bends, cp):
    """
    Compute a Bezier curve and its derivative.

    Args:
        num_bends (int): The number of bends in the curve.
        cp (ndarray): Control points of the curve.

    Returns:
        tuple: A tuple containing two functions:
            - f (function): The Bezier curve function that takes a parameter `t`(only single t value) and returns the corresponding point on the curve.
            - derivative_bezier (function): The derivative of the Bezier curve function that takes a parameter `t` and returns the derivative at that point.
    """
    bezier_coeff_inv = bezier_coeff_fn(num_bends - 1)
    bezier_coeff = bezier_coeff_fn(num_bends)

    def derivative_bezier(t):
        n = cp.shape[0] - 1
        coeff = bezier_coeff_inv(t)
        cp_diff = cp[1:] - cp[:-1]
        return jnp.einsum('j,j...->...', coeff, cp_diff) * n

    # forward function
    @custom_jvp
    def f(t):
        p_t = jnp.einsum('k,k...->...', bezier_coeff(t), cp)
        return p_t

    def f_jvp(primals, tangents):
        t = primals[0]
        tangent_out = derivative_bezier(t)
        primal_out = f(t)
        return primal_out, tangent_out*tangents[0]

    f.defjvp(f_jvp)
    return f, derivative_bezier


class OrthoSpan:
    """
    A class that defines the transformation between the subspace spanned by the control points and the weight space.

    Args:
        cp (torch.Tensor): A tensor representing the set of vectors that span the subspace. shape (k+1, dim)

    Attributes:
        mu (torch.Tensor): The mean vector of the input vectors. shape (dim,)
        pi (torch.Tensor): The projection matrix from the subspace to the weight space. shape (k, dim)
        pi_inv (torch.Tensor): The inverse projection matrix. shape (dim, k)

    Methods:
        __call__(self, p): Applies the projection matrix to the input tensor.
        inv(self, p): Applies the inverse projection matrix to the input tensor.

    """

    def __init__(self, cp):
        super().__init__()
        subspace_dim = cp.shape[0] - 1
        self.mu = cp.mean(0)
        cp -= self.mu
        u, self.s, vh = jnp.linalg.svd(cp, full_matrices=False) # u: (k+1, k+1), s: (k+1,), vh: (k+1, dim)
        self.pi = self.s[:subspace_dim, None] * vh[:subspace_dim]
        self.pi_inv = vh[:subspace_dim].T @ jnp.diag(1/self.s[:subspace_dim])

    def __call__(self, p) -> jnp.ndarray:
        """
        Applies the projection matrix to the input tensor.

        Args:
            p (torch.Tensor): The input tensor. shape (n, k)

        Returns:
            torch.Tensor: The tensor after applying the projection matrix.

        """
        return p @ self.pi + self.mu

    def inv(self, p) -> jnp.ndarray:
        """
        Applies the inverse projection matrix to the input tensor.

        Args:
            p (torch.Tensor): The input tensor. shape (n, dim)

        Returns:
            torch.Tensor: The tensor after applying the inverse projection matrix.

        """
        return (p - self.mu) @ self.pi_inv
    

# class OrthoSpanQR(OrthoSpan):
#     def __init__(self, cp):
#         subspace_dim = cp.shape[0] - 1
#         self.mu = cp.mean(0)
#         cp -= self.mu
#         Q, R = jnp.linalg.qr(cp)
#         self.pi = self.s[:subspace_dim, None] * vh[:subspace_dim]
#         self.pi_inv = vh[:subspace_dim].T @ jnp.diag(1/self.s[:subspace_dim])


class BezierTspaceUnifrom(dist.Uniform):
    def __init__(self, d_bezier, validate_args=None):
        self.d_bezier = d_bezier
        tt = jnp.linspace(0., 1., 1_000)
        bezier_grad = jax.vmap(d_bezier)(tt)
        length = jnp.trapezoid(jnp.linalg.norm(bezier_grad, axis=-1), tt)
        self.total_length = length
        print("Total length: ", length)
        super(BezierTspaceUnifrom, self).__init__(validate_args=validate_args)

    @staticmethod
    def arc_length(t, d_bezier):
        tt = jnp.linspace(0., t, 1_000)
        grads = jax.vmap(d_bezier)(tt)
        return jnp.trapezoid(jnp.linalg.norm(grads, axis=-1), tt)

    def t_from_length(self, length):
        def loss(t):
            return self.arc_length(t, self.d_bezier) - length
        root_solver = Bisection(loss, lower=0., upper=1.,
                                check_bracket=False, tol=1e-12, maxiter=30)
        return root_solver.run()

    def sample(self, key, sample_shape=()):
        s = random.uniform(key, sample_shape) * self.total_length
        shape = s.shape
        t = jax.vmap(self.t_from_length)(s.flatten())
        return t.params.reshape(shape)
        # return t

    def log_prob(self, value):
        absdet = jnp.linalg.norm(jax.vmap(self.d_bezier)(value), axis=-1)
        return jnp.log(absdet) - jnp.log(self.total_length)

    def cdf(self, value):
        raise NotImplementedError

    def icdf(self, q):
        raise NotImplementedError

    def mean(self):
        raise NotImplementedError

    def variance(self):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError


class SubspaceModel:
    """
    A class representing a subspace model which can be used to train the curve model.

    Parameters:
    - model: The underlying model used in the subspace model.
    - k: The number of parameter sets to sample.

    Attributes:
    - model: The underlying model used in the subspace model.
    - k: The number of parameter sets to sample.
    - bezier: The BezierCoeff object for computing Bezier coefficients.

    Methods:
    - init_params(x): Initializes the parameters of the model.
    - __call__(params, t, x): Computes the output of the model for given parameters, time, and input.
    - nll(params, t, x, y): Computes the negative log-likelihood loss of the model.
    - compute_loss(key, params, x, y, n_samples=1): Computes the loss of the model.
    - train_step(key, params, x, y, opt_state, optimizer): Performs a single training step.

    """

    def __init__(self, model, k, n_samples=1, out_scale=0.05, optimize_distparams=False):
        self.model = model
        self.k = k
        self.bezier = jax.vmap(bezier_coeff_fn(k + 1))
        self.n_samples = n_samples
        self.out_scale = out_scale
        self.optimize_distparams = optimize_distparams


    def init_params(self, key, x):
            """
            Initializes the parameters of the model.

            Parameters:
            - key: The random key for parameter initialization.
            - x: The input data.

            Returns:
            - The initialized parameters.

            """
            keys = random.split(key, self.k+1)
            # list of params and variables (such as batch statistics)
            params_and_var = [self.model.init(k, x) for k in keys]
            # now pytree of stacked params (each leaf has leading dim of k+1)
            stacked_params_var = jax.tree.map(
                lambda *x: jnp.stack(x), *params_and_var)
            
            stacked_params_var['dist_params'] = {'log_scale': jnp.log(
                jnp.array(self.out_scale))}
            # only trainable params from pytree
            return stacked_params_var

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
        # forward pass per sample
        out = jax.vmap(self.model.apply, in_axes=(0, None))(sample_param, x)
        return out

    @partial(jit, static_argnums=(0,))
    def nll(self, params, t, x, y):
        """
        Computes the negative log-likelihood loss of the model.

        Parameters:
        - params: The parameters of the model.
        - t: The time parameter.
        - x: The input data.
        - y: The target output.

        Returns:
        - The negative log-likelihood loss.

        """
        out = self(params['params'], t, x)  # shape (n_samples, n_data, output_dim)
        nll = -jax.scipy.stats.norm.logpdf(y,
                                           loc=out.squeeze(axis=-1), scale=jnp.exp(params['dist_params']['log_scale'])+1e-8)
        return nll

    @partial(jit, static_argnums=(0, 5))
    def compute_loss(self, key, params, x, y, n_samples=1):
        """
        Computes the loss of the model.

        Parameters:
        - key: The random key for generating random numbers.
        - params: The parameters of the model.
        - x: The input data.
        - y: The target output.
        - n_samples: The number of samples to use for computing the loss.

        Returns:
        - The computed loss.

        """
        # Sample t param of Bezier curve
        t = random.uniform(key, (n_samples,), minval=0., maxval=1.)
        loss = self.nll(params, t, x, y)
        return loss.mean()
    
    @partial(jit, static_argnums=(0, 5))
    def compute_loss_natural_t(self, key, params, x, y, n_samples=1):
        """
        Computes the loss of the model.

        Parameters:
        - key: The random key for generating random numbers.
        - params: The parameters of the model.
        - x: The input data.
        - y: The target output.
        - n_samples: The number of samples to use for computing the loss.

        Returns:
        - The computed loss.

        """
        # Sample t param of Bezier curve

        t = random.uniform(key, (n_samples,), minval=0., maxval=1.)
        loss = self.nll(params, t, x, y).mean(-1)
        # jax.debug.print(f'loss shape: {loss.shape}')

        cp_w = pytree_to_matrix(params['params'], self.k)
        curve, d_bezier = bezier_curve(self.k+1, cp_w)
        tt = jnp.linspace(0., 1., 10_000)
        bezier_grad = jax.vmap(d_bezier)(tt)
        normalized_grad = jnp.trapezoid(
            jnp.linalg.norm(bezier_grad, axis=-1), tt)
        grads = jax.vmap(d_bezier)(t)
        weights = jnp.linalg.norm(grads, axis=-1) / normalized_grad

        # t_dist = BezierTspaceUnifrom(d_bezier)
        # t = t_dist.sample(key, (n_samples,))

        weighted_loss = jnp.einsum('n,n->', weights, loss)/n_samples
        return weighted_loss

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
        return loss, params, opt_state
    

class UniformTSubspace(SubspaceModel):
    # slow implementation and not working because of gradient computation
    # @partial(jit, static_argnums=(0, 5))
    # def compute_loss(self, key, params, x, y, n_samples=1):
    #     """
    #     Computes the loss of the model.

    #     Parameters:
    #     - key: The random key for generating random numbers.
    #     - params: The parameters of the model.
    #     - x: The input data.
    #     - y: The target output.
    #     - n_samples: The number of samples to use for computing the loss.

    #     Returns:
    #     - The computed loss.

    #     """
    #     # Sample t param of Bezier curve
    #     cp_w = pytree_to_matrix(params['params'], self.k)
    #     curve, d_bezier = bezier_curve(self.k+1, cp_w)
    #     t_dist = BezierTspaceUnifrom(d_bezier)
    #     t = t_dist.sample(key, (n_samples,))
    #     loss = self.nll(params, t, x, y)
    #     return loss.mean()
    
    @partial(jit, static_argnums=(0, 5))
    def compute_loss(self, key, params, x, y, n_samples=1):
        """
        Computes the loss of the model.

        Parameters:
        - key: The random key for generating random numbers.
        - params: The parameters of the model.
        - x: The input data.
        - y: The target output.
        - n_samples: The number of samples to use for computing the loss.

        Returns:
        - The computed loss.

        """
        # Sample t param of Bezier curve

        t = random.uniform(key, (n_samples,), minval=0., maxval=1.)
        loss = self.nll(params, t, x, y).mean(-1)
        # jax.debug.print(f'loss shape: {loss.shape}')

        cp_w = pytree_to_matrix(params['params'], self.k)
        curve, d_bezier = bezier_curve(self.k+1, cp_w)
        tt = jnp.linspace(0., 1., 10_000)
        bezier_grad = jax.vmap(d_bezier)(tt)
        curve_length = jnp.trapezoid(
            jnp.linalg.norm(bezier_grad, axis=-1), tt)
        grads = jax.vmap(d_bezier)(t)
        weights = jnp.linalg.norm(grads, axis=-1) / curve_length

        # t_dist = BezierTspaceUnifrom(d_bezier)
        # t = t_dist.sample(key, (n_samples,))

        weighted_loss = jnp.einsum('n,n->', weights, loss)/n_samples
        return weighted_loss

class CategorySubspace(SubspaceModel):
    def __init__(self, model, k, n_samples):
        super().__init__(model, k, n_samples, np.inf, False)

    @partial(jit, static_argnums=(0,))
    def nll(self, params, t, x, y):
        """
        Computes the negative log-likelihood loss of the model.

        Parameters:
        - params: The parameters of the model.
        - t: The time parameter.
        - x: The input data.
        - y: The target output.

        Returns:
        - The negative log-likelihood loss.

        """
        logits = self(params['params'], t,
                      x)  # shape (n_samples, n_data, output_dim)

        nll = jax.vmap(
            optax.softmax_cross_entropy_with_integer_labels, in_axes=(0, None))(logits, y)
        return nll


class UniformTCategory(CategorySubspace, UniformTSubspace):
    pass


#@partial(jit, static_argnums=(1))
def pytree_to_matrix(pytree, k):
    """
    Converts a pytree of the subspace model into a matrix.

    Args:
        pytree: The pytree to be converted.

    Returns:
        matrix: The matrix representation of the pytree with shape (k+1, n_params).
    """
    # first argument of reduce fn is the accumulator and second is the current value
    return jax.tree.reduce(lambda x, y: jnp.concatenate(
        [x, y.reshape(k+1, -1)], axis=-1), pytree)


# @partial(jit, static_argnums=(1))
def pytree_to_vec(pytree):
    """
    Converts a pytree of the subspace model into a matrix.

    Args:
        pytree: The pytree to be converted.

    Returns:
        matrix: The matrix representation of the pytree with shape (k+1, n_params).
    """
    # first argument of reduce fn is the accumulator and second is the current value
    return jax.tree.reduce(lambda x, y: jnp.concatenate(
        [x, y.reshape(-1)], axis=-1), pytree)

@jit
def vec_to_single_pytree(vec, subspace_params):
    """
    Converts a vector representation of parameters to a pytree structure with no loading dimension.
    opposite of pytree_to_vec function.

    Args:
        vec: A 2-D numpy array or JAX array representing the flattened parameters; shape (k, D).
        subspace_params: A pytree of the subspace parameters. Used to inherit the structure and leaf shapes.

    Returns:
        A pytree structure with the same structure as `subspace_params` and with
        the parameters replaced by the values from `vec`. (Converts the whole parameterset)

    Example:
        vec = np.array([[1, 2, 3, 4]])
        subspace_params = {'a': np.array([[0, 0],]), 'b': np.array([[0, 0, 0],])}
        result = matrix_to_pytree(vec, subspace_params)
        # Output: {'a': np.array([[1, 2]]), 'b': np.array([[3, 4, 0]])}
    """
    leafs_params, structure = jax.tree.flatten(subspace_params)
    # get leaf shapes without stacking dimension
    leaf_vec_shapes = [p.shape for p in leafs_params]

    flatten_leafs = []
    index = 0
    for s in leaf_vec_shapes:
        upper_index = index + np.prod(s)
        flatten_leafs.append(jnp.reshape(vec[index: upper_index], s))
        index = upper_index
    return jax.tree.unflatten(structure, flatten_leafs)


@jit
def vec_to_pytree(vec, subspace_params):
    """
    Converts a vector representation of parameters to a pytree structure.
    With leading dimension.

    Args:
        vec: A 1-D numpy array or JAX array representing the flattened parameters.
        subspace_params: A pytree of the subspace parameters. Used to inherit the structure and leaf shapes.

    Returns:
        A pytree structure with the same structure as `subspace_params` (Without leading dimension), but with
        the parameters replaced by the values from `vec`.

    Example:
        vec = np.array([1, 2, 3, 4])
        subspace_params = {'a': np.array([[0, 0],]), 'b': np.array([[0, 0, 0],])}
        result = vec_to_pytree(vec, subspace_params)
        # Output: {'a': np.array([1, 2]), 'b': np.array([3, 4, 0])}
    """
    leafs_params, structure = jax.tree.flatten(subspace_params)
    # get leaf shapes without stacking dimension
    leaf_vec_shapes = [p.shape[1:] for p in leafs_params]

    flatten_leafs = []
    index = 0
    for s in leaf_vec_shapes:
        upper_index = index + np.prod(s)
        flatten_leafs.append(jnp.reshape(vec[index: upper_index], s))
        index = upper_index
    return jax.tree.unflatten(structure, flatten_leafs)

@jit
def matrix_to_pytree(vec, subspace_params):
    """
    Converts a vector representation of parameters to a pytree structure.

    Args:
        vec: A 2-D numpy array or JAX array representing the flattened parameters; shape (k, D).
        subspace_params: A pytree of the subspace parameters. Used to inherit the structure and leaf shapes.

    Returns:
        A pytree structure with the same structure as `subspace_params` and with
        the parameters replaced by the values from `vec`. (Converts the whole parameterset)

    Example:
        vec = np.array([[1, 2, 3, 4]])
        subspace_params = {'a': np.array([[0, 0],]), 'b': np.array([[0, 0, 0],])}
        result = matrix_to_pytree(vec, subspace_params)
        # Output: {'a': np.array([[1, 2]]), 'b': np.array([[3, 4, 0]])}
    """
    leafs_params, structure = jax.tree.flatten(subspace_params)
    # get leaf shapes without stacking dimension
    leaf_vec_shapes = [p.shape for p in leafs_params]

    flatten_leafs = []
    index = 0
    for s in leaf_vec_shapes:
        upper_index = index + np.prod(s[1:])
        flatten_leafs.append(jnp.reshape(vec[:, index: upper_index], s))
        index = upper_index
    return jax.tree.unflatten(structure, flatten_leafs)
