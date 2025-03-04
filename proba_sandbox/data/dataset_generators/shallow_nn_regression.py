"""Generate a dataset for regression with a shallow neural network."""
import numpy as np
import typer


def generate_shallow_net_samples(
    n_vars,
    mask_vars,
    n_samples,
    n_hidden_nodes,
    activation,
    identifier,
    std,
    marginals,
    error_std,
    seed=0,
    mask_std=0,
    path='data/generated_data/',
):
    """
    Generate a dataset for regression with a shallow neural network.

    run with `python data/dataset_generators/shallow_nn_regression.py`

    Args:
        n_vars (int): The number of independent variables.
        mask_vars (int): The number of independent variables to mask out.
        n_samples (int): The number of samples to generate.
        n_hidden_nodes (int): The number of nodes in the hidden layer.
        activation (str): The activation function.
        identifier (str): A unique identifier for the generated dataset.
        std (float): The standard deviation of the independent variables.
        marginals (str): A comma-separated string of the marginal distributions
        to use for each independent variable.
        error_std (float, optional): The standard deviation of the error term.
        Defaults to 1.0.
        seed (int, optional): The random seed. Defaults to 0.
        mask_std (int, optional): The standard deviation of the masked variables.
        Defaults to 0.
        path (str, optional): The path to save the generated dataset. Defaults to
        'data/generated_data/'.

    Returns:
        None
    """
    if identifier is None:
        # concat the input args into a string
        identifier = '_'.join(
            [
                str(n_vars),
                str(mask_vars),
                str(n_samples),
                str(n_hidden_nodes),
                activation,
                str(std),
                marginals,
                str(error_std),
            ]
        )
    marginals = marginals.split(',')
    if len(marginals) < n_vars:
        marginals = marginals * n_vars
    np.random.seed(seed)
    # Generate the independent variable matrix X and error term vector epsilon
    X = np.zeros((n_samples, n_vars))
    epsilon = np.random.normal(np.zeros(n_samples), error_std)

    # Apply the specified marginal distributions to each column of X
    for i in range(n_vars):
        if marginals[i % len(marginals)] == 'normal':
            X[:, i] = np.random.normal(scale=std, size=n_samples)
        elif marginals[i % len(marginals)] == 'student_t':
            X[:, i] = np.random.standard_t(2, size=n_samples)
        elif marginals[i % len(marginals)] == 'exponential':
            X[:, i] = np.random.exponential(scale=std, size=n_samples)
        elif marginals[i % len(marginals)] == 'gamma':
            X[:, i] = np.random.gamma(shape=2, scale=std / 2, size=n_samples)

    # Generate the dependent variable vector y
    y = np.zeros((n_samples, 1))

    w1 = np.random.normal(0, 1, (n_vars, n_hidden_nodes))
    if mask_vars > 0:
        # set the rows of w1 corresponding to the masked variables close to 0
        if mask_std == 0:
            w1[:mask_vars, :] = 0
        else:
            w1[:mask_vars, :] = np.random.normal(
                0, mask_std, (mask_vars, n_hidden_nodes)
            )

    # save the weights
    np.savetxt(
        f'{path}shallow_nn_weights1_{identifier}.csv',
        w1,
        delimiter=',',
    )

    b1 = np.random.normal(0, 1, (n_hidden_nodes,))
    w2 = np.random.normal(0, 1, (n_hidden_nodes, 1))
    b2 = np.random.normal(0, 1, (1,))

    # Apply the activation function
    if activation == 'tanh':
        h = np.tanh(np.dot(X, w1) + b1)
    elif activation == 'relu':
        h = np.dot(X, w1) + b1
        h = h * (h > 0)
    elif activation == 'sigmoid':
        h = 1 / (1 + np.exp(-(np.dot(X, w1) + b1)))
    elif activation == 'linear':
        h = np.dot(X, w1) + b1

    # Generate the dependent variable vector y
    y = np.dot(h, w2) + b2 + epsilon.reshape(-1, 1)

    res = np.hstack((X, y))
    print(f'Data Shape: {res.shape}')
    # Save the resulting numpy array as ../shallow_nn_regression_<identifier>.csv
    if path is not None:
        np.savetxt(
            f'{path}shallow_nn_regression_{identifier}.csv',
            np.hstack((X, y)),
            delimiter=',',
        )
        print(f'Saved to {path}shallow_nn_regression_{identifier}.csv')
    else:
        return np.hstack((X, y))


def main_shallow_nn(
    n_vars: int = typer.Option(3, prompt=True, help='Number of independent variables'),
    mask_vars: int = typer.Option(
        0, prompt=True, help='Number of independent variables to mask out'
    ),
    n_samples: int = typer.Option(
        400, prompt=True, help='Number of samples to generate'
    ),
    n_hidden_nodes: int = typer.Option(
        5, prompt=True, help='Number of nodes in the hidden layer'
    ),
    activation: str = typer.Option('tanh', prompt=True, help='Activation function'),
    identifier: str = typer.Option(
        'test', prompt=True, help='Identifier for the generated dataset'
    ),
    std: float = typer.Option(
        '1.0', prompt=True, help='Standard deviation of the error term'
    ),
    marginals: str = typer.Option(
        'normal',
        prompt=True,
        help='Marginal distributions for each independent variable',
    ),
    error_std: float = typer.Option(
        1.0, prompt=True, help='Standard deviation of the error term'
    ),
):
    """Generate a dataset for linear regression."""
    generate_shallow_net_samples(
        n_vars,
        mask_vars,
        n_samples,
        n_hidden_nodes,
        activation,
        identifier,
        std,
        marginals,
        error_std,
    )


if __name__ == '__main__':
    typer.run(main_shallow_nn)
