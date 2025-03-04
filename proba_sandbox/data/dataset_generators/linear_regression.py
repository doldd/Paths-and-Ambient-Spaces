"""Generate a dataset for linear regression."""
import numpy as np
import typer


def generate_dataset(
    n_vars,
    coeffs,
    identifier,
    std,
    n_samples,
    marginals,
    error_std,
    seed=0,
    path='data/generated_data/',
):
    """
    Generate a dataset for linear regression.

    run with `python data/dataset_generators/linear_regression.py`

    Args:
        n_vars (int): The number of independent variables.
        coeffs (str): A comma-separated string of the true coefficients for
        each independent variable.
        identifier (str): A unique identifier for the generated dataset.
        std (float): The standard deviation of the independent variables.
        n_samples (int): The number of samples to generate.
        marginals (str): A comma-separated string of the marginal distributions
        to use for each independent variable.
        error_std (float, optional): The standard deviation of the error term.
        Defaults to 1.0.

    Returns:
        None
    """
    np.random.seed(seed)
    if identifier is None:
        # concat the input args into a string
        identifier = '_'.join(
            [str(n_vars), coeffs, str(std), str(n_samples), marginals, error_std]
        )
    coeffs = coeffs.split(',')
    coeffs = [float(x) for x in coeffs]
    marginals = marginals.split(',')
    if len(marginals) < n_vars:
        marginals = marginals * n_vars

    # Define the coefficient values
    coeffs = np.array(coeffs)

    # Generate the independent variable matrix X and error term vector epsilon
    X = np.ones((n_samples, n_vars))
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
    y = np.dot(X, coeffs) + epsilon
    y = y.reshape(-1, 1)

    # Save the resulting numpy array as ../linear_regression_<identifier>.csv
    if path is not None:
        np.savetxt(
            f'{path}linear_regression_{identifier}.csv',
            np.hstack((X, y)),
            delimiter=',',
        )
    else:
        return np.hstack((X, y))


def main(
    n_vars: int = typer.Option(3, prompt=True, help='Number of independent variables'),
    coeffs: str = typer.Option('3.0,2.0,0.1', prompt=True, help='Coefficient values'),
    identifier: str = typer.Option(
        'test', prompt=True, help='Identifier for the generated dataset'
    ),
    std: float = typer.Option(
        '1.0', prompt=True, help='Standard deviation of the error term'
    ),
    n_samples: int = typer.Option(
        400, prompt=True, help='Number of samples to generate'
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
    generate_dataset(n_vars, coeffs, identifier, std, n_samples, marginals, error_std)


if __name__ == '__main__':
    typer.run(main)
