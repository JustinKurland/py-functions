import numpy as np
from scipy.stats import wasserstein_distance

def w_permutation_test(reference, production, n_permutations=1000):
    """
    Calculate the Wasserstein Distance between two samples and leverage a 
    permutation approach and Monte Carlo simulation to calculate the p-value.

    Parameters
    ----------
    reference : np.ndarray
        A 1-D array containing the reference sample.
    production : np.ndarray
        A 1-D array containing the production sample.
    n_permutations : int, optional
        The number of permutations to use for the permutation test (default=1000).

    Returns
    -------
    None
        The function prints the Wasserstein Distance and the 
        permutation test p-value.

    References
    ----------
    [1] "Wasserstein distance" in SciPy documentation
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html
    [2] Vallender, S. S. (1974). Calculation of the Wasserstein distance between 
        probability distributions on the line. Theory of Probability & Its Applications, 
        18(4), 784-786.
    [3] Oden, N. L. (1991). Allocation of effort in Monte Carlo simulation for 
        power of permutation tests. Journal of the American Statistical Association, 
        86(416), 1074-1076.
    """
    # Calculate the Wasserstein Distance between the reference and production samples
    wasserstein_dist = wasserstein_distance(reference, production)

    n_ref = len(reference)
    n_prod = len(production)
    n = min(n_ref, n_prod)

    permutation_stats = np.zeros(n_permutations)

    for i in range(n_permutations):
        # Shuffle the original vectors
        permuted_reference = np.random.permutation(reference)
        permuted_production = np.random.permutation(production)

        # Generate a uniform random number for each index
        rand_nums = np.random.uniform(size=n)

        # Shuffle values between the two samples
        for j in range(n):
            if rand_nums[j] >= 0.5:
                if j < n_ref and j < n_prod:
                    temp = permuted_reference[j]
                    permuted_reference[j] = permuted_production[j]
                    permuted_production[j] = temp

        # Calculate the Wasserstein Distance for this permutation
        permutation_stats[i] = wasserstein_distance(permuted_reference, permuted_production)

    # Calculate the p-value
    p_value = np.sum(permutation_stats >= wasserstein_dist) / n_permutations

    print(f'Wasserstein distance: {wasserstein_dist:.3f}, p-value: {p_value:.3f}')