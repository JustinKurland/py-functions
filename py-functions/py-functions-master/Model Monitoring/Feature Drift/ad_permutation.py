import numpy as np
from scipy.stats import anderson_ksamp
import warnings
warnings.filterwarnings("ignore")

def ad_permutation_test(reference, production, n_permutations=1000):
    """
    Calculate the Anderson-Darling Test-statistic and p-value between two samples, 
    along with a permutation test p-value.

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
        The function prints the AD-statistic and the permutation test p-value.

    References
    ----------
    [1] "K-Sample Anderson-Darling Test" in SciPy documentation
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.anderson_ksamp.html
    [2] Scholz, F. W., & Stephens, M. A. (1987). K-sample Anderson–Darling tests. 
        Journal of the American Statistical Association, 82(399), 918-924.
    [3] Oden, N. L. (1991). Allocation of effort in Monte Carlo simulation for 
        power of permutation tests. Journal of the American Statistical Association, 
        86(416), 1074-1076. 
    """
    # Calculate the Anderson-Darling test statistic between the reference and production samples
    statistic, critical_values, _ = anderson_ksamp([reference, production])

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

        # Calculate the AD-test statistic for this permutation
        permutation_stats[i], critical_values, _ = anderson_ksamp([permuted_reference, permuted_production])

    # Calculate the p-value
    p_value = np.sum(permutation_stats >= statistic) / n_permutations

    print(f'AD-statistic: {statistic:.3f}, p-value: {p_value:.3f}')