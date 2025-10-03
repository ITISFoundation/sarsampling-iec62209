import numpy as np
from scipy.stats import kstest, qmc
from typing import Callable
from numpy.typing import ArrayLike
from sar_sampling.lhs import lhs, LhSampler

def _test_stratification(lhsarray: np.ndarray) -> None:
    """
    Test that each marginal is perfectly stratified.
    
    Each of the k intervals along each axis contains exactly one sample.
    This is a fundamental property of Latin Hypercube sampling.
    
    Parameters:
        lhsarray (np.ndarray): Latin Hypercube sample array of shape (k, n)
        
    Raises:
        AssertionError: If stratification property is violated
    """
    k, n = lhsarray.shape
    for j in range(n):
        # Assign each value to a bin
        bins = (lhsarray[:, j] * k).astype(int)
        # Check all bins are in range
        assert np.all((bins >= 0) & (bins < k)), f"Bins out of range in dim {j}"
        # Check each bin is hit exactly once
        unique, counts = np.unique(bins, return_counts=True)
        assert len(unique) == k, f"not all bins hit in dim {j}"
        assert np.all(counts == 1), f"some bins have multiple samples in dim {j}"

def _test_range(lhsarray: np.ndarray) -> None:
    """
    Test that all samples are in the [0, 1) interval.
    
    Parameters:
        lhsarray (np.ndarray): Latin Hypercube sample array
        
    Raises:
        AssertionError: If any sample is outside the [0, 1) interval
    """
    assert np.all(lhsarray >= 0), "some samples are less than 0"
    assert np.all(lhsarray < 1), "some samples are >= 1"

def _test_uniformity(lhsarray: np.ndarray, alpha: float = 0.1) -> None:
    """
    Test that each marginal is consistent with a uniform distribution using the Kolmogorov-Smirnov test.
    
    Parameters:
        lhsarray (np.ndarray): Latin Hypercube sample array
        alpha (float): Significance level for the KS test (default: 0.1)
        
    Raises:
        AssertionError: If any marginal fails the KS test at the given significance level
    """
    k, n = lhsarray.shape
    for j in range(n):
        stat, pval = kstest(lhsarray[:, j], 'uniform')
        assert pval > alpha, f"KS test failed for dim {j} (p={pval:.3g})"

def _test_discrepancy(lhsarray: np.ndarray, threshold: float = 0.1) -> None:
    """
    Test that the discrepancy is below a reasonable threshold (space-filling property).
    
    Discrepancy measures how well the sample fills the space. Lower values indicate
    better space-filling properties.
    
    Parameters:
        lhsarray (np.ndarray): Latin Hypercube sample array
        threshold (float): Maximum acceptable discrepancy value (default: 0.1)
        
    Raises:
        AssertionError: If discrepancy exceeds the threshold
    """
    disc = qmc.discrepancy(lhsarray)
    assert disc < threshold, f"discrepancy too high: {disc}"

def _test_seed(lhs_func: Callable[..., np.ndarray], n: int = 3, k: int = 10, seed: int = 43) -> None:
    """
    Test reproducibility and randomness of LHS generation.
    
    Verifies that:
    1. Same seed produces identical results
    2. Different seeds produce different results (with high probability)
    
    Parameters:
        lhs_func (callable): LHS generation function to test
        n (int): Number of samples to generate (default: 3)
        k (int): Number of dimensions (default: 10)
        seed (int): Base seed for testing (default: 43)
        
    Raises:
        AssertionError: If reproducibility or randomness properties are violated
    """
    # Generate two samples with the same seed
    sample1 = lhs_func(n, k, seed=seed)
    sample2 = lhs_func(n, k, seed=seed)
    # They must be identical
    assert np.allclose(sample1, sample2), "LHS output is not reproducible with the same seed"
    # Generate a sample with a different seed
    sample3 = lhs_func(n, k, seed=seed+1)
    # With high probability, this should be different
    assert not np.allclose(sample1, sample3), "LHS output is identical for different seeds"

def _test_quantind() -> None:
    """
    Test the quantind_1d and quantind_2d functions for inverse CDF mapping.
    
    Tests both 1D and 2D inverse cumulative distribution function mapping
    to ensure proper bin assignment based on weights and quantiles.
    """
    from sar_sampling.lhs import quantind_1d, quantind_2d
    
    # Test quantind_1d with uniform weights
    quantiles = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    indices = quantind_1d(quantiles, weights)
    expected_indices = np.array([0, 1, 2, 3, 4])
    assert np.array_equal(indices, expected_indices), f"quantind_1d uniform weights failed: {indices}"
    
    # Test quantind_1d with non-uniform weights
    weights_nonuniform = np.array([2.0, 1.0, 1.0, 1.0, 1.0])
    indices_nonuniform = quantind_1d(quantiles, weights_nonuniform)
    # With double weight on first bin, more quantiles should map to index 0
    assert indices_nonuniform[0] == 0, f"quantind_1d non-uniform weights failed: {indices_nonuniform}"
    
    # Test quantind_1d with edge cases
    edge_quantiles = np.array([0.0, 0.5, 1.0])
    indices_edge = quantind_1d(edge_quantiles, weights)
    assert indices_edge[0] == 0, f"quantind_1d edge case 0.0 failed: {indices_edge}"
    assert indices_edge[2] == 4, f"quantind_1d edge case 1.0 failed: {indices_edge}"
    
    # Test quantind_2d with uniform weights
    quantiles1 = np.array([0.2, 0.6, 0.8])
    quantiles2 = np.array([0.3, 0.5, 0.9])
    weights_2d = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    indices1, indices2 = quantind_2d(quantiles1, quantiles2, weights_2d)
    assert len(indices1) == len(quantiles1), f"quantind_2d output length mismatch: {len(indices1)}"
    assert len(indices2) == len(quantiles2), f"quantind_2d output length mismatch: {len(indices2)}"
    assert np.all((indices1 >= 0) & (indices1 < 3)), f"quantind_2d indices1 out of range: {indices1}"
    assert np.all((indices2 >= 0) & (indices2 < 3)), f"quantind_2d indices2 out of range: {indices2}"
    
    # Test quantind_2d with non-uniform weights
    weights_2d_nonuniform = np.array([[2.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    indices1_nonuniform, indices2_nonuniform = quantind_2d(quantiles1, quantiles2, weights_2d_nonuniform)
    # Should still produce valid indices
    assert np.all((indices1_nonuniform >= 0) & (indices1_nonuniform < 3)), f"quantind_2d non-uniform indices1 out of range: {indices1_nonuniform}"
    assert np.all((indices2_nonuniform >= 0) & (indices2_nonuniform < 3)), f"quantind_2d non-uniform indices2 out of range: {indices2_nonuniform}"
    
    # Test quantind_2d with zero-weight rows
    weights_2d_zero = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    indices1_zero, indices2_zero = quantind_2d(quantiles1, quantiles2, weights_2d_zero)
    # Should handle zero weights gracefully
    assert np.all((indices1_zero >= 0) & (indices1_zero < 3)), f"quantind_2d zero weights indices1 out of range: {indices1_zero}"
    assert np.all((indices2_zero >= 0) & (indices2_zero < 3)), f"quantind_2d zero weights indices2 out of range: {indices2_zero}"

def _test_lhsampler(k: int = 10) -> None:
    """
    Test consistency between LhSampler and lhs function.
    
    Compares the output of LhSampler with the basic lhs function to ensure
    they produce equivalent results for various configurations including
    uniform and non-uniform histograms.
    
    Parameters:
        k (int): Number of samples/dimensions for testing (default: 10)
        
    Raises:
        AssertionError: If LhSampler output differs from lhs function output
    """
    rng = np.random.RandomState(43)
    seeds = rng.randint(0, 1000000, size=5)
    for seed in seeds:
        # Generate initial random LHS points and compute bin midpoints for each sample
        lhsarray0 = lhs(2, k, seed=seed)
        grid_indices = (lhsarray0 * k).astype(int)
        lhsarray0 = (grid_indices + 0.5) / k
        # build a domain with midpoints
        edges = np.linspace(0, 1, k + 1)
        midpoints = (edges[:-1] + edges[1:]) / 2
        domain = {'a': list(midpoints), 'b': list(midpoints)}
        # compare lhs with lhsampler
        lhsampler1 = LhSampler(domain)
        lhsarray1 = lhsampler1(k, seed=seed).sort_index(axis=1).to_numpy()
        assert np.allclose(lhsarray0, lhsarray1), "LHS sampler is not consistent with lhs function"
        hist1d = np.ones(k)
        lhsampler2 = LhSampler(domain, hist={'a': hist1d})
        lhsarray2 = lhsampler2(k, seed=seed).sort_index(axis=1).to_numpy()
        assert np.allclose(lhsarray0, lhsarray2), "LHS sampler with 1D histogram is not consistent with lhs function"
        hist2d = np.ones((k, k))
        lhsampler3 = LhSampler(domain, hist={('a', 'b'): hist2d})
        lhsarray3 = lhsampler3(k, seed=seed).sort_index(axis=1).to_numpy()
        assert np.allclose(lhsarray0, lhsarray3), "LHS sampler with 2D histogram is not consistent with lhs function"
        # compare with non-uniform histogram
        hist1d[0] = 2
        hist2d[0, :] = 2
        hist2d[:, 0] = 2
        hist2d[0, 0] = 4
        lhsampler4 = LhSampler(domain, hist={('a', 'b'): hist2d})
        lhsarray4 = lhsampler4(k, seed=seed).sort_index(axis=1).to_numpy()
        lhsampler5 = LhSampler(domain, hist={'a': hist1d, 'b': hist1d})
        lhsarray5 = lhsampler5(k, seed=seed).sort_index(axis=1).to_numpy()
        assert np.allclose(lhsarray4, lhsarray5), "LHS sampler with non-uniform histograms are not consistent"

def test_range():
    for method in [None, 'c', 'm', 'cm']:
        lhsarr = lhs(5, 50, method=method)
        _test_range(lhsarr)

def test_stratification():
    for method in [None, 'c', 'm', 'cm']:
        lhsarr = lhs(5, 50, method=method)
        _test_stratification(lhsarr)

def test_uniformity():
    for method in [None, 'c', 'm', 'cm']:
        lhsarr = lhs(5, 50, method=method)
        _test_uniformity(lhsarr)

def test_discrepancy():
    for method in [None, 'c', 'm', 'cm']:
        lhsarr = lhs(5, 50, method=method)
        _test_discrepancy(lhsarr)

def test_seed():
    _test_seed(lhs)

def test_quantind():
    _test_quantind()

def test_lhsampler():
    _test_lhsampler()

def _test_all(lhs_generator: Callable[..., np.ndarray], n: int = 5, k: int = 50) -> None:
    """
    Run comprehensive LHS validation tests.
    
    Tests all sampling methods and validates fundamental LHS properties
    including stratification, range, uniformity, discrepancy, and reproducibility.
    
    Parameters:
        lhs_generator (callable): LHS generation function to test
        n (int): Number of samples for quality tests (default: 5)
        k (int): Number of dimensions for quality tests (default: 50)
        
    Raises:
        AssertionError: If any LHS property is violated
    """
    # lhs quality
    for method in [None, 'c', 'm', 'cm']:
        lhsarr = lhs_generator(n, k, method=method)
        _test_range(lhsarr)
        _test_stratification(lhsarr)
        _test_uniformity(lhsarr)
        _test_discrepancy(lhsarr)
    # lhs seed
    _test_seed(lhs_generator)
    # quantind functions
    _test_quantind()
    # lhsampler
    _test_lhsampler()

def main() -> None:
    """
    Main function to run LHS validation tests.
    
    Executes all LHS quality and consistency tests using the lhs function
    as the reference implementation.
    """
    print("Running LHS validation tests...")
    _test_all(lhs, n=5, k=50)
    print("All LHS validation tests passed.")

if __name__ == "__main__":
    main()
