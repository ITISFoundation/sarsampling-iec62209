import numpy as np
import scipy as sp
import pandas as pd
from typing import Callable

__all__ = ['lhs', 'LhSampler']

# ============================================================================
# LhSampler class:
class LhSampler:
    """
    A Latin Hypercube Sampler for generating structured samples across multiple dimensions.
    
    This class implements Latin Hypercube sampling with support for weighted distributions,
    both 1D and 2D histograms, and various sampling methods. It can handle continuous
    and discrete domains with custom probability distributions.
    
    The sampler works by:
    1. Processing domain definitions and histogram weights
    2. Generating Latin Hypercube samples in the unit hypercube [0,1]^d
    3. Mapping samples to actual domain values using inverse CDF transformations
    4. Supporting both independent and correlated sampling across dimensions
    
    Attributes:
        domains (dict): Processed domain definitions for each dimension
        histograms (list): List of (key, histogram) tuples for weighted sampling
        keys (list): Sorted list of dimension names
    
    Parameters:
        dom (dict): Domain definitions. Keys are dimension names, values define the domain:
            - None: Continuous uniform on [0, 1]
            - [val]: Continuous uniform on [0, val]
            - [min, max]: Continuous uniform on [min, max]
            - [val1, val2, ...]: Discrete values with optional weights
        hist (dict, optional): Histogram weights for weighted sampling:
            - str key: 1D histogram for single dimension
            - (str, str) key: 2D histogram for correlated dimensions
            - Weights must match domain lengths/shapes
    
    Sampling Methods:
        - 'classic': Standard Latin Hypercube sampling
        - 'center'/'c': Centered Latin Hypercube sampling
        - 'maximin'/'m': Maximin distance optimization
        - 'centermaximin'/'cm': Centered with maximin optimization
    
    Example:
        >>> # Simple uniform sampling
        >>> sampler = LhSampler({'x': [0, 10], 'y': [0, 20]})
        >>> samples = sampler(100, method='maximin')
        
        >>> # Weighted sampling with histogram
        >>> hist = {'x': [1, 2, 1], 'y': [1, 1, 2]}
        >>> sampler = LhSampler({'x': [1, 2, 3], 'y': [4, 5, 6]}, hist=hist)
        >>> samples = sampler(50)
    """
    def __init__(
        self, 
        dom: dict[str, None | float | list[float]],
        hist: None | dict[str | tuple[str, str], np.ndarray | list[float]] = None,
    ) -> None:
        """
        dom = {'name': values, ...} dictionary of domain names and values.
            If values is None, the domain is continuous on [0, 1].
            If values is a list of length 1, the domain is sampled uniformly on [0, values[0]].
            If values is a list of length 2, the domain is sampled uniformly on [values[0], values[1]].
            If values is a list of length > 2, the domain is sampled according to the weights.
            Cases where length 1 and 2 become discrete when the corresponding 1d histogram is provided.

        hist = {'key': weights, ...} where key is a string (for 1d histogram) or tuple of strings (for 2d histogram) 
            and weights is a 1D or 2D array of weights. If None, the domain is sampled uniformly.
            If an entry is not provided for a given domain name, the domain is sampled uniformly.
            'key' names can be used at most once in the hist dictionary.
        """
        if isinstance(dom, dict):
            self.keys = sorted(list(dom.keys()))
        else:
            raise ValueError("domain must be a dictionary")
            # track encountered strings to check for duplicates
        
        # variables names and domains: {key : dom}
        domains = {}
        # histograms: [(key, hist) or ((key1, key2), hist)] 
        histograms = []

        # process histograms first
        if hist is not None:
            for key in hist:
                # Check single string keys
                if isinstance(key, str):
                    # Consistency check: domain cannot be None or float
                    if not isinstance(dom[key], list):
                        raise RuntimeError(f"Domain for {key} must be a list")
                    if key in domains:
                        raise ValueError(f"Duplicate string '{key}' found in hist")
                    if key not in self.keys:
                        raise ValueError(f"String '{key}' in hist not found in domain")
                    hist_array = np.asarray(hist[key])
                    if len(dom[key]) != hist_array.shape[0]: # type: ignore
                        raise ValueError(f"Length mismatch: domain '{key}' has length {len(dom[key])} but hist has length {len(hist[key])}") # type: ignore
                    domains[key] = dom[key].copy() # type: ignore
                    histograms.append((key, hist_array.copy()))
                # Check tuple string keys
                elif isinstance(key, tuple):
                    if len(key) != 2:
                        raise ValueError(f"Tuple key {key} must contain exactly 2 strings")
                    if not all(isinstance(k, str) for k in key):
                        raise ValueError(f"All elements in tuple {key} must be strings")
                    for k in key:
                        if k in domains:
                            raise ValueError(f"Duplicate string '{k}' found in hist")
                        if k not in self.keys:
                            raise ValueError(f"String '{k}' in hist not found in domain")
                        domains[k] = dom[k].copy() # type: ignore
                    hist_array = np.asarray(hist[key])
                    if hist_array.shape != (len(dom[key[0]]), len(dom[key[1]])):  # type: ignore
                        raise ValueError(f"Shape mismatch: histogram for {key} has shape {hist_array.shape} but domains have lengths {len(dom[key[0]])} and {len(dom[key[1]])}")  # type: ignore
                    histograms.append((key, hist_array.copy()))
                else:
                    raise ValueError(f"Invalid key type {type(key)} in hist")

        # Build set of remaining domain keys
        remaining_keys = set(k for k in self.keys if k not in domains)
        for key in remaining_keys:
            if dom[key] is None:
                domains[key] = None
                histograms.append((key, None))
            else:
                domains[key] = dom[key].copy() # type: ignore
                l = len(dom[key])  # type: ignore
                if l > 2:
                    histograms.append((key, np.ones(l)))
                else:
                    histograms.append((key, None))

        self.domains = domains
        self.histograms = histograms

    def __call__(
        self, 
        n_samples: int, 
        method: None | str = None, 
        seed: None | int = None,
    ) -> pd.DataFrame:
        """
        Generate Latin Hypercube samples.
        
        Parameters:
            n_samples (int): Number of samples to generate
            method (str, optional): Sampling method ('classic', 'center', 'maximin', etc.)
            seed (int, optional): Random seed for reproducibility
            
        Returns:
            pd.DataFrame: DataFrame with sampled values for each dimension
        """
        lhsample = lhs(len(self.domains), n_samples, method=method, seed=seed)
        data = {}
        for key, hist in self.histograms:
            if hist is None:
                i = self.keys.index(key)
                dom = self.domains[key]
                if dom is None:
                    data[key] = list(lhsample[:, i])
                elif len(dom) >= 2:
                    data[key] = list(dom[0] + lhsample[:, i] * (dom[1] - dom[0]))
                else:
                    data[key] = list(lhsample[:, i] * dom[0])
            elif isinstance(key, str):
                i = self.keys.index(key)
                indices = quantind_1d(lhsample[:, i], hist)
                values = [self.domains[key][idx] for idx in indices]
                data[key] = values
            elif isinstance(key, tuple):
                key1, key2 = key[0], key[1]
                i1 = self.keys.index(key1)
                i2 = self.keys.index(key2)
                indices1, indices2 = quantind_2d(lhsample[:, i1], lhsample[:, i2], hist)
                values1 = [self.domains[key1][idx] for idx in indices1]
                values2 = [self.domains[key2][idx] for idx in indices2]
                data[key1] = values1
                data[key2] = values2
            else:
                raise ValueError(f"Invalid key {key} in histograms")
        df = pd.DataFrame(data)
        return df


# ============================================================================
# utils:

def quantind_1d(quantiles: np.ndarray | list[float], weights: np.ndarray | list[float]) -> np.ndarray:
    """
    For each value in quantiles (assumed in [0,1]), 
    output the index in weights_1d array corresponding to inverse CDF mapping.
    quantiles:      array-like of values in [0,1]
    weights: 1D array (n_bins,) representing the (non-negative) grid/bin weights
    Returns:    array of indices with same shape as quant
    """
    weights_array = np.asarray(weights, dtype=float)
    weights_array /= weights_array.sum()
    n_bins = len(weights_array)
    # Compute CDF over bins
    cdf = np.cumsum(weights_array)
    cdf /= cdf[-1]
    # Input uniform values (n_samples,)
    u = np.asarray(quantiles, dtype=float)
    # Map each u to the bin where cdf >= u, using searchsorted
    bin_idx = np.searchsorted(cdf, u, side="right")
    bin_idx = np.clip(bin_idx, 0, n_bins-1)
    return bin_idx

def quantind_2d(quantiles1: np.ndarray | list[float], quantiles2: np.ndarray | list[float], weights: np.ndarray | list[float]) -> tuple[np.ndarray, np.ndarray]:
    """
    For each pair (u1, u2) in quantiles1, quantiles2 (all in [0,1]), 
    output (x_idx, y_idx) such that grid_weights[x_idx, y_idx] is the assigned bin 
    according to conditional inverse CDF mapping.
    quantiles1:   array-like of values in [0,1]
    quantiles2:   array-like of values in [0,1]  
    weights:      2D array (n_x, n_y)
    Returns:      tuple (bin_x, bin_y) of arrays of indices for each sample
    """
    wx = np.asarray(weights, dtype=float)
    wx /= wx.sum()
    n_x, n_y = wx.shape
    # Marginal CDF for x
    px = wx.sum(axis=1)
    cdf_x = np.cumsum(px)
    cdf_x /= cdf_x[-1]
    u1 = np.asarray(quantiles1, dtype=float)
    u2 = np.asarray(quantiles2, dtype=float)
    # For each u1, get its x row index via marginal CDF
    bin_x = np.searchsorted(cdf_x, u1, side="right")
    bin_x = np.clip(bin_x, 0, n_x-1)  # safety
    # For each row, get appropriate y index using conditional CDF in that row
    bin_y = np.empty_like(bin_x)
    for idx in range(len(bin_x)):
        w_row = wx[bin_x[idx], :]
        if w_row.sum() == 0:
            cdf_row = np.linspace(0, 1, n_y, endpoint=False)
        else:
            cdf_row = np.cumsum(w_row)
            cdf_row /= cdf_row[-1]
        bin_y[idx] = np.searchsorted(cdf_row, u2[idx], side="right")
        bin_y[idx] = np.clip(bin_y[idx], 0, n_y-1)
    return bin_x, bin_y


# ============================================================================
# lhs function:

def lhs(
    dim: int, 
    size: int, 
    method: None | str = None, 
    iter: None | int = None, 
    seed: None | int | np.random.RandomState = None, 
    quant_func: None | Callable[[np.ndarray], np.ndarray] | list[Callable[[np.ndarray], np.ndarray]] = None
) -> np.ndarray:
    """
    Generate a latin-hypercube design
    Parameters
    ----------
    dim : int
        The number of factors to generate samples for
    size : int
        The number of samples to generate for each factor
    Optional
    --------
    method : str
        Allowable values are "center" or "c", "maximin" or "m",
        "centermaximin" or "cm". If no value
        given, the design is simply randomized.
    iter : int
        The number of iterations in the maximin and correlations algorithms
        (Default: 5).
    seed : np.random.RandomState, int
         The seed and random draws
    quant_func : inverse cumulative distribution function or array of len 
         dim of such functions applied to each factor. 
    Returns
    -------
    H : 2d-array
        An k=size by n=dim 2d array that has been normalized so factor values
        are uniformly spaced between zero and one.
    """
    H: np.ndarray | None = None

    if seed is None:
        seed = np.random.RandomState()
    elif not isinstance(seed, np.random.RandomState):
        seed = np.random.RandomState(seed)

    if method is not None:
        if not method.lower() in ('center', 'c', 'maximin', 'm', 'centermaximin', 'cm'):
            raise ValueError('Invalid value for "method": {}'.format(method))
    else:
        method = 'classic'

    if iter is None:
        iter = 5

    ml = method.lower()
    if ml in ('classic'):
        H = _lhsclassic(dim, size, seed)
    elif ml in ('center', 'c'):
        H = _lhscentered(dim, size, seed)
    elif ml in ('maximin', 'm'):
        H = _lhsmaximin(dim, size, iter, 'maximin', seed)
    elif ml in ('centermaximin', 'cm'):
        H = _lhsmaximin(dim, size, iter, 'centermaximin', seed)

    if H is not None and quant_func is not None:
        if callable(quant_func):
            H = np.apply_along_axis(quant_func, 0, H)
        elif len(quant_func) == dim:
            HT = H.T
            for i, f in enumerate(quant_func):
                if callable(f):
                    HT[i] = f(HT[i])
            H = HT.T
    if H is not None:
        return H
    else:
        raise RuntimeError("Error while generating sample")

def _lhsclassic(n: int, k: int, randomstate: np.random.RandomState) -> np.ndarray:
    # generate the intervals
    cut = np.linspace(0, 1, k + 1)

    # fill points uniformly in each interval
    u = randomstate.rand(k, n)
    a = cut[:k]
    b = cut[1:k + 1]
    rdpoints = np.zeros_like(u)
    for j in range(n):
        rdpoints[:, j] = u[:, j]*(b-a) + a

    # make the random pairings
    H = np.zeros_like(rdpoints)
    for j in range(n):
        order = randomstate.permutation(range(k))
        H[:, j] = rdpoints[order, j]

    return H

def _lhscentered(n: int, k: int, randomstate: np.random.RandomState) -> np.ndarray:
    # generate the intervals
    cut = np.linspace(0, 1, k + 1)

    u = randomstate.rand(k, n)
    a = cut[:k]
    b = cut[1:k + 1]
    _center = (a + b)/2

    # make the random pairings
    H = np.zeros_like(u)
    for j in range(n):
        H[:, j] = randomstate.permutation(_center)

    return H

def _lhsmaximin(
    n: int, 
    k: int, 
    iter: int, 
    lhstype: str, 
    randomstate: np.random.RandomState
) -> np.ndarray | None:
    maxdist = 0

    H : np.ndarray | None = None

    # maximize the minimum distance between points
    for i in range(iter):
        if lhstype=='maximin':
            Hcandidate = _lhsclassic(n, k, randomstate)
        else:
            Hcandidate = _lhscentered(n, k, randomstate)

        d = sp.spatial.distance.pdist(Hcandidate, 'euclidean')
        if maxdist<np.min(d):
            maxdist = np.min(d)
            H = Hcandidate.copy()

    return H


