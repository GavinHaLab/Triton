# Robert Patton, rpatton@fredhutch.org
# v2.0.1, 10/14/2025

import numpy as np
import pandas as pd

# Pre-compute a mapping array for faster lookup
mapping_array = np.zeros(91, dtype=np.uint8)  # ASCII 'Z' is 90
mapping_array[65:91] = 0  # A-Z default to 0
mapping_array[ord('A')] = 1
mapping_array[ord('C')] = 2
mapping_array[ord('G')] = 3
mapping_array[ord('T')] = 4


def str2int(v):
    """
    Stand-in type to read string/int values from the command line
        Parameters:
            v (string): input int/string
        Returns:
            int or None
    """
    return int(v) if v.isdigit() else None


def get_index(header, names, default, message):
    for name in names:
        try:
            return header.index(name)
        except ValueError:
            continue
    print(message.format(default, header[default] if default is not None and default < len(header) else 'N/A'))
    return default


# Standard BED field order, assumed for site lists without a header line
BED_FIELDS = ['chrom', 'chromStart', 'chromEnd', 'name', 'score', 'strand']


def bed_header(first_line):
    """
    Column names of a BED-like site list, given its first line.
        Returns:
            header (list of str): the first line's fields if it is a header; if it is already a site (integer start
                and end in columns 2 and 3), the standard BED field names for that many columns.
            has_header (bool): whether the first line is a header (and so must not be read as a site).
    """
    tokens = first_line.strip().split('\t')
    if len(tokens) >= 3 and tokens[1].isdigit() and tokens[2].isdigit():
        extra = [f'column{index + 1}' for index in range(len(BED_FIELDS), len(tokens))]
        return BED_FIELDS[:len(tokens)] + extra, False
    return tokens, True


def parse_strand(token):
    """
    Normalizes a strand field to '+', '-', or None (unstranded: '.', '*', 'NA' or empty).
    Raises ValueError for any other value, so that a mislabeled site is skipped rather than oriented by guess.
    """
    value = token.strip().lower()
    if value in ('+', '+1', '1', 'plus', 'forward', 'fwd', 'f'):
        return '+'
    if value in ('-', '-1', 'minus', 'reverse', 'rev', 'r'):
        return '-'
    if value in ('.', '*', '', 'na', 'nan', 'none'):
        return None
    raise ValueError(f'Unrecognized strand value: {token!r}')


def one_hot_encode(seq):
    """One-hot encode a nucleotide sequence"""
    # Convert sequence to ASCII codes and map
    indices = np.frombuffer(seq.encode(), dtype=np.uint8)
    return np.eye(5, dtype=np.int8)[mapping_array[indices]]


def get_gc_bias_dict(bias_path):
    """
    Modified from Griffin pipeline, credit: Anna-Lisa Doebley
    https://github.com/adoebley/Griffin
    https://doi.org/10.1038/s41467-022-35076-w
    Reads in a GC bias file and returns a dictionary of GC bias values
        Parameters:
            bias_path (string): path to GC bias output by Griffin or in Griffin format
        Returns:
            dict: GC bias in dictionary form (returns bias given [length][gc content]
    """
    bias = pd.read_csv(bias_path, sep='\t')
    # Apply filtering directly; clipped to reasonable bounds
    bias.loc[bias['smoothed_GC_bias'] < 0.10, 'smoothed_GC_bias'] = 0.10
    bias.loc[bias['smoothed_GC_bias'] > 10.0, 'smoothed_GC_bias'] = 10.0
    
    # Create dictionary directly
    bias_dict = {}
    for _, row in bias[['length', 'num_GC', 'smoothed_GC_bias']].iterrows():
        length, gc, bias_val = int(row['length']), int(row['num_GC']), row['smoothed_GC_bias']
        if length not in bias_dict:
            bias_dict[length] = {}
        bias_dict[length][gc] = bias_val
    
    return bias_dict


def frag_metrics(frag_prob_dist, bins=None, n_eff=None):
    """
    Returns various metrics of fragment lengths given a probability distribution.
    
    Parameters:
        frag_prob_dist (np.array): 1D array of fragment length probabilities (sums to 1.0), where index is fragment length - 1 (1-indexed)
        bins (list/tuple): Pre-computed bins as (bin_start_indices, num_bins). Used only for entropy/Gini-Simpson calculation.
        n_eff (float): effective number of fragments behind the distribution; if given, entropy and Gini-Simpson are
            corrected for finite-sample bias (see diversity_metrics).
    
    Returns:
        Tuple containing:
            - mean (float) - computed on 1-bp resolution
            - stdev (float) - computed on 1-bp resolution
            - skew (float) - computed on 1-bp resolution
            - kurtosis (float) - computed on 1-bp resolution
            - ratio (float) - computed on 1-bp resolution
            - entropy (float) - computed on binned data
            - gini_simpson (float) - computed on binned data
    """
    # Effective number of distinct fragment lengths, 1 / sum(p_i^2). It cannot exceed the effective number of
    # fragments, so requiring 5 also guarantees enough fragments for the bias corrections below.
    sum_p_squared = np.sum(frag_prob_dist ** 2)
    if sum_p_squared <= 0 or 1.0 / sum_p_squared < 5:
        return (np.nan,) * 7

    # Subnucleosomal ratio: computed on 1-bp resolution (threshold at 147 bp)
    frag_short = np.sum(frag_prob_dist[:146])
    frag_long = np.sum(frag_prob_dist[146:])
    ratio = np.nan if frag_long == 0 else frag_short / frag_long
    
    # Diversity metrics - always computed on binned data for smoothing
    entropy, gini_simpson = diversity_metrics(frag_prob_dist, bins=bins, n_eff=n_eff)

    # Statistical moments - computed on 1-bp resolution for accurate distribution representation
    indices = np.arange(1, len(frag_prob_dist) + 1)
    weights = frag_prob_dist
    
    # Mean (in bp units)
    mean = np.average(indices, weights=weights)
    
    # Variance and standard deviation (in bp units)
    delta = indices - mean
    variance = np.average(delta**2, weights=weights)
    stdev = np.sqrt(variance)
    
    # Higher moments (skewness and kurtosis)
    if stdev > 1e-10:
        skew = np.average(delta**3, weights=weights) / (stdev**3)
        kurtosis = np.average(delta**4, weights=weights) / (stdev**4) - 3  # Excess kurtosis
    else:
        skew, kurtosis = 0.0, 0.0

    return mean, stdev, skew, kurtosis, ratio, entropy, gini_simpson


def diversity_metrics(prob_dist, bins=None, n_eff=None):
    """
    Returns the max-normalized Shannon Entropy and Gini-Simpson index (1 - sum(p_i^2)) for binned probability distribution (5bp bins)
        Parameters:
            prob_dist (list/array): 1D array of fragment length probabilities (sums to 1.0), where index is fragment length -1 (1-indexed).
            bins (list/tuple): Pre-computed bins as (bin_start_indices, num_bins). If None, bins will be computed.
            n_eff (float): effective number of fragments behind prob_dist, (sum of weights)^2 / (sum of squared weights).
                If given, both indices are corrected for the downward bias of plug-in estimates from few fragments:
                Gini-Simpson by the exact factor n_eff / (n_eff - 1), entropy by the first-order Miller-Madow term
                (occupied bins - 1) / (2 n_eff ln 2) bits.
        Returns:
            tuple: (float: Max-normalized Shannon Entropy, float: Gini-Simpson index (max-normalized))
    """
    # If bins are not provided, compute them
    if bins is None:
        bin_size = 5
        prob_dist_len = len(prob_dist)
        num_bins = int(np.ceil(prob_dist_len / bin_size))
        
        # Early exit for empty or single bin
        if prob_dist_len == 0 or num_bins <= 1:
            return np.nan, np.nan
            
        # Faster binning using reshape and sum
        # Handle case where prob_dist is not evenly divisible by bin_size
        if prob_dist_len % bin_size != 0:
            # Pad the array with zeros to make it evenly divisible
            padding_size = bin_size - (prob_dist_len % bin_size)
            padded_prob = np.pad(prob_dist, (0, padding_size), 'constant')
            reshaped = padded_prob.reshape(-1, bin_size)
        else:
            reshaped = prob_dist.reshape(-1, bin_size)
        
        binned_probs = np.sum(reshaped, axis=1)
        
        # Remove empty bin from padding if it exists
        if prob_dist_len % bin_size != 0:
            binned_probs = binned_probs[:num_bins]
    else:
        # Use pre-computed bins
        bin_edges, num_bins = bins
        
        # Early exit for empty or single bin
        if len(prob_dist) == 0 or num_bins <= 1:
            return np.nan, np.nan
            
        # Compute binned probabilities using the provided bin edges
        binned_probs = np.zeros(num_bins)
        for i in range(num_bins):
            start_idx = bin_edges[i]
            end_idx = bin_edges[i+1] if i < num_bins-1 else len(prob_dist)
            binned_probs[i] = np.sum(prob_dist[start_idx:end_idx])
    
    # binned_probs should already sum to ~1.0 since prob_dist sums to 1.0
    # But use it directly without re-normalization to preserve the probability mass
    total = np.sum(binned_probs)
    if total <= 0:
        return np.nan, np.nan
        
    # Ensure normalization for numerical stability
    pdist = binned_probs / total if total > 0 else binned_probs
    
    # Gini-Simpson calculation (vectorized)
    gs_index = 1.0 - np.sum(pdist * pdist)
    if n_eff is not None:
        gs_index *= n_eff / (n_eff - 1.0)
    max_gs = 1.0 - 1.0/num_bins
    gs_index = gs_index / max_gs if max_gs > 0 else np.nan
    
    # Entropy calculation - vectorized with masked operations
    mask = pdist > 0
    entropy = 0.0
    if np.any(mask):
        # Use masked array for cleaner calculation
        p_nonzero = pdist[mask]
        log2_p = np.log2(p_nonzero)
        entropy_bits = -np.sum(p_nonzero * log2_p)
        if n_eff is not None:
            entropy_bits += (np.count_nonzero(mask) - 1) / (2.0 * n_eff * np.log(2))
        entropy = entropy_bits / np.log2(num_bins)
    
    return entropy, gs_index


def gini_simpson_profile(binned_profile, weight_sq=None):
    """
    Max-normalized Gini-Simpson index of the fragment length distribution at every position.
        Parameters:
            binned_profile (np.array): (fragment length bin x position) matrix of GC-corrected fragment depth, or of
                site-averaged probability distributions in composite mode.
            weight_sq (np.array): per-position sum of squared fragment weights. If given, the estimator is unbiased:
                (W^2 - sum_i W_i^2) / (W^2 - sum_j w_j^2), the chance that two distinct fragments covering the position
                (drawn in proportion to their weights w_j) fall in different length bins. This equals the plug-in
                index times n_eff / (n_eff - 1) with n_eff = W^2 / sum_j w_j^2, so its expectation does not depend
                on depth. If None, the plug-in index 1 - sum_i p_i^2 is returned (composite mode, where the
                distribution is pooled over many sites and the plug-in bias is negligible).
        Returns:
            np.array: index per position; NaN where there are no fragments or, with weight_sq, where n_eff < 5.
    """
    num_bins = binned_profile.shape[0]
    total_sq = binned_profile.sum(axis=0, dtype=np.float64) ** 2
    bin_sq = np.einsum('ij,ij->j', binned_profile, binned_profile, dtype=np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        if weight_sq is None:
            gs_index = 1.0 - bin_sq / total_sq
            valid = total_sq > 0
        else:
            gs_index = (total_sq - bin_sq) / (total_sq - weight_sq)
            valid = total_sq >= 5 * weight_sq
    gs_index /= 1.0 - 1.0 / num_bins
    gs_index[~valid | (total_sq <= 0)] = np.nan
    return gs_index


def local_peaks(ys):
    """
    Finds LOCAL (min or max relative to neighbors) peaks given a 1D array; only appropriate for very smooth data
    (E.G. an inverse fourier transform with high frequencies removed)
        Parameters:
            ys (np.array): signal height
        Returns:
            max_values: np.array of maxima values
            max_indices: np.array of corresponding maxima indices
            min_values: np.array of minima values
            min_indices: np.array of corresponding minima indices
    """
    # Check for edge cases
    n = len(ys)
    if n <= 2:
        # Not enough points for peaks - return empty arrays
        # We don't return np.nan here because these are arrays that may be used for indexing later
        return np.array([]), np.array([], dtype=int), np.array([]), np.array([], dtype=int)
    
    # Use NumPy's diff function for efficiency
    diff = np.diff(ys)
    
    # Find where derivative changes sign - doing this in one step
    # This pre-allocates the result arrays for better performance
    sign_changes_mask = np.empty(len(diff)-1, dtype=bool)
    np.not_equal(np.sign(diff[:-1]), np.sign(diff[1:]), out=sign_changes_mask)
    sign_changes = np.nonzero(sign_changes_mask)[0] + 1
    
    if len(sign_changes) == 0:
        # No sign changes means no peaks
        return np.array([]), np.array([], dtype=int), np.array([]), np.array([], dtype=int)
    
    # Calculate the directions once
    directions = diff[sign_changes-1] > 0
    
    # Get maxima and minima using boolean masking
    max_indices = sign_changes[directions]
    min_indices = sign_changes[~directions]
    
    # Get the values from original array
    max_values = ys[max_indices]
    min_values = ys[min_indices]
    
    return max_values, max_indices, min_values, min_indices

