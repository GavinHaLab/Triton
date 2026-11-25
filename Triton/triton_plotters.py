# Robert Patton, rpatton@fredhutch.org (Ha Lab)
# v2.0.0, 09/17/2025

# Utility for plotting Triton profile outputs
# N.B. Peak locations plotting is supported. Nucleotide-frequency plotting removed
# as nucleotide frequency data is no longer available in composite mode.

"""
N.B. the numpy array ordering of profile objects:
1: Depth (GC-corrected, normalized by mean flanking depth)
2: 5' fragment end density (GC-corrected, normalized by mean flanking 5' end density)
3: 3' fragment end density (GC-corrected, normalized by mean flanking 3' end density)
4: PN profile (normalized by mean flanking signal in NC signal, pre-FFT)
5: Subnucleosomal depth (fragments < 147 bp, GC-corrected, normalized by mean flanking subnucleosomal depth)
6: Nucleosomal depth (fragments >= 147 bp, GC-corrected, normalized by mean flanking nucleosomal depth)
7: FL Gini-Simpson index (difference from the median flanking index)
8: PN peak locations (-1: trough, 1: peak)
9: A (Adenine)**
10: C (Cytosine)**
11: G (Guanine)**
12: T (Thymine)**
** Nucleotide channels are removed as they are not available in composite mode.
"""

import os
import argparse
import warnings
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import trapezoid

# Suppress the seaborn UserWarning about tight layout
warnings.filterwarnings('ignore', 'The figure layout has changed to tight', UserWarning)

# pretty column names (nucleotide frequencies removed)
cols = ['raw-depth', 'frag-ends-5', 'frag-ends-3', 'pn-signal', 'short-depth', 'long-depth',
        'fl-gini-simpson', 'peaks']
pretty_cols = ['Coverage (GC-Corrected, Flank-Normalized)', '5\' Fragment End Density (Flank-Normalized, Smoothed)', '3\' Fragment End Density (Flank-Normalized, Smoothed)', 'Phased Nucleosomal Signal (Flank-Normalized)',
               'Subnucleosomal Coverage (<147bp, Flank-Normalized)', 'Nucleosomal Coverage (>=147bp, Flank-Normalized)',
               'FL Gini-Simpson Index', 'Called Peaks']
col_map = {col: pretty_cols[i] for i, col in enumerate(cols)}
profile_labels = {'raw-depth': 'Depth',
                  'frag-ends-5': 'End density',
                  'frag-ends-3': 'End density',
                  'pn-signal': 'PN depth',
                  'short-depth': 'Depth',
                  'long-depth': 'Depth',
                  'fl-gini-simpson': 'Difference from flanks'}


def smooth_data(data, window=75, fill_nans=True, fill_value=0.0):
    """
    Performs Gaussian kernel smoothing with options for NaN handling
        Parameters:
            data (vector-like array): signal profile from Triton
            window (int): window-size for smoothing (determines sigma of Gaussian)
            fill_nans (bool): if True, NaN values are replaced with fill_value; if False, NaNs are preserved
            fill_value (float): value to use for filling NaNs when fill_nans=True
        Returns:
            smoothed_data
    """
    # Convert input to numpy array
    data_arr = np.asarray(data)
    n = len(data_arr)
    # Preserve the original data type
    dtype = data_arr.dtype
    
    # Create a copy of the data 
    data_filled = np.copy(data_arr)
    nan_mask = np.isnan(data_arr)
    
    # If all values are NaN, return an array of NaNs
    if np.all(nan_mask):
        return np.full_like(data_arr, np.nan, dtype=dtype)
        
    # Handle NaN values based on the fill_nans parameter
    if fill_nans:
        # Replace NaNs with the specified fill value
        data_filled[nan_mask] = fill_value
    
    # Initialize smoothed data array
    smoothed_data = np.zeros_like(data_arr, dtype=dtype)
    
    # Create Gaussian kernel
    sigma = window / 6.0  # Set sigma so ~99% of Gaussian falls within the window
    kernel_size = window * 2 + 1
    kernel_half = kernel_size // 2
    x = np.arange(-kernel_half, kernel_half + 1)
    kernel = np.exp(-(x**2) / (2 * sigma**2))
    kernel = kernel / np.sum(kernel)  # Normalize the kernel to sum to 1
    
    # Apply the kernel to smooth the data with edge handling
    for i in range(n):
        # If we're preserving NaNs and the current point is NaN, keep it as NaN
        if not fill_nans and np.isnan(data_arr[i]):
            smoothed_data[i] = np.nan
            continue
            
        # Calculate kernel boundaries with edge handling
        k_start = max(0, kernel_half - i)  # How much kernel to skip at start
        k_end = min(kernel_size, kernel_size - (i + kernel_half - n + 1))  # How much kernel to use at end
        # Calculate data boundaries with edge handling
        d_start = max(0, i - kernel_half)
        d_end = min(n, i + kernel_half + 1)
        # Extract appropriate portion of kernel and data
        used_kernel = kernel[k_start:k_end]
        used_data = data_filled[d_start:d_end]
        
        # If preserving NaNs, we need to adjust the kernel based on valid data points
        if not fill_nans:
            # Create a mask for non-NaN values in the data window
            valid_mask = ~np.isnan(used_data)
            # If no valid data in the window, result is NaN
            if not np.any(valid_mask):
                smoothed_data[i] = np.nan
                continue
            # Otherwise, use only valid data points and adjust kernel weights
            used_data = used_data[valid_mask]
            used_kernel = used_kernel[valid_mask]
            # Re-normalize kernel to sum to 1
            if np.sum(used_kernel) > 0:
                used_kernel = used_kernel / np.sum(used_kernel)
        
        # Apply kernel
        if len(used_kernel) > 0:
            smoothed_data[i] = np.sum(used_kernel * used_data)
        else:
            # Fall back if we have issues
            smoothed_data[i] = np.nan if not fill_nans else np.mean(used_data)
            
    return smoothed_data


def _bootstrap_mean_ci_by_position(sample_data, loc_name, n_boot=1000, ci=0.95, seed=0, chunk_size=200):
    """Compute a bootstrap CI of the mean profile at each position.

    Returns:
        x_vals (list[float|int]), means (list[float]), ci_lower (list[float]), ci_upper (list[float])

    Notes:
        - Bootstraps across samples within a group.
        - Handles missing values by using NaN-aware means/percentiles.
        - Uses chunked bootstrapping to control memory usage.
    """
    if sample_data.empty:
        return [], [], [], []

    # Wide matrix: rows = samples, cols = positions
    wide = sample_data.pivot(index='sample', columns=loc_name, values='value').sort_index(axis=1)
    x_vals = wide.columns.to_numpy()
    mat = wide.to_numpy(dtype=np.float32)
    n_samples = mat.shape[0]

    means = np.nanmean(mat, axis=0)
    if n_samples <= 1:
        # With one sample, CI collapses to the mean.
        return x_vals.tolist(), means.tolist(), means.tolist(), means.tolist()

    alpha = 1.0 - ci
    q_lo = 100.0 * (alpha / 2.0)
    q_hi = 100.0 * (1.0 - alpha / 2.0)

    rng = np.random.default_rng(seed)
    boot_means_accum = []

    remaining = int(n_boot)
    while remaining > 0:
        b = min(int(chunk_size), remaining)
        idx = rng.integers(0, n_samples, size=(b, n_samples))
        # mat[idx] => (b, n_samples, n_pos)
        with np.errstate(all='ignore'):
            boot_means = np.nanmean(mat[idx, :], axis=1)  # (b, n_pos)
        boot_means_accum.append(boot_means)
        remaining -= b

    boot_means_all = np.concatenate(boot_means_accum, axis=0)
    ci_lower = np.nanpercentile(boot_means_all, q_lo, axis=0)
    ci_upper = np.nanpercentile(boot_means_all, q_hi, axis=0)

    return x_vals.tolist(), means.tolist(), ci_lower.tolist(), ci_upper.tolist()


# Nucleotide variation function removed - not applicable without nucleotide frequency data


def plot_profiles(name, data, region, fragmentation_data, samples, palette=None, categories=None):
    """
    Plots all profiles in a two-column layout with fragment length distributions.
        Parameters:
            name (string): name of site being plotted
            data (pandas long df): signal profile(s) from Triton
            region (bool): whether to plot region-based profiles (0 at left side) or not (0 in middle)
            fragmentation_data (list): fragmentation profile data for each sample
            samples (list): sample names corresponding to fragmentation_data
            palette (target:color dictionary): palette for subtypes if categories is passed
            categories (dict): mapping from sample names to category labels
    """
    sns.set_context("notebook", font_scale=1.1)

    # Define the profiles we want to plot and their organization (nucleotide variation removed)
    left_profiles = ['pn-signal', 'raw-depth', 'frag-ends-5', 'frag-ends-3']
    right_profiles = ['short-depth', 'long-depth', 'fl-gini-simpson']
    
    # Filter data to only include our desired profiles
    all_profiles = left_profiles + right_profiles
    data = data[data['profile'].isin(all_profiles)].copy()

    # End densities are sparse per-bp counts (linear tracks), so they are smoothed for display, preserving NaNs
    for end_profile in ('frag-ends-5', 'frag-ends-3'):
        for sample in data[data['profile'] == end_profile]['sample'].unique():
            mask = (data['profile'] == end_profile) & (data['sample'] == sample)
            values = data.loc[mask, 'value'].values
            original_dtype = values.dtype
            smoothed = smooth_data(values, fill_nans=False)
            if len(smoothed) == len(values):
                data.loc[mask, 'value'] = smoothed.astype(original_dtype)

    # Update column mappings (nucleotide variation removed)
    extended_col_map = col_map.copy()
    
    # Update profile labels (nucleotide variation removed)
    extended_profile_labels = profile_labels.copy()
    
    data.loc[:, 'profile'] = data['profile'].map(extended_col_map)
    loc_name = 'Location relative to start (bp)' if region else 'Location relative to center (bp)'
    data = data.rename(columns={'profile': 'Profile', 'label': 'Sample / Group', 'loc': loc_name})

    xmin, xmax = data[loc_name].min(), data[loc_name].max() + 1

    # Create custom subplot layout: 2 columns with different heights
    fig = plt.figure(figsize=(16, 12))
    
    # Left column: 4 signal plots (nucleotide variation removed)
    left_gs = fig.add_gridspec(4, 1, left=0.05, right=0.48, top=0.88, bottom=0.1, hspace=0.4)
    
    # Right column: 3 signal plots plus the fragment length density row
    right_signal_gs = fig.add_gridspec(4, 2, left=0.55, right=0.95, top=0.88, bottom=0.1, hspace=0.4, wspace=0.3)

    # Add column titles (closer to signals)
    fig.text(0.265, 0.93, 'Fragment Coverage Signals', ha='center', va='top', fontsize=14, fontweight='bold')
    fig.text(0.75, 0.93, 'Fragment Length-Derived Signals', ha='center', va='top', fontsize=14, fontweight='bold')
    
    # Plot left column profiles
    left_profile_names = [extended_col_map[p] for p in left_profiles]
    first_ax = None  # Keep reference for legend
    for i, profile_name in enumerate(left_profile_names):
        ax = fig.add_subplot(left_gs[i])
        if first_ax is None:
            first_ax = ax  # Save first axis for legend
        profile_data = data[data['Profile'] == profile_name]
        
        # Add faint vertical line at x=0 behind signals
        ax.axvline(x=0, color='lightgray', linestyle=':', alpha=0.5, zorder=0)
        
        # Plot each sample group
        for sample_group in profile_data['Sample / Group'].unique():
            sample_data = profile_data[profile_data['Sample / Group'] == sample_group]
            color = palette.get(sample_group) if palette else None
            
            if categories is not None:
                # When categories are used, plot a bootstrap 95% CI of the mean profile at each position.
                x_vals, means, ci_lower, ci_upper = _bootstrap_mean_ci_by_position(
                    sample_data=sample_data,
                    loc_name=loc_name,
                    n_boot=1000,
                    ci=0.95,
                    seed=0,
                    chunk_size=200,
                )
                
                # Plot mean line and CI band
                line = ax.plot(x_vals, means, label=sample_group, alpha=0.8, color=color, linewidth=2)[0]
                band_color = color if color is not None else line.get_color()
                ax.fill_between(x_vals, ci_lower, ci_upper, alpha=0.2, color=band_color)
            else:
                # Individual sample plotting (original behavior)
                ax.plot(sample_data[loc_name], sample_data['value'], 
                       label=sample_group, alpha=0.8, color=color, linewidth=1.5)
        
        ax.set_ylabel(extended_profile_labels[left_profiles[i]], fontsize=10)
        ax.set_title(profile_name, fontsize=11, pad=5)
        
        # Only show x-axis label on bottom plot
        if i == len(left_profiles) - 1:
            ax.set_xlabel(loc_name, fontsize=10)
        
        ax.set_xlim(xmin, xmax)
    
    # Plot right column profiles (first 3 rows, spanning both columns)
    right_profile_names = [extended_col_map[p] for p in right_profiles]
    for i, profile_name in enumerate(right_profile_names):
        ax = fig.add_subplot(right_signal_gs[i, :])  # Span both columns for signals
        profile_data = data[data['Profile'] == profile_name]
        
        # Add faint vertical line at x=0 behind signals
        ax.axvline(x=0, color='lightgray', linestyle=':', alpha=0.5, zorder=0)
        
        # Plot each sample group
        for sample_group in profile_data['Sample / Group'].unique():
            sample_data = profile_data[profile_data['Sample / Group'] == sample_group]
            color = palette.get(sample_group) if palette else None
            
            if categories is not None:
                # When categories are used, plot a bootstrap 95% CI of the mean profile at each position.
                x_vals, means, ci_lower, ci_upper = _bootstrap_mean_ci_by_position(
                    sample_data=sample_data,
                    loc_name=loc_name,
                    n_boot=1000,
                    ci=0.95,
                    seed=0,
                    chunk_size=200,
                )
                
                # Plot mean line and CI band
                line = ax.plot(x_vals, means, label=sample_group, alpha=0.8, color=color, linewidth=2)[0]
                band_color = color if color is not None else line.get_color()
                ax.fill_between(x_vals, ci_lower, ci_upper, alpha=0.2, color=band_color)
            else:
                # Individual sample plotting (original behavior)
                ax.plot(sample_data[loc_name], sample_data['value'], 
                       label=sample_group, alpha=0.8, color=color, linewidth=1.5)
        
        ax.set_ylabel(extended_profile_labels[right_profiles[i]], fontsize=10)
        ax.set_title(profile_name, fontsize=11, pad=5)
        
        # Only show x-axis label on last signal plot
        if i == len(right_profiles) - 1:
            ax.set_xlabel(loc_name, fontsize=10)
        
        ax.set_xlim(xmin, xmax)
    
    # Fragment plot should align with a single signal plot height but be 2/3 width
    # Get the position of the last row for height alignment
    row3_subplot_spec = right_signal_gs[3, :]  # Full last row
    row3_pos = row3_subplot_spec.get_position(fig)
    
    # Calculate 2/3 width of the signal plot area
    signal_width = row3_pos.x1 - row3_pos.x0  # Full width of signal area
    frag_width = signal_width * (2.0/3.0)  # 2/3 of signal width
    
    # Position fragment plot at left edge with 2/3 width
    frag_left = row3_pos.x0
    frag_bottom = row3_pos.y0
    frag_height = row3_pos.y1 - row3_pos.y0
    
    ax_frag = fig.add_axes([frag_left, frag_bottom, frag_width, frag_height])
    
    # Legend positioned to the right of the fragment plot (in the remaining 1/3 space)
    legend_left = frag_left + frag_width + 0.02  # Small gap after fragment plot
    legend_width = (row3_pos.x1 - legend_left) * 0.8  # Use 80% of remaining space
    legend_height = frag_height * 0.6  # 60% of fragment plot height
    legend_bottom = frag_bottom + (frag_height - legend_height) / 2  # Center vertically
    
    ax_legend = fig.add_axes([legend_left, legend_bottom, legend_width, legend_height])
    ax_legend.axis('off')  # Hide axes
    
    # Create legend using handles from the first signal plot (which has the actual colors matplotlib assigned)
    if first_ax is not None:
        handles, labels = first_ax.get_legend_handles_labels()
        ax_legend.legend(handles=handles, labels=labels, loc='upper left', fontsize=10, frameon=True)
    
    # Add faint vertical line at fragment length 147
    ax_frag.axvline(x=147, color='lightgray', linestyle=':', alpha=0.5, zorder=0)
    
    # Plot fragment length distributions
    frag_lengths = np.arange(1, 501)  # 1 to 500 (as per Triton.py)
    
    # Collect fragment data by category
    frag_data_by_category = {}
    
    for sample_idx, (sample, frag_item) in enumerate(zip(samples, fragmentation_data)):
        if frag_item is None:
            # optional input
            continue

        # Support either an already-open NpzFile or a file path.
        # For large sample sets we prefer file paths to avoid exhausting OS file descriptors.
        if isinstance(frag_item, str):
            if not os.path.exists(frag_item):
                continue
            with np.load(frag_item, allow_pickle=True) as frag_data:
                _collect_fragment_data(
                    ax_frag=ax_frag,
                    name=name,
                    frag_data=frag_data,
                    sample=sample,
                    categories=categories,
                    palette=palette,
                    frag_lengths=frag_lengths,
                    frag_data_by_category=frag_data_by_category,
                )
        else:
            frag_data = frag_item
            _collect_fragment_data(
                ax_frag=ax_frag,
                name=name,
                frag_data=frag_data,
                sample=sample,
                categories=categories,
                palette=palette,
                frag_lengths=frag_lengths,
                frag_data_by_category=frag_data_by_category,
            )

    # Plot fragment data by category
    for sample_group, group_data in frag_data_by_category.items():
        sample_color = palette.get(sample_group) if palette else None
        
        # (moved into helper)
        
        if categories is not None and len(group_data) > 1:
            # Plot 95% CI band for categories with multiple samples
            group_array = np.array(group_data)
            means = np.mean(group_array, axis=0)
            stds = np.std(group_array, axis=0)
            n = len(group_data)
            
            # Calculate 95% CI
            sems = stds / np.sqrt(n)
            ci_lower = means - 1.96 * sems  # Approximate 95% CI
            ci_upper = means + 1.96 * sems
            
            # Plot mean line and CI band
            ax_frag.plot(frag_lengths, means, label=f'{sample_group}', alpha=0.8, 
                        color=sample_color, linewidth=2)
            ax_frag.fill_between(frag_lengths, ci_lower, ci_upper, alpha=0.2, color=sample_color)
        else:
            # Plot individual sample (original behavior)
            for data in group_data:
                ax_frag.plot(frag_lengths, data, label=f'{sample_group}', alpha=0.8, 
                           color=sample_color, linewidth=1.5)
    
    ax_frag.set_xlabel('Fragment Length (bp)', fontsize=10)
    ax_frag.set_ylabel('Density', fontsize=10)
    # Remove the title as requested
    ax_frag.set_xlim(1, 500)
    # The plot dimensions are already set to be square-ish by the add_axes call above
    
    # Save the figure without tight_layout to avoid warnings with manual gridspec
    plt.savefig(name + '-Profiles_all.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    return


def _collect_fragment_data(
    *,
    ax_frag,
    name,
    frag_data,
    sample,
    categories,
    palette,
    frag_lengths,
    frag_data_by_category,
):
    # Skip if sample not in categories (consistent with signal plotting logic)
    if categories is not None and sample not in categories:
        return

    # Access the fragment_profiles data (new format only)
    if 'fragment_data' not in frag_data.files or 'fragment_columns' not in frag_data.files:
        return

    fragment_data = frag_data['fragment_data']
    fragment_columns = frag_data['fragment_columns']

    # Reconstruct DataFrame from saved data and columns
    df = pd.DataFrame(fragment_data, columns=fragment_columns)

    if 'site' not in df.columns:
        return

    site_match = df[df['site'] == name]
    if not site_match.empty:
        frag_cols = [col for col in df.columns if col.startswith('frag_len_')]
        site_frag_data = site_match[frag_cols].iloc[0].values
    else:
        # Fallback to first available site
        if len(df) == 0:
            return
        frag_cols = [col for col in df.columns if col.startswith('frag_len_')]
        site_frag_data = df[frag_cols].iloc[0].values

    site_frag_data = site_frag_data.astype(np.float64)
    if len(site_frag_data) != len(frag_lengths):
        print(f"ERROR: Shape mismatch for sample '{sample}' at site '{name}'")
        return

    area = trapezoid(site_frag_data, frag_lengths)
    if area > 0:
        site_frag_data = site_frag_data / area

    sample_group = categories.get(sample, sample) if categories else sample
    if sample_group not in frag_data_by_category:
        frag_data_by_category[sample_group] = []

    smoothed_data = gaussian_filter1d(site_frag_data, sigma=3)
    frag_data_by_category[sample_group].append(smoothed_data)


def load_file_data(file_path, delimiter='\t', header=None):
    """
    Helper function to load data from files
    """
    if file_path:
        return pd.read_table(file_path, sep=delimiter, header=header).set_index(0).to_dict()[1]
    return None


def _expand_inputs(input_args):
    """Expand globs and normalize inputs into a flat list of existing paths."""
    expanded = []
    for item in input_args:
        if any(ch in item for ch in ['*', '?', '[', ']']):
            matches = glob.glob(item)
            if matches:
                expanded.extend(matches)
            else:
                expanded.append(item)
        else:
            expanded.append(item)
    return expanded


def _discover_samples(input_args):
    """Return list of sample records from input paths.

    Accepts either:
    - explicit sample directories (each containing <sample>_TritonSignalProfiles.npz)
    - a root results directory (containing per-sample subdirectories)
    """
    sample_records = []
    seen_signal_paths = set()

    for path in _expand_inputs(input_args):
        if not os.path.isdir(path):
            continue

        path = path.rstrip('/')
        base = os.path.basename(path)
        direct_signal = os.path.join(path, f'{base}_TritonSignalProfiles.npz')

        if os.path.exists(direct_signal):
            signal_path = direct_signal
            sample_name = base
            frag_path = os.path.join(path, f'{sample_name}_TritonFragmentationProfiles.npz')
            if signal_path not in seen_signal_paths:
                sample_records.append({
                    'sample': sample_name,
                    'dir': path,
                    'signal': signal_path,
                    'frag': frag_path if os.path.exists(frag_path) else None,
                })
                seen_signal_paths.add(signal_path)
            continue

        # Otherwise treat as a root directory: look one level down for sample dirs.
        try:
            children = [os.path.join(path, d) for d in os.listdir(path)]
        except OSError:
            continue

        for child in sorted(children):
            if not os.path.isdir(child):
                continue
            child_base = os.path.basename(child.rstrip('/'))

            # Prefer the conventional naming if present.
            child_signal = os.path.join(child, f'{child_base}_TritonSignalProfiles.npz')
            if os.path.exists(child_signal):
                sample_name = child_base
                signal_path = child_signal
            else:
                # Otherwise, accept any *_TritonSignalProfiles.npz in the child dir.
                matches = glob.glob(os.path.join(child, '*_TritonSignalProfiles.npz'))
                if not matches:
                    continue
                signal_path = matches[0]
                sample_name = os.path.basename(signal_path).removesuffix('_TritonSignalProfiles.npz')

            frag_path = os.path.join(child, f'{sample_name}_TritonFragmentationProfiles.npz')

            if signal_path in seen_signal_paths:
                continue
            sample_records.append({
                'sample': sample_name,
                'dir': child,
                'signal': signal_path,
                'frag': frag_path if os.path.exists(frag_path) else None,
            })
            seen_signal_paths.add(signal_path)

    return sample_records


def main():
    parser = argparse.ArgumentParser(description='\n### triton_plotters.py ### plots Triton output profiles')
    parser.add_argument('-i', '--input', help='one or more directories containing Triton output files; '
                                              'directory names will be used as sample names.', nargs='*', required=True)
    parser.add_argument('-c', '--categories', help='tsv file containing matched sample names (column 1) and '
                                                   'categories (column 2) to color samples by composite '
                                                   'categories rather than individually, with a .95 confidence '
                                                   'interval; sample names in column 1 MUST match the sample '
                                                   'names passed to Triton (i.e. sample1_TritonProfiles.npz). '
                                                   'Any passed inputs not present in this file will be dropped. '
                                                   'This file should NOT contain a header. DEFAULT = None (plot '
                                                   'all samples individually).',
                        required=False, default=None)
    parser.add_argument('-p', '--palette', help='tsv file containing matched categories/samples (column 1) and HEX '
                                                'color codes (column 2, e.g. #0077BB) to specify what color to use '
                                                'for samples/categories. Sample/category names must exactly match '
                                                'sample names passed to Triton, or category names passed with the '
                                                '-c/--categories option. Will error if inputs/categories include '
                                                'labels not present in palette. This file should NOT contain a header. '
                                                'DEFAULT = None (use Seaborn default colors).',
                        required=False, default=None)
    parser.add_argument('-s', '--sites', help='file containing a list (row-wise) of sites to plot. This file should '
                                              'NOT contain a header. DEFAULT = None (plotting all available sites).',
                        required=False, default=None)
    parser.add_argument('-r', '--region_axis', help='if set, data is "region" - sets 0-point to the left side instead of the middle. '
                                               'DEFAULT = False.', action='store_true')

    args = parser.parse_args()
    print('### Running triton_plotters.py ###')

    # Load configuration data
    categories = load_file_data(args.categories)
    palette = load_file_data(args.palette)
    sites = open(args.sites).read().splitlines() if args.sites else None

    # Discover samples from inputs (accept either sample dirs or a root results/ dir)
    sample_records = _discover_samples(args.input)
    if not sample_records:
        print('Error: No valid samples found. Please check your input directories.')
        quit()

    # Filter samples by categories and show warnings once
    if categories is not None:
        filtered = []
        for rec in sample_records:
            if rec['sample'] not in categories:
                print(f"Sample {rec['sample']} not found in categories. Skipping.")
            else:
                filtered.append(rec)
        sample_records = filtered
        if not sample_records:
            print('Error: No valid samples remain after category filtering.')
            quit()

    samples = [rec['sample'] for rec in sample_records]
    fragmentation_data = [rec['frag'] for rec in sample_records]  # paths (or None)

    print(f'Loaded {len(samples)} samples: {", ".join(samples[:20])}' + (' ...' if len(samples) > 20 else ''))

    # Determine sites to iterate (from the first sample's signal npz)
    with np.load(sample_records[0]['signal'], allow_pickle=True) as first_npz:
        available_sites = list(first_npz.files)


    # Process each site (streaming per site, opening/closing per-sample npz to avoid FD exhaustion)
    for site in available_sites:
        if sites is not None and site not in sites:
            continue

        print(f'*** Processing samples for {site}')

        dfs = []
        for rec in sample_records:
            sample = rec['sample']
            signal_path = rec['signal']

            with np.load(signal_path, allow_pickle=True) as test_data:
                if site not in test_data.files:
                    continue

                data_array = test_data[site]
                if pd.isnull(data_array).all():
                    continue

                if len(data_array.shape) != 2:
                    continue

                # Create a DataFrame for the sample - use only the signal channels (no nucleotide channels)
                data_to_use = data_array[:len(cols), :]
                channels_used = data_to_use.shape[0]

                tdf = pd.DataFrame(data_to_use.T, columns=cols[:channels_used])
                tdf['loc'] = np.arange(len(tdf))
                if not args.region_axis:
                    tdf['loc'] -= len(tdf) / 2
                tdf['sample'] = sample
                tdf['label'] = categories.get(sample, sample) if categories else sample
                dfs.append(tdf)

        if not dfs:
            print(f'No data for {site}. Skipping.')
            continue

        df = pd.concat(dfs) if len(dfs) > 1 else dfs[0]
        print(f'*** Plotting {site}')

        if categories:
            df = df[df['label'].notna()]
            if len(df['label'].unique()) < 1:
                print('No samples to plot after matching against the provided categories file. Exiting.')
                quit()

        df_melted = pd.melt(df, id_vars=['sample', 'loc', 'label'], value_vars=cols, var_name='profile')
        plot_profiles(site, df_melted, args.region_axis, fragmentation_data, samples, palette=palette, categories=categories)

if __name__ == "__main__":
    main()
