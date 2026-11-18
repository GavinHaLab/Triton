# Robert Patton, rpatton@fredhutch.org
# v2.0.0, 09/30/2025

import os
import sys
import pysam
import pickle
import argparse
import pandas as pd
import multiprocessing
from functools import partial
import numpy as np
from scipy.fft import rfft, irfft, rfftfreq
from triton_helpers import *

# constants and defaults
window_size = 2000  # default window size for windowed modes: +/-1000bp from center
flank_size = 5000  # default flank/buffer size, added to both sides of the region or window
base_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
iupac_trans = str.maketrans('ACGTURYSWKMBDHVNX',
                            '06600336033422430')  # likelihood of GC content for each bp code x6 (common factor)
freq_max = 1.0 / 147.0  # theoretical minimum nucleosome period is 147 bp -> f = 0.006803
low_1, high_1 = 0.00565, 0.00680  # range_1: T = 147-177 bp (f = 0.00565 - 0.00680, "small linker"/compact chromatin)
low_2, high_2 = 0.00483, 0.00565  # range_2: T = 177-207 bp (f = 0.00483 - 0.00565, "large linker"/open chromatin)

# Global variables for pool initializer
global_gc_bias = None
global_fdict = None

def pool_initializer(gc_bias, fdict):
    """
    Pool initializer function to preload shared data across workers.
    This reduces memory overhead by sharing GC bias dict and fragment dict.
    """
    global global_gc_bias, global_fdict
    global_gc_bias = gc_bias
    global_fdict = fdict


def process_fragments_fast(site_reads, start_pos, frag_range, map_q, gc_bias, window_ref_sequence, 
                          depth, nc_signal, fragment_lengths, fragment_length_profile, 
                          fragment_5end_profile, fragment_3end_profile, fdict, 
                          roi_length, flank_size):
    """
    Fast fragment processing using one-mate rule and TLEN, updating profiles in place.
    No intermediate lists - direct profile updates for maximum efficiency.
    """
    min_frag_size = frag_range[0]
    max_frag_size = frag_range[1]
    has_gc_bias = gc_bias is not None

    # Raw fragment counters (pre-GC correction):
    # - window: fragment center falls within the window (excluding flanks)
    # - roi: fragment fully contained within the ROI (window + flanks)
    n_frag_raw_window = 0
    n_frag_raw_roi = 0
    
    for read in site_reads:
        # Fast filtering - fail early on common exclusions
        if (not read.is_proper_pair or read.mate_is_unmapped or
            read.is_secondary or read.is_supplementary or read.is_duplicate or
            read.mapping_quality < map_q or read.reference_id != read.next_reference_id):
            continue
        tlen = read.template_length
        if tlen == 0:
            continue
            
        # One-mate rule: use Read 1 regardless of TLEN sign
        if not read.is_read1:
            continue
        frag_len = abs(tlen)
        if frag_len < min_frag_size or frag_len > max_frag_size:
            continue
            
        # Calculate fragment coordinates
        frag_start = min(read.reference_start, read.next_reference_start) - start_pos
        frag_end = frag_start + frag_len
        
        # Check bounds
        if frag_start < 0 or frag_end > len(window_ref_sequence):
            continue
            
        # GC bias correction
        if has_gc_bias:
            # Calculate GC content using IUPAC translate
            fragment_seq = window_ref_sequence[frag_start:frag_end].translate(iupac_trans)
            fragment_gc_content = int((np.array(list(fragment_seq), dtype=int).sum() / 6).round())
            # Look up bias factor
            try:
                fragment_bias = gc_bias[frag_len][fragment_gc_content]
                if not (fragment_bias > 0):  # Skip NaN or non-positive bias
                    continue
            except (KeyError, IndexError):
                continue
            bias_factor = 1.0 / fragment_bias
        else:
            bias_factor = 1.0

        # Calculate fragment center position
        center_pos = frag_start + frag_len // 2
        center_in_window = flank_size <= center_pos <= roi_length - (flank_size + 1)
        
        # Update fragment length histogram if fragment center is in window
        if center_in_window:
            n_frag_raw_window += 1
            fragment_lengths[frag_len - 1] += bias_factor
        
        # Get nucleosome density profile if applicable
        nc_density = fdict[frag_len] if 147 <= frag_len <= 500 else None
        
        # Vectorized fragment spanning - update depth and nc_signal profiles in place
        if 0 <= frag_start < roi_length and 0 <= frag_end <= roi_length:
            n_frag_raw_roi += 1
            depth[frag_start:frag_end] += bias_factor
            if nc_density is not None:
                nc_slice = nc_density[:frag_len] * bias_factor
                nc_signal[frag_start:frag_end] += nc_slice
            
            # Update fragment length profile for entire ROI (including flanks) to enable proper flank extraction
            fragment_length_profile[frag_len - 1, frag_start:frag_end] += bias_factor
        
        # Update fragment end profiles - 5' end is always leftmost, 3' end is rightmost
        end_5_pos = frag_start
        end_3_pos = frag_end - 1
            
        # Track ends across full ROI
        if 0 <= end_5_pos < roi_length:
            fragment_5end_profile[end_5_pos] += bias_factor
        if 0 <= end_3_pos < roi_length:
            fragment_3end_profile[end_3_pos] += bias_factor
    return n_frag_raw_window, n_frag_raw_roi


def generate_profile(region, params):
    """
    Generates single or composite signal profiles and extracts features (fragmentation, nucleosome phasing,
    and profile shape-based) for a single sample. Utilizes functions from triton_helpers.py.
        Parameters:
            region (string): either a file path pointing to a BED-like file (composite) or a BED-like single annotation
            params (list): bam_path, frag_range, gc_bias, ref_seq_path, map_q, fdict, run_mode, chr_idx, start_idx, stop_idx, site_idx, strand_idx, pos_idx
        Returns:
            site: annotation name if stacked, "name" from BED file for each region otherwise
                ### Fragmentation (FL) Features (using all fragments in passed length range) ###
            fl-mean: FL distribution mean (absolute, in bp)
            fl-stdev: FL distribution standard deviation (absolute, in bp)
            fl-skew: FL distribution skewness (absolute)
            fl-kurtosis: FL distribution kurtosis (absolute)
            fl-subnucleosomal-ratio: FL subnucleosomal (x < 147 / x >= 147) ratio (log2)
            fl-entropy: FL distribution Normalized Shannon Entropy / Pielou's Evenness (absolute)
            fl-gini-simpson: FL distribution Gini-Simpson index (absolute)
                ### Phased-Nucleosome (PN) Features (FFT-based, using >= 147bp fragments) ###
            pn-compaction-score: ratio of compact (small-linker, 147-177bp) to open (large-linker, 177-207bp) PN signal amplitude (log2)
            pn-mean-spacing: mean spacing between PN peaks (absolute, in bp)
            pn-mean-amplitude: mean amplitude (half peak-to-trough divided by flanking mean) of PN peaks
                ### Profiling Features (Fl or PN signal-based) ###
            mean-region-depth: mean depth in the region (GC-corrected, normalized in composite, raw in region/window)
            central-depth*: PN signal profile value at window center (mean +/-5 bp, normalized by mean flanking PN signal)
            central-entropy*: fragmentation Shannon Entropy signal at window center (mean +/-5 bp, robust z-score using median and MAD of flanking FL entropy)
            central-gini-simpson*: fragmentation Gini-Simpson index signal at window center (mean +/-5 bp, robust z-score using median and MAD of flanking FL Gini-Simpson)
            window-depth: PN signal profile mean in window (normalized by mean flanking PN signal)
            window-entropy: fragmentation Shannon Entropy signal mean in window (robust z-score using median and MAD of flanking FL entropy)
            window-gini-simpson: fragmentation Gini-Simpson signal mean in window (robust z-score using median and MAD of flanking FL Gini-Simpson)
                ### Region-level signal profiles (nt-resolution) ###
            numpy array: shape 8xL (composite) and 12xL (window/region with sequence) containing:
                1: Depth (GC-corrected, normalized by mean flanking depth)
                2: Fragment end coverage (fraction of 5'+3' fragment ends / total fragments at each position, normalized by mean flanking end coverage)
                3: Fragment end orientation asymmetry ((5' - 3') / (5' + 3') at each position)
                4: PN profile (normalized by mean flanking signal in NC signal, pre-FFT)
                5: FL subnucleosomal ratio [log2(x < 147 / x >= 147)]
                6: FL Shannon Entropy / Pielou's Evenness (robust z-score using median and MAD of flanking signal)
                7: FL Gini-Simpson index (robust z-score using median and MAD of flanking signal)
                8: PN peak locations (-1: trough, 1: peak)
                9: A (Adenine)**
                10: C (Cytosine)**
                11: G (Guanine)**
                12: T (Thymine)**
            skipped-sites: list of annotation lines skipped due to truncation or outlier peaks with MAD > 10 (composite only) 
            * these features are centered region specific, and output as np.nan if window == None
            ** nt signals represent the one-hot encoded reference sequence, and are reported in region/window modes only.
            N.B. "flanking" regions represent the outer +/-flank_size (default +/-5000 bp) surrounding the window or region,
            and are used for smooth signal production at the edges of the window/region and for normalization. Sites with flanking regions that
            overlap chromosome ends or contig breaks will have flanks shortened to as short as 500bp, otherwise they are skipped.
            
            ### Composite Mode Normalization ###
            In composite-window mode, all signals undergo per-site normalization and weighting before aggregation:
            1. Flank normalization (stable): Composite depth and nc_signal are normalized to flanks using a ratio-of-sums estimator
                    (Σ w_i * signal_i) / (Σ w_i * flank_mean_i), which is robust to noisy per-site flank means at low depth
            2. Fragment-based probability normalization: Fragment length distributions, fragment length profiles, and fragment end 
               profiles are converted to probability distributions (normalized by their respective totals) to ensure contribution 
               is proportional to fragmentation patterns rather than absolute fragment counts
            3. Weighting: Each site receives weight w = sqrt(n_frag_raw) to balance reliability vs. over-dominance
            4. Weighted aggregation: All normalized signals and probability distributions are multiplied by w and summed
            5. Final normalization: All aggregated signals are divided by total cumulative weight W = Σw
            6. No additional flanking normalization is applied post-aggregation since signals are already per-site normalized
            
            This ensures that sites contribute based on their fragmentation patterns rather than raw fragment counts, 
            with weighting proportional to sqrt(n_frag_raw) to balance reliability while preventing over-dominance.
    """
    bam_path, frag_range, gc_bias, ref_seq_path, map_q, fdict, run_mode, chr_idx, start_idx, stop_idx, site_idx, strand_idx, pos_idx = params
    
    # Use global variables if available (from pool initializer)
    if global_gc_bias is not None:
        gc_bias = global_gc_bias
    if global_fdict is not None:
        fdict = global_fdict
    
    if run_mode == 'composite-window':
        window, stack = window_size, True
    elif run_mode == 'window':
        window, stack = window_size, False
    else:
        window, stack = None, False
    # CRAM files require reference for proper reading
    is_cram = bam_path.lower().endswith('.cram')
    bam = pysam.AlignmentFile(bam_path, 'rb', reference_filename=ref_seq_path if is_cram else None)
    ref_seq = pysam.FastaFile(ref_seq_path)
    fragment_lengths = np.zeros(frag_range[1], dtype=np.float32)  # accumulates fragment length counts (converted to PD before use)
    skipped_sites = []  # for storing skipped sites in composite-window mode
    
    # Pre-compute fragment length bins for diversity metrics (bin width default = 5)
    bin_size = 5
    num_bins = int(np.ceil(frag_range[1] / bin_size))
    # Create bin edges (start indices of each bin)
    bin_edges = np.arange(0, frag_range[1], bin_size)
    # Store bins as tuple for reuse
    frag_bins = (bin_edges, num_bins)

    if stack:
    # assemble depth and fragment profiles for composite sites ---------------------------------------------------------
        site = os.path.basename(region).split('.')[0]  # use the file name as the feature name
        roi_length = window + 2*flank_size  # buffers are added to each end for a smooth FFT at the ROI edges and for flank normalization
        # NOTE: In composite mode, per-site flank normalization should not be performed as a mean-of-ratios
        # (normalize each site by its flank mean, then average) because low-depth sites can have noisy flank
        # means that explode the normalized profile.
        #
        # Instead, use a ratio-of-sums estimator:
        #   composite_depth(x) = (Σ w_i * depth_i(x)) / (Σ w_i * flank_mean_i)
        # This stabilizes the composite in ULP settings while preserving full fragment-span coverage.
        depth_num = np.zeros(roi_length, dtype=np.float32)
        nc_num = np.zeros(roi_length, dtype=np.float32)
        depth_den = 0.0
        nc_den = 0.0
        fragment_length_profile = np.zeros((frag_range[1], roi_length), dtype=np.float32)  # nt-level fragment histogram matrix across full ROI
        # Initialize weighted sum arrays for per-position averaging with NaN masking
        EPC_sum = np.zeros(window_size, dtype=np.float32)  # End coverage per depth sum
        EPC_w = np.zeros(window_size, dtype=np.float32)    # End coverage weights
        ASYM_sum = np.zeros(window_size, dtype=np.float32) # Orientation asymmetry sum
        ASYM_w = np.zeros(window_size, dtype=np.float32)   # Orientation asymmetry weights
        
        # Initialize cumulative weight for per-site normalization
        total_weight = 0.0
        
        # Initialize accumulator for absolute mean depth (GC-corrected, before normalization)
        absolute_mean_depth_sum = 0.0
        
        with open(region.strip(), 'r') as sites_file:
            for entry in sites_file:  # iterate through regions in this particular BED file
                site_depth, site_nc_signal = np.zeros(roi_length, dtype=np.float32), np.zeros(roi_length, dtype=np.float32)
                site_5ends, site_3ends = np.zeros(roi_length, dtype=np.float32), np.zeros(roi_length, dtype=np.float32)
                site_fragment_lengths = np.zeros(frag_range[1], dtype=np.float32)
                site_fragment_length_profile = np.zeros((frag_range[1], roi_length), dtype=np.float32)  # Full ROI for FL profiles
                bed_tokens = entry.strip().split('\t')
                if pos_idx is not None and not bed_tokens[pos_idx].isdigit():
                    continue  # header or typo line
                chrom = str(bed_tokens[chr_idx])
                # Use position column if available, otherwise compute midpoint
                if pos_idx is not None:
                    center_pos = int(bed_tokens[pos_idx])
                else:
                    center_pos = (int(bed_tokens[start_idx]) + int(bed_tokens[stop_idx])) // 2
                if strand_idx is not None:
                    strand = str(bed_tokens[strand_idx])
                    strands_to_process = [strand]  # Process only the specified strand
                else:  # Process both strands and average them with 1/2 weight each
                    """ If strand context is unknown, it would be implying a directionality which is not necessarily present
                    to assume any particular strand. Instead of randomly choosing, we process both strands and average with 1/2 weight each."""
                    strands_to_process = ['+', '-']
                if chrom == 'chrMT':  # in future, change to "if not in list of chromosomes"
                    skipped_sites.append(entry.strip() + '\t' + site + '\tMT_contig')
                    continue
                # Calculate desired positions with full flanks
                desired_start_pos = center_pos - int(roi_length / 2)
                desired_stop_pos = center_pos + int(roi_length / 2)
                # Check if start position would be negative
                if desired_start_pos < 0:
                    skipped_sites.append(entry.strip() + '\t' + site + '\tinsufficient_left_flank')
                    continue
                start_pos = desired_start_pos
                # Process all fragments falling inside the ROI
                site_reads = bam.fetch(chrom, start_pos, desired_stop_pos)
                try:
                    window_ref_sequence = ref_seq.fetch(chrom, start_pos, desired_stop_pos).upper()
                except Exception:
                    # Handle case where stop_pos extends beyond chromosome end
                    skipped_sites.append(entry.strip() + '\t' + site + '\tinsufficient_right_flank')
                    continue
                if len(window_ref_sequence) < roi_length:
                    skipped_sites.append(entry.strip() + '\t' + site + '\tinsufficient_reference')
                    continue
                # Initialize temporary arrays for this site
                temp_depth = np.zeros(roi_length, dtype=np.float32)
                temp_nc_signal = np.zeros(roi_length, dtype=np.float32)
                temp_fragment_length_profile = np.zeros((frag_range[1], roi_length), dtype=np.float32)  # Full ROI for FL profiles
                # Track ends across the full ROI in composite mode to avoid window-edge artifacts
                temp_5ends = np.zeros(roi_length, dtype=np.float32)
                temp_3ends = np.zeros(roi_length, dtype=np.float32)
                temp_fragment_lengths = np.zeros(frag_range[1], dtype=np.float32)
                
                # Process fragments directly into temporary arrays
                n_frag_raw_window, n_frag_raw_roi = process_fragments_fast(
                    site_reads, start_pos, frag_range, map_q, gc_bias, window_ref_sequence,
                    temp_depth, temp_nc_signal, temp_fragment_lengths, temp_fragment_length_profile,
                    temp_5ends, temp_3ends, fdict, roi_length, flank_size
                )
                
                # Now apply strand-specific processing
                if len(strands_to_process) == 1:
                    # Single strand specified - use directly or flip if negative
                    if strands_to_process[0] == '+':
                        site_depth = np.array(temp_depth)
                        site_nc_signal = np.array(temp_nc_signal)
                        site_fragment_length_profile = np.array(temp_fragment_length_profile)
                        site_5ends = np.array(temp_5ends)
                        site_3ends = np.array(temp_3ends)
                        site_fragment_lengths = np.array(temp_fragment_lengths)
                    else:  # negative strand
                        site_depth = temp_depth[::-1]
                        site_nc_signal = temp_nc_signal[::-1]
                        site_fragment_length_profile = np.fliplr(temp_fragment_length_profile)
                        site_5ends = temp_3ends[::-1]  # 5' and 3' are swapped and flipped
                        site_3ends = temp_5ends[::-1]
                        site_fragment_lengths = np.array(temp_fragment_lengths)
                else:
                    # Both strands - average with equal weight to ensure symmetry
                    # Positive strand contribution (as is)
                    pos_depth = temp_depth
                    pos_nc_signal = temp_nc_signal
                    pos_fragment_length_profile = temp_fragment_length_profile
                    pos_5ends = temp_5ends
                    pos_3ends = temp_3ends
                    
                    # Negative strand contribution (mirrored)
                    neg_depth = temp_depth[::-1]
                    neg_nc_signal = temp_nc_signal[::-1]
                    neg_fragment_length_profile = np.fliplr(temp_fragment_length_profile)
                    neg_5ends = temp_3ends[::-1]  # Swap 5' and 3' ends after flipping for true strand symmetry
                    neg_3ends = temp_5ends[::-1]
                    
                    # Average both orientations - this ensures fragment end orientation asymmetry is symmetric about 0
                    site_depth = (pos_depth + neg_depth) * 0.5
                    site_nc_signal = (pos_nc_signal + neg_nc_signal) * 0.5
                    site_fragment_length_profile = (pos_fragment_length_profile + neg_fragment_length_profile) * 0.5
                    site_5ends = (pos_5ends + neg_5ends) * 0.5
                    site_3ends = (pos_3ends + neg_3ends) * 0.5
                    site_fragment_lengths = temp_fragment_lengths  # Fragment lengths are strand-independent
                # Check for problematic outliers using MAD-based filtering
                site_total_coverage = np.sum(site_depth)
                if site_total_coverage == 0:
                    skipped_sites.append(entry.strip() + '\t' + site + '\tzero_coverage')
                    continue
                
                # Use median-based robust outlier detection only for sites with reasonable coverage
                if site_total_coverage > 100:  # Only apply MAD filtering to sites with sufficient coverage
                    raw_median = np.median(site_depth)
                    raw_mad = np.median(np.abs(site_depth - raw_median))
                    
                    # Check for extreme outliers that suggest technical artifacts
                    if raw_mad > 0:  # Avoid division by zero
                        # Use robust scaling: modified z-score with MAD
                        modified_z_scores = 0.6745 * (site_depth - raw_median) / raw_mad
                        # Filter out extreme positive outliers (>10 modified z-score)
                        if np.any(modified_z_scores > 10):
                            skipped_sites.append(entry.strip() + '\t' + site + '\tMAD_outlier')
                            continue
                
                # Per-site normalization and weighting for composite mode
                # A) Calculate flank mean coverage for both signals.
                # Exclude the outermost ~max fragment length from the flanks to avoid ROI-boundary truncation bias.
                edge_exclude = min(int(frag_range[1]), int(flank_size - 1))
                if edge_exclude < 1:
                    edge_exclude = 1
                left_flank_depth = site_depth[edge_exclude:flank_size]
                right_flank_depth = site_depth[-flank_size:-edge_exclude]
                left_flank_nc = site_nc_signal[edge_exclude:flank_size]
                right_flank_nc = site_nc_signal[-flank_size:-edge_exclude]
                site_flank_depth_mean = (np.mean(left_flank_depth) + np.mean(right_flank_depth)) / 2
                site_flank_nc_mean = (np.mean(left_flank_nc) + np.mean(right_flank_nc)) / 2
                # Skip sites with zero flank coverage to avoid division by zero
                if site_flank_depth_mean <= 0 or site_flank_nc_mean <= 0:
                    skipped_sites.append(entry.strip() + '\t' + site + '\tzero_flank_coverage')
                    continue

                # C) Calculate weighting factor based on raw fragments (pre-GC correction).
                # Use ROI-contained fragments (window + flanks) so the weight reflects evidence contributing
                # to both the numerator (full ROI signal) and the denominator (flank mean estimate).
                site_total_fragments_raw = int(n_frag_raw_roi)
                if site_total_fragments_raw == 0:
                    skipped_sites.append(entry.strip() + '\t' + site + '\tzero_fragments')
                    continue
                site_weight = np.sqrt(site_total_fragments_raw)
                
                # Accumulate absolute mean depth (GC-corrected) before normalization for composite mean_depth feature
                site_window_mean_depth = np.mean(site_depth[flank_size:-flank_size])
                absolute_mean_depth_sum += site_window_mean_depth * site_weight
                
                # D) Normalize fragment-based distributions to probability distributions before weighting
                # This ensures contribution is proportional to patterns, not absolute fragment counts
                # Fragment length histogram - convert to probability distribution
                site_total_fragments_gc = np.sum(site_fragment_lengths)
                if site_total_fragments_gc > 0:
                    site_fragment_lengths_prob = site_fragment_lengths / site_total_fragments_gc
                else:
                    site_fragment_lengths_prob = np.zeros_like(site_fragment_lengths, dtype=np.float32)
                
                # Fragment length profile - normalize each position by total fragments at that position
                counts = site_fragment_length_profile
                pos_tot = counts.sum(axis=0, keepdims=True)
                # Positions with no fragments stay at 0
                with np.errstate(divide='ignore', invalid='ignore'):
                    site_fragment_length_profile_prob = np.where(
                        pos_tot > 0,
                        counts / pos_tot,
                        0.0  # Positions with no fragments get 0 probability
                    )
                
                # Calculate EPC (ends per depth) and orientation asymmetry at site level.
                # Compute on the full ROI and flank-normalize using TRUE flanks (outside the window),
                # then slice to window for aggregation.
                eps = 1e-6
                site_total_fragment_ends_roi = site_5ends + site_3ends
                with np.errstate(divide='ignore', invalid='ignore'):
                    epc_roi = site_total_fragment_ends_roi / (site_depth + eps)
                    asym_roi = (site_5ends - site_3ends) / (site_total_fragment_ends_roi + eps)

                # Flank-normalize EPC using flanks (excluding the window)
                left_flank_epc = epc_roi[edge_exclude:flank_size]
                right_flank_epc = epc_roi[-flank_size:-edge_exclude]
                flank_epc = (np.nanmean(left_flank_epc) + np.nanmean(right_flank_epc)) / 2
                if (not np.isfinite(flank_epc)) or flank_epc <= 0:
                    flank_epc = 1.0

                epc = epc_roi[flank_size:-flank_size] / flank_epc
                asym = asym_roi[flank_size:-flank_size]
                
                # Use site weight for per-position weighted averaging
                w_pos = site_weight
                
                # Mask valid values and accumulate weighted sums
                valid_epc = np.isfinite(epc)
                valid_asym = np.isfinite(asym)
                
                EPC_sum[valid_epc] += epc[valid_epc] * w_pos
                EPC_w[valid_epc] += w_pos
                ASYM_sum[valid_asym] += asym[valid_asym] * w_pos
                ASYM_w[valid_asym] += w_pos
                
                # E) Weight the normalized signals and probability distributions, then accumulate
                # Ratio-of-sums flank normalization
                depth_num += site_depth * site_weight
                nc_num += site_nc_signal * site_weight
                depth_den += site_flank_depth_mean * site_weight
                nc_den += site_flank_nc_mean * site_weight
                fragment_lengths += site_fragment_lengths_prob * site_weight  # Weight probability distribution by sqrt(n)
                fragment_length_profile += site_fragment_length_profile_prob * site_weight
                # Fragment end coverage and asymmetry already accumulated above with per-position weighting
                # Accumulate total weight for final normalization
                total_weight += site_weight
        
        # E) Final normalization by cumulative weight W after all sites processed
        if total_weight > 0 and depth_den > 0 and nc_den > 0:
            # Finalize ratio-of-sums normalized composite signals
            depth = depth_num / depth_den
            nc_signal = nc_num / nc_den
            fragment_lengths /= total_weight  # Normalize weighted probability distribution
            fragment_length_profile /= total_weight
            # Compute per-position weighted averages for end coverage and asymmetry
            fragment_end_coverage_profile = EPC_sum / (EPC_w + 1e-12)
            orientation_asymmetry_profile = ASYM_sum / (ASYM_w + 1e-12)
            
            # Calculate absolute mean depth for composite mode (GC-corrected, weighted average across sites)
            composite_absolute_mean_depth = absolute_mean_depth_sum / total_weight
            
            # NOTE: fragment_end_coverage_profile is already flank-normalized per-site using TRUE flanks.
            # Do not renormalize using window edges here (this introduces window-edge artifacts).
        else:
            # If no sites contributed (shouldn't happen but safety check)
            return (site,) + (np.nan,) * 17 + ([],) + ([],) + (skipped_sites,)
    else:
    # assemble depth and fragment profiles for sites individually -------------------------------------------------------
        bed_tokens = region.strip().split('\t')
        chrom, site = str(bed_tokens[chr_idx]), str(bed_tokens[site_idx])
        if strand_idx is not None:
            strand = str(bed_tokens[strand_idx])
        else:  # assume + strand if not specified
            strand = '+'
        if window is not None:
            roi_length = window + 2*flank_size
            # Use position column if available, otherwise compute midpoint
            if pos_idx is not None:
                center_pos = int(bed_tokens[pos_idx])
            else:
                center_pos = (int(bed_tokens[start_idx]) + int(bed_tokens[stop_idx])) // 2
            desired_start_pos = center_pos - int(roi_length / 2)
            desired_stop_pos = center_pos + int(roi_length / 2)
            # Adjust start position if it's negative, allow truncated flanks but ensure minimum of 500bp
            start_pos = max(0, desired_start_pos)
            stop_pos = desired_stop_pos
            # In window mode (non-composite), we allow truncated flanks but still need minimum 500bp
            if center_pos - start_pos < 500:  # Check if we have minimum flanking size
                # Return all NaN with reason for insufficient flank (omit redundant site information)
                skipped_msg = f"{region.strip()}\tinsufficient_left_flank"
                return (site,) + (np.nan,) * 17 + ([],) + ([],) + ([skipped_msg],)
            fragment_length_profile = np.zeros((frag_range[1], roi_length), dtype=np.float32)  # Full ROI for FL profiles
            site_5ends, site_3ends = np.zeros(roi_length, dtype=np.float32), np.zeros(roi_length, dtype=np.float32)
            fragment_5end_profile, fragment_3end_profile = site_5ends, site_3ends
            
            # Pre-compute fragment length bins for diversity metrics (bin width = 5)
            bin_size = 5
            num_bins = int(np.ceil(frag_range[1] / bin_size))
            # Create bin edges (start indices of each bin)
            bin_edges = np.arange(0, frag_range[1], bin_size)
            # Store bins as tuple for reuse
            frag_bins = (bin_edges, num_bins)
        else:
            desired_start_pos = int(bed_tokens[start_idx]) - flank_size
            desired_stop_pos = int(bed_tokens[stop_idx]) + flank_size
            # Adjust start position if it's negative, allow truncated flanks but ensure minimum of 500bp
            start_pos = max(0, desired_start_pos)
            stop_pos = desired_stop_pos
            # In region/window mode, we can allow truncated flanks but still need at least 500bp
            if int(bed_tokens[start_idx]) - start_pos < 500:  # Check if we have minimum flanking size
                # Return all NaN with reason for insufficient flank
                skipped_msg = f"{region.strip()}\tinsufficient_left_flank"
                return (site,) + (np.nan,) * 17 + ([],) + ([],) + ([skipped_msg],)
            roi_length = stop_pos - start_pos
            fragment_length_profile = np.zeros((frag_range[1], roi_length), dtype=np.float32)  # Full ROI for FL profiles
            site_5ends, site_3ends = np.zeros(roi_length, dtype=np.float32), np.zeros(roi_length, dtype=np.float32)
            fragment_5end_profile, fragment_3end_profile = site_5ends, site_3ends
            
            # Pre-compute fragment length bins for diversity metrics (bin width = 5)
            bin_size = 5
            num_bins = int(np.ceil(frag_range[1] / bin_size))
            # Create bin edges (start indices of each bin)
            bin_edges = np.arange(0, frag_range[1], bin_size)
            # Store bins as tuple for reuse
            frag_bins = (bin_edges, num_bins)
        depth, nc_signal = np.zeros(roi_length, dtype=np.float32), np.zeros(roi_length, dtype=np.float32)
        # process all fragments falling inside the site
        site_reads = bam.fetch(bed_tokens[0], start_pos, stop_pos)
        try:
            window_ref_sequence = ref_seq.fetch(bed_tokens[0], start_pos, stop_pos).upper()
        except Exception:
            # For non-composite modes, try to fetch with adjusted stop_pos if it extends beyond chromosome end
            try:
                # Try to get at least some reference sequence up to chromosome end
                chrom_len = ref_seq.get_reference_length(bed_tokens[0])
                adjusted_stop = min(stop_pos, chrom_len)
                if adjusted_stop <= start_pos + 500:  # Need at least 500bp total
                    skipped_msg = f"{region.strip()}\tinsufficient_reference"
                    return (site,) + (np.nan,) * 17 + ([],) + ([],) + ([skipped_msg],)
                # This gets truncated reference which is acceptable in non-composite mode
                window_ref_sequence = ref_seq.fetch(bed_tokens[0], start_pos, adjusted_stop).upper()
            except Exception:
                # If that still fails, skip this site
                skipped_msg = f"{region.strip()}\ttruncated_reference"
                return (site,) + (np.nan,) * 17 + ([],) + ([],) + ([skipped_msg],)
        # Process fragments directly into profiles - no intermediate lists
        process_fragments_fast(
            site_reads, start_pos, frag_range, map_q, gc_bias, window_ref_sequence,
            depth, nc_signal, fragment_lengths, fragment_length_profile,
            fragment_5end_profile, fragment_3end_profile, fdict, roi_length, flank_size
        )

        # Create reference sequence profile for non-composite mode
        if not stack:
            # Create one-hot encoded reference sequence - use window size (no flanks)
            window_width = roi_length - 2*flank_size
            ref_seq_segment = window_ref_sequence[flank_size:-flank_size] if len(window_ref_sequence) >= roi_length else window_ref_sequence
            if len(ref_seq_segment) >= window_width:
                oh_seq = np.zeros((window_width, 4), dtype=np.float32)
                for i, base in enumerate(ref_seq_segment[:window_width]):
                    if base.upper() in base_to_idx:
                        oh_seq[i, base_to_idx[base.upper()]] = 1.0
            else:
                oh_seq = np.zeros((window_width, 4), dtype=np.float32)
        else:
            oh_seq = None
        if strand == "-":
            # Flip all profile data for negative strand to ensure 5'->3' orientation
            depth = depth[::-1]
            nc_signal = nc_signal[::-1]
            fragment_length_profile = np.fliplr(fragment_length_profile)
            # Flip fragment end profiles
            fragment_5end_profile = fragment_5end_profile[::-1]
            fragment_3end_profile = fragment_3end_profile[::-1]
            # Also swap 5' and 3' profiles since their meaning is reversed on the negative strand
            fragment_5end_profile, fragment_3end_profile = fragment_3end_profile, fragment_5end_profile
            if not stack:  # Only flip sequence data in non-composite mode  
                oh_seq = oh_seq[::-1, :]  # Flip reference sequence for negative strand
    
    # generate phased profile and phasing/region-level features --------------------------------------------------------
    if np.count_nonzero(depth > 0) < 0.8 * roi_length:  # skip sites with less than 80% base coverage
        if not stack:  # For non-stacked mode, add the site to skipped_sites
            # Just include the original line plus the reason code
            skipped_sites = ['{}\tinsufficient_coverage'.format(region.strip())]
        return (site,) + (np.nan,) * 17 + ([],) + ([],) + (skipped_sites,)
    
    # In composite mode, use the absolute weighted-average mean depth (GC-corrected)
    # In single/window mode, compute from the window region
    if stack:
        mean_depth = composite_absolute_mean_depth
    else:
        mean_depth = np.mean(depth[flank_size:-flank_size])
    
    # Pre-normalize nc_signal by flanking mean before FFT (matching composite mode approach)
    # In composite mode, nc_signal is already per-site normalized, so this is skipped
    if not stack:
        # Use inner flanks (exclude the outermost ~max fragment length) to avoid ROI-edge truncation bias
        edge_exclude = min(int(frag_range[1]), int(flank_size - 1))
        if edge_exclude < 1:
            edge_exclude = 1
        left_nc = nc_signal[edge_exclude:flank_size]
        right_nc = nc_signal[-flank_size:-edge_exclude]
        flanking_nc_mean = (np.mean(left_nc) + np.mean(right_nc)) / 2
        if flanking_nc_mean > 0:
            nc_signal = nc_signal / flanking_nc_mean
        # If flanking_nc_mean is 0, leave nc_signal as is (will produce NaN in downstream calcs)
    
    fourier = rfft(nc_signal)
    freqs = rfftfreq(roi_length)
    range_1 = np.where((freqs >= low_1) & (freqs <= high_1))
    range_2 = np.where((freqs >= low_2) & (freqs <= high_2))
    primary_amp_1 = np.mean(np.abs(fourier[range_1])) / roi_length
    primary_amp_2 = np.mean(np.abs(fourier[range_2])) / roi_length
    # Apply log2 transformation to compaction score (ratio)
    raw_pn_score = primary_amp_1 / primary_amp_2 if primary_amp_1 > 0 and primary_amp_2 > 0 else np.nan
    pn_score = np.log2(raw_pn_score) if not np.isnan(raw_pn_score) else np.nan
    drop_freqs = np.where(freqs > freq_max)  # low-pass filter
    fourier[drop_freqs] = 0  # this fourier transform is now scrubbed of high frequencies
    phased_signal = irfft(fourier, n=roi_length)  # convert from filtered-freq space back to signal space
    
    # Calculate fragment end coverage and orientation asymmetry
    if stack:  # Composite mode - use pre-computed weighted ratios
        fragment_end_profile = fragment_end_coverage_profile
        orientation_asymmetry = orientation_asymmetry_profile
    else:  # Non-composite mode - calculate from raw counts
        # Sum fragment_length_profile across all fragment lengths to get total fragments per position
        fragment_counts_per_position = np.sum(fragment_length_profile, axis=0)
        fragment_counts_window = fragment_counts_per_position[flank_size:-flank_size]
        fragment_ends_5_window = fragment_5end_profile[flank_size:-flank_size]
        fragment_ends_3_window = fragment_3end_profile[flank_size:-flank_size]
        total_fragment_ends = fragment_ends_5_window + fragment_ends_3_window
        
        with np.errstate(divide='ignore', invalid='ignore'):
            fragment_end_profile = np.true_divide(total_fragment_ends, fragment_counts_window)
            # Set positions with no fragments to NaN
            fragment_end_profile[fragment_counts_window == 0] = np.nan
            # Set positions with fragments but no ends to 0
            fragment_end_profile[(fragment_counts_window > 0) & (total_fragment_ends == 0)] = 0.0

        # Calculate fragment end orientation asymmetry (also before normalization)
        with np.errstate(divide='ignore', invalid='ignore'):
            orientation_asymmetry = np.true_divide((fragment_ends_5_window - fragment_ends_3_window), total_fragment_ends)
            # Only positions with no ends should have NaN
            orientation_asymmetry[total_fragment_ends == 0] = np.nan

    # Apply flank normalization to depth and fragment end coverage
    # PN signal (nc_signal) is already pre-normalized before FFT in non-composite mode
    # In composite mode, signals are already per-site flank-normalized, so avoid double normalization
    if stack:
        flanking_depth_signal = 1.0
        flanking_fec = 1.0
    else:
        # Use inner flanks (exclude the outermost ~max fragment length) to avoid ROI-edge truncation bias
        edge_exclude = min(int(frag_range[1]), int(flank_size - 1))
        if edge_exclude < 1:
            edge_exclude = 1

        left_depth = depth[edge_exclude:flank_size]
        right_depth = depth[-flank_size:-edge_exclude]
        flanking_depth_signal = (np.mean(left_depth) + np.mean(right_depth)) / 2

        # Calculate flanking fragment end coverage for normalization using TRUE flanks (outside the window),
        # not the edges of the window. This avoids window-edge artifacts.
        total_fragment_ends_roi = fragment_5end_profile + fragment_3end_profile
        with np.errstate(divide='ignore', invalid='ignore'):
            epc_roi = np.true_divide(total_fragment_ends_roi, fragment_counts_per_position)
            epc_roi[fragment_counts_per_position == 0] = np.nan

        left_flank_fec = epc_roi[edge_exclude:flank_size]
        right_flank_fec = epc_roi[-flank_size:-edge_exclude]
        left_fec = np.nanmean(left_flank_fec)
        right_fec = np.nanmean(right_flank_fec)
        flanking_fec = (left_fec + right_fec) / 2 if (np.isfinite(left_fec) and np.isfinite(right_fec)) else 1.0
        if not np.isfinite(flanking_fec) or flanking_fec <= 0:
            flanking_fec = 1.0  # Fallback to no normalization if flanking FEC is invalid
    
    # Normalize fragment end coverage by flanking FEC (for both composite and non-composite modes)
    fragment_end_profile = fragment_end_profile / flanking_fec
    
    # Trim flanks and normalize depth
    depth = depth[flank_size:-flank_size] / flanking_depth_signal
    # Trim flanks from phased_signal (already normalized before FFT in non-composite mode)
    phased_signal = phased_signal[flank_size:-flank_size]
    window_depth = np.mean(phased_signal)
    center = int(len(phased_signal) / 2)

    # perform peak calling --------------------------------------------------------------------------------------------
    max_values, peaks, min_values, troughs = local_peaks(phased_signal)
    peak_profile = np.full((roi_length - 2*flank_size), np.nan, dtype=float)  # Use NaN as default
    if len(peaks) > 0:
        peak_profile[peaks] = 1.0
    if len(troughs) > 0:
        peak_profile[troughs] = -1.0
    if window is not None:
        central_depth = np.mean(phased_signal[(center - 5):(center + 6)])  # mean +/-5bp
    else:
        central_depth = np.nan
    if len(max_values) > 1 and len(peaks) > 1:
        pn_spacing = (peaks[-1] - peaks[0]) / (len(peaks) - 1)
        pn_amp = 0.5 * (np.mean(max_values) - np.mean(min_values)) / flanking_depth_signal
    else:
        pn_spacing, pn_amp = np.nan, np.nan

    # generate fragment profiles and fragmentation features ------------------------------------------------------------
    # In composite mode, fragment_lengths_pd is already a weighted probability distribution
    # In non-composite mode, convert raw counts to probability distribution
    if not stack:
        fragment_total = np.sum(fragment_lengths)
        if fragment_total > 0:
            # Convert to probability distribution
            fragment_lengths_pd = fragment_lengths / fragment_total
        else:
            # No fragments at all - return NaN for all features
            return (site,) + (np.nan,) * 17 + (np.full((8 if stack else 12, roi_length - 2*flank_size), np.nan),) + ([],) + ([] if not stack else skipped_sites,)
    else:
        # In composite mode, fragment_lengths is already a probability distribution
        fragment_lengths_pd = fragment_lengths
    
    # Ensure fragment_lengths_pd has no NaN or inf values before passing to frag_metrics
    fragment_lengths_pd = np.nan_to_num(fragment_lengths_pd, nan=0.0, posinf=0.0, neginf=0.0)
    
    fl_mean, fl_stdev, fl_skew, fl_kurt, fl_ratio_raw, fl_ent, fl_gsi = frag_metrics(fragment_lengths_pd, bins=frag_bins)
    # Apply log2 transformation to subnucleosomal ratio
    fl_ratio = np.log2(fl_ratio_raw) if not np.isnan(fl_ratio_raw) and fl_ratio_raw > 0 else np.nan
    # fragment profiles | 3 x len(window) where the 3 ordered are: ratio, entropy, gini-simpson
    # Create a partial function with bins pre-set for apply_along_axis
    frag_metrics_with_bins = partial(frag_metrics, reduce=True, bins=frag_bins)
    
    # For composite mode, fragment_length_profile is already probability distributions per position
    # For non-composite mode, convert raw counts to probability distributions per position
    if not stack:
        # Convert each position's fragment length counts to a probability distribution
        counts = fragment_length_profile
        pos_tot = counts.sum(axis=0, keepdims=True)
        # Positions with no fragments (pos_tot == 0) will remain all zeros
        with np.errstate(divide='ignore', invalid='ignore'):
            fragment_length_profile_pd = np.where(
                pos_tot > 0,
                counts / pos_tot,
                0.0  # Positions with no fragments get 0 probability
            )
    else:
        # In composite mode, fragment_length_profile is already probability distributions
        fragment_length_profile_pd = fragment_length_profile
    
    # Apply frag_metrics to each position (column) to get diversity metrics
    fl_signals_raw = np.apply_along_axis(frag_metrics_with_bins, axis=0, arr=fragment_length_profile_pd)
    # Apply log2 transformation to ratio signal (first row)
    fl_signals = fl_signals_raw.copy()
    ratio_signal = fl_signals_raw[0, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        fl_signals[0, :] = np.where((ratio_signal > 0) & np.isfinite(ratio_signal), 
                                   np.log2(ratio_signal), np.nan)
    
    # Compute robust z-scores for Shannon Entropy (row 1) and Gini-Simpson (row 2) signals
    # Use the median and MAD from the flanking entropy/GSI signals of the full ROI
    
    # Get flanking regions of the FL signals (donut normalization - exclude window, use only flanks)
    roi_width = fl_signals.shape[1]  # This is now the full ROI length including flanks
    if roi_width >= 2 * flank_size:
        # Use only the actual flanking regions, excluding the window (donut normalization)
        left_flank_entropy = fl_signals_raw[1, :flank_size]  # Left flank only
        right_flank_entropy = fl_signals_raw[1, -flank_size:]  # Right flank only
        left_flank_gsi = fl_signals_raw[2, :flank_size]  # Left flank only
        right_flank_gsi = fl_signals_raw[2, -flank_size:]  # Right flank only
    else:
        # For smaller ROIs, use all available data for flanking stats
        left_flank_entropy = fl_signals_raw[1, :]
        right_flank_entropy = np.array([])  # Empty array
        left_flank_gsi = fl_signals_raw[2, :]
        right_flank_gsi = np.array([])
    
    # Combine left and right flanking signals
    flank_entropy_values = np.concatenate([left_flank_entropy, right_flank_entropy])
    flank_gsi_values = np.concatenate([left_flank_gsi, right_flank_gsi])
    
    # Remove NaN and infinite values for robust statistics
    flank_entropy_clean = flank_entropy_values[np.isfinite(flank_entropy_values)]
    flank_gsi_clean = flank_gsi_values[np.isfinite(flank_gsi_values)]
    
    # Compute robust z-scores for entropy and GSI signals
    # Need at least 10 valid values for robust statistics to be meaningful
    if len(flank_entropy_clean) >= 10:
        entropy_median = np.median(flank_entropy_clean)
        entropy_mad = np.median(np.abs(flank_entropy_clean - entropy_median))
        if entropy_mad > 1e-10:  # Use small epsilon to avoid division by near-zero
            # Apply robust z-score transformation: (x - median) / (1.4826 * MAD)
            # Factor 1.4826 approximates std dev for normal distribution
            with np.errstate(divide='ignore', invalid='ignore'):
                fl_signals[1, :] = (fl_signals_raw[1, :] - entropy_median) / (1.4826 * entropy_mad)
        else:
            # If MAD is zero, all values are essentially the same - set to 0 where valid
            fl_signals[1, :] = np.where(np.isfinite(fl_signals_raw[1, :]), 0.0, np.nan)
    else:
        # Insufficient valid data for normalization - keep raw values but mask invalid
        fl_signals[1, :] = np.where(np.isfinite(fl_signals_raw[1, :]), fl_signals_raw[1, :], np.nan)
    
    if len(flank_gsi_clean) >= 10:
        gsi_median = np.median(flank_gsi_clean)
        gsi_mad = np.median(np.abs(flank_gsi_clean - gsi_median))
        if gsi_mad > 1e-10:  # Use small epsilon to avoid division by near-zero
            # Apply robust z-score transformation
            with np.errstate(divide='ignore', invalid='ignore'):
                fl_signals[2, :] = (fl_signals_raw[2, :] - gsi_median) / (1.4826 * gsi_mad)
        else:
            # If MAD is zero, all values are essentially the same - set to 0 where valid
            fl_signals[2, :] = np.where(np.isfinite(fl_signals_raw[2, :]), 0.0, np.nan)
    else:
        # Insufficient valid data for normalization - keep raw values but mask invalid
        fl_signals[2, :] = np.where(np.isfinite(fl_signals_raw[2, :]), fl_signals_raw[2, :], np.nan)
    # For window calculations, use only the window portion (trim flanks)
    window_entropy_data = fl_signals[1, flank_size:-flank_size]
    window_gsi_data = fl_signals[2, flank_size:-flank_size]
    
    # Return NaN if all values are NaN, otherwise use nanmean
    window_ent = np.nan if np.all(np.isnan(window_entropy_data)) else np.nanmean(window_entropy_data)
    window_gsi = np.nan if np.all(np.isnan(window_gsi_data)) else np.nanmean(window_gsi_data)
    
    if window is not None:
        # For central calculations, use center position relative to window (after trimming flanks)
        window_center = center  # center is already relative to trimmed signal
        central_entropy_data = window_entropy_data[(window_center - 5):(window_center + 6)]
        central_gsi_data = window_gsi_data[(window_center - 5):(window_center + 6)]
        
        # Return NaN if all values are NaN or slice is empty, otherwise use nanmean
        central_ent = np.nan if len(central_entropy_data) == 0 or np.all(np.isnan(central_entropy_data)) else np.nanmean(central_entropy_data)
        central_gsi = np.nan if len(central_gsi_data) == 0 or np.all(np.isnan(central_gsi_data)) else np.nanmean(central_gsi_data)
    else:
        central_ent, central_gsi = np.nan, np.nan
    # Prepare sequence profile
    if stack:  # composite-window mode - no sequence data
        seq_profile = None  # No sequence tracks in composite mode
    else:  # window/region mode - use reference sequence
        if oh_seq is not None and isinstance(oh_seq, np.ndarray) and oh_seq.shape[1] >= 4:
            # Transpose to get 4xN shape (4 rows for A,C,G,T)
            seq_profile = oh_seq[:, :4].T
        else:
            # Fallback if oh_seq isn't properly formatted - use window width (after flank trimming)
            window_width = roi_length - 2*flank_size
            seq_profile = np.zeros((4, window_width), dtype=np.float32)
    
    # combine and save profiles ----------------------------------------------------------------------------------------
    out_array = np.nan
    if window is not None and window <= 5000:  # do not print profiles longer than 5kb to avoid memory issues
        # Extract window portion of FL signals (trim flanks for output)
        fl_signals_window = fl_signals[:, flank_size:-flank_size]
        signal_array = np.vstack((depth, fragment_end_profile, orientation_asymmetry, phased_signal, fl_signals_window, peak_profile))
        if seq_profile is not None:
            out_array = np.concatenate((signal_array, seq_profile), axis=0)
        else:
            out_array = signal_array
        
    bam.close()
    ref_seq.close()

    return site, fl_mean, fl_stdev, fl_skew, fl_kurt, fl_ratio, fl_ent, fl_gsi, pn_score, pn_spacing, pn_amp,\
           mean_depth, central_depth, central_ent, central_gsi, window_depth, window_ent, window_gsi, out_array,\
           fragment_lengths_pd, skipped_sites


def main():
    """
    Takes in input parameters for a single sample Triton run, evaluates the site list format, manages breaking up
    sites into chunks to be run in parallel with generate_profile(), then combines outputs from each core and saves
    them to two files: sample_name_TritonFeatures.tsv and sample_name_TritonProfiles.npz containing region-level
    features and signal profiles, respectively. In composite mode, an additional file sample_name_SkippedSites.bed
    will be output with skipped sites if sites were skipped.
    """

    # parse command line arguments:
    parser = argparse.ArgumentParser(description='\n### Triton.py ### main Triton pipeline')
    parser.add_argument('-n', '--sample_name', help='sample identifier', required=True)
    parser.add_argument('-i', '--input', help='input BAM file', required=True)
    parser.add_argument('-b', '--bias', help='input-matched GC bias file (Griffin formatted; optional)', required=False, default=None)
    parser.add_argument('-a', '--annotation', help='regions of interest as a BED or list of BEDs', required=True)
    parser.add_argument('-g', '--reference_genome', help='reference genome (.fa)', required=True)
    parser.add_argument('-r', '--results_dir', help='directory for results', required=True)
    parser.add_argument('-m', '--run_mode', help='run mode ("region", "window", or "composite-window")', type=str, required=True)
    parser.add_argument('-q', '--map_quality', help='minimum mapping quality; default=20', type=int, default=20)
    parser.add_argument('-f', '--size_range', help='fragment size range (bp); default/bounds=15 500', nargs=2, type=int, default=(15, 500))
    parser.add_argument('-c', '--cpus', help='number of CPUs to use for parallel processing; default=all', type=int, required=False)
    parser.add_argument('-d', '--frag_dict', help='dictionary of probable nucleosome center-frag displacements; default=nc_fitting/NCDict.pkl', required=False, default='nc_fitting/NCDict.pkl')
    args = parser.parse_args()

    print('Loading input files . . .')
    sample_name, bam_path, bias_path, sites_path, ref_seq_path, results_dir, map_q, size_range, cpus, run_mode, dict_path = \
        args.sample_name, args.input, args.bias, args.annotation, args.reference_genome, args.results_dir, args.map_quality, args.size_range, args.cpus, args.run_mode, args.frag_dict

    if cpus is None: # If no argument was passed to cpus, use all available CPUs
        cpus = multiprocessing.cpu_count()

    print(f'\n### arguments provided:\n'
          f'\tsample name = "{sample_name}"\n'
          f'\tinput bam path = "{bam_path}"\n'
          f'\tGC bias dict path = "{bias_path}"\n'
          f'\treference genome path = "{ref_seq_path}"\n'
          f'\tsite list = "{sites_path}"\n'
          f'\tresults directory = "{results_dir}"\n'
          f'\tfragment size range = {size_range}\n'
          f'\tminimum mapping quality = {map_q}\n'
          f'\tCPUs to use = {cpus}\n'
          f'\trun mode = {run_mode}\n'
          f'\tfragment re-weighting dictionary = {dict_path}\n')
    sys.stdout.flush()

    # Basic input validation for clearer failures up front.
    if not os.path.exists(bam_path):
        raise FileNotFoundError(f'Input alignment file not found: {bam_path}')
    if not os.path.exists(ref_seq_path):
        raise FileNotFoundError(f'Reference genome FASTA not found: {ref_seq_path}')
    if not os.path.exists(sites_path):
        raise FileNotFoundError(f'Annotation file not found: {sites_path}')
    if not os.path.exists(dict_path):
        raise FileNotFoundError(f'Fragment center dictionary not found: {dict_path}')
    if map_q < 0:
        raise ValueError(f'Invalid map_quality ({map_q}): must be >= 0')
    if len(size_range) != 2 or size_range[0] < 1 or size_range[1] < size_range[0]:
        raise ValueError(
            f'Invalid size_range ({size_range}): expected two integers with 1 <= min <= max'
        )
    if cpus is not None and cpus < 1:
        raise ValueError(f'Invalid cpus ({cpus}): must be >= 1')

    if bias_path is not None:
        if not os.path.exists(bias_path):
            raise FileNotFoundError(f'GC bias file not found: {bias_path}')
        gc_bias = get_gc_bias_dict(bias_path)
    else:
        print('WARNING: No GC bias file provided. Running without GC bias correction (all bias factors set to 1).')
        gc_bias = None
    sites = [region for region in open(sites_path, 'r')]
    frag_dict = pickle.load(open(dict_path, 'rb'))
    if run_mode in ['region', 'window']:
        header = sites.pop(0).strip().split('\t')
    elif run_mode == 'composite-window':
        with open(sites[0].strip(), 'r') as f:
            header = f.readline().strip().split('\t')
            print(header)
    else:
        raise ValueError(f"Invalid run_mode ({run_mode}): must be one of 'region', 'window', or 'composite-window'")
       
    chr_idx = get_index(header, ['chrom', 'Chrom'], 0, 'No "chrom" column found in BED file(s): defaulting to index {} ({})')
    start_idx = get_index(header, ['chromStart', 'start', 'Start'], 1, 'No "chromStart" column found in BED file(s): defaulting to index {} ({})')
    stop_idx = get_index(header, ['chromEnd', 'end', 'End'], 2, 'No "chromEnd" column found in BED file(s): defaulting to index {} ({})')
    if run_mode in ['region', 'window']:
        site_idx = get_index(header, ['name', 'Gene'], 3, 'No "name" column found in BED file(s): defaulting to index {} ({})')
        strand_idx = get_index(header, ['strand', 'Strand'], None, 'WARNING: No "strand" column found in BED file(s): defaulting to "+" for all sites')
    else:
        strand_idx = get_index(header, ['strand', 'Strand'], None, 'WARNING: No "strand" column found in BED file(s): both strands will be averaged for each site')
        site_idx = None
    pos_idx = get_index(header, ['position'], None, 'No "position" column found in BED file(s): will compute midpoint from start+end') if run_mode != 'region' else None

    params = [bam_path, size_range, gc_bias, ref_seq_path, map_q, frag_dict, run_mode,
              chr_idx, start_idx, stop_idx, site_idx, strand_idx, pos_idx]

    print('\n### Running Triton on {} region sets ###\n'.format(len(sites)))

    if len(sites) < cpus:
        print('More CPUs passed than sites - scaling down to {}'.format(len(sites)))
        cpus = len(sites)

    # Use parallel processing by default for production runs with pool initializer
    print(f'Processing {len(sites)} sites in parallel using {cpus} CPUs...')
    with multiprocessing.Pool(processes=cpus, initializer=pool_initializer, 
                             initargs=(gc_bias, frag_dict)) as pool:
        # Optimize chunk size based on number of sites and CPUs: larger chunks reduce overhead but may cause load imbalance
        chunksize = max(50, len(sites)//(cpus*5))
        print(f'Using chunk size of {chunksize} sites per worker')
    
        # Track progress
        completed = 0
        results = []
        
        # Use imap to process sites with progress tracking
        for result in pool.imap(partial(generate_profile, params=params), sites, chunksize=chunksize):
            results.append(result)
            completed += 1
            if completed % 100 == 0 or completed == len(sites):
                print(f'Processed {completed}/{len(sites)} sites ({(completed/len(sites)*100):.1f}%)')
    
    # Sequential version - uncomment for debugging
    # results = []
    # for i, site in enumerate(sites):
    #     results += [generate_profile(site, params)]
    #     if (i+1) % 100 == 0:
    #         print(f'Processed {i+1}/{len(sites)} sites ({(i+1)/len(sites)*100:.1f}%)')
    
    print('All sites processed. Combining results...')

    print('Merging and saving results . . .')
    signal_dict, fl_dict = {}, {}
    all_skipped_sites = []
    
    # Import pandas here if not already imported
    import pandas as pd

    # ORDERED output feature keys
    keys = ['fl-mean',
            'fl-stdev',
            'fl-skew',
            'fl-kurtosis',
            'fl-subnucleosomal-ratio', 
            'fl-entropy',
            'fl-gini-simpson',
            'pn-compaction-score',
            'pn-mean-spacing',
            'pn-mean-amplitude', 
            'mean-region-depth',
            'central-depth',
            'central-entropy',
            'central-gini-simpson',
            'window-depth', 
            'window-entropy',
            'window-gini-simpson',]

    # Prepare data for wide-form DataFrame (sites as rows, features as columns)
    sites_data = []
    
    for result in results:
        """ result[0] = site
            result[1-17] = features in order of 'keys' list above
            result[18] = numpy array of signal profiles (or np.nan if window > 5000bp)
            result[19] = fragment length probability distribution (1-500bp)
            result[20] = skipped sites list (composite-window mode only; empty list if none skipped) """
            
        # Create a row for this site with all its features
        site_row = {'site': result[0], 'sample': sample_name}
        for i, key in enumerate(keys):
            site_row[key] = result[i+1]
        
        sites_data.append(site_row)
        
        # Store signal profiles and fragment length probability distributions separately
        signal_dict[result[0]] = result[18]
        fl_dict[result[0]] = result[19]
        
        if len(result) == 21 and result[20]:  # skipped sites list exists and is not empty
            # Debug print to see what the skipped sites look like
            for line in result[20]:
                # Split the line to get components
                parts = line.split('\t')
                if len(parts) >= 1:  # Ensure there's at least one part
                    all_skipped_sites.append(line)

    # Create a DataFrame from fragment length probability distributions for the additional output
    fl_df = pd.DataFrame.from_dict(fl_dict, orient='index')
    fl_df.columns = [f"frag_len_{i+1}" for i in range(fl_df.shape[1])]  # Create column names like frag_len_1, frag_len_2, etc.
    fl_df = fl_df.astype(np.float32)  # Convert to float32 to save space
    
    # Ensure the index (site names) is explicitly preserved as a column for downstream access
    fl_df.reset_index(inplace=True)
    fl_df.rename(columns={'index': 'site'}, inplace=True)

    # Create sample-specific subdirectory
    sample_dir = os.path.join(results_dir, sample_name)
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    # First create wide-form dataframe (sites as rows, features as columns)
    df_wide = pd.DataFrame(sites_data)
    
    # Convert numerical columns to float32 to save space
    numeric_columns = df_wide.select_dtypes(include=[np.number]).columns
    df_wide[numeric_columns] = df_wide[numeric_columns].astype(np.float32)
    
    # Save files to sample subdirectory
    out_file = os.path.join(sample_dir, sample_name + '_TritonFeatures.tsv')
    signal_file = os.path.join(sample_dir, sample_name + '_TritonSignalProfiles.npz')
    frag_file = os.path.join(sample_dir, sample_name + '_TritonFragmentationProfiles.npz')
    
    # Save output in wide-form format (features as columns)
    df_wide.to_csv(out_file, sep='\t', index=False)
    
    # Convert any float64 arrays to float32 in the signal dictionary to save space
    signal_dict_float32 = {}
    for key, value in signal_dict.items():
        if isinstance(value, np.ndarray) and value.dtype == np.float64:
            signal_dict_float32[key] = value.astype(np.float32)
        else:
            signal_dict_float32[key] = value
    np.savez_compressed(signal_file, **signal_dict_float32)
    
    # Save the fragment length histogram dataframe as compressed numpy archive
    # Save both the data and column names to preserve DataFrame structure
    np.savez_compressed(frag_file, 
                       fragment_data=fl_df.values, 
                       fragment_columns=fl_df.columns.values)

    # Always create skipped sites report, regardless of run mode
    if all_skipped_sites:  # If there are any skipped sites
            # Summarize reasons for skipped sites
            reason_counts = {}
            for entry in all_skipped_sites:
                parts = entry.split('\t')
                if len(parts) >= 2:
                    reason = parts[-1]  # Last element is always the reason
                    if reason not in reason_counts:
                        reason_counts[reason] = 0
                    reason_counts[reason] += 1
            
            # Print summary of skipped sites by reason
            print("Summary of skipped sites by reason:")
            for reason, count in reason_counts.items():
                print(f"  {reason}: {count}")
            
            # Create a text file with skipped sites information
            skipped_file = os.path.join(sample_dir, sample_name + '_SkippedSites.txt')
            
            # Format the output differently based on run mode
            if run_mode in ['region', 'window']:
                # For non-composite mode, use original header plus skip_reason column
                # First get a copy of the original header
                header_copy = header.copy()
                header_copy.append("skip_reason")  # Add the skip_reason column
                
                with open(skipped_file, 'w') as f:
                    # Write the header (original BED columns + skip_reason)
                    f.write('\t'.join(header_copy) + '\n')
                    
                    # For each skipped site entry
                    for entry in all_skipped_sites:
                        parts = entry.split('\t')
                        if len(parts) >= 2:  # Ensure we have at least a region and reason
                            # The last part is the reason code
                            reason = parts[-1]
                            # Get the original region info
                            region_parts = parts[:-1]  # Everything except the reason
                            # Write out the original region info plus the reason code
                            f.write('\t'.join(region_parts) + '\t' + reason + '\n')
            else:
                # For composite mode, use a format that matches the region/window mode format
                # First get the header from the first site file mentioned in the composite list
                with open(sites[0].strip(), 'r') as f:
                    first_header = f.readline().strip().split('\t')
                
                # Make a copy and add site and skip_reason columns
                header_copy = first_header.copy()
                if "site" not in header_copy:
                    header_copy.append("site")
                header_copy.append("skip_reason")
                
                with open(skipped_file, 'w') as f:
                    # For composite mode, we're now using a format where site name is embedded in the entry
                    # So the header should reflect that properly
                    f.write('\t'.join(header_copy) + '\n')
                    
                    # For each skipped site entry
                    for entry in all_skipped_sites:
                        parts = entry.split('\t')
                        if len(parts) >= 2:  # Ensure we have at least a region and reason
                            # Extract region parts and reason
                            reason = parts[-1]  # Last part is reason
                            region_parts = parts[:-1]  # All but reason
                            
                            # In composite mode, site name is already included in the message
                            # Format is: original_entry \t site_name \t reason
                            if len(parts) >= 3:
                                site_name = parts[-2]  # Site name is second-to-last element
                            else:
                                # Fallback: Extract site name from the appropriate file
                                for site_entry in sites:
                                    if site_entry.strip() in entry:
                                        site_name = os.path.basename(site_entry.strip()).split('.')[0]
                                        break
                                else:
                                    site_name = "unknown"
                                
                            # Format output line with region parts, site name, and reason
                            # If region parts has fewer columns than header, pad with empty strings
                            while len(region_parts) < len(first_header):
                                region_parts.append("")
                            
                            # Add site column if it's not already present
                            if len(region_parts) == len(first_header):
                                region_parts.append(site_name)
                                
                            # Write out the line with reason appended
                            f.write('\t'.join(region_parts) + '\t' + reason + '\n')
            
            print(f'Skipped sites details written to: {skipped_file}')
            # print(f'Total skipped sites: {len(all_skipped_sites)} ({len(all_skipped_sites)/len(sites)*100:.1f}% of input)')
            print(f'\nFinished! Results saved to: {sample_dir}')

if __name__ == "__main__":
    main()
