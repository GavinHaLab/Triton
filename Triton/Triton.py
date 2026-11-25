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
# likelihood of GC content for each IUPAC bp code x6 (common factor), indexed by ASCII code; 255 marks other characters
gc_score_table = np.full(256, 255, dtype=np.uint8)
gc_score_table[np.frombuffer(b'ACGTURYSWKMBDHVNX', dtype=np.uint8)] = [int(score) for score in '06600336033422430']
freq_max = 1.0 / 147.0  # theoretical minimum nucleosome period is 147 bp -> f = 0.006803
low_1, high_1 = 0.00565, 0.00680  # range_1: T = 147-177 bp (f = 0.00565 - 0.00680, "small linker"/compact chromatin)
low_2, high_2 = 0.00483, 0.00565  # range_2: T = 177-207 bp (f = 0.00483 - 0.00565, "large linker"/open chromatin)

# Per-worker state, set once by the pool initializer
global_bam = None
global_ref_seq = None
global_contig_lengths = None
global_gc_bias = None
global_fdict = None
mitochondrial_contigs = {'chrM', 'chrMT', 'M', 'MT'}

def pool_initializer(bam_path, ref_seq_path, gc_bias, fdict):
    """
    Pool initializer: each worker opens the alignment and reference files once (so their indices are
    loaded once per worker rather than once per site) and holds the GC bias and fragment dictionaries.
    """
    global global_bam, global_ref_seq, global_contig_lengths, global_gc_bias, global_fdict
    # CRAM files require reference for proper reading
    is_cram = bam_path.lower().endswith('.cram')
    global_bam = pysam.AlignmentFile(bam_path, 'rb', reference_filename=ref_seq_path if is_cram else None)
    global_ref_seq = pysam.FastaFile(ref_seq_path)
    # Contigs a site can be profiled on: present in both the reference and the alignment header
    global_contig_lengths = {name: length for name, length in zip(global_ref_seq.references, global_ref_seq.lengths)
                             if global_bam.get_tid(name) >= 0}
    global_gc_bias = gc_bias
    global_fdict = fdict


def site_skip_reason(tokens, required_columns, coordinate_columns):
    """Reason a site line cannot be parsed (too few columns or non-integer coordinates), or None."""
    if len(tokens) <= max(required_columns) or not all(tokens[index].isdigit() for index in coordinate_columns):
        return 'malformed_line'
    return None


def roi_skip_reason(chrom, start_pos, stop_pos):
    """
    Reason a region of interest (site plus full flanks) cannot be profiled, or None. A region reaching past either
    end of its contig is skipped rather than truncated: every profile is indexed from the region start, so a
    shortened flank would shift the site within its window and bias the flank normalization.
    """
    if chrom in mitochondrial_contigs:
        return 'MT_contig'
    if chrom not in global_contig_lengths:
        return 'unknown_contig'
    if start_pos < 0:
        return 'insufficient_left_flank'
    if stop_pos > global_contig_lengths[chrom]:
        return 'insufficient_right_flank'
    return None


def skipped_site_result(site, region, reason):
    """generate_profile return value for a window/region site that is skipped: no features, the site line and reason."""
    return (site,) + (np.nan,) * 15 + ([],) + ([],) + ([f'{region.strip()}\t{reason}'],)


def new_fragment_profiles(num_bins, roi_length):
    """
    Per-position accumulators filled by process_fragments_fast (window and composite-window modes only):
        length: (fragment length bin x position) GC-corrected depth, for the Gini-Simpson profile
        weight_sq: per-position sum of squared fragment weights, for the effective fragment count
        short / long: GC-corrected depth from fragments < 147 bp / >= 147 bp
        end5 / end3: GC-corrected 5' (leftmost) / 3' (rightmost) fragment end counts
    """
    profiles = {name: np.zeros(roi_length, dtype=np.float32) for name in ('weight_sq', 'short', 'long', 'end5', 'end3')}
    profiles['length'] = np.zeros((num_bins, roi_length), dtype=np.float32)
    return profiles


def mirror_fragment_profiles(profiles):
    """Reverse-strand view of fragment profiles: positions are reversed and the 5' and 3' ends exchange roles."""
    mirrored = {name: profile[..., ::-1] for name, profile in profiles.items()}
    mirrored['end5'], mirrored['end3'] = mirrored['end3'], mirrored['end5']
    return mirrored


def inner_flank_mean(signal, edge_exclude):
    """Mean of both flanks, excluding the outermost edge_exclude positions (ROI-boundary truncation bias)."""
    return (np.mean(signal[edge_exclude:flank_size]) + np.mean(signal[-flank_size:-edge_exclude])) / 2


def process_fragments_fast(site_reads, start_pos, frag_range, map_q, gc_bias, window_ref_sequence,
                          depth, nc_signal, fragment_lengths, fdict, roi_length, flank_size,
                          profiles=None, bin_size=5):
    """
    Fast fragment processing using one-mate rule and TLEN, updating profiles in place.
    No intermediate lists - direct profile updates for maximum efficiency.
    profiles (see new_fragment_profiles) is None in region mode, where no per-position fragment profiles are needed.
    Returns the raw fragment counts (window-centered, ROI-contained) and the sum of squared weights of the
    window-centered fragments (with their summed weights, this gives the effective fragment count).
    """
    min_frag_size = frag_range[0]
    max_frag_size = frag_range[1]
    has_gc_bias = gc_bias is not None
    if has_gc_bias:
        # Cumulative GC score along the ROI: a fragment's GC content is then one subtraction instead of a per-base sum
        gc_scores = gc_score_table[np.frombuffer(window_ref_sequence.encode('ascii'), dtype=np.uint8)]
        if (gc_scores == 255).any():
            raise ValueError('Reference sequence contains characters outside the IUPAC nucleotide codes')
        gc_cumulative = [0] + np.cumsum(gc_scores, dtype=np.int64).tolist()

    # Raw fragment counters (pre-GC correction):
    # - window: fragment center falls within the window (excluding flanks)
    # - roi: fragment fully contained within the ROI (window + flanks)
    n_frag_raw_window = 0
    n_frag_raw_roi = 0
    weight_sq_window = 0.0
    
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
            fragment_gc_content = round((gc_cumulative[frag_end] - gc_cumulative[frag_start]) / 6)
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
            weight_sq_window += bias_factor * bias_factor
        
        # Get nucleosome density profile if applicable
        nc_density = fdict[frag_len] if 147 <= frag_len <= 500 else None
        
        # Vectorized fragment spanning - update depth and nc_signal profiles in place
        if 0 <= frag_start < roi_length and 0 <= frag_end <= roi_length:
            n_frag_raw_roi += 1
            depth[frag_start:frag_end] += bias_factor
            if nc_density is not None:
                nc_slice = nc_density[:frag_len] * bias_factor
                nc_signal[frag_start:frag_end] += nc_slice

            if profiles is not None:
                # Full ROI (including flanks), so the flanks can serve as the local reference level
                profiles['length'][(frag_len - 1) // bin_size, frag_start:frag_end] += bias_factor
                profiles['weight_sq'][frag_start:frag_end] += bias_factor * bias_factor
                profiles['short' if frag_len < 147 else 'long'][frag_start:frag_end] += bias_factor

        # Fragment ends - 5' end is always leftmost, 3' end is rightmost
        if profiles is not None:
            if 0 <= frag_start < roi_length:
                profiles['end5'][frag_start] += bias_factor
            if 0 <= frag_end - 1 < roi_length:
                profiles['end3'][frag_end - 1] += bias_factor
    return n_frag_raw_window, n_frag_raw_roi, weight_sq_window


def generate_profile(region, params):
    """
    Generates single or composite signal profiles and extracts features (fragmentation, nucleosome phasing,
    and profile shape-based) for a single sample. Utilizes functions from triton_helpers.py.
        Parameters:
            region (string): either a file path pointing to a BED-like file (composite) or a BED-like single annotation
            params (list): frag_range, map_q, run_mode, chr_idx, start_idx, stop_idx, site_idx, strand_idx, pos_idx
                (the alignment file, reference, GC bias and fragment dictionary are per-worker state; see pool_initializer)
        Returns:
            site: annotation name if stacked, "name" from BED file for each region otherwise
                ### Fragmentation (FL) Features (using all fragments in passed length range) ###
            fl-mean: FL distribution mean (absolute, in bp)
            fl-stdev: FL distribution standard deviation (absolute, in bp)
            fl-skew: FL distribution skewness (absolute)
            fl-kurtosis: FL distribution kurtosis (absolute)
            fl-subnucleosomal-ratio: FL subnucleosomal (x < 147 / x >= 147) ratio (log2)
            fl-entropy: FL distribution Normalized Shannon Entropy / Pielou's Evenness (absolute; Miller-Madow bias-corrected in region/window modes)
            fl-gini-simpson: FL distribution Gini-Simpson index (absolute; unbiased estimator in region/window modes)
                ### Phased-Nucleosome (PN) Features (FFT-based, using >= 147bp fragments) ###
            pn-compaction-score: ratio of compact (small-linker, 147-177bp) to open (large-linker, 177-207bp) PN signal amplitude (log2)
            pn-mean-spacing: mean spacing between PN peaks (absolute, in bp)
            pn-mean-amplitude: mean amplitude (half peak-to-trough of the flank-normalized PN signal) of PN peaks
                ### Profiling Features (Fl or PN signal-based) ###
            mean-region-depth: mean depth in the region (GC-corrected, normalized in composite, raw in region/window)
            central-depth*: PN signal profile value at window center (mean +/-5 bp, normalized by mean flanking PN signal)
            central-gini-simpson*: fragmentation Gini-Simpson index signal at window center (mean +/-5 bp, difference from the median flanking FL Gini-Simpson)
            window-depth: PN signal profile mean in window (normalized by mean flanking PN signal)
            window-gini-simpson*: fragmentation Gini-Simpson signal mean in window (difference from the median flanking FL Gini-Simpson)
                ### Region-level signal profiles (nt-resolution) ###
            numpy array: shape 8xL (composite) and 12xL (window with sequence) containing:
                1: Depth (GC-corrected, normalized by mean flanking depth)
                2: 5' fragment end density (GC-corrected, normalized by mean flanking 5' end density)
                3: 3' fragment end density (GC-corrected, normalized by mean flanking 3' end density)
                4: PN profile (normalized by mean flanking signal in NC signal, pre-FFT)
                5: Subnucleosomal depth (fragments < 147 bp, GC-corrected, normalized by mean flanking subnucleosomal depth)
                6: Nucleosomal depth (fragments >= 147 bp, GC-corrected, normalized by mean flanking nucleosomal depth)
                7: FL Gini-Simpson index (unbiased estimator in window mode; difference from the median flanking index)
                8: PN peak locations (-1: trough, 1: peak)
                9: A (Adenine)**
                10: C (Cytosine)**
                11: G (Guanine)**
                12: T (Thymine)**
            skipped-sites: list of annotation lines skipped, each with a reason: malformed_line, unrecognized_strand, MT_contig,
                unknown_contig, insufficient_left_flank / insufficient_right_flank (region plus flanks reaches past a contig end),
                insufficient_coverage, and in composite mode zero_coverage, MAD_outlier, zero_flank_coverage, zero_fragments
            * these features are centered region specific, and output as np.nan if window == None
            ** nt signals represent the one-hot encoded reference sequence of the site's own strand read 5'->3' (the reverse
            complement of the reference for - strand sites, matching the orientation of the signals), reported in window mode only.
            N.B. "flanking" regions represent the outer +/-flank_size (default +/-5000 bp) surrounding the window or region,
            and are used for smooth signal production at the edges of the window/region and for normalization. Sites with flanking regions that
            reach past a contig end are skipped.
            
            ### Composite Mode Normalization ###
            In composite-window mode, all signals undergo per-site normalization and weighting before aggregation:
            1. Flank normalization (stable): Composite depth, subnucleosomal/nucleosomal depth and nc_signal are normalized to flanks using a ratio-of-sums estimator
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
    frag_range, map_q, run_mode, chr_idx, start_idx, stop_idx, site_idx, strand_idx, pos_idx = params
    bam, ref_seq, gc_bias, fdict = global_bam, global_ref_seq, global_gc_bias, global_fdict
    
    if run_mode == 'composite-window':
        window, stack = window_size, True
    elif run_mode == 'window':
        window, stack = window_size, False
    else:
        window, stack = None, False
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
        # Subnucleosomal/nucleosomal depth and 5'/3' end densities use the same ratio-of-sums estimator
        coverage_tracks = ('short', 'long', 'end5', 'end3')
        coverage_num = {name: np.zeros(roi_length, dtype=np.float32) for name in coverage_tracks}
        coverage_den = dict.fromkeys(coverage_tracks, 0.0)
        fragment_length_profile = np.zeros((num_bins, roi_length), dtype=np.float32)  # nt-level binned fragment length distributions across full ROI
        
        # Initialize cumulative weight for per-site normalization
        total_weight = 0.0
        
        # Initialize accumulator for absolute mean depth (GC-corrected, before normalization)
        absolute_mean_depth_sum = 0.0
        
        with open(region.strip(), 'r') as sites_file:
            for line_number, entry in enumerate(sites_file):  # iterate through regions in this particular BED file
                if not entry.strip():
                    continue
                bed_tokens = entry.strip().split('\t')
                coordinate_columns = [start_idx, stop_idx] if pos_idx is None else [start_idx, stop_idx, pos_idx]
                required_columns = [chr_idx] + coordinate_columns + ([] if strand_idx is None else [strand_idx])
                reason = site_skip_reason(bed_tokens, required_columns, coordinate_columns)
                if reason is not None:
                    if line_number > 0:  # the first line may be a header
                        skipped_sites.append(entry.strip() + '\t' + site + '\t' + reason)
                    continue
                chrom = str(bed_tokens[chr_idx])
                # Use position column if available, otherwise compute midpoint
                if pos_idx is not None:
                    center_pos = int(bed_tokens[pos_idx])
                else:
                    center_pos = (int(bed_tokens[start_idx]) + int(bed_tokens[stop_idx])) // 2
                # If strand context is unknown, assuming a strand would imply a directionality which is not
                # necessarily present; both orientations are instead averaged with 1/2 weight each
                strand = None
                if strand_idx is not None:
                    try:
                        strand = parse_strand(bed_tokens[strand_idx])
                    except ValueError:
                        skipped_sites.append(entry.strip() + '\t' + site + '\tunrecognized_strand')
                        continue
                strands_to_process = ['+', '-'] if strand is None else [strand]
                # Full flanks are required on both sides
                start_pos = center_pos - int(roi_length / 2)
                desired_stop_pos = center_pos + int(roi_length / 2)
                reason = roi_skip_reason(chrom, start_pos, desired_stop_pos)
                if reason is not None:
                    skipped_sites.append(entry.strip() + '\t' + site + '\t' + reason)
                    continue
                # Process all fragments falling inside the ROI
                site_reads = bam.fetch(chrom, start_pos, desired_stop_pos)
                window_ref_sequence = ref_seq.fetch(chrom, start_pos, desired_stop_pos).upper()
                # Initialize temporary arrays for this site
                temp_depth = np.zeros(roi_length, dtype=np.float32)
                temp_nc_signal = np.zeros(roi_length, dtype=np.float32)
                site_fragment_lengths = np.zeros(frag_range[1], dtype=np.float32)  # strand-independent
                site_profiles = new_fragment_profiles(num_bins, roi_length)
                
                # Process fragments directly into temporary arrays
                n_frag_raw_window, n_frag_raw_roi, _ = process_fragments_fast(
                    site_reads, start_pos, frag_range, map_q, gc_bias, window_ref_sequence,
                    temp_depth, temp_nc_signal, site_fragment_lengths, fdict, roi_length, flank_size,
                    profiles=site_profiles, bin_size=bin_size
                )
                
                # Now apply strand-specific processing
                if strands_to_process == ['+']:
                    site_depth, site_nc_signal = temp_depth, temp_nc_signal
                elif len(strands_to_process) == 1:  # negative strand
                    site_depth, site_nc_signal = temp_depth[::-1], temp_nc_signal[::-1]
                    site_profiles = mirror_fragment_profiles(site_profiles)
                else:
                    # Both strands - average the two orientations with equal weight so the composite is symmetric
                    # (the mirrored 5' ends are the 3' ends, so the two end densities become mirror images)
                    site_depth = (temp_depth + temp_depth[::-1]) * 0.5
                    site_nc_signal = (temp_nc_signal + temp_nc_signal[::-1]) * 0.5
                    mirrored_profiles = mirror_fragment_profiles(site_profiles)
                    site_profiles = {name: (profile + mirrored_profiles[name]) * 0.5 for name, profile in site_profiles.items()}
                site_fragment_length_profile = site_profiles['length']
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
                
                # E) Weight the normalized signals and probability distributions, then accumulate
                # Ratio-of-sums flank normalization
                depth_num += site_depth * site_weight
                nc_num += site_nc_signal * site_weight
                depth_den += site_flank_depth_mean * site_weight
                nc_den += site_flank_nc_mean * site_weight
                for name in coverage_tracks:
                    coverage_num[name] += site_profiles[name] * site_weight
                    coverage_den[name] += inner_flank_mean(site_profiles[name], edge_exclude) * site_weight
                fragment_lengths += site_fragment_lengths_prob * site_weight  # Weight probability distribution by sqrt(n)
                fragment_length_profile += site_fragment_length_profile_prob * site_weight
                # Accumulate total weight for final normalization
                total_weight += site_weight
        
        # E) Final normalization by cumulative weight W after all sites processed
        if total_weight > 0 and depth_den > 0 and nc_den > 0:
            # Finalize ratio-of-sums normalized composite signals
            depth = depth_num / depth_den
            nc_signal = nc_num / nc_den
            # A composite with no flanking signal for a track has no defined level for it
            coverage_profiles = {name: coverage_num[name] / coverage_den[name] if coverage_den[name] > 0
                                 else np.full(roi_length, np.nan, dtype=np.float32) for name in coverage_tracks}
            fragment_lengths /= total_weight  # Normalize weighted probability distribution
            fragment_length_profile /= total_weight
            
            # Calculate absolute mean depth for composite mode (GC-corrected, weighted average across sites)
            composite_absolute_mean_depth = absolute_mean_depth_sum / total_weight
        else:
            # If no sites contributed (shouldn't happen but safety check)
            return (site,) + (np.nan,) * 15 + ([],) + ([],) + (skipped_sites,)
    else:
    # assemble depth and fragment profiles for sites individually -------------------------------------------------------
        bed_tokens = region.strip().split('\t')
        coordinate_columns = [start_idx, stop_idx] if pos_idx is None else [start_idx, stop_idx, pos_idx]
        required_columns = [chr_idx, site_idx] + coordinate_columns + ([] if strand_idx is None else [strand_idx])
        reason = site_skip_reason(bed_tokens, required_columns, coordinate_columns)
        if reason is not None:
            return skipped_site_result(bed_tokens[site_idx] if len(bed_tokens) > site_idx else region.strip(), region, reason)
        chrom, site = str(bed_tokens[chr_idx]), str(bed_tokens[site_idx])
        # Without a strand column, or for unstranded sites, the + strand is assumed
        strand = '+'
        if strand_idx is not None:
            try:
                strand = parse_strand(bed_tokens[strand_idx]) or '+'
            except ValueError:
                return skipped_site_result(site, region, 'unrecognized_strand')
        if window is not None:
            roi_length = window + 2*flank_size
            # Use position column if available, otherwise compute midpoint
            if pos_idx is not None:
                center_pos = int(bed_tokens[pos_idx])
            else:
                center_pos = (int(bed_tokens[start_idx]) + int(bed_tokens[stop_idx])) // 2
            start_pos = center_pos - int(roi_length / 2)
            stop_pos = center_pos + int(roi_length / 2)
            fragment_profiles = new_fragment_profiles(num_bins, roi_length)
        else:
            start_pos = int(bed_tokens[start_idx]) - flank_size
            stop_pos = int(bed_tokens[stop_idx]) + flank_size
            roi_length = stop_pos - start_pos
            # Region mode reports no per-position profiles, so the (length bin x position) accumulators,
            # whose size grows with the region, are not built
            fragment_profiles = None
        # Full flanks are required on both sides
        reason = roi_skip_reason(chrom, start_pos, stop_pos)
        if reason is not None:
            return skipped_site_result(site, region, reason)
        depth, nc_signal = np.zeros(roi_length, dtype=np.float32), np.zeros(roi_length, dtype=np.float32)
        # process all fragments falling inside the site
        site_reads = bam.fetch(chrom, start_pos, stop_pos)
        window_ref_sequence = ref_seq.fetch(chrom, start_pos, stop_pos).upper()
        # Process fragments directly into profiles - no intermediate lists
        _, _, weight_sq_window = process_fragments_fast(
            site_reads, start_pos, frag_range, map_q, gc_bias, window_ref_sequence,
            depth, nc_signal, fragment_lengths, fdict, roi_length, flank_size,
            profiles=fragment_profiles, bin_size=bin_size
        )

        # One-hot encoded reference sequence of the window (no flanks); bases other than A/C/G/T stay all-zero
        oh_seq = None
        if window is not None:
            oh_seq = np.zeros((window, 4), dtype=np.float32)
            for i, base in enumerate(window_ref_sequence[flank_size:-flank_size]):
                if base in base_to_idx:
                    oh_seq[i, base_to_idx[base]] = 1.0
        if strand == "-":
            # Flip all profile data for negative strand to ensure 5'->3' orientation
            depth = depth[::-1]
            nc_signal = nc_signal[::-1]
            if fragment_profiles is not None:
                fragment_profiles = mirror_fragment_profiles(fragment_profiles)
            if oh_seq is not None:
                # Reverse complement, so the sequence is the site's own strand read 5'->3' like the signal
                # profiles: reversing the A/C/G/T columns exchanges A with T and C with G
                oh_seq = oh_seq[::-1, ::-1]
    
    # generate phased profile and phasing/region-level features --------------------------------------------------------
    if np.count_nonzero(depth > 0) < 0.8 * roi_length:  # skip sites with less than 80% base coverage
        if not stack:  # For non-stacked mode, add the site to skipped_sites
            # Just include the original line plus the reason code
            skipped_sites = ['{}\tinsufficient_coverage'.format(region.strip())]
        return (site,) + (np.nan,) * 15 + ([],) + ([],) + (skipped_sites,)
    
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
    
    # Apply flank normalization to depth and, in window mode, to the subnucleosomal/nucleosomal depth and the
    # 5'/3' fragment end densities (each divided by its own flanking mean)
    # PN signal (nc_signal) is already pre-normalized before FFT in non-composite mode
    # In composite mode, signals are already per-site flank-normalized, so avoid double normalization
    if stack:
        flanking_depth_signal = 1.0
    else:
        # Use inner flanks (exclude the outermost ~max fragment length) to avoid ROI-edge truncation bias
        edge_exclude = min(int(frag_range[1]), int(flank_size - 1))
        if edge_exclude < 1:
            edge_exclude = 1

        left_depth = depth[edge_exclude:flank_size]
        right_depth = depth[-flank_size:-edge_exclude]
        flanking_depth_signal = (np.mean(left_depth) + np.mean(right_depth)) / 2

        if fragment_profiles is not None:
            coverage_profiles = {}
            for name in ('short', 'long', 'end5', 'end3'):
                flank_mean = inner_flank_mean(fragment_profiles[name], edge_exclude)
                # A site with no flanking signal for a track has no defined level for it
                coverage_profiles[name] = (fragment_profiles[name] / flank_mean if flank_mean > 0
                                           else np.full(roi_length, np.nan, dtype=np.float32))
    
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
        # phased_signal is already flank-normalized (pre-FFT in window/region modes, per-site in composite mode)
        pn_amp = 0.5 * (np.mean(max_values) - np.mean(min_values))
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
            return (site,) + (np.nan,) * 15 + (np.full((8 if stack else 12, roi_length - 2*flank_size), np.nan, dtype=np.float32),) + ([],) + ([] if not stack else skipped_sites,)
        # Effective fragment count (Kish): GC weights make fragments count unequally
        n_eff_window = fragment_total ** 2 / weight_sq_window
    else:
        # In composite mode, fragment_lengths is already a probability distribution pooled over sites,
        # so there is no single fragment count to correct for
        fragment_lengths_pd = fragment_lengths
        n_eff_window = None
    
    # Ensure fragment_lengths_pd has no NaN or inf values before passing to frag_metrics
    fragment_lengths_pd = np.nan_to_num(fragment_lengths_pd, nan=0.0, posinf=0.0, neginf=0.0)
    
    fl_mean, fl_stdev, fl_skew, fl_kurt, fl_ratio_raw, fl_ent, fl_gsi = frag_metrics(fragment_lengths_pd, bins=frag_bins, n_eff=n_eff_window)
    # Apply log2 transformation to subnucleosomal ratio
    fl_ratio = np.log2(fl_ratio_raw) if not np.isnan(fl_ratio_raw) and fl_ratio_raw > 0 else np.nan

    # Per-position Gini-Simpson profile (window modes), reported as its difference from the flank median:
    # an effect size on the index's own 0-1 scale, whose expectation does not depend on sequencing depth
    # (dividing by the flank spread would not: that spread shrinks as depth grows)
    window_gsi, central_gsi = np.nan, np.nan
    if stack:
        gini_raw = gini_simpson_profile(fragment_length_profile)
    elif fragment_profiles is not None:
        gini_raw = gini_simpson_profile(fragment_profiles['length'], fragment_profiles['weight_sq'])
    else:
        gini_raw = None
    if gini_raw is not None:
        # Donut reference: the flanks only, excluding the window
        flank_gini = np.concatenate([gini_raw[:flank_size], gini_raw[-flank_size:]])
        flank_gini = flank_gini[np.isfinite(flank_gini)]
        # Fewer than 10 valid flank positions give no usable reference level
        gini_signal = gini_raw - np.median(flank_gini) if len(flank_gini) >= 10 else np.full(roi_length, np.nan)
        gini_signal = gini_signal[flank_size:-flank_size]
        if not np.all(np.isnan(gini_signal)):
            window_gsi = np.nanmean(gini_signal)
        central_gini = gini_signal[(center - 5):(center + 6)]
        if len(central_gini) > 0 and not np.all(np.isnan(central_gini)):
            central_gsi = np.nanmean(central_gini)
    # Sequence profile (window mode only): 4 x N, rows A, C, G, T
    seq_profile = oh_seq.T if not stack and oh_seq is not None else None
    
    # combine and save profiles ----------------------------------------------------------------------------------------
    out_array = np.nan
    if window is not None and window <= 5000:  # do not print profiles longer than 5kb to avoid memory issues
        signal_array = np.vstack((depth, coverage_profiles['end5'][flank_size:-flank_size], coverage_profiles['end3'][flank_size:-flank_size],
                                  phased_signal, coverage_profiles['short'][flank_size:-flank_size], coverage_profiles['long'][flank_size:-flank_size],
                                  gini_signal, peak_profile))
        if seq_profile is not None:
            out_array = np.concatenate((signal_array, seq_profile), axis=0)
        else:
            out_array = signal_array
        # Profiles are saved as float32; casting here halves what each worker sends back and the parent holds
        out_array = out_array.astype(np.float32)

    return site, fl_mean, fl_stdev, fl_skew, fl_kurt, fl_ratio, fl_ent, fl_gsi, pn_score, pn_spacing, pn_amp,\
           mean_depth, central_depth, central_gsi, window_depth, window_gsi, out_array,\
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
    sites = [region for region in open(sites_path, 'r') if region.strip()]
    if not sites:
        raise ValueError(f'Annotation file is empty: {sites_path}')
    frag_dict = pickle.load(open(dict_path, 'rb'))
    if run_mode in ['region', 'window']:
        header, has_header = bed_header(sites[0])
        if has_header:
            sites.pop(0)
    elif run_mode == 'composite-window':
        with open(sites[0].strip(), 'r') as f:
            header, has_header = bed_header(f.readline())
    else:
        raise ValueError(f"Invalid run_mode ({run_mode}): must be one of 'region', 'window', or 'composite-window'")
    if has_header:
        print(f'BED header: {header}')
    else:
        print(f'WARNING: No header line found in BED file(s): assuming standard BED column order {header}')
       
    chr_idx = get_index(header, ['chrom', 'Chrom', '#chrom', 'chr', 'chromosome', 'Chromosome'], 0, 'No "chrom" column found in BED file(s): defaulting to index {} ({})')
    start_idx = get_index(header, ['chromStart', 'start', 'Start'], 1, 'No "chromStart" column found in BED file(s): defaulting to index {} ({})')
    stop_idx = get_index(header, ['chromEnd', 'end', 'End'], 2, 'No "chromEnd" column found in BED file(s): defaulting to index {} ({})')
    if run_mode in ['region', 'window']:
        site_idx = get_index(header, ['name', 'Gene'], 3, 'No "name" column found in BED file(s): defaulting to index {} ({})')
        strand_idx = get_index(header, ['strand', 'Strand'], None, 'WARNING: No "strand" column found in BED file(s): defaulting to "+" for all sites')
    else:
        strand_idx = get_index(header, ['strand', 'Strand'], None, 'WARNING: No "strand" column found in BED file(s): both strands will be averaged for each site')
        site_idx = None
    pos_idx = get_index(header, ['position'], None, 'No "position" column found in BED file(s): will compute midpoint from start+end') if run_mode != 'region' else None
    if site_idx is not None and site_idx >= len(header):
        raise ValueError(f'{run_mode} mode needs a site name column ("name", or the 4th column), but the BED file has {len(header)} columns')

    params = [size_range, map_q, run_mode,
              chr_idx, start_idx, stop_idx, site_idx, strand_idx, pos_idx]

    print('\n### Running Triton on {} region sets ###\n'.format(len(sites)))

    if len(sites) < cpus:
        print('More CPUs passed than sites - scaling down to {}'.format(len(sites)))
        cpus = len(sites)

    # Use parallel processing by default for production runs with pool initializer
    print(f'Processing {len(sites)} sites in parallel using {cpus} CPUs...')
    with multiprocessing.Pool(processes=cpus, initializer=pool_initializer, 
                             initargs=(bam_path, ref_seq_path, gc_bias, frag_dict)) as pool:
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
            'central-gini-simpson',
            'window-depth', 
            'window-gini-simpson',]

    # Prepare data for wide-form DataFrame (sites as rows, features as columns)
    sites_data = []
    
    for result in results:
        """ result[0] = site
            result[1-15] = features in order of 'keys' list above
            result[16] = numpy array of signal profiles (or np.nan in region mode)
            result[17] = fragment length probability distribution (1-500bp)
            result[18] = skipped sites list (composite-window mode only; empty list if none skipped) """
            
        # Create a row for this site with all its features
        site_row = {'site': result[0], 'sample': sample_name}
        for i, key in enumerate(keys):
            site_row[key] = result[i+1]
        
        sites_data.append(site_row)
        
        # Store signal profiles and fragment length probability distributions separately
        signal_dict[result[0]] = result[16]
        fl_dict[result[0]] = result[17]
        
        if len(result) == 19 and result[18]:  # skipped sites list exists and is not empty
            # Debug print to see what the skipped sites look like
            for line in result[18]:
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
    
    np.savez_compressed(signal_file, **signal_dict)
    
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
                first_header = header
                
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
