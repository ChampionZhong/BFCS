"""
Regenerated Google-style docstrings for module 'spikeinterface'.
README source: others/readme/spikeinterface/README.md
Generated at: 2025-12-02T04:04:41.874667Z

Total functions: 159
"""


import numpy

################################################################################
# Source: spikeinterface.comparison.comparisontools.compare_spike_trains
# File: spikeinterface/comparison/comparisontools.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_comparison_comparisontools_compare_spike_trains(
    spiketrain1: numpy.ndarray,
    spiketrain2: numpy.ndarray,
    delta_frames: int = 10
):
    """Compares two spike trains and labels each spike as a true positive (TP), false negative (FN), or false positive (FP) for benchmarking spike sorting outputs. This function is intended for use in the SpikeInterface comparison/benchmarking workflow where spiketrain1 is treated as the ground-truth spike times and spiketrain2 is treated as the spike times produced by a sorter or another detection method. The function finds one-to-one matches between spikes in spiketrain1 and spikes in spiketrain2 within a symmetric time window defined by delta_frames (the effective half-width is delta_frames // 2 frames). If multiple spikes in spiketrain2 fall within the matching window of a single ground-truth spike, only the first unmatched spiketrain2 spike is paired (the others remain labeled as false positives). The comparison uses integer-frame arithmetic for spiketrain2 by casting it to int via numpy.astype(int), so inputs should generally represent spike times in frame indices.
    
    Args:
        spiketrain1 (numpy.ndarray): 1-D array of spike times for the first spike train. In the SpikeInterface benchmarking context this array is expected to represent ground-truth spike times in recording frame indices. Each element is the time (frame index) of a spike. The function iterates over these values and attempts to find a matching spike in spiketrain2 within the delta_frames window. If spiketrain1 contains non-integer values they will be compared against spiketrain2.astype(int) (see notes below); to avoid ambiguity, provide integer frame indices when possible.
        spiketrain2 (numpy.ndarray): 1-D array of spike times for the second spike train (e.g., output of a spike sorter). This array is converted to integers using numpy.astype(int) inside the function before computing absolute differences; fractional values in spiketrain2 will be truncated toward zero prior to matching. Each element corresponds to a detected spike time (frame index). The function labels each element in this array as "TP" if it is paired to a ground-truth spike, otherwise as "FP".
        delta_frames (int): Positive integer window size in frames used to decide whether two spikes match. The function computes matches using the condition abs(spiketrain2.astype(int) - n_sp) <= delta_frames // 2 where n_sp is a value from spiketrain1. The effective half-window is computed with integer division (//), so delta_frames = 10 corresponds to a ±5-frame tolerance. Default is 10. Supplying zero or a negative integer will make the matching window empty or negative (no matches will be found) and is not meaningful for typical benchmarking; the function does not explicitly validate positivity and may therefore return only FN/FP labels in such cases.
    
    Returns:
        lab_st1 (numpy.ndarray): 1-D numpy array of string labels for each spike in spiketrain1, of the same length and relative ordering as spiketrain1. Possible labels are:
            "TP" for a ground-truth spike that was paired with a single spike in spiketrain2 within the matching window,
            "FN" for a ground-truth spike that remained unpaired (false negative).
        lab_st2 (numpy.ndarray): 1-D numpy array of string labels for each spike in spiketrain2, of the same length and relative ordering as spiketrain2. Possible labels are:
            "TP" for a detected spike that was paired to one ground-truth spike,
            "FP" for a detected spike that remained unpaired (false positive).
    
    Behavior, side effects, defaults, and failure modes:
        - The function treats spiketrain1 as the ground truth and attempts to create one-to-one pairings with spiketrain2; it intentionally avoids counting multiple detections in spiketrain2 as multiple true positives for the same ground-truth spike. When more than one spiketrain2 spike falls inside the matching window of a single ground-truth spike, only the first unmatched spiketrain2 spike (by index order) is paired; the remaining nearby spiketrain2 spikes remain labeled "FP".
        - The matching condition uses spiketrain2.astype(int) and delta_frames // 2; therefore fractional values in spiketrain2 are truncated to integers prior to matching, and delta_frames is effectively halved via integer division. To avoid truncation ambiguity and ensure precise benchmarking, provide spike times as integer frame indices when possible.
        - The function returns new numpy arrays lab_st1 and lab_st2 and does not modify the input arrays in-place.
        - Performance: the implementation loops over spiketrain1 and for each element computes a vectorized absolute difference against spiketrain2, so runtime scales approximately with len(spiketrain1) * len(spiketrain2) in the worst case. For very large spike trains this may be slow or memory intensive.
        - Input validation: the function does not perform comprehensive type or value validation. If spiketrain1 or spiketrain2 are not numeric numpy arrays, or contain NaNs, or are not one-dimensional, behavior may be undefined or a Python/NumPy exception may be raised. If delta_frames is not an integer, Python will attempt to use it in integer division; non-integer types may raise an error.
        - Label strings are exact and uppercase: "TP", "FN", "FP". Initially all entries are "UNPAIRED" internally, then converted as described; the returned arrays contain only the labels documented above.
        - Intended use: this routine is intended for spike sorting comparison and benchmarking workflows within the SpikeInterface framework (see project README), where concise per-spike labels are required to compute metrics such as precision, recall, and F1 score.
    """
    from spikeinterface.comparison.comparisontools import compare_spike_trains
    return compare_spike_trains(spiketrain1, spiketrain2, delta_frames)


################################################################################
# Source: spikeinterface.comparison.comparisontools.compute_agreement_score
# File: spikeinterface/comparison/comparisontools.py
# Category: valid
################################################################################

def spikeinterface_comparison_comparisontools_compute_agreement_score(
    num_matches: int,
    num1: int,
    num2: int
):
    """spikeinterface.comparison.comparisontools.compute_agreement_score computes an agreement score between two spike trains or spike-sorting units. In the SpikeInterface domain this function is used when comparing and benchmarking spike sorting outputs (for example, when matching units from two sortings or comparing a sorter output to ground truth) to quantify how many events (spikes) are shared relative to the total distinct events across both spike trains.
    
    Args:
        num_matches (int): Number of matching events (spike times) between the two spike trains. This is expected to be an integer count produced by a matching procedure (for example, counting coincident spikes within a tolerance) and represents the intersection size of the event sets.
        num1 (int): Number of events (spikes) in spike train 1. This is the integer count of events for the first unit/sorting output and represents the size of the first set in the comparison.
        num2 (int): Number of events (spikes) in spike train 2. This is the integer count of events for the second unit/sorting output and represents the size of the second set in the comparison.
    
    Behavior:
        The function computes the score as num_matches / (num1 + num2 - num_matches), where the denominator is the total number of distinct events across both spike trains (the union size), assuming num_matches counts events that are present in both. If the denominator is zero (which occurs when num1 and num2 are both zero and num_matches is zero), the function returns 0. The function performs no I/O, has no side effects, and runs in constant time O(1).
    
    Practical significance:
        The returned agreement score provides a normalized measure of overlap between two sets of spike events and is useful in benchmarking and comparing spike sorting outputs within the SpikeInterface framework. When inputs are valid non-negative counts and num_matches ≤ min(num1, num2), the score reflects the fraction of shared events relative to the union of events and is commonly used to rank or threshold matches during comparison workflows.
    
    Failure modes and input validation:
        The function does not perform explicit validation of input values beyond the integer typing in the signature. Supplying negative integers or values inconsistent with set semantics (for example, num_matches > min(num1, num2)) will produce a numeric result that may be semantically meaningless for spike-count comparisons. In the specific coded behavior, a zero union (denominator == 0) is handled by returning 0 rather than raising an error.
    
    Returns:
        float: Agreement score computed as num_matches / (num1 + num2 - num_matches). If the denominator (num1 + num2 - num_matches) equals zero, returns 0. The return value is a floating-point scalar representing the normalized overlap between the two spike trains.
    """
    from spikeinterface.comparison.comparisontools import compute_agreement_score
    return compute_agreement_score(num_matches, num1, num2)


################################################################################
# Source: spikeinterface.comparison.comparisontools.count_match_spikes
# File: spikeinterface/comparison/comparisontools.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_comparison_comparisontools_count_match_spikes(
    times1: numpy.ndarray,
    all_times2: list,
    delta_frames: int
):
    """Computes, for one reference spike train, the number of matching spikes found in each spike train of a second collection of spike trains.
    
    This function is used in the spike sorting comparison and benchmarking workflows of SpikeInterface to quantify how many spikes from a single unit (times1) are matched by units from another sorting (all_times2). For each spike train in all_times2 the function calls count_matching_events(times1, times2, delta=delta_frames) to count spikes in times2 that occur within a tolerance of delta_frames (measured in sample frames) of spikes in times1. The result is an array of integer counts with one entry per element of all_times2 in the same order. The function has no side effects on its inputs and is intended for use in comparing sorted outputs, computing per-unit overlap, and building confusion/matching matrices for benchmarking.
    
    Args:
        times1 (numpy.ndarray): 1-D array of spike times expressed as integer frame indices for the reference spike train (unit) from sorting 1. In the SpikeInterface comparison context this represents the event times for one unit whose matches across another sorting are being counted. The array is not modified by this function; if it is not a numeric 1-D array, downstream routines (count_matching_events) may raise errors.
        all_times2 (list): List of spike train arrays (e.g., numpy.ndarray objects) where each element is a 1-D array of spike times expressed as integer frame indices from sorting 2. Each list element corresponds to one candidate unit in the other sorting; the function iterates over this list in order and counts matches for each element.
        delta_frames (int): Integer tolerance in sample frames used to decide whether a spike in times2 matches a spike in times1. This value is passed directly to count_matching_events as its delta argument and therefore controls the temporal window for matching events during comparisons between sortings.
    
    Returns:
        matching_event_counts (numpy.ndarray): 1-D numpy array of dtype int64 containing the number of matching events for each spike train in all_times2. The length and ordering of this array match len(all_times2) and the order of elements in all_times2 respectively. If inputs have incorrect types or shapes, or if the underlying count_matching_events function raises an error for the provided delta_frames, those exceptions will propagate to the caller.
    """
    from spikeinterface.comparison.comparisontools import count_match_spikes
    return count_match_spikes(times1, all_times2, delta_frames)


################################################################################
# Source: spikeinterface.comparison.comparisontools.do_confusion_matrix
# File: spikeinterface/comparison/comparisontools.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_comparison_comparisontools_do_confusion_matrix(
    event_counts1: dict,
    event_counts2: dict,
    match_12: dict,
    match_event_count: numpy.ndarray
):
    """Compute the confusion matrix between one ground-truth sorting and another sorting.
    
    This function is used in SpikeInterface's comparison and benchmarking workflow to aggregate per-unit true positive, false negative, and false positive event counts when comparing two spike-sortings (for example, a ground-truth sorting versus an automated sorter). It takes per-unit event counts for each sorting, a mapping from units in sorting1 to units in sorting2 (the match), and a matrix of matched-event counts between units, and returns a pandas DataFrame that encodes for each pair of matched units the number of matched events and, for each unit, the remaining false negatives (FN) and false positives (FP). The resulting confusion matrix is suitable for downstream quality metrics, summaries, and reports in the SpikeInterface benchmarking pipeline.
    
    Args:
        event_counts1 (pd.Series): Number of events per unit in sorting1 (ground-truth). The Series index must contain unit identifiers for sorting1 that correspond to the row index of match_event_count and to the index of match_12. Values are integer event counts for each unit in sorting1; these counts are used to compute false negatives (FN) per unit as event_counts1[unit1] - matched_events. Missing indices or non-integer values will typically raise a KeyError or result in incorrect FN values.
        event_counts2 (pd.Series): Number of events per unit in sorting2 (the compared sorting). The Series index must contain unit identifiers for sorting2 that correspond to the column index of match_event_count and to the values referenced by match_12. Values are integer event counts for each unit in sorting2; these counts are used to compute false positives (FP) per unit as event_counts2[unit2] - matched_events. Missing indices or non-integer values will typically raise a KeyError or result in incorrect FP values.
        match_12 (pd.Series): Mapping from units in sorting1 to units in sorting2. This Series uses sorting1 unit identifiers as its index and has integer values equal to the matching sorting2 unit identifier, or -1 to indicate that a unit in sorting1 is unmatched. match_12 is typically produced by a matching procedure such as the Hungarian or best-match algorithm used in SpikeInterface comparisons. Entries equal to -1 are treated as unmatched units; non-existent unit identifiers or mismatched labels relative to match_event_count will raise KeyError when the function accesses the match counts.
        match_event_count (pd.DataFrame): Matrix of matched-event counts between units, as produced by make_match_count_matrix in SpikeInterface. Rows must be indexed by sorting1 unit identifiers and columns by sorting2 unit identifiers. Each cell at (unit1, unit2) contains the integer number of matched events between those units. This matrix is used to fill the confusion matrix's per-pair true positive counts. Missing cells, misaligned indices/columns, or non-integer entries will raise KeyError or produce incorrect results.
    
    Returns:
        pd.DataFrame: Confusion matrix of integer counts with shape (N1 + 1, N2 + 1) where N1 is the number of units in sorting1 (as given by match_event_count.index) and N2 is the number of units in sorting2 (as given by match_event_count.columns). The DataFrame index lists units from sorting1 ordered with matched units first followed by unmatched units, and an additional final row labeled "FP" that accumulates per-unit false positives for sorting2. The DataFrame columns list units from sorting2 ordered with matched units first followed by unmatched units, and an additional final column labeled "FN" that accumulates per-unit false negatives for sorting1. For each matched pair (u1, u2) the cell at (u1, u2) contains the number of matched events from match_event_count; the cell at (u1, "FN") is event_counts1[u1] - matched_events; the cell at ("FP", u2) is event_counts2[u2] - matched_events. For unmatched units, the "FN" or "FP" entries are set to the full event_counts1 or event_counts2 value respectively. The returned DataFrame uses integer dtype and is a pure functional output (no side effects on inputs).
    
    Failure modes and notes:
        - The function assumes indices and labels in event_counts1, event_counts2, match_12, and match_event_count are consistent. Mismatched labels or missing indices will raise KeyError when accessing Series/DataFrame entries.
        - Values in event_counts1, event_counts2, and match_event_count are expected to be integer event counts; non-integer or negative values will produce meaningless FN/FP results but will not be explicitly validated by this function.
        - match_12 values must be unit identifiers present in match_event_count.columns or -1 for unmatched; otherwise a KeyError will occur when reading matched counts.
        - This function does not modify any input objects; it constructs and returns a new pandas DataFrame representing the confusion matrix.
    """
    from spikeinterface.comparison.comparisontools import do_confusion_matrix
    return do_confusion_matrix(
        event_counts1,
        event_counts2,
        match_12,
        match_event_count
    )


################################################################################
# Source: spikeinterface.comparison.comparisontools.do_count_score
# File: spikeinterface/comparison/comparisontools.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_comparison_comparisontools_do_count_score(
    event_counts1: dict,
    event_counts2: dict,
    match_12: dict,
    match_event_count: dict
):
    """Do a per-ground-truth-unit count of true positives (tp), false negatives (fn), false positives (fp) and related bookkeeping used when comparing two spike sortings (for example a ground-truth sorting vs a tested sorting) within the SpikeInterface comparison workflow.
    
    This function iterates over each unit in event_counts1 (treated as the ground-truth units) and uses the provided mapping from sorting1 to sorting2 and the precomputed match event counts to compute, for each ground-truth unit, how many events were correctly matched (tp), missed (fn), and extra in the tested unit (fp). The result is a pandas DataFrame with one row per ground-truth unit and columns that summarize counts necessary for downstream accuracy/score computations and benchmarking of spike sorting outputs.
    
    Args:
        event_counts1 (pd.Series): Number of events (spikes) per unit in sorting1. In the SpikeInterface context this is typically the ground-truth per-unit event count produced by counting spike times in the ground-truth SortingExtractor or equivalent representation. The Series index must contain the unit identifiers for sorting1 and is used as the set of ground-truth unit ids iterated by this function. Values must be non-negative integers. The function uses event_counts1.at[unit_id] to obtain num_gt for the unit.
        event_counts2 (pd.Series): Number of events (spikes) per unit in sorting2 (the tested sorting). In benchmarking workflows this represents the per-unit spike counts produced by the sorter being evaluated. The Series index must contain the unit identifiers for sorting2 that appear as targets in match_12. Values must be non-negative integers. The function uses event_counts2.at[tested_unit_id] when computing num_tested and fp.
        match_12 (pd.Series): Mapping from units in sorting1 to units in sorting2. This Series models the matching (for example the Hungarian or best-match mapping) where match_12.at[unit1_id] yields the corresponding unit id in sorting2 that best matches unit1_id. If a ground-truth unit is unmatched, match_12 should contain either -1 or the empty string "" for that unit; the function treats those values as indicating no tested unit and records zero tested events and zero true positives. The index of this Series must align with event_counts1.index (the ground-truth unit ids).
        match_event_count (pd.DataFrame): Matrix of matched-event counts produced by make_match_count_matrix. Rows are unit ids from sorting1 (ground truth) and columns are unit ids from sorting2 (tested). The entry match_event_count.at[unit1_id, unit2_id] is the number of events that are considered matched between unit1_id and unit2_id. The function reads this matrix to determine the true positive count for a matched pair. Missing rows/columns or mismatched indices will raise a KeyError.
    
    Behavior, side effects, defaults, and failure modes:
        This function has no external side effects (it does not modify inputs) and returns a new pandas DataFrame. The returned DataFrame has index equal to event_counts1.index and index name set to "gt_unit_id". The DataFrame columns are exactly: "tp", "fn", "fp", "num_gt", "num_tested", "tested_id". For each ground-truth unit u1:
            - If match_12[u1] is -1 or "", the unit is considered unmatched: "tested_id" is set to that value, "num_tested" is set to 0, "tp" and "fp" are set to 0, and "fn" and "num_gt" are set to event_counts1[u1].
            - Otherwise let u2 = match_12[u1] and num_match = match_event_count.at[u1, u2]; then "tp" = num_match, "fn" = event_counts1.at[u1] - num_match, "fp" = event_counts2.at[u2] - num_match, "num_gt" = event_counts1.at[u1], and "num_tested" = event_counts2.at[u2].
        The function assumes counts in event_counts1, event_counts2, and match_event_count are non-negative integers; negative values may lead to misleading outputs. If indices do not align (for example unit ids present in match_12 are missing from event_counts2 or match_event_count), pandas will raise KeyError when accessing .at[...] and the function will fail. The function does not validate types beyond relying on pandas indexing operations; callers should pass pandas Series/DataFrame as documented.
        This per-unit counting is intended to be used as part of spike sorting comparison and benchmarking pipelines in SpikeInterface (for example to compute precision/recall or to build summary tables used in reports), and the returned counts should be interpreted in that domain: "tp" are matched spikes between a ground-truth unit and its best-matching tested unit, "fn" are ground-truth spikes not matched, and "fp" are extra tested spikes attributed to the tested unit beyond the matched ones.
    
    Returns:
        pd.DataFrame: A new pandas DataFrame indexed by ground-truth unit ids (index name "gt_unit_id") containing one row per ground-truth unit and exactly the columns ["tp", "fn", "fp", "num_gt", "num_tested", "tested_id"]. Column semantics: "tp" is the number of matched events between the ground-truth unit and its mapped tested unit; "fn" is the number of ground-truth events not matched; "fp" is the number of extra events in the tested unit beyond matches; "num_gt" is the total number of ground-truth events for the unit; "num_tested" is the total number of events in the matched tested unit (zero if unmatched); "tested_id" is the id of the tested unit matched to the ground-truth unit (or -1 / "" if unmatched).
    """
    from spikeinterface.comparison.comparisontools import do_count_score
    return do_count_score(event_counts1, event_counts2, match_12, match_event_count)


################################################################################
# Source: spikeinterface.comparison.comparisontools.make_agreement_scores_from_count
# File: spikeinterface/comparison/comparisontools.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_comparison_comparisontools_make_agreement_scores_from_count(
    match_event_count: numpy.ndarray,
    event_counts1: numpy.ndarray,
    event_counts2: numpy.ndarray
):
    """Compute pairwise agreement (Jaccard-like) scores between units from two spike-sorting outputs using a precomputed matrix of matched event counts. This function is used in the comparison/benchmarking stage of SpikeInterface to quantify how well units from two sortings correspond to each other: each score expresses the fraction of shared events relative to the union of events for the two units being compared. It implements the same agreement definition used by make_agreement_scores but accepts a precomputed match-event-count matrix to avoid recomputing matches.
    
    Args:
        match_event_count (numpy.ndarray): A 2-D matrix where element (i, j) is the number of matched events between unit i from sorting A and unit j from sorting B. In the SpikeInterface comparison workflow this typically comes from a cross-matching procedure and is used as the numerator when computing pairwise agreement scores. The implementation expects that this argument provides .values for numeric data and, if available, .index and .columns to preserve unit labels when constructing the returned DataFrame; in practice this is commonly a pandas.DataFrame with shape (n_units_A, n_units_B) but the numeric content must be compatible with numpy array semantics.
        event_counts1 (numpy.ndarray): A 1-D array (length n_units_A) of integer event counts (spike counts) for each unit in sorting A. These counts represent the total number of events assigned to each unit in sorting A and are used as part of the denominator in the agreement score formula (see Returns). The function accesses event_counts1.values if present (e.g., when a pandas.Series is passed) or uses the array-like numeric values directly.
        event_counts2 (numpy.ndarray): A 1-D array (length n_units_B) of integer event counts for each unit in sorting B. These counts represent the total number of events assigned to each unit in sorting B and are used as part of the denominator in the agreement score formula. As with event_counts1, the function will use .values if the argument is a pandas.Series or otherwise use the numeric content.
    
    Returns:
        pandas.DataFrame: A DataFrame of shape (n_units_A, n_units_B) containing the pairwise agreement scores computed as match_event_count / (event_counts1 + event_counts2 - match_event_count) for each unit pair. The returned DataFrame preserves the row and column labels from match_event_count.index and match_event_count.columns when match_event_count provides them; otherwise, numeric 0-based integer labels are used implicitly by pandas. Scores follow a Jaccard-like definition bounded between 0 and 1 inclusive for valid non-negative counts: 0 indicates no shared events; 1 indicates perfect agreement (all events shared). Implementation notes and failure modes: the function computes the denominator with numpy broadcasting as event_counts1.values[:, None] + event_counts2.values[None, :] - match_event_count.values. To avoid division-by-zero, any denominator entries that evaluate to 0 are temporarily set to -1; because match_event_count is also 0 in those cases, the resulting score is 0. If input shapes are incompatible for broadcasting (lengths not matching the rows/columns of match_event_count) a numpy broadcasting or indexing error will be raised. The function assumes non-negative integer counts; negative or non-numeric entries may produce invalid scores or raise exceptions. The return value has no side effects beyond constructing and returning the DataFrame.
    """
    from spikeinterface.comparison.comparisontools import make_agreement_scores_from_count
    return make_agreement_scores_from_count(
        match_event_count,
        event_counts1,
        event_counts2
    )


################################################################################
# Source: spikeinterface.comparison.comparisontools.make_matching_events
# File: spikeinterface/comparison/comparisontools.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_comparison_comparisontools_make_matching_events(
    times1: numpy.ndarray,
    times2: numpy.ndarray,
    delta: int
):
    """Compute matching (colliding) spike events between two spike trains and return the indices
    of the matching spikes together with their frame difference. This helper is used by
    spikeinterface.comparison tools for collision detection when comparing spike-sorted outputs:
    it identifies pairs of spikes (one from each train) that occur within a given frame window
    (delta) of each other and returns the corresponding indices into the original spike trains.
    The implementation concatenates the two input time arrays, tags membership (train 1 or train 2),
    sorts by time, and then selects adjacent cross-train pairs whose time difference is <= delta.
    Only adjacent cross-train neighbors in the sorted time sequence are considered (not all
    pairwise combinations within the window), and the final array is sorted by index1.
    
    Args:
        times1 (numpy.ndarray): 1D array of frame/sample indices for spike train 1. Each element
            is the frame number (integer) at which a spike was detected in train 1. This array is
            used to produce index1 values in the output; indices are zero-based positions into
            this array (dtype int64 in the returned structured array). The function does not
            modify times1.
        times2 (numpy.ndarray): 1D array of frame/sample indices for spike train 2. Each element
            is the frame number (integer) at which a spike was detected in train 2. This array is
            used to produce index2 values in the output; indices are zero-based positions into
            this array (dtype int64 in the returned structured array). The function does not
            modify times2.
        delta (int): Threshold in frames for considering two spikes a matching event (collision).
            Two spikes (one from times1 and one from times2) are considered matching if the
            difference in their frame indices is less than or equal to delta. delta should be an
            integer (typically non-negative); a negative delta will result in no matches because
            no non-negative time differences satisfy the <= condition.
    
    Returns:
        numpy.ndarray: A 1D structured numpy array with dtype([('index1','int64'),
            ('index2','int64'), ('delta_frame','int64')]). Each record represents a detected
            matching event (collision) between the two spike trains:
        - index1: integer index into the input times1 array (zero-based) identifying the spike from
          train 1 participating in the match.
        - index2: integer index into the input times2 array (zero-based) identifying the spike from
          train 2 participating in the match.
        - delta_frame: non-negative integer equal to the absolute difference in frames between the
          matched spikes (<= delta).
        The returned array is sorted by index1 in ascending order. If no matching events are found,
        an empty structured numpy array with the same dtype is returned.
    
    Behavior, side effects, and failure modes:
        - The function is pure with no side effects; inputs times1 and times2 are not modified.
        - Inputs are expected to be 1D numpy.ndarray of integer frame indices; passing arrays with
          extra dimensions or non-integer types may lead to unintended behavior or numpy errors
          during concatenation and sorting.
        - The matching logic only considers adjacent cross-train neighbors after time sorting.
          For closely clustered spikes where more than two events fall within delta, this method
          pairs adjacent neighbors in time rather than enumerating all possible pairwise matches
          inside the window.
        - If times1 or times2 are empty, the function returns an empty structured array.
        - If inputs are not compatible with numpy.concatenate (for example, non-array objects
          lacking shape/size attributes), numpy will raise an exception.
    """
    from spikeinterface.comparison.comparisontools import make_matching_events
    return make_matching_events(times1, times2, delta)


################################################################################
# Source: spikeinterface.core.channelsaggregationrecording.aggregate_channels
# File: spikeinterface/core/channelsaggregationrecording.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_core_channelsaggregationrecording_aggregate_channels(
    recording_list_or_dict: list = None,
    renamed_channel_ids: numpy.ndarray = None,
    recording_list: list = None
):
    """Aggregates channels from multiple recordings into a single ChannelsAggregationRecording wrapper used by SpikeInterface for preprocessing, sorting, waveform extraction, visualization, and other multi-recording operations.
    
    This function constructs and returns a ChannelsAggregationRecording that logically concatenates or combines channels provided by multiple BaseRecording objects so they can be treated as one recording by downstream SpikeInterface components. It accepts either a list or a dict of recordings, an optional array-like of renamed channel ids to override channel identifiers in the aggregate, and an optional alternate list parameter. The function itself is a thin factory that forwards these arguments to ChannelsAggregationRecording and performs no additional I/O; actual data access semantics (lazy, memory-mapped, or in-memory) depend on the underlying BaseRecording implementations.
    
    Args:
        recording_list_or_dict (list | dict): List or dict of BaseRecording objects to aggregate. When a list is passed, each element must be a BaseRecording instance representing an extracellular recording (single or multi-channel) from the SpikeInterface ecosystem. When a dict is passed, keys can be used as identifiers and values must be BaseRecording instances. At least one of recording_list_or_dict or recording_list should be provided; if both are None, ChannelsAggregationRecording will typically raise an error. The practical significance is that this argument defines which recordings contribute channels to the aggregated recording so downstream tools (preprocessing, sorters, visualization) operate on the combined channel set.
        renamed_channel_ids (array-like): If given, provides the new channel ids to assign to the aggregated channels. This should be an array-like object (for example, numpy array or any sequence) whose length matches the total number of channels resulting from the aggregation of the provided recordings. The role of this argument is to override or standardize channel identifiers across recordings (for instance to unify numbering or to avoid id collisions) before downstream processing. If the length or contents are invalid (mismatch with total channels, duplicate ids where not allowed, or incompatible types), a ValueError or TypeError may be raised by the underlying ChannelsAggregationRecording implementation.
        recording_list (list): Alternate way to pass a list of BaseRecording objects to aggregate. This parameter is forwarded to ChannelsAggregationRecording as provided. It exists to support different calling patterns; when both recording_list_or_dict and recording_list are supplied, resolution of any conflict (for example duplicate recordings or inconsistent metadata) is deferred to ChannelsAggregationRecording and may result in an error.
    
    Returns:
        ChannelsAggregationRecording: The aggregated recording object. This wrapper represents the combined channel set from the supplied BaseRecording inputs and is intended for use with SpikeInterface workflows (preprocessing, running sorters, waveform extraction, visualization, quality metrics, export). The returned object is constructed by ChannelsAggregationRecording and is responsible for validating compatibility of recordings (for example sampling frequency, channel properties) and for exposing a unified recording API. No data copying or additional side effects are guaranteed by this function itself; data access behavior depends on the underlying recordings and the ChannelsAggregationRecording implementation.
    
    Raises:
        TypeError: If provided inputs are of incorrect types (for example recording_list_or_dict is not a list or dict, or renamed_channel_ids is not array-like) the underlying ChannelsAggregationRecording constructor may raise a TypeError.
        ValueError: If renamed_channel_ids length does not match the total aggregated channel count, if there are irreconcilable channel id conflicts, or if recordings have incompatible properties required by ChannelsAggregationRecording, a ValueError may be raised.
    """
    from spikeinterface.core.channelsaggregationrecording import aggregate_channels
    return aggregate_channels(
        recording_list_or_dict,
        renamed_channel_ids,
        recording_list
    )


################################################################################
# Source: spikeinterface.core.core_tools.check_paths_relative
# File: spikeinterface/core/core_tools.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_core_core_tools_check_paths_relative(
    input_dict: dict,
    relative_folder: str
):
    """spikeinterface.core.core_tools.check_paths_relative checks whether all filesystem paths contained in a dictionary that describes a BaseExtractor can be converted to relative paths with respect to a given folder.
    
    This function is used in the SpikeInterface framework (a unified framework for spike sorting) to decide if a dataset description (for example the dict returned by BaseExtractor.to_dict()) can be made portable by rewriting absolute file paths as paths relative to a chosen folder. It examines each path discovered in input_dict (these typically refer to recording files, probe files, or other data artifacts referenced by an extractor) and returns True only if every path can be expressed relative to relative_folder according to the same rules used by the library (URL and remote paths are excluded, Windows drive mismatches prevent relativity, and the Path.relative_to operation must succeed). The function performs checks such as string-based URL detection ("http" substring), remote-path detection via is_path_remote, drive equality for Windows paths (including UNC semantics where the host/share is treated as the drive), and a final attempt to compute a relative path using the library's _relative_to helper. The function does not modify files or the input_dict.
    
    Args:
        input_dict (dict): A dictionary describing a BaseExtractor obtained by BaseExtractor.to_dict(). In the SpikeInterface domain this dict encodes metadata and file references for recordings or sortings; the function will extract any file-system-like paths from this dict (via the internal _get_paths_list) and evaluate whether each such path can be made relative to relative_folder for the purpose of making the extractor description portable across environments.
        relative_folder (str or Path): The candidate base folder to which paths should be made relative. The function will convert this argument to a Path and call resolve().absolute() on it before testing. In practice this is the folder you intend to use as the root of a portable dataset (for example a project or repository folder).
    
    Returns:
        bool: True if every path found in input_dict can be converted to a path relative to relative_folder according to the checks performed; False if at least one path cannot be made relative. A False result indicates one or more of the following practical conditions: a path was detected as an HTTP URL (contains "http"), a path was detected as a remote path (is_path_remote returned True, e.g., s3 or other remote storage), a Windows path exists on a different drive than relative_folder (drive mismatch, including UNC semantics), or computing the relative path raised a ValueError (the path and relative_folder do not share a common ancestry according to Path.relative_to logic). Note that the function returns a boolean only and does not change input_dict or the filesystem.
    
    Behavioral details, side effects, and failure modes:
        The function gathers candidate paths from input_dict using the internal helper _get_paths_list. It treats any path string containing the substring "http" as a URL and marks it as not convertible. It uses is_path_remote to detect remote/virtual filesystems; remote paths are considered not convertible. Both relative_folder and candidate paths are converted to pathlib.Path objects; relative_folder is normalized with resolve().absolute() before checks. On Windows, if a candidate path is a WindowsPath and relative_folder is a WindowsPath, the function compares their drive attributes; if drives differ the path is considered not convertible (this preserves correct handling of drive letters and UNC shares). The function then attempts to compute a relative path via the library's _relative_to helper; if this raises ValueError the candidate path is considered not convertible. The function returns True only when no candidate paths were marked not convertible. The function itself does not modify the input_dict or any files.
    
        Note on exceptions: resolve().absolute() is invoked on relative_folder and, for drive checks, on candidate paths; depending on the Python/runtime environment this may access the filesystem and can raise exceptions such as FileNotFoundError or OSError if strict resolution fails or the underlying OS reports an error. Those exceptions are not caught inside the function and will propagate to the caller. The function only catches ValueError raised when attempting to compute a relative path via _relative_to. If input_dict contains no paths (for example an empty extractor dict), the function will return True (there were no paths preventing relativity).
    """
    from spikeinterface.core.core_tools import check_paths_relative
    return check_paths_relative(input_dict, relative_folder)


################################################################################
# Source: spikeinterface.core.core_tools.convert_bytes_to_str
# File: spikeinterface/core/core_tools.py
# Category: valid
################################################################################

def spikeinterface_core_core_tools_convert_bytes_to_str(byte_value: int):
    """Convert a number of bytes to a human-readable string using IEC binary prefixes.
    
    This utility converts an integer count of bytes into a formatted string that is easier for humans to read and compare in the context of SpikeInterface operations (for example, when reporting file sizes, memory footprints of recordings or processed datasets, export sizes for Phy/reports, or GUI displays). The function selects an appropriate IEC binary unit from bytes (B), kibibytes (KiB = 1024 B), mebibytes (MiB = 1024 KiB), gibibytes (GiB), up to tebibytes (TiB) and formats the numeric value with two decimal places followed by a single space and the unit (for example, "1.00 KiB"). This behavior matches how SpikeInterface presents storage and memory quantities when reading/writing many extracellular file formats, exporting reports, or displaying dataset sizes in the user interface.
    
    Args:
        byte_value (int): The integer number of bytes to convert. In SpikeInterface this represents a file or memory size measured in bytes (for example the size of a recording file or an in-memory buffer). The function expects an int; passing a non-int type that does not support numeric comparison with integers (for example, a string) will raise a TypeError during execution. Negative integers are processed numerically (they will produce a negative formatted value) but negative sizes are semantically unusual for file/memory sizes in the SpikeInterface domain.
    
    Returns:
        str: A human-readable string representation of the input byte count. The numeric portion is formatted with two decimal places and separated from the IEC binary unit by a single space (examples: "512.00 B", "1.00 KiB", "1.00 MiB"). Units are chosen by repeatedly dividing by 1024 until the value is less than 1024 or the largest unit (TiB) is reached; values equal to or larger than 1024**4 are reported in TiB. There are no side effects (the function does not perform any I/O). Note that the implementation converts the value to a floating-point number during scaling and formatting; extremely large integers may be subject to floating-point rounding or precision limitations when represented in the output string.
    """
    from spikeinterface.core.core_tools import convert_bytes_to_str
    return convert_bytes_to_str(byte_value)


################################################################################
# Source: spikeinterface.core.core_tools.convert_seconds_to_str
# File: spikeinterface/core/core_tools.py
# Category: valid
################################################################################

def spikeinterface_core_core_tools_convert_seconds_to_str(
    seconds: float,
    long_notation: bool = True
):
    """spikeinterface.core.core_tools.convert_seconds_to_str converts a duration given in seconds into a human-readable string used throughout SpikeInterface to display recording durations, preprocessing and sorting runtimes, metric computation times, and other time-related values in logs and reports.
    
    Args:
        seconds (float): The duration expressed in seconds. This numeric input is formatted as the primary representation with a thousands separator and two decimal places (for example, 1,234.56s). The function expects a finite floating-point value; if the value cannot be formatted as a float (for example, None or a non-numeric object), Python's built-in formatting will raise a TypeError or ValueError. The numeric value determines which additional unit annotation (milliseconds, minutes, hours, or days) is appended when long_notation is True.
        long_notation (bool): Whether to append an additional unit representation to the primary seconds string. Defaults to True. When True, the function appends one supplementary unit in parentheses according to thresholds used in SpikeInterface for clearer human interpretation: if seconds < 1.0 the function appends milliseconds (seconds * 1000); if seconds < 60 no extra unit is appended because seconds are already the most appropriate unit; if seconds < 3600 it appends minutes (seconds / 60); if seconds < 172800 (2 days) it appends hours (seconds / 3600); otherwise it appends days (seconds / 86400). When False, the function returns only the formatted seconds representation (with two decimals and thousands separator) and no parenthetical annotation.
    
    Returns:
        str: A human-readable string representing the duration. The returned string always includes the primary seconds representation formatted with a thousands separator and two decimal places followed by "s" (for example, "0.50s", "65.00s", "1,234.56s"). If long_notation is True, one additional parenthetical unit annotation is appended according to the thresholds described above (for example, "0.50s (500.00 ms)", "90.00s (1.50 minutes)", "7200.00s (2.00 hours)", "172800.00s (2.00 days)"). The function has no side effects and does not modify external state.
    """
    from spikeinterface.core.core_tools import convert_seconds_to_str
    return convert_seconds_to_str(seconds, long_notation)


################################################################################
# Source: spikeinterface.core.core_tools.convert_string_to_bytes
# File: spikeinterface/core/core_tools.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for convert_string_to_bytes because the docstring has no description for the argument 'memory_string'
################################################################################

def spikeinterface_core_core_tools_convert_string_to_bytes(memory_string: str):
    """Convert a memory size string to the corresponding number of bytes.
    
    This function is spikeinterface.core.core_tools.convert_string_to_bytes and is used within the SpikeInterface framework to translate human-readable memory specifications (commonly provided when configuring caching sizes, chunking windows, memory limits for sorting backends, exporters and other memory-sensitive components) into an integer byte count that the codebase can use for allocation, comparisons, and configuration.
    
    Args:
        memory_string (str): Memory size string containing a numeric value immediately followed by a unit suffix. The function checks the last two characters first and, if that two-character suffix is not present in the module's internal _exponents mapping, falls back to the last single character. Examples of acceptable inputs (based on existing usage in the codebase) include "1G", "512Mi", and "2T". The numeric portion may be an integer or a floating-point value (for example, "1.5G"). The string must not be empty and must include a recognized suffix; the function does not strip leading or trailing whitespace, so callers should pass a trimmed string. In the SpikeInterface domain this argument is typically supplied by user configuration or internal helpers when specifying memory limits for preprocessing, sorting, or export steps.
    
    Returns:
        int: Number of bytes computed as int(float(numeric_value) * _exponents[suffix]). The multiplication uses the module's internal _exponents mapping to map suffixes to their byte multipliers; the final int() conversion truncates any fractional bytes (i.e., rounds toward zero). The returned integer is meant to be used directly for memory allocation, comparisons, and configuration values elsewhere in SpikeInterface.
    
    Behavior, side effects, and failure modes:
        The function has no side effects. It extracts the suffix by first examining memory_string[-2:] and falling back to memory_string[-1], then converts the preceding characters to float and multiplies by the corresponding exponent. It will raise an AssertionError with message "Unknown suffix: {suffix}" if the extracted suffix is not a key in the internal _exponents mapping. It will raise a ValueError if the numeric portion cannot be converted to float, an IndexError if memory_string is empty (or too short to contain a numeric part plus suffix), and will likely raise a TypeError if a non-str is passed. Callers should ensure the string is trimmed and uses a suffix supported by the module's _exponents mapping.
    """
    from spikeinterface.core.core_tools import convert_string_to_bytes
    return convert_string_to_bytes(memory_string)


################################################################################
# Source: spikeinterface.core.core_tools.extractor_dict_iterator
# File: spikeinterface/core/core_tools.py
# Category: fix_docstring
# Reason: Schema parsing failed: ("Couldn't parse this type hint, likely due to a custom class or object: ", typing.Generator[spikeinterface.core.core_tools.extractor_dict_element, NoneType, NoneType])
################################################################################

def spikeinterface_core_core_tools_extractor_dict_iterator(extractor_dict: dict):
    """Iterator for recursive traversal of a dictionary produced by extractors in the SpikeInterface framework. This function explores the nested mapping/list structure returned by BaseExtractor.to_dict() and yields a sequence of extractor_dict_element named tuples that describe every leaf value found in the structure. Each yielded element includes the actual leaf value, a name (the last dict key or the propagated list name), and an access_path tuple that records the sequence of dict keys and list indices required to reach that value from the top-level extractor_dict. In the SpikeInterface domain this is used to inspect, serialize, or process all scalar or non-dict/list entries stored by an extractor (for example metadata, numeric parameters, or paths) without copying the underlying data.
    
    Args:
        extractor_dict (dict): The input mapping to traverse. In SpikeInterface this is expected to be the dictionary returned by BaseExtractor.to_dict(), i.e., a nested structure of dicts and lists representing an Extractor's serializable state. The function treats dict instances as nested mappings and list instances as ordered sequences whose indices are appended to the access_path. Non-dict, non-list objects are treated as leaves and yielded as values. Keys in dicts are appended to access_path as-is (typically strings used by Extractor.to_dict()); list indices are appended as integers. The function does not modify extractor_dict.
    
    Returns:
        Generator[extractor_dict_element, None, None]: A generator that yields extractor_dict_element named tuples for each leaf in extractor_dict. Each extractor_dict_element has three fields: value (the leaf object found in the structure), name (the dict key corresponding to the leaf or the propagated name for items inside lists), and access_path (a tuple of dict keys and list indices showing the path from the top-level extractor_dict to the value). The generator is lazy and yields references to the original objects (no deep copy), preserving insertion order for dict traversal (Python 3.7+). Traversal visits only leaves (objects that are neither dict nor list). Side effects: none on the input structure. Failure modes and important behaviors: deep or cyclic structures can cause RecursionError or infinite recursion; the function only recognizes instances of dict and list as containers (tuples, sets, or other sequence types are treated as leaves); values that are not serializable are still yielded as-is; traversal relies on Python dict iteration order (insertion order) to determine yield order.
    """
    from spikeinterface.core.core_tools import extractor_dict_iterator
    return extractor_dict_iterator(extractor_dict)


################################################################################
# Source: spikeinterface.core.core_tools.is_dict_extractor
# File: spikeinterface/core/core_tools.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for is_dict_extractor because the docstring has no description for the argument 'd'
################################################################################

def spikeinterface_core_core_tools_is_dict_extractor(d: dict):
    """is_dict_extractor(d)
    Determines whether a given Python dict describes an extractor object in the SpikeInterface framework.
    
    This function performs a lightweight, structural check used throughout SpikeInterface when handling serialized extractor descriptions (for example, when saving/loading RecordingExtractor or SortingExtractor metadata, exporting extractor descriptions to disk, or passing extractor definitions between processes). Concretely, it verifies that the input is a dict and that it contains the four keys expected by SpikeInterface extractor descriptions: "module", "class", "version", and "annotations". This check is intentionally shallow: it tests type and key presence only and does not validate the types or contents of the individual values associated with those keys (for example it does not parse or check the format of "version" or the structure of "annotations").
    
    Args:
        d (dict): The object to test. This should be a Python dict that may represent a serialized extractor description used by SpikeInterface. The function first checks that d is an instance of dict; if it is not, the function will return False. If d is a dict, the function then checks for the presence of the exact keys "module", "class", "version", and "annotations". The practical role of this parameter is to provide a candidate serialized extractor representation so callers (such as serializers, deserializers, file readers, or inter-process communication code in spike sorting pipelines) can quickly decide whether the object likely encodes an extractor.
    
    Returns:
        bool: True if and only if d is a dict and contains the keys "module", "class", "version", and "annotations". A return value of True indicates that d structurally matches the minimal extractor description shape used by SpikeInterface and is therefore a candidate for further processing (e.g., reconstruction of an extractor object). A return value of False indicates that either d is not a dict or one or more of the required keys are missing. This function has no side effects and does not raise exceptions for typical inputs; it performs only type and key-presence checks.
    """
    from spikeinterface.core.core_tools import is_dict_extractor
    return is_dict_extractor(d)


################################################################################
# Source: spikeinterface.core.core_tools.make_paths_absolute
# File: spikeinterface/core/core_tools.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_core_core_tools_make_paths_absolute(input_dict: dict, base_folder: str):
    """spikeinterface.core.core_tools.make_paths_absolute: Recursively convert path-like entries in a BaseExtractor description dict into absolute POSIX paths relative to a given base folder.
    
    This function is used in the SpikeInterface framework (a unified framework for spike sorting) to post-process dictionaries produced by BaseExtractor.to_dict() so that any entries representing filesystem paths become absolute and machine-independent POSIX strings. It walks the input extractor dictionary, identifies entries that represent paths using extractor_dict_iterator(...) together with element_is_path(...), resolves each candidate path relative to base_folder, and writes the resolved POSIX path back into a deep copy of the input dictionary using set_value_in_extractor_dict(...). Only entries whose resolved absolute path exists on the filesystem are replaced; non-existing resolved paths are left unchanged. The original input_dict is not modified because the function operates on a deepcopy and returns a modified copy.
    
    Args:
        input_dict (dict): A dictionary describing an extractor obtained from BaseExtractor.to_dict(). In the SpikeInterface domain this dict typically encodes metadata and references to recorded data files or auxiliary resources; the function will examine this structure to locate elements that represent filesystem paths (as identified by extractor_dict_iterator and element_is_path) and may replace those elements with absolute POSIX path strings.
        base_folder (str or pathlib.Path): The filesystem folder used as the reference for resolving relative paths. Each path-like element found in input_dict will be combined with this base_folder (using Path(base_folder) / element.value) and resolved to an absolute path. The function accepts a string or a pathlib.Path as documented in the original implementation.
    
    Returns:
        output_dict (dict): A deep copy of input_dict with path-like elements replaced by absolute POSIX-formatted strings when and only when the resolved absolute path exists on the filesystem. The function returns a new dict so callers can rely on input_dict remaining unchanged. If a candidate path resolves to a location that does not exist, that element is not modified in the returned dict.
    
    Behavior and side effects:
        The function uses pathlib.Path to construct and resolve paths and converts replaced values to POSIX-style strings via Path.as_posix() so that returned paths are consistent across platforms (useful when saving extractor descriptions for later reloading or sharing across systems). It identifies which keys/values to treat as paths by using internal helpers extractor_dict_iterator(...) and element_is_path(...), and writes replacements using set_value_in_extractor_dict(...). It always operates on a deepcopy of the input to avoid mutating the caller's object.
    
    Failure modes and exceptions:
        If input_dict is not a dict in the expected extractor format, the helper functions may raise TypeError or other exceptions; these exceptions are raised from the iterator/helper functions and are not caught here. If base_folder cannot be converted to a pathlib.Path or contains invalid characters, a TypeError or OSError from pathlib.Path may be raised during resolution. Path resolution is attempted for each candidate element; however, replacement in the returned dict only occurs when Path(resolved).exists() is True. The function does not create filesystem entries.
    """
    from spikeinterface.core.core_tools import make_paths_absolute
    return make_paths_absolute(input_dict, base_folder)


################################################################################
# Source: spikeinterface.core.core_tools.measure_memory_allocation
# File: spikeinterface/core/core_tools.py
# Category: valid
################################################################################

def spikeinterface_core_core_tools_measure_memory_allocation(
    measure_in_process: bool = True
):
    """Measure memory allocation at a single point in time for use in spike sorting workflows.
    
    This utility is used within SpikeInterface, a framework for extracellular recording processing and spike sorting, to quantify memory consumption either for the current Python process or for the whole system. Measuring memory allocation is practically significant when loading large recordings, running preprocessors or spike sorters, or benchmarking memory usage of sorting pipelines as described in the SpikeInterface README. The function performs a one-time query using the psutil package and does not modify program state.
    
    Args:
        measure_in_process (bool): If True (default), measure memory allocated by the current Python process only. In this mode the function queries psutil.Process().memory_info().rss, returning the resident set size (RSS) in bytes—which is the portion of memory occupied in RAM and excludes swapped-out pages. If False, measure system-level memory usage by computing psutil.virtual_memory().total - psutil.virtual_memory().available, which yields the approximate number of bytes currently used by the whole system (this accounts for used memory including caches and buffers according to psutil semantics). Choose process-level measurement when profiling a specific sorter or data-loading routine; choose system-level measurement when assessing overall system memory pressure before launching heavy computations.
    
    Returns:
        float: The measured memory amount expressed in bytes as a floating-point number. When measure_in_process is True this is the process RSS (bytes). When False this is the system used memory computed as total minus available (bytes). Callers that need human-friendly units should convert the returned value (for example, divide by 1024**2 to obtain mebibytes). Note that the implementation uses psutil and performs a single instantaneous read; the result represents a snapshot and can change immediately after the call.
    
    Side effects and failure modes:
        This function imports and calls into the psutil library. If psutil is not installed, an ImportError will be raised at import time. Calls into psutil may also raise platform- or permission-related exceptions (for example, psutil.AccessDenied or psutil.NoSuchProcess in rare race conditions), which are propagated to the caller. The function does not allocate persistent resources or modify process memory; it only queries existing system/process state.
    """
    from spikeinterface.core.core_tools import measure_memory_allocation
    return measure_memory_allocation(measure_in_process)


################################################################################
# Source: spikeinterface.core.core_tools.read_python
# File: spikeinterface/core/core_tools.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_core_core_tools_read_python(path: str):
    """Parses a Python script file and returns a metadata dictionary built from the top-level names defined in that file. This function is used in SpikeInterface to load Python-based configuration or metadata files (for example, small modules that define recording parameters, channel maps, preprocessing parameters, or other dataset-specific Python objects) into a programmatic dictionary that downstream spike-sorting components can consume.
    
    The function reads the file at path, applies a small text transformation that converts simple numeric calls to range(...) into list(range(...)) to preserve expected sequence semantics, then executes the resulting code using six.exec_ in an isolated execution environment to collect the top-level names and values defined in the file. After execution, all top-level names are converted to lowercase and returned in a dictionary. Because the function executes arbitrary Python code, it can raise the same runtime or syntax errors as executing that code directly and may have side effects (file I/O, imports, network access, etc.). Use this function only on trusted files.
    
    Args:
        path (str or Path): Path to the Python file to parse. This parameter accepts a string file path or a pathlib.Path object pointing to an existing file on disk. The function resolves the path to an absolute path and asserts that it is a file; if the path does not point to a file the function raises an AssertionError.
    
    Returns:
        dict: A dictionary containing the parsed file's top-level names mapped to their values, with all keys converted to lowercase. The values are the actual Python objects created by executing the file (for example, numbers, strings, lists, dicts, functions, classes, or third-party objects such as numpy arrays) and can be used by SpikeInterface components that expect programmatic configuration/metadata. Key collisions that differ only by case are resolved by keeping the last-defined value for the lowercase key.
    
    Behavior and side effects:
        The function opens and reads the file at path. It performs a regex substitution that replaces simple numeric literal uses of range(...) (pattern matching digits and commas inside the parentheses) with list(range(...)), so occurrences such as range(0,10) become list(range(0,10)). This substitution is limited to the regex pattern used and will not transform uses of range with variables, expressions, or complex arguments.
    
        The (possibly modified) file contents are executed via six.exec_(contents, {}, metadata), where an empty globals mapping and the metadata dict are supplied as the execution environment. Top-level names assigned during execution are captured in the metadata dict. After execution the function lowercases all keys in the metadata dict (metadata = {k.lower(): v for (k, v) in metadata.items()}) and returns that mapping.
    
    Failure modes and exceptions:
        If path does not point to an existing file, the assertion path.is_file() will fail and an AssertionError is raised. Opening the file may raise OSError or related I/O exceptions. The exec_ call may raise SyntaxError, ImportError, NameError, RuntimeError, or any exception raised by the code in the file; those propagate to the caller. Because the file is executed, malicious or buggy code can perform arbitrary side effects (I/O, process spawning, network access, modifying global state), so only trusted files should be parsed with this function.
    
    Security and usage notes:
        Executing untrusted Python files is dangerous. Although the code executes with an empty globals mapping, builtins and imports are still available in the execution environment and the executed code can perform side effects. Prefer safer, structured formats (for example JSON or YAML) for untrusted configuration data when possible. In SpikeInterface, this function is intended for convenience when datasets or toolchains provide configuration as Python scripts.
    """
    from spikeinterface.core.core_tools import read_python
    return read_python(path)


################################################################################
# Source: spikeinterface.core.core_tools.write_python
# File: spikeinterface/core/core_tools.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_core_core_tools_write_python(path: str, dict: dict):
    """spikeinterface.core.core_tools.write_python saves a Python-style assignment file representing a simple dictionary of configuration or metadata values used in the SpikeInterface spike-sorting framework.
    
    This function writes each key/value pair from the provided dictionary as a Python assignment statement to a plain text file at the given path. It is commonly used in SpikeInterface to persist lightweight configuration dictionaries, sorter parameter sets, or small pieces of recording metadata in a human-readable Python format that can be inspected or executed (with caution) by downstream tools. The function performs minimal formatting: string values are wrapped in single quotes, Windows-style file paths are written as raw string literals when the key name suggests a path, and all other values are written using their str() representation. The file at the target path is opened for writing and will be created or overwritten.
    
    Args:
        path (str or Path): Path to save file. This argument is the filesystem location where the function will create or overwrite a plain text file containing Python assignment statements. The implementation accepts either a string path or a pathlib.Path-like object; the path will be opened with Path(path).open("w") so typical I/O errors (for example, permission denied or nonexistent directory) will propagate from the underlying filesystem call.
        dict (dict): dictionary to save. This mapping's items are written one-per-line in the output file as "key = value". Keys are written using str(key) and values are formatted as follows: if a value is an instance of str and does not already start with a single quote character ("'"), the value will be wrapped in single quotes; if the key name contains the substring "path" and the runtime platform contains "win", the value is written as a raw string literal (r'...') to preserve backslashes in Windows file paths. For all other values, the value is written using str(value). Practical consequences: list and numeric values are written with their str() representation (which typically yields valid Python literal syntax), but arbitrary objects will be serialized using str() which may not produce a valid or safe Python literal. Keys that are not valid Python identifiers are still written but may produce an assignment that is not valid Python code.
    
    Returns:
        None: The function does not return a value. Instead, it has the side effect of creating or overwriting a text file at the provided path containing one Python assignment per dictionary entry. If the write operation fails, the underlying exceptions from pathlib.Path.open or the file write operations (for example, OSError/IOError) will propagate to the caller. Users should validate that the produced file meets their safety and syntactic requirements before importing or executing it.
    """
    from spikeinterface.core.core_tools import write_python
    return write_python(path, dict)


################################################################################
# Source: spikeinterface.core.generate.clean_refractory_period
# File: spikeinterface/core/generate.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_core_generate_clean_refractory_period(
    times: numpy.ndarray,
    refractory_period: float
):
    """Remove spike times that violate a refractory period in a spike train.
    
    This function is used in spike sorting and spike-train post-processing to enforce a minimum time interval (the refractory period) between consecutive spike events. Given a 1-D array of spike event times (for example, sample indices or times in seconds or milliseconds), the function returns a new numpy.ndarray containing only those spike times that remain after removing events that occur too soon after a preceding event. The algorithm is greedy and deterministic: it first sorts the input times in ascending order, then iteratively removes the later event in any pair whose inter-event interval is less than or equal to the specified refractory period, repeating until no violations remain. In practice this is used to discard duplicate or temporally implausible detections that would violate a neuron's physiological or algorithmic refractory constraint.
    
    Args:
        times (numpy.ndarray): 1-D numeric array of spike event times. These are the times or sample indices of detected spikes in a recording. The array may be unsorted on input; the function sorts it internally and returns a sorted array. The units of times must match the units of refractory_period (for example, both in samples, seconds, or milliseconds). An empty array is accepted and returned unchanged.
        refractory_period (float): Minimum allowed interval between consecutive spikes, expressed in the same units as times (samples, seconds, or milliseconds). The function treats pairs with inter-spike interval less than or equal to this value as violating the refractory period and removes the later spike in the pair. A non-negative value is expected for typical use; if a negative value is provided the function will sort and return the input times without removing events (because sorted inter-event differences are non-negative).
    
    Returns:
        numpy.ndarray: A 1-D numpy array of spike times after removing spikes that violate the refractory period. The returned array is sorted in ascending order and uses the same units as the input times. The function does not modify the caller's input array in-place (it works on a sorted copy internally) and instead returns a new array containing the filtered spike times.
    
    Behavior, side effects, and failure modes:
        The function sorts the input times and repeatedly removes the later element of any adjacent pair whose difference is <= refractory_period until no such pairs remain; this effectively keeps the earliest spike in any cluster of closely spaced events. If times is empty the function returns it immediately. If times is not a 1-D numeric numpy.ndarray, or contains NaN or non-finite values, operations such as sorting and numeric comparisons may raise exceptions (TypeError, ValueError, or propagate NaNs), so callers should ensure valid numeric input. The function assumes units consistency between times and refractory_period; mismatched units (for example, times in seconds but refractory_period given in samples) will produce incorrect filtering. The algorithm has worst-case runtime proportional to the number of spikes times the number of iterations needed to remove violations in dense bursts; in typical sparse spike trains this is efficient.
    """
    from spikeinterface.core.generate import clean_refractory_period
    return clean_refractory_period(times, refractory_period)


################################################################################
# Source: spikeinterface.core.generate.create_sorting_npz
# File: spikeinterface/core/generate.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_core_generate_create_sorting_npz(num_seg: int, file_path: str):
    """Create a NPZ-format sorting file used for tests and examples within the SpikeInterface framework.
    
    This function builds a minimal, synthetic sorting dataset and writes it to disk using numpy.savez at the given file_path. The produced NPZ file mirrors the small, fixed-format convention used by SpikeInterface test utilities: it contains a unit identifier array, a stored number-of-segments array, a sampling frequency array, and for each segment two arrays named spike_indexes_seg{seg_index} and spike_labels_seg{seg_index} that represent spike times (sample indices) and corresponding unit labels. This helper is intended to generate a predictable, toy sorting file for downstream consumers in the spike sorting pipeline (for example, loader tests, examples, or toy benchmarking), not to produce realistic experimental data.
    
    Args:
        num_seg (int): The number of segments to include in the generated NPZ file. This controls how many pairs of arrays named spike_indexes_seg{seg_index} and spike_labels_seg{seg_index} are written, where seg_index runs from 0 to num_seg-1. If num_seg is zero or negative the function will write no per-segment arrays (the function will still write the other fixed arrays). The function does not validate beyond using range(num_seg) and therefore negative values produce an empty segment loop rather than raising.
        file_path (str): Path (as a string) where the NPZ file will be written. The function calls numpy.savez(file_path, **d) and will overwrite an existing file at that path. Typical failure modes are file system errors such as OSError or PermissionError if the path is invalid or not writable; a TypeError may occur if a non-string incompatible object is passed.
    
    Behavior details, side effects, defaults, and failure modes:
        The function constructs a dictionary of numpy arrays and saves it to disk. The dictionary contents and their practical meaning in the spike-sorting domain are:
        - "unit_ids": numpy.ndarray of dtype int64 with values [0, 1, 2]. These are the integer identifiers for simulated units (neurons) used by the toy sorting; downstream code uses these to enumerate units.
        - "num_segment": numpy.ndarray of dtype int64 containing a single value. In this implementation the value is set to numpy.array([2], dtype="int64") (hardcoded) rather than being derived from the num_seg argument; therefore the stored "num_segment" value may not match the num_seg parameter passed to the function. Users relying on that field to reflect num_seg should be aware of this mismatch.
        - "sampling_frequency": numpy.ndarray of dtype float64 containing a single value 30000.0. This represents the sampling rate in Hertz used by the synthetic spikes; downstream waveform/time conversion assumes this sampling frequency.
        - For each segment index seg_index in range(num_seg) two arrays are written:
          - "spike_indexes_seg{seg_index}": a numpy.ndarray with values numpy.arange(0, 1000, 10). These are spike time sample indices (0, 10, 20, ..., 990) used as a deterministic, evenly spaced spike train for the toy dataset.
          - "spike_labels_seg{seg_index}": a numpy.ndarray of dtype int64 the same length as the corresponding spike_indexes array. Labels are assigned cyclically so that entries 0::3 are 0, 1::3 are 1, and 2::3 are 2, producing a repeating assignment of the three unit_ids to spikes.
        All arrays are stored with the dtypes set in the function (unit_ids and labels as int64, sampling_frequency as float64). The function writes the file using numpy.savez, which will create a .npz archive at file_path and overwrite an existing file with that name.
    
        Practical significance in the SpikeInterface domain: this deterministic, small NPZ file is suitable for unit tests, examples, and tutorials within SpikeInterface where predictable spike times and labels are required to exercise loaders, sorters, post-processing, or visualization components without relying on large experimental recordings.
    
        Limitations and failure modes: the created dataset is synthetic and minimal; it does not model realistic spike timing, noise, or channel geometry. The "num_segment" field is hardcoded to 2 in the saved file and may thus be inconsistent with the actual number of per-segment arrays written when num_seg differs from 2. Filesystem errors (permission, directory not found, disk full) will be raised by numpy.savez; invalid types for file_path may raise TypeError. The function has no explicit return value.
    
    Returns:
        None: The function does not return a Python value. Side effect: a .npz file is created at file_path containing the arrays described above (unit_ids, num_segment, sampling_frequency, and for each segment spike_indexes_seg{seg_index} and spike_labels_seg{seg_index}).
    """
    from spikeinterface.core.generate import create_sorting_npz
    return create_sorting_npz(num_seg, file_path)


################################################################################
# Source: spikeinterface.core.generate.generate_single_fake_waveform
# File: spikeinterface/core/generate.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_core_generate_generate_single_fake_waveform(
    sampling_frequency: float = None,
    ms_before: float = 1.0,
    ms_after: float = 3.0,
    negative_amplitude: float = -1,
    positive_amplitude: float = 0.15,
    depolarization_ms: float = 0.1,
    repolarization_ms: float = 0.6,
    recovery_ms: float = 1.1,
    smooth_ms: float = 0.05,
    dtype: str = "float32"
):
    """generate a single-channel synthetic spike waveform composed of three exponential phases (depolarization, repolarization, recovery), then apply Gaussian smoothing and align the negative peak to the sample index corresponding to ms_before. This helper is intended for use in SpikeInterface (a unified framework for spike sorting) to create simple, repeatable test waveforms for waveform extraction, sorter testing, tutorials, and unit tests where a realistic but parameterizable extracellular spike shape is required.
    
    Args:
        sampling_frequency (float): Sampling frequency in Hz used to convert time in milliseconds to integer sample counts. The function multiplies this value by ms_before and ms_after to compute the number of samples before and after the spike peak (nbefore and nafter). Although the signature default is None, a numeric float must be provided in practice; passing None or a non-numeric value will cause a TypeError when the function attempts arithmetic with sampling_frequency. This parameter determines waveform temporal resolution and directly affects all derived integer sample counts (nbefore, nafter, ndepo, nrepol, nrefac).
        ms_before (float): Duration in milliseconds of the waveform segment preceding the spike peak (default 1.0 ms). This is used with sampling_frequency to compute nbefore = int(sampling_frequency * ms_before / 1000.0). The function asserts that ms_before > depolarization_ms and that the computed ndepo is less than nafter; if these conditions are not met an AssertionError is raised, indicating the provided ms_before is too short for the requested depolarization phase.
        ms_after (float): Duration in milliseconds of the waveform segment following the spike peak (default 3.0 ms). Used with sampling_frequency to compute nafter = int(sampling_frequency * ms_after / 1000.0). The function asserts that ms_after > depolarization_ms + repolarization_ms and later asserts that the sum of the repolarization and recovery sample counts is less than nafter; violating these constraints raises an AssertionError with messages such as "ms_after is too short".
        negative_amplitude (float): Peak negative amplitude (most hyperpolarized point) of the generated waveform, expressed in the same arbitrary amplitude units used by downstream processing (default -1). This value is used as the start or end amplitude for the exponential segments that create the negative deflection and influences the overall waveform scaling prior to smoothing and normalization.
        positive_amplitude (float): Positive (depolarizing) overshoot amplitude following the negative peak (default 0.15). This sets the intermediate positive plateau amplitude used by the repolarization and recovery exponentials and affects the waveform shape used during sorting algorithm tests or visualization.
        depolarization_ms (float): Duration in milliseconds allocated to the depolarization exponential that forms the leading (negative) phase of the spike (default 0.1 ms). The code converts this into ndepo = int(depolarization_ms * sampling_frequency / 1000.0) samples and uses an exponential growth helper to interpolate between baseline and negative_amplitude. If depolarization_ms is longer than ms_before (after conversion), an AssertionError is raised.
        repolarization_ms (float): Duration in milliseconds allocated to the repolarization exponential that transitions from the negative peak toward positive_amplitude (default 0.6 ms). Converted to nrepol samples and used with an exponential growth helper. The function requires ms_after > depolarization_ms + repolarization_ms at the start and will assert that nrefac + nrepol < nafter later to ensure there is room for the recovery phase.
        recovery_ms (float): Duration in milliseconds allocated to the recovery exponential that returns the waveform from positive_amplitude back to baseline (default 1.1 ms). Converted to nrefac = int(recovery_ms * sampling_frequency / 1000.0) samples and filled using an exponential growth helper. If the sum of nrepol and nrefac is not strictly less than nafter an AssertionError is raised ("ms_after is too short").
        smooth_ms (float): Width parameter in milliseconds used to build a Gaussian smoothing kernel (default 0.05 ms). Internally converted to kernel size in samples as smooth_size = smooth_ms / (1 / sampling_frequency * 1000.0), a symmetric kernel over bins = arange(-n, n+1) with n = int(smooth_size * 4). The waveform is convolved with this normalized Gaussian to reduce sharp transitions; after smoothing the waveform is renormalized so its absolute peak amplitude matches the pre-smoothing peak.
        dtype (str): NumPy dtype name used for the returned waveform array (default "float32"). The function constructs the working array with numpy.zeros(width, dtype=dtype) and returns an array of this dtype. The string must be a valid NumPy dtype name recognized by numpy.zeros.
    
    Returns:
        numpy.ndarray: A one-dimensional NumPy array of length width = nbefore + nafter containing the synthetic single-channel spike waveform samples with the requested dtype. The negative peak is aligned to sample index nbefore (the transition point between the "before" and "after" segments) after internal possible shifts performed to correct for smoothing-induced peak displacement. The returned array is suitable for use as a single-channel template in waveform extraction, visualization, or as synthetic input to downstream spike sorting and quality-metric computations.
    
    Behavior and failure modes:
        The function procedurally composes the waveform by filling three contiguous segments (depolarization, repolarization, recovery) using an exponential growth helper, then applies Gaussian smoothing and rescales amplitudes to preserve the original absolute peak. It enforces temporal constraints via assertions: ms_after must be greater than depolarization_ms + repolarization_ms, and ms_before must be greater than depolarization_ms; additionally, the integer-sample conversions must leave space for each exponential segment (asserts with messages "ms_before is too short" or "ms_after is too short"). If sampling_frequency is None or non-numeric, arithmetic operations will raise a TypeError. The function has no external side effects (it does not modify global state or files) and is deterministic given the same inputs.
    """
    from spikeinterface.core.generate import generate_single_fake_waveform
    return generate_single_fake_waveform(
        sampling_frequency,
        ms_before,
        ms_after,
        negative_amplitude,
        positive_amplitude,
        depolarization_ms,
        repolarization_ms,
        recovery_ms,
        smooth_ms,
        dtype
    )


################################################################################
# Source: spikeinterface.core.generate.generate_snippets
# File: spikeinterface/core/generate.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_core_generate_generate_snippets(
    nbefore: int = 20,
    nafter: int = 44,
    num_channels: int = 2,
    wf_folder: str = None,
    sampling_frequency: float = 30000.0,
    durations: list = [10.325, 3.5],
    set_probe: bool = True,
    ndim: int = 2,
    num_units: int = 5,
    empty_units: list = None,
    **job_kwargs
):
    """Generates a synthetic Snippets object together with a corresponding synthetic Sorting object for use in spike-sorting workflows, using helper generators in the SpikeInterface test utilities. This function is used to create small, controlled datasets (recording + sorting) and then extract waveform snippets around detected spike times. It is useful for unit tests, examples, tutorials, and any situation where an in-memory or on-disk set of waveform snippets and a matching sorting are required in the SpikeInterface framework.
    
    Args:
        nbefore (int): Number of samples before the peak to include in each extracted snippet. This integer defines the left context of each waveform snippet relative to the detected spike time and is passed to snippets_from_sorting so the resulting snippet length equals nbefore + nafter samples. Typical use: set to the number of pre-peak samples needed to capture waveform onset.
        nafter (int): Number of samples after the peak to include in each extracted snippet. This integer defines the right context of each waveform snippet and, together with nbefore, determines the temporal window of each snippet used for downstream waveform analysis and visualization.
        num_channels (int): Number of channels in the synthetic recording. This controls how many channels the generated Recording object will have and therefore the channel dimension of returned snippets. In multi-channel spike-sorting contexts this represents the number of electrodes simulated.
        wf_folder (str): Optional path (string) to a folder where extracted waveform snippets will be saved on disk. If wf_folder is None (the default), snippets are kept in memory (no waveform files are written). If a path is provided, disk I/O operations occur and OSError or similar file-system exceptions may be raised if the folder is not writable.
        sampling_frequency (float): Sampling frequency of the synthetic recording and snippets, in Hertz. This float sets the time base for the recording and sorting objects and is used when converting between sample indices and time units.
        durations (list): List of floats describing the duration in seconds of each segment to generate (for example [10.325, 3.5] creates two segments of those lengths). The number of recording segments created equals len(durations); this affects the generated recording timeline and the sorting generation across segments.
        set_probe (bool): If True (default), the probe information (geometry and channel mapping) created on the synthetic Recording will be attached to the returned snippets object via snippets.set_probe(probe). When True, the returned snippets include probe metadata which is important for visualization and spatially-aware processing; when False, probe metadata is not attached.
        ndim (int): Number of spatial dimensions for the probe geometry in the generated Recording (for example 2 for planar probes). This controls the probe layout associated with the recording and therefore the probe that may be attached to snippets when set_probe is True.
        num_units (int): Number of units (putative neurons) to generate in the synthetic Sorting object. This integer controls how many unit IDs and associated spike trains are created by generate_sorting and thus how many rows/labels appear in the returned sorting.
        empty_units (list): Optional list of unit identifiers (as produced by generate_sorting) that should contain no spikes. When provided, these units are created in the sorting but with empty spike trains; useful to test handling of empty units in downstream pipelines.
        job_kwargs (dict): Additional keyword arguments forwarded to snippets_from_sorting. These job-related keyword arguments are used by snippets_from_sorting for task configuration (for example internal parallelization or processing options handled by that function). The keys and semantics are those accepted by snippets_from_sorting and are not interpreted here.
    
    Returns:
        snippets (NumpySnippets): A NumpySnippets object containing extracted waveform snippets organized per unit and per channel. The snippets contain waveform arrays with shape determined by (n_snippets, nbefore + nafter, num_channels) for each unit and include metadata; if wf_folder was provided, waveform data may be stored or cached on disk in that folder. The returned snippets object will have the probe attached if set_probe is True.
        sorting (NumpySorting): A NumpySorting object representing the synthetic spike sorting associated with the recording. This sorting provides unit IDs and spike times (in samples or seconds consistent with sampling_frequency) for each generated unit and segment; empty_units (if provided) will be present with empty spike lists.
    
    Behavior and side effects:
        This function calls generate_recording to create a synthetic Recording with the specified durations, num_channels, sampling_frequency, ndim, and set_probe behavior, then calls generate_sorting to create spike trains (num_units, empty_units, sampling_frequency, durations). It then calls snippets_from_sorting to extract waveform snippets using nbefore, nafter, and wf_folder and forwards any job_kwargs to that function. If set_probe is True the probe object from the generated recording is attached to the returned snippets via snippets.set_probe(probe). If wf_folder is specified, waveform data may be written to disk; ensure the path is writable to avoid filesystem errors.
    
    Defaults:
        If wf_folder is None, snippets remain in memory. Default durations create two segments of lengths [10.325, 3.5] seconds. Default sampling_frequency is 30000.0 Hz. Default nbefore and nafter are 20 and 44 samples, respectively.
    
    Failure modes and exceptions:
        Errors raised by generate_recording, generate_sorting, or snippets_from_sorting will propagate to the caller (for example ValueError for invalid parameter combinations, or OSError for filesystem issues when wf_folder is used). Mismatched or invalid types for parameters (e.g., non-integer nbefore/nafter, non-list durations) may cause downstream exceptions.
    """
    from spikeinterface.core.generate import generate_snippets
    return generate_snippets(
        nbefore,
        nafter,
        num_channels,
        wf_folder,
        sampling_frequency,
        durations,
        set_probe,
        ndim,
        num_units,
        empty_units,
        **job_kwargs
    )


################################################################################
# Source: spikeinterface.core.generate.generate_templates
# File: spikeinterface/core/generate.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_core_generate_generate_templates(
    channel_locations: numpy.ndarray,
    units_locations: numpy.ndarray,
    sampling_frequency: float,
    ms_before: float,
    ms_after: float,
    seed: int = None,
    dtype: numpy.ndarray = "float32",
    upsample_factor: int = None,
    unit_params: dict = None,
    mode: str = "ellipsoid"
):
    """Generate templates (simulated spike waveforms) for multiple units given spatial channel positions and neuron locations.
    
    This function is used in the SpikeInterface framework to create synthetic templates that mimic extracellular action potentials across a probe's channels. It constructs a single-channel waveform per unit with generate_single_fake_waveform(), applies a spatial decay law to distribute that waveform across channels based on distances between neuron and channel locations, and optionally applies propagation delays and temporal upsampling to produce realistic, multi-channel templates suitable for simulation, benchmarking, or algorithm development in spike sorting workflows.
    
    Args:
        channel_locations (numpy.ndarray): Array of channel coordinates. Each row is a channel position. The function accepts 2D coordinates (N, 2) or 3D coordinates (N, 3). If 2D are provided, a zero z-coordinate is appended internally to produce 3D coordinates. This parameter defines the electrode geometry used to compute distance-dependent amplitude decay of templates and thus controls how templates vary across channels in a realistic probe layout.
        units_locations (numpy.ndarray): Array of neuron locations with shape (num_units, 3). Each row must contain 3 values (x, y, z) for a unit. The function asserts that the second dimension equals 3 and will fail if units_locations is not 3D. These locations determine per-channel distances that drive spatial decay, propagation delays, and anisotropy when mode is "ellipsoid".
        sampling_frequency (float): Sampling frequency in Hz used to convert the temporal parameters (ms_before, ms_after) to sample counts and to generate waveforms. If upsample_factor is provided, an effective sampling frequency equal to sampling_frequency * upsample_factor is used for waveform generation to enable sub-sample shifts and jitter. Must be a positive float; nonpositive values are invalid and can produce incorrect nbefore/nafter or runtime errors.
        ms_before (float): Duration in milliseconds to include before the spike peak when generating each template (cut-out before peak). This is converted internally to an integer number of samples nbefore = int(sampling_frequency * ms_before / 1000.0). It determines the left-side temporal length of each template and thus the total template width when combined with ms_after.
        ms_after (float): Duration in milliseconds to include after the spike peak when generating each template (cut-out after peak). This is converted internally to an integer number of samples nafter = int(sampling_frequency * ms_after / 1000.0). Together with ms_before it sets the template width in samples: width = nbefore + nafter.
        seed (int): Optional RNG seed used to initialize internal randomness (via numpy.random.default_rng). The seed affects random choices in _ensure_unit_params when unit_params contain ranges (tuples) and therefore controls reproducibility of generated per-unit parameters (amplitude, durations, angles, etc.). If None, a non-deterministic RNG is used and results will vary across calls.
        dtype (numpy.ndarray): Numpy dtype (or dtype specifier accepted by numpy) used for the returned templates array and for intermediate waveform generation. The function signature default is "float32". The dtype determines numeric precision and memory footprint of the returned templates; common choice is numpy.float32 for balanced precision and memory usage.
        upsample_factor (int): Optional integer upsampling factor. If not None, upsample_factor is cast to int and must be >= 1. When provided, templates are generated at an effective sampling frequency sampling_frequency * upsample_factor and the returned templates gain a new axis (axis=3) with size upsample_factor. In this mode the function fills templates with sub-sampled versions of the upsampled waveforms (templates shape: (num_units, num_samples, num_channels, upsample_factor)), enabling simple random jitter by selecting a particular offset along the new axis. If None, no upsampling is performed and templates have shape (num_units, num_samples, num_channels).
        unit_params (dict): Optional dictionary specifying per-unit parameters used to shape waveforms and spatial behavior. Keys are parameter names and values can be: a numpy array of length num_units, a scalar (applied to all units), or a tuple specifying a range for random sampling. Recognized keys (and their roles/practical significance in spike waveform simulation) include:
            alpha: overall amplitude scaling of the action potential in arbitrary units (used as multiplier for spatial factors). Default typical range: (6000, 9000). Controls per-unit peak amplitude before spatial decay.
            depolarization_ms: duration of the depolarization phase in ms used to shape the single-channel waveform (default range: 0.09–0.14 ms). Affects waveform rising time.
            repolarization_ms: duration of the repolarization phase in ms (default range: 0.5–0.8 ms). Affects early falling phase of the waveform.
            recovery_ms: duration of the recovery interval in ms (default range: 1.0–1.5 ms). Affects late return to baseline.
            positive_amplitude: positive lobe amplitude relative to negative amplitude (negative amplitude is fixed at -1 when generating the base waveform). Range: 0.05–0.15. Controls waveform asymmetry.
            smooth_ms: Gaussian smoothing kernel width in ms applied to the waveform (default range: 0.03–0.07 ms). Affects waveform high-frequency content.
            spatial_decay: spatial decay constant in the same length unit as channel/unit coordinates (default range: 20–40). Used in the exponential decay alpha * exp(-distance / spatial_decay) to scale waveform amplitude across channels.
            propagation_speed: propagation "speed" in micrometers per millisecond to mimic propagation delays across channels (default range: 250–350). If not None, a channel-specific delay is computed from distance and applied using FFT-based sub-sample shifting; providing None disables propagation delays.
            ellipse_shrink: anisotropy factor applied to the y-axis of the ellipsoid distance calculation (used when mode == "ellipsoid"); must be provided or will be set by _ensure_unit_params.
            ellipse_angle: rotation angle in degrees (or radians consistent with get_ellipse) used to orient the ellipsoid when mode == "ellipsoid"; must be provided or will be set by _ensure_unit_params.
        mode (str): Distance calculation mode, either "ellipsoid" or "sphere". "sphere" uses isotropic Euclidean-like distances (calls get_ellipse with x_factor=1, y_factor=1, no rotation). "ellipsoid" introduces anisotropy dependent on per-unit ellipse_shrink and ellipse_angle parameters (calls get_ellipse with y_factor=ellipse_shrink and z_angle set to ellipse_angle). The chosen mode controls how spatial distances between unit and channel are computed and therefore changes channel-wise amplitude scaling and delays.
    
    Behavior and side effects:
        The function constructs templates for all units by calling generate_single_fake_waveform() once per unit to produce a mono-channel temporal waveform (shape: num_samples). It then computes distances between that unit and each channel according to mode and per-unit ellipse parameters, computes channel_factors = alpha * exp(-distances / spatial_decay), and multiplies the mono waveform with channel_factors to obtain a multi-channel waveform wfs of shape (num_samples, num_channels). If propagation_speed is provided for a unit, the function computes per-channel delays from distance and applies sub-sample temporal shifts using FFT (numpy.fft.rfft and numpy.fft.irfft) to produce realistic propagation across channels. If upsample_factor is provided, waveforms are generated at an effective sampling frequency multiplied by upsample_factor and the returned templates include the extra final axis for sub-sample offsets. The function uses numpy.random.default_rng(seed) to make random choices reproducible when seed is provided. Inputs are not modified in place except that a local copy of channel_locations may be replaced when appending a zero z-column for 2D inputs; the original array passed by the caller is not altered by the function.
    
    Failure modes and validations:
        The function asserts that units_locations has a second dimension of size 3 and will raise an AssertionError otherwise. If channel_locations has shape[1] not equal to 2 or 3, subsequent code will likely raise shape errors. If upsample_factor is provided and is not an integer >= 1, an AssertionError is raised. sampling_frequency must be a positive float; zero or negative values will lead to incorrect nbefore/nafter computation or runtime errors. unit_params values that are arrays must match num_units length or the behavior of _ensure_unit_params will raise or produce unexpected shapes. FFT-based propagation delay computation assumes the waveform length n is consistent; extremely large widths or memory-constrained environments can lead to MemoryError when allocating temporary FFT arrays. Any exceptions raised by generate_single_fake_waveform(), get_ellipse(), or _ensure_unit_params will propagate to the caller.
    
    Returns:
        numpy.ndarray: The templates array containing per-unit, per-sample, per-channel simulated waveforms. The shape is:
            * (num_units, num_samples, num_channels) when upsample_factor is None.
            * (num_units, num_samples, num_channels, upsample_factor) when upsample_factor is not None.
        The dtype of the returned array is the dtype specified by the dtype parameter. The returned templates represent synthetic extracellular action potential waveforms across the probe channels and can be used for simulation of recordings, benchmarking spike sorting algorithms, or initializing template-matching procedures.
    """
    from spikeinterface.core.generate import generate_templates
    return generate_templates(
        channel_locations,
        units_locations,
        sampling_frequency,
        ms_before,
        ms_after,
        seed,
        dtype,
        upsample_factor,
        unit_params,
        mode
    )


################################################################################
# Source: spikeinterface.core.generate.generate_unit_locations
# File: spikeinterface/core/generate.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_core_generate_generate_unit_locations(
    num_units: int,
    channel_locations: numpy.ndarray,
    margin_um: float = 20.0,
    minimum_z: float = 5.0,
    maximum_z: float = 40.0,
    minimum_distance: float = 20.0,
    max_iteration: int = 100,
    distance_strict: bool = False,
    distribution: str = "uniform",
    num_modes: int = 2,
    seed: int = None
):
    """spikeinterface.core.generate.generate_unit_locations: Generate random 3D unit locations for simulated neurons or test datasets, constrained by recording channel geometry and inter-unit distance rules.
    
    This function is part of the SpikeInterface framework used to create spatial layouts of neuronal units relative to extracellular recording channels for simulations, benchmarking, waveform extraction, quality-metric evaluation, and other spike-sorting workflows. The function samples num_units 3D coordinates (x, y, z) so that x and y lie within the bounding box of the provided channel_locations expanded by margin_um, z lies between minimum_z and maximum_z, and pairwise Euclidean distances between generated units meet a minimum_distance constraint when requested. Reproducible sampling is supported via an explicit random seed (seed). The function uses NumPy's Generator (numpy.random.default_rng) to draw uniform random values and an internal helper (_generate_multimodal) when distribution="multimodal" to mimic layered (multimodal) y-axis distributions.
    
    Args:
        num_units (int): Number of unit locations to generate. In the spike-sorting domain this corresponds to the number of simulated neurons whose spatial coordinates will be produced.
        channel_locations (numpy.ndarray): 2D array of shape (num_channels, 2) giving the (x, y) coordinates of each recording channel. These coordinates define the recording probe geometry and are used to compute the allowable x and y sampling ranges by taking the channel extents and expanding them by margin_um.
        margin_um (float): Margin in micrometers added to the minimum and maximum x and y channel coordinates when defining the sampling region for unit x and y coordinates. Increasing margin_um allows units to be placed beyond the outermost channels; default is 20.0.
        minimum_z (float): Minimum z coordinate (depth) for generated unit locations. This defines the lower bound of the third spatial dimension for units; default is 5.0.
        maximum_z (float): Maximum z coordinate (depth) for generated unit locations. This defines the upper bound of the third spatial dimension for units; default is 40.0.
        minimum_distance (float): Minimum allowable Euclidean distance between any two generated units in micrometers. If set to None, no inter-unit distance constraint is enforced. When set to a numeric value, the function iteratively resamples units that violate this constraint until a valid configuration is found or max_iteration is reached; default is 20.0.
        max_iteration (int): Maximum number of iterative resampling attempts to satisfy the minimum_distance constraint. The algorithm recomputes pairwise Euclidean distances after each resampling pass and stops early if a solution is found; default is 100.
        distance_strict (bool): If True and no configuration satisfying minimum_distance is found within max_iteration attempts, the function raises a ValueError; if False (default) it emits a warnings.warn message and returns the best configuration produced. Use True to enforce strict failure behavior in automated pipelines.
        distribution (str): Sampling distribution for the y coordinate. Allowed values are "uniform" or "multimodal". "uniform" samples y uniformly across the allowed y-range. "multimodal" uses an internal helper to generate num_modes modes along the y axis to mimic layered unit distributions (e.g., cortical layers). When used with minimum_distance not None there is no guarantee of a perfectly multimodal outcome because units violating distance constraints are resampled and may fall between modes; default is "uniform".
        num_modes (int): When distribution="multimodal", the number of modes (layers) to generate along the y axis. This controls how many peaks the internal multimodal generator will produce; default is 2.
        seed (int or None): Integer seed for numpy.random.default_rng to make sampling reproducible. If None (default), the RNG is not explicitly seeded and results are non-deterministic across runs.
    
    Returns:
        numpy.ndarray: A 2D array of shape (num_units, 3) where each row is the (x, y, z) coordinates of a generated unit location. The returned array has dtype float32. Coordinates are sampled within [min_channel_x - margin_um, max_channel_x + margin_um] for x, [min_channel_y - margin_um, max_channel_y + margin_um] for y (or according to the multimodal generator), and [minimum_z, maximum_z] for z. Pairwise distances are Euclidean (L2 norm) and reflect any enforced minimum_distance constraint.
    
    Behavior, side effects, defaults, and failure modes:
        - The function constructs a NumPy random Generator via numpy.random.default_rng(seed) for all sampling; setting seed produces repeatable layouts useful for reproducible simulations and benchmarking in spike-sorting workflows.
        - If distribution is not "uniform" or "multimodal", the function raises ValueError indicating an unsupported distribution argument.
        - If minimum_distance is not None, the function computes pairwise Euclidean distances between units and iteratively resamples units that violate the distance constraint. Resampling is limited by max_iteration. If a valid configuration is not found within max_iteration:
            - If distance_strict is True, a ValueError is raised indicating no solution was found for the requested minimum_distance and max_iteration.
            - If distance_strict is False, the function issues a warnings.warn describing the failure and returns the current unit locations (which may still contain distances below minimum_distance).
        - When distribution="multimodal" and minimum_distance is specified, resampling required to satisfy distance constraints can disrupt the intended multimodal layering; therefore the returned distribution may not be a true multimodal layout in such cases.
        - The function may raise other runtime errors if channel_locations has invalid shape or contains non-numeric values; channel_locations is expected to be a numeric numpy.ndarray with two columns representing x and y coordinates.
    """
    from spikeinterface.core.generate import generate_unit_locations
    return generate_unit_locations(
        num_units,
        channel_locations,
        margin_um,
        minimum_z,
        maximum_z,
        minimum_distance,
        max_iteration,
        distance_strict,
        distribution,
        num_modes,
        seed
    )


################################################################################
# Source: spikeinterface.core.generate.get_ellipse
# File: spikeinterface/core/generate.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_core_generate_get_ellipse(
    positions: numpy.ndarray,
    center: numpy.ndarray,
    x_factor: float = 1,
    y_factor: float = 1,
    x_angle: float = 0,
    y_angle: float = 0,
    z_angle: float = 0
):
    """Compute per-channel radii (distances) to a specified 3D ellipsoidal volume used to model spatial inhomogeneities when generating templates in spike sorting pipelines. This function is used in SpikeInterface to transform 2D channel coordinates into the ellipsoid's centered, rotated reference frame, rescale with axis factors, and compute the scalar radius R for each channel using the ellipsoid equation R = (X/x_factor)**2 + (Y/y_factor)**2 + (Z/1)**2. The computed distances can be used to derive putative amplitudes or attenuation factors for template generation and spatial weighting in downstream spike sorting and post-processing.
    
    Args:
        positions (numpy.ndarray): 2D array of channel coordinates with shape (n_channels, 2). Each row must contain the x and y coordinate of a recording channel in the same Cartesian units used for center. This function assumes channel z coordinates lie in the recording plane z = 0; the z difference is computed as (0 - center[2]). Supplying coordinates in a different shape or with an included z column will produce incorrect results or raise indexing errors.
        center (numpy.ndarray): 1D array-like with three elements [x0, y0, z0] specifying the ellipsoid center in the same Cartesian units as positions. center[0] and center[1] are subtracted from the corresponding x and y channel coordinates; center[2] is used as the channel-plane offset (channels are assumed at z = 0 so the z difference is -center[2]). Passing a center with length different from 3 will lead to indexing errors.
        x_factor (float): Scaling factor a (semi-axis length) applied to the X axis of the ellipsoid in the rotated, centered frame. The ellipsoid equation uses X/x_factor; values close to zero will produce very large computed radii or division-by-zero runtime warnings/errors. Typical use: increase x_factor to flatten the ellipsoid along X, decrease to elongate along X. Default is 1.
        y_factor (float): Scaling factor b (semi-axis length) applied to the Y axis of the ellipsoid in the rotated, centered frame. The ellipsoid equation uses Y/y_factor; same failure modes as x_factor apply. Default is 1.
        x_angle (float): Rotation angle (in radians) around the X axis used to rotate the ellipsoid reference frame before computing radii. The implementation applies the negative of this angle (numpy.cos(-x_angle), numpy.sin(-x_angle)), corresponding to a passive rotation of the coordinate frame. Provide angles in radians; non-numeric inputs will raise type errors. Default is 0.
        y_angle (float): Rotation angle (in radians) around the Y axis used to rotate the ellipsoid reference frame. As with x_angle, the code uses the negative angle for the rotation matrix. Default is 0.
        z_angle (float): Rotation angle (in radians) around the Z axis used to rotate the ellipsoid reference frame. The combined rotation matrix is Rx @ Ry @ Rz (where each uses the negative of the provided angle). Default is 0.
    
    Returns:
        numpy.ndarray: 1D array of length n_channels containing the computed ellipsoidal radii/distances for each input channel. Each returned value is non-negative and equals sqrt((X/x_factor)**2 + (Y/y_factor)**2 + (Z/1)**2) where [X, Y, Z] is the channel position after translation by center and rotation by the composed rotation matrix. No side effects occur; the function does not modify its inputs.
    
    Behavior, defaults, and failure modes:
        - The function translates channels by subtracting center[:2] from positions[:, :2] and treats channels as lying at z = 0 (thus Z = 0 - center[2]). If your channels have nonzero z coordinates, you must incorporate them into positions and adapt this function accordingly.
        - Rotations are performed by constructing Rx, Ry, Rz using the negative of the supplied angles and composing them as Rx @ Ry @ Rz; this implements a change of reference frame before computing ellipsoidal distances.
        - x_factor and y_factor must be nonzero finite floats; zero or NaN values will cause runtime warnings or invalid results.
        - Inputs must be numpy.ndarray (as declared). Passing other array-like types that cannot be indexed or broadcast as expected will raise errors.
        - In the special case x_factor == y_factor == 1 and all angles == 0, the returned distances reduce to the Euclidean distance from each channel to the ellipsoid center projected into the channel plane (i.e., standard planar Euclidean distance when center[2] == 0).
    """
    from spikeinterface.core.generate import get_ellipse
    return get_ellipse(positions, center, x_factor, y_factor, x_angle, y_angle, z_angle)


################################################################################
# Source: spikeinterface.core.generate.synthesize_poisson_spike_vector
# File: spikeinterface/core/generate.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_core_generate_synthesize_poisson_spike_vector(
    num_units: int = 20,
    sampling_frequency: float = 30000.0,
    duration: float = 60.0,
    refractory_period_ms: float = 4.0,
    firing_rates: float = 3.0,
    seed: int = 0
):
    """synthesize_poisson_spike_vector generates discrete spike times (sample frames) for multiple neuronal units by simulating independent Poisson spike trains per unit and enforcing a refractory period. This function is used in SpikeInterface to create synthetic ground-truth spike data for testing, benchmarking, and development of spike-sorting workflows; it models each unit's spiking as a discrete-time Poisson process (geometric inter-spike counts), compensates for the refractory period by increasing the effective firing rate, discretizes times to sample frames using the provided sampling frequency, and returns sorted global spike frames with corresponding unit indices.
    
    Args:
        num_units (int): Number of neuronal units to simulate. This determines how many independent Poisson processes are generated and therefore the range of returned unit indices (0 .. num_units-1). Affects memory and runtime linearly because inter-spike samples are generated per unit.
        sampling_frequency (float): Sampling frequency in Hz used to discretize continuous spike times into integer sample frames. All returned spike times (spike_frames) are indices in units of samples computed with this sampling frequency. Must be positive; larger values increase the number of discrete time bins and thus the resolution of the simulation.
        duration (float): Duration of the simulated recording in seconds. The function generates spikes up to duration * sampling_frequency samples (internally uses max_frames = int(duration * sampling_frequency) - 1). Spikes beyond this duration are discarded.
        refractory_period_ms (float): Refractory period enforced per unit, in milliseconds. This is converted to an integer number of sample frames via int((refractory_period_ms / 1000.0) * sampling_frequency) and added to inter-spike intervals to prevent two spikes from the same unit occurring within this period. If this refractory period is too large relative to a unit's firing rate (see Failure modes), a ValueError is raised.
        firing_rates (float or array_like or tuple): Desired firing rate(s) in Hz. Can be a single scalar applied uniformly to all units or an array-like/tuple providing one firing rate per unit (length must match num_units). These rates are the target rates after accounting for the enforced refractory period; internally the code increases the raw generation probability to compensate for the dead time so that the observed rate matches the requested rate in expectation.
        seed (int): Seed for the numpy random number generator (numpy.random.default_rng) to allow reproducible simulations. Providing the same seed and identical parameters produces the same spike_frames and unit_indices.
    
    Returns:
        spike_frames (ndarray): 1D numpy array of sorted integer sample frames (indices) at which spikes occur. Frames are in ascending order, within the inclusive range [0, int(duration * sampling_frequency) - 1]. Each entry corresponds to one spike event in the simulated recording. The dtype is integer-like (generated from geometric counts then cumulative-summed to frames).
        unit_indices (ndarray): 1D numpy array of unit indices (dtype uint16 as produced by the implementation) that correspond to each entry of spike_frames. Each value is an integer in [0, num_units-1] indicating which simulated unit produced the spike at the matching frame. The arrays are aligned so that spike_frames[i] is the sample index of a spike from unit_indices[i].
    
    Behavior and implementation details:
        - Inter-spike intervals are generated per unit as discrete geometric random variables with success probability p derived from an adjusted firing rate (modified_firing_rate = firing_rates / (1 - firing_rates * refractory_period_seconds)). This adjustment compensates for the refractory period so that the effective firing rate matches firing_rates in expectation, following the modeling approach described in the original implementation and literature (Deger et al., 2012).
        - The per-bin success probability used for geometric draws is binomial_p_modified = modified_firing_rate / sampling_frequency and is clipped to a maximum of 1.0.
        - The refractory period in frames (refractory_period_frames) is added to every inter-spike interval except the first for each unit to enforce a minimum gap between spikes from the same unit.
        - To bound memory usage, the code estimates an upper bound on the number of spikes per unit (num_spikes_max) from the maximum per-bin probability and duration, generates arrays of shape (num_units, num_spikes_max), and then flattens and filters spikes that fall beyond the requested duration.
        - The global outputs are sorted by spike frame using a stable argsort so the temporal order of spikes is preserved; ties (simultaneous spikes across units) keep their original relative order from the generation procedure.
        - The function is deterministic for a given seed value; different seeds produce different stochastic realizations.
    
    Failure modes and errors:
        - ValueError is raised if the provided refractory_period_ms is too long for the specified firing_rates (specifically when refractory_period_seconds >= 1.0 / firing_rate for any unit), because no valid inter-spike interval distribution can produce the requested average rate under that dead-time constraint.
        - If firing_rates is provided as an array-like, it must provide one rate per unit (length equal to num_units); otherwise supporting utilities called by this function may raise an error.
        - Very large num_units, very high sampling_frequency, or long duration can lead to high memory use because the implementation allocates an intermediate (num_units x num_spikes_max) array; num_spikes_max is estimated from duration and the maximum per-bin probability and can be large if parameters imply high expected spike counts.
    
    Side effects and performance notes:
        - No external state is modified beyond returning the arrays; randomness is controlled only via the provided seed.
        - The implementation attempts to minimize temporary allocations (e.g., by reusing arrays where possible and by writing cumulative sums into preallocated arrays) for performance; nonetheless, memory usage scales with num_units and the estimated maximum number of spikes.
        - The produced output arrays are suitable as synthetic inputs to SpikeInterface sorting pipelines, benchmarking utilities, or any downstream code expecting arrays of spike sample indices and corresponding unit labels.
    """
    from spikeinterface.core.generate import synthesize_poisson_spike_vector
    return synthesize_poisson_spike_vector(
        num_units,
        sampling_frequency,
        duration,
        refractory_period_ms,
        firing_rates,
        seed
    )


################################################################################
# Source: spikeinterface.core.generate.synthesize_random_firings
# File: spikeinterface/core/generate.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_core_generate_synthesize_random_firings(
    num_units: int = 20,
    sampling_frequency: float = 30000.0,
    duration: float = 60,
    refractory_period_ms: float = 4.0,
    firing_rates: list = 3.0,
    add_shift_shuffle: bool = False,
    seed: int = None
):
    """synthesize_random_firings generates a single-segment synthetic dataset of spike times and unit labels for use in spike-sorting development, testing, and benchmarking within the SpikeInterface framework. The function simulates num_units independent units firing with Poisson-like average rates over a recording of given duration and sampling frequency, enforces a refractory period in samples, optionally perturbs half of each unit's spikes to produce less-flat autocorrelograms, and returns concatenated, time-sorted spike times and their corresponding unit labels.
    
    Args:
        num_units (int): Number of units (neurons) to simulate. This controls the number of distinct integer labels produced (0 .. num_units-1). Default: 20. In practical spike-sorting workflows this represents the number of ground-truth units present in the synthetic recording segment.
        sampling_frequency (float): Sampling rate in Hz used to convert between time (seconds) and discrete sample indices. Default: 30000.0. Internally, duration is converted to a segment size in samples as int(sampling_frequency * duration) and the returned times are integer sample indices relative to the start of the segment.
        duration (float): Duration of the simulated segment in seconds. Default: 60. This determines the total number of samples (segment length) and, together with firing_rates, the expected number of spikes per unit.
        refractory_period_ms (float): Minimal allowed inter-spike interval per unit expressed in milliseconds. This is converted internally to an integer refractory_sample = int(refractory_period_ms / 1000.0 * sampling_frequency) and used to remove spikes that violate the refractory constraint. Default: 4.0. Practically, increasing this value reduces occurrences of very close spike times for the same unit.
        firing_rates (float or list[float]): Target firing rate(s) in Hz. If a single float is provided, all units will use that firing rate; if a list (or list-like) of floats is provided it should specify a rate per unit. The function expands or validates this input to produce a per-unit firing rate vector of length num_units using an internal helper. Default: 3.0. These rates are used to compute an expected integer number of spikes per unit as int(rate * duration), which is then sampled and possibly pruned by the refractory constraint.
        add_shift_shuffle (bool): If True, for each unit approximately half of the generated spikes are shifted forward by a small positive integer sample offset to make the autocorrelogram less flat. The shift values are computed as shift = a + (b - a) * x**2 with a = refractory_sample and b = refractory_sample*20 and x drawn uniformly in [0,1). After shifting, spikes falling outside the segment are discarded. Default: False. This option is intended to introduce realistic clustering of some spike times while preserving the refractory-based pruning.
        seed (int): Seed for the random number generator (passed to numpy.random.default_rng) to make the generation deterministic and reproducible. Default: None. When provided, the same seed will yield the same spike times and labels across runs; when None behavior is nondeterministic.
    
    Behavior, defaults, and failure modes:
        The function uses a NumPy Generator (numpy.random.default_rng) initialized with seed to draw integer spike candidate times per unit and to perform random choices for subsampling and shifting. For each unit the function:
        - computes an expected spike count n_spikes = int(firing_rate * duration) and samples a slightly larger pool n = int(n_spikes + 10 * sqrt(n_spikes)) of integer times in the segment to allow pruning,
        - sorts these candidate times, optionally shifts roughly half of them when add_shift_shuffle is True, removes spikes that violate the refractory period by detecting adjacent differences smaller than refractory_sample, and if more spikes remain than n_spikes randomly subsamples without replacement to n_spikes.
        - assigns the unit index (int) as the label for that unit's spikes.
    
        The returned spike times are integer sample indices in the range [0, segment_size-1] where segment_size = int(sampling_frequency * duration). To convert returned times to seconds divide by sampling_frequency. The function casts times and labels to dtype int64 before concatenation.
    
        No files are written and there are no external side effects beyond the deterministic behavior controlled by seed. Possible failure modes include invalid firing_rates input (e.g., a list of a length not compatible with num_units) which may raise an error in the internal helper that expands/validates firing_rates; extremely small firing_rates or very large refractory_period_ms may produce zero spikes for some or all units; because candidate spikes are pruned by the refractory constraint and then randomly subsampled if too many remain, the final number of spikes per unit may be less than or equal to int(firing_rate * duration).
    
    Returns:
        times (numpy.array): 1-D NumPy array of concatenated and time-sorted spike times expressed as integer sample indices (dtype int64). The array contains one entry per spike across all units in the segment, sorted in non-decreasing order.
        labels (numpy.array): 1-D NumPy array of concatenated and time-sorted integer unit labels (dtype int64). labels[i] is the unit index (0 .. num_units-1) for the spike at times[i]. The two arrays have the same length and correspondence by index.
    """
    from spikeinterface.core.generate import synthesize_random_firings
    return synthesize_random_firings(
        num_units,
        sampling_frequency,
        duration,
        refractory_period_ms,
        firing_rates,
        add_shift_shuffle,
        seed
    )


################################################################################
# Source: spikeinterface.core.generate.synthetize_spike_train_bad_isi
# File: spikeinterface/core/generate.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_core_generate_synthetize_spike_train_bad_isi(
    duration: float,
    baseline_rate: float,
    num_violations: int,
    violation_delta: float = 1e-05
):
    """Create a synthetic spike train with mostly uniform inter-spike intervals (ISIs) and a specified number of close temporal contaminating spikes that produce ISI violations. This function is intended for use in spike-sorting and quality-metric workflows (as in SpikeInterface) to simulate recordings with controlled firing rate and a controllable number of contaminating events that violate the expected refractory period, so downstream algorithms and metrics can be validated against known violations.
    
    Args:
        duration (float): Length of the simulated recording in seconds. This controls the nominal time span over which spikes are generated. Expected to be positive; passing a non-positive value will lead to zero or invalid spike counts or a ValueError from downstream operations.
        baseline_rate (float): Firing rate for the "true" spikes, in spikes per second. The function constructs uniform inter-spike intervals of 1.0 / baseline_rate and repeats them int(duration * baseline_rate) times to approximate this rate over the requested duration. Must be positive; a value of zero will cause a division-by-zero error, and negative values produce ISIs with reversed sign and are not meaningful in the spike-sorting context.
        num_violations (int): Number of contaminating spikes to insert that create ISI violations relative to the true spikes. The first num_violations spikes from the generated true spike train are each duplicated with a small temporal offset (violation_delta) to simulate contamination. Expected to be a non-negative integer; if num_violations is larger than the number of generated true spikes, only the available early spikes will be duplicated. A negative value will produce unintended slicing behavior and is not supported.
        violation_delta (float): Temporal offset in seconds for each contaminating spike relative to its corresponding true spike. Default: 1e-5. Typical use is a very small positive value to create near-coincident contaminating spikes (bad ISI). If violation_delta is zero, exact duplicate times will be created. If violation_delta is large enough that a contaminating spike falls at or beyond duration, that contaminating spike is discarded (see behavior below). Negative values are allowed by the code but will place contaminating spikes earlier than the originals, which may be outside the intended simulation assumptions.
    
    Returns:
        spike_train (numpy.array): One-dimensional numpy array of spike times (in seconds) sorted in non-decreasing order. The array contains the uniformly spaced "true" spikes (inter-spike interval = 1.0 / baseline_rate) and the additional contaminating spikes (up to num_violations) offset by violation_delta. Contaminating spikes that fall at or beyond duration are removed before the final sort. The returned times are suitable for use as simulated spike timestamps in spike-sorting pipelines and quality-metric computations.
    
    Behavior and important notes:
        - The function generates int(duration * baseline_rate) true spikes with ISI = 1.0 / baseline_rate. Because the count uses truncation via int(...), the effective realized duration and number of spikes are subject to integer truncation.
        - Contaminating spikes are created by taking the first num_violations true spikes and adding violation_delta to each. Contaminating spikes with times >= duration are filtered out (they are not included in the returned spike_train).
        - The final spike_train is the sorted concatenation of true spikes and valid contaminating spikes; if violation_delta is zero this will produce duplicate timestamps, which is a valid representation of coincident events for testing but may be treated specially by downstream code.
        - No side effects occur: the function does not modify external state and returns a new numpy array.
        - Failure modes include division by zero or invalid behavior if baseline_rate is zero or non-positive, unexpected results if duration is non-positive, and unintended slicing if num_violations is negative. The caller should ensure duration > 0, baseline_rate > 0, and num_violations >= 0 for meaningful simulations.
        - This function is deterministic given identical inputs and is intended for controlled synthetic data generation for validating spike-sorting algorithms, computing ISI-based quality metrics, and benchmarking contamination handling in the SpikeInterface ecosystem.
    """
    from spikeinterface.core.generate import synthetize_spike_train_bad_isi
    return synthetize_spike_train_bad_isi(
        duration,
        baseline_rate,
        num_violations,
        violation_delta
    )


################################################################################
# Source: spikeinterface.core.globals.set_global_dataset_folder
# File: spikeinterface/core/globals.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_core_globals_set_global_dataset_folder(folder: str):
    """Set the global dataset folder used by SpikeInterface.
    
    This function records a filesystem location (provided as a string) as the global dataset folder for the running Python process. The supplied folder string is converted internally to a pathlib.Path and stored in the module-level global variable dataset_folder. The module-level boolean dataset_folder_set is also set to True. Many SpikeInterface utilities that manage example datasets, dataset downloaders, and I/O helpers consult this global dataset_folder as the default location to read from or write to; calling this function therefore changes the default storage/lookup location for those components across the entire SpikeInterface runtime.
    
    Args:
        folder (str): Filesystem path to use as the global dataset folder, expressed as a string. The path may be absolute or relative; it is converted internally to a pathlib.Path via Path(folder) and assigned to the global variable dataset_folder. This parameter is the canonical way, within SpikeInterface, to specify where recordings, sorting outputs, and example datasets should be stored or looked up by dataset-related helpers.
    
    Returns:
        None: This function does not return a value. Instead, it has the side effects of setting the module-level globals dataset_folder (to Path(folder)) and dataset_folder_set (to True). After this call, other SpikeInterface functions that rely on these globals will use the newly set folder as their default location.
    
    Behavior, side effects, and failure modes:
    - No filesystem validation or creation is performed. The function does not check that the path exists, is writable, or points to a directory; it only converts the provided string to a pathlib.Path and stores it. Subsequent I/O operations that assume the folder exists may raise FileNotFoundError, PermissionError, or other OS-level errors.
    - The function mutates global state in the spikeinterface.core.globals module. This global mutation affects all SpikeInterface modules in the same interpreter and process; callers should be aware that changing this value changes defaults globally.
    - The function is not synchronized for concurrent use. Concurrent calls from multiple threads or processes may race, leading to indeterminate ordering of updates to dataset_folder and dataset_folder_set.
    - The parameter is annotated as str in the function signature; callers should pass a string. If a non-string value is passed, behavior depends on pathlib.Path(folder) and may raise an exception; callers should follow the documented signature and provide a string.
    """
    from spikeinterface.core.globals import set_global_dataset_folder
    return set_global_dataset_folder(folder)


################################################################################
# Source: spikeinterface.core.globals.set_global_tmp_folder
# File: spikeinterface/core/globals.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_core_globals_set_global_tmp_folder(folder: str):
    """Set the global temporary folder path used by SpikeInterface.
    
    This function defines the module-level temporary folder that many SpikeInterface components use to store intermediate files, caches, logs, or files passed to external sorters and exporters (for example during preprocessing, running sorters, waveform extraction, and export). Calling this function updates internal global state so that subsequent operations in the same Python process will read and write temporary artifacts under the provided path.
    
    Args:
        folder (str): Filesystem path, expressed as a string, that will be used as the global temporary folder for SpikeInterface. The function converts this string to a pathlib.Path and assigns it to the module-level variable temp_folder. The exact practical significance is that components which create intermediate files (e.g., preprocessing pipelines, sorter wrappers, waveform extractors, exporters to Phy) will use this location instead of any previously configured location.
    
    Returns:
        None: This function does not return a value. Instead it has the side effect of setting the module-level globals temp_folder (a pathlib.Path created from the provided string) and temp_folder_set (a boolean set to True). After this call, any SpikeInterface code that checks these globals will observe the new temporary folder.
    
    Behavior, side effects, and failure modes:
        This is a global, process-wide change: it immediately affects all subsequent SpikeInterface operations in the current Python process that rely on the module's temporary folder. The function does not create the directory on disk, check that it exists, or validate permissions; it only stores the path. Therefore, if the provided path does not exist or is not writable, later file operations that attempt to write into this folder may raise filesystem errors (for example FileNotFoundError or PermissionError) when those operations run. The function performs no synchronization, so concurrent threads that call or read the globals can race; use external synchronization if you need thread-safe updates. If the argument is not a string, pathlib.Path(folder) conversion may raise a TypeError; callers should pass a string path as documented.
    """
    from spikeinterface.core.globals import set_global_tmp_folder
    return set_global_tmp_folder(folder)


################################################################################
# Source: spikeinterface.core.job_tools.split_job_kwargs
# File: spikeinterface/core/job_tools.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_core_job_tools_split_job_kwargs(mixed_kwargs: dict):
    """spikeinterface.core.job_tools.split_job_kwargs.
    Splits a mixed dictionary of keyword arguments into two separate dictionaries: one containing job-level control options used by SpikeInterface job management utilities and one containing function-specific parameters used by spike-sorting, preprocessing, or postprocessing routines.
    
    This function is used in the SpikeInterface framework (a unified framework for spike sorting) to allow functions with a generic signature that mix execution/control parameters (for example, parallelization, scheduling, or job-dispatch options) and algorithm- or dataset-specific parameters to separate those concerns. The implementation iterates over the provided mixed_kwargs and classifies each key as a job-level key if it is present in the module-level job_keys collection; all other keys are classified as specific (domain) kwargs. After collecting job-level keys, the job_kwargs dict is passed to the module-level helper fix_job_kwargs to normalize, validate, and possibly augment job-level settings (for example to set defaults or normalize formats used by downstream job runners). The original mixed_kwargs mapping is not mutated; two new dict objects are returned.
    
    Args:
        mixed_kwargs (dict): A mapping of keyword argument names to values that mixes both job-level control parameters and domain-specific parameters. In the SpikeInterface domain this commonly contains execution or orchestration options (job-level keys, e.g., parallelization/scheduling related entries) together with spike-sorting, recording, or preprocessing parameters (specific keys). The function expects a dict-like object supporting the items() method; passing a non-mapping may raise an AttributeError or TypeError.
    
    Returns:
        tuple: A 2-tuple (specific_kwargs, job_kwargs) where:
            specific_kwargs (dict): A new dict containing keys from mixed_kwargs that are not present in the module-level job_keys collection. These are the domain- or algorithm-specific parameters that will be forwarded to sorting, preprocessing, or postprocessing functions.
            job_kwargs (dict): A new dict containing the keys from mixed_kwargs that are present in job_keys. This dict has been passed through fix_job_kwargs, and so may have been normalized, validated, or supplemented with defaults by that helper. The job_kwargs dict represents execution/control options for SpikeInterface job management utilities.
    
    Behavior and side effects:
        This function performs classification only and does not mutate the input mixed_kwargs mapping; it creates and returns two new dict objects. It depends on two module-level symbols: job_keys (an iterable of key names treated as job-level) and fix_job_kwargs (a callable that accepts and returns a dict of job kwargs). If job_keys or fix_job_kwargs are not defined in the module namespace a NameError will be raised. If fix_job_kwargs performs additional validation it may raise its own exceptions (ValueError, TypeError, etc.) which propagate to the caller.
    
    Defaults and failure modes:
        No defaults are introduced by this function itself; any defaults or normalization applied to job-level parameters are the responsibility of fix_job_kwargs. Passing a non-dict or non-mapping for mixed_kwargs will likely raise an exception when items() is accessed. Keys not recognized as job-level are always placed in specific_kwargs. The exact set of job-level keys and the normalization rules are determined by the module-level job_keys and fix_job_kwargs implementations.
    """
    from spikeinterface.core.job_tools import split_job_kwargs
    return split_job_kwargs(mixed_kwargs)


################################################################################
# Source: spikeinterface.core.motion.ensure_time_bins
# File: spikeinterface/core/motion.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_core_motion_ensure_time_bins(
    time_bin_centers_s: numpy.ndarray = None,
    time_bin_edges_s: numpy.ndarray = None
):
    """Ensure that both time bin centers and time bin edges are available for use in spike-sorting workflows (e.g., motion estimation, per-bin metrics, or temporal alignment). This function accepts either bin centers or bin edges (for a single recording segment or for multiple segments as a list) and reconstructs the missing representation. When converting edges to centers the midpoint of each adjacent edge pair is used. When converting centers to edges the function pads the left and right sides with the outer center values and computes midpoints for interior edges. This behavior ensures consistent 1D time-bin definitions in seconds across single- and multi-segment data used throughout SpikeInterface for preprocessing, quality metrics, and visualization.
    
    Args:
        time_bin_centers_s (None or numpy.array or list[numpy.array]): Time bin center values in seconds. For a single recording segment this must be a 1D numpy array of center times (dtype preserved in reconstructed edges). For multi-segment data this may be a list of 1D numpy arrays, one per segment. If provided and time_bin_edges_s is None, the function will construct time bin edges from these centers by creating an array of length (n_centers + 1), setting the first and last edge equal to the first and last center, and filling interior edges with midpoints of adjacent centers. If time_bin_centers_s is None and time_bin_edges_s is provided, centers will be reconstructed from edges. Default is None.
        time_bin_edges_s (None or numpy.array or list[numpy.array]): Time bin edge values in seconds. For a single recording segment this must be a 1D numpy array with at least two entries (each pair of adjacent edges defines one bin). For multi-segment data this may be a list of such 1D numpy arrays. If provided and time_bin_centers_s is None, the function will compute bin centers as the midpoint of each adjacent edge pair. If time_bin_edges_s is None and time_bin_centers_s is provided, edges will be reconstructed as described above. Default is None.
    
    Returns:
        time_bin_centers_s, time_bin_edges_s (tuple): A tuple of (time_bin_centers_s, time_bin_edges_s) where each element is either a 1D numpy array (for single-segment inputs) or a list of 1D numpy arrays (for multi-segment inputs). The first element contains bin center times in seconds; the second element contains bin edge times in seconds. For single-segment arrays, the returned edges have length equal to centers.size + 1 and preserve the dtype of the input centers when constructed. For multi-segment lists, each centers list element corresponds to the same-index edges element.
    
    Raises:
        ValueError: If both time_bin_centers_s and time_bin_edges_s are None, because at least one representation is required to reconstruct the other.
        AssertionError: If time_bin_edges_s is provided as a single-segment array but is not one-dimensional or has fewer than two elements; this enforces that edges define at least one bin.
    
    Notes:
        - The function supports recursion to handle lists of per-segment arrays: if a list is provided for either argument, each element is processed independently and the function returns a list of reconstructed arrays.
        - The conversion from edges to centers uses simple midpoints: center[i] = 0.5 * (edge[i] + edge[i+1]).
        - The conversion from centers to edges pads outer edges with the outermost centers and sets interior edges to midpoints of neighboring centers.
        - The function does not introduce new units; all times are expected and returned in seconds, as indicated by the _s suffix in parameter names.
        - The function returns newly constructed arrays/lists and does not mutate input arrays/lists in-place.
    """
    from spikeinterface.core.motion import ensure_time_bins
    return ensure_time_bins(time_bin_centers_s, time_bin_edges_s)


################################################################################
# Source: spikeinterface.core.node_pipeline.check_graph
# File: spikeinterface/core/node_pipeline.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_core_node_pipeline_check_graph(nodes: list):
    """Validate the ordering and composition of a SpikeInterface node pipeline.
    
    This function enforces the structural rules required by the peak-processing pipeline used in SpikeInterface (the unified spike sorting framework). It inspects the provided list of pipeline nodes and verifies: (1) the first element is a PeakSource (for example a PeakDetector, PeakRetriever, or SpikeRetriever), which is required because the pipeline must start from a source of detected peaks; (2) every element is an instance of PipelineNode, the base type used for nodes in the node pipeline; and (3) for each node, every declared parent (node.parents) is included in the provided list and appears earlier in the list so that parents precede children in execution order. The function treats a falsy node.parents value (None or empty) as no parents. This validation is intended to be called before running or composing pipeline stages to avoid runtime errors during spike detection, preprocessing, or downstream sorting/post-processing.
    
    Args:
        nodes (list): Ordered sequence of pipeline node objects that represent stages in a peak-processing pipeline in SpikeInterface. Each element is expected to be an instance of PipelineNode and may expose a parents attribute (truthy iterable of other PipelineNode objects) that lists upstream dependencies. The order of elements expresses the intended execution order; parents must appear earlier in this list. The practical significance is that this list defines the dataflow for operations such as peak detection, peak retrieval, and subsequent processors used throughout SpikeInterface for extracellular spike sorting and post-processing.
    
    Returns:
        list: The same list object passed in as nodes. Returning the original list allows callers to perform this validation inline as part of pipeline construction or to chain validators without copying or modifying the pipeline definition. This function does not mutate the list or its elements; its effect is purely validation.
    
    Raises:
        ValueError: If the first element of nodes is not an instance of PeakSource (for example not a PeakDetector, PeakRetriever, or SpikeRetriever). This enforces the domain rule that the pipeline must start from a peak-producing source.
        AssertionError: If any element in nodes is not an instance of PipelineNode; if a node declares a parent that is not present in nodes; or if a parent appears after its child in the provided ordering. These assertions indicate structural errors in the pipeline definition that must be corrected before execution.
    """
    from spikeinterface.core.node_pipeline import check_graph
    return check_graph(nodes)


################################################################################
# Source: spikeinterface.core.node_pipeline.find_parent_of_type
# File: spikeinterface/core/node_pipeline.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_core_node_pipeline_find_parent_of_type(
    list_of_parents: list,
    parent_type: tuple
):
    """Find a single parent node of a specified type from a list of pipeline parents.
    
    This function is used within SpikeInterface's node_pipeline utilities to locate an upstream pipeline node (PipelineNode) of a particular class or classes when traversing or inspecting processing/sorting pipelines. In the spike sorting domain this is useful to find a specific kind of parent node (for example a preprocessing, extractor, or analysis node) among a node's parents so downstream logic can access configuration or results produced earlier in the pipeline.
    
    Args:
        list_of_parents (list of PipelineNode): List of parent pipeline nodes to search through. Each element is expected to be a PipelineNode instance representing an upstream component in a spike sorting or preprocessing pipeline. If this argument is None or an empty list, the function will immediately return None.
        parent_type (type | tuple of types): The type or tuple of types to search for among the parents. This should be a Python class (type) or a tuple containing multiple classes. The function uses type-matching semantics (i.e., checking whether a parent is an instance of parent_type) to decide matches. Passing a value that is not a type or a tuple of types may raise a TypeError during the underlying type checks.
    
    Returns:
        PipelineNode or None: The first parent from list_of_parents whose type matches parent_type. If multiple parents match, the parent that appears earliest in list_of_parents is returned. If no matching parent is found, or if list_of_parents is None, the function returns None. The function has no side effects: it does not modify the input list or parent objects; it only inspects them.
    """
    from spikeinterface.core.node_pipeline import find_parent_of_type
    return find_parent_of_type(list_of_parents, parent_type)


################################################################################
# Source: spikeinterface.core.node_pipeline.find_parents_of_type
# File: spikeinterface/core/node_pipeline.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_core_node_pipeline_find_parents_of_type(
    list_of_parents: list,
    parent_type: tuple
):
    """Find all parents of a given type or types from a provided list of pipeline parent nodes used in SpikeInterface node pipelines.
    
    Args:
        list_of_parents (list of PipelineNode): List of parents to search through. In the SpikeInterface node pipeline domain a "parent" is expected to be a PipelineNode instance that represents an upstream step in a processing pipeline (for example preprocessing, filtering, or a sorter component). If this argument is None the function treats it as an empty collection and returns an empty list. The function does not modify this input list; it only reads its elements and returns a new list of matches.
        parent_type (type | tuple of types): The type or tuple of types to match against each parent using Python's isinstance check. In SpikeInterface this is typically used to locate parents of a specific PipelineNode subclass (for example to find upstream nodes implementing a particular processing step). If parent_type is not a valid type or tuple of types, Python's isinstance will raise a TypeError.
    
    Returns:
        list of PipelineNode: A new list containing all elements from list_of_parents for which isinstance(parent, parent_type) is True. The order of matching parents in the returned list preserves their original order in list_of_parents. If list_of_parents is None or no elements match parent_type, an empty list is returned.
    
    Behavior and failure modes:
        This function performs a shallow, non-recursive scan of the provided list_of_parents and uses isinstance to test each element. It does not traverse nested parent relationships, nor does it inspect attributes of PipelineNode objects beyond their type. Because matching uses isinstance, subclasses of the provided parent_type are considered matches. The function is side-effect free: it does not modify list_of_parents or the matched PipelineNode objects and allocates a new list for the results. If elements of list_of_parents are not PipelineNode instances, they will simply not match unless their type matches parent_type; they will not cause modification but may be skipped. If parent_type is not a type or tuple of types, a TypeError will be raised by isinstance. Performance is linear in the length of list_of_parents.
    """
    from spikeinterface.core.node_pipeline import find_parents_of_type
    return find_parents_of_type(list_of_parents, parent_type)


################################################################################
# Source: spikeinterface.core.recording_tools.check_probe_do_not_overlap
# File: spikeinterface/core/recording_tools.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_core_recording_tools_check_probe_do_not_overlap(probes: list):
    """Check that multiple probe objects do not overlap in 2D space so their channel/contact positions can be safely concatenated.
    
    This function is used in the SpikeInterface framework (a unified framework for spike sorting) to verify that several Probe-like objects, each representing a recording probe with spatial contact positions, occupy disjoint axis-aligned rectangular areas in the recorded coordinate frame. It computes the axis-aligned bounding box for each probe from probe.contact_positions (the array of contact coordinates), and then checks pairwise that no contact position from one probe lies within the bounding box of another. This check is commonly performed before concatenating channel positions from multiple probes into a single Recording object or channel map to avoid ambiguous or conflicting channel spatial assignments.
    
    Args:
        probes (list): A list of probe-like objects to check for overlap. Each element must expose a contact_positions attribute that is a numpy.ndarray or array-like with shape (n_contacts, 2) where column 0 is the x coordinate and column 1 is the y coordinate for each contact. All probes must use the same coordinate system and units. The list order determines the pairwise checks (the function compares probes[i] against probes[j] for j > i).
    
    Returns:
        None: The function returns None when the overlap check is successful and has no side effects on the probe objects. It performs only read-only inspections of probe.contact_positions.
    
    Raises:
        Exception: If any probe pair is found to overlap, an Exception is raised with the message "Probes are overlapping! Retrieve locations of single probes separately". Overlap is defined inclusively: a contact lying exactly on the boundary of another probe's axis-aligned bounding box is considered overlapping.
    
    Notes and failure modes:
        - The function uses axis-aligned bounding boxes computed as the minimum and maximum of the x and y columns of contact_positions. It does not consider probe rotations or arbitrary polygonal probe shapes; probes that do not overlap in bounding boxes may still be close but are considered non-overlapping by this check.
        - If a probe object is missing the contact_positions attribute, or if contact_positions is not indexable as a 2D array with at least two columns, the function will raise an AttributeError, IndexError, or a numpy-related TypeError/ValueError coming from the attempted numeric operations. These errors indicate invalid probe input rather than detected spatial overlap.
        - The algorithm performs pairwise comparisons (O(P^2) with P = number of probes) and checks each contact position; for very large numbers of probes or contacts this may have non-negligible computational cost.
        - Ensure all probes use the same spatial reference frame and units before calling this function; mismatched coordinate systems can produce false positives or false negatives.
    """
    from spikeinterface.core.recording_tools import check_probe_do_not_overlap
    return check_probe_do_not_overlap(probes)


################################################################################
# Source: spikeinterface.core.sorting_tools.generate_unit_ids_for_merge_group
# File: spikeinterface/core/sorting_tools.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_core_sorting_tools_generate_unit_ids_for_merge_group(
    old_unit_ids: numpy.ndarray,
    merge_unit_groups: list,
    new_unit_ids: list = None,
    new_id_strategy: str = "append"
):
    """spikeinterface.core.sorting_tools.generate_unit_ids_for_merge_group generates identifiers for newly created merged units during a unit-merging procedure in SpikeInterface sorting workflows. This function is used when curating spike sorting results to combine multiple existing units into single units; it either validates user-provided new ids or produces new ids according to a chosen strategy that is compatible with the dtype of the existing unit ids.
    
    Args:
        old_unit_ids (numpy.ndarray): Array of existing unit identifiers from the sorter (e.g., numpy array of ints or strings). This is used to determine dtype and numeric/string characteristics for automatic id generation. The function converts this input to a numpy array internally and does not mutate the provided array.
        merge_unit_groups (list): A list (or tuple) whose elements are lists/tuples of unit identifiers to be merged together. Each inner list/tuple represents one merge group and must contain at least two members (two or more units to merge). The ordering of groups determines the ordering of produced new ids when ids are auto-generated.
        new_unit_ids (list | None): Optional list of new unit identifiers to use for the merged groups. If provided, its length must equal len(merge_unit_groups). When supplied, the function performs consistency checks only and returns this list unchanged if checks pass. If a supplied new id already exists in old_unit_ids it is allowed only when that id appears inside the corresponding merge group (i.e., the new id is one of the units being merged). The function does not modify the provided list.
        new_id_strategy (str): Strategy for generating new ids when new_unit_ids is None. Accepted values are "append", "take_first", and "join". The default is "append". Behavior:
            "append": Create new ids by appending ids that do not collide with existing ids. If old_unit_ids has integer dtype, new ids are consecutive integers starting at max(old_unit_ids)+1. If old_unit_ids has string dtype and all strings are decimal digits, numeric interpretation is used to find the max numeric id and new string ids are numeric strings starting from that max+1. If old_unit_ids has string dtype but not all entries are digits, generated ids are the literal strings "merge0", "merge1", ... (one per merge group).
            "take_first": Use the first unit id from each merge group as the new id (i.e., the merged unit takes the id of the first member of its group).
            "join": If old_unit_ids are strings, form new ids by joining the members of each group with "-" (e.g., "u1-u2"). If old_unit_ids are numeric, this strategy falls back to the same behavior as "append" for numeric types (consecutive integers starting after the current max).
            Invalid values for new_id_strategy raise a ValueError.
    
    Returns:
        list: A list of new unit identifiers corresponding one-to-one to merge_unit_groups. The elements of the returned list will be of types consistent with decisions above (strings or numeric types derived from old_unit_ids). When new_unit_ids was provided, the same list (after consistency checks) is returned. The function returns a new list and does not modify old_unit_ids or merge_unit_groups.
    
    Behavior notes, defaults, and failure modes:
        - If new_unit_ids is provided and len(new_unit_ids) != len(merge_unit_groups), an AssertionError is raised.
        - If a provided new unit id already exists in old_unit_ids but is not present in the corresponding merge group, an AssertionError is raised to prevent accidental id collisions across unrelated units.
        - If new_unit_ids is None, the function uses new_id_strategy to generate ids; the default strategy is "append".
        - The "join" strategy requires string-like old_unit_ids to produce readable joined ids; if old_unit_ids are numeric, "join" falls back to numeric id generation identical to "append".
        - For string old_unit_ids that are all decimal-digit strings, numeric semantics are used to compute a next numeric id and new ids are returned as digit strings to avoid mixing numeric-seeming ids with arbitrary text.
        - The function raises ValueError for unrecognized new_id_strategy values.
        - The function performs no I/O and has no side effects on input arrays/lists; it only returns a list of identifiers.
    """
    from spikeinterface.core.sorting_tools import generate_unit_ids_for_merge_group
    return generate_unit_ids_for_merge_group(
        old_unit_ids,
        merge_unit_groups,
        new_unit_ids,
        new_id_strategy
    )


################################################################################
# Source: spikeinterface.core.sorting_tools.generate_unit_ids_for_split
# File: spikeinterface/core/sorting_tools.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_core_sorting_tools_generate_unit_ids_for_split(
    old_unit_ids: numpy.ndarray,
    unit_splits: dict,
    new_unit_ids: list = None,
    new_id_strategy: str = "append"
):
    """Function to generate new unit ids used when splitting existing units during spike-sorting curation or automated split operations in SpikeInterface. This function is used by higher-level curation utilities to produce or validate identifiers for the new sub-units created when an existing unit is split (for example, during manual curation or algorithmic post-processing of a SortingExtractor/Sorting object). It either validates user-provided new ids against the requested splits or generates new ids according to a chosen strategy while avoiding id collisions with the existing unit id set.
    
    Args:
        old_unit_ids (numpy.ndarray): Array of existing unit ids before splitting. The function converts this input with numpy.asarray and inspects its dtype to decide how to generate new ids (integer dtype vs. string/character dtype). This array represents the current set of unit identifiers in a sorting result (e.g., sorting.unit_ids) and is used to check for id collisions and to compute new ids that are greater than the current maximum when using the "append" strategy.
        unit_splits (dict): Mapping that describes how existing units should be split. Each key is an existing unit id (matching elements of old_unit_ids) and each value is an iterable (e.g., list or array) of split indices or descriptors that indicate the number of resulting sub-units for that original unit. The function iterates over unit_splits to produce one sublist of new ids per key, in the same iteration order as unit_splits (for Python 3.7+, insertion order is used).
        new_unit_ids (list | None): Optional explicit new unit ids for the splits. If provided, this must be a list where each element is itself a list of new ids corresponding to one entry in unit_splits, and the length of each inner list must equal the number of splits for that unit. When provided, the function performs consistency checks (lengths match and provided ids do not already exist outside the respective split groups) and then returns this list unchanged. If None (the default), the function generates new ids automatically according to new_id_strategy.
        new_id_strategy (str): Strategy to use when generating new unit ids if new_unit_ids is None. Must be either "append" or "split". Default is "append". Behavior:
            - "append": New ids are created so they are numerically or lexicographically greater than the existing max id and are appended to the current set of ids to avoid collisions across multiple splits. For integer-typed old_unit_ids, numeric ids are generated as max(old_unit_ids) + 1, +2, ... using the same integer dtype. For string-typed old_unit_ids, if all existing strings are digit-only (e.g., "1", "2"), numeric string ids are generated by converting to int, taking max + 1, and converting back to strings. If string ids are not all digits, names of the form "{unit_to_split}-split{i}" are generated.
            - "split": New ids are generated by formatting the original unit id and an index as "{unit_to_split}-{i}" (for i in 0..num_splits-1). This strategy only works for non-integer old_unit_ids; if old_unit_ids have an integer dtype and "split" is requested, the function emits a warning and switches to the "append" strategy.
            The function checks new_id_strategy at the start and asserts it is one of the two allowed strings.
    
    Behavior, side effects, defaults, and failure modes:
        - The function asserts that new_id_strategy is either "append" or "split"; otherwise an AssertionError is raised with message "new_id_strategy should be 'append' or 'split'".
        - The function converts old_unit_ids to a numpy array and uses its dtype to choose generation logic. If old_unit_ids cannot be converted to a numpy array, numpy may raise an error.
        - If new_unit_ids is provided, the function performs two consistency checks and raises AssertionError on failure:
            1) For each split group, the provided new ids list length must equal the number of requested splits for that group; otherwise AssertionError with "new_unit_ids should have the same len as unit_splits.values".
            2) Provided new ids must not already exist in old_unit_ids except where they overlap with the same split group; otherwise AssertionError with "new_unit_ids already exists but outside the split groups".
        - If new_unit_ids is None and new_id_strategy == "split" while old_unit_ids are of integer dtype, the function issues a warnings.warn message stating the incompatibility and switches new_id_strategy to "append".
        - When generating ids with "append", the function keeps a running current_unit_ids array that is extended with each set of newly created ids so that subsequent splits use an updated maximum and avoid collisions across multiple generated groups.
        - Generated new ids will be strings when the original dtype is string, or integers (numpy integer type) when the original dtype is integer (except when numeric strings are generated for an all-digit string dtype).
        - The function relies on Python dict iteration order for unit_splits; the order of produced sublists matches the iteration order of unit_splits (unit_splits.items() when generating, unit_splits.values() when validating provided new_unit_ids).
        - Possible exceptions include AssertionError for invalid inputs as described, and standard numpy errors if old_unit_ids cannot be treated as an array.
    
    Returns:
        list of lists: A list where each element is a list of new unit ids for a single original unit split. The outer list is ordered in the same order as iteration over unit_splits (when generating ids) or in the same order as the provided new_unit_ids zipped with unit_splits.values() (when validating provided ids). If new_unit_ids was provided, this returned value is the validated new_unit_ids; if None, the returned value is the newly generated ids according to new_id_strategy.
    """
    from spikeinterface.core.sorting_tools import generate_unit_ids_for_split
    return generate_unit_ids_for_split(
        old_unit_ids,
        unit_splits,
        new_unit_ids,
        new_id_strategy
    )


################################################################################
# Source: spikeinterface.core.sorting_tools.spike_vector_to_indices
# File: spikeinterface/core/sorting_tools.py
# Category: valid
################################################################################

def spikeinterface_core_sorting_tools_spike_vector_to_indices(
    spike_vector: list[numpy.array],
    unit_ids: numpy.array,
    absolute_index: bool = False
):
    """spikeinterface.core.sorting_tools.spike_vector_to_indices: Convert a spike-associated vector (for example spike amplitudes or spike locations) into a nested dictionary of spike indices organized by recording segment and by sorted unit. This function is used in the SpikeInterface spike-sorting workflow to take a global or per-segment vector that aligns one value per spike and split it back into per-unit spike-index arrays so downstream post-processing (quality metrics, waveform extraction, visualization, exporting) can access values for each unit and segment.
    
    Args:
        spike_vector (list[numpy.array]): List of spike vectors obtained with sorting.to_spike_vector(concatenated=False) or a concatenated vector when concatenated=True. Each element of the list corresponds to one recording segment and is expected to be a numpy structured/recarray-like array that contains at least a field named "unit_index" giving, for each spike in that segment (or in the concatenated vector), the integer index of the owning unit (these indices are integer indices into unit_ids, not unit IDs themselves). The practical role of this argument is to provide the per-spike values (amplitude, location, etc.) that must be redistributed to each unit and segment. If this structure is not respected (for example missing "unit_index"), a KeyError or similar will be raised. The function does not modify the provided arrays in-place.
        unit_ids (numpy.array): 1D numpy array of unit identifiers (the external unit IDs used by the sorter). The order and size of this array determine num_units used to allocate per-unit lists: the code uses unit_ids.size to derive the expected number of units and maps computed per-unit indices back to these unit IDs as dictionary keys. Unit IDs are used as keys in the returned per-segment dictionaries and therefore should be unique; if the integer indices found in spike_vector["unit_index"] are outside the range implied by unit_ids.size, an IndexError or incorrect mapping may occur.
        absolute_index (bool): If False (default), spike indices returned for each segment are relative to that segment (i.e., they range from 0 to number_of_spikes_in_segment-1). If True, the function returns absolute spike indices computed by cumulatively offsetting indices across segments (useful when spike_vector is a single concatenated vector covering all segments). The practical significance is that absolute_index=True should be passed when spike_vector represents a unique/concatenated spike vector (so returned indices index into that concatenated vector), whereas absolute_index=False is appropriate when spike_vector is already a list by segment. If absolute_index=True is used incorrectly with per-segment vectors, the indices will be offset and will not align with per-segment arrays.
    
    Returns:
        dict[dict]: A nested dictionary mapping segment_index (int, the segment position in the input spike_vector list) to a dictionary that maps each unit_id (elements of the provided unit_ids numpy.array) to a numpy.ndarray of spike indices (dtype numpy.int64) for that unit in that segment. Practically, the return value is suitable for indexing back into other spike-aligned vectors or arrays: for a given segment s and unit u, spike_indices[s][u] is a numpy array of integer indices pointing to the spikes belonging to unit u (either relative to the segment or absolute across the concatenated vector depending on absolute_index). If a unit has no spikes in a segment, its entry is an empty numpy array. No in-place side effects occur to the inputs; internally the function will use numba-accelerated logic when numba is installed and a numpy fallback otherwise. Failure modes include missing "unit_index" field in spike_vector elements (KeyError), unit_index values out of range relative to unit_ids.size (IndexError or incorrect mapping), and non-integer or malformed unit_index values causing type or conversion errors.
    """
    from spikeinterface.core.sorting_tools import spike_vector_to_indices
    return spike_vector_to_indices(spike_vector, unit_ids, absolute_index)


################################################################################
# Source: spikeinterface.core.sorting_tools.spike_vector_to_spike_trains
# File: spikeinterface/core/sorting_tools.py
# Category: valid
################################################################################

def spikeinterface_core_sorting_tools_spike_vector_to_spike_trains(
    spike_vector: list[numpy.array],
    unit_ids: numpy.array
):
    """spikeinterface.core.sorting_tools.spike_vector_to_spike_trains computes per-segment spike trains for every unit given a spike vector produced by SpikeInterface sorting.to_spike_vector(concatenated=False). It converts a list of per-segment spike records (each containing sample and unit indices) into a dictionary that maps each segment index to an inner dictionary mapping unit identifiers to numpy arrays of spike sample indices. This transformation is commonly used in post-processing, metric computation, exporting results (for example to phy), and visualization workflows within the SpikeInterface framework.
    
    Args:
        spike_vector (list[numpy.array]): List of spike vectors obtained with sorting.to_spike_vector(concatenated=False). Each element corresponds to one recording segment and must be an array-like object indexable with the keys/field names "sample_index" and "unit_index" (e.g., a structured numpy array or dict-like object). "sample_index" must contain integer sample indices that reference the timeline of that segment; "unit_index" must contain integer indices that enumerate units in the range [0, unit_ids.size - 1]. The function converts sample indices to numpy.int64 internally. The practical significance is that this argument provides the raw per-spike information (where and which unit fired) from which per-unit spike trains are reconstructed.
        unit_ids (numpy.array): 1D numpy array of unit identifiers (for example integer or string labels) that correspond, in order, to the integer unit indices used inside spike_vector (unit_index values). The size of this array defines num_units used during reconstruction. The elements of this array are used verbatim as keys in the returned per-segment inner dictionaries; therefore the order and uniqueness of unit_ids determine which spike train is associated with each unit identifier in the result.
    
    Returns:
        dict[dict[str, numpy.array]]: A dictionary keyed by integer segment indices (0-based) where each value is an inner dictionary mapping unit identifiers (the exact values taken from unit_ids) to numpy arrays of spike sample indices (numpy.array with dtype convertible to numpy.int64) representing the spike train for that unit in that segment. Units with no spikes in a segment are present in the inner dictionary and map to an empty numpy array. The sample indices are expressed with the same reference as the input spike_vector (per-segment sample indexing used by the original Recording/Sorting objects). Note: although the annotated inner-key type is shown as str in the return type, the function uses the values from unit_ids verbatim as keys; these keys may be integers, strings, or numpy scalar types depending on unit_ids.
    
    Behavior and side effects:
        This function does not modify the input spike_vector or unit_ids arrays. Internally, if the optional dependency numba is available it will use a numba-accelerated routine for speed; otherwise it falls back to a numpy implementation (vector_to_list_of_spiketrain_numpy). The function casts sample_index and unit_index arrays to numpy.int64 (copy=False where possible) before processing. The runtime is approximately linear in the total number of spikes across all segments.
    
    Failure modes and input validation notes:
        If a per-segment element in spike_vector does not expose "sample_index" or "unit_index" keys/fields, a KeyError (or equivalent indexing error) will be raised. If unit_index values are not integers or contain values outside the range [0, unit_ids.size - 1], the underlying vector-to-spiketrain routine may raise an IndexError, ValueError, or produce incorrect results; therefore ensure unit_index values are zero-based integer indices matching unit_ids ordering. If unit_ids contains duplicate values, keys in the returned inner dictionaries will be overwritten for duplicates, resulting in loss of earlier mappings. Incorrect types for spike_vector elements (non-array-like) or non-1D unit_ids may raise TypeError or ValueError during conversion.
    """
    from spikeinterface.core.sorting_tools import spike_vector_to_spike_trains
    return spike_vector_to_spike_trains(spike_vector, unit_ids)


################################################################################
# Source: spikeinterface.core.sorting_tools.vector_to_list_of_spiketrain_numpy
# File: spikeinterface/core/sorting_tools.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_core_sorting_tools_vector_to_list_of_spiketrain_numpy(
    sample_indices: numpy.ndarray,
    unit_indices: numpy.ndarray,
    num_units: int
):
    """Slower implementation of vector_to_dict for a single recording segment using a NumPy boolean mask. In the SpikeInterface framework this function is used during post-processing of spike sorting outputs to convert a vectorized representation of spikes (sample time indices and corresponding unit labels for one segment) into a Python list of per-unit spike trains (numpy arrays). This is useful when preparing per-unit spike trains for downstream tasks such as waveform extraction, quality metrics computation, visualization, or export.
    
    Args:
        sample_indices (numpy.ndarray): 1D numpy array of sample indices (timepoints) for all detected spikes in one recording segment. Each entry denotes the sample index of one spike measured on the recording timeline. The dtype and integer nature are preserved in the returned per-unit arrays. Length must equal the length of unit_indices.
        unit_indices (numpy.ndarray): 1D numpy array of integer unit labels for each spike in the same order as sample_indices. Each entry assigns the corresponding spike in sample_indices to a unit index. Values are expected to be integers in the range [0, num_units - 1]. Length must equal the length of sample_indices.
        num_units (int): Total number of units (clusters) expected for the segment. This sets the length of the returned list and determines which unit indices are collected. The function iterates u from 0 to num_units - 1 and selects spikes where unit_indices == u.
    
    Returns:
        list[numpy.ndarray]: A Python list of length num_units. Element u of the list is a 1D numpy.ndarray containing the sample indices from sample_indices for which unit_indices == u. The order of spikes within each returned array follows the order they appeared in the input arrays (i.e., original spike order is preserved). If no spikes are assigned to a particular unit index u, the corresponding element is an empty 1D numpy.ndarray.
    
    Behavior and notes:
    - This implementation uses a boolean mask for each unit (sample_indices[unit_indices == u]) and is therefore O(num_units * N) in typical scenarios, which can be slower than grouped or vectorized approaches for large numbers of units or spikes. The code is intentionally simple and clear for the single-segment case.
    - The function performs no in-place modification; it returns new numpy arrays (boolean indexing produces copies) inside a Python list.
    - Preconditions: sample_indices and unit_indices must be 1D arrays of identical length. unit_indices should contain integer labels appropriate for the expected num_units. If these preconditions are violated (for example, mismatched lengths, non-1D inputs, or incompatible shapes), NumPy will raise an indexing or broadcasting error.
    - If unit_indices contains values outside the range [0, num_units - 1], those values will not be collected into any list element; spikes with negative or out-of-range labels will effectively be ignored by this routine (no implicit error is raised), which may lead to silent loss of spikes unless validated beforehand.
    - Use this function when you need a straightforward per-unit list of spike times for a single segment and when clarity is preferred over maximum performance. For large datasets or multiple segments, consider alternative, more optimized grouping utilities in the codebase (e.g., vector_to_dict or other aggregation functions).
    """
    from spikeinterface.core.sorting_tools import vector_to_list_of_spiketrain_numpy
    return vector_to_list_of_spiketrain_numpy(sample_indices, unit_indices, num_units)


################################################################################
# Source: spikeinterface.core.sortinganalyzer.get_default_analyzer_extension_params
# File: spikeinterface/core/sortinganalyzer.py
# Category: valid
################################################################################

def spikeinterface_core_sortinganalyzer_get_default_analyzer_extension_params(
    extension_name: str
):
    """spikeinterface.core.sortinganalyzer.get_default_analyzer_extension_params: Retrieve the default parameter values declared by a SortingAnalyzer extension's _set_params method.
    
    Args:
        extension_name (str): The registered name of an analyzer extension within SpikeInterface's SortingAnalyzer extension registry. In the SpikeInterface domain, extensions implement additional analysis functionality on sorting outputs; extension_name is used to locate the extension class via get_extension_class(extension_name). This function expects the extension to define a class method or instance method named _set_params whose signature contains parameters with optional default values.
    
    Returns:
        dict: A mapping from parameter name (str) to its default value for every parameter in the extension class's _set_params signature that has an explicit default. The returned dict is suitable for use when constructing or initializing analyzer extensions, populating user interfaces with sensible defaults, or programmatically determining which parameters are optional. Parameters named "self" and any parameters without an explicit default (inspect.Parameter.empty) are excluded from the returned mapping. If no parameters with defaults are present, an empty dict is returned.
    
    Behavior and side effects:
        The function performs pure introspection and has no side effects on recorded data or global state. It calls get_extension_class(extension_name) to obtain the extension class, then inspects the signature of extension_class._set_params using inspect.signature and collects default values. It does not validate default value types beyond extracting them from the signature, nor does it call _set_params itself. The function is intended for use in the SortingAnalyzer workflow within SpikeInterface to obtain defaults for analyzer extension configuration and to support GUI or programmatic parameter handling.
    
    Failure modes:
        If extension_name is not registered, get_extension_class(extension_name) will propagate whatever exception it raises (for example, a KeyError or a custom registry-related error) indicating the extension was not found. If the located extension class does not expose an attribute _set_params, an AttributeError will be raised. inspect.signature may raise a ValueError or TypeError if the _set_params object is not introspectable; those exceptions are propagated. Users should catch these exceptions if extension names may be invalid or extension implementations nonconformant.
    """
    from spikeinterface.core.sortinganalyzer import get_default_analyzer_extension_params
    return get_default_analyzer_extension_params(extension_name)


################################################################################
# Source: spikeinterface.core.sortinganalyzer.get_extension_class
# File: spikeinterface/core/sortinganalyzer.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_core_sortinganalyzer_get_extension_class(
    extension_name: str,
    auto_import: bool = True
):
    """spikeinterface.core.sortinganalyzer.get_extension_class: Retrieve an extension class by name and verify it is registered with the internal extension registry used by the SortingAnalyzer API in SpikeInterface.
    
    This function looks up a registered extension class by its declared extension_name in the global registry (_possible_extensions). It is used throughout the SortingAnalyzer and extension machinery of SpikeInterface to locate the concrete class that implements an extension (for example, post-processing, metrics, or other SortingAnalyzer extensions) so that the analyzer can instantiate or query that extension. If the requested extension is not yet registered but is a known builtin extension (listed in _builtin_extensions), the function can optionally import the module that provides and registers the extension, updating the global registry as a side effect. The function therefore both performs a registry lookup and may perform dynamic import to ensure builtin extensions become available to the SortingAnalyzer workflow.
    
    Args:
        extension_name (str): The name identifier of the extension to retrieve. This must match the extension.extension_name attribute of a class that has been registered in the global _possible_extensions registry. In practical SpikeInterface use, this name corresponds to the string used to refer to an extension when interacting with SortingAnalyzer (for example, requesting a particular post-processing or metric extension).
        auto_import (bool): If True (default), and the extension_name is not currently registered but exists in the _builtin_extensions mapping, the function will import the corresponding module via importlib.import_module(module). This import has the side effect of allowing that module to run its registration code and update the global _possible_extensions registry so the extension becomes available. If False, the function will not import modules automatically and will raise a ValueError instructing the caller to import the related module manually before use. Note that importing may be relatively expensive and can raise ImportError or other import-time exceptions if the module or its dependencies are not available.
    
    Returns:
        ext_class (type): The Python class object that implements the requested extension. This class is the registered extension class whose extension_name attribute equals the extension_name argument and is intended to be used by SortingAnalyzer to create extension instances, call class methods, or inspect extension capabilities.
    
    Raises:
        ValueError: If the extension_name is not registered and is not present in the _builtin_extensions mapping, a ValueError is raised indicating the extension is unknown (suggesting it may be an external extension or a typo). If the extension_name is a known builtin but not registered and auto_import is False, a ValueError is raised advising to import the related module first (message suggests "import {module}").
        ImportError (or other import-time exceptions): If auto_import is True and importlib.import_module(module) fails, the underlying import exception is propagated, indicating the builtin extension module or one of its dependencies could not be imported.
    
    Side effects:
        When auto_import is True and a builtin extension module is imported, the global _possible_extensions registry is refreshed/updated by that module's registration logic, and subsequent calls can find the newly registered extension. This function does not persist changes beyond the current Python process; it only affects in-memory registration state used by SortingAnalyzer within the running SpikeInterface session.
    """
    from spikeinterface.core.sortinganalyzer import get_extension_class
    return get_extension_class(extension_name, auto_import)


################################################################################
# Source: spikeinterface.core.sortinganalyzer.load_sorting_analyzer
# File: spikeinterface/core/sortinganalyzer.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_core_sortinganalyzer_load_sorting_analyzer(
    folder: str,
    load_extensions: bool = True,
    format: str = "auto",
    backend_options: dict = None
):
    """Load a SortingAnalyzer object from a folder on disk or in cloud storage so it can be used by SpikeInterface code to inspect, post-process, and compute quality metrics on spike sorting outputs.
    
    Args:
        folder (str or Path): The folder or zarr folder where the analyzer is stored. This is the filesystem path or remote URL that contains the saved SortingAnalyzer data. In the SpikeInterface domain this folder was previously produced by Saving/Export routines and contains the analyzer metadata, sorting results references, and optional extensions. If the folder is remote (for example an S3 or other fsspec-accessible path), backend_options can supply credentials; if credentials are not provided and the remote storage supports anonymous access, the loader will attempt to open the folder in anonymous mode (anon=True).
        load_extensions (bool): Whether to load all extensions associated with the saved SortingAnalyzer. Default True. Extensions are additional datasets attached to a SortingAnalyzer that provide extra annotations, metrics, or derived data needed for post-processing and quality assessment; loading them may require additional I/O and memory. Set this to False to avoid loading extensions when only core sorting metadata is required.
        format ("auto" | "binary_folder" | "zarr"): The on-disk serialization format of the analyzer folder. "auto" (default) instructs the loader to detect the format from the folder contents. Use "binary_folder" when the analyzer was saved as a collection of binary/JSON files, or "zarr" when saved as a zarr hierarchy. Providing an incorrect format may cause the loader to fail with a ValueError or similar error indicating an unrecognized layout.
        backend_options (dict | None): Backend options passed to the underlying storage backend (for example fsspec). Default None. When provided, this dictionary can include keys such as "storage_options" (a dict of fsspec-compatible storage options, e.g., credentials, anon flags, endpoint URLs) and "saving_options" (a dict of additional saving or opening options used when datasets were created). Use this to supply cloud credentials, custom endpoints, or other backend-specific configuration required to access remote folders.
    
    Returns:
        SortingAnalyzer: An instance of SortingAnalyzer loaded from the specified folder. This returned object is the in-memory representation used by SpikeInterface to analyze and post-process spike sorting outputs (compute quality metrics, inspect waveforms, manage extensions, etc.). The returned object is created by calling SortingAnalyzer.load(...) and will reflect the state stored on disk, including any loaded extensions when load_extensions=True.
    
    Behavior and side effects:
        This function performs I/O to read metadata and (optionally) extensions from the provided folder. For remote folders it uses fsspec-style backends and will attempt anonymous access if backend_options are not supplied and the remote endpoint allows it. Loading extensions can incur additional network or disk reads and can increase memory usage; set load_extensions=False to avoid that overhead if only basic metadata is needed. When format="auto", the function inspects the folder layout to choose the appropriate loader. The function delegates loading to SortingAnalyzer.load(folder, load_extensions=..., format=..., backend_options=...), so any additional behavior, validation, or errors raised by that method (for example FileNotFoundError if the folder is inaccessible, or ValueError for an unrecognized format) will propagate to the caller.
    
    Failure modes:
        If the folder does not exist or is not reachable with the provided backend_options, a filesystem-related exception (for example FileNotFoundError or an fsspec/backend-specific exception) will be raised. If the format cannot be detected when format="auto" or if a specified format does not match the folder contents, the loader will raise an error (typically ValueError). Providing malformed backend_options may raise TypeError or backend-specific configuration errors. Loading large extensions may fail due to memory limits or network timeouts depending on environment and backend configuration.
    """
    from spikeinterface.core.sortinganalyzer import load_sorting_analyzer
    return load_sorting_analyzer(folder, load_extensions, format, backend_options)


################################################################################
# Source: spikeinterface.core.waveform_tools.split_waveforms_by_units
# File: spikeinterface/core/waveform_tools.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_core_waveform_tools_split_waveforms_by_units(
    unit_ids: list,
    spikes: numpy.ndarray,
    all_waveforms: numpy.ndarray,
    sparsity_mask: numpy.ndarray = None,
    folder: str = None
):
    """split_waveforms_by_units splits a single contiguous buffer of extracted waveforms into per-unit waveform collections. In SpikeInterface workflows (spike sorting, waveform extraction and postprocessing), this function is used to separate a global waveform buffer that contains waveforms for all detected spikes into a dictionary keyed by unit identifier or, alternatively, to save each unit's waveforms to separate .npy files on disk and return memory-mapped views. The splitting uses the integer unit index stored in the provided spikes vector to group waveforms for each unit id in the same order as unit_ids.
    
    Args:
        unit_ids (list or numpy array): List or array of unit identifiers. Each element is a unit identifier (e.g., integer or string) whose order corresponds to the integer unit indices referenced in the spikes array. The function enumerates unit_ids and uses the enumeration index (0-based) to match spikes["unit_index"] entries; therefore the practical significance is that unit_ids defines the mapping from integer unit_index values found in spikes to the external unit identifiers used in SpikeInterface pipelines.
        spikes (numpy array): Structured numpy array or record array containing spike metadata for all waveforms in all_waveforms. This array must include an integer field named "unit_index" such that spikes["unit_index"] == unit_index selects the rows (spikes) that belong to the unit at position unit_index in unit_ids. The function relies on this field to group rows of all_waveforms by unit; missing or incorrectly typed "unit_index" will raise KeyError or produce incorrect grouping.
        all_waveforms (numpy array): Single contiguous numpy array buffer that contains the waveform samples for every spike in spikes. The array is indexed in the code as all_waveforms[mask, :, :] where mask selects spikes for a unit, so in practice it must be a 3-dimensional array with the first axis aligned to the rows of spikes (all spikes), and the remaining axes holding sample/time and channel dimensions produced by waveform extraction. The practical role is to provide the full set of extracted waveforms so they can be split by unit without re-extracting from the recording.
        sparsity_mask (None or numpy array): Optional boolean numpy array that encodes channel sparsity per unit (for example produced by a sparsity calculation where some channels are omitted for a unit). When provided, the function uses sparsity_mask[unit_index, :] to determine how many channels are stored per unit and slices the last dimension of all_waveforms to keep only the stored channels (it sums the True values to compute num_chans and uses the first num_chans channels of the stored buffer). If None (default), no per-unit channel filtering is applied and the full channel dimension from all_waveforms is preserved.
        folder (None or str or Path): Optional target folder path. If None (default), the function returns per-unit waveforms as in-memory numpy arrays. If a folder is provided, the function will save each unit's waveforms to an .npy file named "waveforms_{unit_id}.npy" using numpy.save and then replace the in-memory array with a numpy.load of that file using mmap_mode="r" so the returned value is a memory-mapped array that avoids keeping all waveforms in memory simultaneously. Side effects: the function attempts to convert folder to a Path and will write files to that directory; errors writing files (e.g., non-existing directory, permission denied) will raise exceptions from Path conversion or numpy.save. Note: the implementation saves files to folder but the subsequent numpy.load call uses a filename without the folder path; this can cause load failures or incorrect file resolution if the working directory differs from folder—callers should ensure the working directory and folder usage are consistent to avoid this failure mode.
    
    Returns:
        waveforms_by_units (dict of array): A dict keyed by unit identifier (elements from unit_ids) whose values are numpy arrays containing the waveforms for that unit. If folder is None, each value is an in-memory numpy array (the subset of all_waveforms for that unit, optionally channel-sliced when sparsity_mask is provided). If folder is not None, the values are numpy.load results with mmap_mode="r" (memory-mapped numpy arrays pointing to the saved "waveforms_{unit_id}.npy" files) intended to reduce RAM usage. Possible failure modes include KeyError if spikes lacks "unit_index", ValueError or IndexError when unit_ids length and unit_index values do not correspond, and IO-related exceptions when saving/loading files to/from folder.
    """
    from spikeinterface.core.waveform_tools import split_waveforms_by_units
    return split_waveforms_by_units(
        unit_ids,
        spikes,
        all_waveforms,
        sparsity_mask,
        folder
    )


################################################################################
# Source: spikeinterface.core.zarrextractors.get_default_zarr_compressor
# File: spikeinterface/core/zarrextractors.py
# Category: valid
################################################################################

def spikeinterface_core_zarrextractors_get_default_zarr_compressor(clevel: int = 5):
    """Return a configured Zarr/Blosc compressor optimized for storing int16
    electrophysiology recordings used in SpikeInterface.
    
    This factory function constructs and returns a numcodecs Blosc compressor
    configured for good performance when saving extracellular electrophysiology
    data (commonly int16 samples) to Zarr stores. The configuration mirrors the
    defaults used in SpikeInterface for Zarr-backed recording storage: codec name
    "zstd" (zstandard) for efficient compression ratio, and BITSHUFFLE to improve
    compressibility of small integer samples. The clevel parameter controls the
    trade-off between compression ratio and CPU cost when writing data: higher
    values generally produce smaller files but require more CPU time.
    
    Args:
        clevel (int): Compression level passed directly to numcodecs.Blosc.
            Higher values typically increase compression ratio at the cost of
            increased CPU usage when compressing. Documented minimum is 1 and
            maximum is 9; the function does not perform explicit range enforcement,
            so values outside this range may be rejected or handled by the
            underlying numcodecs/Blosc implementation and can raise an exception.
            Default is 5. In the SpikeInterface domain this parameter is used when
            saving recordings to Zarr to balance storage size and write performance
            for typical extracellular, int16-formatted datasets.
    
    Returns:
        Blosc.compressor: A configured numcodecs.Blosc compressor instance with
        cname="zstd", shuffle=Blosc.BITSHUFFLE, and clevel set to the provided
        value. This object is suitable to pass as the compressor argument to Zarr
        storage functions (for example, the SpikeInterface save-to-Zarr utilities)
        and is tuned for good compression performance on int16 electrophysiology
        data. No other side effects occur. Possible runtime errors include ImportError
        if numcodecs is not available or codec-specific errors raised by Blosc if
        an invalid clevel or unsupported configuration is supplied.
    """
    from spikeinterface.core.zarrextractors import get_default_zarr_compressor
    return get_default_zarr_compressor(clevel)


################################################################################
# Source: spikeinterface.curation.auto_merge.binom_sf
# File: spikeinterface/curation/auto_merge.py
# Category: valid
################################################################################

def spikeinterface_curation_auto_merge_binom_sf(x: int, n: float, p: float):
    """spikeinterface.curation.auto_merge.binom_sf computes the binomial survival function (sf = 1 - cdf) for a given observed count of successes, using integer-valued binomial survival probabilities computed at nearby integer trial counts and a quadratic interpolation to support non-integer trial inputs. In the SpikeInterface curation/auto_merge context, this function is used to quantify the tail probability that a binomial random variable with probability p would exceed x successes across approximately n trials; this probability can be used as a statistical criterion when automatically deciding whether to merge putative units (for example, when evaluating the significance of coincident spike counts between units).
    
    Args:
        x (int): The observed number of successes. In spike-sorting/curation use this represents a raw count (for example, number of coincident spikes). It must be an integer as given in the function signature; typical practical values are non-negative and at most comparable to the number of trials n, although the underlying scipy.stats.binom.sf behavior governs out-of-range values.
        n (float): The (possibly non-integer) number of trials. The implementation builds an integer grid n_array = arange(floor(n - 2), ceil(n + 3), 1) filtered to non-negative integers and evaluates the binomial survival function at each integer n_ in that grid; a quadratic interpolation (scipy.interpolate.interp1d with kind='quadratic') is then used to estimate the survival function at the supplied non-integer n. Supplying n as a float allows using this interpolated approximation when an exact integer trial count is not available (e.g., when deriving an effective number of independent comparisons in curation heuristics). Note that if the constructed n_array contains fewer than three distinct integer points (for example, when n is very small), the quadratic interpolation will fail and a ValueError will be raised.
        p (float): The probability of success for each trial, interpreted as the binomial parameter p in [0, 1]. In practice this represents the expected per-trial coincidence probability or similar statistic used in merging criteria. Values of p outside [0, 1] are not valid probabilities and will produce undefined or error behavior from the underlying SciPy routines.
    
    Returns:
        float: The interpolated survival function value sf = 1 - cdf(x) for a Binomial(n, p) model evaluated at the provided n (via quadratic interpolation over nearby integer n_). This return value is the tail probability P(X > x) for X ~ Binomial(n, p) as approximated by the integer-grid computation and interpolation. The function has no other side effects, but it depends on SciPy (scipy.stats and scipy.interpolate) and NumPy/math; calling the function will raise ImportError if SciPy is not available and may raise ValueError from the interpolation step when insufficient grid points exist or when inputs are invalid (for example, p outside [0,1] or very small n leading to too few integer points).
    """
    from spikeinterface.curation.auto_merge import binom_sf
    return binom_sf(x, n, p)


################################################################################
# Source: spikeinterface.curation.auto_merge.estimate_contamination
# File: spikeinterface/curation/auto_merge.py
# Category: valid
################################################################################

def spikeinterface_curation_auto_merge_estimate_contamination(
    spike_train: numpy.ndarray,
    sf: float,
    T: int,
    refractory_period: tuple[float, float]
):
    """Estimate the contamination of a spike train by counting refractory period violations and converting that count into a contamination fraction used in spike sorting curation.
    
    Args:
        spike_train (numpy.ndarray): The unit's spike train as a 1-D numpy array of spike times expressed in sample indices (integer sample timestamps). In the context of SpikeInterface (a unified framework for spike sorting), this array represents the detected spike events for one unit. The function will cast this array to numpy.int64 internally before computing violations; the original array is not modified.
        sf (float): The sampling frequency (in Hz) of the spike train. This value is used to convert refractory period limits given in milliseconds to sample counts (t_c and t_r are computed as refractory_period[*] * 1e-3 * sf). sf must be provided in the same temporal units expected for conversion to samples.
        T (int): The duration of the spike train in samples (integer). This is the total number of samples in the recording epoch used to normalize the contamination estimate; providing an incorrect T (for example smaller than the largest spike index) will yield an incorrect estimate or may trigger runtime errors.
        refractory_period (tuple[float, float]): The censored and refractory period (t_c, t_r) used, expressed in milliseconds. The first element is the censored time t_c (ms) and the second is the refractory time t_r (ms). These two values are converted to sample counts via sf and are used to define the time windows for counting violations and computing the contamination metric.
    
    Returns:
        float: The estimated contamination as a floating-point fraction between 0 and 1. This value quantifies the fraction of spikes that are likely contaminating events (spikes not generated by the neuron of interest) based on refractory period violations. A result of 0.0 indicates no estimated contamination under the model; a result of 1.0 indicates maximum contamination. Internally the function computes:
            t_c = refractory_period[0] * 1e-3 * sf
            t_r = refractory_period[1] * 1e-3 * sf
            n_v = compute_nb_violations(spike_train.astype(numpy.int64), t_r)
            N = len(spike_train)
            D = 1 - n_v * (T - 2 * N * t_c) / (N**2 * (t_r - t_c))
        and returns 1.0 if D < 0 else 1 - math.sqrt(D).
    
    Behavior, side effects, defaults, and failure modes:
        This function performs a pure calculation and has no external side effects (it does not modify files or global state). It casts spike_train to numpy.int64 internally before calling compute_nb_violations, so inputs that cannot be losslessly converted to integer sample indices may produce unexpected results. The function assumes refractory_period is given in milliseconds and sf is in Hz; providing inconsistent units will produce incorrect conversions and results. If spike_train is empty (N == 0), or if t_r == t_c, the denominator N**2 * (t_r - t_c) leads to division by zero and will raise a runtime error; similarly, negative or nonsensical refractory_period values can yield invalid intermediate values. If D computed from the formula is negative (which indicates an estimate outside the model's valid range), the implementation clamps the contamination to 1.0. This contamination estimate is intended as a heuristic metric used during automated curation and merging of units in spike sorting workflows; interpret values in that practical context and verify against additional quality metrics when making curation decisions.
    """
    from spikeinterface.curation.auto_merge import estimate_contamination
    return estimate_contamination(spike_train, sf, T, refractory_period)


################################################################################
# Source: spikeinterface.curation.auto_merge.get_unit_adaptive_window
# File: spikeinterface/curation/auto_merge.py
# Category: valid
################################################################################

def spikeinterface_curation_auto_merge_get_unit_adaptive_window(
    auto_corr: numpy.ndarray,
    threshold: float
):
    """Computes an adaptive window size from a correlogram for unit curation/auto-merge workflows in SpikeInterface. This function identifies the first relevant peak in the correlogram (interpreted as the first peak nearest to the center) by locating peaks in the negative second derivative of the correlogram, filtering those peaks by a minimum amplitude threshold, and selecting the last qualifying peak before the correlogram center. The resulting window size is computed as the distance (in array indices / samples) from that peak to the correlogram center and is intended to be used by downstream curation logic (for example, to define the temporal window for deciding whether two units should be merged based on their cross-correlogram).
    
    Args:
        auto_corr (numpy.ndarray): 1-D numpy array containing the correlogram values used to compute the adaptive window. The correlogram is expected to be centered such that the central bin index is auto_corr.shape[0] // 2. Values represent correlogram amplitudes (e.g., counts or normalized correlation) across time lags. This array is processed by computing its second derivative (via numpy.gradient) and finding peaks in the negated second derivative using scipy.signal.find_peaks; incorrect dimensionality (non-1D) or non-numeric contents may produce exceptions or undefined behavior.
        threshold (float): Minimum amplitude threshold used to filter candidate peaks in the correlogram. Only peaks whose correlogram amplitude at the peak index is greater than or equal to this threshold are kept. If no peaks meet this threshold, the function will recursively retry with threshold / 2 until either a peak is found or the threshold falls below 1e-5. The threshold represents an absolute amplitude cutoff on auto_corr values as used in curation decisions.
    
    Returns:
        int: The adaptive window size expressed as an integer number of indices/samples. This is computed as (auto_corr.shape[0] // 2) - p where p is the last (nearest-to-center) peak index that satisfies the threshold and lies before the correlogram center. This integer is intended to be used as a window radius for merging/curation decisions.
        Special-case behaviors and failure modes:
        - If the correlogram is entirely zero (numpy.sum(numpy.abs(auto_corr)) == 0), the function returns 20.0 (a heuristic default used in the code path). Callers that strictly require an int should cast this value, but be aware the implementation currently returns a float in this degenerate case.
        - If no peaks pass the threshold and the threshold is already below 1e-5, the function returns 1 (an integer fallback window).
        - If no peaks pass the threshold and threshold >= 1e-5, the function recursively calls itself with threshold / 2 until a peak is found or the 1e-5 base case is reached.
        - There are no external side effects (the function does not modify inputs or external state); however, it uses recursion and scipy.signal.find_peaks, and may raise typical numpy/scipy exceptions if auto_corr has incompatible shape or types.
    """
    from spikeinterface.curation.auto_merge import get_unit_adaptive_window
    return get_unit_adaptive_window(auto_corr, threshold)


################################################################################
# Source: spikeinterface.curation.auto_merge.normalize_correlogram
# File: spikeinterface/curation/auto_merge.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for normalize_correlogram because the docstring has no description for the argument 'correlogram'
################################################################################

def spikeinterface_curation_auto_merge_normalize_correlogram(correlogram: numpy.ndarray):
    """spikeinterface.curation.auto_merge.normalize_correlogram normalizes a correlogram so its mean over time bins is 1.
    
    Normalizes an input correlogram array used in spike-sorting curation (for example in automated unit merging) by dividing all values by the arithmetic mean computed across the correlogram's elements (the mean across time bins / lag values). This ensures different correlograms are placed on a common scale so that their mean activity across time is 1, which is useful when comparing shapes of cross-correlograms or autocorrelograms between units during the auto-merge curation workflow in SpikeInterface. If the correlogram is all zeros, the function leaves it unchanged to avoid division by zero.
    
    Args:
        correlogram (numpy.ndarray): Correlogram to normalize. This is a NumPy array representing counts or rates across time lags (time bins) for an autocorrelogram or cross-correlogram between units. The function computes the arithmetic mean across all elements of this array (effectively the mean over time bins). The correlogram should contain numeric values; if it contains NaNs or infinities, the computed mean and the result may be NaN or infinite. If correlogram has a non-numeric dtype that prevents computing a mean, numpy will raise an error.
    
    Returns:
        normalized_correlogram (numpy.ndarray) [time]: A NumPy array of the same shape as the input correlogram containing the normalized correlogram. If the input mean is non-zero, the returned array equals correlogram / mean and therefore has an arithmetic mean of 1 across its elements (time bins). If the input correlogram is zero everywhere (mean == 0), the original array object is returned unchanged to avoid division by zero. Note that when division occurs, numpy will produce a new array (and integer inputs will be cast to a floating dtype); when mean == 0 the original input object is returned without modification. Possible failure modes include propagation of NaNs or infinities if present in the input, or an exception from numpy.mean for non-numeric dtypes.
    """
    from spikeinterface.curation.auto_merge import normalize_correlogram
    return normalize_correlogram(correlogram)


################################################################################
# Source: spikeinterface.curation.auto_merge.resolve_pairs
# File: spikeinterface/curation/auto_merge.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_curation_auto_merge_resolve_pairs(
    existing_merges: dict,
    new_merges: dict
):
    """Resolve nested unit-merge mappings produced when merging units recursively during curation.
    
    This function is used primarily by auto_merge_units within the SpikeInterface curation workflow to produce a condensed, non-nested representation of all unit merges that have been applied so far. The returned mapping is suitable for downstream visualization (for example, the plot_potential_merge widget) and for producing a final list of merged unit identifiers for post-processing and quality-metric calculations. In the SpikeInterface domain, "unit ids" refer to the identifiers used by a SortingExtractor (integers or strings identifying sorted units), and the mappings record which unit ids were merged into which surviving unit id.
    
    Args:
        existing_merges (dict): A dictionary representing merges that have already been applied. The keys are unit ids (the surviving unit id after a previous merge) and the values are lists of unit ids that were merged into that key. For example, an entry {A: [B, C]} means units B and C were merged into unit A. This function expects a dict object; passing None has a documented behavior described below. The function performs a shallow copy of this dict at start, so the top-level mapping passed in will not be mutated, but the lists inside the mapping are not deep-copied by this function.
        new_merges (dict): A dictionary representing new merges to incorporate. The keys are the new surviving unit ids produced by the latest merge step and the values are lists of unit ids to merge into that key. For example, {D: [A, E]} means units A and E should be merged into unit D. The function will inspect these lists to detect references to keys already present in existing_merges and will expand those references by replacing referenced keys with their constituent unit lists from existing_merges. The function expects new_merges to be a dict and note that the lists provided as values in new_merges may be mutated in-place by this function (they are extended and elements removed when nested merges are resolved).
    
    Returns:
        dict: A dictionary mapping surviving unit ids to flattened lists of unit ids that have been merged into them, with nested merges resolved. If existing_merges is None, the function returns new_merges.copy() (a shallow copy of the new_merges dictionary). When existing_merges is provided, the function returns a shallow copy of existing_merges updated to incorporate new_merges: any new merge that references a previously merged key will have that key removed and replaced by the list of unit ids that were previously merged into it. The returned dict uses the same unit id objects (keys and list elements) as provided in the inputs (no deep copy), so callers should copy input lists first if they need to preserve them unchanged.
    
    Behavior and side effects:
        The function resolves nested merges by checking, for each new merge entry, whether any element of the new merge's list references a key present in existing_merges. If no such reference exists, the new entry is added to the resolved mapping. If references exist, each referenced existing key is removed from the new merge's value list and replaced by the list of unit ids previously merged into that existing key; the existing key is removed from the resolved mapping. The algorithm performs shallow dict copies but does not deep-copy the lists of unit ids; therefore, lists taken from new_merges may be modified in-place (items removed and new items appended). The function returns a mapping suitable for visualization and further curation steps within SpikeInterface.
    
    Failure modes and edge cases:
        The function expects both existing_merges and new_merges to be dicts (or existing_merges may be None). Passing other types may raise TypeError or lead to unpredictable behavior. Because list values are manipulated in-place, callers that reuse the same list objects elsewhere should defend by providing copied lists if they must remain unchanged. If lists in new_merges contain duplicate references to the same existing key, the function will remove only one occurrence per detected reference and will append the corresponding previous-merge list once per detected reference; this may lead to duplicate unit ids in the resulting lists if duplicates were present in inputs. The function does not perform validation of unit id types or uniqueness beyond list membership checks; it assumes that unit ids in keys and values correspond to identifiers used by SortingExtractor and by auto_merge_units.
    """
    from spikeinterface.curation.auto_merge import resolve_pairs
    return resolve_pairs(existing_merges, new_merges)


################################################################################
# Source: spikeinterface.curation.auto_merge.smooth_correlogram
# File: spikeinterface/curation/auto_merge.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_curation_auto_merge_smooth_correlogram(
    correlograms: numpy.ndarray,
    bins: numpy.ndarray,
    sigma_smooth_ms: float = 0.6
):
    """spikeinterface.curation.auto_merge.smooth_correlogram smooths cross-correlograms by convolving them with a Gaussian kernel. This function is used in the automatic curation/auto-merge workflow of SpikeInterface to reduce high-frequency noise in computed cross-correlograms (time-lag histograms between spike trains) so that downstream heuristics or metrics that decide whether two units should be merged are more robust. The smoothing is implemented by constructing a normalized Gaussian kernel from the provided bins and sigma (in milliseconds) and applying a fast Fourier-based convolution (scipy.signal.fftconvolve) along the correlogram time-lag axis.
    
    Args:
        correlograms (numpy.ndarray): Array containing one or more correlograms to be smoothed. The function expects a multi-dimensional array where the time-lag bins are on axis 2 (i.e., correlograms.shape[2] must equal the length of bins). In the SpikeInterface auto-merge context this is typically a 3D array (for example, number of correlograms x number of channels/units x number of time-lag bins). Each element represents counts or rates for a given time lag and will be convolved along axis 2. The input array is not modified in-place; a new array is returned.
        bins (numpy.ndarray): 1D array of time-lag bin centers/positions corresponding to the third axis of correlograms. Units must match sigma_smooth_ms (milliseconds). The Gaussian smoothing kernel is computed using these bin coordinates as kernel support (kernel = exp(-bins**2 / (2 * sigma_smooth_ms**2))). The length of bins must equal correlograms.shape[2]; otherwise scipy.signal.fftconvolve will raise an error.
        sigma_smooth_ms (float): Standard deviation of the Gaussian smoothing kernel, expressed in milliseconds. Default is 0.6. This parameter controls the amount of temporal smoothing: larger values produce broader, stronger smoothing across neighboring time-lag bins. Must be a positive, non-zero float; extremely small values may produce negligible smoothing and very large values may oversmooth features important for merging decisions.
    
    Returns:
        numpy.ndarray: A new array with the same shape as correlograms containing the smoothed correlograms. Smoothing is performed by constructing a normalized Gaussian kernel from bins and sigma_smooth_ms, reshaping it to (1, 1, n_bins) to broadcast over the first two axes, and applying scipy.signal.fftconvolve with mode="same" and axes=2 so the output aligns with the input time-lag bins. If correlograms has length zero (len(correlograms) == 0), an empty array with the same shape is returned (dtype float64) because fftconvolve cannot produce the correct shape for empty inputs. No in-place modification of the input is performed.
    
    Notes:
        - The function relies on scipy.signal.fftconvolve; for very large arrays this may incur noticeable memory and computation cost.
        - Ensure bins are centered and expressed in milliseconds to match sigma_smooth_ms; mismatched units will produce incorrect smoothing.
        - If bins length does not match correlograms.shape[2], scipy will raise an error during convolution.
        - The Gaussian kernel is normalized to sum to 1 so that overall correlogram counts are preserved (apart from edge effects introduced by mode="same").
    """
    from spikeinterface.curation.auto_merge import smooth_correlogram
    return smooth_correlogram(correlograms, bins, sigma_smooth_ms)


################################################################################
# Source: spikeinterface.curation.model_based_curation.load_model
# File: spikeinterface/curation/model_based_curation.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_curation_model_based_curation_load_model(
    model_folder: str = None,
    repo_id: str = None,
    model_name: str = None,
    trust_model: bool = False,
    trusted: list = None
):
    """Load a serialized model and its accompanying metadata (model_info) either from a local folder or from a HuggingFace Hub repository. This function is used within the SpikeInterface curation workflow to obtain a trained model and metadata that can be applied to automated, model-based curation of spike-sorting outputs (for example, to score or classify units based on learned features). The returned model is the deserialized object ready for inference (commonly a scikit-learn estimator, pipeline, or other Python object serialized with skops), and model_info contains metadata about the model (for example, training provenance, feature definitions, or versioning) that downstream curation code can inspect to ensure compatibility.
    
    Args:
        model_folder (str or Path): The path to a local folder that contains the serialized model file(s). If provided, the function will attempt to load the model from the local filesystem using internal helper _load_model_from_folder. If model_folder is None and repo_id is provided, the function will instead fetch the model from the HuggingFace Hub. If both model_folder and repo_id are None, the function raises ValueError. Supplying both model_folder and repo_id is also invalid and raises ValueError.
        repo_id (str | Path): Identifier of a HuggingFace Hub repository containing the model (for example 'username/model'). When repo_id is provided (and model_folder is None), the function will attempt to download the model artifact from the HuggingFace Hub using internal helper _load_model_from_huggingface; this implies network access and possible download side effects (written to cache or temporary files depending on the underlying HuggingFace client). If the repository or artifact cannot be reached, a network or I/O-related exception may be raised.
        model_name (str | Path): The filename of the serialized model artifact to load (for example 'my_model.skops'). If model_name is None, the loader will use the first model file it finds in the target location (local folder or HuggingFace repo). Providing a model_name narrows which file is deserialized; if the named file is not present a FileNotFoundError or equivalent I/O error will be raised by the helper loader.
        trust_model (bool): Whether to trust the model's serialized contents when deserializing with skops.load. If True, the function will attempt to automatically infer a safe `trusted` list to pass to skops.load so that common, expected types are allowed to be deserialized. If False (the default), the caller must provide an explicit `trusted` list to indicate which object types/names are allowed; otherwise skops.load will refuse to load objects that are not marked trusted. This flag controls security behavior to avoid executing or instantiating untrusted code during deserialization and thus affects safety of loading unverified model files.
        trusted (list): A list of strings passed directly to skops.load that names the allowed objects/types in the serialized file. When trust_model is False, this parameter must be provided to permit deserialization of the model; when trust_model is True the function will attempt to infer an appropriate trusted list and the provided trusted value may be ignored. If skops.load detects untrusted objects in the dump (i.e., objects not in this list and not inherently allowed), it will raise an exception and loading will fail.
    
    Returns:
        tuple: (model, model_info) where
            model: The deserialized model object returned by skops.load (a Python object such as a scikit-learn estimator or pipeline) that can be used for inference in model-based curation steps.
            model_info: Metadata associated with the model (typically a dict-like object) describing the model's provenance, version, expected features, or other information that downstream SpikeInterface curation code uses to validate compatibility and interpret model outputs.
    
    Behavior and side effects:
        Exactly one of model_folder or repo_id must be provided; otherwise a ValueError is raised. When loading from repo_id the function will perform network I/O and may write or read cache files via the HuggingFace client. Deserialization is performed using skops.load; if the serialized file contains objects not allowed by the trusted policy, skops.load will raise and load will fail. If model_name is omitted the loader selects the first compatible model file found; this may produce unexpected results if multiple model artifacts exist. Common failure modes include ValueError for invalid argument combinations, FileNotFoundError if the specified model file is absent, network errors when contacting HuggingFace, and skops-related exceptions during deserialization due to untrusted or incompatible objects.
    """
    from spikeinterface.curation.model_based_curation import load_model
    return load_model(model_folder, repo_id, model_name, trust_model, trusted)


################################################################################
# Source: spikeinterface.curation.train_manual_curation.check_metric_names_are_the_same
# File: spikeinterface/curation/train_manual_curation.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_curation_train_manual_curation_check_metric_names_are_the_same(
    metrics_for_each_analyzer: list
):
    """Check that the set of metric names computed by each analyzer is identical.
    
    Args:
        metrics_for_each_analyzer (list): A list where each element corresponds to the metrics computed by one SortingAnalyzer (in the SpikeInterface curation/training workflow). In practice each element is expected to be a dataframe-like object (for example a pandas.DataFrame) or any mapping-like object that exposes a .keys() method returning the available metric names for that analyzer. Each element represents the collection of computed quality metrics for a single analyzer; the function uses the set of keys from each element to determine the metric names present.
    
    Behavior:
        The function iterates over all distinct pairs of elements in metrics_for_each_analyzer and compares the sets of keys returned by each element's .keys() method. The comparison is symmetric and performed only once per unordered pair (the implementation compares pairs where the first index is greater than the second to avoid duplicate checks). If all analyzers expose exactly the same set of metric names, the function completes normally and returns None. If any pair has a different set of metric names, the function constructs a descriptive error message that identifies the two analyzer indices involved (using the list indices) and lists which metric names are present in one analyzer but missing in the other, then raises an Exception with that message.
    
    Side effects and failure modes:
        This function does not modify the input list or its elements. It performs O(n^2) pairwise set comparisons, where n is len(metrics_for_each_analyzer), so runtime grows quadratically with the number of analyzers. If an element does not implement .keys(), a runtime AttributeError will be raised when the function attempts to call .keys() on that element. If metric name sets differ between any two analyzers, the function raises a generic Exception containing a message of the form:
        "Computed metrics are not equal for sorting_analyzers #<j> and #<i>\n" plus additional lines indicating which metrics are missing in which analyzer. The indices <j> and <i> correspond to the positions of the analyzers in metrics_for_each_analyzer.
    
    Returns:
        None: The function returns None on success (when all analyzers have identical metric name sets). On mismatch it raises an Exception describing which analyzers differ and which metric names are missing in each.
    """
    from spikeinterface.curation.train_manual_curation import check_metric_names_are_the_same
    return check_metric_names_are_the_same(metrics_for_each_analyzer)


################################################################################
# Source: spikeinterface.curation.train_manual_curation.train_model
# File: spikeinterface/curation/train_manual_curation.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_curation_train_manual_curation_train_model(
    mode: str = "analyzers",
    labels: list = None,
    analyzers: list = None,
    metrics_paths: list = None,
    folder: str = None,
    metric_names: list = None,
    imputation_strategies: list = None,
    scaling_techniques: list = None,
    classifiers: list = None,
    test_size: float = 0.2,
    overwrite: bool = False,
    seed: int = None,
    search_kwargs: dict = None,
    verbose: bool = True,
    enforce_metric_params: bool = False,
    **job_kwargs
):
    """Trains and evaluates machine learning models to assist manual curation of spike sorting outputs within the SpikeInterface framework. The function orchestrates creation of a CurationModelTrainer, validates inputs, loads metrics either from SortingAnalyzer objects or CSV files, preprocesses metrics (selection of metric names, imputation, scaling), searches classifier hyperparameters, evaluates model configurations on a train/test split, and saves evaluation artifacts and the best model to the specified output folder. This function is used in spike sorting curation workflows to build models that predict curated labels (for example, unit quality or accept/reject decisions) from computed quality metrics produced by SpikeInterface analyzers or from CSV exports of those metrics.
    
    Args:
        mode (str): Mode to use for training. Must be either "analyzers" or "csv". In "analyzers" mode the function expects a list of SortingAnalyzer-like objects (see analyzers) from which metrics and labels are loaded; in "csv" mode the function expects a list of file paths to CSV metric files (see metrics_paths). A wrong value raises an Exception.
        labels (list): List of lists of curated labels to use as the supervised targets for training. Each inner list corresponds to the curated labels for one dataset and must be in the same order as the metrics provided/loaded. This argument is required and if None the function raises an Exception instructing the user to supply labels = [[...],[...],...]. The labels are used by the CurationModelTrainer to form y values for fitting and evaluation.
        analyzers (list): List of SortingAnalyzer-like objects or None. When mode is "analyzers", this list must be provided and contain objects that expose computed metrics (quality metrics and any parameters used to compute them) and align with the provided labels. If analyzers is None while mode is "analyzers", an AssertionError is raised. The analyzers are passed to the trainer.load_and_preprocess_analyzers method for loading and preprocessing.
        metrics_paths (list): List of strings or None. When mode is "csv", this must be a list of file paths to CSV files containing the metrics data. Before loading, each path is asserted to be an existing file; a non-existing path causes an AssertionError identifying the offending path. The files are passed to trainer.load_and_preprocess_csv for loading.
        folder (str): Path to the output folder where models, evaluation metrics, and other artifacts produced by training and evaluation will be saved. This parameter is mandatory and if None the function raises an Exception instructing the user to supply folder='path/to/folder/'. If overwrite is False and the folder already exists, the function asserts and raises an AssertionError indicating the folder exists; if overwrite is True the function proceeds (the CurationModelTrainer will be initialized with this folder and is responsible for any further handling).
        metric_names (list): List of strings or None. Explicit list of metric names (column names of metric data) to use for training. If None, default metric selections defined by the trainer are used. When provided, these names are forwarded to the CurationModelTrainer to select which metrics are included in preprocessing and model fitting.
        imputation_strategies (list): List of strings or None. A list of imputation strategies to try during preprocessing. Allowed example strategies include "knn", "iterative", "median", and "most_frequent"; the function documentation refers to strategies compatible with sklearn SimpleImputer or other supported imputers. If None, the trainer will use the default strategies ["median", "most_frequent", "knn", "iterative"]. These strategies are explored during model configuration evaluation to handle missing metric values.
        scaling_techniques (list): List of strings or None. A list of scaling techniques to try during preprocessing, for example "standard_scaler", "min_max_scaler", or "robust_scaler". If None, the trainer will try all available techniques by default. These determine how metric features are scaled prior to model fitting.
        classifiers (list): List, dict, or None. A list of classifier identifiers or a dictionary mapping classifier identifiers to hyperparameter search spaces. If None, default classifiers and their default hyperparameter search spaces (as defined by CurationModelTrainer.get_classifier_search_space) are used. When a dict is provided, it must follow the format expected by the trainer for hyperparameter search spaces; this argument controls which model families and parameter searches the evaluation will consider.
        test_size (float): Proportion of the dataset used as the test split for evaluation, passed to sklearn.model_selection.train_test_split. Must be a float between 0.0 and 1.0 inclusive; values less than 0.0 or greater than 1.0 cause an Exception with a message that test_size must be between 0.0 and 1.0. The default is 0.2, meaning 20% of the data is held out for testing.
        overwrite (bool): If False (default), the function asserts that the specified folder does not already exist and raises an AssertionError if it does; this prevents accidental overwriting. If True, the function does not perform that existence assertion and proceeds to initialize the trainer with the given folder (the trainer is then responsible for how existing contents are handled).
        seed (int): Integer seed for randomness to improve reproducibility of splits and model search. If None (default), a random seed will be generated/used by the trainer. The seed value is forwarded to the CurationModelTrainer and influences train/test splitting and any randomized search procedures.
        search_kwargs (dict): Dictionary of keyword arguments forwarded to the hyperparameter search routine (e.g., BayesSearchCV or RandomizedSearchCV). If None (default), the function uses or forwards default search_kwargs equivalent to {'cv': 3, 'scoring': 'balanced_accuracy', 'n_iter': 25}. These kwargs influence cross-validation folds, scoring metric, number of iterations, and other search behavior.
        verbose (bool): If True (default), the function prints progress and useful informational messages during initialization, loading, preprocessing, and evaluation. If False, the function suppresses such informational printing; warnings and errors from underlying libraries may still be shown.
        enforce_metric_params (bool): If True, the function enforces that metric computation parameters are identical across provided SortingAnalyzer objects; if analyzers were produced with different metric parameters and this flag is True, an error will be raised when loading analyzers. If False (default), the function will not raise on differing metric parameters but inconsistent metrics may affect model training.
        job_kwargs (dict): Additional keyword arguments captured from the caller and forwarded verbatim to the CurationModelTrainer constructor. These are used to pass backend, job scheduling, parallelization, or other trainer-specific configuration options accepted by CurationModelTrainer.
    
    Returns:
        CurationModelTrainer: The initialized and executed CurationModelTrainer instance used for training and evaluation. The returned trainer has performed data loading, preprocessing, hyperparameter search/evaluation (via trainer.evaluate_model_config), and has saved evaluation results and the selected best model to the given folder. Inspecting the returned object gives access to fitted models, evaluation metrics, search results, and any artifacts written to disk.
    
    Behavior, side effects, and failure modes:
        The function validates inputs and raises explicit Exceptions or AssertionError in the following situations: folder is None (Exception), labels is None (Exception), mode not in {"analyzers", "csv"} (Exception), test_size outside the [0.0, 1.0] range (Exception), if overwrite is False and folder already exists (AssertionError), if mode is "analyzers" and analyzers is None (AssertionError), and if mode is "csv" and any entry in metrics_paths does not point to an existing file (AssertionError). After validation, a CurationModelTrainer is instantiated with the provided arguments and job_kwargs. In "analyzers" mode, trainer.load_and_preprocess_analyzers is called (and may raise errors if analyzers are invalid or metrics are inconsistent when enforce_metric_params is True). In "csv" mode, trainer.load_and_preprocess_csv is called for the provided paths. Finally, trainer.evaluate_model_config() is executed to run preprocessing combinations, hyperparameter searches, and evaluation; this method saves models and evaluation artifacts into folder. The function returns the trainer object for programmatic access to results.
    """
    from spikeinterface.curation.train_manual_curation import train_model
    return train_model(
        mode,
        labels,
        analyzers,
        metrics_paths,
        folder,
        metric_names,
        imputation_strategies,
        scaling_techniques,
        classifiers,
        test_size,
        overwrite,
        seed,
        search_kwargs,
        verbose,
        enforce_metric_params,
        **job_kwargs
    )


################################################################################
# Source: spikeinterface.extractors.bids.read_bids
# File: spikeinterface/extractors/bids.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_extractors_bids_read_bids(folder_path: str):
    """spikeinterface.extractors.bids.read_bids loads a BIDS-formatted folder of electrophysiology data into SpikeInterface extractor objects for downstream preprocessing and spike sorting.
    
    Args:
        folder_path (str or Path): Path to the BIDS folder on disk. This is the filesystem path to a directory that follows BIDS conventions for electrophysiology data (for example containing files such as <suffix>_channels.tsv, <suffix>_contacts.tsv, <suffix>_ephys.nwb, <suffix>_probes.tsv). The function accepts either a string path or a pathlib.Path object (the implementation immediately converts the input to a Path). The provided folder must exist and be readable by the running process; if it does not exist or is not a directory, iteration over the folder will raise a FileNotFoundError or appropriate OSError.
    
    Behavior and practical details:
        This function iterates the top-level entries in folder_path and inspects file suffixes to discover supported data containers. It recognizes Neurodata Without Borders files with suffix ".nwb" and NIX files with suffix ".nix". For each recognized file it:
        - Derives a BIDS base name (bids_name) from the file stem and annotates the created extractor with bids metadata via rec.annotate(bids_name=bids_name). This annotation stores the BIDS identifier on the returned extractor so downstream SpikeInterface components (preprocessing, sorters, post-processing) can reference the dataset origin.
        - For ".nwb" files, calls read_nwb(file_path, load_recording=True, load_sorting=False, electrical_series_name=None) to load the recording data. The function intentionally requests the recording but not sorting to produce RecordingExtractor-style objects for processing and sorting pipelines.
        - For ".nix" files, imports neo and instantiates neo.rawio.NIXRawIO(file_path) to discover signal_stream ids; then calls read_nix(file_path, stream_id=stream_id) for each stream_id to produce one extractor per stream.
        - Attempts to attach probe metadata to each recording by calling an internal helper _read_probe_group(file_path.parent, bids_name, rec.channel_ids) and then rec.set_probegroup(probegroup). This attaches Probe/ProbeGroup information (channel geometry and contact metadata typically described in <suffix>_probes.tsv and <suffix>_channels.tsv) to the extractor so downstream waveform extraction, channel mapping, and visualization use correct spatial/channel layouts.
        - Marks the recording as having an extra requirement of the pandas package by extending rec.extra_requirements with "pandas". This is a metadata flag indicating that full parsing of associated TSV probe/channel files relied on pandas; it does not install pandas automatically. If pandas is not available, downstream code that needs probe metadata may raise ImportError when attempting to use pandas-dependent routines.
        Files that do not have a recognized suffix are ignored. If read_nwb/read_nix or the neo library raise errors, those exceptions propagate to the caller (e.g., file corruption, incompatible NWB/NIX schema, or missing optional dependencies will result in an exception). If the folder contains no supported files, the function returns an empty list.
    
    Returns:
        list: A list of extractor objects (RecordingExtractor-like objects) corresponding to the loaded recordings. Each returned extractor represents a single recording stream discovered in the BIDS folder, is annotated with bids_name, has been marked to require pandas for full metadata parsing, and has had Probe/ProbeGroup information attached when available. The returned extractors are ready for use with SpikeInterface preprocessing, spike sorting, and downstream post-processing workflows.
    """
    from spikeinterface.extractors.bids import read_bids
    return read_bids(folder_path)


################################################################################
# Source: spikeinterface.extractors.cbin_ibl.extract_stream_info
# File: spikeinterface/extractors/cbin_ibl.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_extractors_cbin_ibl_extract_stream_info(meta_file: str, meta: dict):
    """spikeinterface.extractors.cbin_ibl.extract_stream_info extracts channel and stream-level metadata from a SpikeGLX-style metadata dictionary and the metadata filename, returning a normalized info dictionary used by SpikeInterface to interpret binary stream files for spike sorting and downstream processing.
    
    This function is used within the SpikeInterface framework to translate SpikeGLX .meta file contents (and the metadata filename) into a canonical set of values needed when reading and preprocessing extracellular recordings: number of saved channels, sampling rate, stream kind and name (e.g., "ap" or "lf"), per-channel gains converted to microvolts, channel names, sample count, and whether a sync trace is present. The returned info dictionary is intended to be consumed by extractors and readers that map raw binary samples to physical units and channel labels for spike sorting, visualization, and quality metrics computation.
    
    Args:
        meta_file (str): Path or filename of the SpikeGLX metadata file. The function uses Path(meta_file).stem to derive the base filename and infers the stream_kind as the last dot-separated component of that stem (for example, a stem ending with ".ap" indicates an AP stream). The stem is also used to compute the session identifier and the canonical stream_name (session + "." + stream_kind). Supplying an incorrect filename may cause stream_kind/session to be inferred incorrectly, which will propagate to stream_name and fname fields in the returned info.
        meta (dict): Dictionary parsed from the SpikeGLX .meta file. This function expects keys documented by SpikeGLX metadata formats: "nSavedChans" (int-like string), "snsApLfSy" (optional, to detect sync trace presence), "snsChanMap" (list-like of channel labels), "fileSizeBytes" (bytes in the corresponding raw binary file), and either NP1.0 or NP2.0 fields used to compute per-channel gains (for example "imroTbl", "imAiRangeMax", "imMaxInt", "imChan0apGain", "imDatPrb_type", "niSampRate", "imSampRate"). The meta dict is not modified. If required keys are missing or contain unexpected formats (non-numeric strings where numeric conversion is performed), the function may raise KeyError or ValueError.
    
    Behavior and practical details:
        - num_chan is derived from meta["nSavedChans"] and cast to int; it represents the number of recorded channels reported in the metadata and is used throughout gain and sample length calculations.
        - has_sync_trace is determined from meta["snsApLfSy"] when present: this string is split into three comma-separated integers (ap, lf, sy) and has_sync_trace is True if sy == 1. If "snsApLfSy" is absent (typical for NIDQ cases), has_sync_trace is set to False.
        - fname is the stem of meta_file (Path(meta_file).stem) and used to build session and stream_kind. session is computed by removing the final dot-separated component from fname; stream_kind is the final dot-separated component. stream_name is session + "." + stream_kind.
        - units are set to "uV" (microvolts). Note that channel gain computations include a factor of 1e6 to convert from volts to microvolts, consistent with SpikeInterface conventions for physical units.
        - Per-channel gains are computed differently depending on meta["imDatPrb_type"]:
            - For NP 1.0 and certain legacy/imDatPrb_type values (imDatPrb_type absent, "0", or in ("1015","1022","1030","1031","1032")), the code expects meta["imroTbl"] entries and extracts a gain-related token at index 3 for AP streams and index 4 for LF streams. For each channel index c in range(num_chan - 1), the token is parsed and used to set per-channel gain as 1.0 / float(token). The last channel is treated as a fake channel and left at a default of 1.0 until the global gain_factor is applied.
            - For NP 2.0 probe types (imDatPrb_type in ("21","24","2003","2004","2013","2014")), the code uses meta["imChan0apGain"] when available; otherwise it falls back to a default per-channel AP gain of 1/80.0 for all but the last channel. max_int is read from meta["imMaxInt"] when present, otherwise 8192 is used. These values are combined with imAiRangeMax to compute the global gain_factor.
        - gain_factor is computed from meta["imAiRangeMax"] divided by an integer full-scale value (512 for NP1.0 case, or imMaxInt / default 8192 for NP2.0); final per-channel channel_gains are gain_factor * per_channel_gain * 1e6 (microvolt conversion).
        - channel_names is built from meta["snsChanMap"] by taking each entry and splitting on ";" to use the first token as the display channel name. channel_offsets is set to a numpy array of zeros with length num_chan.
        - sampling_rate is read preferentially from meta keys "niSampRate" or "imSampRate" when present and converted to float.
        - sample_length is computed as int(meta["fileSizeBytes"]) // 2 // num_chan. The division by 2 assumes 2 bytes per sample (int16 storage in SpikeGLX raw files); this value yields the number of time samples in the corresponding raw binary file and is critical for mapping file size to time duration.
        - The function makes no external I/O calls; it only reads the provided meta dict and meta_file string. It does however rely on meta contents matching expected SpikeGLX metadata formats; malformed or missing keys will raise standard Python exceptions (KeyError, ValueError) or a NotImplementedError for unsupported imDatPrb_type values.
    
    Failure modes and exceptions:
        - NotImplementedError is raised when meta["imDatPrb_type"] is present and not in the handled sets for NP1.0 or NP2.0; this indicates the metadata version/probe type is not implemented by this function and callers should handle this case upstream.
        - KeyError or ValueError may occur if required meta keys are missing or not convertible to numeric types (for example "nSavedChans", "fileSizeBytes", "imAiRangeMax", entries in "imroTbl", or numeric gains). Callers should validate the meta dict or catch these exceptions.
        - If meta_file does not contain a dot-separated final component indicating stream kind, stream_kind will still be set to the final token from fname.split(".") and may be incorrect; callers should provide correct filenames (typical SpikeGLX naming conventions) to avoid mislabeling stream_kind.
    
    Returns:
        dict: A dictionary with normalized stream and channel information used by SpikeInterface readers and extractors. The dictionary contains at least the following keys and associated types and meanings:
            fname (str): The stem of the provided meta_file (Path(meta_file).stem). Used as the canonical base name for the metadata/stream.
            meta (dict): The original meta dictionary passed into the function (unchanged). Included so downstream code has access to raw metadata.
            sampling_rate (float, optional): The inferred sampling rate in Hz if "niSampRate" or "imSampRate" was present in meta. If neither key exists, this key will be absent.
            num_chan (int): Number of saved channels (int(meta["nSavedChans"])), used for shaping and indexing channel-related arrays.
            sample_length (int): Number of time samples in the corresponding raw binary stream, computed as int(meta["fileSizeBytes"]) // 2 // num_chan. This value is essential for mapping file offsets to time indices.
            stream_kind (str): The stream type inferred from the filename stem (e.g., "ap" or "lf"). Used to choose processing paths for high-pass (AP) vs low-frequency (LF) streams.
            stream_name (str): Canonical stream name composed as session + "." + stream_kind. Useful for labeling streams across sessions and for filename derivation in extractors.
            units (str): Physical units for channel_gains, set to "uV" (microvolts).
            channel_names (list of str): List of display channel names extracted from meta["snsChanMap"], each taken as the substring before ";" for each entry. These names are used for plotting and channel selection in downstream tools.
            channel_gains (numpy.ndarray): 1D numpy array of dtype float64 containing per-channel multiplicative gains to convert raw integer samples to units (microvolts). These gains incorporate probe-specific per-channel factors and global ADC range conversion.
            channel_offsets (numpy.ndarray): 1D numpy array (float64) of zeros with length num_chan. Present to allow callers to apply additive offsets if required; currently all offsets are zero because SpikeGLX metadata does not provide per-channel offsets in this code path.
            has_sync_trace (bool): True if a synchronization trace is present (derived from "snsApLfSy" when available), otherwise False.
    
    No external side effects are performed (the function does not read the binary data file or write files); it purely computes and returns the info dictionary or raises exceptions described above.
    """
    from spikeinterface.extractors.cbin_ibl import extract_stream_info
    return extract_stream_info(meta_file, meta)


################################################################################
# Source: spikeinterface.extractors.mcsh5extractors.openMCSH5File
# File: spikeinterface/extractors/mcsh5extractors.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_extractors_mcsh5extractors_openMCSH5File(filename: str, stream_id: int):
    """spikeinterface.extractors.mcsh5extractors.openMCSH5File opens a Multi Channel Systems (MCS) HDF5 recording file and extracts metadata required by SpikeInterface for downstream spike-sorting workflows. The function reads the specified AnalogStream inside the file, verifies timestamps and channel information, converts stored amplitude calibration to microvolts, computes the sampling frequency, and returns a dictionary of recording-level information used by SpikeInterface extractors to construct a RecordingExtractor.
    
    Args:
        filename (str): Path to an existing MCS HDF5 file on disk. This is passed verbatim to h5py.File(filename, "r") to open the file read-only. If the file does not exist or cannot be opened, h5py will raise an IOError/OSError. The opened filehandle is returned inside the result dictionary under the key "filehandle" and is left open; the caller (or downstream SpikeInterface components) is responsible for closing it when finished.
        stream_id (int): Integer identifier of the AnalogStream to read. The function expects a group named "Stream_{stream_id}" under "/Data/Recording_0/AnalogStream". If that named stream is not present, the function raises an AssertionError listing the available streams. This parameter selects which recorded stream (for example when multiple simultaneous streams were recorded) will be parsed to obtain channel data, timestamps, and channel calibration info.
    
    Returns:
        dict: A dictionary containing recording metadata and objects required by SpikeInterface to represent the MCS recording. The dictionary contains the following keys and values exactly as produced by the function:
            "filehandle": an open h5py.File instance for the MCS HDF5 file; left open so callers can read raw datasets if needed. Close this filehandle explicitly to release resources.
            "num_frames": int, the number of time samples (frames) in the ChannelData array (data.shape[1]).
            "sampling_frequency": float, computed as the reciprocal of the average timestep derived from the stream's ChannelDataTimeStamps. Time tick values are converted by dividing the stored Tick value by 1e6 (Tick in seconds) as implemented in the source.
            "num_channels": int, the number of recorded channels (data.shape[0]).
            "channel_ids": list of str, channel identifiers constructed as "Ch{channel_id}" from the MCS InfoChannel ChannelID field; these are the channel identifiers used by SpikeInterface for channel mapping.
            "electrode_labels": list of str, human-readable electrode labels decoded from the InfoChannel Label field.
            "gain": numpy.ndarray (one element per channel), conversion factors expressed in microvolts (uV) computed from the stored ConversionFactor and Exponent fields and scaled by 1e6 to represent microvolts. These gains are used to convert raw stored values to physical units.
            "dtype": numpy.dtype, the raw data dtype of the ChannelData dataset read from the file; useful for downstream readers to interpret stored binary values.
            "offset": numpy.ndarray (one element per channel), additive offsets in microvolts (uV) computed from the InfoChannel ADZero and other fields. These offsets are already scaled to microvolts and are intended to be applied after gain conversion to obtain calibrated voltages.
    
    Behavior, side effects, defaults, and failure modes:
        - The function opens the HDF5 file read-only via h5py.File(filename, "r") and returns the open filehandle in the result dictionary; it does not close the file before returning. Callers must close the filehandle to avoid resource leaks.
        - The function validates that the requested stream exists under "/Data/Recording_0/AnalogStream". If not present, it raises an AssertionError with a message that includes the list of available analog streams.
        - The function reads datasets named "ChannelData", "ChannelDataTimeStamps", and "InfoChannel" from the stream group. Missing datasets or malformed HDF5 structure will cause h5py to return None or raise exceptions when accessed; subsequent operations may raise TypeError/AttributeError/AssertionError.
        - Timestamp handling: the function asserts that timestamps[0][0] < timestamps[0][2] and computes TimeVals as an integer range multiplied by Tick (Tick is divided by 1e6 in code to convert to seconds). It computes average, min, and (by implementation) min again for timestep checks; it asserts that relative variation from the average is less than 1e-6 (1 ppm). If timestamps are invalid or time steps vary beyond this tolerance, an AssertionError is raised.
        - Units and calibration: the function expects voltage units to be stored as b"V". If Unit differs, it prints a warning stating the unexpected unit but continues, assuming values represent volts. Gain and offset are converted and scaled to microvolts (uV) as implemented in the source code; these numeric arrays align with channel ordering in InfoChannel.
        - Duplicate channel IDs: the function asserts that channel_ids are unique; if duplicates are found, an AssertionError is raised.
        - Sampling frequency is derived from the timestamp tick conversion and the computed average timestep; if timestamp resolution is inconsistent the sampling frequency may be invalid and the function will raise assertions as above.
        - All numeric arrays returned (gain, offset) are produced by NumPy operations as in the source and preserve per-channel ordering from the MCS InfoChannel.
    """
    from spikeinterface.extractors.mcsh5extractors import openMCSH5File
    return openMCSH5File(filename, stream_id)


################################################################################
# Source: spikeinterface.extractors.neoextractors.mearec.read_mearec
# File: spikeinterface/extractors/neoextractors/mearec.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_extractors_neoextractors_mearec_read_mearec(file_path: str):
    """spikeinterface.extractors.neoextractors.mearec.read_mearec reads a MEArec HDF5 file and returns a pair of SpikeInterface extractor objects: a MEArecRecordingExtractor that provides programmatic access to the extracellular recording (continuous traces and channel metadata) and a MEArecSortingExtractor that provides the corresponding ground-truth spike times/units for benchmarking and analysis.
    
    Args:
        file_path (str or Path): Path to a MEArec HDF5 file produced by the MEArec simulator. This parameter must be a filesystem path (either a string or a pathlib.Path) that points to a readable MEArec-format .h5 file. In the SpikeInterface domain, this file encapsulates simulated extracellular voltage traces and the simulator's ground-truth spike sorting; providing this path allows the function to construct extractor objects that integrate with the SpikeInterface pipeline for preprocessing, running sorters, computing quality metrics, visualization, and benchmarking.
    
    Returns:
        tuple[MEArecRecordingExtractor, MEArecSortingExtractor]: A tuple (recording, sorting) where:
            recording (MEArecRecordingExtractor): An extractor object that exposes the recording data (sampling frequency, number of channels, channel ids, and methods to retrieve traces). In practical use within SpikeInterface, this recording extractor is used as the input to preprocessing steps, waveform extraction, and spike sorting algorithms without requiring the caller to manually parse the HDF5 file.
            sorting (MEArecSortingExtractor): An extractor object that exposes the ground-truth unit spike trains (unit ids and spike times). This sorting extractor is used for benchmarking and validation of spike sorting outputs produced from the associated recording.
    
    Behavior and side effects:
        The function constructs and returns extractor objects by calling MEArecRecordingExtractor(file_path) and MEArecSortingExtractor(file_path). These constructors create Python objects that reference the underlying MEArec file; they typically avoid copying the full dataset into memory and instead provide accessors to load data or metadata on demand. The function itself does not modify the input file.
    
    Failure modes and exceptions:
        If the given file_path does not exist or is not accessible, underlying constructors will raise FileNotFoundError or OSError. If the file exists but is not a valid MEArec HDF5 file or is missing required groups/datasets, the underlying extractor constructors may raise ValueError or format-specific errors. Any exceptions raised originate from MEArecRecordingExtractor or MEArecSortingExtractor constructors and should be handled by the caller.
    
    Usage significance:
        This function is a convenience entry point within SpikeInterface for loading simulated datasets created with the MEArec simulator, enabling reproducible benchmarking workflows where the recording (signals) and the ground-truth sorting (spike identities and times) are required together.
    """
    from spikeinterface.extractors.neoextractors.mearec import read_mearec
    return read_mearec(file_path)


################################################################################
# Source: spikeinterface.extractors.neoextractors.neo_utils.get_neo_num_blocks
# File: spikeinterface/extractors/neoextractors/neo_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_extractors_neoextractors_neo_utils_get_neo_num_blocks(
    extractor_name: str,
    *args,
    **kwargs
):
    """Returns the number of NEO blocks for a given NEO-based extractor.
    
    This utility is part of the SpikeInterface framework (a unified framework for spike sorting) and is used by extractor callers and higher-level I/O utilities to determine how many NEO "Block" records a dataset exposes before attempting to read data. Knowing the number of blocks is important in multi-block extracellular recordings (for example, concatenated recording sessions, segmented acquisitions, or datasets where temporal segments are stored as separate NEO Block objects) so downstream code can select a block via the block_index argument when calling the corresponding read_**extractor_name**() reader.
    
    Args:
        extractor_name (str): The extractor identifier string. This must be the name of a NEO-backed extractor registered in the SpikeInterface neo extractors registry (for example, those available via se.recording_extractor_full_dict). The function uses this name to resolve a neo_extractor object via get_neo_extractor(extractor_name) and then queries that object's block count. In practical terms, pass the exact extractor key used by SpikeInterface to refer to the file format or dataset handler you intend to probe.
        args (tuple): Positional, extractor-specific arguments forwarded directly to the underlying neo_extractor.get_num_blocks(...) implementation. These are the same positional parameters you would pass to the corresponding read_**extractor_name**() function (for example file paths, device identifiers, or other positional options required by that extractor). They are domain-specific and interpreted by the concrete neo extractor implementation.
        kwargs (dict): Keyword, extractor-specific arguments forwarded directly to the underlying neo_extractor.get_num_blocks(...) implementation. Commonly used keyword arguments include block_index in caller code to select or query particular blocks, file-specific options, or format-specific flags. These keyword parameters must match what the specific extractor's read_**extractor_name**() and get_num_blocks(...) expect; consult the extractor's documentation or the read_**extractor_name**? help for details.
    
    Returns:
        int: The number of NEO Block objects reported by the resolved neo_extractor for the dataset described by the provided arguments. This integer is used by SpikeInterface callers to decide whether the dataset is single-block (common) or multi-block and to validate block_index values before reading. A return value of 1 indicates a single-block dataset (the most common case); values >1 indicate multiple temporal/structural blocks.
    
    Behavior and side effects:
        This function performs no I/O of raw signal data itself beyond whatever lightweight probing the underlying neo extractor implements to determine block count. It resolves an extractor implementation via get_neo_extractor(extractor_name) and then calls neo_extractor.get_num_blocks(*args, **kwargs). All positional and keyword parameters are forwarded unchanged to that call. There are no persistent side effects (no files are modified); the call is intended to be read-only and quick in typical extractor implementations.
    
    Failure modes and errors:
        If extractor_name does not correspond to a registered NEO extractor, get_neo_extractor(extractor_name) will raise an error which is propagated to the caller. If the underlying neo_extractor cannot determine the number of blocks with the provided arguments, it may raise exceptions (for example ValueError, TypeError, or format-specific errors) which are also propagated. Callers should validate extractor_name and the extractor-specific args/kwargs according to the extractor's documentation (see read_**extractor_name**? or the extractor registry) to avoid these errors.
    
    Notes:
        Most extracellular datasets handled by SpikeInterface contain a single block; callers can use the returned value to implement logic for multi-block handling (e.g., iterating over blocks, selecting a specific block_index, or raising an informative error when a requested block_index is out of range).
    """
    from spikeinterface.extractors.neoextractors.neo_utils import get_neo_num_blocks
    return get_neo_num_blocks(extractor_name, *args, **kwargs)


################################################################################
# Source: spikeinterface.extractors.neoextractors.neo_utils.get_neo_streams
# File: spikeinterface/extractors/neoextractors/neo_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_extractors_neoextractors_neo_utils_get_neo_streams(
    extractor_name: str,
    *args,
    **kwargs
):
    """spikeinterface.extractors.neoextractors.neo_utils.get_neo_streams: Return the NEO stream names and stream ids associated with a dataset so callers (for example read_<extractor_name> functions in SpikeInterface) can enumerate available substreams and select which stream to read.
    
    This function is part of the SpikeInterface framework for reading many extracellular file formats via NEO-based extractors. It looks up the registered NEO extractor implementation by name, forwards the provided positional and keyword arguments to that extractor, and returns the extractor-reported list of stream names and the corresponding list of stream ids. For multi-stream datasets, these values let downstream code or users select the desired substream (for example by passing stream_id or stream_name to read_<extractor_name>).
    
    Args:
        extractor_name (str): The extractor name to look up in the SpikeInterface NEO extractor registry. This string must match a key available through the recording extractor registry used by SpikeInterface (for example entries in se.recording_extractor_full_dict). The name identifies the NEO-backed driver implementation that will be queried for available streams and determines the semantics of the forwarded arguments.
        args (tuple): Positional, extractor-specific arguments forwarded unchanged to the underlying NEO extractor's get_streams method. In practice these typically include path, folder, or other positional parameters required to open the dataset for that particular extractor. Consult the corresponding read_<extractor_name> documentation for the exact positional parameters accepted by that extractor.
        kwargs (dict): Keyword, extractor-specific arguments forwarded unchanged to the underlying NEO extractor's get_streams method. Common kwargs include stream selection keys such as stream_id or stream_name (used to select or filter streams) and other extractor-specific options. These are the same keyword arguments accepted by the read_<extractor_name> functions in SpikeInterface.
    
    Returns:
        list: List of NEO stream names returned by the underlying extractor. These are human-readable stream identifiers used by NEO and SpikeInterface to present or select substreams within a multi-stream dataset.
        list: List of NEO stream ids returned by the underlying extractor. These are the corresponding stream identifiers (numeric or string ids as reported by the extractor) that can be used programmatically to request a specific stream when reading data.
    
    Notes:
        The implementation performs a lookup via get_neo_extractor(extractor_name) and then calls that extractor's get_streams(*args, **kwargs). There are no side effects such as writing files; the function only queries the extractor for metadata. If the requested extractor_name is not registered or the underlying NEO extractor fails to enumerate streams (for example due to an invalid file path or unsupported file format), the underlying lookup or extractor call will raise an exception (for example KeyError, ValueError, or extractor-specific I/O exceptions). For single-stream datasets the returned lists will typically contain a single element.
    """
    from spikeinterface.extractors.neoextractors.neo_utils import get_neo_streams
    return get_neo_streams(extractor_name, *args, **kwargs)


################################################################################
# Source: spikeinterface.extractors.neoextractors.neuroscope.read_neuroscope
# File: spikeinterface/extractors/neoextractors/neuroscope.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_extractors_neoextractors_neuroscope_read_neuroscope(
    file_path: str,
    stream_id: str = None,
    keep_mua_units: bool = False,
    exclude_shanks: list = None,
    load_recording: bool = True,
    load_sorting: bool = False
):
    """Read neuroscope recording and sorting and return SpikeInterface extractor objects for downstream spike-sorting workflows.
    
    This function is a convenience loader used by the SpikeInterface framework to import NeuroScope-format data into the unified spike-sorting pipeline described in the project README. It expects an XML recording descriptor (file_path) and, optionally, associated clustering files (.res and .clu) in the same folder as that XML file. The function creates and returns NeuroScopeRecordingExtractor and/or NeuroScopeSortingExtractor instances (from the local neoextractors integration) depending on the load_recording and load_sorting flags. These extractor objects are the standard SpikeInterface abstractions for a recorded extracellular signal (recording) and a set of spike times with unit assignments (sorting) and can be used immediately for preprocessing, running sorters, computing quality metrics, visualization, and export as described in the README.
    
    Args:
        file_path (str): The path to the NeuroScope XML file describing the recording. This parameter must point to the .xml file used by NeuroScope; the function assumes any associated .res and .clu files are located in the same directory as this XML. The path is used both to construct a NeuroScopeRecordingExtractor (when load_recording is True) and to determine the folder for NeuroScopeSortingExtractor (when load_sorting is True).
        stream_id (str or None): The stream identifier within the NeuroScope XML to load for the recording. If None (the default), the function will load the first available stream found by NeuroScopeRecordingExtractor. In practical terms, this selects which data stream (for example, a particular continuous channel group or recording stream) is exposed by the returned recording extractor.
        keep_mua_units (bool): Whether to keep multi-unit activity (MUA) units when creating the sorting extractor. Default is False. When True, units labeled as MUA in the NeuroScope clustering files will be included in the returned NeuroScopeSortingExtractor; when False, those units will be filtered out. This affects downstream analyses (e.g., quality metrics, curation, or visualization) because MUA units typically represent pooled activity rather than single-unit spikes.
        exclude_shanks (list): Optional list of shank indices to ignore when creating the sorting extractor. If provided, the NeuroScopeSortingExtractor will exclude spikes/units coming from the specified shank indices. If None (the default), the set of all available shank indices is determined automatically by examining the final integer in pairs of .res.%i and .clu.%i filenames found in the folder containing file_path. This parameter controls which physical probe shanks are included in the sorting, which can be useful for excluding damaged or irrelevant probe regions.
        load_recording (bool): If True (default), construct and return a NeuroScopeRecordingExtractor for the specified file_path and stream_id. This extractor is the SpikeInterface object representing the continuous recording and is used throughout the SpikeInterface workflow for preprocessing and running sorters. If False, no recording extractor is created.
        load_sorting (bool): If True (default: False), construct and return a NeuroScopeSortingExtractor for the folder containing file_path, honoring keep_mua_units and exclude_shanks. This extractor represents spike times and unit labels parsed from .res and .clu files and is used for post-processing, benchmarking, and visualization within SpikeInterface. If False, no sorting extractor is created.
    
    Returns:
        NeuroScopeRecordingExtractor or NeuroScopeSortingExtractor or tuple: If only one of load_recording or load_sorting is True, the single corresponding extractor object is returned (NeuroScopeRecordingExtractor when load_recording is True and load_sorting is False; NeuroScopeSortingExtractor when load_sorting is True and load_recording is False). If both load_recording and load_sorting are True, a tuple (recording, sorting) is returned where recording is the NeuroScopeRecordingExtractor and sorting is the NeuroScopeSortingExtractor, in that order. If neither load_recording nor load_sorting is True, an empty tuple is returned. The returned extractor objects are the standard SpikeInterface API objects that let downstream code perform preprocessing, sorting, metric computation, visualization, and export as described in the README.
    
    Behavior, side effects, defaults, and failure modes:
        This function does not write to disk; its side effect is the construction (and any file handles opened by those constructors) of NeuroScopeRecordingExtractor and/or NeuroScopeSortingExtractor instances. The function uses Path(file_path).parent to locate clustering files for the sorting extractor. The implementation currently does not perform exhaustive existence checks before constructing extractors (there is a TODO in the source); therefore, common failure modes include FileNotFoundError or OSError if the XML or associated .res/.clu files are missing or inaccessible, and parsing errors or ValueError raised by the underlying extractor constructors if files are malformed or inconsistent. Consumers should handle these exceptions or validate file presence beforehand. The default behavior is to load only the recording (load_recording=True, load_sorting=False) to avoid needing clustering files unless explicitly requested.
    """
    from spikeinterface.extractors.neoextractors.neuroscope import read_neuroscope
    return read_neuroscope(
        file_path,
        stream_id,
        keep_mua_units,
        exclude_shanks,
        load_recording,
        load_sorting
    )


################################################################################
# Source: spikeinterface.extractors.neoextractors.openephys.read_openephys
# File: spikeinterface/extractors/neoextractors/openephys.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_extractors_neoextractors_openephys_read_openephys(
    folder_path: str,
    **kwargs
):
    """spikeinterface.extractors.neoextractors.openephys.read_openephys: Read an Open Ephys folder and return a SpikeInterface RecordingExtractor that can be used downstream in the SpikeInterface spike-sorting pipeline.
    
    Args:
        folder_path (str): Path to the Open Ephys folder on disk. This is the filesystem location of a recording saved by the Open Ephys system in either the "open ephys legacy" format (files with the suffix ".continuous") or the newer "binary" format. The function inspects the folder contents to auto-detect the format: if any file with ".continuous" appears, the folder is treated as legacy format and OpenEphysLegacyRecordingExtractor is instantiated; otherwise OpenEphysBinaryRecordingExtractor is instantiated. A non-existing or inaccessible folder will raise a FileNotFoundError or an OSError coming from the underlying filesystem access.
        kwargs (dict): Additional keyword arguments forwarded to the selected Open Ephys RecordingExtractor constructor. These keyword arguments control which experiment/stream is loaded, annotation behavior, synchronization handling, and timestamp handling; they mirror options used by the Open Ephys neo extractors and are important for correct interpretation of recorded channels and times when preparing data for preprocessing, sorting, and visualization in SpikeInterface. Supported keys (with their role and practical significance) include:
            experiment_name (str, default: None): Name of the experiment to load (for binary format only). Use when the folder contains multiple named experiments (for example "experiment1", "experiment2"). This cannot be used together with block_index.
            stream_id (str, default: None): Identifier of the stream to load when multiple streams are present. Use this to select the appropriate data stream (e.g., a particular recording device stream) to include in the returned RecordingExtractor.
            stream_name (str, default: None): Alternative to stream_id to select which stream to load by name when multiple streams exist.
            block_index (int, default: None): Zero-based index of the block/experiment to load as an alternative to experiment_name. Use this to select a block by position rather than by name. Cannot be used together with experiment_name.
            all_annotations (bool, default: False): If True, load exhaustively all annotations available from neo into the RecordingExtractor. Annotations can include experimental metadata and event labels that are useful for downstream curation and analysis. False (default) loads only essential annotations used to construct the recording.
            load_sync_channel (bool, default: False): DEPRECATED. Historically used to request loading a SYNC analog channel (for devices such as Neuropixels) into the analog signals. If False (default) and a SYNC channel is present it is not loaded; if True it is loaded and available in analog signals. Prefer specifying stream_name or stream_id for sync streams in current usage. This option applies to the open ephys binary format only.
            load_sync_timestamps (bool, default: False): If True, the extractor will load synchronized timestamps and set them as the recording times so that non-uniform or externally aligned timestamps are used. If False (default), the recording uses t_start and sampling rate, assuming uniformly spaced timestamps. This option applies to the open ephys binary format only and is important when precise, externally synchronized timing is required for analysis.
            ignore_timestamps_errors (bool, default: False): If True, ignore discontinuous timestamp errors reported by neo when reading legacy Open Ephys data. Use when legacy recordings contain known timestamp discontinuities and the user accepts potential timing inaccuracies. This option applies to the open ephys legacy format only.
        Notes:
            - kwargs are forwarded directly to the chosen extractor class; type checks and additional validations are performed by those extractor constructors. Conflicting options (for example providing both experiment_name and block_index) are not permitted and will be flagged by the underlying extractor code.
            - The function makes a format decision by checking for ".continuous" files; if such files are present anywhere in the folder, the legacy extractor is chosen. If the folder contains a mix of legacy and binary-style files, the presence of any ".continuous" file forces legacy mode.
            - This function is intended as an entry point to produce a RecordingExtractor used throughout SpikeInterface for preprocessing, running spike sorters, computing metrics, and visualization as documented in the SpikeInterface README.
    
    Returns:
        recording (OpenEphysLegacyRecordingExtractor or OpenEphysBinaryRecordingExtractor): A RecordingExtractor instance appropriate for the detected Open Ephys format. The returned object provides the standardized RecordingExtractor API used by SpikeInterface to access raw analog traces, channel locations, sampling rate, t_start, and annotations. The precise class depends on the auto-detected format: OpenEphysLegacyRecordingExtractor when legacy ".continuous" files are present, otherwise OpenEphysBinaryRecordingExtractor. The returned extractor may raise errors later when its methods are used if the provided kwargs were inconsistent with the folder contents or if files are corrupted.
    """
    from spikeinterface.extractors.neoextractors.openephys import read_openephys
    return read_openephys(folder_path, **kwargs)


################################################################################
# Source: spikeinterface.extractors.neoextractors.openephys.read_openephys_event
# File: spikeinterface/extractors/neoextractors/openephys.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_extractors_neoextractors_openephys_read_openephys_event(
    folder_path: str,
    experiment_name: str = None,
    block_index: int = None
):
    """spikeinterface.extractors.neoextractors.openephys.read_openephys_event reads Open Ephys events from an Open Ephys "binary" folder and returns an OpenEphysBinaryEventExtractor instance that exposes event timestamps and metadata for use in SpikeInterface workflows (for example, aligning events with recordings, epoching, or triggering downstream spike-sorting steps).
    
    This function inspects the given folder to determine whether the session uses the Open Ephys binary format and then constructs an OpenEphysBinaryEventExtractor for the requested experiment/block. It is intended for use in SpikeInterface pipelines that need programmatic access to Open Ephys event data produced by the Open Ephys GUI in binary format.
    
    Args:
        folder_path (str or pathlib.Path): Path to the top-level Open Ephys folder containing experiment subfolders and binary event files. The function uses pathlib.Path(folder_path).iterdir() to list files; if the path does not exist or is not a directory, a FileNotFoundError or NotADirectoryError will be raised by pathlib. The folder_path determines which Open Ephys session is inspected for event files and is required.
        experiment_name (str or None): Name of the experiment subfolder to load (for example, "experiment1" or "experiment2"). This parameter selects the experiment by name when the Open Ephys folder contains multiple experiments. Default is None. experiment_name is mutually exclusive with block_index: do not provide both. If both are provided, the values are forwarded to the underlying OpenEphysBinaryEventExtractor; this combination is not supported and may result in an error from that extractor.
        block_index (int or None): Zero-based index specifying which experiment block to load when multiple blocks (experiments) are present in the Open Ephys folder. For example, 0 selects the first experiment, 1 the second, and so on. Default is None. block_index is mutually exclusive with experiment_name: do not provide both. If both are provided, the values are forwarded to the underlying OpenEphysBinaryEventExtractor; this combination is not supported and may result in an error from that extractor.
    
    Behavior and side effects:
        The function auto-guesses the Open Ephys format by listing files in folder_path and checking for filenames that start with "Continuous". If any such "Continuous" files are present, the session is not the supported binary format for events and the function raises Exception("Events can be read only from 'binary' format"). If the format check passes, the function constructs and returns an OpenEphysBinaryEventExtractor(folder_path, experiment_name=experiment_name, block_index=block_index). Constructing the extractor is the primary side effect; the extractor object encapsulates access to event timestamps and related metadata and is used downstream in SpikeInterface for aligning events with recordings and other analysis steps. The function itself does not perform full file parsing beyond the format check and defers detailed reading to the returned extractor.
        Errors you may encounter: FileNotFoundError or NotADirectoryError if folder_path is invalid; Exception("Events can be read only from 'binary' format") if the folder appears to contain Open Ephys "continuous" (non-binary) files; and possible errors from the underlying OpenEphysBinaryEventExtractor if incompatible or unsupported combinations of experiment_name and block_index are supplied or if the binary files are malformed.
    
    Returns:
        OpenEphysBinaryEventExtractor: An extractor object that provides programmatic access to event timestamps, event labels, and related metadata read from the Open Ephys binary event files in the specified folder and experiment/block. This extractor is intended for use within SpikeInterface pipelines to align events with recordings, create epochs, or drive downstream spike sorting and analysis.
    """
    from spikeinterface.extractors.neoextractors.openephys import read_openephys_event
    return read_openephys_event(folder_path, experiment_name, block_index)


################################################################################
# Source: spikeinterface.extractors.neoextractors.spikeglx.read_spikeglx_event
# File: spikeinterface/extractors/neoextractors/spikeglx.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_extractors_neoextractors_spikeglx_read_spikeglx_event(
    folder_path: str,
    block_index: int = None
):
    """Read SpikeGLX events and return a SpikeGLXEventExtractor suitable for use in the SpikeInterface pipeline.
    
    Args:
        folder_path (str or Path): Path to the SpikeGLX / OpenEphys folder that contains the event files to read. In the SpikeInterface context this is the on-disk location produced by SpikeGLX/OpenEphys acquisition where event metadata and timestamp files live; the function uses this path to locate and parse the SpikeGLX event files so they can be exposed to downstream spike-sorting, visualization, and quality-metric components. The path must point to a directory with the expected SpikeGLX/OpenEphys file layout; if it does not exist or does not contain readable SpikeGLX event files the underlying extractor will raise an error.
        block_index (int, optional): If the dataset contains multiple recording blocks (multiple experiments or recording sessions within the same folder), specify the zero-based index of the block to load. Defaults to None, which selects the only block when there is a single block or relies on the underlying SpikeGLXEventExtractor to choose the default block. Supplying an integer selects that specific block; supplying an out-of-range index will result in an error from the extractor indicating an invalid block selection.
    
    Returns:
        SpikeGLXEventExtractor: An extractor object that wraps the parsed SpikeGLX event data and exposes it in the SpikeInterface extractor API. This object does not modify source files; instead it provides programmatic access to event times, event labels/types, and related metadata so that events can be used by downstream SpikeInterface components (preprocessing, synchronization, sorting, visualization, and export). The returned extractor may load metadata immediately or lazily depending on the extractor implementation; attempts to access non-existent blocks or malformed event files will raise errors from the extractor (e.g., file-not-found or parsing errors).
    """
    from spikeinterface.extractors.neoextractors.spikeglx import read_spikeglx_event
    return read_spikeglx_event(folder_path, block_index)


################################################################################
# Source: spikeinterface.extractors.neuropixels_utils.get_neuropixels_channel_groups
# File: spikeinterface/extractors/neuropixels_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_extractors_neuropixels_utils_get_neuropixels_channel_groups(
    num_channels: int = 384,
    num_channels_per_adc: int = 12
):
    """spikeinterface.extractors.neuropixels_utils.get_neuropixels_channel_groups returns groups of simultaneously sampled channels on a Neuropixels probe and documents the sampling grouping used to support operations that must act on synchronously digitized channels (for example, preprocessing.common_reference).
    
    Args:
        num_channels (int): The total number of channels in a recording. This value determines the range of valid channel indices used to form groups. All currently available Neuropixels variants have 384 channels, and the function default preserves that common configuration (default: 384). In practice, num_channels defines the maximum channel index + 1 that will be considered when computing ADC-synchronous groups.
        num_channels_per_adc (int): The number of channels handled by each ADC on the probe. This controls how many concurrently-sampled groups are generated (the function produces exactly num_channels_per_adc groups). Typical Neuropixels hardware configurations are documented here: Neuropixels 1.0 probes use 32 ADCs with 12 channels per ADC (hence the default 12), and Neuropixels 2.0 probes use 24 ADCs with 16 channels per ADC. The function uses this parameter to compute the interleaving pattern of channels that are digitized by the same ADC.
    
    Detailed behavior and practical significance:
        The function computes groups of channel indices that are digitized simultaneously by the same analog-to-digital converter (ADC) on a Neuropixels probe. This grouping is required by downstream preprocessing routines (for example, common-reference estimation) that must operate only on channels sampled synchronously to avoid mixing samples taken at different time instants.
        The grouping follows the Neuropixels ADC sampling pattern where pairs of even and odd channels are interleaved across ADCs (even and odd channels are digitized by separate ADCs). Concretely, for each i in range(num_channels_per_adc) the function concatenates two arithmetic progressions: one starting at channel index i*2 and advancing by step (num_channels_per_adc * 2), and another starting at i*2 + 1 with the same step, then sorts that concatenation to produce a single group. The output therefore has length equal to num_channels_per_adc, and each group is a sorted list of integer channel indices that were sampled at the same time by a given ADC.
        Defaults and typical usage: By default the function is configured for Neuropixels 1.0 probes (384 total channels, 12 channels per ADC). If you are using a different Neuropixels variant, set num_channels and num_channels_per_adc to match the probe geometry you have recorded with so the computed groups meaningfully reflect the device sampling pattern.
        Failure modes and edge cases: The function assumes integer inputs for num_channels and num_channels_per_adc. Non-integer or negative values will lead to errors or undefined behavior from the underlying numpy range operations. If the provided parameters do not correspond to an actual Neuropixels configuration, the produced groups will still be computed mechanically but may not reflect any real ADC sampling layout; callers should ensure parameters match the recording hardware. Also note that the function produces groups even when some groups may be empty if num_channels is small relative to the interleaving step; such empty groups are a natural consequence of the arithmetic pattern and should be handled by callers if necessary.
        Deprecation and side effects: Calling this function emits a DeprecationWarning indicating it is deprecated and will be removed in version 0.104.0; the recommended replacement is to use the adc_group contact annotation available from a Probe object (Probe contact annotations -> "adc_group"). The function has no other side effects beyond returning the computed groups and issuing the deprecation warning.
    
    Returns:
        list: A list of lists of integer channel indices. Each inner list contains the indices of channels that are sampled synchronously by the same ADC according to the Neuropixels interleaved sampling pattern. The number of inner lists equals num_channels_per_adc.
    """
    from spikeinterface.extractors.neuropixels_utils import get_neuropixels_channel_groups
    return get_neuropixels_channel_groups(num_channels, num_channels_per_adc)


################################################################################
# Source: spikeinterface.extractors.nwbextractors.read_nwb
# File: spikeinterface/extractors/nwbextractors.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_extractors_nwbextractors_read_nwb(
    file_path: str,
    load_recording: bool = True,
    load_sorting: bool = False,
    electrical_series_path: str = None
):
    """spikeinterface.extractors.nwbextractors.read_nwb reads an NWB (Neurodata Without Borders) file and constructs SpikeInterface extractor objects for extracellular recordings and/or spike sorting outputs. This function is used within the SpikeInterface framework (a unified framework for spike sorting) to import data stored in NWB format so downstream preprocessing, sorting, post-processing, quality metrics, and visualization tools in SpikeInterface can operate on the data.
    
    Args:
        file_path (str or Path): Path to the NWB file to read. This should point to a file that contains extracellular recording data and/or sorting results in NWB format. In practice this is the file path used by the underlying NWB reader helpers (read_nwb_recording and read_nwb_sorting). Supplying an invalid path will raise a FileNotFoundError or an error from the underlying NWB library.
        load_recording (bool): If True (default: True), the function will load and return a RecordingExtractor constructed from the NWB ElectricalSeries or recording group. The RecordingExtractor represents the extracellular voltage traces and channel metadata in SpikeInterface and is typically used for preprocessing, visualization, and as input to spike sorters.
        load_sorting (bool): If True (default: False), the function will load and return a SortingExtractor constructed from spike times / units stored in the NWB file. The SortingExtractor represents detected or curated spike times per unit and is used for post-processing, computing quality metrics, comparison and visualization. Both load_recording and load_sorting can be True to obtain both objects.
        electrical_series_path (str or None): The name or path identifying the NWB ElectricalSeries to use when multiple ElectricalSeries objects are present in the file. Default: None. If the NWB file contains a single ElectricalSeries, the helper readers will use it automatically; if multiple ElectricalSeries exist, provide this parameter to disambiguate. If this parameter is omitted when multiple ElectricalSeries are present, the underlying reader helpers may raise an error indicating ambiguity or missing specification.
    
    Returns:
        RecordingExtractor or SortingExtractor or tuple: A single RecordingExtractor or a single SortingExtractor if only one of load_recording/load_sorting was requested, or a tuple containing (RecordingExtractor, SortingExtractor) in that order if both load_recording and load_sorting are True. The RecordingExtractor and SortingExtractor are instances of the respective SpikeInterface extractor classes and encapsulate the recording traces, channel information, and spike train/unit information for downstream analysis. Errors encountered while parsing the NWB file (for example, invalid NWB schema, missing expected datasets, or an invalid electrical_series_path) are propagated from the underlying helpers (read_nwb_recording/read_nwb_sorting) and will raise exceptions.
    """
    from spikeinterface.extractors.nwbextractors import read_nwb
    return read_nwb(file_path, load_recording, load_sorting, electrical_series_path)


################################################################################
# Source: spikeinterface.extractors.sinapsrecordingextractors.parse_sinapse_h5
# File: spikeinterface/extractors/sinapsrecordingextractors.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_extractors_sinapsrecordingextractors_parse_sinapse_h5(filename: str):
    """Open and parse a SiNAPS HDF5 recording file and return a dictionary of recording metadata needed by SpikeInterface for downstream processing (e.g., preprocessing, waveform extraction, and spike sorting).
    
    This function is used in the SpikeInterface extractors submodule to read SiNAPS-format HDF5 files produced by SiNAPS/real-time acquisition systems. It opens the file in read-only mode, locates expected groups and datasets (RealTimeProcessedData/FilteredData and Parameters, plus Advanced Recording Parameters), and extracts essential recording parameters and a live file handle for later data access. The returned information is intended to let SpikeInterface and downstream tools interpret the stored samples (shape, dtype, gain/offset, sampling frequency, probe description, and ADC resolution) and to provide a ready filehandle for reading raw/filtered data without re-opening the file.
    
    Args:
        filename (str): Path to the SiNAPS HDF5 file on disk. This must be a filesystem-accessible string pointing to an HDF5 file that contains the expected SiNAPS groups and datasets (for example, a recording produced by the SiNAPS acquisition/processing pipeline). The function opens this path with h5py.File(filename, "r"), so the caller must have read permission.
    
    Returns:
        dict: A dictionary with the following keys (types shown in parentheses) describing the opened recording. This dictionary is constructed directly from groups/datasets in the HDF5 file and is intended for use by SpikeInterface extractors and downstream processing:
            filehandle (h5py.File): Open h5py file object returned by h5py.File(filename, "r"). Side effect: the file remains open after this function returns; the caller is responsible for closing this handle (for example, filehandle.close()) when finished to release system resources.
            num_frames (int): Number of time frames/samples per channel (second dimension of the FilteredData dataset). Represents the temporal length of the recording and is used to index and slice time ranges.
            sampling_frequency (int or float): Sampling frequency value taken from Parameters/SamplingFrequency[0]. Represents samples per second and is required to convert sample indices to time (seconds) in SpikeInterface workflows.
            num_channels (int): Number of recorded channels (first dimension of the FilteredData dataset). Used to validate channel counts and to size channel-related arrays.
            channel_ids (numpy.ndarray): Zero-based integer array of length num_channels produced by numpy.arange(num_channels). Represents the extractor’s internal channel identifiers mapping to rows of the FilteredData dataset.
            gain (numeric): Numeric gain value read from Parameters/VoltageConverter[0]. This value is the channel-independent voltage conversion factor stored in the file and is intended to convert raw stored values (ADC units) to physical voltages when required by downstream processing.
            offset (int or float): Offset applied to voltage conversion. In this implementation the offset is set to 0 (constant) and included so downstream code can apply offset + gain * raw if needed.
            dtype (numpy.dtype): NumPy dtype of the FilteredData dataset (data.dtype). Indicates the in-file sample storage type (e.g., int16, float32) and is necessary for correct interpretation of bytes when reading data.
            probe_type (str): Probe type string read from Advanced Recording Parameters/Probe/probeType. Provides an identifier for the electrode/probe geometry or manufacturer-specific probe description that downstream code or visualization tools can use to interpret channel layout.
            num_bits (int): Integer ADC resolution computed as int(log2(nbADCLevels)) where nbADCLevels is read from Advanced Recording Parameters/DAQ/nbADCLevels[0]. Represents the number of effective ADC bits and can be used to validate gain/scale or for metadata reporting.
    
    Behavior and side effects:
        - The function opens the HDF5 file with h5py.File(filename, "r") and returns the open file object as filehandle inside the returned dict. The file is not closed by this function.
        - The function reads the dataset RealTimeProcessedData/FilteredData to determine channel and frame counts and dtype, and reads parameters from the Parameters and Advanced Recording Parameters groups to populate gain, sampling frequency, probe type, and ADC resolution.
        - The offset field is set to 0 by this implementation (no file-provided offset is applied).
    
    Failure modes and exceptions:
        - If filename does not exist or is not an HDF5 file, h5py will raise an IOError/OSError or a subclass (file open failure).
        - If expected groups or datasets (for example, "RealTimeProcessedData", "FilteredData", "Parameters", or "Advanced Recording Parameters/Probe/probeType") are missing or have unexpected shapes, h5py will raise KeyError or dataset-related exceptions when attempting to access them.
        - If numeric fields have unexpected shapes or types, casting or arithmetic (for example, computing num_bits via log2) may raise ValueError or TypeError.
        - Because the filehandle is returned open, failing to call filehandle.close() may lead to resource leaks; consider using the returned filehandle within a context manager or close it explicitly when finished.
    
    Practical significance in SpikeInterface:
        - This parser provides the minimal recording metadata required by SpikeInterface extractors to interpret SiNAPS-formatted recordings and to drive downstream processing such as filtering, spike detection, waveform extraction, and visualization. The returned sampling_frequency, gain, dtype, and probe_type are particularly important to ensure correct unit conversions, correct memory interpretation of on-disk samples, and correct mapping of channels to probe geometry.
    """
    from spikeinterface.extractors.sinapsrecordingextractors import parse_sinapse_h5
    return parse_sinapse_h5(filename)


################################################################################
# Source: spikeinterface.generation.drift_tools.interpolate_templates
# File: spikeinterface/generation/drift_tools.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_generation_drift_tools_interpolate_templates(
    templates_array: numpy.ndarray,
    source_locations: numpy.ndarray,
    dest_locations: numpy.ndarray,
    interpolation_method: str = "cubic"
):
    """Interpolate templates_array to new channel positions for use in spike sorting tasks such as simulating probe motion or remapping templates from one probe geometry to another.
    
    Args:
        templates_array (numpy.ndarray): A dense array of spike templates. This represents template waveforms used in spike sorting and has shape = (num_templates, num_samples, num_channels). The first dimension indexes distinct templates (units), the second indexes time samples of each waveform, and the third indexes recording channels. The array dtype is preserved in the returned result.
        source_locations (numpy.ndarray): The 2D spatial coordinates of the channels that correspond to the third dimension of templates_array. This array must have shape = (num_channels, 2) and provides the (x, y) locations used as input points for spatial interpolation of template amplitudes across channels.
        dest_locations (numpy.ndarray): The target 2D coordinates to which templates_array will be interpolated. This can be either a single new probe geometry with shape = (num_channels, 2) or multiple geometries for producing multiple motions/realizations with shape = (num_motions, num_channels, 2). When dest_locations.ndim == 3 the function broadcasts over the first dimension (num_motions) and returns an extra leading dimension in the output corresponding to each motion.
        interpolation_method (str): The interpolation method string forwarded to scipy.interpolate.griddata for spatial interpolation of channel amplitudes. Default is "cubic". This parameter controls how values between source locations are estimated (for example, typical methods include "linear", "nearest", "cubic" as supported by scipy). The chosen method is passed directly to griddata.
    
    Returns:
        new_templates_array (numpy.ndarray): An array containing the templates interpolated at dest_locations. If dest_locations has ndim == 2, the return shape is (num_templates, num_samples, num_channels) where num_channels == len(dest_locations). If dest_locations has ndim == 3, the return shape is (num_motions, num_templates, num_samples, num_channels) where num_motions == dest_locations.shape[0] and num_channels == dest_locations.shape[1]. The dtype of new_templates_array matches templates_array.dtype. Values outside the convex hull of source_locations are filled with 0 because scipy.interpolate.griddata is called with fill_value=0.
    
    Behavior, side effects, and failure modes:
        This function performs spatial interpolation independently for each template and each time sample: for every template_index and sample_index the channel amplitudes (1D over channels) are interpolated from source_locations to dest_locations using scipy.interpolate.griddata. The function allocates and returns a new numpy array; it does not modify the input arrays (no in-place side effects). Performance can be time- and memory-intensive for large numbers of templates, long waveforms (num_samples), or many destination geometries (num_motions) because the interpolation loop iterates over templates and samples and calls griddata repeatedly. If dest_locations.ndim is not 2 or 3 a ValueError is raised indicating incorrect dimensions for dest_locations. The function relies on scipy.interpolate.griddata semantics for handling interpolation boundaries and supported method strings.
    """
    from spikeinterface.generation.drift_tools import interpolate_templates
    return interpolate_templates(
        templates_array,
        source_locations,
        dest_locations,
        interpolation_method
    )


################################################################################
# Source: spikeinterface.generation.drift_tools.make_linear_displacement
# File: spikeinterface/generation/drift_tools.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_generation_drift_tools_make_linear_displacement(
    start: numpy.ndarray,
    stop: numpy.ndarray,
    num_step: int = 10
):
    """make_linear_displacement generates a sequence of 2D positions that interpolate linearly between a start and stop position. In the SpikeInterface generation/drift_tools context, this function is used to create a temporal sequence of probe or source displacements (x,y coordinates) for simulating linear drift of an extracellular recording setup; the returned sequence can be fed into synthetic recording generation or drift simulation routines to model gradual movement of the probe relative to neural tissue.
    
    Args:
        start (numpy.ndarray): A 1D array of 2 elements giving the start position [x, y]. This represents the initial probe (or source) displacement in the same units used by downstream simulation code (e.g., micrometers). The function treats this as the position at the first timestep.
        stop (numpy.ndarray): A 1D array of 2 elements giving the stop position [x, y]. This represents the final probe (or source) displacement and is treated as the position at the last timestep. Both start and stop must have exactly two components corresponding to the two spatial dimensions.
        num_step (int): The number of discrete timesteps (positions) to generate between start and stop, inclusive. Default: 10. When num_step > 1, the function returns num_step positions that are linearly spaced from start to stop (start is the first row, stop is the last row). When num_step == 1, the function returns a single position equal to the midpoint (start + stop) / 2.
    
    Returns:
        numpy.ndarray: A 2D array of shape (num_step, 2) containing the sequence of displacements. Each row is a 2-element [x, y] position corresponding to successive timesteps from start to stop. The array uses the same numeric dtype and units as the input arrays.
    
    Raises:
        ValueError: If num_step < 1. The function requires at least one step to produce a valid displacement sequence.
    
    Notes:
        - The function is pure and has no side effects: it does not modify the input arrays and only returns a newly allocated numpy.ndarray.
        - The interpolation is linear and computed with vectorized numpy operations; numerical precision and dtype follow numpy semantics based on the input arrays.
        - Typical usage in SpikeInterface is to simulate linear probe drift for synthetic extracellular recordings, where the resulting displacement sequence is applied across time to shift electrode positions or spike templates.
    """
    from spikeinterface.generation.drift_tools import make_linear_displacement
    return make_linear_displacement(start, stop, num_step)


################################################################################
# Source: spikeinterface.generation.drifting_generator.generate_drifting_recording
# File: spikeinterface/generation/drifting_generator.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_generation_drifting_generator_generate_drifting_recording(
    num_units: int = 250,
    duration: float = 600.0,
    sampling_frequency: float = 30000.0,
    probe_name: str = "Neuropixels1-128",
    generate_probe_kwargs: dict = None,
    generate_unit_locations_kwargs: dict = {'margin_um': 20.0, 'minimum_z': 5.0, 'maximum_z': 45.0, 'minimum_distance': 18.0, 'max_iteration': 100, 'distance_strict': False, 'distribution': 'uniform'},
    generate_displacement_vector_kwargs: dict = {'displacement_sampling_frequency': 5.0, 'drift_start_um': [0, 20], 'drift_stop_um': [0, -20], 'drift_step_um': 1, 'motion_list': [{'drift_mode': 'zigzag', 'non_rigid_gradient': None, 't_start_drift': 60.0, 't_end_drift': None, 'period_s': 200}]},
    generate_templates_kwargs: dict = {'ms_before': 1.5, 'ms_after': 3.0, 'mode': 'ellipsoid', 'unit_params': {'alpha': (100.0, 500.0), 'spatial_decay': (10, 45), 'ellipse_shrink': (0.4, 1), 'ellipse_angle': (0, 6.283185307179586)}},
    generate_sorting_kwargs: dict = {'firing_rates': (2.0, 8.0), 'refractory_period_ms': 4.0},
    generate_noise_kwargs: dict = {'noise_levels': (6.0, 8.0), 'spatial_decay': 25.0},
    extra_outputs: bool = False,
    seed: int = None
):
    """spikeinterface.generation.drifting_generator.generate_drifting_recording
    Generates a pair of synthetic extracellular recordings that share identical spike times and unit identities but differ in motion: one "static" recording with no probe drift and one "drifting" recording with unit positions moved over time. This function is intended for benchmarking spike sorters and motion-correction algorithms in extracellular electrophysiology (see SpikeInterface README). It composes probe generation, unit placement, time-varying displacement vector generation, template generation (including multiple spatial positions for drift steps), and noise generation, and then injects the resulting drifting templates into the noise recording to produce Recording objects usable by the SpikeInterface API.
    
    Args:
        num_units (int): Number of simulated neural units (putative neurons) to create. In practice this controls how many templates and spike trains are generated and therefore the density of simultaneous spiking activity in the output Recordings. Default 250 in the function signature.
        duration (float): Total recording duration in seconds. This value is used to determine the number of samples produced (duration * sampling_frequency) and to generate spike times and displacement time series. Default 600.0 seconds.
        sampling_frequency (float): Sampling frequency in Hz used for template generation, noise generation, and conversion between times and sample counts. Typical Neuropixels usage is 30000.0 Hz. Default 30000.0.
        probe_name (str): Name of a built-in probe configuration to use when generate_probe_kwargs is None. The code maps this name to an entry in an internal _toy_probes mapping and constructs a probe with generate_multi_columns_probe. If the name is not present in that mapping, a KeyError will be raised by the lookup. Default "Neuropixels1-128".
        generate_probe_kwargs (dict or None): If provided (dict), this dictionary is passed to generate_multi_columns_probe and supersedes probe_name; it allows callers to specify a custom probe layout, contact positions, and any probe-specific options accepted by the probe generator. If None the function uses the probe configuration indexed by probe_name. Default None.
        generate_unit_locations_kwargs (dict): Keyword arguments forwarded to generate_unit_locations(), which places num_units relative to probe channel positions. Contains placement control parameters (for example margin_um, minimum_z, maximum_z, minimum_distance, max_iteration, distance_strict, distribution). These control geometry, minimum inter-unit distance, sampling strategy and iteration limits; placement can fail or be slower if constraints are infeasible (e.g., too many units for available space) and may raise an error or return fewer placed units depending on the underlying generator. Default dict in signature includes margin_um=20.0, minimum_z=5.0, maximum_z=45.0, minimum_distance=18.0, max_iteration=100, distance_strict=False, distribution="uniform".
        generate_displacement_vector_kwargs (dict): Keyword arguments forwarded to generate_displacement_vector(), which defines how units move over time (drift). Expected keys (as in the default) include displacement_sampling_frequency, drift_start_um, drift_stop_um, drift_step_um, and a motion_list describing drift modes and timing. The function returns per-unit and per-time displacement vectors; incompatible motion specifications or inconsistent durations/periods can cause runtime errors. Default contains a displacement_sampling_frequency of 5.0 Hz and a zigzag motion_list starting at t_start_drift=60.0 s.
        generate_templates_kwargs (dict): Keyword arguments forwarded to generate_templates() used to synthesize per-unit templates on the probe contacts. Must include ms_before and ms_after (ms window around spike), mode (for spatial template shape), and unit_params (distributions or fixed values for template amplitude, spatial decay, shape). The function will call _ensure_unit_params to expand unit_params to num_units using the provided seed. These templates are generated at the static unit locations and then re-generated at each displacement step to produce templates_array_moved for drifting behavior. Default contains ms_before=1.5, ms_after=3.0, mode="ellipsoid", and unit_params ranges for alpha, spatial_decay, ellipse_shrink, ellipse_angle.
        generate_sorting_kwargs (dict): Keyword arguments forwarded to generate_sorting() to synthesize spike trains (ground-truth Sorting object), such as firing_rates (tuple or distribution parameters) and refractory_period_ms. The returned Sorting object is used as the ground-truth for both the static and drifting recordings and will have properties set for gt_unit_locations and max_channel_index based on generated unit placements. Default includes firing_rates=(2.0, 8.0) and refractory_period_ms=4.0.
        generate_noise_kwargs (dict): Keyword arguments forwarded to generate_noise() to synthesize background noise recording that serves as the parent_recording into which templates are injected. Typical keys include noise_levels and spatial_decay; the resulting Recording serves as the same background for both static and drifting outputs. Default includes noise_levels=(6.0, 8.0) and spatial_decay=25.0.
        extra_outputs (bool): If True the function returns an additional dictionary with internal intermediate results useful for motion benchmarking and debugging (see Returns section). If False only the three primary objects (static_recording, drifting_recording, sorting) are returned. Default False.
        seed (int or None): Seed or None passed to _ensure_seed to control randomness across probe/unit placement, template synthesis, spike times, displacement vectors, and noise generation. Providing an integer makes generation deterministic and reproducible; None lets the internal _ensure_seed produce a random seed. The function uses _ensure_seed(seed) at the start and passes the resulting seed to sub-generators.
    
    Returns:
        tuple: When extra_outputs is False returns a 3-tuple:
            static_recording (Recording): A Recording object created by InjectDriftingTemplatesRecording where displacement_vectors and displacement_unit_factor are zero, so templates remain fixed in space. This recording represents the same neurons and spike trains as the drifting recording but without motion; it is intended as a baseline for assessing drift effects on spike sorting.
            drifting_recording (Recording): A Recording object created by InjectDriftingTemplatesRecording that applies the generated displacement_vectors and displacement_unit_factor to drifting_templates, producing time-varying template projections on the probe contacts. This recording is intended to simulate probe or tissue motion and to be used to benchmark motion-correction and drift-robust sorting algorithms.
            sorting (Sorting): The ground-truth Sorting object containing the generated spike trains for num_units and properties set by this function. It will have properties "gt_unit_locations" (unit_locations as generated) and "max_channel_index" (index of nearest contact per unit). This Sorting object is the canonical ground truth for both returned Recordings.
        tuple: When extra_outputs is True returns a 4-tuple (static_recording, drifting_recording, sorting, extra_infos) where extra_infos is a dict containing additional intermediate data useful for analysis and benchmarking:
            extra_infos (dict): Dictionary with keys:
                displacement_vectors: numpy array of per-time displacement vectors produced by generate_displacement_vector; these describe global or per-region motion used to move templates.
                displacement_sampling_frequency: sampling frequency (Hz) of the displacement vector time series.
                unit_locations: numpy array of unit locations (x, y, z) in the same spatial units used for template generation; these are the base positions before any applied drift.
                displacement_unit_factor: array or factor describing how displacement vectors map to per-unit displacements (used internally when non-rigid motion is simulated).
                unit_displacements: per-unit displacement over time derived from displacement_vectors and displacement_unit_factor.
                templates: the Templates object produced from static templates (Templates) before conversion to DriftingTemplates.
            These entries are provided exactly as produced by the internal generators (typically numpy arrays and SpikeInterface Template objects) and are intended for detailed motion-benchmarking, visualization, or validation.
    
    Behavioral notes, side effects, and failure modes:
        - If generate_probe_kwargs is None the function looks up probe configuration by probe_name in an internal _toy_probes mapping and calls generate_multi_columns_probe; a missing probe_name key will raise a KeyError.
        - Unit placement is performed by generate_unit_locations with generate_unit_locations_kwargs; infeasible placement constraints (for example too many units given minimum_distance and margin_um) can cause the placement routine to fail or to exhaust max_iteration and either raise an error or return an unexpected distribution of locations depending on the underlying implementation.
        - The function calls _ensure_unit_params to expand generate_templates_kwargs["unit_params"] into per-unit parameters; these per-unit parameters are then used to generate templates deterministically from the provided seed. If unit_params are malformed, template generation will raise an error.
        - To model drift, templates are regenerated at each displacement step rather than relying on precomputed interpolation; the code populates drifting_templates.templates_array_moved by repeatedly calling generate_templates at shifted unit locations for each displacement step. This approach avoids interpolation edge effects but increases memory and compute cost; templates_array_moved has shape (num_displacement_steps, num_units, num_samples, num_channels).
        - The function generates a noise Recording via generate_noise and uses it as parent_recording for both InjectDriftingTemplatesRecording outputs; both Recordings therefore share the same noise realization and ground-truth spike trains, differing only in the applied displacement vectors and displacement_unit_factor.
        - Because templates are regenerated for each displacement step, memory usage may be substantial for long recordings, many displacement steps, high sampling_frequency, or large num_units. Users should reduce duration, sampling_frequency, or num_units or adjust drift_step_um / displacement_sampling_frequency to limit memory footprint.
        - The function sets sorting properties "gt_unit_locations" and "max_channel_index" based on generated unit locations and nearest probe contact; downstream code may rely on these properties for evaluation and visualization.
        - Invalid or inconsistent kwargs passed to the underlying generators (generate_templates, generate_sorting, generate_noise, generate_displacement_vector) will propagate exceptions from those functions; callers should validate kwargs against the respective generator documentation.
        - The returned Recording objects are SpikeInterface Recording objects and may need to be serialized or processed further (e.g., waveform extraction, spike sorting) using the rest of the SpikeInterface API.
    """
    from spikeinterface.generation.drifting_generator import generate_drifting_recording
    return generate_drifting_recording(
        num_units,
        duration,
        sampling_frequency,
        probe_name,
        generate_probe_kwargs,
        generate_unit_locations_kwargs,
        generate_displacement_vector_kwargs,
        generate_templates_kwargs,
        generate_sorting_kwargs,
        generate_noise_kwargs,
        extra_outputs,
        seed
    )


################################################################################
# Source: spikeinterface.generation.drifting_generator.make_one_displacement_vector
# File: spikeinterface/generation/drifting_generator.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_generation_drifting_generator_make_one_displacement_vector(
    drift_mode: str = "zigzag",
    duration: float = 600.0,
    amplitude_factor: float = 1.0,
    displacement_sampling_frequency: float = 5.0,
    t_start_drift: float = None,
    t_end_drift: float = None,
    period_s: float = 200,
    bump_interval_s: tuple = (30, 90.0),
    seed: int = None
):
    """spikeinterface.generation.drifting_generator.make_one_displacement_vector generates a single, time-indexed displacement vector used to simulate probe drift for spike-sorting workflows in SpikeInterface. The function constructs a normalized displacement shape (base range [-0.5, 0.5]) sampled at a specified rate over a recording duration, then scales it by amplitude_factor so it can be interpreted as a spatial displacement trace (commonly interpreted in micrometers when amplitude_factor is given in micrometers). This displacement vector can be applied to simulated extracellular recordings or waveform templates to model slow drift (zigzag), discrete alternating bumps, or a constrained random walk during a specified drift interval.
    
    Args:
        drift_mode (str): Mode of the drift shape. Supported values implemented by this function are "zigzag", "bump", and "random_walk". "zigzag" produces a triangular/periodic sawtooth-like waveform with period controlled by period_s. "bump" produces piecewise constant segments that alternate between +0.5 and -0.5 at randomly spaced bump times drawn from bump_interval_s. "random_walk" produces a cumulative sum of +/-1 steps (constrained to the drift interval) normalized to fit in the base [-0.5, 0.5] range. The chosen mode determines how displacement evolves over time and therefore how simulated probe movement affects spike projections.
        duration (float): Total recording duration in seconds. The output vector covers the full interval [0, duration). Internally the number of samples is computed as ceil(displacement_sampling_frequency * duration), so the vector length equals this sample count.
        amplitude_factor (float): Multiplicative scaling applied to the normalized shape (base range [-0.5, 0.5]). Use this parameter to convert the normalized displacement into physical units (for example set amplitude_factor in micrometers to obtain a displacement trace in micrometers). A value of 1.0 returns the base normalized shape; other values scale amplitude proportionally.
        displacement_sampling_frequency (float): Sampling frequency in Hz used to discretize the displacement vector. Determines the temporal resolution of the returned vector. Sample indices are computed as integer indices from times using this sampling rate.
        t_start_drift (float): Start time in seconds when non-zero drift begins. If None, drift begins at 0.0 s. Values are converted to a start index using displacement_sampling_frequency. Values must be less than duration (asserted).
        t_end_drift (float): End time in seconds when the drift stops. If None, drift ends at duration. Values are converted to an end index using displacement_sampling_frequency. Must be <= duration (asserted). Outside the [t_start_drift, t_end_drift) interval the returned vector is held at zero (or the last triangular value for zigzag after end).
        period_s (float): Period in seconds used by the "zigzag" mode to define the triangular waveform frequency (freq = 1.0 / period_s). Only used when drift_mode == "zigzag".
        bump_interval_s (tuple): Two-element tuple (min_interval_s, max_interval_s) specifying the uniform range, in seconds, from which inter-bump intervals are drawn in "bump" mode. The implementation draws successive intervals with numpy.random.default_rng(seed) and cumsums them to create bump times in [t_start_drift, t_end_drift). Bump segments alternate between +0.5 and -0.5.
        seed (int): Optional integer seed for the random number generator used in "bump" and "random_walk" modes to make bump times or random steps reproducible. If None, the RNG is nondeterministic.
    
    Behavior, defaults, and failure modes:
        The function first resolves t_start_drift to 0.0 and t_end_drift to duration when they are None. It asserts that t_start_drift < duration and t_end_drift <= duration; failing these assertions raises AssertionError with a descriptive message. The output length is computed as num_samples = ceil(displacement_sampling_frequency * duration) and the returned numpy array has one entry per sample covering the full recording interval.
        For "zigzag": a triangular waveform is generated using scipy.signal.sawtooth, then shifted to the base range [-0.5, 0.5]; samples before t_start_drift are zero, samples in [t_start_drift, t_end_drift) follow the triangle, and samples after t_end_drift are set to the last triangle value to hold constant.
        For "bump": the function generates bump times by drawing intervals uniformly in bump_interval_s until reaching t_end_drift (using numpy.random.default_rng with seed), then fills successive segments between bump times with +0.5 or -0.5 alternating. The displacement outside the drift interval remains zero.
        For "random_walk": a sequence of integer steps in {0,1} is sampled and mapped to {-1,+1}; steps outside the drift window are forced to 0. The cumulative sum of these steps is normalized by (2 * max_abs_value) so the base range is roughly within [-0.5, 0.5]. If the random walk has no variability (max absolute value equals zero) the normalization will divide by zero, which can produce NaNs or runtime warnings; the implementation does not explicitly check for this condition.
        The function uses numpy and scipy.signal internally; providing an unsupported drift_mode (anything other than "zigzag", "bump", or "random_walk") raises ValueError.
        No external state is modified except that providing seed controls the reproducibility of the random number generator used for bump/random_walk generation. The function does not write files or alter global randomness state beyond using numpy.random.default_rng with the provided seed.
    
    Returns:
        numpy.ndarray: 1-D numpy array of floats with length ceil(displacement_sampling_frequency * duration). The array contains the time-series displacement vector produced according to the chosen drift_mode, scaled by amplitude_factor. The base (pre-scale) shape is in the range [-0.5, 0.5]; final returned values equal base_shape * amplitude_factor. This vector is intended to be sampled at displacement_sampling_frequency and can be applied to simulated recordings or templates to emulate probe drift.
    
    Raises:
        AssertionError: If t_start_drift >= duration or t_end_drift > duration.
        ValueError: If drift_mode is not one of the implemented modes ("zigzag", "bump", "random_walk").
    """
    from spikeinterface.generation.drifting_generator import make_one_displacement_vector
    return make_one_displacement_vector(
        drift_mode,
        duration,
        amplitude_factor,
        displacement_sampling_frequency,
        t_start_drift,
        t_end_drift,
        period_s,
        bump_interval_s,
        seed
    )


################################################################################
# Source: spikeinterface.postprocessing.amplitude_scalings.find_collisions
# File: spikeinterface/postprocessing/amplitude_scalings.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_postprocessing_amplitude_scalings_find_collisions(
    spikes: numpy.ndarray,
    spikes_within_margin: numpy.ndarray,
    delta_collision_samples: int,
    sparsity_mask: numpy.ndarray
):
    """spikeinterface.postprocessing.amplitude_scalings.find_collisions identifies temporal and spatial collisions between spikes recorded and sorted across units, for use during spike sorting post-processing (for example, in amplitude scaling steps where overlapping spikes can bias amplitude estimates).
    
    Finds, for each spike in the provided spikes array, other spikes that (1) occur within a temporal window defined by delta_collision_samples around the spike peak sample index, and (2) are produced by units that have overlapping spatial footprint on any recording channel as indicated by sparsity_mask. Temporal overlap is determined by comparing the spike["sample_index"] values and using numpy.searchsorted on spikes_within_margin["sample_index"] to efficiently locate candidate spikes within the time window. Spatial overlap is determined by comparing unit indices via the provided sparsity_mask (a boolean unit-by-channel mask) using the internal _are_units_spatially_overlapping check. The function returns a dictionary keyed by the index of the spike in the input spikes array; each value is a numpy.ndarray of spike records that overlap with that spike, where the spike itself appears at position 0 of the array.
    
    Args:
        spikes (numpy.ndarray): Structured numpy array of spikes to analyze. Each element must be a record (for example a structured/recarray) exposing the fields "sample_index", "channel_index", "amplitude", "segment_index", "unit_index", and "in_margin" (in that or an equivalent layout). This array provides the primary list of spikes for which collisions are sought and its indices are used as the keys in the returned dictionary. The "sample_index" field is used to define the temporal collision window; the "unit_index" field is used to test spatial overlap via sparsity_mask.
        spikes_within_margin (numpy.ndarray): Structured numpy array of spikes with the same record format as spikes (same fields as described above). This array must contain, for every record present in spikes, an identical record instance so that equality comparison (spikes_within_margin == spike) can locate the corresponding position. The implementation relies on spikes_within_margin["sample_index"] being sorted in ascending order so that numpy.searchsorted returns a correct temporal-window range of candidate overlapping spikes.
        delta_collision_samples (int): Integer count of samples that defines the temporal half-width for collision detection. For each spike, the function defines a temporal window [sample_index - delta_collision_samples, sample_index + delta_collision_samples] and considers any spike in spikes_within_margin whose "sample_index" lies within that window as a temporal candidate for collision. This parameter is interpreted as a non-negative integer representing the maximum allowed sample distance for temporal overlap.
        sparsity_mask (numpy.ndarray): Boolean numpy array of shape (num_units, num_channels) where True indicates that a given unit produces signal on a given channel. This mask is used to determine spatial overlap between two units: two spikes are considered spatially overlapping if their corresponding unit rows in sparsity_mask share any True entry (i.e., they have at least one channel in common). The function uses unit indices from the spike records to index into this mask.
    
    Returns:
        collision_spikes_dict (dict): A dictionary mapping spike indices (int, indices into the input spikes array) to numpy.ndarray values. Each value is a numpy.ndarray of spike records (same dtype/structure as the input spikes arrays) that temporally and spatially overlap the keyed spike; by construction the keyed spike itself appears at position 0 of the array. If a given spike has no overlapping spikes (other than itself), the spike index will not appear in the dictionary. If no collisions are found for any spikes, the function returns an empty dict.
    
    Behavior, side effects, and failure modes:
        This function does not modify its input arrays; it only allocates and returns a new dictionary and arrays of overlapping spike records. It iterates over every element of spikes and for each element performs a search in spikes_within_margin to obtain candidate temporal neighbors and then filters those candidates by spatial overlap via sparsity_mask.
        The function assumes that spikes_within_margin contains an identical record for every record in spikes; if a given spike record is not present in spikes_within_margin, the equality lookup numpy.where(spikes_within_margin == spike)[0][0] will raise an IndexError. If spikes_within_margin["sample_index"] is not sorted ascending, numpy.searchsorted will yield incorrect temporal ranges and collisions may be missed or misidentified. If the structured arrays do not expose the required fields ("sample_index" and "unit_index" at minimum), KeyError or similar access errors will occur. If sparsity_mask does not have rows indexed by unit indices present in the spike records, indexing errors will occur when checking spatial overlap.
        Performance note: the current implementation searches per-spike and concatenates arrays iteratively; for large datasets this can be relatively slow (the source contains a TODO to refactor for speed). No in-place changes are made to inputs; memory usage increases proportional to the number and sizes of detected collisions because arrays of overlapping spike records are created for each collision entry.
    """
    from spikeinterface.postprocessing.amplitude_scalings import find_collisions
    return find_collisions(
        spikes,
        spikes_within_margin,
        delta_collision_samples,
        sparsity_mask
    )


################################################################################
# Source: spikeinterface.postprocessing.amplitude_scalings.fit_collision
# File: spikeinterface/postprocessing/amplitude_scalings.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_postprocessing_amplitude_scalings_fit_collision(
    collision: numpy.ndarray,
    traces_with_margin: numpy.ndarray,
    nbefore: int,
    all_templates: numpy.ndarray,
    sparsity_mask: numpy.ndarray,
    cut_out_before: int,
    cut_out_after: int
):
    """Compute the best-fit non-negative scaling factors for a collision between a target spike and its temporally and spatially overlapping spikes in the SpikeInterface postprocessing workflow.
    
    This function is used in SpikeInterface to estimate how much each colliding spike's unit template contributes to an observed multi-channel waveform that contains an overlapping set of spikes. It constructs a multivariate linear regression problem where the observed local waveform (y) is modeled as a non-negative linear combination of temporally shifted unit templates (columns of X). Each returned coefficient is the estimated scaling factor for the corresponding spike in the input collision array and can be used to decompose observed amplitudes into contributions from the spike of interest and its colliders, which is useful for amplitude-scaling corrections, deconvolution, or collision-aware waveform analyses.
    
    Args:
        collision (numpy.ndarray): A one-dimensional structured numpy array of shape (n_colliding_spikes,) containing the set of colliding spikes. The first element is the spike of interest and subsequent entries are spikes that overlap it in time. Each array element is expected to be a record with fields at least: sample_index (int), channel_index (int), amplitude (float), segment_index (int), unit_index (int), in_margin (bool). The function uses sample_index to align templates temporally and unit_index to select the corresponding unit template from all_templates. The order of entries in this array determines the order of the returned scaling factors.
        traces_with_margin (numpy.ndarray): A two-dimensional numpy array of shape (n_samples, n_channels) containing the raw or preprocessed extracellular traces with an added margin around the central event. This array provides the observed waveform y from which contributions of colliding spikes are fit. The temporal indices in collision["sample_index"] refer to indices into this array.
        nbefore (int): The number of samples before the nominal spike center that were used when creating unit templates (i.e., the template center offset). This integer is used to slice each unit template to the same cutout window as used for the traces (together with cut_out_before and cut_out_after) so templates are temporally aligned with traces_with_margin.
        all_templates (numpy.ndarray): A three-dimensional numpy array of shape (n_units, n_samples_template, n_channels) containing unit templates for all units. Each template is indexed by unit_index (as present in the collision structured array) and is assumed to have the same channel ordering as traces_with_margin. The function slices these templates using nbefore, cut_out_before and cut_out_after to create temporally shifted template contributions for each colliding spike.
        sparsity_mask (numpy.ndarray): A two-dimensional boolean numpy array of shape (n_units, n_channels) indicating whether a given unit is represented on a given channel (True when present). The function computes the union across units present in collision to select a reduced set of channels (sparse_indices) to include in the regression; this reduces computation and matches SpikeInterface sparsity assumptions.
        cut_out_before (int): The number of samples to include before each spike center when constructing the local template cutout. This integer, together with cut_out_after, defines the temporal window (cut_out_before + cut_out_after samples) around each spike used to slice unit templates and to place them into the design matrix X.
        cut_out_after (int): The number of samples to include after each spike center when constructing the local template cutout. See cut_out_before for how this defines the temporal window used for slicing and aligning templates.
    
    Returns:
        numpy.ndarray: A one-dimensional numpy array of length equal to the number of colliding spikes (len(collision)). Each element is the fitted, non-negative scaling factor (regression coefficient) for the corresponding spike in the input collision array (same order). Coefficients are estimated using sklearn.linear_model.LinearRegression with fit_intercept=True and positive=True, so returned values are >= 0. The intercept term is fitted internally but not returned. If the regression is degenerate (e.g., insufficient channels selected, a completely empty local waveform, or exact linear dependence), sklearn's least-squares solver behavior determines the coefficients; in practice this can produce zeros or numerically unstable coefficients and may raise warnings from sklearn.
    
    Notes on behavior, side effects, and failure modes:
        - The function selects channels to include in the fit by taking the logical OR across sparsity_mask rows for all units present in collision. If no channel is selected (all False), the function will attempt to build an empty observation vector y and will likely raise a downstream error (e.g., from sklearn) or return an empty coefficients array. Callers should validate sparsity_mask and collision to ensure at least one channel is active.
        - The local temporal window used for fitting spans from max(0, min(sample_index) - cut_out_before) to min(traces_with_margin.shape[0], max(sample_index) + cut_out_after). If this window has zero length (e.g., due to invalid indices or empty traces_with_margin), the function will fail when constructing y or X.
        - Templates that extend beyond the edges of the local window are clipped; the code handles left and right border clipping so partial template contributions are inserted appropriately into the design matrix.
        - The function imports sklearn.linear_model.LinearRegression internally; an ImportError will be raised if scikit-learn is not installed in the runtime environment.
        - The function does not modify its input arrays (traces_with_margin, all_templates, collision, sparsity_mask) and has no external side effects.
        - Returned scalings correspond to collision entries in the same order and can be used downstream for amplitude-correction, demixing, or reporting within the SpikeInterface postprocessing pipeline.
    """
    from spikeinterface.postprocessing.amplitude_scalings import fit_collision
    return fit_collision(
        collision,
        traces_with_margin,
        nbefore,
        all_templates,
        sparsity_mask,
        cut_out_before,
        cut_out_after
    )


################################################################################
# Source: spikeinterface.postprocessing.correlograms.correlogram_for_one_segment
# File: spikeinterface/postprocessing/correlograms.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_postprocessing_correlograms_correlogram_for_one_segment(
    spike_times: numpy.ndarray,
    spike_unit_indices: numpy.ndarray,
    window_size: int,
    bin_size: int
):
    """Compute cross-correlograms for one recording segment using an optimized algorithm adapted from the Phy package (originally by Cyrille Rossant). This function is used in SpikeInterface post-processing to quantify temporal relationships between spike trains produced by spike sorting: it bins time differences between spikes from all pairs of units within a specified window (measured in samples) and returns integer counts per lag-time bin for every unit pair. The implementation is optimized for large spike collections by iteratively shifting the spike train, masking out spikes outside the window, binning delays by floor division, and using ravel_multi_index plus bincount to increment counts efficiently.
    
    Args:
        spike_times (numpy.ndarray): A 1-D array of spike times in samples (not seconds). This array must contain the timestamps for spikes from all units in the segment and must have the same length as spike_unit_indices. The algorithm assumes spike_times are sorted in non-decreasing order (temporal order); incorrect ordering will produce incorrect correlograms. Each element is an integer sample index representing when a spike occurred in the recording.
        spike_unit_indices (numpy.ndarray): A 1-D array of integer labels indicating the unit associated with each spike in spike_times. This array must have the same shape and ordering as spike_times (i.e., spike_unit_indices[i] is the unit label for spike_times[i]). Unit labels are used as array indices when building the correlogram; in typical usage they should map to 0..(num_units-1) indices (contiguous integer labels), otherwise indexing errors or out-of-range exceptions may occur.
        window_size (int): The half-window size (in samples) over which to search for matching spikes for cross-correlation. The function computes the number of lag bins from this window size and bin_size (via the internal _compute_num_bins call). window_size must be a positive integer expressed in the same sample units as spike_times and bin_size. Larger window_size increases the number of lag bins and thus memory and computation.
        bin_size (int): The size (in samples) of each lag-time bin. bin_size must be a positive integer expressed in samples. Delays between spikes are converted to bin indices using floor division by bin_size; the binning determines the temporal resolution of the correlograms.
    
    Returns:
        correlograms (numpy.ndarray): A 3-D integer array of shape (num_units, num_units, num_bins) and dtype int64. Here num_units is the number of unique labels in spike_unit_indices and num_bins is computed from window_size and bin_size. Element correlograms[i, j, k] is the count of spike pairs where a spike from unit i occurred with a lag that falls into bin k relative to a spike from unit j; positive lag values correspond to spikes from unit i occurring after spikes from unit j. Auto-correlograms appear on the diagonal (i == j); note that zero-lag self-coincidences are not counted because the algorithm starts with a shift of 1 (no zero-shift pairs are included). The returned array contains integer counts only; no timestamp information is returned.
    
    Behavior, side effects, and failure modes:
        - This function is pure (no external side effects) and returns a newly allocated numpy.ndarray correlograms. It does not modify the input arrays.
        - The function is optimized: it iteratively increases a shift between spike indices, computes spike time differences within the shifting window, bins them by floor division by bin_size, applies masks to ignore differences outside half the window, and accumulates counts using ravel_multi_index and numpy.bincount for speed.
        - Inputs must be consistent in length and ordering: spike_times and spike_unit_indices must have identical shapes, and spike_times must be sorted ascending; otherwise results will be incorrect or an IndexError may be raised.
        - bin_size and window_size must be positive integers; non-positive values will lead to incorrect behavior or exceptions from internal computations.
        - Unit labels in spike_unit_indices must be suitable for direct indexing into an array of shape (num_units, ...). Using non-contiguous or large integer labels (e.g., labels that do not start at 0 and run contiguously) may result in IndexError or an incorrect mapping; if labels are non-contiguous, they should be remapped to 0..(num_units-1) before calling this function.
        - Memory and compute scale with num_units^2 * num_bins. For large numbers of units or very fine temporal resolution (small bin_size) over large windows, the returned array can be large and consume substantial memory; plan resources accordingly.
        - Temporal units: all timing inputs and outputs are in samples. To interpret lags in seconds, convert bin indices to time using the recording sampling rate outside this function (lag_time_seconds = (bin_index - center_bin) * bin_size / sampling_rate).
    
    Usage context:
        This function is intended for post-processing spike sorting outputs within SpikeInterface to compute auto- and cross-correlograms for analyzing synchrony, latency relationships, or refractory period structure between sorted units. It implements a well-tested algorithm from the Phy project and is designed for efficiency on typical extracellular spike datasets.
    """
    from spikeinterface.postprocessing.correlograms import correlogram_for_one_segment
    return correlogram_for_one_segment(
        spike_times,
        spike_unit_indices,
        window_size,
        bin_size
    )


################################################################################
# Source: spikeinterface.postprocessing.localization_tools.enforce_decrease_shells_data
# File: spikeinterface/postprocessing/localization_tools.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_postprocessing_localization_tools_enforce_decrease_shells_data(
    wf_data: numpy.ndarray,
    maxchan: int,
    radial_parents: dict,
    in_place: bool = False
):
    """spikeinterface.postprocessing.localization_tools.enforce_decrease_shells_data enforces a radial decrease constraint on a one-dimensional per-channel data vector used in localization postprocessing within the SpikeInterface framework. In the spike-sorting/localization domain (see README), this function is used to regularize per-channel scalar metrics (for example per-channel waveform amplitudes, energies, or localization scores computed for a single waveform or template) so that values on channels in outer radial shells do not exceed the maximum value of their parent (closer-to-center) channels; any channel value that exceeds its parents' maximum is scaled down to that maximum.
    
    Args:
        wf_data (numpy.ndarray): A one-dimensional NumPy array of length C containing scalar values for each channel for a single waveform or template. The function expects wf_data.shape to unpack as (C,). These values are the per-channel metrics (amplitude/energy/score) that will be checked and possibly scaled to enforce a nonincreasing radial profile from the center channel outward. The array's dtype is preserved on return. If wf_data is not one-dimensional, a ValueError will be raised by the attempted shape unpacking.
        maxchan (int): Integer channel index that identifies the central (reference) channel for the radial parent relationships provided in radial_parents. The function looks up radial_parents[maxchan] to obtain the iteration order and parent sets; if maxchan is not present as a key in radial_parents, a KeyError will be raised.
        radial_parents (dict): Dictionary used to describe the radial-parent structure for localization shells. For the key equal to maxchan, radial_parents[maxchan] must be an iterable of pairs (c, parents_rel) where c is an integer channel index to process and parents_rel is an indexable/iterable of integer channel indices (for example a list or NumPy array) indicating the parent channels of c (closer to the center). For each pair the function computes parent_max = decreasing_data[parents_rel].max() and enforces decreasing_data[c] <= parent_max by scaling down decreasing_data[c] when needed. If any indices in parents_rel are out of bounds for wf_data, an IndexError will be raised.
        in_place (bool): If True, the operation is performed in-place on the provided wf_data array and the same object is returned (mutating the caller's array). If False (default), the function works on and returns a shallow copy of wf_data, leaving the original array unmodified. Use in_place=True to avoid an additional array allocation when mutation is acceptable.
    
    Returns:
        numpy.ndarray: A one-dimensional NumPy array of the same shape and dtype as the input wf_data containing the modified per-channel values after enforcing the radial decrease constraint. If in_place is True, the returned array is the same object as the input wf_data (modified in place); otherwise it is a copy and the original wf_data remains unchanged.
    
    Raises:
        ValueError: If wf_data does not have a one-dimensional shape that can be unpacked as (C,).
        KeyError: If maxchan is not a key in radial_parents.
        IndexError: If any channel index referenced in radial_parents[maxchan] or its parents_rel lists is out of bounds for wf_data.
        TypeError: If wf_data is not indexable with the integer indices found in radial_parents, or if types do not support comparison/division used in the algorithm.
        ZeroDivisionError: In the unlikely case that a channel value decreasing_data[c] equals zero while the code attempts to scale it (this occurs if decreasing_data[c] > parent_max and decreasing_data[c] == 0), a division by zero can occur. Validate input values if zero entries combined with negative parent maxima are possible in your preprocessing pipeline.
    """
    from spikeinterface.postprocessing.localization_tools import enforce_decrease_shells_data
    return enforce_decrease_shells_data(wf_data, maxchan, radial_parents, in_place)


################################################################################
# Source: spikeinterface.postprocessing.localization_tools.get_grid_convolution_templates_and_weights
# File: spikeinterface/postprocessing/localization_tools.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_postprocessing_localization_tools_get_grid_convolution_templates_and_weights(
    contact_locations: numpy.ndarray,
    radius_um: float = 40,
    upsampling_um: float = 5,
    margin_um: float = 50,
    weight_method: dict = {'mode': 'exponential_3d'}
):
    """Get an upsampled 2D grid of artificial templates from a probe contact layout for use in localization and post-processing of spike-sorted extracellular recordings.
    
    This function is used in SpikeInterface post-processing to construct a dense, regularly spaced set of template positions (a grid) over the spatial extent of a probe's recorded contacts. The grid is created in physical units of micrometers (um) and is intended for downstream localization and template-weighting operations (for example, to estimate spike source locations or to compute convolutional template weights for channels). The function computes pairwise distances between each probe contact and each grid template, creates a sparsity mask of nearby templates per contact based on a radius, and delegates to get_convolution_weights() (via the weight_method argument) to compute per-channel template weights and any z-axis scaling factors required by the chosen weight model.
    
    Args:
        contact_locations (array): The 2D positions of the probe channels in micrometers. This must be an array with shape (n_channels, 2) where each row is (x, y) in um. The positions define the spatial extent used to build the upsampled grid and to compute distances to templates; incorrect dimensionality or units will produce incorrect grids or distance calculations.
        radius_um (float): Radius in micrometers used to define sparsity of templates relative to channels. A template is considered "near" a contact when its Euclidean distance to that contact is <= radius_um. This parameter controls the boolean nearest_template_mask output and therefore which templates are treated as relevant per channel. Default: 40.
        upsampling_um (float): Spacing of the upsampled grid in micrometers. The grid of artificial templates is generated by stepping in x and y from the contact extent ± margin_um using this spacing. Smaller values produce a denser grid (higher nb_templates) and increased memory/time cost; larger values produce a coarser grid. Default: 5.
        margin_um (float): Additional margin in micrometers added on all sides of the min/max extents of contact_locations before creating the grid. This ensures templates extend beyond the outermost contacts by margin_um. Choosing a large margin increases the grid area and computational cost; too small a margin may omit templates needed for edge localization. Default: 50.
        weight_method (dict): Dictionary of parameters forwarded to get_convolution_weights() to compute template weights from the contact-to-template distance matrix. In practice this dictionary must include a "mode" key indicating the weighting model (e.g., "gaussian_2d" for a 2D Gaussian KS-like model or "exponential_3d" for the default exponential 3D model). The exact semantics and any additional keys are those expected by get_convolution_weights(); this argument determines how the distances are converted into per-channel, per-template weights and any z-axis scaling factors.
    
    Behavior, side effects, defaults, and failure modes:
        - The function computes x and y ranges from contact_locations, expands them by margin_um, then constructs a regular meshgrid with spacing upsampling_um. An internal small epsilon (upsampling_um / 10) is added to the upper bounds to include the final endpoint.
        - It computes pairwise Euclidean distances between each contact and each template position using sklearn.metrics.pairwise_distances (sklearn is imported inside the function). The resulting distance matrix has shape (n_channels, nb_templates).
        - nearest_template_mask is produced as a boolean mask (same shape as the distance matrix) where entries are True if distance <= radius_um.
        - weights and z_factors are produced by calling get_convolution_weights(dist, **weight_method). The exact numerical meaning and shape of z_factors depend on the chosen weight_method (for example, exponential_3d may return z scaling factors used to modulate weights along the third dimension). weights will correspond to per-channel weights for each template consistent with the distance matrix.
        - Default parameter values are radius_um=40, upsampling_um=5, margin_um=50, and weight_method={"mode": "exponential_3d"} as given in the function signature.
        - Potential failure modes include invalid input shapes for contact_locations (e.g., not two columns), extremely small upsampling_um or very large margin_um producing very large nb_templates that can cause high memory usage or MemoryError, and providing a weight_method that is incompatible with get_convolution_weights() which may raise an exception. The function does not validate units; contact_locations must already be in micrometers for meaningful results.
    
    Returns:
        template_positions (array): Array of shape (nb_templates, 2) containing the (x, y) positions in micrometers of the upsampled grid templates. These positions are the coordinates of the artificial templates used for localization and weighting relative to the probe contacts.
        weights (array): Per-channel weights for each template. This array is produced by get_convolution_weights() from the contact-to-template distance matrix and the provided weight_method. Practically, these weights are used to combine channel signals or to score template relevance per contact; the array shape matches the distance matrix (n_channels, nb_templates) as returned by sklearn.metrics.pairwise_distances.
        nearest_template_mask (array): Boolean mask with shape (n_channels, nb_templates). True indicates that the template is within radius_um of the given contact and therefore considered spatially "near" that contact. This mask is useful to sparsify computations by restricting attention to nearby templates.
        z_factors (array): Array of z-axis scaling factors (or other auxiliary factors) returned by get_convolution_weights() that were used to generate the weights along the third dimension when using 3D-aware weight models (for example, exponential_3d). The exact interpretation and length of this array depend on the selected weight_method and follow the semantics of get_convolution_weights().
    """
    from spikeinterface.postprocessing.localization_tools import get_grid_convolution_templates_and_weights
    return get_grid_convolution_templates_and_weights(
        contact_locations,
        radius_um,
        upsampling_um,
        margin_um,
        weight_method
    )


################################################################################
# Source: spikeinterface.postprocessing.localization_tools.make_radial_order_parents
# File: spikeinterface/postprocessing/localization_tools.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_postprocessing_localization_tools_make_radial_order_parents(
    geom: numpy.ndarray,
    neighbours_mask: numpy.ndarray,
    n_jumps_per_growth: int = 1,
    n_jumps_parent: int = 3
):
    """Pre-computes a radial parent lookup structure used by enforce_decrease_shells in the localization post-processing tools of SpikeInterface. This function analyzes the spatial probe geometry and a per-channel neighbor mask to determine, for each channel, an ordered list of "child" neighbor channels and for each child the set of already-seen "parent" neighbor indices that lie closer to the source channel. The resulting structure is intended to be used by enforce_decrease_shells to enforce a radial decrease constraint (for example, decreasing spike amplitude or score with radial distance) when post-processing localization or quality metrics.
    
    Args:
        geom (numpy.ndarray): Array containing spatial coordinates for each recording channel. The function uses len(geom) as the number of channels and passes geom to the internal make_shell and make_shells helpers to compute spatial shells. Practically, geom represents the probe geometry (one row per channel, coordinate columns depend on probe dimensionality) and determines how "closeness" and shells are computed for radial growth.
        neighbours_mask (numpy.ndarray): Per-channel neighbor mask that identifies which channels should be considered neighbors of each channel. Each element iterated from neighbours_mask is interpreted as a boolean-like mask or array convertible to indices via numpy.flatnonzero; the function converts each row to a list of neighbor channel indices and only considers neighbors with indices < n_channels. In practice this mask restricts the set of candidate channels for which radial parent relationships are computed (e.g., channels within a recording adjacency or a pruning radius).
        n_jumps_per_growth (int): Number of shell "jumps" to use when growing the search for progressively more distant channels during parent discovery. Defaults to 1. This parameter controls the granularity of each growth step: larger values make each growth step reach farther channels, which reduces the number of iterations but coarsens the radial ordering used to find parents. It is passed to make_shell when expanding the local neighborhood for a channel.
        n_jumps_parent (int): Number of shell jumps used when precomputing per-channel parent shells via make_shells. Defaults to 3. This parameter controls the radius used to determine candidate parent channels for any channel and is used to compute the static shells array (shells[new_chan]) that is intersected with already-seen channels to identify parents.
    
    Returns:
        list: A list of length equal to the number of channels (len(geom)). Each element radial_parents[channel] is itself a list of tuples (child_pos, parents_rel) describing how neighbor channels of `channel` should be connected to previously seen parent channels for radial ordering. For each tuple:
            child_pos (int): the index (position) of the child channel inside the channel's neighbor list (i.e., an integer equal to the position in the array produced by numpy.flatnonzero(neighbours_mask[channel])).
            parents_rel (numpy.ndarray): a one-dimensional numpy.ndarray of integer indices. Each integer is an index into the same neighbor list (the neighbor array obtained from neighbours_mask[channel]) and identifies which neighbors of `channel` are considered parents of the child channel. These parent indices reference neighbors that are spatially closer (according to shells computed from geom) and are used by enforce_decrease_shells to propagate or check radial constraints.
        The returned structure is intended for read-only use by downstream routines; the function does not modify its inputs.
    
    Behavior, side effects, defaults, and failure modes:
        - The function calls make_shells(geom, n_jumps=n_jumps_parent) to precompute per-channel shells used as parent candidates, and repeatedly calls make_shell(channel, geom, n_jumps=...) to grow the local neighborhood around a given channel.
        - For each channel, the immediate closest shell (computed with n_jumps_per_growth) is marked as already seen and is not processed for parent relationships; the function then iteratively expands to more distant shells until all neighbors listed in neighbours_mask for that channel have been considered or seen.
        - If, for a newly discovered child channel, the intersection between that child's precomputed shell and the already-seen channels is empty, that child is skipped (this is logged in code as a benign condition for unusual geometries). Such skipped children are not added to the channel's parent list.
        - The function returns a nested list structure and does not write any files or modify global state; it performs only CPU-bound numpy computations.
        - Defaults are n_jumps_per_growth=1 and n_jumps_parent=3, chosen to balance finer-grained growth with a moderate parent-shell radius; callers can adjust these to trade off iteration count versus radial granularity.
        - Failure modes: if geom and neighbours_mask are inconsistent (for example, neighbours_mask refers to channel indices that are never reached by expanding shells computed from geom), the iterative loop may take many iterations to find all neighbors or may fail to progress meaningfully; in pathological or malformed geometries the function may effectively stall or perform many iterations because there is no explicit iteration cap or timeout. The function also assumes neighbours_mask rows can be converted to neighbor indices by numpy.flatnonzero; passing non-boolean, non-indexable structures will raise numpy errors.
    """
    from spikeinterface.postprocessing.localization_tools import make_radial_order_parents
    return make_radial_order_parents(
        geom,
        neighbours_mask,
        n_jumps_per_growth,
        n_jumps_parent
    )


################################################################################
# Source: spikeinterface.postprocessing.localization_tools.make_shell
# File: spikeinterface/postprocessing/localization_tools.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_postprocessing_localization_tools_make_shell(
    channel: int,
    geom: numpy.ndarray,
    n_jumps: int = 1
):
    """make_shell computes the set of channel indices forming a single "shell" of neighbors around a reference channel in electrode geometry. It is a helper used by spikeinterface.postprocessing.localization_tools.make_shells and by other localization/post-processing utilities in SpikeInterface to select neighbouring channels for waveform localization, template extraction, and downstream quality metrics.
    
    This function measures Euclidean distances between the reference channel and all channels using scipy.spatial.distance.cdist, selects the nth unique nonzero distance (controlled by n_jumps) as the shell radius, and returns the indices of all channels whose distance is less than or equal to that radius (within a floating-point tolerance). The returned indices exclude the reference channel itself.
    
    Args:
        channel (int): Index of the reference channel in geom for which the shell is computed. This must be a valid integer index into geom (0 <= channel < geom.shape[0]). If channel is out of bounds, an IndexError will be raised by array indexing.
        geom (numpy.ndarray): 2-D array of channel coordinates with shape (n_channels, n_dimensions). Each row geom[i] is the coordinate vector (e.g., x,y or x,y,z) of channel i. The function uses these coordinates to compute pairwise Euclidean distances from the reference channel.
        n_jumps (int): Number of unique distance "jumps" to include when defining the shell radius. The function identifies the sorted unique distances from the reference channel to other channels, ignores the zero distance to itself, and picks the n_jumps-th unique distance as the radius. n_jumps must be a positive integer (default 1) to select the first nonzero unique distance. If n_jumps is zero, negative, or larger than the number of available unique nonzero distances, the function will raise an IndexError due to attempting to index a non-existent unique distance.
    
    Returns:
        numpy.ndarray: One-dimensional array of integer channel indices (dtype int) that are within the computed shell radius from the reference channel, excluding the reference channel. The indices are returned in ascending order and are unique. No in-place modification of geom or other inputs occurs; the function has no side effects. Floating-point tolerance of 1e-8 is applied when comparing distances to the radius to account for numerical precision.
    
    Notes:
        - The distance computation uses scipy.spatial.distance.cdist and Euclidean distance behavior; ensure scipy is available in the environment.
        - The function assumes geom is densely structured as a numeric numpy.ndarray; passing non-array types may still work if they are array-like, but types incompatible with numpy operations will raise TypeError.
        - If multiple channels share identical coordinates or distances, those ties are handled by numpy.unique and the selection of the n_jumps-th unique distance. If there are insufficient unique distances to satisfy n_jumps, an IndexError will be raised.
        - Typical practical use in SpikeInterface: select neighbouring channels around a detection site to compute localized waveforms or to build shells for spatial aggregation in post-processing and localization algorithms.
    """
    from spikeinterface.postprocessing.localization_tools import make_shell
    return make_shell(channel, geom, n_jumps)


################################################################################
# Source: spikeinterface.postprocessing.localization_tools.make_shells
# File: spikeinterface/postprocessing/localization_tools.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_postprocessing_localization_tools_make_shells(
    geom: numpy.ndarray,
    n_jumps: int = 1
):
    """Get neighbor-channel "shells" for each electrode channel based on inter-channel distances.
    
    This function is used in spike sorting post-processing (spikeinterface.postprocessing.localization_tools) to determine, for every recording channel, which other channels lie within successive nearest-neighbor distance shells. In practical spike-sorting workflows this is useful to restrict computations (for example waveform extraction, localization, template updates, or quality metric calculations) to channels that are spatially close to a given channel. The radius that defines each shell is computed from the empirical distances between channels: first the distance to the closest other channel, then the next distinct larger distance, and so on, up to n_jumps distinct distance levels.
    
    Args:
      geom (numpy.ndarray): Array of channel coordinates. Expected to have shape (n_channels, n_dimensions) where each row gives the spatial coordinate of one electrode channel. The function uses these coordinates to compute pairwise Euclidean distances between channels and to determine the successive nearest-neighbor distance thresholds. The channel ordering (row index) defines the channel indices returned in the output; indices are zero-based and correspond to the rows of geom.
      n_jumps (int): Number of successive nearest-neighbor distance levels (shells) to include for each channel. A value of 1 includes only the channels at the minimum nonzero distance from the reference channel (the immediate neighbors). A value of 2 includes those and also the channels at the next larger distinct distance, and so on. Must be a positive integer (>= 1). If n_jumps exceeds the number of distinct nonzero inter-channel distance levels, the returned shells for a channel will include all other channels (i.e., the shell grows until no further distinct distances remain).
    
    Behavior and edge cases:
      The function computes pairwise Euclidean distances between channel coordinates in geom, identifies distinct increasing distance values from a reference channel, and collects indices of channels whose distances fall within the first n_jumps distinct distance levels. A channel is never included in its own shell (the reference channel index is excluded). If multiple channels lie at the same distance level, they are all included in that shell. If geom contains duplicate coordinates (zero distance between distinct channels), the corresponding channels will be treated as belonging to the same (possibly the first) distance level; this can lead to shells that include multiple physically co-located channels. There are no side effects: geom is not modified. Computation cost scales approximately with O(n_channels^2) due to pairwise distance calculations, so for very large channel counts this function can be computationally intensive.
    
    Returns:
      list: A list of length geom.shape[0] (the number of channels). The ith entry in the list is an array with the indices of the neighbors of the ith channel according to the first n_jumps distance shells. The reference index i is not included in its own array. These arrays contain integer channel indices corresponding to rows of geom.
    """
    from spikeinterface.postprocessing.localization_tools import make_shells
    return make_shells(geom, n_jumps)


################################################################################
# Source: spikeinterface.postprocessing.template_metrics.fit_velocity
# File: spikeinterface/postprocessing/template_metrics.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_postprocessing_template_metrics_fit_velocity(
    peak_times: numpy.ndarray,
    channel_dist: numpy.ndarray
):
    """Fit velocity from peak times and channel distances using a robust Theil–Sen linear estimator.
    
    This function is used in SpikeInterface post-processing (template_metrics) to estimate the propagation velocity of a spike waveform across recording channels. Given a set of peak times (for example, times of maximum amplitude of a template on each channel) and the corresponding channel distances from a chosen reference channel, the function fits a linear model of the form channel_dist = slope * peak_times + intercept using sklearn.linear_model.TheilSenRegressor. The fitted slope is interpreted as the propagation velocity (distance per unit time) when peak_times and channel_dist are provided in consistent physical units (for example, seconds and micrometers). The method is robust to outliers compared to ordinary least squares and returns a goodness-of-fit score (coefficient of determination, R^2) from the estimator.
    
    Args:
        peak_times (numpy.ndarray): 1-D array of peak times corresponding to each channel or observation. Each element represents the time of the peak (for example, sample index or seconds) for a specific channel or spatial location. peak_times is treated as the independent variable in the linear fit. It must be a 1-D numpy.ndarray with the same length as channel_dist and use a time unit consistent with the unit of channel_dist to allow meaningful slope interpretation.
        channel_dist (numpy.ndarray): 1-D array of channel distances corresponding to each entry in peak_times. Each element represents the distance of the channel from a reference location (for example, micrometers or electrode index). channel_dist is treated as the dependent variable in the linear fit. It must be a 1-D numpy.ndarray with the same length as peak_times and use a distance unit consistent with peak_times.
    
    Returns:
        tuple:
            slope (float): Estimated linear slope from Theil–Sen regression. Interpreted as propagation velocity in units of channel_dist per unit of peak_times (for example, micrometers per second) when input arrays use consistent physical units. This value is the model coefficient theil.coef_[0] returned by sklearn.
            intercept (float): Estimated intercept of the linear model (theil.intercept_), representing the predicted channel distance at time zero in the same distance units as channel_dist.
            score (float): Coefficient of determination R^2 computed by the Theil–Sen estimator (theil.score). This provides a measure of goodness-of-fit where values closer to 1 indicate a better linear fit under the estimator's criteria.
    
    Behavior, side effects, defaults, and failure modes:
        The function reshapes peak_times to a 2-D column vector internally (peak_times.reshape(-1, 1)) because sklearn estimators expect a 2-D feature matrix. No global state is modified; there are no side effects beyond returning the fitted parameters and score. The function relies on sklearn.linear_model.TheilSenRegressor and therefore requires scikit-learn to be available in the runtime environment. Inputs must be finite numeric arrays; supplying arrays containing NaN or infinite values may cause the underlying estimator to raise an error. The function will also raise an exception if peak_times and channel_dist have mismatched lengths or incompatible shapes. For very small sample sizes (for example, fewer than two valid observations) the estimator may fail or return unreliable parameters. Ensure that peak_times and channel_dist use consistent physical units if the numeric slope is to be interpreted physically as a velocity.
    """
    from spikeinterface.postprocessing.template_metrics import fit_velocity
    return fit_velocity(peak_times, channel_dist)


################################################################################
# Source: spikeinterface.postprocessing.template_metrics.get_exp_decay
# File: spikeinterface/postprocessing/template_metrics.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_postprocessing_template_metrics_get_exp_decay(
    template: numpy.ndarray,
    channel_locations: numpy.ndarray,
    sampling_frequency: float = None,
    **kwargs
):
    """Compute the exponential spatial decay constant of a template waveform's channel peak
    amplitudes used in SpikeInterface post-processing template metrics (units um/s).
    This function is intended for use in the postprocessing stage of SpikeInterface to
    quantify how a unit's template amplitude falls off with distance across recording
    channels. The returned decay constant is useful as a compact quality metric for
    assessing the spatial spread/localization of a putative neuron (e.g., for curation
    or comparison of sorters).
    
    Args:
        template (numpy.ndarray): The template waveform array with shape (num_samples, num_channels).
            This is the per-unit template produced by waveform extraction or a TemplateExtractor
            within SpikeInterface. The function computes a per-channel peak amplitude by
            applying the selected peak function along axis=0 (across samples) and taking the
            absolute value; these per-channel amplitudes are the observations used to fit the
            exponential decay.
        channel_locations (numpy.ndarray): Array of channel coordinates with shape (num_channels, 2).
            Each row corresponds to the (x, y) location of a channel (unit must match the
            template's channel order). Euclidean distances are computed between channels and the
            channel with maximal amplitude to produce the independent variable for the fit.
            The coordinate units should be consistent with the intended output units (the
            implementation documents decay in units "um/s").
        sampling_frequency (float): Optional sampling frequency of the template in Hz. Default is None.
            This parameter is accepted for API compatibility with other template-metric functions
            in SpikeInterface but is not used in the current exponential decay computation.
            Providing it does not change the computation performed by this function.
        kwargs: Dictionary of required keyword arguments that control fitting behavior and acceptance.
            The function asserts that the following keys are present and will raise AssertionError
            if they are missing.
            - exp_peak_function: str, required. Specifies how to compute the per-channel peak amplitude.
              Acceptable values are "ptp" to use numpy.ptp (peak-to-peak across samples) or "min"
              to use numpy.min across samples. The chosen function is applied along axis=0 and then
              absolute values are taken to form the amplitude vector used for fitting.
            - min_r2_exp_decay: float, required. The minimum R^2 value that the exponential fit must
              achieve to be considered valid. If the computed R^2 (using sklearn.metrics.r2_score)
              is below this threshold, the function returns numpy.nan to indicate an unreliable fit.
    
    Behavior, defaults, side effects, and failure modes:
        The function computes per-channel amplitudes using the chosen exp_peak_function, finds
        the channel with maximum amplitude and computes Euclidean distances from that channel to
        all channels. Distances and amplitudes are cast to numpy.longdouble for fitting stability
        where available. The function fits the model amp0 * exp(-decay * x) + offset using
        scipy.optimize.curve_fit with the following behavior: initial parameter guess p0=[1e-3, amp0, offset0]
        and bounds bounds=([1e-5, amp0 - 0.5*amp0, 0], [2, amp0 + 0.5*amp0, 2*offset0]). The fitted
        decay parameter is interpreted as the exponential decay constant describing how amplitude
        decreases with distance (reported in the documented units "um/s"). After fitting, the
        coefficient of determination R^2 between observed amplitudes and the model prediction is
        computed; if R^2 is less than min_r2_exp_decay the fit is considered invalid and numpy.nan
        is returned.
        The function asserts that the kwargs keys "exp_peak_function" and "min_r2_exp_decay" exist
        and will raise AssertionError if they are absent. Any exceptions raised during the numeric
        fitting (for example, from scipy or due to degenerate data) are caught internally; in such
        cases the function returns numpy.nan rather than propagating the exception. No external
        state is modified by this function. The function uses scipy.optimize.curve_fit and
        sklearn.metrics.r2_score; these packages must be available in the environment.
    
    Returns:
        float: The fitted exponential decay constant (decay parameter) as a Python float (popt[0]).
        If the fit fails for any reason or if the fit's R^2 is below min_r2_exp_decay, the function
        returns numpy.nan to signal an invalid or unreliable exponential-decay estimate.
    """
    from spikeinterface.postprocessing.template_metrics import get_exp_decay
    return get_exp_decay(template, channel_locations, sampling_frequency, **kwargs)


################################################################################
# Source: spikeinterface.postprocessing.template_metrics.get_half_width
# File: spikeinterface/postprocessing/template_metrics.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_postprocessing_template_metrics_get_half_width(
    template_single: numpy.ndarray,
    sampling_frequency: float,
    trough_idx: int = None,
    peak_idx: int = None,
    **kwargs
):
    """Return the half width of a single-unit template waveform in seconds.
    
    This function is part of SpikeInterface's postprocessing template metrics used in spike sorting quality assessment and curation. Given a 1D template waveform for a single unit (for example, the average or median spike waveform on a channel), get_half_width computes the temporal width at half the trough amplitude (half-width) measured in seconds. The half-width is a commonly used spike metric that captures spike duration and helps distinguish cell types and assess sorting quality. If trough_idx or peak_idx are not provided, they are inferred by calling get_trough_and_peak_idx(template_single). The function assumes a baseline of 0 when computing the half-amplitude threshold (threshold = 0.5 * trough_value). No in-place modifications are made to template_single.
    
    Args:
        template_single (numpy.ndarray): A 1D numpy array containing the template waveform for a single unit. This is the waveform used by SpikeInterface postprocessing to compute metrics; it must be indexed with standard zero-based indexing. The function expects a 1D array representing the waveform samples in temporal order.
        sampling_frequency (float): The sampling frequency of the template in Hz (samples per second). This value is used to convert a difference in sample indices into seconds by dividing the sample-count difference by sampling_frequency. Must be a finite float; a non-positive or zero sampling_frequency will make the result invalid (division by zero) and should be avoided by the caller.
        trough_idx (int): The integer index (zero-based) of the trough (minimum deflection) in template_single. If None (the default), the function computes trough_idx (and peak_idx) by calling get_trough_and_peak_idx(template_single). Supplying a valid trough_idx avoids that automatic detection step.
        peak_idx (int): The integer index (zero-based) of the peak in template_single. If None (the default), the function computes peak_idx (and trough_idx) by calling get_trough_and_peak_idx(template_single). The function treats peak_idx == 0 as an invalid case and returns numpy.nan immediately.
        kwargs (dict): Additional keyword arguments. Present for backward compatibility with other code paths in SpikeInterface; this function does not use these arguments and they are ignored. Callers may pass an empty dict or arbitrary keyword args without side effects, but relying on any behavior from these arguments is unsupported.
    
    Returns:
        hw (float): The half width in seconds. This is computed by finding sample indices on the waveform flanks around the trough where the waveform crosses half the trough amplitude (threshold = 0.5 * trough_value, assuming baseline 0), taking the difference between the post-trough and pre-trough crossing indices, and dividing by sampling_frequency. If the function cannot determine valid crossing points (for example, if no samples before or after the trough fall below the threshold, or if peak_idx == 0), it returns numpy.nan to indicate the half width is undefined. Callers should handle numpy.nan results when aggregating metrics.
    
    Behavior, defaults, and failure modes:
        If trough_idx or peak_idx are None, get_trough_and_peak_idx(template_single) is invoked; errors or unexpected outputs from that helper will propagate. The function searches for indices where template_single is below the half-amplitude threshold on the left and right flanks surrounding the trough; if either side has no such crossing, the function sets hw to numpy.nan. The function does not modify template_single. The caller must ensure sampling_frequency is a valid positive float; otherwise the returned value will be invalid or numpy.nan. This function is intended for use in spike sorting postprocessing and quality-metric pipelines within SpikeInterface; it returns a single scalar metric per template that can be aggregated across units or used for visualization and curation.
    """
    from spikeinterface.postprocessing.template_metrics import get_half_width
    return get_half_width(
        template_single,
        sampling_frequency,
        trough_idx,
        peak_idx,
        **kwargs
    )


################################################################################
# Source: spikeinterface.postprocessing.template_metrics.get_num_negative_peaks
# File: spikeinterface/postprocessing/template_metrics.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_postprocessing_template_metrics_get_num_negative_peaks(
    template_single: numpy.ndarray,
    sampling_frequency: float,
    **kwargs
):
    """Count the number of negative peaks in a single template waveform used for spike-sorting
    postprocessing. This function is part of the spikeinterface.postprocessing.template_metrics
    module and is used to quantify how many distinct negative deflections (negative-going
    peaks) appear in a neuron's average spike waveform (template). This metric is useful in
    spike sorting and template quality assessment to characterize waveform shape (for example,
    to detect multi-phasic waveforms or split templates).
    
    Args:
        template_single (numpy.ndarray): A 1D NumPy array containing the template waveform
            (average spike waveform) for a single unit/channel. The function treats this array
            as a time series of voltage (or amplitude) samples. The role of this argument is to
            supply the waveform in which negative peaks will be detected. The function takes
            the absolute maximum of this array to compute a relative detection threshold.
            Practical significance: each entry corresponds to a sample point at a known sampling
            frequency and the array should represent a single waveform (not a stack of waveforms).
        sampling_frequency (float): The sampling frequency (in Hz) at which template_single
            was sampled. This value is used to convert a peak width given in milliseconds
            (see kwargs) into samples for peak detection. The sampling frequency must reflect
            the same timebase as template_single (e.g., 30000.0 for 30 kHz recordings). A
            non-positive or incorrect sampling_frequency will yield an incorrect conversion
            from milliseconds to samples and thus incorrect peak-width handling.
        kwargs (dict): Additional required keyword arguments controlling peak detection.
            This function requires the following keys to be present; missing keys raise an
            AssertionError:
            - peak_relative_threshold: a float giving the relative height threshold used to
              detect peaks. It is interpreted as a fraction of the maximum absolute amplitude of
              template_single (max(abs(template_single))). Practically, only peaks whose
              absolute height is >= peak_relative_threshold * max_abs_value will be considered.
            - peak_width_ms: a float giving the minimum peak width in milliseconds. The value
              is converted to samples by int(peak_width_ms / 1000 * sampling_frequency) and
              that integer number of samples is passed to the peak detection routine as the
              minimum width. Practically, this controls how broad a trough must be to count as
              a peak and helps ignore very narrow noise transients.
    
    Behavior:
        The function computes max_value = max(abs(template_single)) to establish a scale for
        relative thresholding. It converts peak_width_ms into an integer sample count using the
        provided sampling_frequency. It then uses scipy.signal.find_peaks on the negated
        waveform (-template_single) with the computed height threshold and width to detect
        negative-going peaks. The function returns the number of detected negative peaks,
        implemented as the length of the peak indices array returned by find_peaks.
    
    Side effects and defaults:
        - The function does not modify template_single or any external state.
        - It relies on scipy.signal.find_peaks for detection behavior (height and width logic).
        - peak_width_ms is converted with int(...); small values can yield zero, which affects
          the width constraint passed to find_peaks (behavior then follows scipy's handling of
          width=0).
        - The function will raise AssertionError if the required keys in kwargs are missing.
        - The function may raise standard NumPy or SciPy errors if template_single contains
          non-numeric values (NaN/inf) or has incompatible shape; input is expected to be a
          one-dimensional numeric array.
    
    Failure modes:
        - Missing required kwargs results in AssertionError with explanatory messages.
        - An incorrect sampling_frequency (e.g., zero or negative) produces an incorrect
          width conversion and thus incorrect peak detection; such misuse can lead to
          undercounting or overcounting peaks.
        - Passing a multidimensional array or an empty array may produce NumPy or SciPy errors
          (e.g., when taking the maximum or running find_peaks). The caller should ensure
          template_single is a non-empty 1D numeric array.
    
    Returns:
        num_negative_peaks (int): The number of detected negative peaks in template_single.
            This integer is computed as the number of peak indices returned by scipy.signal.find_peaks
            when applied to -template_single with a height threshold of peak_relative_threshold * max_abs_value
            and a minimum width of peak_width_ms converted to samples. Practical significance:
            this count summarizes how many distinct negative deflections the average spike
            waveform contains and can be used as a feature for template characterization,
            quality metrics, or downstream curation in spike sorting pipelines.
    """
    from spikeinterface.postprocessing.template_metrics import get_num_negative_peaks
    return get_num_negative_peaks(template_single, sampling_frequency, **kwargs)


################################################################################
# Source: spikeinterface.postprocessing.template_metrics.get_num_positive_peaks
# File: spikeinterface/postprocessing/template_metrics.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_postprocessing_template_metrics_get_num_positive_peaks(
    template_single: numpy.ndarray,
    sampling_frequency: float,
    **kwargs
):
    """Count the number of positive peaks in a single-unit template waveform.
    
    This function is part of SpikeInterface post-processing template metrics and is used to quantify a template's morphology for spike sorting validation and curation. It inspects a 1D template waveform for a single unit (template_single) and returns how many positive (local maximum) peaks exceed a threshold relative to the waveform's maximum absolute amplitude. The function uses scipy.signal.find_peaks with a height threshold computed as peak_relative_threshold * max(abs(template_single)) and a width constraint derived from peak_width_ms converted to samples using sampling_frequency. This metric can help characterize multi-peaked templates which may influence downstream quality metrics and manual curation decisions.
    
    Args:
        template_single (numpy.ndarray): The 1D template waveform for a single unit. This array is interpreted as the discrete-time waveform (samples along time) produced by waveform extraction in SpikeInterface. The function computes max(abs(template_single)) and searches for local maxima in this array; providing a non-1D or non-numeric array may raise numpy or scipy exceptions.
        sampling_frequency (float): The sampling frequency (in Hz) used to convert a width expressed in milliseconds into a width in samples. It is used in the conversion peak_width_samples = int(peak_width_ms / 1000 * sampling_frequency). Supplying an incorrect sampling frequency will change the width constraint applied when detecting peaks and therefore change the reported count.
        kwargs (dict): Additional required keyword arguments that control peak detection. Required keys:
            - peak_relative_threshold: a scalar multiplier applied to max(abs(template_single)) to compute the minimum peak height passed to scipy.signal.find_peaks (i.e., height = peak_relative_threshold * max_value). This parameter determines how large a positive deflection must be, relative to the template's overall amplitude, to be counted as a peak.
            - peak_width_ms: the width threshold specified in milliseconds. This value is converted to an integer number of samples via int(peak_width_ms / 1000 * sampling_frequency) and passed as the width argument to scipy.signal.find_peaks. The width parameter constrains detected peaks to have at least this width (in samples) as interpreted by scipy.signal.find_peaks.
    
    Behavior, side effects, defaults, and failure modes:
        The function computes max_value = numpy.max(numpy.abs(template_single)) and uses scipy.signal.find_peaks(template_single, height=peak_relative_threshold * max_value, width=peak_width_samples) to detect positive peaks. The width passed to scipy is the integer conversion described above. There are no external side effects; the function returns an integer count only.
        If either required kwargs 'peak_relative_threshold' or 'peak_width_ms' is missing, the function raises an AssertionError (the code asserts their presence). If template_single contains non-finite values or invalid types, numpy or scipy functions may raise exceptions. If max_value equals zero (e.g., a flat zero waveform), the computed height threshold will be zero and the detection behavior will depend on scipy.signal.find_peaks semantics; results in that case may be uninterpretable for waveform quality assessment. If sampling_frequency is zero or negative, the width conversion may yield zero or negative integers which can change peak detection behavior or trigger exceptions from scipy.
    
    Returns:
        number_positive_peaks (int): The number of detected positive peaks (local maxima) that satisfy the configured relative height and width constraints. This returned integer is suitable for use in downstream template metrics, unit quality summaries, or curation rules within the SpikeInterface postprocessing pipeline.
    """
    from spikeinterface.postprocessing.template_metrics import get_num_positive_peaks
    return get_num_positive_peaks(template_single, sampling_frequency, **kwargs)


################################################################################
# Source: spikeinterface.postprocessing.template_metrics.get_peak_to_valley
# File: spikeinterface/postprocessing/template_metrics.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_postprocessing_template_metrics_get_peak_to_valley(
    template_single: numpy.ndarray,
    sampling_frequency: float,
    trough_idx: int = None,
    peak_idx: int = None,
    **kwargs
):
    """spikeinterface.postprocessing.template_metrics.get_peak_to_valley returns the peak-to-valley duration of a single template waveform in seconds, a common spike-sorting quality metric used in SpikeInterface post-processing to characterize waveform shape and temporal relationships between trough and peak.
    
    This function computes the time difference between a waveform peak and trough expressed in seconds. In the SpikeInterface context, this measurement is used in post-processing and quality metrics to help assess unit isolation and waveform morphology. If trough and peak sample indices are not provided, the function will call get_trough_and_peak_idx(template_single) to determine them automatically from the 1D template waveform. The computed value is (peak_idx - trough_idx) / sampling_frequency, so the sign indicates whether the peak occurs after (positive) or before (negative) the trough.
    
    Args:
        template_single (numpy.ndarray): The 1D template waveform for a single unit from which the peak and trough are identified. This should be a NumPy array containing sampled voltage values (time series) corresponding to one averaged spike waveform produced during spike-sorting post-processing. The function expects a one-dimensional array; if a different shape is provided, NumPy operations or the helper get_trough_and_peak_idx may raise an exception.
        sampling_frequency (float): The sampling frequency (in samples per second, Hz) used to convert a sample index difference into seconds. This value must be the same sampling rate that produced template_single; a non-positive or incorrect sampling_frequency will yield meaningless durations or a division error.
        trough_idx (int): The sample index (zero-based integer) of the trough (local minimum) in template_single. If provided, this index is used directly without re-detection. It must refer to the same indexing convention as template_single and be an integer; out-of-bounds or non-integer values will lead to errors or incorrect results.
        peak_idx (int): The sample index (zero-based integer) of the peak (local maximum) in template_single. If provided, this index is used directly without re-detection. It must refer to the same indexing convention as template_single and be an integer; out-of-bounds or non-integer values will lead to errors or incorrect results.
        kwargs (dict): Additional keyword arguments accepted for API compatibility with other template-metric functions in SpikeInterface. These are not used by this function and are ignored; they are present to allow callers that pass extra parameters to shared utility routines without raising an unexpected-argument error.
    
    Returns:
        float: The peak-to-valley duration in seconds computed as (peak_idx - trough_idx) / sampling_frequency. This scalar float represents the temporal offset from trough to peak for the provided template. A positive value means the peak occurs after the trough; a negative value means the peak occurs before the trough. Exceptions raised by this function stem from invalid inputs (for example, non-1D template_single, indices out of range, or invalid sampling_frequency) or from the helper get_trough_and_peak_idx when trough_idx and/or peak_idx are not provided.
    """
    from spikeinterface.postprocessing.template_metrics import get_peak_to_valley
    return get_peak_to_valley(
        template_single,
        sampling_frequency,
        trough_idx,
        peak_idx,
        **kwargs
    )


################################################################################
# Source: spikeinterface.postprocessing.template_metrics.get_peak_trough_ratio
# File: spikeinterface/postprocessing/template_metrics.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_postprocessing_template_metrics_get_peak_trough_ratio(
    template_single: numpy.ndarray,
    sampling_frequency: float = None,
    trough_idx: int = None,
    peak_idx: int = None,
    **kwargs
):
    """Compute the peak-to-trough ratio of a single template waveform used in SpikeInterface post-processing metrics.
    
    This function is used in the SpikeInterface postprocessing.template_metrics module to quantify waveform asymmetry for a single 1D template waveform (for example, an average extracellular spike waveform computed by the framework). The ratio is computed as the sample value at peak_idx divided by the sample value at trough_idx and is returned as a unitless float that can be used as one feature in spike sorting quality assessment and curation workflows.
    
    Args:
        template_single (numpy.ndarray): The 1D template waveform array containing sampled waveform values (one-dimensional sequence of floats). This is the template for a single unit produced by SpikeInterface waveform extraction or template computation and must be indexable with Python integer indices. The function expects a single-channel (1D) template; passing a multi-dimensional array is not supported and may cause indexing errors.
        sampling_frequency (float): The sampling frequency (in Hz) associated with template_single. This parameter documents the temporal sampling context of the waveform for users of SpikeInterface metrics; in the current implementation it is accepted for API compatibility with other metrics but is not used in the computation. Provide the acquisition sampling rate when available for clarity and future compatibility.
        trough_idx (int): The integer index (Python 0-based) of the trough sample within template_single to use as the denominator in the ratio. If set to None, the function will attempt to infer the trough index by calling the internal helper get_trough_and_peak_idx(template_single). The provided index must be a valid index into template_single; if it is out of bounds an IndexError may be raised. If the sample at trough_idx is zero, a ZeroDivisionError will occur.
        peak_idx (int): The integer index (Python 0-based) of the peak sample within template_single to use as the numerator in the ratio. If set to None, the function will attempt to infer the peak index by calling the internal helper get_trough_and_peak_idx(template_single). The provided index must be a valid index into template_single; if it is out of bounds an IndexError may be raised.
        kwargs (dict): Additional keyword arguments accepted for forward compatibility with the SpikeInterface metrics API. These extra arguments are ignored by the current implementation and have no effect on the computation; they exist so this function can be called with uniform argument sets across the metrics suite.
    
    Returns:
        float: The peak-to-trough ratio computed as template_single[peak_idx] / template_single[trough_idx]. This is a unitless scalar feature describing the relative amplitude of the waveform peak compared to its trough and is intended for use as a quality metric in spike sorting post-processing. The function has no side effects (it does not modify template_single). Possible failure modes include IndexError for invalid indices, ZeroDivisionError if the trough sample equals zero, and propagation of NaN/Inf values if present in template_single.
    """
    from spikeinterface.postprocessing.template_metrics import get_peak_trough_ratio
    return get_peak_trough_ratio(
        template_single,
        sampling_frequency,
        trough_idx,
        peak_idx,
        **kwargs
    )


################################################################################
# Source: spikeinterface.postprocessing.template_metrics.get_recovery_slope
# File: spikeinterface/postprocessing/template_metrics.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_postprocessing_template_metrics_get_recovery_slope(
    template_single: numpy.ndarray,
    sampling_frequency: float,
    peak_idx: int = None,
    **kwargs
):
    """Return the recovery slope of a single template waveform after its peak, measured as change in amplitude per unit time (unit of template / second). In the SpikeInterface framework for spike sorting and post-processing, this function quantifies the post-spike repolarization/hyperpolarization trend (the slope of the waveform after the action potential peak) within a user-defined time window. This metric is useful for quality metrics and waveform characterization in spike sorting analyses. By default, templates may be scaled to microvolts (uV) when produced by SortingAnalyzer (controlled by sorting_analyzer.return_in_uV); in that case the returned slope will be in uV/s.
    
    The function computes the slope by performing a linear regression (scipy.stats.linregress) of the template amplitude versus time over the samples from the peak index up to a time corresponding to recovery_window_ms after the peak. Time is derived from sample indices divided by sampling_frequency (seconds). If peak_idx is not provided, the function calls get_trough_and_peak_idx(template_single) to determine the peak index. The window endpoint is clipped to the template length. The function asserts that the required kwarg recovery_window_ms is present and will raise an AssertionError if it is not. If the peak is at index 0 the function returns numpy.nan. Note that sampling_frequency is used as a divisor to compute time; providing sampling_frequency equal to zero will lead to a division error.
    
    Args:
        template_single (numpy.ndarray): The 1D template waveform array for a single unit. Each element is an amplitude sample in the same physical units as produced by the upstream processing pipeline (e.g., volts or microvolts). The function treats this array as a time series sampled at sampling_frequency and uses its length to clip the regression window.
        sampling_frequency (float): The sampling frequency (samples per second) of the template. This value is used to convert sample indices to time in seconds (time = index / sampling_frequency). Must be a non-zero numeric sampling rate; a zero sampling_frequency will cause a division error.
        peak_idx (int): The integer index in template_single that marks the peak of the action potential. If None, the function will compute the peak index by calling get_trough_and_peak_idx(template_single) and use the returned peak index. If the resolved peak index equals 0 the function returns numpy.nan because no post-peak samples are available for regression.
        kwargs (dict): Additional keyword arguments. Required key:
            recovery_window_ms: the length of the post-peak window, in milliseconds, over which to compute the recovery slope. This numeric value is converted to seconds internally and used together with sampling_frequency to determine the number of samples included in the linear regression. This kwarg is mandatory; the function asserts its presence and will raise an AssertionError if it is missing. The computed window is clipped at the end of template_single so the actual number of samples used may be smaller than requested.
    
    Returns:
        float: The slope of the template after the peak in units of (unit of template) per second (for example, uV/s if templates are scaled to microvolts). The slope is the result.slope value returned by scipy.stats.linregress performed on the time vector (seconds) and the template amplitudes over the specified recovery window. Possible outcomes and failure modes include:
        - Returns numpy.nan if peak_idx equals 0.
        - If the regression window contains fewer than two samples (for example, if the requested recovery_window_ms is effectively zero or the peak is at or near the end of the template), scipy.stats.linregress may return nan or raise an error depending on SciPy behavior.
        - Raises AssertionError if the required kwarg recovery_window_ms is not provided.
        - A ZeroDivisionError or invalid time values will occur if sampling_frequency is zero.
    """
    from spikeinterface.postprocessing.template_metrics import get_recovery_slope
    return get_recovery_slope(template_single, sampling_frequency, peak_idx, **kwargs)


################################################################################
# Source: spikeinterface.postprocessing.template_metrics.get_repolarization_slope
# File: spikeinterface/postprocessing/template_metrics.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_postprocessing_template_metrics_get_repolarization_slope(
    template_single: numpy.ndarray,
    sampling_frequency: float,
    trough_idx: int = None,
    **kwargs
):
    """Return slope of the repolarization period of a single average spike template between its trough (maximum hyperpolarization) and the baseline.
    
    This function is used in SpikeInterface postprocessing (template_metrics) to quantify how quickly a neuron's membrane potential recovers after the trough of an action potential. The repolarization slope is computed as the linear regression slope (dV/dT) of the template waveform between the trough index and the first sample after the trough where the waveform returns to baseline (value >= 0). The returned value has units of (unit of template) per second; when templates are scaled to microvolts (uV) by SortingAnalyzer.return_in_uV, the result is in uV/s. The implementation derives time points from the sample indices using the provided sampling_frequency and uses scipy.stats.linregress to compute the slope.
    
    Args:
        template_single (numpy.ndarray): The 1D template waveform for a single unit, provided as a NumPy array. This array represents the average extracellular voltage trace of an action potential across time samples. The function expects a one-dimensional time series where indices correspond to successive samples; trough_idx (if provided) indexes into this array.
        sampling_frequency (float): The sampling frequency (samples per second) used to convert sample indices to time in seconds. It is used to build a times vector as numpy.arange(template_single.shape[0]) / sampling_frequency. If sampling_frequency is zero or invalid this will lead to division errors or invalid times.
        trough_idx (int or None): The integer index of the trough (the sample of maximum negative/most hyperpolarized deflection) within template_single. If None (the default), the function determines the trough index by calling get_trough_and_peak_idx(template_single) as done in the source code. If trough_idx is 0, the function cannot compute a repolarization interval and returns numpy.nan.
        kwargs (dict): Additional keyword arguments provided for API compatibility with other callers (for example SortingAnalyzer workflows). These keyword arguments are not used by this function and are ignored; they exist to allow callers to pass through extra options without raising errors.
    
    Returns:
        slope (float): The repolarization slope (dV/dT) computed by linear regression of the template between trough_idx and the first sample after the trough where template_single >= 0 (the return-to-baseline index). The slope is expressed in (unit of template) per second (for example uV/s when templates are in microvolts). If the function cannot compute a valid slope it returns numpy.nan. Specific failure modes that yield numpy.nan are: trough_idx equals 0 (no pre-trough samples to define interval), no sample after trough reaches baseline (no return-to-baseline detected), or the interval from trough to return-to-baseline contains fewer than three samples (insufficient points for reliable linear regression). The function performs no side effects and is deterministic given the same inputs; it uses scipy.stats.linregress internally to estimate the slope.
    """
    from spikeinterface.postprocessing.template_metrics import get_repolarization_slope
    return get_repolarization_slope(
        template_single,
        sampling_frequency,
        trough_idx,
        **kwargs
    )


################################################################################
# Source: spikeinterface.postprocessing.template_metrics.get_spread
# File: spikeinterface/postprocessing/template_metrics.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_postprocessing_template_metrics_get_spread(
    template: numpy.ndarray,
    channel_locations: numpy.ndarray,
    sampling_frequency: float,
    **kwargs
):
    """Compute the spatial spread (in micrometers) of a unit template's amplitude along a specified depth axis.
    
    This function is used in the SpikeInterface postprocessing pipeline to quantify how far, across probe depth, a template's amplitude is distributed. It is intended for use after spike sorting and template extraction to provide a simple, interpretable metric (spread in um) that helps validate and compare units (for example, to identify spatially compact versus distributed templates). The computation first optionally restricts channels to a lateral column_range, sorts channels by depth, computes per-channel peak-to-peak amplitude of the template, optionally smooths that amplitude profile in depth using a Gaussian kernel (spread_smooth_um), normalizes the profile to its maximum, selects channels whose normalized amplitude exceeds spread_threshold, and returns the peak-to-peak extent (ptp) of those channel depths.
    
    Args:
        template (numpy.ndarray): The template waveform array with shape (num_samples, num_channels). Each column corresponds to one recording channel's waveform for a sorted unit. The function computes per-channel amplitudes as the peak-to-peak (ptp) value across the time axis (axis 0). Practical significance: these per-channel amplitudes represent how strongly the unit is detected on each physical channel and are the basis for computing spatial spread.
        channel_locations (numpy.ndarray): The channel coordinates with shape (num_channels, 2). Each row is a (x, y) coordinate pair (units in micrometers, um) that locates the corresponding column of template in physical probe space. The function uses these coordinates to compute depths along the requested depth_direction and to filter/sort channels; mismatch between num_channels here and template.shape[1] will raise alignment errors when slicing or reordering.
        sampling_frequency (float): Sampling frequency of the template in Hz. This value is included to match the Template/Sorting data context and is used for plotting/debugging time axes in the original implementation; it is not used in the numeric spread computation itself. Provide the recording sampling rate from the preprocessing/sorting workflow so the function call context is consistent with other SpikeInterface utilities.
        kwargs (dict): Additional required keyword arguments (must be supplied). The function asserts the presence of all of these keys and will raise AssertionError with a clear message if any are missing:
            depth_direction (str): Direction/coordinate name to treat as depth when computing spread. Conventionally one of "x", "y", or "z" as used in probe geometry descriptions. Implementation detail: if depth_direction == "y" the function uses the second column (index 1) of channel_locations as depth; otherwise it uses the first column (index 0). Practical significance: selecting the correct axis ensures spread is measured along the probe depth rather than along lateral/column axes.
            spread_threshold (float): Normalized amplitude threshold (unitless, between 0 and 1) used to select channels that contribute to the spread. After computing per-channel peak-to-peak amplitudes and normalizing by the maximum amplitude, channels with normalized amplitude > spread_threshold are considered part of the unit footprint. Practical significance: higher thresholds yield more spatially compact footprints; lower thresholds include more distant channels.
            spread_smooth_um (float or None): Smoothing kernel size in micrometers used to smooth the per-channel amplitude profile along depth. If None or <= 0, no smoothing is applied. When > 0, a Gaussian filter is applied with sigma = spread_smooth_um / median_spacing_between_unique_depths. Practical significance: smoothing reduces the influence of single-channel noise and produces a more robust estimate of spatial extent. Note: applying smoothing requires scipy.ndimage; ImportError will be raised if scipy is unavailable.
            column_range (tuple or sequence): Range in micrometers in the x-direction used to restrict which channels (columns) are considered before computing spread. This value is passed to transform_column_range to select channels whose x-coordinate falls within the provided window. Practical significance: in multi-column probes or 2D layouts, limiting to a lateral column ensures spread is measured within a local column around the unit.
    
    Behavior, side effects, defaults, and failure modes:
        - The function first calls transform_column_range(template, channel_locations, column_range) to filter channels by the provided lateral (x) range. It then calls sort_template_and_locations(template, channel_locations, depth_direction) to reorder channels so depth is monotonic. These helper functions return new arrays; the original input arrays are not modified in place by this function (no persistent side effects).
        - Per-channel amplitude is computed as numpy.ptp(template, axis=0), i.e., peak-to-peak value across time samples. That amplitude profile is optionally smoothed (if spread_smooth_um > 0) using scipy.ndimage.gaussian_filter1d with sigma derived from physical spacing of channels.
        - The amplitude profile is normalized by dividing by its maximum (MM = MM / numpy.max(MM)). If the maximum amplitude is zero (e.g., an all-zero template) this division can produce invalid values and downstream comparisons will result in no channels exceeding the threshold; in practice this leads to a returned spread of 0.0.
        - Channels with normalized amplitude > spread_threshold are selected; the spread is computed as the peak-to-peak extent (numpy.ptp) of their depths (in the same units as channel_locations, typically micrometers). If no channels exceed the threshold (empty selection) numpy.ptp over an empty array yields 0.0 in the current NumPy behavior, and the function returns 0.0. If SciPy is requested for smoothing but not installed, an ImportError will be raised.
        - sampling_frequency is not used in the numerical spread computation; it appears in the original implementation only for optional plotting/debugging of the template time axis.
        - Input validation is limited: missing required kwargs cause AssertionError with explicit messages (e.g., "depth_direction must be given as kwarg"); incompatible array shapes (mismatched num_channels between template and channel_locations) or non-numeric entries will raise standard NumPy errors when indexing, sorting, or computing numeric operations.
        - The function relies on channel_locations being expressed in micrometers (um) for the returned spread to be interpretable in that unit. Ensure probe geometry coordinates are provided in um.
    
    Returns:
        float: The spatial spread of the template amplitude along the requested depth axis, expressed in the same length units as channel_locations (typically micrometers, um). The value is the peak-to-peak depth extent of channels whose normalized amplitude exceeds spread_threshold; a return value of 0.0 indicates either no channels passed the threshold, a zero-amplitude template, or that the selected column_range produced no channels to evaluate.
    """
    from spikeinterface.postprocessing.template_metrics import get_spread
    return get_spread(template, channel_locations, sampling_frequency, **kwargs)


################################################################################
# Source: spikeinterface.postprocessing.template_metrics.get_trough_and_peak_idx
# File: spikeinterface/postprocessing/template_metrics.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_postprocessing_template_metrics_get_trough_and_peak_idx(
    template: numpy.ndarray
):
    """spikeinterface.postprocessing.template_metrics.get_trough_and_peak_idx returns the indices into a 1D numpy template waveform corresponding to the detected trough (the minimum value) and the subsequent peak (the maximum value at or after the trough). This function is used in SpikeInterface postprocessing and template metrics to locate trough-to-peak features of averaged spike waveforms for tasks such as waveform alignment, amplitude measurement, and quality metric computation.
    
    The function expects a single-channel (1D) template waveform and assumes a negative trough followed by a positive peak. It does not modify the input array; it only reads values and computes two integer indices. The trough index is computed with numpy.argmin over the entire template. The peak index is computed relative to the trough as trough_idx + numpy.argmax(template[trough_idx:]), so the peak is searched for at or after the trough index. Because it uses numpy.argmin/argmax, tie-breaking follows numpy semantics (the first occurrence is returned). If the input is not 1-dimensional an AssertionError is raised by the internal assertion. If the template is empty, numpy.argmin/argmax will raise a ValueError. Presence of NaNs or other non-finite values will influence results according to numpy.argmin/argmax behavior.
    
    Args:
        template (numpy.ndarray): The 1D template waveform array representing an averaged extracellular spike waveform for a single unit or channel. In the SpikeInterface postprocessing context, this array is typically a time-series vector (shape (n_samples,)) containing the waveform amplitudes sampled over time. The function requires template.ndim == 1 and will assert otherwise. The function assumes the physiologically expected pattern of a negative-going trough followed by a positive-going peak; if this assumption is violated the returned indices may not correspond to a meaningful trough-to-peak pair.
    
    Returns:
        trough_idx (int): The integer index into template of the detected trough (minimum value). This index is a Python int (returned from numpy.argmin) and identifies the sample time of the waveform minimum used as the trough reference for downstream metrics (for example trough-to-peak amplitude or latency). If multiple minima are equal, the first occurrence is returned per numpy semantics.
        peak_idx (int): The integer index into template of the detected peak (maximum value at or after the trough). This is computed as trough_idx + numpy.argmax(template[trough_idx:]) and is a Python int. The peak index may equal trough_idx if the maximum in the tail slice occurs at the trough location (e.g., flat or monotonically increasing data). The index identifies the sample time of the waveform peak for use in peak-to-trough calculations and other waveform-based quality metrics.
    """
    from spikeinterface.postprocessing.template_metrics import get_trough_and_peak_idx
    return get_trough_and_peak_idx(template)


################################################################################
# Source: spikeinterface.postprocessing.template_metrics.get_velocity_above
# File: spikeinterface/postprocessing/template_metrics.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_postprocessing_template_metrics_get_velocity_above(
    template: numpy.ndarray,
    channel_locations: numpy.ndarray,
    sampling_frequency: float,
    **kwargs
):
    """get_velocity_above computes the propagation velocity of the template waveform above its maximum-amplitude channel, returning the velocity in micrometers per second (um/s) as used in spike sorting post-processing to quantify vertical spike propagation across channels.
    
    This function is used in the SpikeInterface post-processing pipeline to estimate how quickly the peak of a template travels away from the channel with the largest (most negative) deflection. The input template is a waveform extracted from extracellular recordings (num_samples, num_channels), and channel_locations gives the spatial coordinates of each channel (num_channels, 2). The function converts sample indices to times using sampling_frequency (Hz), selects channels located "above" the max channel according to depth_direction, fits a linear model relating peak time to distance, and returns the fitted slope as the velocity. The function will return numpy.nan when the velocity cannot be robustly estimated (insufficient channels or low fit quality).
    
    Args:
        template (numpy.ndarray): The template waveform array with shape (num_samples, num_channels). Each column is the waveform on one recording channel. In spike sorting, templates represent the average spike waveform for a unit across channels; this argument supplies the waveforms used to estimate propagation velocity.
        channel_locations (numpy.ndarray): Array of channel coordinates with shape (num_channels, 2). Each row corresponds to the (x, y) (or other 2D) location of the matching channel in the same order as template columns. These coordinates are used to compute Euclidean distances (in micrometers) from the channel with the template peak.
        sampling_frequency (float): Sampling frequency of the template in Hz. This is used to convert sample indices of waveform peaks into times in milliseconds (ms) via sample_index / sampling_frequency * 1000 before fitting the velocity.
        kwargs (dict): Additional required keyword arguments that control selection and fit behavior. The following keys are required and their meanings are:
            depth_direction (str): Axis name indicating which coordinate in channel_locations corresponds to depth for determining "above" vs "below". Expected values include "x", "y", or "z" as used in the codebase; the implementation maps this to an index with depth_dim = 1 if depth_direction == "y" else 0, so "y" selects the second column and other values select the first column of channel_locations. This choice determines which coordinate is compared to the max channel coordinate to select channels considered "above".
            min_channels_for_velocity (int): Minimum number of channels that must lie above the max-amplitude channel for a velocity estimate to be attempted. If the number of channels above is less than this threshold, the function returns numpy.nan. This prevents unstable fits when too few data points are available.
            min_r2_velocity (float): Minimum acceptable R^2 (fit score) for the linear fit between peak times and distances. If the fitted score is below this threshold, the function returns numpy.nan to indicate an unreliable velocity estimate.
            column_range (numeric or sequence): Range in micrometers in the x-direction used by transform_column_range to restrict channels considered for velocity. This argument is forwarded to transform_column_range and affects which channels and waveform columns are kept before sorting and fitting.
    
    Behavior and side effects:
        The function asserts the presence of the four required keys in kwargs and will raise AssertionError with a descriptive message if any are missing. Internally, it calls transform_column_range(template, channel_locations, column_range, depth_direction) to restrict channels by column_range and then sort_template_and_locations(template, channel_locations, depth_direction) to order channels by depth_direction; these calls return new template and channel_locations arrays and do not mutate external state beyond returning their results.
        The function locates the channel of maximum amplitude using numpy.argmin(template) (the code treats the template peak as the minimum value, consistent with negative-going extracellular spikes), converts that sample index to time in milliseconds using sampling_frequency, and computes distances (Euclidean norms in micrometers) from that max channel to all channels considered "above" (channels with channel_locations[:, depth_dim] >= max_channel_location[depth_dim]).
        It computes peak times of the selected channels in milliseconds relative to the max-channel peak and fits a linear relationship between peak time (ms) and distance (um) via fit_velocity, which returns (velocity, intercept, score). The returned velocity is the fitted slope reported by the function and is provided in units of micrometers per second (um/s) as stated in the module documentation.
        If the number of channels above is less than min_channels_for_velocity or if the fit score is less than min_r2_velocity, the function returns numpy.nan to signal that a robust velocity estimate could not be obtained.
        No plotting or I/O occurs in normal operation; commented DEBUG plotting code exists but is inactive.
    
    Failure modes:
        Missing required kwargs -> AssertionError is raised with a message indicating which key is missing.
        Insufficient channels above the max channel -> function returns numpy.nan (no exception).
        Poor linear fit quality (score < min_r2_velocity) -> function returns numpy.nan (no exception).
        If template and channel_locations shapes do not match (num_channels mismatch) or have incorrect dimensions, downstream indexing or linear algebra calls (e.g., numpy.unravel_index, numpy.linalg.norm) may raise IndexError or ValueError.
    
    Returns:
        float or numpy.nan: The estimated propagation velocity above the template's max-amplitude channel expressed in micrometers per second (um/s). If a robust estimate could not be obtained because there are too few channels above the max channel or the linear fit quality is below min_r2_velocity, the function returns numpy.nan to indicate an invalid/unreliable velocity.
    """
    from spikeinterface.postprocessing.template_metrics import get_velocity_above
    return get_velocity_above(template, channel_locations, sampling_frequency, **kwargs)


################################################################################
# Source: spikeinterface.postprocessing.template_metrics.get_velocity_below
# File: spikeinterface/postprocessing/template_metrics.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_postprocessing_template_metrics_get_velocity_below(
    template: numpy.ndarray,
    channel_locations: numpy.ndarray,
    sampling_frequency: float,
    **kwargs
):
    """Compute the propagation velocity of the spike waveform below the maximum-amplitude channel of a template. The function is used in spike sorting post-processing to estimate how quickly a spike peak travels across electrodes positioned in space (channel_locations). In the SpikeInterface workflow this metric helps characterize unit depth/propagation and can be used for quality metrics or anatomical interpretation. The original documentation states the velocity is reported in units um/s; note that the implementation computes peak times in milliseconds and distances in micrometers and fits a linear model, so the fitted slope returned by fit_velocity is derived from distances (um) versus times (ms) (see "Returns" for unit clarification and a note about a common units mismatch).
    
    Args:
        template (numpy.ndarray): The template waveform with shape (num_samples, num_channels). Each column is the waveform recorded on one channel. This array is used to find the channel with the maximal spike deflection (the code locates the global minimum of the template, since extracellular spikes are typically negative-going) and to compute per-channel peak times used for velocity fitting.
        channel_locations (numpy.ndarray): The channel coordinates with shape (num_channels, 2). Coordinates are in spatial units consistent with the code (interpreted as micrometers in downstream computations). These locations are used to compute Euclidean distances from the max channel to other channels that lie below it along the chosen depth direction.
        sampling_frequency (float): Sampling frequency of the template in Hz. Used to convert sample indices to time (the implementation converts to milliseconds by dividing sample index by sampling_frequency and multiplying by 1000) so that per-channel peak times are expressed in milliseconds for the velocity fit.
        kwargs (dict): Additional required keyword arguments. The function asserts that the following keys are present; if any are missing an AssertionError is raised with the messages used in the implementation.
            - depth_direction (str): The axis along which "above" and "below" are defined; allowed values in the implementation are "x", "y", or "z". The code maps depth_direction to a depth dimension index (depth_dim = 1 if "y" else 0) and then sorts/filter channels accordingly via sort_template_and_locations.
            - min_channels_for_velocity (int): Minimum number of channels located below the max channel that must be available to attempt a velocity fit. If fewer channels are present the function returns numpy.nan. This guards against unreliable fits when spatial sampling is insufficient.
            - min_r2_velocity (float): Minimum coefficient of determination (R^2) that the linear fit (fit_velocity) must achieve for the velocity to be accepted. If the fit score is below this threshold the function returns numpy.nan to indicate an unreliable estimate.
            - column_range (value accepted by transform_column_range): The spatial range (interpreted by transform_column_range) used to filter channels in the orthogonal column direction (documented as the range in micrometers in the x-direction in the original docstring). The function first calls transform_column_range(template, channel_locations, column_range) to restrict channels to this range, then sort_template_and_locations(template, channel_locations, depth_direction) to order channels by depth.
    
    Behavior and algorithmic details:
        1. The function enforces presence of the four required kwargs via assertions; missing keys raise AssertionError with explicit messages.
        2. transform_column_range is called to restrict the template and channel_locations to the requested column_range (this function is part of the postprocessing pipeline and filters channels in the orthogonal axis).
        3. sort_template_and_locations is called to sort the template columns and channel_locations according to the requested depth_direction (so "below" corresponds to indices with coordinate <= max channel coordinate along depth dimension).
        4. The channel with the maximal spike deflection is identified with numpy.argmin over the template (returns the sample and channel indices of the minimal value). The code treats that minimum as the spike peak.
        5. Peak times for channels below the max channel are computed as sample index of each channel's minimum divided by sampling_frequency, converted to milliseconds, and then referenced to the max channel peak time (so these are relative times in ms).
        6. Euclidean distances from each considered channel to the max channel are computed using the provided channel_locations and are interpreted as micrometers.
        7. fit_velocity is called with peak times (ms) and distances (um) and returns (velocity, intercept, score). The returned velocity is the slope of distance versus time used here to represent propagation speed.
        8. If the number of available channels below the max channel is less than min_channels_for_velocity, or if the fit score is below min_r2_velocity, the function returns numpy.nan to indicate that a reliable velocity estimate could not be produced.
    
    Failure modes and side effects:
        - AssertionError is raised if any required key is missing from kwargs, with the exact messages used in the implementation ("depth_direction must be given as kwarg", "min_channels_for_velocity must be given as kwarg", "min_r2_velocity must be given as kwarg", "column_range must be given as kwarg").
        - The function returns numpy.nan (float) when there are not enough channels below the max channel or when the fit score is below the provided threshold; this is the function's primary failure signal for downstream code.
        - The function calls transform_column_range and sort_template_and_locations which may reorder and/or filter the input arrays; these operations are applied internally and the function does not modify caller objects in place (it operates on the returned arrays from those helper functions).
        - The implementation treats the template minimum per channel as the spike peak (typical for extracellular negative-going spikes). If templates or coordinate conventions differ, results may be incorrect; users must ensure their templates and channel_locations follow the expected conventions (template shape and channel coordinate units consistent with micrometers).
    
    Returns:
        float: The fitted propagation velocity below the max-amplitude channel. The numeric value is the slope produced by fitting distances (in micrometers) versus peak time differences (in milliseconds). Because distances are in um and times are in ms, the raw fitted slope has units of um/ms. The original module documentation states the velocity is reported in um/s; callers should be aware of this implementation detail (if um/s are required, multiply the returned value by 1000). If the velocity could not be estimated because of insufficient channels or a poor fit, numpy.nan is returned to indicate an invalid/unreliable estimate.
    """
    from spikeinterface.postprocessing.template_metrics import get_velocity_below
    return get_velocity_below(template, channel_locations, sampling_frequency, **kwargs)


################################################################################
# Source: spikeinterface.postprocessing.template_metrics.sort_template_and_locations
# File: spikeinterface/postprocessing/template_metrics.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_postprocessing_template_metrics_sort_template_and_locations(
    template: numpy.ndarray,
    channel_locations: numpy.ndarray,
    depth_direction: str = "y"
):
    """spikeinterface.postprocessing.template_metrics.sort_template_and_locations sorts a template waveform array and the corresponding channel location coordinates by electrode depth. This function is used in spike sorting post-processing to reorder channels and their spatial coordinates so that downstream metrics, visualizations, or curation workflows see channels in increasing depth order (e.g., from superficial to deep electrodes). The function implements a deterministic ascending sort of channels based on a single spatial axis (depth), where the axis used is selected by the depth_direction parameter.
    
    Args:
        template (numpy.ndarray): 2D array containing template waveforms for a unit/channel set. In common SpikeInterface usage this array has shape (n_timepoints, n_channels) where each column corresponds to the waveform recorded on one channel. The function returns a new array with the same dtype and first dimension preserved, but with columns reordered to match the sorted channel order. This parameter must be indexable along its second axis (columns) because columns are permuted by the computed sort indices.
        channel_locations (numpy.ndarray): 2D array where each row corresponds to the spatial coordinates of a channel/electrode. In typical usage this is an array of shape (n_channels, n_spatial_dims) (for example, n_spatial_dims == 2 for x,y coordinates). The function uses one column of this array (selected by depth_direction) to compute the sort order and returns a reordered copy of this array with rows permuted to match the sorted channel order. If this array has fewer columns than expected (for example, no column for the requested depth dimension), an IndexError may occur.
        depth_direction (str): Which coordinate column of channel_locations represents electrode depth. If equal to the exact string "y" the function treats the second column (index 1) of channel_locations as the depth coordinate; for any other value it treats the first column (index 0) as depth. The default is "y". Note that the comparison is exact and case-sensitive: "Y" or other strings are treated as not equal to "y" and will cause the function to use column 0.
    
    Returns:
        tuple: A pair of numpy.ndarray objects (sorted_template, sorted_channel_locations).
            sorted_template (numpy.ndarray): The input template with its columns permuted according to increasing depth coordinate. The first axis (typically time samples) is unchanged; the second axis (channels) is reordered.
            sorted_channel_locations (numpy.ndarray): The input channel_locations with its rows permuted to match sorted_template so that each row still corresponds to the same channel as the corresponding column in sorted_template.
    
    Behavior and side effects:
        The function computes sort_indices = numpy.argsort(channel_locations[:, depth_dim]) where depth_dim is 1 if depth_direction == "y" else 0, and then returns template[:, sort_indices] and channel_locations[sort_indices, :]. The sort is ascending (shallow to deep by numeric value). The operation does not modify the input arrays in-place; it returns new views or copies according to NumPy semantics. Because the depth_direction check is a simple equality test, callers must pass the exact string "y" to select the second column; any other string selects the first column.
    
    Failure modes and notes for users:
        If the number of columns in template does not match the number of rows in channel_locations (i.e., n_channels mismatch), the returned arrays will be inconsistent for downstream code that assumes one-to-one correspondence between template columns and channel_locations rows. If channel_locations has fewer columns than required for the selected depth_dim, an IndexError will be raised. The function relies on numpy.argsort and therefore has the usual performance characteristics of NumPy sorts (O(n log n) time for n channels).
    """
    from spikeinterface.postprocessing.template_metrics import sort_template_and_locations
    return sort_template_and_locations(template, channel_locations, depth_direction)


################################################################################
# Source: spikeinterface.postprocessing.template_metrics.transform_column_range
# File: spikeinterface/postprocessing/template_metrics.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_postprocessing_template_metrics_transform_column_range(
    template: numpy.ndarray,
    channel_locations: numpy.ndarray,
    column_range: float,
    depth_direction: str = "y"
):
    """Transform template and channel locations to include only channels within a given column range around the template's peak channel.
    
    This function is used in the SpikeInterface postprocessing/template_metrics workflow to restrict a spike template and its associated channel coordinates to a spatial subset (a "column") centered on the channel that has the largest peak-to-peak amplitude in the provided template. The result is commonly used when computing template-based quality metrics or visualizing waveforms on a subset of electrode columns in extracellular recording arrays.
    
    Args:
        template (numpy.ndarray): The template waveform array representing averaged spike waveforms across recording channels. In code this is used with numpy.ptp(template, axis=0), so the array is expected to have timepoints along axis 0 and channels along axis 1 (typical shape (n_timepoints, n_channels)). The function identifies the channel with the largest peak-to-peak amplitude from this array to center the column selection.
        channel_locations (numpy.ndarray): Array of channel spatial coordinates for each recording channel. The function indexes rows of this array by channel index (the same ordering as template's channel axis) and reads coordinate components by integer column indices. Typically this is a 2D array with shape (n_channels, n_coords) such as (n_channels, 2) holding x/y coordinates; the function uses column 0 when computing the reference coordinate (see behavior).
        column_range (float): Half-width of the column selection in the same spatial units as channel_locations. If column_range is None (the function explicitly checks for None), no spatial filtering is applied and the original inputs are returned unchanged. When a float is provided, channels whose coordinate (along the chosen column axis) differs from the reference coordinate by at most column_range are kept (comparison uses <=).
        depth_direction (str): Direction string that controls which coordinate column of channel_locations is used to form the column mask. If depth_direction == "y" then column_dim is set to 0 and the mask compares channel_locations[:, 0] to the reference coordinate; if depth_direction is any other value then column_dim is set to 1 and the mask compares channel_locations[:, 1]. The default is "y". Note: the reference coordinate used to center the selection is always taken from channel_locations[..., 0] at the channel index with maximum peak-to-peak amplitude in template (see behavior), so if depth_direction != "y" the code will still use the 0th coordinate of the peak channel as the center while comparing along column_dim.
    
    Returns:
        template_column_range (numpy.ndarray): The template array restricted to the selected columns (same dtype as input template). If column_range is None this is the original template object (no copy guaranteed). If filtering is applied this has the same timepoint axis as template and a reduced channel axis corresponding to the channels that satisfy the spatial mask. If no channels satisfy the mask an array with zero columns is returned.
        channel_locations_column_range (numpy.ndarray): The subset of channel_locations corresponding to the selected channels (same dtype/ndim as input). If column_range is None this is the original channel_locations object. If filtering is applied this contains only the rows for channels within the column selection; it may be empty if no channels match.
    
    Behavior and failure modes:
        The function first determines column_dim = 0 if depth_direction == "y" else 1. If column_range is None, the function returns the input template and channel_locations unchanged. Otherwise the function computes per-channel peak-to-peak amplitudes via numpy.ptp(template, axis=0), finds the channel index with maximum amplitude using numpy.argmax, and reads that channel's coordinate at column index 0 from channel_locations to form the reference coordinate (variable max_channel_x in the source). It then builds a boolean mask of channels whose coordinate at column_dim is within column_range of that reference coordinate, applies that mask to the channel axis of template and to the rows of channel_locations, and returns the two filtered arrays.
    
        The function does not modify the input arrays in-place (it returns views or new arrays depending on numpy's indexing behavior) and has no other side effects. Potential failure modes include IndexError or unexpected behavior if template and channel_locations do not have the same number of channels in the same ordering (the function assumes channel indices are aligned between the two inputs), or if template has an unexpected shape such that numpy.ptp(template, axis=0) is invalid. Because the reference coordinate is always taken from channel_locations[:, 0] at the peak channel index, using depth_direction values other than "y" may produce unintuitive centering if your coordinate system is not organized with the intended axis in column 0.
    """
    from spikeinterface.postprocessing.template_metrics import transform_column_range
    return transform_column_range(
        template,
        channel_locations,
        column_range,
        depth_direction
    )


################################################################################
# Source: spikeinterface.postprocessing.template_similarity.check_equal_template_with_distribution_overlap
# File: spikeinterface/postprocessing/template_similarity.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_postprocessing_template_similarity_check_equal_template_with_distribution_overlap(
    waveforms0: numpy.ndarray,
    waveforms1: numpy.ndarray,
    template0: numpy.ndarray = None,
    template1: numpy.ndarray = None,
    num_shift: int = 2,
    quantile_limit: float = 0.8,
    return_shift: bool = False
):
    """Check whether two sets of spike waveforms come from the same underlying waveform distribution by measuring overlap of their projected scalar-value distributions along the vector between their templates. This function is used in the SpikeInterface postprocessing context (for example internally by tridesclous for the automatic merge step) and can also serve as a simple distance metric between two clusters of spikes when building or evaluating sorting pipelines.
    
    The function projects each waveform onto the normalized vector from template0 to template1, computes per-cluster scalar projections, and compares specified quantiles of these scalar distributions across a range of discrete sample shifts. If the chosen quantile of waveforms0 is greater than or equal to the complementary quantile of waveforms1 for any tested shift, the clusters are considered equal (i.e., too overlapping to be reliably distinct). The tested shifts allow the comparison to be robust to small temporal misalignments between clusters.
    
    Args:
        waveforms0 (numpy.ndarray): First set of waveforms with shape (num_spikes, num_samples, num_chans). Each entry is an individual spike waveform for a candidate cluster. num_spikes may differ between waveforms0 and waveforms1. This array must be "spasifyed" (preprocessed and aligned) outside this function as expected by the calling pipeline.
        waveforms1 (numpy.ndarray): Second set of waveforms with shape (num_spikes, num_samples, num_chans). Must have the same num_samples and num_chans as waveforms0; otherwise an AssertionError is raised. This is the other candidate cluster to compare against waveforms0.
        template0 (numpy.ndarray): Mean template for waveforms0 with shape (num_samples, num_chans). If None, the function computes template0 = numpy.mean(waveforms0, axis=0). The template represents the average waveform of the cluster and defines the reference for projection and shift comparisons.
        template1 (numpy.ndarray): Mean template for waveforms1 with shape (num_samples, num_chans). If None, the function computes template1 = numpy.mean(waveforms1, axis=0). template1 is used together with template0 to form the projection vector used to compare distributions.
        num_shift (int): Number of discrete sample shifts to test on each side of the central alignment (default 2). The function tests shifts in the integer range [ -num_shift, ..., 0, ..., +num_shift ] by slicing waveform windows and corresponding template windows; the tested shift that yields overlap below the threshold is returned when return_shift is True. num_shift must be non-negative and small enough so that slicing template[num_shift:-num_shift] yields a non-empty central window; otherwise an error (e.g., empty slices or IndexError) will occur.
        quantile_limit (float): Quantile in the interval [0, 1] that defines the overlap threshold (default 0.8). For each tested shift, the function computes l0 = quantile(scalar_projections_of_waveforms0, quantile_limit) and l1 = quantile(scalar_projections_of_waveforms1, 1 - quantile_limit). If l0 >= l1, the distributions are considered overlapping enough to declare the clusters equal. quantile_limit must be between 0 and 1 inclusive; values outside this range will produce undefined behavior from numpy.quantile.
        return_shift (bool): If True, the function returns a tuple (equal, final_shift) where final_shift is the integer sample offset (shift - num_shift) at which equality was detected; if no equality is detected final_shift is None. If False (default), only the boolean equal is returned.
    
    Returns:
        bool: When return_shift is False, returns a single boolean equal indicating whether the two waveform sets are considered to come from the same distribution (True) or not (False).
        tuple(bool, int or None): When return_shift is True, returns (equal, final_shift). equal is the same boolean as above. final_shift is the integer shift (negative means template1 is shifted earlier relative to template0) at which equality was first detected; if no shift yields equality final_shift is None.
    
    Behavior, side effects, defaults, and failure modes:
        - The function asserts that waveforms0 and waveforms1 have identical num_samples and num_chans; an AssertionError is raised if this is not true.
        - If template0 or template1 is None, they are computed as the mean across the first axis of the corresponding waveforms (numpy.mean(..., axis=0)). This introduces no external side effects but does allocate the computed template.
        - The function extracts a central window of the templates via template[num_shift:-num_shift, :]. Therefore num_shift must be strictly smaller than half of num_samples to produce a non-empty window; otherwise slicing may produce empty arrays and subsequent operations will fail.
        - For each tested integer shift in 0..(2 * num_shift), the code aligns a windowed version of template1 and waveforms1 and forms vector_0_1 = template1_window - template0_window. This vector is normalized by dividing by the scalar numpy.sum(vector_0_1**2). If the sum of squares is zero (for example when templates are identical across the window), this normalization will divide by zero and produce inf or nan values, which will lead to incorrect or undefined comparison results; callers should ensure templates are not identically zero in the evaluated window or handle nan/inf after the call.
        - The projection is computed as the elementwise inner product between (waveform - template0_window) and the normalized vector over sample and channel axes, producing one scalar per waveform. The function then compares the specified quantiles of the two scalar distributions to decide equality.
        - No external state is modified. Memory copies are made when slicing waveforms for shift evaluation.
        - Typical reasons for failure include mismatched waveform dimensions, inappropriate num_shift relative to waveform length, quantile_limit outside [0, 1], or degenerate templates that lead to division by zero during normalization.
        - This algorithm is simple and intended for use as a heuristic in postprocessing/auto-merge workflows; it does not replace more sophisticated statistical tests but provides a practical, fast overlap-based criterion used in SpikeInterface and tridesclous.
    """
    from spikeinterface.postprocessing.template_similarity import check_equal_template_with_distribution_overlap
    return check_equal_template_with_distribution_overlap(
        waveforms0,
        waveforms1,
        template0,
        template1,
        num_shift,
        quantile_limit,
        return_shift
    )


################################################################################
# Source: spikeinterface.preprocessing.detect_bad_channels.detect_bad_channels_ibl
# File: spikeinterface/preprocessing/detect_bad_channels.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_preprocessing_detect_bad_channels_detect_bad_channels_ibl(
    raw: numpy.ndarray,
    fs: float,
    psd_hf_threshold: float,
    dead_channel_thr: float = -0.5,
    noisy_channel_thr: float = 1.0,
    outside_channel_thr: float = -0.75,
    n_neighbors: int = 11,
    nyquist_threshold: float = 0.8,
    welch_window_ms: float = 0.3,
    outside_channels_location: str = "top"
):
    """Bad channels detection for Neuropixel probes developed by the International Brain Laboratory (IBL). This function is intended for pre-processing extracellular recordings in the SpikeInterface pipeline: it analyzes per-channel coherence and high-frequency power to flag channels that are likely dead (very low coherence/amplitude), noisy (high high-frequency power and/or high coherence), or located outside of the brain (contiguous channels at probe extremes with low trend coherence). The function implements the IBL heuristic using a Welch power spectral density (PSD) estimate and a local-detrending coherence measure computed across probe channels; it is typically used before spike sorting to exclude or mark channels that would degrade sorting quality.
    
    Args:
        raw (numpy.ndarray): Raw voltage traces with shape (num_samples, n_channels). This is the input extracellular recording (time x channels). The function computes per-channel mean removal (raw - mean) internally; the input array in the caller is not modified in-place by this function (the implementation assigns a new array to the local variable). The array must be two-dimensional; if not, a runtime error may occur.
        fs (float): Sampling frequency in Hz. Used to compute the Welch PSD frequency axis and convert welch_window_ms to nperseg. Accurate sampling frequency is required because frequency bands (including the Nyquist-based HF band) are computed from fs.
        psd_hf_threshold (float): Threshold applied to the mean PSD in the high-frequency band (see nyquist_threshold). For each channel, the function computes the mean PSD over frequencies above (fs / 2 * nyquist_threshold); channels whose mean PSD in that band exceeds psd_hf_threshold are candidate noisy channels (they are flagged noisy only if they also satisfy the coherence/noise criteria described below).
        dead_channel_thr (float): Threshold for the local coherence indicator below which channels are labeled as dead (default: -0.5). The coherence indicator is derived from the channel-wise dot-product with a median reference and a local detrending operation using n_neighbors. Channels with coherence below this threshold are marked as dead (label 1), indicating very low similarity and/or amplitude compared to neighboring channels.
        noisy_channel_thr (float): Threshold for the local coherence indicator above which channels are labeled as noisy (default: 1.0). Channels whose local coherence exceeds this value or whose high-frequency PSD exceeds psd_hf_threshold are flagged as noisy (label 2). This helps detect channels dominated by high-frequency noise or excessively correlated artifacts.
        outside_channel_thr (float): Threshold applied to the distant-trend coherence (default: -0.75). The function computes a distant coherence trend (xcorr_distant) by removing the local trend from the coherence signal; contiguous channels where xcorr_distant is below outside_channel_thr are candidate outside-of-brain channels. To be marked as outside (label 3), these contiguous channels must also be at an extreme of the probe as determined by outside_channels_location.
        n_neighbors (int): Number of neighboring channels used by the detrend operation to compute the local coherence trend (default: 11). This parameter controls the spatial window for estimating local channel coherence; larger values produce a smoother local trend and affect which channels are considered outliers versus locally coherent.
        nyquist_threshold (float): Fraction of the Nyquist frequency used to define the high-frequency PSD band (default: 0.8). The high-frequency band is defined as frequencies f > (fs/2 * nyquist_threshold). This value therefore determines what part of the PSD is considered when computing psd_hf for the noisy-channel test.
        welch_window_ms (float): Window length in milliseconds for scipy.signal.welch that is converted to an integer nperseg via int(welch_window_ms * fs / 1000) (default: 0.3 ms). This controls the frequency resolution and variance of the PSD estimate. If the computed nperseg is less than 1 or incompatible with scipy.signal.welch, scipy will raise an error; ensure fs and welch_window_ms produce a valid nperseg.
        outside_channels_location (str): One of "top", "bottom", or "both" indicating which probe extremes may be labeled as outside-of-brain channels (default: "top"). The implementation assumes channels are ordered bottom-to-top along the probe (index 0 = bottom, index n_channels-1 = top). If "top", only contiguous candidate outside channels that include the last channel index (n_channels-1) can be labeled outside; if "bottom", only those that include index 0 can be labeled outside; if "both", candidate contiguous runs at either extreme may be labeled outside. Passing other string values will be treated by the implementation like "both" (i.e., both extremes considered) but callers should use the documented three choices.
    
    Returns:
        numpy.ndarray: 1D integer array of length n_channels giving channel labels for each input channel. Values are: 0 = good (no flag), 1 = dead (low coherence/amplitude), 2 = noisy (high HF PSD and/or high coherence), 3 = outside of the brain (contiguous extreme channels with low distant-trend coherence). The returned array has dtype int (as constructed in the implementation) and is suitable for masking or indexing channels in downstream SpikeInterface preprocessing and spike sorting steps.
    
    Behavior, defaults, and failure modes:
        The function performs the following high-level steps: per-channel mean removal, computation of a reference median waveform across time, computation of a channel-wise coherence measure via a dot product with the reference, local detrending of that coherence using n_neighbors, estimation of PSD via scipy.signal.welch with nperseg = int(welch_window_ms * fs / 1000), computation of mean high-frequency PSD above (fs/2 * nyquist_threshold), and application of thresholds to produce the channel labels described above. Default thresholds are chosen based on the IBL Neuropixel heuristics (dead_channel_thr=-0.5, noisy_channel_thr=1.0, outside_channel_thr=-0.75) but may need tuning for different probes or recording conditions.
        The function imports and uses scipy.signal.welch internally; if SciPy is not installed, an ImportError will be raised. If raw is not a 2D numpy.ndarray or is extremely large, the function may raise errors or consume substantial memory/time when computing the PSD and coherence; callers should ensure adequate memory and that raw shape is (num_samples, n_channels). If welch_window_ms and fs lead to nperseg < 1, scipy.signal.welch will raise an error; choose welch_window_ms and fs consistent with the required nperseg. The function does not modify the caller's raw array in-place (it reassigns a local array after subtracting the mean), but the local computation may allocate large temporary arrays.
        This detection is heuristic: false positives and false negatives are possible, particularly for atypical probe geometries, ordering of channels, very noisy recordings, or non-Neuropixel probes. The labels produced are intended as recommendations to be used by downstream curation (e.g., masking, exclusion, or manual review) within the SpikeInterface workflow.
    """
    from spikeinterface.preprocessing.detect_bad_channels import detect_bad_channels_ibl
    return detect_bad_channels_ibl(
        raw,
        fs,
        psd_hf_threshold,
        dead_channel_thr,
        noisy_channel_thr,
        outside_channel_thr,
        n_neighbors,
        nyquist_threshold,
        welch_window_ms,
        outside_channels_location
    )


################################################################################
# Source: spikeinterface.preprocessing.detect_bad_channels.detrend
# File: spikeinterface/preprocessing/detect_bad_channels.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_preprocessing_detect_bad_channels_detrend(x: numpy.ndarray, nmed: int):
    """spikeinterface.preprocessing.detect_bad_channels.detrend subtracts a median-filtered trend from a 1-D signal (vector) using endpoint tapering. This function is used in spike preprocessing (for example, in detect_bad_channels) to remove slowly varying baseline or trend from an extracellular recording channel prior to computing metrics for bad-channel detection or other downstream analyses.
    
    Args:
        x (numpy.ndarray): Input 1-D array (vector) containing the sampled signal for a single channel or trace. In the SpikeInterface preprocessing context, this typically represents a time series of extracellular voltage values for one channel. The function treats x as read-only and returns a new array; it does not modify x in place. x must be non-empty because the implementation accesses x[0] and x[-1].
        nmed (int): Number of points used for the median filter kernel (median filter length). This integer controls the time scale of the trend that is removed: larger values remove slower-varying trends. There is no default; the caller must supply a positive integer. The implementation computes ntap = ceil(nmed / 2) and pads the signal by repeating the first and last samples ntap times before applying scipy.signal.medfilt with kernel size nmed.
    
    Returns:
        numpy.ndarray: A new numpy array of the same length as x containing the detrended signal computed as x - trend, where trend is the median-filtered and tapered version of x. The returned array preserves the sample-wise alignment with the input trace and is intended to remove baseline drift or slow components so that short-timescale features (e.g., spikes) are emphasized.
    
    Behavior and side effects:
        The function pads the input signal on both ends by repeating the first sample ntap times and the last sample ntap times (ntap = ceil(nmed / 2)), applies scipy.signal.medfilt with kernel length nmed to compute a trend, removes the padding, and subtracts this trend from the original signal. The function performs an import of scipy.signal at call time; if SciPy is not installed, an ImportError will be raised. The function does not modify the input array x in place and has no other side effects.
    
    Failure modes and notes:
        If x is empty, accessing x[0] or x[-1] raises an IndexError. The function assumes x is a 1-D numpy.ndarray (a vector); behavior for higher-dimensional arrays is not intended and may be unexpected. The function does not validate that nmed is positive or that it is smaller than the signal length; providing an invalid nmed (for example, non-positive) may raise an error from numpy or scipy or produce an unexpected result. The exact boundary behavior for extreme kernel sizes is governed by scipy.signal.medfilt.
    """
    from spikeinterface.preprocessing.detect_bad_channels import detrend
    return detrend(x, nmed)


################################################################################
# Source: spikeinterface.preprocessing.highpass_spatial_filter.agc
# File: spikeinterface/preprocessing/highpass_spatial_filter.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_preprocessing_highpass_spatial_filter_agc(
    traces: numpy.ndarray,
    window: numpy.ndarray,
    epsilon: float = 1e-08
):
    """Automatic gain control (AGC) applied to multichannel time series used in SpikeInterface preprocessing. This function computes a local amplitude envelope per time sample and channel by convolving the absolute traces with a 1-D window kernel, uses a small epsilon regularizer to avoid division by zero, and divides the input traces by the computed gain to produce AGC-normalized traces. In the spike-sorting domain (SpikeInterface preprocessing of extracellular recordings) this reduces slow amplitude variations across time and channels, aiding consistent spike detection and downstream sorting.
    
    Args:
        traces (numpy.ndarray): Multichannel time series to be AGC-normalized. The implementation expects samples along axis 0 and channels along axis 1 (shape convention (n_samples, n_channels) as used in the source code). This array is used as input and is modified in-place for channels that are not marked as "dead" (see behavior below). In typical SpikeInterface workflows this contains preprocessed extracellular recordings prior to spike detection or filtering.
        window (numpy.ndarray): One-dimensional convolution kernel (window) provided in sample units (length in samples). The kernel is convolved with numpy.abs(traces) along axis 0 using scipy.signal.fftconvolve(mode="same", axes=0) to compute a local amplitude (gain) envelope. The kernel should be appropriate for the desired temporal smoothing (for example a rectangular or tapered window of N samples corresponding to the target time window given the recording sampling rate). The function expects a 1-D numpy.ndarray; incompatible shapes will cause scipy.signal.fftconvolve to raise an error.
        epsilon (float): Small non-negative scalar regularizer added to the per-channel gain to avoid division by zero and excessive amplification of synthetic or near-zero signals. Default is 1e-08. The code adds a channel-wise offset proportional to (sum(gain, axis=0) * epsilon / n_samples) to the computed gain before division. Supplying a negative epsilon is not supported and may lead to incorrect behavior.
    
    Returns:
        tuple: A two-item tuple (agc_traces, gain) where both elements are numpy.ndarray and have the same shape as the input traces (samples along axis 0, channels along axis 1).
        agc_traces (numpy.ndarray): The traces after AGC normalization. For channels with non-zero computed gain the function divides traces by gain elementwise and writes the result into the provided traces array (in-place modification). Channels detected as "dead" (channels whose gain sums to zero) are left unchanged in the returned traces array.
        gain (numpy.ndarray): The computed non-negative gain envelope used to normalize the data. gain is obtained by fftconvolving numpy.abs(traces) with window along axis 0, then adding the epsilon-based offset. Channels whose gain is identically zero are reported as such (their gain column will be zeros).
    
    Behavior, side effects, and failure modes:
        - The convolution is performed with scipy.signal.fftconvolve(..., mode="same", axes=0), so edge effects follow the 'same' convolution semantics.
        - The function modifies the input traces array in-place for channels that are not marked as dead; callers that need to retain the original data must pass a copy of traces.
        - Channels whose gain sums to zero across time are considered dead and are not divided (their traces remain as in the input and their gain remains zero).
        - The function does not validate that traces contains floating-point values; if integer arrays are provided, division may upcast or produce unintended results—prefer floating-point numpy arrays for audio/seismic/neural recordings.
        - If window is not a 1-D numpy.ndarray or its shape is incompatible with fftconvolve and traces, scipy.signal.fftconvolve will raise an error (e.g., a ValueError). If epsilon is excessively large it will bias the normalization toward no-op behavior; if epsilon is negative behavior is undefined.
        - The function is intended for preprocessing within SpikeInterface to normalize amplitude variations in extracellular recordings for spike sorting and should be used with a window chosen according to the recording sampling rate and the desired temporal smoothing.
    """
    from spikeinterface.preprocessing.highpass_spatial_filter import agc
    return agc(traces, window, epsilon)


################################################################################
# Source: spikeinterface.preprocessing.highpass_spatial_filter.fcn_cosine
# File: spikeinterface/preprocessing/highpass_spatial_filter.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_preprocessing_highpass_spatial_filter_fcn_cosine(bounds: tuple):
    """Return a soft-thresholding callable that applies a cosine taper between two numeric bounds used by preprocessing highpass_spatial_filter in SpikeInterface.
    
    This function constructs and returns a lambda that implements a smooth, cosine-shaped transition (taper) between two threshold bounds. In the SpikeInterface preprocessing context, this is used to produce a soft thresholding function for spatial high-pass filtering or related preprocessing steps where abrupt clipping of values is undesirable; the cosine taper yields a gradual interpolation between passing values unchanged below the lower bound and clamping values to the upper bound above the upper bound.
    
    Args:
        bounds (tuple): A length-2 tuple containing the lower and upper threshold values in the order (lower, upper). These two numeric bounds define the regions of behavior for the returned function: for inputs less than or equal to bounds[0] the returned function will leave values unchanged; for inputs strictly between bounds[0] and bounds[1] the returned function will apply a cosine taper (using the internal formula (1 - cos((x - bounds[0])/(bounds[1] - bounds[0]) * pi))/2 to generate the taper); for inputs greater than or equal to bounds[1] the returned function will clamp or map values to bounds[1]. In the spike-sorting preprocessing domain, choose bounds to represent the numeric thresholds for transition between unmodified signal and clipped/attenuated signal. If bounds is not a sequence with two accessible elements, or if bounds[1] == bounds[0], calling the returned function may raise an exception (for example, TypeError or ZeroDivisionError) because the implementation divides by (bounds[1] - bounds[0]).
    
    Returns:
        callable: A lambda function that implements the soft-thresholding behavior described above. The callable accepts an input x (the original code expects x to be a numeric scalar or array-like object appropriate for the surrounding preprocessing pipeline) and returns the thresholded/transformed value(s). The returned function is pure (it does not modify external state) and is intended to be used wherever a smooth transition between passing original values and clamping to an upper bound is required in the SpikeInterface preprocessing highpass_spatial_filter workflow.
    
    Raises:
        TypeError: If bounds does not support indexing or is not a tuple-like sequence of two elements when the returned function is invoked.
        ZeroDivisionError: If bounds[1] == bounds[0], because the cosine taper calculation divides by (bounds[1] - bounds[0]).
    """
    from spikeinterface.preprocessing.highpass_spatial_filter import fcn_cosine
    return fcn_cosine(bounds)


################################################################################
# Source: spikeinterface.preprocessing.motion.get_motion_parameters_preset
# File: spikeinterface/preprocessing/motion.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_preprocessing_motion_get_motion_parameters_preset(preset: str):
    """Get the parameters tree for a named preset used by the motion-correction steps in the preprocessing pipeline.
    
    This function is part of SpikeInterface's preprocessing.motion utilities and returns a fully-resolved parameters dictionary (a "parameters tree") for a given preset name. In the spike sorting domain, motion correction is a preprocessing stage that compensates for probe and tissue movement by running a sequence of algorithmic steps (for example: selecting peaks, estimating drift, splitting/merging segments). Each step can have multiple methods and method-specific parameters. Presets are high-level named configurations that select methods and override a subset of parameters for those steps. This function takes the named preset, deep-copies the preset definition from the internal registry, merges it with the framework's default motion parameters for each step and method, and returns the resulting parameter tree that the motion-correction pipeline consumes to configure and run each step.
    
    Args:
        preset (str): The preset name to resolve into a parameters tree. The string must be a key present in the internal preset registry (motion_options_preset) and typically corresponds to a predefined motion-correction configuration provided by SpikeInterface. The original implementation documented a default of None; in practice the function expects a valid preset name and will raise a KeyError if the provided name is not found in the preset registry. Use spikeinterface.preprocessing.get_motion_presets() to list available preset names. The preset controls which methods are selected for each motion-correction step and any preset-specific parameter overrides; this function will merge those overrides with the framework defaults for each selected method.
    
    Returns:
        dict: A parameters tree (dictionary) mapping motion-correction step names to either:
            - a string: a documentation key or descriptive text for that step (when the preset stores a doc key),
            - an empty dict: indicating the step should be skipped (for example, an explicitly empty configuration for select_peaks),
            - a dict of parameters: where method-specific default parameters (obtained from _get_default_motion_params()) have been copied and then updated with any overrides provided by the preset. This returned dict is intended to be passed directly to the motion-correction pipeline within spikeinterface.preprocessing.motion to configure each step and its chosen method.
        The function performs a deep copy of the preset definition and merges defaults; it does not modify the global preset registry. Failure modes include KeyError if the preset name is not present in the internal registry, and ValueError if the preset structure is not a recognized format (the function validates that each step entry is a string, an empty dict, or a dict containing optional "method" and parameter entries). Other potential KeyError or lookup errors can occur if expected default parameters for a referenced step/method are missing from the framework defaults. The function has no external side effects other than reading global preset and default parameter structures.
    """
    from spikeinterface.preprocessing.motion import get_motion_parameters_preset
    return get_motion_parameters_preset(preset)


################################################################################
# Source: spikeinterface.preprocessing.motion.load_motion_info
# File: spikeinterface/preprocessing/motion.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_preprocessing_motion_load_motion_info(folder: str):
    """Loads motion-related metadata and Motion object from a folder produced by SpikeInterface preprocessing. This function reads JSON metadata, optional numpy arrays, and either a saved Motion object (current implementation) or a legacy set of numpy files and returns a dictionary summarizing all loaded motion information. In the SpikeInterface domain this is used to recover motion/drift estimation outputs produced during preprocessing so they can be inspected, visualized, or used by downstream postprocessing and quality-metric computations.
    
    Args:
        folder (str): Filesystem path (as a string) to the folder that contains the motion information to load. The folder is expected to contain at minimum parameters.json and run_times.json produced by SpikeInterface motion routines. Optionally it may contain peaks.npy and peak_locations.npy (each loaded with numpy.load), a subdirectory named "motion" that can be loaded using spikeinterface.core.motion.Motion.load, or legacy files spatial_bins.npy, temporal_bins.npy, and motion.npy that will be converted to a Motion instance. The argument is treated as a path string pointing to an existing directory; if required files are missing or unreadable, file system or parsing errors will be raised (see Failure modes).
    
    Returns:
        dict: A dictionary ("motion_info") containing the loaded motion information with these keys and meanings:
            "parameters": the parsed JSON object (dict or list) loaded from parameters.json; this contains motion-related parameters recorded by the preprocessing step and is used to interpret the motion estimates.
            "run_times": the parsed JSON object (dict or list) loaded from run_times.json; this records runtime metadata for motion computation.
            "peaks": a numpy.ndarray if peaks.npy exists, otherwise None. When present this array contains peak amplitudes or related per-event values saved by the preprocessing pipeline and is loaded into memory with numpy.load.
            "peak_locations": a numpy.ndarray if peak_locations.npy exists, otherwise None. When present this array contains spatial locations for peaks as saved by preprocessing and is loaded into memory with numpy.load.
            "motion": an instance of spikeinterface.core.motion.Motion when a Motion object could be loaded either from the "motion" subdirectory (using Motion.load) or reconstructed from the legacy files spatial_bins.npy, temporal_bins.npy, and motion.npy. If no Motion data is found, the value is None.
    
    Behavior, side effects, defaults, and failure modes:
        This function always attempts to open and parse parameters.json and run_times.json in the supplied folder; missing or unreadable JSON files will raise standard IO or JSON parsing exceptions (for example FileNotFoundError or json.JSONDecodeError). For peaks.npy and peak_locations.npy the function checks for file existence and sets the corresponding dictionary entry to None if the file is absent; if the file exists but is corrupt, numpy.load will raise its usual exceptions. If a subdirectory named "motion" exists, the function delegates to Motion.load to reconstruct the Motion object; errors raised by Motion.load propagate to the caller. If the "motion" subdirectory does not exist, the function checks for the legacy trio of files spatial_bins.npy, temporal_bins.npy, and motion.npy; if all three are present they are loaded and used to construct a Motion instance (temporal_bins and displacement are wrapped in single-element lists to match the Motion constructor expectations). If neither the "motion" directory nor the legacy files are present, the function emits a runtime warning (via warnings.warn) stating that no Motion object was found and sets motion_info["motion"] to None. Note that loading .npy files can be memory intensive for large motion arrays; numpy.load will allocate arrays in memory. The function may also emit warnings when falling back to the legacy format or when no motion object is available.
    """
    from spikeinterface.preprocessing.motion import load_motion_info
    return load_motion_info(folder)


################################################################################
# Source: spikeinterface.preprocessing.motion.save_motion_info
# File: spikeinterface/preprocessing/motion.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_preprocessing_motion_save_motion_info(
    motion_info: dict,
    folder: str,
    overwrite: bool = False
):
    """spikeinterface.preprocessing.motion.save_motion_info saves motion-analysis results produced by compute_motion to disk in a structured folder so they can be reloaded for downstream preprocessing, drift correction, benchmarking, and reproducibility in spike sorting workflows provided by SpikeInterface.
    
    This function expects the motion_info dictionary returned by compute_motion and writes several files into the target folder: a JSON-serialized copy of the analysis parameters, a JSON file of run times, NumPy .npy files for detected peaks and their locations, and, if present, a serialized motion object saved via its own save method. The function creates the folder (including parent directories) when needed, and enforces an explicit overwrite policy to avoid accidental data loss.
    
    Args:
        motion_info (dict): The dictionary returned by compute_motion that contains motion estimation results and metadata. Required keys used by this function are "parameters" (a mapping of algorithm parameters; must be JSON-serializable via SIJsonEncoder), "run_times" (a mapping of timing information suitable for JSON encoding), "peaks" (an array-like collection of detected peak indices or times that will be written with numpy.save to peaks.npy), "peak_locations" (an array-like collection of corresponding peak spatial locations written with numpy.save to peak_locations.npy), and "motion" (either None or an object exposing a save(path) method; if not None, motion.save(folder / "motion") will be called to persist the motion object). The role of this argument is to provide the full set of results produced by compute_motion so they can be persisted and later reloaded for further preprocessing or analysis steps in the SpikeInterface pipeline.
        folder (str | Path): Filesystem path where motion_info will be saved. The function will create this directory (folder.mkdir with parents=True). If the folder already exists and overwrite is False, a FileExistsError is raised to prevent accidental overwrites; if overwrite is True the existing folder and its contents are removed with shutil.rmtree before creating a fresh folder. Practical significance: choose a path dedicated to storing motion-analysis artifacts for a given recording/session to keep downstream processing reproducible and organized.
        overwrite (bool): Whether to remove an existing folder at the target path before saving. Default is False. If False and the folder already exists, the function raises FileExistsError. If True, the function deletes the existing folder tree with shutil.rmtree which will remove all files under that path (risk of data loss), then recreates the folder and writes the files. Use True only when you intend to replace existing saved motion information.
    
    Behavior and side effects:
    This function performs filesystem writes and may raise errors related to missing expected keys in motion_info (KeyError), JSON serialization issues for "parameters" (TypeError or json.JSONEncodeError if objects are not serializable by SIJsonEncoder), numpy save errors if "peaks" or "peak_locations" are not array-like, permission or I/O errors when creating/deleting directories or writing files, and errors from motion.save if the motion object cannot be serialized. The function writes parameters.json using json.dumps with SIJsonEncoder, run_times.json using json.dumps, and saves peaks.npy and peak_locations.npy using numpy.save. If motion_info["motion"] is not None, motion.save(folder / "motion") is invoked; if it is None, no motion object file is produced. The function ensures the target folder exists after successful execution.
    
    Returns:
        None: This function does not return a value. Its purpose is to persist motion analysis artifacts to disk as side effects. On success, the following files and/or directories will exist under the provided folder path: parameters.json, run_times.json, peaks.npy, peak_locations.npy, and, when motion is not None, a saved motion object at folder/"motion".
    """
    from spikeinterface.preprocessing.motion import save_motion_info
    return save_motion_info(motion_info, folder, overwrite)


################################################################################
# Source: spikeinterface.preprocessing.phase_shift.apply_frequency_shift
# File: spikeinterface/preprocessing/phase_shift.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_preprocessing_phase_shift_apply_frequency_shift(
    signal: numpy.ndarray,
    shift_samples: numpy.ndarray,
    axis: int = 0
):
    """Apply a sub-sample-accurate frequency (phase) shift to a multi-channel signal buffer.
    
    This function is intended for preprocessing extracellular recordings in SpikeInterface prior to spike sorting or waveform extraction. It shifts each channel by a (possibly fractional) number of samples using the Fourier shift theorem: the signal is transformed to the frequency domain with a real FFT (rFFT), a complex phase rotation that corresponds to the requested time shift is applied per-frequency-bin and per-channel, and the result is transformed back to the time domain with an inverse real FFT (irFFT). This produces time shifts that are accurate below the sampling-period resolution and so are useful for aligning channels, correcting propagation delays between channels, or fine temporal registration of multichannel recordings before downstream spike-sorting, quality-metric computation, or waveform extraction.
    
    Args:
        signal (numpy.ndarray): Input real-valued signal array to be shifted. This array contains the time samples along the axis specified by the axis parameter and the channel layout on the remaining axes. For typical SpikeInterface preprocessing, signal is a 2D array with shape (num_time_samples, num_channels) and dtype compatible with scipy.fft.rfft/irfft. The function will perform an rFFT along the specified axis to compute the frequency-domain representation.
        shift_samples (numpy.ndarray): 1-D array of sample shifts, one entry per channel. Each value is the desired shift for the corresponding channel expressed in units of samples; fractional values are allowed to request sub-sample shifts. Conceptually, the time shift in seconds equals shift_samples / sampling_rate (sampling_rate is not an argument to this function and must be handled by the caller). The array length (shift_samples.size) must match the number of channels in signal along the non-shift axis (for axis=0 this means the second dimension: signal.shape[1] when signal is 2D). If sizes do not match, NumPy broadcasting or elementwise multiplication will fail and a ValueError (or related broadcasting exception) can be raised.
        axis (int): Axis along which to perform the shift. Default is 0. Currently, only axis=0 is supported by this implementation and passing any other value will cause a NotImplementedError. The axis specifies the time/sample axis over which the rFFT/irFFT are computed and therefore which axis is translated in time by shift_samples.
    
    Returns:
        numpy.ndarray: Time-domain signal array with the same overall shape as the input signal and with the requested per-channel sub-sample shifts applied along the specified axis. The returned array is the result of scipy.fft.irfft applied to the phase-rotated frequency-domain signal; dtype and exact memory layout follow scipy.fft.irfft behavior.
    
    Behavior, side effects, and failure modes:
        - Method: The function computes an rFFT of signal along axis, constructs a frequency grid using numpy.fft.rfftfreq(signal_length) and converts it to angular frequency (2*pi*freq). It multiplies that frequency grid by shift_samples to produce per-frequency-bin phase offsets (in radians), forms complex rotations exp(-1j * phase), multiplies the rFFT result by these rotations (elementwise, per channel), and finally performs an irFFT to return to the time domain.
        - In-place/overwrite behavior: The implementation calls scipy.fft.rfft(..., overwrite_x=True) and scipy.fft.irfft(..., overwrite_x=True) to reduce memory usage. As a consequence, the input signal buffer may be overwritten or reused internally by the FFT functions; callers must not rely on the original contents of signal being preserved after the call. If preservation is required, callers should pass a copy of signal.
        - Axis support: Only axis=0 is implemented. Calling with axis != 0 raises NotImplementedError.
        - Shape and broadcasting requirements: shift_samples must be shaped and ordered so that after adding a leading frequency-dimension it can broadcast to the rFFT output shape. In the common 2D case (time x channels) this requires shift_samples.size == signal.shape[1]. Mismatched sizes will cause broadcasting errors (ValueError) or incorrect results.
        - Input expectations: signal is expected to represent real-valued time-domain samples along the chosen axis (the code uses rFFT). While complex-valued arrays may be accepted by scipy.fft.rfft, the intended use in SpikeInterface is for real-valued extracellular recordings.
        - Numerical considerations: Because the method operates in the frequency domain, very large shifts or signals with non-periodic edges may introduce wrap-around effects inherent to FFT-based circular shifts; callers should window or pad signals appropriately if linear (non-circular) shift behavior is required.
        - Exceptions: The function can raise NotImplementedError for unsupported axis values, ValueError or broadcasting-related exceptions when shift_samples does not match the channel layout of signal, and typical scipy/NumPy errors for invalid input arrays or insufficient memory.
    """
    from spikeinterface.preprocessing.phase_shift import apply_frequency_shift
    return apply_frequency_shift(signal, shift_samples, axis)


################################################################################
# Source: spikeinterface.preprocessing.phase_shift.apply_fshift_ibl
# File: spikeinterface/preprocessing/phase_shift.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_preprocessing_phase_shift_apply_fshift_ibl(
    w: numpy.ndarray,
    s: numpy.ndarray,
    axis: int = 0,
    ns: int = None
):
    """apply_fshift_ibl shifts a 1D or 2D signal in the frequency domain to implement accurate non-integer sample shifts used in spike preprocessing and alignment.
    
    This function is adapted from IBLIB (https://github.com/int-brain-lab/ibllib/blob/master/ibllib/dsp/fourier.py) and is used in SpikeInterface preprocessing.phase_shift to perform sub-sample shifts on extracellular recordings or their frequency-domain representations. It multiplies the Fourier transform of the input by a phase factor corresponding to the requested shift(s), allowing precise alignment of waveforms across channels or time without relying on interpolation in the time domain.
    
    Args:
        w (numpy.ndarray): Input signal. In the common time-domain use, w is a real-valued array containing samples along the axis dimension; the function will compute an rFFT internally, apply the phase shift, and return a real time-domain array with the same shape and dtype as w. Alternatively, w may be a frequency-domain array produced by scipy.fft.rfft (complex-valued). In that case, the function treats w as already transformed and applies the phase shift directly in frequency space; when providing a frequency-domain array, also provide ns to disambiguate the original time-domain length. The axis dimension of w is the sample axis along which shifts are applied.
        s (numpy.ndarray): Shift amount(s) in samples. Positive values shift the signal forward in time (toward increasing sample indices). s may be a scalar (numpy.isscalar accepted) or a numpy.ndarray. If s is an array, it will be reshaped and broadcast so that a different shift can be applied along other axes while remaining constant along the sample axis; its shape must be compatible with w for broadcasting along the requested axis. Typical use: provide non-integer shifts (floats) to implement sub-sample alignment when aligning spike waveforms or channels.
        axis (int, optional): Axis along which to shift (index of the sample/time axis). Default is 0. This parameter identifies which dimension of w corresponds to samples so the rFFT/irFFT and phase multiplication are applied along that axis. If axis is out of range for w, an IndexError will be raised.
        ns (int, optional): Number of time-domain samples corresponding to the rFFT length. Default is None. When w is a frequency-domain rFFT array (complex), ns disambiguates the original time-domain length required by scipy.fft.irfft for reconstruction. If ns is None and w is complex, the function will infer ns from w.shape[axis] when possible, but this can be ambiguous for rFFT arrays and may lead to incorrect inverse transforms; therefore, when passing frequency-domain data, explicitly provide ns equal to the original number of time-domain samples.
    
    Behavior and side effects:
        - If w is real (non-complex), the function computes rfft(w, axis=axis), multiplies by a complex phase factor corresponding to s, then computes the real inverse irfft and returns a real array. The returned array will be cast to the original dtype of w.
        - If w is complex (assumed to be an rFFT frequency-domain array), the function treats it as already transformed, applies the phase factor in frequency space, and returns the (complex) frequency-domain array. To recover time-domain samples after passing a frequency-domain w, call scipy.fft.irfft with the same ns used here.
        - The function does not modify the input array w in-place; it returns a new array.
        - When s is non-scalar, it will be reshaped with its dimension along axis set to 1 and broadcast over the other dimensions of w so per-channel or per-trace shifts can be applied in a single call.
        - The function preserves array shapes (except that frequency-domain arrays have rFFT length along the axis) and attempts to preserve dtype for time-domain outputs.
    
    Failure modes and errors:
        - If axis is outside the valid range for w, an IndexError will be raised.
        - If s has a shape incompatible with w for the required reshape/broadcast, a ValueError will be raised by numpy broadcasting or reshape operations.
        - If w is a complex array that is not a valid rFFT result for the given ns, the inverse transform (if requested externally) may produce incorrect time-domain signals; therefore, supply ns when passing frequency-domain inputs.
        - Non-numeric or non-array inputs for w or s can raise TypeError or other numpy/scipy-related exceptions.
    
    Returns:
        numpy.ndarray: The shifted signal. If the input w was a real time-domain array, the return is a real numpy.ndarray of the same shape and dtype as w containing the time-domain signal shifted by s samples (supports sub-sample shifts via phase manipulation in the Fourier domain). If the input w was a frequency-domain rFFT array (complex), the return is a complex numpy.ndarray representing the frequency-domain data after the phase shift; in that case, call scipy.fft.irfft with the appropriate ns to obtain time-domain samples.
    """
    from spikeinterface.preprocessing.phase_shift import apply_fshift_ibl
    return apply_fshift_ibl(w, s, axis, ns)


################################################################################
# Source: spikeinterface.preprocessing.pipeline.get_preprocessing_dict_from_analyzer
# File: spikeinterface/preprocessing/pipeline.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_preprocessing_pipeline_get_preprocessing_dict_from_analyzer(
    analyzer_folder: str,
    format: str = "auto",
    backend_options: dict = None
):
    """spikeinterface.preprocessing.pipeline.get_preprocessing_dict_from_analyzer generates a preprocessing dictionary from a saved analyzer folder so the dictionary can be passed to PreprocessingPipeline to recreate the same preprocessing steps used when the analyzer was created. This function is used within the SpikeInterface framework to extract preprocessing configuration (e.g., filters, referencing, channel selection) that was stored by an Analyzer or the newer SortingAnalyzer during preprocessing and recording preparation, enabling reproducible downstream processing and consistent post-processing or quality-metric computation.
    
    Args:
        analyzer_folder (str or Path): Path to the saved analyzer folder on disk or a remote path accepted by is_path_remote. In the SpikeInterface workflow this folder is produced by saving an Analyzer/SortingAnalyzer and contains metadata about the original recording and preprocessing. If the path is local (is_path_remote returns False) the function will convert the value to a pathlib.Path for local filesystem operations. If the path is remote, it is passed unchanged to remote-aware open routines (for example, super_zarr_open for zarr storage).
        format ("auto" | "binary_folder" | "zarr"): The format of the saved analyzer. If "auto" (default), the function infers the format by checking whether the analyzer_folder string ends with ".zarr" and selects "zarr" in that case, otherwise it selects "binary_folder". If "binary_folder", the function searches the folder for a file matching "*recording.*" and extracts the preprocessing dict from that file using get_preprocessing_dict_from_file. If "zarr", the function opens the zarr store (using super_zarr_open) and attempts to read the "recording" field to build the preprocessing dict. This parameter must be one of the three specified literal values; passing any other string will lead to a runtime error because the function does not handle additional format values.
        backend_options (dict | None): Backend options passed only when format == "zarr". Default is None. When None the function uses an empty dict; otherwise it reads backend_options.get("storage_options", {}) and forwards those storage options to super_zarr_open to control remote storage access (for example, credentials or s3 configuration). This argument has no effect for "binary_folder" format.
    
    Returns:
        preprocessing_dict (dict): A dictionary representing the preprocessing pipeline reconstructed from the analyzer's recorded metadata. For "binary_folder" this is produced by parsing the first file matching "*recording.*" in the analyzer_folder via get_preprocessing_dict_from_file. For "zarr" this is produced by reading the "recording" entry from the zarr root (if present) and converting it with _make_pipeline_dict_from_recording_dict; if no "recording" field is present an empty recording dict is used and the resulting preprocessing_dict may be empty or minimal. The returned dict is intended to be passed directly to PreprocessingPipeline to instantiate the same preprocessing sequence used when the analyzer was saved.
    
    Behavior, side effects, defaults, and failure modes:
        - The function will call is_path_remote(analyzer_folder). If the path is not remote it will convert analyzer_folder to pathlib.Path for local file operations; remote paths are left as-is for use with remote-aware backends.
        - When format == "auto", the function infers "zarr" when analyzer_folder string ends with ".zarr"; otherwise it infers "binary_folder".
        - For format == "binary_folder", the function lists files in the analyzer_folder and expects at least one file matching the glob "*recording.*". If no such file is found a FileNotFoundError is raised with a message indicating the missing recording file in the analyzer folder.
        - For format == "zarr", backend_options is set to {} when None. The function extracts storage_options = backend_options.get("storage_options", {}) and passes them to super_zarr_open(str(analyzer_folder), mode="r", storage_options=storage_options). Errors opening the zarr store (e.g., network/authentication/storage misconfiguration) will propagate from super_zarr_open and are not caught by this function.
        - If the zarr root contains a "recording" field, the function reads the first element (rec_field[0]) as the recording metadata; if that field is missing an empty recording dictionary is used and the produced preprocessing_dict will reflect that (typically resulting in an empty or default pipeline).
        - The function only supports the three documented format values; providing an unsupported format value will lead to undefined/erroneous behavior because preprocessing_dict will not be set before return (this manifests as a runtime error). Users should pass one of "auto", "binary_folder", or "zarr".
        - No files are written by this function; it only reads metadata from the analyzer folder or zarr store and returns an in-memory dict suitable for recreating the preprocessing pipeline.
    """
    from spikeinterface.preprocessing.pipeline import get_preprocessing_dict_from_analyzer
    return get_preprocessing_dict_from_analyzer(
        analyzer_folder,
        format,
        backend_options
    )


################################################################################
# Source: spikeinterface.preprocessing.pipeline.get_preprocessing_dict_from_file
# File: spikeinterface/preprocessing/pipeline.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_preprocessing_pipeline_get_preprocessing_dict_from_file(
    recording_dictionary_path: str
):
    """spikeinterface.preprocessing.pipeline.get_preprocessing_dict_from_file generates a preprocessing dictionary from a saved recording descriptor file so it can be passed to apply_preprocessing_pipeline or used to construct a PreprocessPipeline. This function is part of the SpikeInterface preprocessing utilities for spike-sorting workflows; it reads a recording dictionary previously saved to disk (by SpikeInterface recording save routines) and extracts only the preprocessing steps that can be applied globally to any recording (it deliberately excludes per-channel or per-frame operations such as ChannelSlice and FrameSlice). The returned dictionary maps preprocessing step names to their keyword-argument dictionaries and is suitable for automated reproducibility of preprocessing pipelines or for re-applying the same global preprocessing to other recordings.
    
    Args:
        recording_dictionary_path (str or Path): Path to the saved recording descriptor file produced by SpikeInterface recording save functionality. The function accepts a filesystem path to a file ending with ".json" (expected to contain a JSON-serialized recording dictionary) or ".pkl" / ".pickle" (expected to contain a Python pickle of the recording dictionary). The recording dictionary is the serialized metadata and preprocessing record for a recording; providing this file allows reconstruction of the global preprocessing pipeline used when the recording was saved.
    
    Returns:
        pipeline_dict (dict): Dictionary containing preprocessing steps and their kwargs extracted from the saved recording dictionary. The keys are preprocessing step identifiers and the values are dictionaries of keyword arguments for each step. This dictionary is formatted so it can be passed directly to spikeinterface.preprocessing.pipeline.apply_preprocessing_pipeline and to the PreprocessPipeline class to re-create the same global preprocessing sequence.
    
    Behavior and side effects:
        The function will open and read the file specified by recording_dictionary_path. If the path ends with ".json" the file is parsed with the json module; if it ends with ".pkl" or ".pickle" the file is read with the pickle module. After reading, the function calls an internal helper (_make_pipeline_dict_from_recording_dict) to construct the preprocessing dictionary from the loaded recording dictionary. No modifications are written back to disk by this function; its only side effect is reading the input file.
    
    Failure modes and exceptions:
        If recording_dictionary_path does not point to an existing file, a FileNotFoundError will be raised when attempting to open it. If the file ends with ".json" but contains invalid JSON, a json.JSONDecodeError (or subclass) will be raised. If the file is a pickle but is not a valid pickle for the expected recording dictionary, pickle.UnpicklingError or other exceptions from the pickle module may be raised. If the file extension is not one of ".json", ".pkl", or ".pickle", the function will not read the file and a subsequent error will occur when attempting to build the pipeline (for example, an UnboundLocalError or NameError because the variable holding the recording dictionary will be undefined); callers should ensure the input file uses one of the supported extensions. The function assumes the recording dictionary was produced by SpikeInterface save routines and contains the expected structure required by _make_pipeline_dict_from_recording_dict; if that structure is absent or malformed, the helper may raise KeyError or other exceptions while constructing the pipeline dictionary.
    """
    from spikeinterface.preprocessing.pipeline import get_preprocessing_dict_from_file
    return get_preprocessing_dict_from_file(recording_dictionary_path)


################################################################################
# Source: spikeinterface.preprocessing.preprocessing_tools.get_kriging_channel_weights
# File: spikeinterface/preprocessing/preprocessing_tools.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_preprocessing_preprocessing_tools_get_kriging_channel_weights(
    contact_positions1: numpy.ndarray,
    contact_positions2: numpy.ndarray,
    sigma_um: float,
    p: float,
    weight_threshold: float = 0.005
):
    """Calculate kriging interpolation weights between two sets of recording contacts. This function computes a kernel-based weight matrix that maps values on a source set of contacts (contact_positions1) to a target set of contacts (contact_positions2) using a spatial kernel controlled by sigma_um and p. Very small weights (below weight_threshold) are zeroed to improve numerical stability and sparsity. The implementation delegates kernel computation to get_kriging_kernel_distance and then normalizes weights so that, for each target contact (each column), the weights sum to 1 when possible. This function is used in extracellular preprocessing and bad-channel interpolation workflows in spike sorting pipelines (for example, the International Brain Laboratory pipeline referenced in the source) to estimate channel contributions from nearby contacts when reconstructing or interpolating signals.
    
    Args:
        contact_positions1 (numpy.ndarray): Array of spatial coordinates for the source contacts (the contacts providing values to be interpolated). The array must be a numeric numpy.ndarray (e.g., float) containing coordinate values in the same spatial units as sigma_um. The practical role is to identify which recorded channels contribute to interpolation; each row corresponds to one source contact.
        contact_positions2 (numpy.ndarray): Array of spatial coordinates for the target contacts (the contacts where interpolated values are needed). This is a numeric numpy.ndarray with the same coordinate units as contact_positions1 and sigma_um. Each row corresponds to one target contact. The function computes weights that map from contact_positions1 to contact_positions2.
        sigma_um (float): Kernel spatial scale parameter (in the same spatial units as contact_positions arrays; the name indicates micrometers). It controls the spatial decay of the kriging kernel: larger sigma_um produces broader spatial influence of source contacts. In spike sorting preprocessing, sigma_um sets the locality over which nearby channels influence an interpolated channel.
        p (float): Power/exponent parameter passed to the kriging kernel distance computation. This parameter modifies the shape of the kernel (for example, a power-law or exponent applied inside the kernel function implemented by get_kriging_kernel_distance). The practical significance is tuning how rapidly kernel weights decay with distance.
        weight_threshold (float): Threshold for small weights. Default is 0.005. Any computed weight strictly below this threshold is set to 0 to enforce sparsity and numerical stability. After initial thresholding, weights are column-normalized; entries that remain below the threshold or become NaN during normalization are set to 0.
    
    Returns:
        numpy.ndarray: A 2-D array of kriging weights. Columns correspond to target contacts (contact_positions2) and rows correspond to source contacts (contact_positions1); i.e., each column contains the weights to combine source-contact values to estimate the value at one target contact. Behavior notes: weights are computed by get_kriging_kernel_distance(contact_positions1, contact_positions2, sigma_um, p), then any value < weight_threshold is set to 0, then the matrix is normalized so each column sums to 1 where possible. Division-by-zero or invalid-value warnings during normalization are suppressed; any resulting NaNs or values below weight_threshold after normalization are set to 0. If all contributions to a target contact are below threshold (or effectively zero after thresholding), that column will be all zeros (no interpolation contribution). This function has no side effects on its inputs.
    """
    from spikeinterface.preprocessing.preprocessing_tools import get_kriging_channel_weights
    return get_kriging_channel_weights(
        contact_positions1,
        contact_positions2,
        sigma_um,
        p,
        weight_threshold
    )


################################################################################
# Source: spikeinterface.preprocessing.preprocessing_tools.get_kriging_kernel_distance
# File: spikeinterface/preprocessing/preprocessing_tools.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_preprocessing_preprocessing_tools_get_kriging_kernel_distance(
    locations_1: numpy.ndarray,
    locations_2: numpy.ndarray,
    sigma_um: list,
    p: float,
    distance_metric: str = "euclidean"
):
    """Get the kriging kernel between two sets of spatial channel locations for use in spike-sorting preprocessing.
    
    Args:
        locations_1 (numpy.ndarray): 2D array of shape (N1, D) giving spatial coordinates of the first set of channels/contacts. N1 is the number of channels in the first set and D is the spatial dimensionality (e.g., 2 for [x, y]). In the SpikeInterface domain this represents channel/contact positions (typically in micrometers) used to compute how much one channel influences another for kriging-style spatial interpolation or channel-based weighting in preprocessing.
        locations_2 (numpy.ndarray): 2D array of shape (N2, D) giving spatial coordinates of the second set of channels/contacts. N2 is the number of channels in the second set and D must match the dimensionality of locations_1. The returned kernel is evaluated pairwise between each location in locations_1 and each location in locations_2.
        sigma_um (float or list): Scale parameter(s) for the Gaussian kernel, typically given in micrometers and representing the spatial decay length of channel influence. If a scalar float is provided, a single isotropic scale is used and the function computes pairwise distances with scipy.spatial.distance.cdist using distance_metric and then applies kernal = exp(-(dist / sigma_um) ** p). If a list is provided, it must contain two elements (sigma_x, sigma_y) to mimic Kilosort2.5 behavior where separate scales are applied per spatial dimension; in this branch the code explicitly computes absolute differences along the first two coordinates and applies kernal = exp(- (|dx|/sigma_x)**p - (|dy|/sigma_y)**p). When sigma_um is a list the function forces a cityblock-like separable treatment of dimensions and ignores the distance_metric argument.
        p (float): Exponent applied to the normalized distance in the kernel: kernel = exp(- (distance / sigma) ** p) (or the separable per-dimension equivalent when sigma_um is a list). In practice p controls how sharply the kernel decays with distance (common choices are p=2 for Gaussian decay). For typical, physically meaningful kernels p should be positive; nonpositive values may produce non-decaying or numerically unstable results.
        distance_metric (str, optional): Metric name passed to scipy.spatial.distance.cdist when sigma_um is a scalar. Default is "euclidean". This parameter is ignored when sigma_um is a list (the list case uses per-dimension absolute differences and does not call cdist).
    
    Returns:
        numpy.ndarray: A 2D array of shape (N1, N2) (locations_1 x locations_2) containing the kriging kernel values (Gaussian-style kernel) between each pair of locations. Each element is computed as exp(-((dist / sigma) ** p)) for the scalar-sigma case or exp(-((|dx|/sigma_x)**p) - ((|dy|/sigma_y)**p)) for the two-sigma list case. Under the common assumptions sigma_um > 0 and p > 0 the returned values lie in (0, 1], with 1 on the diagonal for zero distances and values tending toward 0 for large separations.
    
    Behavior, side effects, and failure modes:
        This function is pure (no side effects or in-place mutation of inputs). When sigma_um is scalar the function imports scipy inside the branch and calls scipy.spatial.distance.cdist; if scipy is not available an ImportError will be raised. If sigma_um is a list, the function expects at least two spatial dimensions (D >= 2) and a two-element list (sigma_x, sigma_y); otherwise an IndexError or ValueError may occur. Mismatched input shapes (locations_1 and locations_2 must be 2D and have the same number of columns D) will raise errors from NumPy or scipy distance utilities. The function assumes numeric coordinate arrays; non-numeric dtypes can raise TypeError during distance computation. The function preserves the numeric dtype semantics of NumPy operations for the returned array.
    """
    from spikeinterface.preprocessing.preprocessing_tools import get_kriging_kernel_distance
    return get_kriging_kernel_distance(
        locations_1,
        locations_2,
        sigma_um,
        p,
        distance_metric
    )


################################################################################
# Source: spikeinterface.preprocessing.preprocessing_tools.get_spatial_interpolation_kernel
# File: spikeinterface/preprocessing/preprocessing_tools.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_preprocessing_preprocessing_tools_get_spatial_interpolation_kernel(
    source_location: numpy.ndarray,
    target_location: numpy.ndarray,
    method: str = "kriging",
    sigma_um: float = 20.0,
    p: int = 1,
    num_closest: int = 4,
    sparse_thresh: float = None,
    dtype: str = "float32",
    force_extrapolate: bool = False
):
    """Compute the spatial interpolation kernel used for linear spatial interpolation of extracellular recordings.
    
    This function is used in SpikeInterface preprocessing to interpolate signals from a set of source electrode/contact locations to a set of target locations. Typical use cases in the SpikeInterface framework include replacing bad channels by spatial interpolation and correcting drift by interpolating between contacts. The function implements three methods: "kriging" (the default, matching the approach used in kilosort and an adaptation of pykilosort), "idw" (inverse distance weighting), and "nearest" (assign the nearest source). The returned kernel is a weight matrix that maps source-channel signals to target locations: output_signal = interpolation_kernel.T @ source_signals (or equivalently each column of the kernel contains weights for a target location). The function is pure (no side effects on inputs) and performs numerical regularization for kriging; it may produce warnings or NaNs if extreme inputs (for example all-zero weights for a target) occur.
    
    Args:
        source_location (numpy.ndarray): Array of shape (m, 2) giving the 2D coordinates (typically in micrometers, x and y) of m source recording contacts or channels. In SpikeInterface this represents the original electrode/contact layout whose signals will be used as inputs for interpolation. The function uses source_location to compute distances and to determine the bounding box used for deciding which target points are "inside" the convex axis-aligned range of the sources.
        target_location (numpy.ndarray): Array of shape (n, 2) giving the 2D coordinates (typically in micrometers, x and y) of n desired target locations where interpolated signals are needed. In SpikeInterface this represents positions of virtual contacts, corrected contact positions after drift estimation, or locations of bad channels to replace. The function computes weights mapping source channels to these target positions.
        method (str): Choice of interpolation method. Allowed values are "kriging", "idw", and "nearest". Default is "kriging". "kriging" implements the kernel-based approach used in kilosort (it builds source-source and target-source covariance-like kernels and applies a pseudoinverse with small regularization), "idw" performs inverse-distance-weighted interpolation using a limited number of closest sources, and "nearest" assigns weight 1 to the single closest source for each target. A ValueError is raised if an unsupported method string is provided.
        sigma_um (float or list): Parameter used by the "kriging" kernel distance function to scale spatial covariance in micrometers. When a single float is provided it is used isotropically for x and y; when a list is provided it must have 2 elements (separate scales for x and y). In SpikeInterface kriging this parameter controls the spatial length scale of interpolation; it is forwarded to get_kriging_kernel_distance. The default is 20.0.
        p (int): Exponent parameter used in the kriging distance/kernel formula. In the kriging implementation derived from pykilosort this integer controls the functional form of the kernel; keep the default of 1 unless reproducing a specific pipeline. Used only by the "kriging" method.
        num_closest (int): Number of closest source channels to consider for each target when method == "idw" (inverse distance weighting). For each target the algorithm selects the num_closest source locations, assigns weights proportional to 1/distance, and leaves all other source weights zero. Default is 4.
        sparse_thresh (None or float): When not None and method == "kriging", values in the computed kriging interpolation kernel below this threshold are set to zero to produce a sparse kernel. After thresholding the function renormalizes columns corresponding to target points considered "inside" the axis-aligned bounding box of the sources so that their column sums equal 1. Default is None (no sparsification).
        dtype (str): String name of the numpy dtype to use for the returned interpolation kernel (for example "float32"). The final returned array is cast to this dtype. Default is "float32".
        force_extrapolate (bool): Controls handling of target locations that fall outside the axis-aligned bounding box of source_location. If False (default) any target location outside the bounding box is forced to have all-zero weights (no extrapolation). If True, extrapolation is performed using the chosen method's formula for those targets; in that case column sums are not forced to 1 for outside targets and the kernel may contain values outside the normal convex-combination range.
    
    Returns:
        numpy.ndarray: interpolation_kernel of shape (m, n) and dtype given by the dtype parameter. Each column j contains weights that map the m source channels to target_location[j]. For targets determined to be inside the axis-aligned bounding box of the sources and when force_extrapolate is False, columns are normalized so their sum is 1 (for "nearest" columns contain a single 1). For "kriging" the implementation computes Kyx @ pinv(Kxx + 1e-6 * I) (with a 1e-6 diagonal regularization) and then transposes to shape (m, n). For "idw" weights are nonzero only for up to num_closest sources per target and are proportional to 1/distance, with exact-zero handling when a source coincides with a target (that source receives weight 1). For "nearest" each column has a single 1 at the index of the closest source. The function may raise a ValueError for an invalid method string. Note that numerical issues can occur if a column sums to zero (for example after sparsification), which can produce divisions by zero and lead to inf/NaN values or runtime warnings; the kriging path uses a pseudoinverse to mitigate singular Kxx but does not explicitly guard against zero-sum columns after thresholding.
    """
    from spikeinterface.preprocessing.preprocessing_tools import get_spatial_interpolation_kernel
    return get_spatial_interpolation_kernel(
        source_location,
        target_location,
        method,
        sigma_um,
        p,
        num_closest,
        sparse_thresh,
        dtype,
        force_extrapolate
    )


################################################################################
# Source: spikeinterface.preprocessing.whiten.compute_sklearn_covariance_matrix
# File: spikeinterface/preprocessing/whiten.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_preprocessing_whiten_compute_sklearn_covariance_matrix(
    data: numpy.ndarray,
    regularize_kwargs: dict
):
    """Compute the covariance matrix using a scikit-learn covariance estimator for preprocessing (whitening) of extracellular recordings in the SpikeInterface spike-sorting workflow.
    
    This function is used during preprocessing/whitening of extracellular electrophysiology data (as performed in SpikeInterface) to estimate the channels/features covariance matrix required to decorrelate or whiten the data prior to spike detection and sorting. It delegates estimation to a class from sklearn.covariance specified by the "method" entry in regularize_kwargs, forces centered-data behavior, and fits the estimator on the provided data after converting it to float64 (required by scikit-learn). Note that scikit-learn covariance implementations that are provided as standalone functions (rather than classes in sklearn.covariance) are not supported by this wrapper.
    
    Args:
        data (numpy.ndarray): Input data array to estimate covariance from. In the SpikeInterface/whitening context this represents recorded extracellular signals arranged so that rows correspond to observations/samples and columns correspond to channels or features. The array will be converted to dtype float64 before fitting because scikit-learn covariance estimators require float64 input; this conversion can increase memory usage for large arrays.
        regularize_kwargs (dict): Keyword arguments passed to the chosen scikit-learn covariance estimator class. This dictionary must contain a "method" key whose value is the name (string) of a class in sklearn.covariance (for example, "LedoitWolf" or "OAS"). The function will mutate this dict: it pops the "method" entry and then sets "assume_centered" to True inside the dict before instantiating the estimator. If "assume_centered" is present in regularize_kwargs and is explicitly False, the function raises a ValueError because the wrapper forces centered-data behavior required for whitening in this preprocessing pipeline. Any remaining entries in regularize_kwargs are forwarded as keyword arguments to the sklearn covariance estimator constructor; if they are invalid for the chosen estimator, the estimator construction or fit will raise the corresponding scikit-learn error.
    
    Returns:
        numpy.ndarray: The estimated covariance matrix returned by the scikit-learn estimator (estimator.covariance_). In the context of SpikeInterface preprocessing, this is the channel/feature covariance matrix used for whitening. The returned array is a NumPy array with dtype float64 as produced by scikit-learn.
    
    Raises:
        ValueError: If regularize_kwargs contains "assume_centered" set to False (the function requires or forces assume_centered=True).
        AttributeError: If the value of regularize_kwargs["method"] does not correspond to a class attribute in sklearn.covariance.
        TypeError, ValueError, or sklearn-specific exceptions: If provided regularize_kwargs entries are invalid for the chosen sklearn estimator or if estimator.fit fails on the provided data (for example due to incompatible shapes).
    """
    from spikeinterface.preprocessing.whiten import compute_sklearn_covariance_matrix
    return compute_sklearn_covariance_matrix(data, regularize_kwargs)


################################################################################
# Source: spikeinterface.preprocessing.whiten.compute_whitening_from_covariance
# File: spikeinterface/preprocessing/whiten.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_preprocessing_whiten_compute_whitening_from_covariance(
    cov: numpy.ndarray,
    eps: float
):
    """spikeinterface.preprocessing.whiten.compute_whitening_from_covariance computes the ZCA whitening matrix from a provided covariance matrix for use in preprocessing extracellular recordings (for example, to decorrelate channels prior to spike detection and sorting). The function performs a singular value decomposition (SVD) of the covariance matrix and builds a whitening transform that regularizes small or zero eigenvalues using the supplied eps parameter to ensure numerical stability.
    
    Args:
        cov (numpy.ndarray): The covariance matrix from which to compute the whitening matrix. In the spike sorting preprocessing context, this is typically the empirical covariance of recorded channels or features derived from an extracellular recording. The array is expected to represent a square, symmetric, positive semi-definite covariance matrix (shape compatible with a channel-by-channel covariance). If cov contains non-finite values (NaN or Inf) or is not a valid covariance matrix, the function may fail during SVD.
        eps (float): A non-negative regularization constant added to the eigenvalues (diagonal values from the SVD) before inversion. Practically, eps prevents division-by-zero or amplification of numerical noise when eigenvalues are zero or very small, and therefore controls numerical stability of the whitening. Use a small positive value (for example, 1e-6) in typical usage; supplying a negative eps can lead to invalid (complex) results or runtime warnings.
    
    Returns:
        numpy.ndarray: The computed whitening matrix W of the same shape as cov that implements ZCA whitening via W = U @ diag(1 / sqrt(S + eps)) @ Ut where U, S, Ut are from the SVD of cov. In the spike sorting preprocessing pipeline, applying this matrix to data (for example, X_whitened = W @ X) approximately decorrelates channels and scales variances so that W @ cov @ W.T is close to the identity matrix. The function has no side effects and returns the whitening matrix as a new numpy.ndarray.
    
    Notes on behavior and failure modes:
        The implementation uses numpy.linalg.svd(cov, full_matrices=True) to obtain U, S, Ut. If cov contains NaNs/Infs or is otherwise invalid, numpy.linalg.svd may raise a numpy.linalg.LinAlgError or propagate invalid values. The eps parameter must be chosen to balance numerical stability and preservation of signal structure; overly large eps will oversmooth and reduce the effect of whitening, while eps == 0 will allow division by zero when eigenvalues are exactly zero. The function is deterministic for a given cov and eps and introduces no in-place modification to inputs.
    """
    from spikeinterface.preprocessing.whiten import compute_whitening_from_covariance
    return compute_whitening_from_covariance(cov, eps)


################################################################################
# Source: spikeinterface.qualitymetrics.misc_metrics.amplitude_cutoff
# File: spikeinterface/qualitymetrics/misc_metrics.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_qualitymetrics_misc_metrics_amplitude_cutoff(
    amplitudes: numpy.ndarray,
    num_histogram_bins: int = 500,
    histogram_smoothing_value: int = 3,
    amplitudes_bins_min_ratio: int = 5
):
    """spikeinterface.qualitymetrics.misc_metrics.amplitude_cutoff: Compute an approximate fraction of spikes that are missing for a single unit by analyzing the empirical distribution of spike amplitudes. This function is used in spike sorting quality assessment to estimate how many low-amplitude spikes may have been lost because they fall below the detection threshold; it is intended to be used on the amplitudes (in microvolts) of all detected spikes for one unit and is referenced by higher-level helpers such as compute_amplitude_cutoffs.
    
    Args:
        amplitudes (numpy.ndarray): 1D array of spike amplitudes for a single unit, expressed in microvolts (uV). These are the measured peak or trough amplitudes used to characterize the amplitude distribution of the unit. The function treats this array as the empirical sample from which to build a probability density function (PDF) via a histogram; amplitudes must therefore contain at least one value and should represent the same quantity (same sign convention and units) used across units in the recording/analysis pipeline.
        num_histogram_bins (int): The number of bins to use when constructing the amplitude histogram (passed to numpy.histogram). A larger number gives finer resolution of the empirical PDF but requires more samples; default is 500. This parameter controls the discretization of the amplitude axis used to estimate the PDF and therefore influences sensitivity to small-scale features in the distribution.
        histogram_smoothing_value (int): Controls the amount of smoothing applied to the raw histogram to estimate a continuous PDF. The value is passed as the sigma parameter to scipy.ndimage.gaussian_filter1d and defaults to 3. Higher values produce a smoother PDF that reduces sensitivity to high-frequency noise in the histogram, while lower values retain more fine detail; choose based on expected sample size and noise.
        amplitudes_bins_min_ratio (int): Minimum required ratio between the number of amplitude samples (len(amplitudes)) and num_histogram_bins. If len(amplitudes) / num_histogram_bins is less than this integer threshold, the function will not attempt a robust histogram-based estimation and will return numpy.nan. Default is 5. This guard prevents unreliable estimates when the per-bin sample count would be too small.
    
    Behavior and algorithmic details:
        The function first checks the sample-size-to-bin count ratio and returns numpy.nan immediately if there are insufficient amplitude samples relative to num_histogram_bins (see amplitudes_bins_min_ratio). If sufficient samples are available, the function computes a density histogram of amplitudes using numpy.histogram(..., density=True) to obtain a raw PDF estimate. It then smooths that PDF with scipy.ndimage.gaussian_filter1d using histogram_smoothing_value as the smoothing parameter. The support used for bin positions is the left edges of the histogram bins and the mean bin width is used as the bin size (in uV). The algorithm locates the global maximum of the smoothed PDF (the main mode) and then searches for the first minimum of the PDF on the higher-amplitude side relative to that peak. The estimated fraction of missing spikes is computed as the integral (sum over bins multiplied by bin size) of the smoothed PDF for amplitudes larger than that minimum; this integral represents the probability mass to the right of the cut point and is interpreted as the fraction of spikes above the cutoff that are observed, so the complementary interpretation here yields the estimated missing fraction. The returned value is clipped to a maximum of 0.5 to reflect the method's intended operating region.
    
    Side effects and warnings:
        The function performs an on-demand import of scipy.ndimage.gaussian_filter1d; this requires SciPy to be installed in the Python environment. If the PDF analysis finds that the absolute-difference vector used to locate the minimum contains a non-unique minimum (multiple equal minimal values), the function issues a warnings.warn message indicating that the amplitude PDF does not have a unique minimum and that more spikes may be required for a stable amplitude_cutoff estimate. No files or external state are modified.
    
    Defaults and failure modes:
        If the sample-to-bin ratio check fails (len(amplitudes) / num_histogram_bins < amplitudes_bins_min_ratio) the function returns numpy.nan to indicate that a reliable estimate cannot be produced with the provided data. If SciPy is not available, an ImportError will propagate when the function attempts to import gaussian_filter1d. The estimate may be inaccurate if the amplitude distribution is multimodal, heavily skewed, or has insufficient samples even if the ratio threshold is met; in ambiguous cases a warning is emitted but a numeric value is still returned (subject to the clipping to 0.5).
    
    Returns:
        float: Estimated fraction of missing spikes for the provided unit based on the amplitude distribution. The value is a floating-point number in which numpy.nan indicates that the estimate could not be computed due to insufficient data (per amplitudes_bins_min_ratio). Valid numeric estimates are capped at 0.5. This returned fraction is meant to be interpreted in the spike-sorting quality-control context as the approximate proportion of spikes not detected because their amplitudes fell below the effective detection threshold inferred from the empirical amplitude PDF.
    """
    from spikeinterface.qualitymetrics.misc_metrics import amplitude_cutoff
    return amplitude_cutoff(
        amplitudes,
        num_histogram_bins,
        histogram_smoothing_value,
        amplitudes_bins_min_ratio
    )


################################################################################
# Source: spikeinterface.qualitymetrics.misc_metrics.isi_violations
# File: spikeinterface/qualitymetrics/misc_metrics.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_qualitymetrics_misc_metrics_isi_violations(
    spike_trains: list,
    total_duration_s: float,
    isi_threshold_s: float = 0.0015,
    min_isi_s: float = 0
):
    """Calculate Inter-Spike Interval (ISI) violations for a single sorted unit across one or more recording segments.
    
    This function is used in SpikeInterface quality metrics to quantify refractory-period violations that indicate contamination or mis-sorting of a unit. It counts adjacent spike pairs whose inter-spike interval (ISI) is below a biophysical threshold (isi_threshold_s), aggregates counts across provided recording segments, and computes two summary metrics: (1) isi_violations_ratio, the ratio of the violation rate to the overall firing rate (dimensionless fraction); and (2) isi_violations_rate, the absolute rate of violating spikes in spikes per second. The algorithm follows the implementation documented in compute_isi_violations: for each segment the function computes ISIs via successive differences, sums ISIs < isi_threshold_s to get num_violations, and computes violation_time = 2 * num_spikes * (isi_threshold_s - min_isi_s) to normalize the violation rate.
    
    Args:
        spike_trains (list): The spike times for each recording segment for a single unit, in seconds. Each element is expected to be a 1-D numpy.ndarray (or array-like) of spike times sorted in ascending order for that segment. These per-segment arrays correspond to contiguous recording intervals or trials; the function concatenates their contributions by summing counts and lengths. Supplying unsorted times or non-numeric values will lead to incorrect ISI calculations or runtime errors.
        total_duration_s (float): The total duration of the recording, in seconds. This scalar is used to compute the overall firing rate (num_spikes / total_duration_s) and to convert the total number of violating spikes into an absolute violating-spike rate. Must be positive; passing zero or negative values will lead to division-by-zero or invalid rates.
        isi_threshold_s (float): Threshold for classifying adjacent spikes as an ISI violation, in seconds. This represents the biophysical refractory period used to identify contaminating spikes and defaults to 0.0015 (1.5 ms). Typical use is to set this to a conservative refractory-time estimate below which two spikes cannot physiologically originate from the same neuron.
        min_isi_s (float): Minimum possible inter-spike interval, in seconds, representing any artificial refractory period enforced by acquisition hardware or post-processing (default 0). This value is subtracted from isi_threshold_s when computing the available violation window; it must be less than isi_threshold_s to yield a positive violation_time.
    
    Behavior, defaults, side effects, and failure modes:
        The function computes ISIs per segment using numpy.diff on each spike_trains element, increments num_spikes by the number of spikes in each segment, and counts violations where ISI < isi_threshold_s. It then computes violation_time = 2 * num_spikes * (isi_threshold_s - min_isi_s), the total time window in which a violating spike could occur given the thresholds. If num_spikes > 0, the function computes total_rate = num_spikes / total_duration_s and violation_rate = num_violations / violation_time, then returns isi_violations_ratio = violation_rate / total_rate, isi_violations_rate = num_violations / total_duration_s, and isi_violations_count = num_violations.
        Default values: isi_threshold_s defaults to 0.0015 s and min_isi_s defaults to 0 s as in common spike-sorting practice; these defaults are applied when arguments are omitted.
        If num_spikes == 0 (no detected spikes across all segments), the function returns NaN for the ratio and rate values (consistent with inability to estimate firing/violation rates) and NaN for the count as implemented. If total_duration_s <= 0 or isi_threshold_s <= min_isi_s (which makes violation_time non-positive), the function may raise a division-by-zero error or return infinite/invalid values; callers should validate inputs to avoid these conditions. The function does not modify its inputs (pure calculation, no side effects like I/O or in-place mutation).
        Practical significance: isi_violations_ratio is a dimensionless contamination metric used in unit curation and benchmarking (higher values indicate greater refractory-period violations and likely contamination). isi_violations_rate gives the absolute rate of violating spikes (spikes per second) and isi_violations_count gives the raw number of violating ISI pairs detected across all segments; users often threshold these metrics when accepting or rejecting sorted units.
    
    Returns:
        tuple: Three values describing ISI violations for the input spike_trains.
            isi_violations_ratio (float): Dimensionless ratio of the violation rate to the overall firing rate (violation_rate / total_rate). Used as a normalized contamination metric in spike sorting quality assessment.
            isi_violations_rate (float): Absolute rate of violating spikes in spikes per second (num_violations / total_duration_s). Higher values indicate more contaminating events per unit time.
            isi_violations_count (int): Integer count of ISI violations detected across all segments (number of adjacent spike pairs with ISI < isi_threshold_s).
    
    References and usage note:
        This function is part of SpikeInterface quality metrics used to validate and curate spike sorting outputs. The computed metrics are commonly compared against thresholds during automated or manual curation to identify units with excessive refractory-period violations. See compute_isi_violations documentation within the codebase for additional theoretical background and references.
    """
    from spikeinterface.qualitymetrics.misc_metrics import isi_violations
    return isi_violations(spike_trains, total_duration_s, isi_threshold_s, min_isi_s)


################################################################################
# Source: spikeinterface.qualitymetrics.misc_metrics.presence_ratio
# File: spikeinterface/qualitymetrics/misc_metrics.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_qualitymetrics_misc_metrics_presence_ratio(
    spike_train: numpy.ndarray,
    total_length: int,
    bin_edges: numpy.ndarray = None,
    num_bin_edges: int = None,
    bin_n_spikes_thres: int = 0
):
    """Calculate the presence ratio for a single unit across a recording by dividing the number of temporal bins in which the unit is "active" by the total number of temporal bins. This metric is used in spike sorting quality assessment (see SpikeInterface quality metrics) to quantify how consistently a sorted unit is present throughout the recording: values close to 1 indicate the unit fires across most of the recording, values close to 0 indicate the unit is only present in a small fraction of the recording.
    
    Args:
        spike_train (numpy.ndarray): 1-D array of spike times for this unit, expressed in samples. These are the event times used to assign spikes to temporal bins; they do not need to be sorted. In the spike-sorting domain this represents the sample indices of detected spikes for a single unit produced by a sorter or post-processing step.
        total_length (int): Total length of the recording in samples. This parameter documents the expected recording duration (in samples) for the spike_train and is part of the function API in SpikeInterface quality metrics; note that the current implementation does not use this value internally but callers should supply the recording length in samples for API consistency and future compatibility.
        bin_edges (numpy.ndarray): Optional. An explicit array of bin edge positions (in samples) to use to partition the recording into temporal bins. Mutually exclusive with num_bin_edges. If provided, these edges are passed directly to numpy.histogram and the effective number of histogram bins is len(bin_edges) - 1. Values in spike_train that fall outside the provided edges are not counted in the returned histogram bins (consistent with numpy.histogram behavior). Providing bin_edges allows callers to define uneven or externally computed bin boundaries (for example, to align bins to behavioral epochs).
        num_bin_edges (int): Optional. The number of bin edges to use to compute the presence ratio (mutually exclusive with bin_edges). Interpreted as the number of edges, so the effective number of histogram bins used is num_bin_edges - 1. If bin_edges is not provided, num_bin_edges is passed to numpy.histogram as the bins argument (an integer), which causes numpy to compute an even partitioning across the range of spike times. This parameter controls the temporal resolution of the presence ratio: larger num_bin_edges (hence more bins) yields a finer-grained assessment of presence across time.
        bin_n_spikes_thres (int): Minimum number of spikes required within a bin for that bin to be considered "active" (default: 0). In practice, with the default 0 a bin is considered active if it contains at least one spike; setting this to a larger integer makes the presence ratio stricter by requiring more spikes within a bin to count it as presence. Must be >= 0.
    
    Behavior, side effects, defaults, and failure modes:
        The function computes a histogram of spike counts per temporal bin using numpy.histogram, then computes the fraction of bins whose spike count is strictly greater than bin_n_spikes_thres. The returned presence ratio is the number_of_active_bins / number_of_bins, where number_of_bins is (num_bin_edges - 1) (this is why num_bin_edges should represent the number of edges). The function has no side effects (it does not modify inputs or external state).
        The function asserts that exactly one of bin_edges or num_bin_edges is provided; if both are None an AssertionError is raised with message "Use either bin_edges or num_bin_edges". It also asserts bin_n_spikes_thres >= 0 and will raise an AssertionError if a negative threshold is supplied. If num_bin_edges is <= 1 (which makes the effective number of bins zero or negative), the code will perform a division by zero when computing the ratio and a ZeroDivisionError may be raised; callers must ensure num_bin_edges >= 2 when using integer bin counts.
        When bin_edges is provided as a sequence, spike times outside the provided edge range will not contribute to any bin counts (consistent with numpy.histogram). When bins are specified by an integer (via num_bin_edges), numpy.histogram determines the bin edges from the data range; in this case, the implicit bin edges depend on the spike_train values (if spike_train is empty or constant-valued, numpy.histogram behavior may yield degenerate bins — ensure spike_train has representative times).
        The function expects spike times to be in the same sample units as total_length and bin_edges; mismatched units will produce meaningless results.
    
    Returns:
        presence_ratio (float): Fraction in [0, 1] (under normal valid inputs) representing the fraction of temporal bins in which the unit is active according to bin_n_spikes_thres. This value quantifies the temporal ubiquity of the unit across the recording and is used in SpikeInterface quality metrics to flag units that are transient or consistently present. Note that a ZeroDivisionError can occur if num_bin_edges <= 1; AssertionError can occur if both bin_edges and num_bin_edges are None or if bin_n_spikes_thres < 0.
    """
    from spikeinterface.qualitymetrics.misc_metrics import presence_ratio
    return presence_ratio(
        spike_train,
        total_length,
        bin_edges,
        num_bin_edges,
        bin_n_spikes_thres
    )


################################################################################
# Source: spikeinterface.qualitymetrics.misc_metrics.slidingRP_violations
# File: spikeinterface/qualitymetrics/misc_metrics.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_qualitymetrics_misc_metrics_slidingRP_violations(
    spike_samples: numpy.ndarray,
    sample_rate: float,
    duration: float,
    bin_size_ms: float = 0.25,
    window_size_s: float = 1,
    exclude_ref_period_below_ms: float = 0.5,
    max_ref_period_ms: float = 10,
    contamination_values: numpy.ndarray = None,
    return_conf_matrix: bool = False
):
    """A metric developed in the IBL / SteinmetzLab tradition that estimates the minimum contamination (fraction of non-physiological or mis-assigned spikes) consistent with observed refractory-period violations using a sliding-refractory-period method. This function is part of SpikeInterface quality-metrics utilities for post-processing spike-sorted data: it computes autocorrelograms from spike times, tests a range of candidate refractory-period durations and candidate contamination fractions, and returns the smallest contamination fraction that yields >90% confidence of explaining the observed violations. The method is useful when validating/curating spike sorting results to detect units with excessive refractory-period violations indicative of contamination or merging.
    
    Args:
        spike_samples (numpy.ndarray): Spike times expressed in sample indices. In practice this may be a 1D numpy array of integer sample indices for a single recording segment, or (as supported by the implementation and original documentation) a list/sequence of such arrays for multi-segment recordings; the function concatenates multi-segment spike lists to compute global spike counts and firing rate. Each element is interpreted as the time of a spike measured in samples relative to the start of its segment. Supplying non-integer or out-of-range values will likely cause downstream errors in correlogram computation. The caller is responsible for ensuring spike times are in the same sampling reference as sample_rate and duration.
        sample_rate (float): Acquisition sampling rate in Hz (samples per second). This value is used to convert bin sizes and window sizes specified in seconds or milliseconds into integer sample counts used by the correlogram routine. Passing non-positive or zero sample_rate will lead to invalid conversions and likely runtime errors.
        duration (float): Total recording duration in seconds used to compute the firing rate (n_spikes / duration). For multi-segment inputs this should be the total duration across segments. A duration of zero or a negative value will cause a division error or nonsensical firing-rate estimates.
        bin_size_ms (float): Bin width in milliseconds used to build the autocorrelogram (default: 0.25). Internally this value is converted to seconds and to an integer number of samples via sample_rate; the code ensures at least one sample per bin. Very small values relative to sample_rate will be clamped to one sample per bin.
        window_size_s (float): Window size in seconds (default: 1). This is the half-window used when computing the autocorrelogram around each spike; it is converted to sample counts by multiplying with sample_rate. Large values increase computation and memory for correlograms; very small values reduce the range over which refractory-period violations are counted.
        exclude_ref_period_below_ms (float): Refractory periods (tested durations) below this threshold in milliseconds are excluded from the determination of the minimum contamination (default: 0.5 ms). This is to avoid counting biologically implausible ultra-short refractory periods. The threshold is converted to seconds for internal comparisons.
        max_ref_period_ms (float): Maximum refractory-period duration (in milliseconds) to test (default: 10 ms). The function builds a vector of refractory period centers from 0 up to (but not exceeding) this maximum value in steps of bin_size_ms. Increasing this value tests longer candidate refractory periods at additional computational cost.
        contamination_values (numpy.ndarray): 1D array of candidate contamination fractions to test (each value should be in fractional units, e.g., 0.01 for 1%). If None, the function sets contamination_values to numpy.arange(0.5, 35, 0.5) / 100 (i.e., 0.5% to 34.5% in 0.5% steps). This vector defines the x-axis of the confidence grid: increasing its length increases computation proportionally.
        return_conf_matrix (bool): If True, also return the full confidence matrix (default: False). The confidence matrix has shape (n_contamination_values, n_ref_periods) and contains the computed confidence that the observed cumulative refractory-period violations are explained by each (contamination, refractory-period) pair.
    
    Returns:
        float or tuple:
        If return_conf_matrix is False, returns:
            min_cont_with_90_confidence (float): The smallest tested contamination fraction (same units as contamination_values) for which the computed confidence exceeds 0.90 (90%) for at least one tested refractory-period duration greater than exclude_ref_period_below_ms. If no contamination value achieves >90% confidence, numpy.nan is returned. The value is a float (or numpy scalar) and represents the estimated lower bound on contamination consistent with the observed violations under the sliding-RP model.
        If return_conf_matrix is True, returns:
            (min_cont_with_90_confidence, conf_matrix) where conf_matrix is a numpy.ndarray of shape (n_contamination_values, n_ref_periods): conf_matrix[i, j] is the computed confidence (between 0 and 1) that contamination value contamination_values[i] and refractory period rp_centers[j] explain the observed cumulative refractory-period violations. The caller can inspect conf_matrix to visualize confidence as a function of contamination and refractory period.
    
    Behavioral notes, defaults, and failure modes:
        - The function computes per-segment autocorrelograms using an internal correlogram_for_one_segment helper, sums them to produce a global autocorrelogram, and then analyses the positive-lag cumulative counts up to each tested refractory-period center.
        - Default contamination_values (when contamination_values is None) is numpy.arange(0.5, 35, 0.5) / 100 (0.005 to 0.345 step 0.005). rp_centers are produced from 0 to max_ref_period_ms/1000 with steps bin_size_ms/1000 and then shifted to bin centers.
        - The function converts bin_size_ms and exclude_ref_period_below_ms from milliseconds to seconds internally and converts to sample counts using sample_rate; at least one sample per bin is enforced.
        - The firing rate is computed as total_spike_count / duration. If duration is zero or extremely small this will raise a ZeroDivisionError or produce unstable estimates; callers must provide a valid recording duration in seconds.
        - If spike_samples is empty (no spikes) the function will compute n_spikes=0 and may return numpy.nan for the minimum contamination since there are no observed violations; behavior depends on downstream routines but will not produce a valid positive contamination estimate.
        - The function expects spike times in sample units consistent with sample_rate; supplying times in seconds without conversion will produce incorrect results.
        - The returned conf_matrix values and the derived min_cont_with_90_confidence depend on the assumptions of the underlying sliding-RP statistical model implemented in _compute_violations; as with any model, extreme or pathological spike trains (very low spike counts, extremely high firing rates concentrated in short intervals, or inconsistent multi-segment durations) can yield unreliable or non-informative outputs.
        - No in-place modification of input arrays is performed; this function is read-only with respect to its inputs.
    
    References and provenance:
        - The implementation and core algorithm are adapted from the SteinmetzLab slidingRP codebase (see https://github.com/SteinmetzLab/slidingRefractory) and are intended for use within SpikeInterface quality-metrics pipelines for spike sorting validation.
    """
    from spikeinterface.qualitymetrics.misc_metrics import slidingRP_violations
    return slidingRP_violations(
        spike_samples,
        sample_rate,
        duration,
        bin_size_ms,
        window_size_s,
        exclude_ref_period_below_ms,
        max_ref_period_ms,
        contamination_values,
        return_conf_matrix
    )


################################################################################
# Source: spikeinterface.qualitymetrics.pca_metrics.lda_metrics
# File: spikeinterface/qualitymetrics/pca_metrics.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_qualitymetrics_pca_metrics_lda_metrics(
    all_pcs: numpy.ndarray,
    all_labels: numpy.ndarray,
    this_unit_id: int
):
    """spikeinterface.qualitymetrics.pca_metrics.lda_metrics computes a d-prime separability measure for a single sorted unit using Linear Discriminant Analysis (LDA). This function is used within the SpikeInterface quality-metrics workflow to quantify how well the principal-component (PC) feature distribution of spikes assigned to a target unit (this_unit_id) is separated from the distribution of all other spikes; a larger d-prime indicates better separability and therefore higher presumed unit quality during spike sorting validation and curation.
    
    Args:
        all_pcs (numpy.ndarray): A 2-D array of principal-component scores for every spike in the recording, organized as [num_spikes, num_PCs]. In the SpikeInterface workflow this array is typically produced by applying PCA to extracted spike waveforms and is used here as the feature matrix X for LDA. The function requires a numeric numpy.ndarray; shape must be (N, P) where N is the number of spikes and P is the number of retained PCs. If this array is not two-dimensional or its first dimension does not match the length of all_labels, the function will produce an error or invalid results.
        all_labels (numpy.ndarray): A 1-D array of integer cluster labels for each spike, with length equal to the number of spikes N. These labels come from a spike-sorting output (the cluster assignments to be evaluated). The function builds a boolean target mask by comparing all_labels to this_unit_id; therefore all_labels must be indexable and align elementwise with all_pcs rows. If lengths mismatch or the array is not 1-D, the behavior is undefined and an exception may be raised.
        this_unit_id (int): The integer identifier of the unit (cluster) to evaluate. This identifier is compared against all_labels to create the positive class for LDA. In practice this is the unit ID returned by a spike sorting algorithm or a SortingExtractor; the function measures how distinct the PC features of spikes labeled this_unit_id are from all other spikes.
    
    Behavior and practical details:
        The function constructs a boolean target vector y where entries equal True for spikes with label this_unit_id and False otherwise. It fits sklearn.discriminant_analysis.LinearDiscriminantAnalysis with n_components=1 to project the multi-dimensional PC features onto a single discriminant axis (Fisher LDA). The projection values for the target unit and all other spikes are separated into two one-dimensional arrays. The d-prime is computed as the difference of the two class means divided by the pooled standard deviation: (mean_target - mean_other) / sqrt(0.5*(var_target + var_other)). This scalar summary quantifies effect size (separability) along the LDA axis and is the same metric used in quality evaluation literature (see Hill et al. reference in original code). The function does not modify its inputs; it creates local copies/views to fit the LDA and compute statistics.
    
    Defaults and implementation notes:
        The implementation fixes LinearDiscriminantAnalysis(n_components=1) and uses the default solver/settings from sklearn for LDA. No random seeds are required because LDA is deterministic for a given input. The function returns a single float value.
    
    Failure modes and edge cases:
        If all_pcs has zero variance along the direction needed for discrimination or if one of the classes has zero variance (e.g., a class with identical projected values), the pooled standard deviation in the denominator may be zero and the result may be inf or NaN. If the target class (this_unit_id) has zero spikes (no entries in all_labels equal to this_unit_id) or if there are fewer than two samples in a class, sklearn's LDA.fit_transform may raise a ValueError or produce degenerate output; callers should validate that the target unit has a sufficient number of spikes before calling this function. If the shapes of all_pcs and all_labels are inconsistent (first dimension of all_pcs not equal to length of all_labels) a shape-related exception can occur. Any exceptions raised by sklearn (for example due to singular covariance estimates in extreme cases) will propagate to the caller.
    
    Returns:
        float: The computed d-prime value for this_unit_id. This scalar quantifies the separability of the unit's projected PC distribution from all other spikes along the LDA axis; positive values indicate the target class mean is greater than the other-class mean along the discriminant, negative values indicate the opposite, and larger magnitude indicates stronger separation.
    """
    from spikeinterface.qualitymetrics.pca_metrics import lda_metrics
    return lda_metrics(all_pcs, all_labels, this_unit_id)


################################################################################
# Source: spikeinterface.qualitymetrics.pca_metrics.mahalanobis_metrics
# File: spikeinterface/qualitymetrics/pca_metrics.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_qualitymetrics_pca_metrics_mahalanobis_metrics(
    all_pcs: numpy.ndarray,
    all_labels: numpy.ndarray,
    this_unit_id: int
):
    """spikeinterface.qualitymetrics.pca_metrics.mahalanobis_metrics calculates two commonly used spike-sorting unit quality metrics, isolation distance and L-ratio, from Mahalanobis distances computed in PCA feature space. These metrics quantify how well spikes assigned to a given unit (this_unit_id) are separated from spikes assigned to other units, using the empirical covariance of the unit's principal components. This function is used in SpikeInterface quality metrics pipelines to evaluate unit isolation and aid automated or manual curation of spike-sorting results.
    
    Args:
        all_pcs (numpy.ndarray): 2D array of principal component projections for all detected spikes. The expected organization is [num_spikes, num_pcs], where num_spikes is the total number of spikes across all clusters and num_pcs is the number of PCA features per spike. In the SpikeInterface context, these PCs are typically extracted per-spike waveforms and are used as the feature space for distance-based quality metrics.
        all_labels (numpy.ndarray): 1D array of integer cluster labels for all spikes. Must have length equal to num_spikes (the first dimension of all_pcs). Labels partition the spikes into units/clusters; this function separates the rows of all_pcs into those belonging to this_unit_id and those belonging to other units using this array. If lengths do not match the first dimension of all_pcs, indexing operations will fail (typically raising an exception).
        this_unit_id (int): Integer identifier of the unit (cluster) for which to compute the metrics. The function selects spikes where all_labels == this_unit_id as the "self" set and all other spikes as the "other" set. In typical SpikeInterface workflows, this corresponds to a single sorted unit whose isolation is being assessed.
    
    Behavior and computation details:
        The function computes the sample mean of the PCA features for spikes assigned to this_unit_id and the sample covariance matrix from those same spikes. It then attempts to compute the inverse of that covariance matrix and uses scipy.spatial.distance.cdist with metric "mahalanobis" to compute Mahalanobis distances from the unit mean to every spike in both the "self" (pcs_for_this_unit) and "other" (pcs_for_other_units) sets, using the inverse covariance as the VI parameter.
        Degrees of freedom (dof) for the chi-square distribution is set to the number of PCA features (num_pcs), i.e., pcs_for_this_unit.shape[1].
        Let n_self be the number of spikes in the unit (pcs_for_this_unit.shape[0]) and n_other be the number of spikes in other units (pcs_for_other_units.shape[0]). The variable n used in the metric definitions is min(n_self, n_other).
        Isolation distance is defined as the squared Mahalanobis distance of the (n)-th nearest other spike to the unit mean (i.e., pow(mahalanobis_other[n - 1], 2)), which corresponds to the Mahalanobis radius that would contain as many other spikes as the unit contains.
        L-ratio is computed as the sum over other spikes of the tail probability 1 - chi2.cdf(mahalanobis_other**2, dof), divided by the number of spikes in the unit (mahalanobis_self.shape[0]). This yields a normalized measure of how much of the other-spike Mahalanobis mass lies near the unit center; smaller L-ratio indicates better isolation.
    
    Side effects, defaults, and failure modes:
        The function has no side effects (does not modify inputs or external state); it performs numerical computations and returns two floats.
        If the sample covariance matrix of pcs_for_this_unit is singular or not invertible, numpy.linalg.inv will raise a LinAlgError which the function catches; in that case the function returns (numpy.nan, numpy.nan) for (isolation_distance, l_ratio). This indicates the metrics could not be computed due to degenerate feature covariance (e.g., too few spikes or collinear features).
        If n < 2 (i.e., either the unit has fewer than two spikes or there are fewer than two other spikes), the function returns (numpy.nan, numpy.nan) because the metrics are undefined for too-small sample sizes.
        If all_labels and all_pcs lengths do not match or have incompatible shapes for boolean indexing, the indexing operations will raise a Python exception (e.g., IndexError or ValueError) prior to metric computation.
        The function relies on scipy.spatial.distance.cdist and scipy.stats.chi2; unexpected input shapes or non-finite values (NaN/Inf) in all_pcs may lead to exceptions or propagate NaNs in the outputs.
    
    Returns:
        isolation_distance (float): Squared Mahalanobis distance threshold (as described above) for this_unit_id. Represents the Mahalanobis radius (squared) at which as many other-unit spikes fall inside as the number of spikes in the unit. Returns numpy.nan if the covariance is singular or there are insufficient spikes to define the metric.
        l_ratio (float): L-ratio for this_unit_id, computed as the summed tail probabilities of other-unit spikes under a chi-square distribution with degrees of freedom equal to the number of PCA features, normalized by the unit spike count. Lower values indicate better isolation. Returns numpy.nan if the covariance is singular or there are insufficient spikes to define the metric.
    
    References:
        Based on metrics described in Schmitzer-Torbert et al.; these metrics are standard in spike-sorting quality assessment and are used in SpikeInterface for automated quality metrics and curation workflows.
    """
    from spikeinterface.qualitymetrics.pca_metrics import mahalanobis_metrics
    return mahalanobis_metrics(all_pcs, all_labels, this_unit_id)


################################################################################
# Source: spikeinterface.qualitymetrics.pca_metrics.nearest_neighbors_metrics
# File: spikeinterface/qualitymetrics/pca_metrics.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_qualitymetrics_pca_metrics_nearest_neighbors_metrics(
    all_pcs: numpy.ndarray,
    all_labels: numpy.ndarray,
    this_unit_id: int,
    max_spikes: int,
    n_neighbors: int
):
    """spikeinterface.qualitymetrics.pca_metrics.nearest_neighbors_metrics: Calculate nearest-neighbor based contamination metrics for a single sorted unit using PCA features.
    
    This function is used in the SpikeInterface quality-metrics pipeline to quantify cluster contamination and isolation by examining nearest neighbors in PCA feature space computed from spike waveforms. It constructs a feature matrix that places spikes from the target unit first and then all other spikes, optionally subsamples spikes when max_spikes is smaller than the total number of spikes, fits a scikit-learn NearestNeighbors model (ball_tree), excludes the self-nearest neighbor, and computes two scalar metrics: the hit rate (how often neighbors of target-unit spikes are from the target unit) and the miss rate (how often neighbors of other-unit spikes are from the target unit). These metrics are useful to assess contamination and separation of a putative unit after spike sorting.
    
    Args:
        all_pcs (numpy.ndarray): Array of principal-component feature vectors for all spikes. Shape is (num_spikes, num_pcs). Each row is the PCA feature vector for a single spike. This array provides the feature space in which nearest-neighbor search is performed; the number of rows must match the length of all_labels.
        all_labels (numpy.ndarray): One-dimensional array of integer cluster labels for each spike. Must have length equal to all_pcs.shape[0]. Values identify which sorted unit each spike belongs to; this_unit_id is matched against these entries to select the target unit's spikes.
        this_unit_id (int): Integer label identifying the target unit (cluster) for which to compute the metrics. Spikes with all_labels equal to this_unit_id are treated as the target cluster and are placed first in the nearest-neighbor search matrix.
        max_spikes (int): Maximum number of spikes to use in the computation, applied as an approximate per-cluster cap via uniform subsampling when max_spikes is smaller than the total number of spikes. Internally, the function computes ratio = max_spikes / total_spikes and, if ratio < 1, selects evenly spaced indices using numpy.arange with step 1/ratio to subsample the concatenated feature matrix. Note that very large values (original code warns about values >20000) increase computational cost and memory usage; very small values can reduce metric reliability.
        n_neighbors (int): Number of neighbors requested from sklearn.neighbors.NearestNeighbors (the first neighbor is expected to be the point itself and is removed before metric computation). The function fits a NearestNeighbors model with algorithm="ball_tree" on the (possibly subsampled) feature matrix and then uses the returned neighbor indices excluding the first column. If n_neighbors is less than or equal to 1 the neighbor-exclusion step yields empty neighbor lists and the resulting hit/miss rates may be NaN; choose n_neighbors >= 2 for meaningful non-empty neighbor sets.
    
    Returns:
        hit_rate (float): Fraction of nearest neighbors for spikes from the target cluster that are also labeled as the target cluster. Computed as mean(this_cluster_nearest < num_obs_this_unit) where this_cluster_nearest are neighbor indices for the first num_obs_this_unit rows (target spikes) after excluding self-neighbors. Values are in [0.0, 1.0] when computed over non-empty neighbor sets; may be NaN if no neighbors are considered.
        miss_rate (float): Fraction of nearest neighbors for spikes from other clusters that are labeled as the target cluster. Computed as mean(other_cluster_nearest < num_obs_this_unit) where other_cluster_nearest are neighbor indices for the rows corresponding to non-target spikes. Values are in [0.0, 1.0] when computed over non-empty neighbor sets; may be NaN if no neighbors are considered.
    
    Behavior, side effects, and failure modes:
        The function concatenates target-unit PCA rows first then other-unit PCA rows so that integer index comparisons (index < num_obs_this_unit) identify membership in the target cluster. If there is only one unique label in all_labels (no other units present), the function issues a python warnings.warn and returns (1.0, 0.0) as the best possible hit/miss result for that degenerate case. The function expects all_pcs.shape[0] == len(all_labels); if these do not match, indexing operations will raise an exception (e.g., IndexError or ValueError). Large values of max_spikes increase runtime and memory because NearestNeighbors is constructed on up to that many samples; the original implementation notes potential slowness for values >20000. Identical feature vectors, duplicate spikes, or extremely small n_neighbors can produce degenerate behavior (including NaN results) because the implementation removes the first neighbor per point (assumed to be the point itself) before computing rates. The function uses sklearn.neighbors.NearestNeighbors with algorithm="ball_tree", so sklearn must be available in the environment.
    """
    from spikeinterface.qualitymetrics.pca_metrics import nearest_neighbors_metrics
    return nearest_neighbors_metrics(
        all_pcs,
        all_labels,
        this_unit_id,
        max_spikes,
        n_neighbors
    )


################################################################################
# Source: spikeinterface.qualitymetrics.pca_metrics.silhouette_score
# File: spikeinterface/qualitymetrics/pca_metrics.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_qualitymetrics_pca_metrics_silhouette_score(
    all_pcs: numpy.ndarray,
    all_labels: numpy.ndarray,
    this_unit_id: int
):
    """spikeinterface.qualitymetrics.pca_metrics.silhouette_score calculates the silhouette score for a single sorted unit using principal components (PCs) of spike waveforms. This score is a standard cluster-quality metric that ranges from -1 (poor clustering, spikes likely misassigned) to +1 (well-separated cluster). In the SpikeInterface domain, this function is used as a quality metric to validate and curate spike-sorting outputs by quantifying how well spikes assigned to a given unit separate from spikes of other units in PC feature space.
    
    The function computes pairwise Euclidean distances (via scipy.spatial.distance.cdist) between spikes in the PC space, estimates the average distance from spikes of the target unit to each other cluster, selects the nearest other cluster (the one with the smallest mean inter-cluster distance), and then computes per-spike silhouette distances using the standard formula (b - a) / max(a, b) where a is the intra-unit pairwise distance and b is the distance to the nearest other cluster. The returned unit silhouette score is the mean of those per-spike silhouette distances.
    
    Args:
        all_pcs (numpy.ndarray): A 2D array of principal components used as features for clustering, organized as [num_spikes, num_pcs]. Each row corresponds to one spike waveform projected into PC space. This array provides the feature vectors on which pairwise Euclidean distances are computed to assess cluster separability in spike sorting.
        all_labels (numpy.ndarray): A 1D array of integer cluster labels for all spikes. Must have length equal to the number of rows in all_pcs (num_spikes). Labels identify which spikes belong to which sorted unit; this mapping is used to select the spikes belonging to this_unit_id and to iterate over other clusters for inter-cluster distance comparisons.
        this_unit_id (int): The integer label (ID) of the unit for which to calculate the silhouette score. This ID must appear in all_labels for the function to compute a meaningful score for that unit.
    
    Behavior, assumptions, and failure modes:
        The function uses scipy.spatial.distance.cdist with the default Euclidean metric to compute pairwise distances. For the target unit, it computes the intra-unit distance matrix between all spikes of this_unit_id and itself. For each other label present in all_labels, it computes the distance matrix between spikes of that other label and spikes of this_unit_id and uses the mean of that matrix as the inter-cluster distance to that other label. The other cluster with the smallest mean inter-cluster distance is selected as the "nearest other cluster" and its full distance matrix to the target unit is used to compute per-spike silhouette distances.
        The function expects at least one spike assigned to this_unit_id and at least one other label with at least one spike; otherwise, the intermediate arrays may be empty and numpy operations such as mean may produce NaN or raise exceptions. In particular, if pcs_for_this_unit has zero rows, or if there are no other clusters with spikes, the result is undefined and may raise an error or return NaN.
        The implementation performs elementwise arithmetic between the intra-unit distance matrix and the selected inter-cluster distance matrix; these arrays must be shape/broadcast-compatible for numpy operations. If cluster sizes differ such that the distance matrices are not broadcast-compatible, numpy may raise a ValueError due to shape mismatch. Users should ensure that their data satisfy these conditions or handle exceptions accordingly.
        This function is pure (no external side effects) and relies on numpy and scipy for computations. It does not modify inputs.
    
    Returns:
        unit_silhouette_score (float): The mean silhouette score for the specified unit computed across spikes of that unit. Values are theoretically in the interval [-1, 1], where values close to 1 indicate that spikes of this unit are well separated from spikes of other units in PC space (high cluster quality for spike sorting), values near 0 indicate overlap or ambiguous clustering, and values near -1 indicate that spikes may be misassigned. If inputs are invalid or degenerate (e.g., no spikes for this unit or no other clusters), the returned value may be NaN or an exception may be raised depending on numpy/scipy behavior.
    """
    from spikeinterface.qualitymetrics.pca_metrics import silhouette_score
    return silhouette_score(all_pcs, all_labels, this_unit_id)


################################################################################
# Source: spikeinterface.qualitymetrics.pca_metrics.simplified_silhouette_score
# File: spikeinterface/qualitymetrics/pca_metrics.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_qualitymetrics_pca_metrics_simplified_silhouette_score(
    all_pcs: numpy.ndarray,
    all_labels: numpy.ndarray,
    this_unit_id: int
):
    """spikeinterface.qualitymetrics.pca_metrics.simplified_silhouette_score: Compute the simplified silhouette score for a single cluster (unit) using principal components (PCs) of spike waveforms. This metric is a centroid-based approximation of the classic silhouette score used in clustering and spike sorting quality assessment; it is intended to provide a fast, interpretable measure of how well a unit's spikes are separated from spikes of other units in PC space and is used in SpikeInterface workflows for validating and curating spike sorting outputs.
    
    Args:
        all_pcs (numpy.ndarray): 2D array of principal components for all spikes, organized as [num_spikes, num_PCs]. Each row corresponds to a single spike projection in PC space. This array provides the feature space in which cluster separability is measured; it must be two-dimensional and have one row per spike.
        all_labels (numpy.ndarray): 1D array of integer cluster labels for all spikes. Must have length equal to the number of rows in all_pcs. Labels identify the unit (cluster) assignment for each spike produced by a spike sorter; this function selects spikes with label equal to this_unit_id as the focal cluster.
        this_unit_id (int): Integer identifier of the unit (cluster) for which to calculate the simplified silhouette score. This should match the label values contained in all_labels and designates the "own" cluster whose intra-cluster distances and nearest other-cluster distances will be compared.
    
    Returns:
        unit_silhouette_score (float): Mean simplified silhouette score for spikes of the specified unit. Computed by: (b_i - a_i) / max(a_i, b_i) for each spike i in this unit, where a_i is the distance from the unit centroid to spike i (intra-cluster distance) and b_i is the mean distance from the spike i to the centroid of the nearest other cluster (inter-cluster distance, using centroid-to-spike distances). The returned value is the arithmetic mean of these per-spike values and is intended to lie in the interval [-1, 1], where values near 1 indicate that the unit is well separated from other units in PC space, values near 0 indicate overlap, and values near -1 indicate poor separation. The implementation uses scipy.spatial.distance.cdist to compute centroid-to-spike distances and numpy to compute means.
    
    Behavior and assumptions:
        The function implements a simplified silhouette by using centroids for distance calculations rather than expensive pairwise spike-to-spike distances. It selects the centroid of the focal unit (mean of its spikes in PC space) and, for each other label, the centroid of that label; it then finds the other-cluster centroid with the minimum mean distance to the focal unit's spikes and uses that mean to compute inter-cluster distances b_i. This design reduces computational complexity relative to the standard silhouette metric and is suitable for large spike datasets typically encountered in extracellular electrophysiology workflows described in the SpikeInterface README.
    
    Preconditions and failure modes:
        The function assumes:
            - all_pcs is a numpy.ndarray with ndim == 2 and shape[0] equal to len(all_labels).
            - all_labels is a 1D numpy.ndarray with one label per spike.
            - At least one spike exists for this_unit_id (i.e., at least one occurrence of this_unit_id in all_labels).
            - At least one other distinct label exists in all_labels (so a nearest other-cluster centroid can be found).
        If these preconditions are violated, the function may raise exceptions (for example, from numpy operations or scipy.spatial.distance.cdist) or return NaN values. Callers should validate inputs and handle or prevent the following conditions:
            - Empty or mismatched arrays (lengths inconsistent between all_pcs and all_labels) will lead to errors.
            - No spikes for this_unit_id will cause mean computations over empty arrays, producing NaN or raising an error.
            - If no other labels exist (only one cluster present), the algorithm cannot compute a nearest other-cluster centroid and will fail; callers should detect this case and decide how to handle it (for example, by skipping the metric or returning a sentinel value).
        The function has no side effects and does not modify its inputs.
    
    Implementation notes:
        This function is intended for use within the SpikeInterface quality metrics suite to provide a fast, interpretable estimate of cluster separability in PC space during post-processing and curation of spike sorting results. It trades the exactness of the full silhouette calculation for reduced computation by comparing per-spike distances to two centroids (own and nearest-other) rather than all pairwise distances.
    """
    from spikeinterface.qualitymetrics.pca_metrics import simplified_silhouette_score
    return simplified_silhouette_score(all_pcs, all_labels, this_unit_id)


################################################################################
# Source: spikeinterface.sorters.container_tools.find_recording_folders
# File: spikeinterface/sorters/container_tools.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_sorters_container_tools_find_recording_folders(d: dict):
    """spikeinterface.sorters.container_tools.find_recording_folders finds the minimal set of filesystem folders that contain recording file paths described in a SpikeInterface-style dictionary and prepares them for use as container mount points.
    
    This function is used in the container tooling of SpikeInterface (a framework to run spike sorters in Docker/Singularity containers) to determine which host folders need to be mounted into a container so the sorter can access the recording files. It extracts file paths from the provided dictionary using the internal helper _get_paths_list, resolves each path, reduces them to their parent directories, and attempts to compute a single common parent folder when possible to minimize the number of mounts. The function performs only path computations and does not perform any I/O, create mounts, or modify the input dictionary.
    
    Args:
        d (dict): A dictionary containing recording metadata and file paths in the format expected by SpikeInterface container utilities. The helper _get_paths_list(d=d) is called to extract a list of raw file-system paths (strings) from this dictionary. The caller is responsible for providing a dictionary where recording file paths can be found by that helper; if _get_paths_list raises an exception for an unexpected structure or missing keys, this function will not catch it.
    
    Returns:
        list[pathlib.Path]: A list of pathlib.Path objects representing folders to mount. Each entry is the resolved parent folder of a recording path extracted from d. If all recording paths share a single common parent folder on the same filesystem, the returned list will contain that single common parent (resolved). If recording paths reside on different root devices or filesystems and os.path.commonpath raises ValueError, the function returns the list of individual parent folders. To avoid returning an overly broad mount like the filesystem root, if the computed common parent serializes to a one-character string (e.g., "/"), the function instead returns the individual parent folders. The returned paths are intended to be used as mount source paths for containers and should be portable to the container runtime; the function does not validate container-side paths or perform mount operations.
    """
    from spikeinterface.sorters.container_tools import find_recording_folders
    return find_recording_folders(d)


################################################################################
# Source: spikeinterface.sorters.container_tools.path_to_unix
# File: spikeinterface/sorters/container_tools.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_sorters_container_tools_path_to_unix(path: str):
    """spikeinterface.sorters.container_tools.path_to_unix converts a filesystem path string to a POSIX-style (Unix) path string. This helper is intended for use inside the SpikeInterface container tooling (spikeinterface.sorters.container_tools) when preparing host filesystem paths for use with Unix-based container runtimes (for example Docker or Singularity) that do not use Windows drive letters. It makes Windows-style paths compatible with Unix-style containers by removing the drive letter and colon and returning a forward-slash-separated path.
    
    Args:
        path (str): Filesystem path on the host machine provided as a Python string. In the SpikeInterface domain this is typically a path to recording files, sorter output directories, or other data files that must be mounted or referenced inside a Unix-based container when running spike sorters. The function will convert this string to a POSIX-style path. The implementation uses pathlib.Path(path) internally, so if an invalid type is passed (not a str) a TypeError raised by pathlib.Path may occur; callers should pass a string as required by the function signature.
    
    Returns:
        str: A POSIX-style path string suitable for use in Unix environments and container commands. Behavior:
            - On Windows (platform.system() == "Windows"): the function strips the drive letter and the colon by taking the substring starting immediately after the first ":" character and then returns that path in POSIX form (forward slashes). This makes a Windows path like "C:\\some\\dir" usable as "/some/dir" inside a Unix-like container context. Note that the function only removes the drive letter and colon; it does not attempt to perform any mount mapping, create directories, or validate that the resulting path exists inside the container.
            - On non-Windows platforms: the function returns path.as_posix(), which replaces OS-specific separators with "/" but otherwise leaves the path content unchanged.
        Side effects and failure modes: This function is pure (no filesystem side effects) and does not access or modify files. It does not verify that the returned path will be valid or accessible inside any container; callers must ensure appropriate bind mounts or volume mappings when launching containers. If the input string contains no ":", the slicing logic leaves the string intact and returns its POSIX form. If a non-string is provided, pathlib.Path may raise a TypeError.
    """
    from spikeinterface.sorters.container_tools import path_to_unix
    return path_to_unix(path)


################################################################################
# Source: spikeinterface.sorters.external.yass.merge_params_dict
# File: spikeinterface/sorters/external/yass.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_sorters_external_yass_merge_params_dict(yass_params: dict, params: dict):
    """spikeinterface.sorters.external.yass.merge_params_dict merges a default YASS parameter dictionary with caller-provided SpikeInterface-level parameters to produce a single configuration dictionary that the YASS external sorter wrapper can use. This function is used inside the SpikeInterface YASS sorter integration to translate generic sorter parameters (provided by SpikeInterface or the user) into specific entries expected by YASS, and to expose these combined parameters for running or configuring the YASS pipeline.
    
    Args:
        yass_params (dict): The base/default YASS configuration dictionary (the "default" parameters produced by or for the YASS sorter). This dictionary is treated as the starting point for the merge. The function creates a shallow copy of this dict and updates specific nested keys. Because the copy is shallow, nested sub-dictionaries referenced by yass_params may be mutated by this function; if the caller must preserve the original yass_params and all nested objects, they should pass a deep copy.
        params (dict): A dictionary of higher-level SpikeInterface parameters that should override or supply values for specific YASS configuration fields. This dict is expected to contain keys referenced in the implementation (for example "freq_min", "freq_max", "neural_nets_path", "multi_processing", "n_processors", "n_gpu_processors", "n_sec_chunk", "n_sec_chunk_gpu_detect", "n_sec_chunk_gpu_deconv", "gpu_id", "generate_phy", "phy_percent_spikes", "spatial_radius", "spike_size_ms", "clustering_chunk", "update_templates", "neuron_discover", "template_update_time"). Values should be appropriate for their intended fields (e.g., numeric frequency bounds for "freq_min"/"freq_max", a path-like string for "neural_nets_path"), because missing keys or incompatible types will raise exceptions at runtime.
    
    Behavior and side effects:
        The function produces a merged configuration by performing a shallow copy of yass_params and then setting or overriding specific nested keys using values from params. The exact assignments performed by the function are:
        merge_params["preprocess"]["filter"]["low_pass_freq"] is set to params["freq_min"]
        merge_params["preprocess"]["filter"]["high_factor"] is set to params["freq_max"]
        merge_params["neuralnetwork"]["detect"]["filename"] is set to os.path.join(params["neural_nets_path"], "detect.pt")
        merge_params["neuralnetwork"]["denoise"]["filename"] is set to os.path.join(params["neural_nets_path"], "denoise.pt")
        merge_params["resources"]["multi_processing"] is set to params["multi_processing"]
        merge_params["resources"]["n_processors"] is set to params["n_processors"]
        merge_params["resources"]["n_gpu_processors"] is set to params["n_gpu_processors"]
        merge_params["resources"]["n_sec_chunk"] is set to params["n_sec_chunk"]
        merge_params["resources"]["n_sec_chunk_gpu_detect"] is set to params["n_sec_chunk_gpu_detect"]
        merge_params["resources"]["n_sec_chunk_gpu_deconv"] is set to params["n_sec_chunk_gpu_deconv"]
        merge_params["resources"]["gpu_id"] is set to params["gpu_id"]
        merge_params["resources"]["generate_phy"] is set to params["generate_phy"]
        merge_params["resources"]["phy_percent_spikes"] is set to params["phy_percent_spikes"]
        merge_params["recordings"]["spatial_radius"] is set to params["spatial_radius"]
        merge_params["recordings"]["spike_size_ms"] is set to params["spike_size_ms"]
        merge_params["recordings"]["clustering_chunk"] is set to params["clustering_chunk"]
        merge_params["deconvolution"]["update_templates"] is set to params["update_templates"]
        merge_params["deconvolution"]["neuron_discover"] is set to params["neuron_discover"]
        merge_params["deconvolution"]["template_update_time"] is set to params["template_update_time"]
        The filenames for the neural network models are constructed with os.path.join using params["neural_nets_path"] and the literal filenames "detect.pt" and "denoise.pt".
    
    Defaults:
        The function does not introduce defaults beyond those already present in yass_params. It assumes yass_params contains the nested structure accessed in the assignments. If a value is missing in params, no default is provided by this function and a KeyError will be raised when attempting to read the missing key.
    
    Failure modes:
        If either yass_params or params does not contain the expected nested keys, the function will raise KeyError when attempting to access or assign the referenced entries. If params["neural_nets_path"] is not a path-like string, os.path.join may raise a TypeError. Because the function uses a shallow copy of yass_params, unintended mutations of nested structures in the original yass_params may occur; this is a common source of unexpected side effects if callers assume full immutability.
    
    Returns:
        dict: A dictionary (the merged YASS configuration) that is a shallow copy of yass_params with the listed nested fields updated from params. This returned dict is intended to be used by the SpikeInterface YASS sorter wrapper to run or configure the YASS pipeline. The original params dict is not modified by this function, but nested sub-dictionaries inside the original yass_params may be modified due to the shallow copy approach.
    """
    from spikeinterface.sorters.external.yass import merge_params_dict
    return merge_params_dict(yass_params, params)


################################################################################
# Source: spikeinterface.sorters.launcher.run_sorter_jobs
# File: spikeinterface/sorters/launcher.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_sorters_launcher_run_sorter_jobs(
    job_list: list,
    engine: str = "loop",
    engine_kwargs: dict = None,
    return_output: bool = False
):
    """Run several run_sorter() calls either sequentially or using a parallel/asynchronous engine.
    This function is part of the SpikeInterface suite for running spike sorter jobs described
    in job dictionaries (see SpikeInterface README and run_sorter usage). Each entry in
    job_list is treated as the keyword arguments to a single call to run_sorter(...). The
    function selects an execution engine (loop, joblib, processpoolexecutor, dask, or slurm)
    and dispatches the jobs according to the chosen engine, merging provided engine-specific
    defaults with the engine_kwargs argument. It manages per-job flags required to request
    that run_sorter return outputs (by injecting the with_output key into each job dict),
    creates and submits batch scripts for slurm, and may return the list of Sorting objects
    produced by run_sorter when supported by the chosen engine.
    
    Args:
        job_list (list): A list of dict. Each dict contains keyword arguments that are
            passed verbatim to spikeinterface.sorters.launcher.run_sorter(...). In the
            SpikeInterface domain, each job dict typically contains a "recording" object
            (a RecordingExtractor-like object that implements to_dict()) and sorter-specific
            parameters such as output folder paths and sorter options. For the "slurm"
            engine the function reads recording.to_dict() and therefore each recording value
            must implement a to_dict() method. The function mutates each dict by adding the
            key "with_output" (set to True or False depending on return_output) before
            calling or dispatching run_sorter.
        engine (str): Execution engine name. Supported engine strings (checked against the
            module-level _implemented_engine) include: "loop", "joblib", "processpoolexecutor",
            "dask", and "slurm". The choice affects blocking behavior and how jobs are
            dispatched:
            - "loop": run_sorter() is called sequentially in the main process (blocking).
            - "joblib": uses joblib.Parallel to run jobs in parallel (blocking until all finish).
            - "processpoolexecutor": uses concurrent.futures.ProcessPoolExecutor (blocking).
            - "dask": submits tasks to a provided dask.distributed.Client (requires client in engine_kwargs;
              this implementation will call result() on each task, so it will block until results are ready).
            - "slurm": writes per-job Python scripts to tmp_script_folder, submits them with sbatch,
              and returns immediately (asynchronous from the caller perspective). For slurm the
              function will create temporary script files and invoke the sbatch command; results
              must be retrieved later (for example with spikeinterface.sorters.read_sorter_folder()).
            The function asserts the provided engine is present in _implemented_engine and will
            raise an AssertionError otherwise.
        engine_kwargs (dict): Parameters passed to the underlying engine. If None, an empty
            dict is used and then merged with engine-specific defaults read from the module's
            _default_engine_kwargs for the chosen engine. Known engine-specific keys (from the
            launcher implementation) include:
            - For "loop": no engine_kwargs are used (defaults are applied but empty).
            - For "joblib":
                n_jobs (int): maximum number of concurrently running jobs (default comes from
                    _default_engine_kwargs; typical default is -1 to use all CPUs).
                backend (str): joblib backend implementation (default from defaults, e.g. "loky").
            - For "processpoolexecutor":
                max_workers (int): maximum number of worker processes (default from defaults).
                mp_context (str or None): multiprocessing context passed to ProcessPoolExecutor
                    (default from defaults).
            - For "dask":
                client (dask.distributed.Client): required Dask client to submit tasks.
                    The function asserts client is not None for the dask engine.
            - For "slurm":
                tmp_script_folder (str or pathlib.Path): folder in which per-job python
                    scripts are written. If None (default), a temporary directory is created
                    using tempfile.mkdtemp with prefix "spikeinterface_slurm_".
                sbatch_args (dict): dictionary of sbatch arguments that will be translated
                    into command-line flags by prefixing keys with "--" (for example
                    {"cpus-per-task": 1, "mem": "1G"}). The function warns or errors if
                    unsupported legacy keys are used: passing "cpus_per_task" (underscore)
                    will raise a ValueError instructing to use "cpus-per-task" (hyphenated).
            The function merges user-provided engine_kwargs with built-in defaults; keys not
            recognized by the engine are ignored by this function but may be used by the
            underlying execution mechanism. For "slurm", temporary script files are created,
            their permissions are set to user read/write/execute (os.fchmod with S_IRWXU),
            and sbatch is invoked via subprocess.run; stdout is printed and non-empty stderr
            triggers a warnings.warn call.
        return_output (bool): If True, the function requests each run_sorter call to return
            its Sorting object by injecting "with_output": True into each job dict before
            dispatch. If False, "with_output" is set to False and the function returns None.
            Only a subset of engines support collecting and returning the outputs: the code
            requires engine to be one of "loop", "joblib", or "processpoolexecutor" when
            return_output=True; otherwise an AssertionError is raised. When return_output=True
            and supported, the function collects the Sorting objects returned by run_sorter
            from each dispatched job and returns them as a list in the same order as job_list.
    
    Returns:
        None or list: If return_output is False the function returns None and its observable
        side effects are the execution of the jobs (running sorters, creating output folders,
        writing files produced by run_sorter, and for "slurm" creating and submitting sbatch
        scripts). If return_output is True and the chosen engine supports returning outputs
        ("loop", "joblib", or "processpoolexecutor"), the function returns a list of Sorting
        objects (the exact type and contents are those produced by spikeinterface.sorters.launcher.run_sorter).
        In all cases exceptions raised by run_sorter propagate through this function (for
        parallel engines, exceptions raised in worker processes/threads will be raised when
        results are collected via result() or when joblib re-raises them). For asynchronous
        engines such as "slurm" the function returns None almost immediately and no automatic
        mechanism is provided to detect job completion; users should use spikeinterface utilities
        (for example read_sorter_folder()) to inspect or retrieve results after the external jobs finish.
    """
    from spikeinterface.sorters.launcher import run_sorter_jobs
    return run_sorter_jobs(job_list, engine, engine_kwargs, return_output)


################################################################################
# Source: spikeinterface.sorters.runsorter.read_sorter_folder
# File: spikeinterface/sorters/runsorter.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_sorters_runsorter_read_sorter_folder(
    folder: str,
    register_recording: bool = True,
    sorting_info: bool = True,
    raise_error: bool = True
):
    """spikeinterface.sorters.runsorter.read_sorter_folder loads a sorting result produced by a SpikeInterface-compatible sorter from a sorter output folder and reconstructs the in-memory sorting object for downstream post-processing, validation, visualization, or export.
    
    This function expects the folder to contain a SpikeInterface run log file named "spikeinterface_log.json" that documents the sorter name and the run outcome. It uses the "sorter_name" entry in that log to select the corresponding sorter implementation from the internal sorter registry (sorter_dict) and then delegates reconstruction to that sorter's get_result_from_folder method. This is typically used in the SpikeInterface workflow to programmatically re-load sorting outputs created by run_sorter so they can be inspected, have recordings attached, have sorting metadata re-associated, or be passed to quality metric and comparison tools.
    
    Args:
        folder (Pth or str): The path to the sorter output folder created by a sorter run. The folder must contain a file named "spikeinterface_log.json". This argument may be a string path or a Path-like object (the code converts it to a pathlib.Path). The folder identifies the on-disk location where sorting results, optional recording serializations (json or pickle), and the spikeinterface_log.json summary are stored.
        register_recording (bool): If True (default True), attempt to attach the original recording to the returned sorting object when the recording has been saved alongside the sorting results (for example as json or pickle in the same folder). Attaching the recording restores an association between the Sorting object and its Recording (useful for waveform extraction, visualization, and post-processing). When False, the returned sorting will not have the recording attached even if a serialized recording is present on disk.
        sorting_info (bool): If True (default True), request that the sorter's reconstruction attach sorting metadata (sorting info) to the returned sorting object. Sorting info can include sorter-specific parameters, metrics, channel maps, or other auxiliary data saved by the sorter run. When False, this auxiliary information will not be requested during reconstruction.
        raise_error (bool): If True (default True), raise a SpikeSortingError when the log indicates the sorter run failed (log["error"] is truthy). If False, and the run failed according to the log, the function returns None instead of raising; this allows callers to handle failed runs programmatically without exceptions.
    
    Behavior, side effects, defaults, and failure modes:
        - The function converts the provided folder argument to a pathlib.Path and looks for the file "spikeinterface_log.json" in that folder. If that file is missing, the function raises a generic Exception indicating the folder does not contain spikeinterface_log.json.
        - The log file is parsed with json.load; invalid JSON will raise the underlying json.JSONDecodeError.
        - The function expects the parsed log to contain at least the keys "error" and "sorter_name". It interprets log["error"] as a boolean indicating whether the sorter run failed; if True and raise_error is True, a SpikeSortingError is raised describing the failed run and the folder path. If log["error"] is True and raise_error is False, the function returns None.
        - If the run did not fail (log["error"] is falsy), the function reads log["sorter_name"] and looks up the corresponding sorter class in the internal registry sorter_dict. If the sorter name is not present in sorter_dict, a KeyError will be raised by the lookup.
        - The function calls SorterClass.get_result_from_folder(folder, register_recording=register_recording, sorting_info=sorting_info). Any exceptions raised by the sorter-specific get_result_from_folder implementation (for example due to missing files, incompatible versions, or corrupted serialized objects) propagate to the caller.
        - When register_recording is True and a serialized recording is present, attaching the recording may load additional files (json or pickle) and increase memory usage; callers should be prepared for I/O and deserialization costs.
        - Default behaviors: register_recording defaults to True, sorting_info defaults to True, and raise_error defaults to True to favor strict failure reporting in automated pipelines.
    
    Returns:
        sorting (object): The sorting result object returned by the sorter's get_result_from_folder method. This is the reconstructed in-memory sorting representation provided by the specific sorter implementation and is intended for use by SpikeInterface post-processing (quality metric computation, waveform extraction, visualization, comparison, export). If the log indicates the run failed and raise_error is False, the function returns None instead of raising an exception.
    """
    from spikeinterface.sorters.runsorter import read_sorter_folder
    return read_sorter_folder(folder, register_recording, sorting_info, raise_error)


################################################################################
# Source: spikeinterface.sorters.sorterlist.get_default_sorter_params
# File: spikeinterface/sorters/sorterlist.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_sorters_sorterlist_get_default_sorter_params(sorter_name_or_class: str):
    """Returns the default parameter dictionary for a given spike sorter implementation, resolving the sorter either from a registered name or from a Sorter class. This function is part of the SpikeInterface unified framework for spike sorting and is used to obtain canonical runtime configuration values that downstream code (sorting pipelines, benchmarking, GUIs, containerized runs) rely on to run a sorter with its default settings.
    
    Args:
        sorter_name_or_class (str or SorterClass): Identifier of the sorter whose default parameters are requested. This argument can be either:
            - a string key that must exist in the module-level sorter_dict mapping (the registered name for a sorter implementation), or
            - a SorterClass object that must be present in the module-level sorter_full_list (the class implementing a sorter).
            The value is used to resolve the concrete SorterClass and then obtain its defaults. In the spike sorting domain, callers pass this to retrieve the canonical configuration for a specific sorter implementation so they can run the sorter, display defaults in a user interface, or compare default settings across sorters.
    
    Behavior and side effects:
        If sorter_name_or_class is a str, the function looks up SorterClass = sorter_dict[sorter_name_or_class] and then returns SorterClass.default_params(). If sorter_name_or_class is a SorterClass that is present in sorter_full_list, the function uses that class directly and returns SorterClass.default_params(). The function itself does not modify global registry structures when successful; it simply delegates to the SorterClass.default_params() class method to produce the dictionary of defaults.
        The returned value is exactly whatever SorterClass.default_params() returns. If that method returns a mutable dictionary that is not a defensive copy, mutating the returned dictionary may affect future uses of that same object; callers that need isolation should copy the dictionary themselves.
    
    Failure modes and exceptions:
        If sorter_name_or_class is a str but not a key in sorter_dict, the lookup sorter_dict[sorter_name_or_class] raises a KeyError. If sorter_name_or_class is not a str and is not present in sorter_full_list, the function raises a ValueError with a message indicating an unknown sorter was given. The function may also propagate any exceptions raised by SorterClass.default_params().
    
    Returns:
        dict: A dictionary containing the default parameters for the resolved sorter. These defaults represent the canonical configuration used by that sorter implementation within SpikeInterface; they are intended to be used to configure sorter runs, to provide baseline values for GUIs or pipelines, and for benchmarking/comparison workflows.
    """
    from spikeinterface.sorters.sorterlist import get_default_sorter_params
    return get_default_sorter_params(sorter_name_or_class)


################################################################################
# Source: spikeinterface.sorters.sorterlist.get_sorter_description
# File: spikeinterface/sorters/sorterlist.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_sorters_sorterlist_get_sorter_description(sorter_name_or_class: str):
    """spikeinterface.sorters.sorterlist.get_sorter_description: Return the brief parameter description dictionary for a specified sorter used by the SpikeInterface framework.
    
    This function looks up and returns the sorter_description attribute defined on a registered sorter class. In the SpikeInterface domain (a unified framework for spike sorting), a "sorter" is a wrapper class that encapsulates a specific spike sorting algorithm and its configurable parameters; the returned description dictionary is used to document sorter parameters, drive GUIs, validate user-supplied parameters, generate help text for automated pipelines, and to assist in containerized or programmatic execution of sorters.
    
    Args:
        sorter_name_or_class (str or SorterClass): The sorter to retrieve the description from. This may be either:
            - a string name that is a key in the module-level sorter_dict (the canonical registry of available sorters), in which case the function will resolve the name to the corresponding SorterClass; or
            - a SorterClass object that is present in the module-level sorter_full_list (the list of available sorter classes).
            The SorterClass represents the SpikeInterface wrapper for a concrete spike sorting algorithm; its sorter_description attribute contains metadata about the algorithm's parameters (for example parameter names, help text, types, and defaults as provided by that sorter implementation). Pass exactly the registered name or the class object; other input types or unregistered classes will cause an error as described below.
    
    Returns:
        dict: The sorter_description dictionary taken directly from the resolved SorterClass (SorterClass.sorter_description). This dictionary contains the parameter descriptions for the sorter and is intended for downstream uses such as building user interfaces, validating parameters prior to running the sorter, generating documentation, or composing command lines for containerized execution. Note that the function returns the actual attribute from the SorterClass (a reference), not a deep copy; modifying the returned dictionary will modify the SorterClass.sorter_description attribute for that sorter, which may affect other code that relies on the class metadata.
    
    Raises:
        ValueError: If sorter_name_or_class is a string not found in sorter_dict, or if it is not a string and not a member of sorter_full_list, a ValueError is raised with a message of the form "Unknown sorter {sorter_name_or_class} has been given". This function performs no other side effects besides the lookup; it does not modify global registries or sorter classes itself.
    """
    from spikeinterface.sorters.sorterlist import get_sorter_description
    return get_sorter_description(sorter_name_or_class)


################################################################################
# Source: spikeinterface.sorters.sorterlist.get_sorter_params_description
# File: spikeinterface/sorters/sorterlist.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_sorters_sorterlist_get_sorter_params_description(
    sorter_name_or_class: str
):
    """Returns a description of the parameters supported by a specific spike sorter.
    
    This function is part of SpikeInterface, a unified framework for spike sorting (see README). It retrieves the parameter documentation that a given sorter exposes so callers (CLI, GUIs, workflow code, container wrappers, or benchmarking scripts) can present, validate, or programmatically set the parameters needed to run that sorter. The function accepts either the registered sorter name or the Sorter class object itself and returns the sorter-provided parameters description without modifying global state.
    
    Args:
        sorter_name_or_class (str or SorterClass): Identifier of the sorter whose parameters description is requested. If a string is provided, it must be the key of a sorter registered in the module-level sorter_dict registry (the canonical name used by SpikeInterface to reference available sorters). If a SorterClass object is provided, it must be one of the classes listed in the module-level sorter_full_list registry. This argument determines which SorterClass.params_description() method is invoked to obtain the description. Supplying the sorter name is the typical usage when code or user interfaces allow selecting a sorter by name; supplying the SorterClass is useful when the caller already has the class object (for example when iterating over sorter classes).
    
    Raises:
        ValueError: If sorter_name_or_class is a string not present in the sorter_dict registry, or if it is not a known SorterClass in sorter_full_list. This indicates the requested sorter is not recognized by the SpikeInterface installation and callers should handle the error by using a valid registered sorter name or class.
    
    Returns:
        dict: A dictionary containing the parameter description returned by the selected SorterClass.params_description() method. The returned dictionary is the sorter-specific parameter description that the sorter class exposes to SpikeInterface; typical consumers use it to display human-readable parameter information, determine default values, or validate user-supplied parameters before launching the sorting algorithm. The function has no side effects beyond reading the module-level sorter registries.
    """
    from spikeinterface.sorters.sorterlist import get_sorter_params_description
    return get_sorter_params_description(sorter_name_or_class)


################################################################################
# Source: spikeinterface.sorters.utils.misc.get_git_commit
# File: spikeinterface/sorters/utils/misc.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_sorters_utils_misc_get_git_commit(git_folder: str, shorten: bool = True):
    """Get commit to generate sorters version.
    
    Args:
        git_folder (str): Path to a Git repository folder as a string. In the SpikeInterface project and its sorter integrations, this is the filesystem directory where the sorter's source or the containing repository is located. The function runs the Git command "git rev-parse HEAD" with this directory as the current working directory to obtain the repository's current commit hash. If git_folder is None, the function immediately returns None (no external commands are invoked).
        shorten (bool): If True (default), return a shortened form of the commit identifier by taking the first 12 characters of the commit hash. This is useful in SpikeInterface for concise sorter version strings, logging, and metadata where the full 40-character SHA-1 is not required. If False, the full commit hash produced by "git rev-parse HEAD" (typically 40 hex characters) is returned.
    
    Behavior and side effects:
        The function invokes the external "git" command via subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=git_folder), decodes the resulting bytes using UTF-8, and strips trailing newline characters to produce the commit string. If shorten is True the function truncates the commit string to its first 12 characters. The function does not modify the repository contents; its only side effect is spawning a subprocess to run Git. If git_folder is None, no subprocess is spawned and the function returns None immediately.
    
    Failure modes and defaults:
        If git is not installed, the provided path is not a Git repository, the git command fails for any reason, or any other exception occurs during command execution or decoding, the function catches the exception and returns None. No exceptions are propagated to the caller. The default behavior (shorten=True) returns a 12-character prefix of the commit; callers that require the full commit hash should call get_git_commit(..., shorten=False).
    
    Returns:
        str or None: The repository commit identifier obtained from "git rev-parse HEAD" as a UTF-8 decoded string, shortened to 12 characters when shorten is True. Returns None if git_folder is None or if the git command fails for any reason (for example, git not installed, invalid repository path, or other subprocess errors). The returned value is typically used by SpikeInterface to generate sorter version metadata for reproducibility, logging, or packaging.
    """
    from spikeinterface.sorters.utils.misc import get_git_commit
    return get_git_commit(git_folder, shorten)


################################################################################
# Source: spikeinterface.sortingcomponents.clustering.isosplit_isocut.ensure_continuous_labels
# File: spikeinterface/sortingcomponents/clustering/isosplit_isocut.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_sortingcomponents_clustering_isosplit_isocut_ensure_continuous_labels(
    labels: numpy.ndarray
):
    """Ensure that a set of integer labels is remapped to a contiguous zero-based range [0, N-1].
    
    Args:
        labels (numpy.ndarray): Array of labels produced by a clustering or sorting step in SpikeInterface. In the SpikeInterface clustering/isotonic components (for example, isosplit/isocut), labels are often used as indices into arrays or as initial integer identifiers for clusters or spikes; this function remaps the existing label values to a compact, continuous set of integers starting at 0 so they can be used reliably as array indices or initial indices. The array is inspected with numpy.unique to determine the distinct label values; the original dtype is preserved when allocating the output buffer.
    
    Returns:
        numpy.ndarray: A new 1-D numpy array of length labels.size (shape (labels.size,)) containing remapped labels in the range 0..(K-1) where K is the number of unique values in the input. The mapping is ordered by numpy.unique (sorted order of the unique input values): the smallest unique input label is mapped to 0, the next to 1, and so on. The returned array is newly allocated (the input array is not modified) and uses the same dtype as the input labels array.
    
    Behavior, side effects, defaults, and failure modes:
        This function creates and returns a new array (no in-place modification of the input). It determines the unique label values via numpy.unique(labels) and assigns consecutive integer indices starting at 0 to each unique label value. Because the output buffer is created with dtype=labels.dtype, integer indices are cast to the input dtype when stored; if the input dtype cannot represent the integer indices without loss or cannot accept integer values, the result may be implicitly cast or may raise an error depending on numpy's casting rules. If labels is empty (labels.size == 0), an empty 1-D array of the same dtype is returned. If labels is not a numpy.ndarray, operations such as numpy.unique, attribute access to .size or .dtype, or boolean masking may raise exceptions; callers should pass a numpy.ndarray to avoid such errors. The ordering of the remapped labels follows numpy.unique and therefore depends on numpy's sorting behavior of the unique values. This function is intended to be used in SpikeInterface clustering/sorting components where label values must correspond to contiguous indices (for example, when a label value is used directly as an index into an array of per-cluster data).
    """
    from spikeinterface.sortingcomponents.clustering.isosplit_isocut import ensure_continuous_labels
    return ensure_continuous_labels(labels)


################################################################################
# Source: spikeinterface.sortingcomponents.clustering.isosplit_isocut.isosplit
# File: spikeinterface/sortingcomponents/clustering/isosplit_isocut.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_sortingcomponents_clustering_isosplit_isocut_isosplit(
    X: numpy.ndarray,
    initial_labels: numpy.ndarray = None,
    n_init: int = 200,
    max_iterations_per_pass: int = 500,
    min_cluster_size: int = 10,
    isocut_threshold: float = 2.0,
    seed: int = None
):
    """spikeinterface.sortingcomponents.clustering.isosplit_isocut.isosplit
    Performs clustering of feature vectors using a Python/numba implementation of the Isosplit algorithm (Jeremy Magland's isosplit6) tailored for spike sorting components. This function is typically used in spike sorting pipelines to cluster spike waveform features or low-dimensional embeddings so that putative neural units (clusters) are automatically discovered from extracellular recording features. The implementation first optionally initializes cluster assignments with k-means (many centroids) and then iteratively agglomerates clusters using a dip-test based "isocut" criterion to merge similar clusters and enforce a minimum cluster size.
    
    Args:
        X (numpy.ndarray): Input data matrix containing samples to cluster. Expected shape is (num_samples, num_dim), where num_samples is the number of spike-derived feature vectors (e.g., PCA components or waveform features) and num_dim is the dimensionality of each feature vector. The values and dtype in X are preserved during centroid and covariance computations; invalid shapes (not 2-D) or incompatible dtypes will raise standard NumPy errors from downstream numerical routines.
        initial_labels (numpy.ndarray): Optional initial integer label array of length num_samples to use as starting cluster assignments instead of performing the internal k-means initialization. When provided, these labels are normalized internally to contiguous 0-based integer labels. Use this parameter to resume clustering from a previous assignment or to provide a domain-specific initialization (for example, pre-clustered spike events). If None (default), the routine generates initial labels by running scipy.cluster.vq.kmeans2 with n_init centers. The provided array must have one label per row of X; mismatched lengths will result in standard array-shape errors.
        n_init (int): Initial number of k-means centroids to use when initial_labels is None. Default is 200. In the spike-sorting context this parameter controls the granularity of the initial partitioning of spike feature space: larger n_init can help separate fine-grained structure but increases runtime. The implementation guards against excessively large n_init relative to the number of samples and min_cluster_size: if n_init >= num_samples or n_init > num_samples // min_cluster_size, a warning is emitted and n_init is reduced to max(1, num_samples // (min_cluster_size * 2)).
        max_iterations_per_pass (int): Maximum number of pairwise comparison iterations allowed in a single agglomerative pass. Default is 500. Each pass attempts to merge cluster pairs and redistribute labels; if this limit is exceeded the function emits a warning ("isosplit : max iterations per pass exceeded") and proceeds to the next control step. This protects against pathological label oscillations or very slow convergence.
        min_cluster_size (int): Minimum allowed cluster size in number of samples. Default is 10. Clusters with fewer samples than min_cluster_size are considered too small and are merged with neighboring clusters during the agglomerative passes. In spike sorting, this helps avoid creating spurious units from a few noisy waveforms. Changing this value affects the algorithm's propensity to merge small clusters and therefore affects unit count stability.
        isocut_threshold (float): Threshold used by the isocut dip-test merge criterion. Default is 2.0. During pairwise cluster comparisons, a "dipscore" (measure from the isocut procedure) is computed; when dipscore < isocut_threshold the algorithm treats the pair as not sufficiently separated and merges them. Lowering this value makes merging stricter (fewer merges), raising it makes merging more permissive.
        seed (int): Optional random seed forwarded to scipy.cluster.vq.kmeans2 when initial_labels is None to control the k-means initialization. Default is None, which yields non-deterministic initialization subject to the execution environment's RNG. Providing an integer produces reproducible k-means initializations across runs (useful for deterministic pipeline behavior and reproducible spike-sorting experiments).
    
    Behavior and side effects:
        This function implements the Isosplit algorithm described in https://arxiv.org/abs/1508.04841 and the reference implementation at https://github.com/magland/isosplit6. When initial_labels is None, the function uses scipy.cluster.vq.kmeans2(minit="points") to compute an initial partition with n_init centroids; in the current Python/numba implementation, roughly half of execution time is typically spent in this kmeans2 step. The implementation includes safeguards that may reduce n_init and emit warnings when n_init is too large relative to the available samples and the requested min_cluster_size. The clustering proceeds in passes: within each pass the algorithm repeatedly selects closest pairs of active clusters that have not yet been compared in the pass, applies the isocut dip-test (and min_cluster_size logic) to decide merges/redistributions, updates centroids and covariance estimates for changed clusters, and records comparisons to avoid redundant tests. If any merges occur in a pass, another pass is performed; after convergence one final pass redistributes labels. If max_iterations_per_pass is exceeded, a warning is emitted and the iteration loop breaks for that pass. Internal state includes active label masks, centroids, covariance matrices, and a comparisons_made matrix; these are managed internally and do not persist outside the call.
    
    Performance and implementation notes:
        This is a pure Python/numba translation and is reported to be approximately 2x slower than the C++ implementation found in Magland's original code; the kmeans2 step is a common hotspot. Tuning n_init downward can reduce runtime at the risk of lowering initial over-segmentation. The function uses helper routines such as ensure_continuous_labels, compute_centroids_and_covmats, get_pairs_to_compare, and compare_pairs to implement the agglomerative and redistribution logic.
    
    Failure modes and warnings:
        The function will raise standard NumPy/SciPy errors for invalid inputs (for example, mismatched array lengths, non-2D X, or non-numeric data). The function emits warnings in several situations: when n_init is reduced because it was too large relative to the data, when kmeans2 fails to find the expected number of clusters (suppressed in the kmeans call but may still warn elsewhere), and when max_iterations_per_pass is exceeded. Because the algorithm relies on pairwise statistical tests and iterative merges, pathological datasets may result in many label changes or slower convergence; in such cases consider adjusting n_init, min_cluster_size, or isocut_threshold.
    
    Returns:
        labels (numpy.ndarray): Integer label array of length num_samples with the final 0-based contiguous cluster assignment for each input sample in X. The dtype is integer and labels are renumbered internally to ensure continuity (no gaps in label values). These labels represent putative neural units or clusters discovered by the Isosplit procedure and can be used downstream in spike sorting components for unit-level analysis, waveform extraction, or quality metric computation.
    """
    from spikeinterface.sortingcomponents.clustering.isosplit_isocut import isosplit
    return isosplit(
        X,
        initial_labels,
        n_init,
        max_iterations_per_pass,
        min_cluster_size,
        isocut_threshold,
        seed
    )


################################################################################
# Source: spikeinterface.sortingcomponents.clustering.merging_tools.merge_peak_labels_from_templates
# File: spikeinterface/sortingcomponents/clustering/merging_tools.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_sortingcomponents_clustering_merging_tools_merge_peak_labels_from_templates(
    peaks: numpy.ndarray,
    peak_labels: numpy.ndarray,
    unit_ids: list,
    templates_array: numpy.ndarray,
    template_sparse_mask: numpy.ndarray,
    similarity_metric: str = "l1",
    similarity_thresh: float = 0.8,
    num_shifts: int = 3,
    use_lags: bool = False
):
    """Low-level utility that merges peak labels by comparing template waveforms and grouping templates that exceed a similarity threshold. This function is used in the clustering and merging steps of SpikeInterface sorting components to clean likely oversplits (multiple units that should be a single unit) by computing pairwise template similarities and applying a merge mask to peak labels and templates. The function calls spikeinterface.postprocessing.template_similarity.compute_similarity_with_templates_array to compute a similarity matrix (and optional lag offsets) between templates and then delegates the application of the binary pairwise merge mask to an internal helper that rebuilds cleaned labels, merged template waveforms, and a merged sparsity mask.
    
    Args:
        peaks (numpy.ndarray): Array of detected peaks for the recording. In the SpikeInterface clustering pipeline this is typically a structured array or 2D array where each row corresponds to a detected peak event (for example containing time sample and channel). Note: this function does not read or modify peak waveform samples directly, but peaks is accepted for API compatibility with higher-level merging workflows and to document the association between labels and detected events. The length/number of peak rows should match peak_labels.
        peak_labels (numpy.ndarray): 1-D integer array of unit labels assigned to each entry in peaks prior to merging. Each element maps the corresponding peak (by index) to a unit id that is present in unit_ids. This array is used to produce clean_labels where some original labels may be replaced by merged unit ids. The returned clean_labels has the same shape as this input.
        unit_ids (list): List of unit identifiers (typically integers or strings) that index the rows of templates_array. The function asserts that len(unit_ids) == templates_array.shape[0]; a mismatch raises an AssertionError. The returned new_unit_ids is the list of unit identifiers after merging, preserving the domain-level mapping between template rows and unit identifiers used elsewhere in SpikeInterface.
        templates_array (numpy.ndarray): Dense template waveform array with shape (num_templates, num_total_channels, ...) or (num_templates, num_features) depending on calling context; in common SpikeInterface usage this is a 2-D array where the first dimension indexes templates (num_templates) and the remaining dimensions represent concatenated channel/time waveform samples per template. The original source documents this as a dense array shaped (num_templates, num_total_channel) with a companion template_sparse_mask. This array is used to compute pairwise similarities between templates and to rebuild merged templates after applying the pairwise merge mask.
        template_sparse_mask (numpy.ndarray): Companion sparsity mask for templates_array that indicates which channels (or channel/time entries) are present for each template. This mask is supplied as the sparsity argument to compute_similarity_with_templates_array and is propagated to the internal recompute routine. The function expects template_sparse_mask to be compatible with templates_array (same first-dimension length) and will pass it as both sparsity and other_sparsity to the similarity computation.
        similarity_metric (str): Similarity metric name passed to compute_similarity_with_templates_array to compute pairwise template similarity. Defaults to "l1". This string should be one of the methods supported by the underlying template similarity routine; an unsupported value will typically raise an error inside compute_similarity_with_templates_array. The chosen metric controls how waveform similarity is quantified when deciding merges.
        similarity_thresh (float): Threshold in the similarity metric space above which a pair of templates is considered for merging. Defaults to 0.8. After computing the pairwise similarity matrix, pairs with similarity > similarity_thresh are marked in the binary pair mask used to merge labels and recompute templates. Choosing a higher threshold makes merges more conservative (fewer merges); lower thresholds permit more aggressive merging.
        num_shifts (int): Number of temporal shifts considered when computing similarity between templates (passed to compute_similarity_with_templates_array). Defaults to 3. This controls how many relative time lags are evaluated to account for small temporal offsets between templates; larger values increase computational cost.
        use_lags (bool): If True, per-pair lag offsets returned by the similarity computation are passed to the label-application and template-recomputation step so merged templates may account for relative shifts. If False (default), lag information is ignored and set to None before recomputing merged templates. Use True when the merging decision should preserve estimated temporal offsets between templates; note that enabling lags affects how templates are recombined and may change the resulting merged waveforms.
    
    Returns:
        tuple: A 4-tuple containing:
            clean_labels (numpy.ndarray): Updated peak label array with the same shape as the input peak_labels. Labels that corresponded to templates marked for merging in the pair mask are replaced so that peaks belonging to merged templates share a single new unit id drawn from new_unit_ids. This output is used downstream in SpikeInterface clustering/curation to reflect merged units.
            merge_template_array (numpy.ndarray): Array of reconstructed template waveforms after applying the merge mask and (optionally) lags. The first dimension matches the length of new_unit_ids. This array provides the new canonical templates for the merged units and is suitable for use in subsequent postprocessing or visualization steps.
            merge_sparsity_mask (numpy.ndarray): Sparsity mask corresponding to merge_template_array. This mask indicates which channels/time entries are present for each merged template and follows the same sparsity conventions used by templates_array/template_sparse_mask.
            new_unit_ids (list): List of unit identifiers after merging, in the same order as the rows of merge_template_array. This list preserves the mapping between returned template rows and domain-level unit identifiers used in SpikeInterface.
    
    Behavior and side effects:
        The function computes an all-vs-all similarity matrix between templates_array rows using spikeinterface.postprocessing.template_similarity.compute_similarity_with_templates_array with method=similarity_metric, num_shifts=num_shifts, support="union", and the provided sparsity masks. It forms a binary pair_mask via similarity > similarity_thresh, optionally retains lag offsets if use_lags is True, and then calls the internal helper _apply_pair_mask_on_labels_and_recompute_templates to apply pairwise merges to peak_labels and to recompute merged templates and sparsity masks. There are no persistent side effects (no on-disk writes); all outputs are returned for the caller to consume. The function asserts that len(unit_ids) equals templates_array.shape[0] and will raise AssertionError on mismatch. Errors in the underlying similarity routine (for example due to invalid similarity_metric or incompatible array shapes) will propagate as exceptions.
    
    Failure modes and notes:
        A mismatched length between unit_ids and the number of template rows raises AssertionError. Invalid similarity_metric values, incompatible shapes between templates_array and template_sparse_mask, or other input inconsistencies will typically raise exceptions from compute_similarity_with_templates_array or the internal recomputation helper. Because peaks is not consulted in this implementation, callers should ensure peak_labels aligns with peaks externally; this function will still return clean_labels shaped like peak_labels. The choice of similarity_thresh and similarity_metric materially affects merging behavior: conservative thresholds reduce false merges but may leave true oversplits unmerged; aggressive thresholds risk merging distinct units.
    """
    from spikeinterface.sortingcomponents.clustering.merging_tools import merge_peak_labels_from_templates
    return merge_peak_labels_from_templates(
        peaks,
        peak_labels,
        unit_ids,
        templates_array,
        template_sparse_mask,
        similarity_metric,
        similarity_thresh,
        num_shifts,
        use_lags
    )


################################################################################
# Source: spikeinterface.sortingcomponents.clustering.tools.aggregate_sparse_features
# File: spikeinterface/sortingcomponents/clustering/tools.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_sortingcomponents_clustering_tools_aggregate_sparse_features(
    peaks: numpy.ndarray,
    peak_indices: numpy.ndarray,
    sparse_feature: numpy.ndarray,
    sparse_target_mask: numpy.ndarray,
    target_channels: numpy.ndarray
):
    """Aggregate sparse features that were computed on per-peak, possibly unaligned channel sets and realign them onto a consistent set of target channels.
    
    This function is used in SpikeInterface sorting components to align back per-peak extracted data (for example waveforms, PCA features, or TSVD features) when detections were performed on differing sparse channel sets. Given a global peaks array and a mapping from global channels to sparse feature channel indices, the function builds a new array of features with a fixed last dimension equal to the number of target_channels. The function does not remove peaks that lack the full set of target channels; instead it flags them so downstream code can decide how to handle them.
    
    Args:
        peaks (numpy.ndarray): Global array of detected peaks. In SpikeInterface this is typically a structured NumPy array of length N_peaks containing per-peak metadata. This function requires that peaks has an integer field named "channel_index" that gives the canonical/global channel index for each peak; that field is used to group peaks and to look up sparse channel mappings.
        peak_indices (numpy.ndarray): 1-D integer array selecting a subset of rows from peaks and corresponding rows in sparse_feature. This array defines the local set of peaks to process: local_peaks = peaks[peak_indices]. The order and values in peak_indices are used to index sparse_feature so the first axis of sparse_feature must correspond to the same global peak ordering referenced by peak_indices.
        sparse_feature (numpy.ndarray): Array containing per-peak features computed on sparse channel subsets. In the code this is indexed as sparse_feature[peak_index, :, sparse_channel_index]. The function expects a 3-D array where the first axis aligns with the global set of peaks (the same indexing as peak_indices), the second axis is the feature dimension (for example waveform sample points or PCA components), and the third axis enumerates the sparse channels used when computing those features. The dtype of sparse_feature is preserved in the returned aligned array.
        sparse_target_mask (numpy.ndarray): 2-D mask mapping global channels to sparse-channel indices. This array is indexed as sparse_target_mask[global_channel, :] and is used to compute sparse_chans = numpy.flatnonzero(sparse_target_mask[chan, :]). For each global channel index present in local_peaks, the function uses the corresponding row of sparse_target_mask to find which sparse-channel indices contributed to features for that global channel. If rows are empty or do not contain all target_channels, the corresponding peaks are flagged in dont_have_channels.
        target_channels (numpy.ndarray): 1-D array of global channel indices that define the desired, consistent channel ordering for the final aligned features. The size of target_channels determines the final channel axis length of the returned aligned_features. The function tests, for each peak group, whether all target_channels are available in the sparse mapping and uses that ordering to select the corresponding sparse feature channels when present.
    
    Behavior, side effects, defaults, and failure modes:
        The function first extracts local_peaks = peaks[peak_indices] and allocates aligned_features as a new NumPy array of shape (local_peaks.size, sparse_feature.shape[1], target_channels.size) filled with zeros and using sparse_feature.dtype. It also creates a boolean dont_have_channels of shape (peak_indices.size,) initialized to False. The function iterates over unique values of local_peaks["channel_index"]. For each such channel, it computes sparse_chans = numpy.flatnonzero(sparse_target_mask[chan, :]) and finds peak_inds, the indices within local_peaks that belong to that channel. If all target_channels are contained in sparse_chans, the function determines source_chans = numpy.flatnonzero(numpy.isin(sparse_chans, target_channels)) and copies sparse_feature rows corresponding to those peaks and source_chans into the aligned_features slice for those peaks. If any target_channels are missing for that channel, the function does not copy feature data for those peaks and instead sets dont_have_channels[peak_inds] = True. The function does not modify any of the input arrays; it returns new arrays. Peaks that lack required target channels are not removed; they are only flagged by dont_have_channels so that downstream code (for example filtering or imputation) can handle them.
    
        Failure modes:
        - If peak_indices contains values outside the valid range for peaks or sparse_feature, an IndexError will be raised when indexing.
        - If sparse_feature does not have at least three dimensions with the described semantics, indexing or shape assumptions may raise IndexError or produce incorrect results.
        - If sparse_target_mask does not index with the global channel indices present in peaks (for example wrong shape or missing rows), numpy.flatnonzero(sparse_target_mask[chan, :]) may return an empty array and those peaks will be flagged in dont_have_channels.
        - The function presumes that the first axis of sparse_feature corresponds to the same global peak ordering referenced by peak_indices; a mismatch will produce incorrect alignments.
    
    Returns:
        aligned_features (numpy.ndarray): New array of aligned features with shape (local_peaks.size, sparse_feature.shape[1], target_channels.size) and dtype equal to sparse_feature.dtype. For peaks that had all target_channels available in the sparse mapping, the corresponding slice contains the re-ordered sparse_feature values. For peaks missing required channels the corresponding slices remain zero (or the dtype zero value).
        dont_have_channels (numpy.ndarray): 1-D boolean array of shape (peak_indices.size,) where True indicates the corresponding peak (in peaks[peak_indices]) did not have all target_channels available in sparse_target_mask and therefore was not fully copied into aligned_features. This allows downstream code to filter out or otherwise handle incomplete peaks.
    """
    from spikeinterface.sortingcomponents.clustering.tools import aggregate_sparse_features
    return aggregate_sparse_features(
        peaks,
        peak_indices,
        sparse_feature,
        sparse_target_mask,
        target_channels
    )


################################################################################
# Source: spikeinterface.sortingcomponents.clustering.tools.apply_waveforms_shift
# File: spikeinterface/sortingcomponents/clustering/tools.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_sortingcomponents_clustering_tools_apply_waveforms_shift(
    waveforms: numpy.ndarray,
    peak_shifts: numpy.ndarray,
    inplace: bool = False
):
    """Apply a per-spike temporal shift to a batch of spike waveforms to realign their troughs before template computation in spike sorting workflows.
    
    This function is part of SpikeInterface sorting components (clustering tools) and is used when clusters or templates are merged but their constituent spikes have different trough (peak) time offsets. It shifts each waveform along the time axis so that spikes with early troughs are moved to the right and spikes with late troughs are moved to the left, facilitating correct averaging when computing templates after merge. Border time samples that would fall outside the original buffer are left unchanged. The operation can be performed in-place to avoid allocating a new array or on a copy to preserve the original buffer.
    
    Args:
        waveforms (numpy.ndarray): A 3-dimensional array of spike waveform snippets with shape (n_waveforms, n_timepoints, n_channels). Each entry represents the recorded voltage waveform around a detected spike. In the SpikeInterface domain, these waveforms are the per-spike buffers used to compute cluster templates or to perform further post-processing. The function preserves the original array shape and data type. The function assumes axis 1 is the temporal axis (n_timepoints) and axis 2 corresponds to channels.
        peak_shifts (numpy.ndarray): A 1-dimensional array of integer shifts with shape (n_waveforms,). Each element is the signed number of time samples to shift the corresponding waveform: a negative value means the trough was detected too early and the waveform will be moved toward the right, and a positive value means the trough was detected too late and the waveform will be moved toward the left. These shifts are applied per spike to realign troughs prior to template calculation. Values should be integers (or integer-valued) and the array length must match waveforms.shape[0].
        inplace (bool): If True, modify the input waveforms array in-place and return the same object reference (this avoids additional memory allocation and is useful when the caller no longer needs the original unshifted buffers). If False (default), operate on a copy of the input and return the new aligned array, leaving the input unmodified.
    
    Returns:
        aligned_waveforms (numpy.ndarray): A numpy.ndarray with the same shape as the input waveforms (n_waveforms, n_timepoints, n_channels). This array contains the time-shifted waveforms ready for operations such as template averaging after cluster merges. If inplace=True, this is the same object as the input waveforms (modified in-place); if inplace=False, this is a new array copy.
    
    Behavior, side effects, and failure modes:
        - The function computes unique shift values from peak_shifts and applies each distinct shift to the subset of waveforms having that shift.
        - Border samples that would be shifted outside the original temporal window are left unchanged; i.e., samples that cannot be replaced by shifted data remain as they were in the input buffer.
        - A shift of zero leaves the corresponding waveforms unchanged.
        - The function asserts that the maximum absolute shift is strictly less than the number of timepoints (waveforms.shape[1]); if this condition is false, an AssertionError is raised.
        - If peak_shifts length does not match waveforms.shape[0], or if peak_shifts contains non-integer values that cannot be used as slice indices, indexing errors or unexpected behavior may occur (for example, IndexError or TypeError). Callers should ensure compatible shapes and integer-valued shifts.
        - Using inplace=True will modify the provided waveforms buffer; callers should copy the array beforehand if the original unaltered data must be preserved.
    """
    from spikeinterface.sortingcomponents.clustering.tools import apply_waveforms_shift
    return apply_waveforms_shift(waveforms, peak_shifts, inplace)


################################################################################
# Source: spikeinterface.sortingcomponents.matching.circus.compress_templates
# File: spikeinterface/sortingcomponents/matching/circus.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_sortingcomponents_matching_circus_compress_templates(
    templates_array: numpy.ndarray,
    approx_rank: int,
    remove_mean: bool = False,
    return_new_templates: bool = True
):
    """spikeinterface.sortingcomponents.matching.circus.compress_templates compresses spike template waveforms using singular value decomposition (SVD) into separable temporal, singular (diagonal/spectral), and spatial components for use in spike sorting workflows. This function is used in the SpikeInterface sorting components to reduce the dimensionality of template waveforms (templates_array), which can speed template matching, storage, and downstream comparisons in sorter pipelines (for example when building or optimizing custom sorters or components such as Circus).
    
    Args:
        templates_array (numpy.ndarray): Array of spike template waveforms with shape (num_templates, num_samples, num_channels). Each entry templates_array[i] is a 2D waveform matrix for template i where rows correspond to time samples and columns correspond to recording channels. This array is read and may be modified in-place if remove_mean is True (see side effects). The function expects a numeric numpy array compatible with numpy.linalg.svd.
        approx_rank (int): Desired rank (number of components) for the compressed template representation. The function allocates output arrays whose third/second dimension equals approx_rank and fills up to min(approx_rank, available_rank) components from each template's SVD. If approx_rank is larger than the intrinsic rank determined by num_samples and num_channels, only the available SVD components are filled and the remaining entries remain zero. approx_rank controls the compression level: smaller values reduce storage and computation at the cost of approximation error.
        remove_mean (bool): If True, subtract the mean value of each template across time and channels (mean computed over axes (1, 2) per template) before performing SVD. Default False. This subtraction is performed in-place via templates_array -= mean, so the original templates_array passed by the caller will be mutated when remove_mean is True.
        return_new_templates (bool): If True (default), the function reconstructs and returns a new templates_array computed from the compressed components (temporal, singular, spatial) using matrix multiplication. If False, the reconstructed templates are not returned and the fourth element of the returned tuple is None.
    
    Returns:
        tuple: A 4-tuple of numpy.ndarray or None representing the compressed components and an optional reconstructed template array:
            temporal (numpy.ndarray): Array of temporal components with shape (num_templates, num_samples, approx_rank) and dtype numpy.float32. For each template i, temporal[i, :, k] is the k-th temporal singular vector (time course) used in the low-rank reconstruction. Only the first min(approx_rank, available_components) entries along the approx_rank axis are filled from SVD; remaining slots (if any) are left as zeros.
            singular (numpy.ndarray): Array of singular values with shape (num_templates, approx_rank) and dtype numpy.float32. For each template i, singular[i, k] is the k-th singular value from the template's SVD. Values beyond the available SVD components are zero when approx_rank exceeds the available component count.
            spatial (numpy.ndarray): Array of spatial components with shape (num_templates, approx_rank, num_channels) and dtype numpy.float32. For each template i, spatial[i, k, :] is the k-th spatial singular vector across channels.
            templates_array (numpy.ndarray or None): If return_new_templates is True, a reconstructed templates array with the same shape as the input (num_templates, num_samples, num_channels) computed as numpy.matmul(temporal * singular[:, numpy.newaxis, :], spatial), dtype numpy.float32. If return_new_templates is False, this value is None. Note that if remove_mean was True, the input templates_array has already been modified in-place prior to SVD; the returned reconstructed array (when requested) is produced from the compressed components and replaces the local templates_array variable but does not restore any original pre-mean-subtraction data unless the caller had preserved a copy.
    
    Behavior and side effects:
        The function iterates over templates and computes numpy.linalg.svd(template_matrix, full_matrices=False) for each template independently, storing the first approx_rank components (or fewer if limited by the template's intrinsic rank). The arrays temporal, singular, and spatial are allocated with dtype numpy.float32. If remove_mean is True the input templates_array is modified in-place by subtracting each template's scalar mean across time and channels. If return_new_templates is True, a reconstructed template array is returned as the fourth element; otherwise the fourth element is None. The function does not perform explicit validation of shapes or of approx_rank; when approx_rank exceeds available components, only available SVD outputs are assigned and the remaining entries remain zero.
    
    Failure modes and errors:
        numpy.linalg.svd may raise numpy.linalg.LinAlgError if the SVD does not converge for any template; such exceptions propagate to the caller. Large inputs may raise MemoryError or be computationally expensive because an SVD is computed per template in a Python loop. The function relies on numpy broadcasting rules when reconstructing templates and assumes the input is compatible with numpy SVD routines.
    """
    from spikeinterface.sortingcomponents.matching.circus import compress_templates
    return compress_templates(
        templates_array,
        approx_rank,
        remove_mean,
        return_new_templates
    )


################################################################################
# Source: spikeinterface.sortingcomponents.matching.tdc_peeler.fit_one_amplitude_with_neighbors
# File: spikeinterface/sortingcomponents/matching/tdc_peeler.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_sortingcomponents_matching_tdc_peeler_fit_one_amplitude_with_neighbors(
    spike: numpy.ndarray,
    neighbors_spikes: numpy.ndarray,
    traces: numpy.ndarray,
    template_sparsity_mask: numpy.ndarray,
    sparse_templates_array: numpy.ndarray,
    template_norms: numpy.ndarray,
    nbefore: int,
    nafter: int
):
    """Fit amplitude for a single spike using its template and optionally fit or subtract contributions from neighboring spikes to account for temporal overlaps in spike sorting deconvolution.
    
    This function is used in SpikeInterface's matching/peeler components (tdc_peeler) to estimate the scalar amplitude of one detected spike relative to a stored sparse template. It either computes a simple dot-product projection when there are no neighbors, or builds a local linear model (design matrix) that includes the target spike and selected neighbor regressors, then solves a linear least-squares problem to jointly estimate amplitudes. The estimate is returned as a single float amplitude used downstream by the peeler to scale the template when subtracting predicted waveforms from the raw traces.
    
    Args:
        spike (numpy.ndarray): Structured numpy array element describing the target spike to fit. The function expects at least the fields "cluster_index" (int, index of the template/cluster) and "sample_index" (int, sample/time index of the spike). If present, the "amplitude" field is read/assigned by the caller code but this function only reads and copies the entry. This argument identifies which template to use (cluster_index) and where to center the temporal window (sample_index) when extracting traces and constructing regressors for fitting.
        neighbors_spikes (numpy.ndarray): Structured numpy array of neighboring spikes (may be None or empty). Each entry is expected to have fields "sample_index" (int), "amplitude" (float) and "cluster_index" (int). When None or empty, no neighbor modeling is performed and the amplitude is computed by projecting the template onto the extracted waveform. When provided, neighbors are used to extend the fitting window; neighbors with amplitude == 0.0 and cluster_index >= 0 are treated as additional unknown regressors whose amplitudes are fitted jointly, and neighbors with nonzero amplitude may be removed from the local traces as known contributions.
        traces (numpy.ndarray): 2D raw recording traces array (time x channels). The function extracts a time window around the spike (and an expanded window if neighbors exist) and selects channels according to template_sparsity_mask. Values are used as the observations y in the linear system solved for amplitudes.
        template_sparsity_mask (numpy.ndarray): 2D boolean or integer mask indexed by [template_index, channel]. For a given cluster_index from spike, template_sparsity_mask[cluster_index, :] selects the channels that are part of the sparse template for that cluster. The mask determines which columns of traces and which channels of sparse_templates_array are used in the projection or in constructing the design matrix.
        sparse_templates_array (numpy.ndarray): Array storing sparse templates. The code indexes this as sparse_templates_array[cluster_index, :, :num_chans] to obtain the template waveform for the selected channels. The array provides the template time × channel waveforms (sparsely stored) that are used as regressors to predict traces and compute dot products.
        template_norms (numpy.ndarray): 1D array of template norm values indexed by cluster_index. When computing a simple projection without neighbors, the dot product between template and waveform window is normalized by template_norms[cluster_index]. If template_norms[cluster_index] == 0.0, the function treats the template as effectively empty and returns 0.0 immediately.
        nbefore (int): Number of samples before the spike's sample_index to include in the temporal window used for template extraction and regression construction. Used both when extracting the target waveform and when building predicted contributions in construct_prediction_sparse.
        nafter (int): Number of samples after the spike's sample_index to include in the temporal window used for template extraction and regression construction.
    
    Behavior and implementation details:
        - If the selected template has zero active channels (sum of chan_sparsity_mask == 0) or template_norms[cluster_index] == 0.0, the function returns 0.0 immediately to protect against degenerate/empty templates.
        - When neighbors_spikes is None or empty, the function:
            - extracts the waveform window traces[start:stop, :] where start = sample_index - nbefore and stop = sample_index + nafter,
            - selects channels using the per-template sparsity mask,
            - loads the corresponding sparse template for cluster_index and computes the scalar amplitude as the normalized dot product: sum(template * wf) / template_norms[cluster_index].
        - When neighbors_spikes is provided and non-empty, the function:
            - computes an expanded local window [lim0:lim1] that covers the target spike window and all neighbor windows (using neighbors' sample_index ± nbefore/nafter),
            - creates a local copy of traces for the selected channels and copies of spike and neighbors to shift their sample indices into the local window,
            - constructs a design tensor x with one regressor for the target spike and additional regressors for neighbors whose "amplitude" == 0.0 and "cluster_index" >= 0 (these neighbors are fitted jointly); neighbors with nonzero amplitude are subtracted from the local_traces as known contributions,
            - reshapes x and the flattened local_traces into a conventional 2D linear system y = X a and solves for a using scipy.linalg.lstsq (driver "gelsd"),
            - returns the first fitted amplitude corresponding to the target spike.
        - The function calls construct_prediction_sparse (internal helper) to build predicted waveform columns for each spike regressor; this helper is responsible for mapping templates into the appropriate local time/channel slices.
        - The returned amplitude is the estimated scalar multiplier for the template associated with spike["cluster_index"] that best explains the observed local traces in a least-squares sense (or the simple projection result when no neighbors are present).
    
    Side effects and mutability:
        - The function does not modify the provided traces, sparse_templates_array, template_sparsity_mask, or template_norms in-place. It makes local copies of spike and neighbors_spikes before modifying sample_index/amplitude fields for local computations.
        - No global state is modified.
    
    Failure modes and edge cases:
        - Returns 0.0 if the template is empty for the cluster (no channels active) or if template_norms[cluster_index] == 0.0.
        - If neighbors_spikes is provided but does not contain the expected structured fields ("sample_index", "amplitude", "cluster_index") or if their values lead to invalid array indexing relative to traces, the function will raise the corresponding numpy KeyError or IndexError.
        - The linear least-squares step may produce numerically unstable or poorly constrained amplitude estimates if the design matrix is degenerate (e.g., highly colinear regressors); such numerical issues are delegated to scipy.linalg.lstsq and may result in large or noisy amplitude estimates.
        - The function assumes that nbefore and nafter define a valid window inside traces when combined with spike and neighbors; callers should ensure traces contains the required samples for the windows computed by this function.
    
    Returns:
        float: Estimated scalar amplitude for the input spike's template. This is the fitted amplitude (first coefficient) from either the simple projection (no neighbors) or the joint least-squares fit (with neighbors). When the template is empty or has zero norm, the function returns 0.0.
    """
    from spikeinterface.sortingcomponents.matching.tdc_peeler import fit_one_amplitude_with_neighbors
    return fit_one_amplitude_with_neighbors(
        spike,
        neighbors_spikes,
        traces,
        template_sparsity_mask,
        sparse_templates_array,
        template_norms,
        nbefore,
        nafter
    )


################################################################################
# Source: spikeinterface.sortingcomponents.matching.wobble.compress_templates
# File: spikeinterface/sortingcomponents/matching/wobble.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_sortingcomponents_matching_wobble_compress_templates(
    templates: numpy.ndarray,
    approx_rank: int
):
    """Compress templates using singular value decomposition for use in spike sorting workflows.
    
    This function, spikeinterface.sortingcomponents.matching.wobble.compress_templates, computes a reduced-rank representation of input spike template waveforms by applying NumPy's singular value decomposition (numpy.linalg.svd with full_matrices=False) independently to each template matrix. In the SpikeInterface domain this is used to reduce temporal and spatial dimensionality of template waveforms (num_templates, num_samples, num_channels) to accelerate matching, reduce memory use, and enable low-rank operations in downstream sorting and post-processing components.
    
    Args:
        templates (numpy.ndarray): Spike template waveforms with shape (num_templates, num_samples, num_channels). Each entry templates[i] is a 2-D matrix representing the i-th template's waveform over time (num_samples) and recording channels (num_channels). This input is the data to be compressed; it must be a numeric ndarray with a leading dimension indexing templates. If templates is not a 3-D numeric array, NumPy will raise an error when computing the SVD.
        approx_rank (int): Desired rank of the compressed template matrices. This integer is treated as the maximum number of singular components to keep for each template. The effective retained rank is min(approx_rank, min(num_samples, num_channels)) because SVD cannot produce more components than the smaller matrix dimension. approx_rank should be a positive integer for meaningful compression; nonpositive values will result in empty component arrays for the retained-dimension axis.
    
    Returns:
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: A tuple (temporal, singular, spatial) containing the compressed SVD components for every template. temporal is a numpy.ndarray of dtype float32 with shape (num_templates, num_samples, r_eff) where r_eff = min(approx_rank, min(num_samples, num_channels)); temporal holds the left-singular vectors (U) for each template but with the sample axis reversed (the function applies numpy.flip along the sample axis). singular is a numpy.ndarray of dtype float32 with shape (num_templates, r_eff) containing the singular values for each template (the diagonal entries typically used in low-rank reconstruction). spatial is a numpy.ndarray of dtype float32 with shape (num_templates, r_eff, num_channels) containing the right-singular vectors (V^H) for each template. Together these components allow approximate reconstruction of the original templates: for template index i, reconstruct the i-th template as U_i_orig @ diag(s_i) @ Vh_i where U_i_orig is obtained by flipping the returned temporal[i] back along the sample axis (i.e., U_i_orig = numpy.flip(temporal[i], axis=0)), diag(s_i) is formed from singular[i], and Vh_i is spatial[i].
    
    Behavior and side effects:
        This function performs an SVD on each template matrix and casts outputs to numpy.float32. It uses numpy.linalg.svd with full_matrices=False and then truncates to at most approx_rank components. The temporal component is flipped along the sample axis after truncation; callers must flip it back when performing exact matrix reconstruction as described above. There are no other side effects (no in-place modification of the input array); the function returns newly allocated arrays.
    
    Failure modes and performance considerations:
        If templates is not a 3-D numeric ndarray, NumPy will raise an exception (ValueError or LinAlgError) when calling numpy.linalg.svd. If approx_rank is larger than the allowable SVD rank, the function silently uses the maximum available rank (min(num_samples, num_channels)). SVD is computationally and memory intensive: runtime and memory scale with the number of templates and the template matrix sizes (num_samples and num_channels). For large datasets, consider lowering approx_rank or using more efficient/batched SVD strategies before calling this function.
    """
    from spikeinterface.sortingcomponents.matching.wobble import compress_templates
    return compress_templates(templates, approx_rank)


################################################################################
# Source: spikeinterface.sortingcomponents.matching.wobble.compute_scale_amplitudes
# File: spikeinterface/sortingcomponents/matching/wobble.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_sortingcomponents_matching_wobble_compute_scale_amplitudes(
    high_resolution_conv: numpy.ndarray,
    norm_peaks: numpy.ndarray,
    scale_min: float,
    scale_max: float,
    amplitude_variance: float
):
    """Compute optimal amplitude scaling factors for spikes and the resulting high-resolution objective used in spike-template matching.
    
    This function is used in spike sorting matching components to adjust template amplitudes for each detected spike and to compute a per-timepoint objective (super-resolution objective) within a small time window around each spike peak. It implements a numerically stable form of the closed-form solution for the amplitude that maximizes a quadratic objective with a Gaussian prior on amplitudes (amplitude scaling factor ~ N(1, amplitude_variance)). To avoid overflow from squaring large intermediate values, the computed scaling b/a is clipped to the provided [scale_min, scale_max] interval; the clipped scaling is then substituted into the objective formula implemented here. The outputs are intended for downstream matching/score computations that operate on super-resolved convolution traces and per-spike template norms.
    
    Args:
        high_resolution_conv (numpy.ndarray): Super-resolution upsampled convolution of the spike templates with the recorded traces restricted to a small time window around each spike peak. Practically, this is the per-superresolution-timepoint match between templates and data used to compute the matching objective. The array must be broadcast-compatible with norm_peaks so that norm_peaks aligns with the second axis (typical shape is (n_timepoints, num_spikes) where the second dimension corresponds to the spikes for which amplitudes are estimated). Values are floating point and represent convolution scores at super-resolved time offsets.
        norm_peaks (numpy.ndarray): Per-spike template magnitude used as the quadratic coefficient in the closed-form amplitude solution. This is a 1-D array of length num_spikes whose elements quantify the template energy (norm) associated with each spike in the spike train; it is used as the denominator term when computing optimal scaling and is broadcast across the time window of high_resolution_conv.
        scale_min (float): Lower bound for the allowed amplitude scaling factor. This hard clip prevents arbitrarily small scaling values and is applied elementwise to the unconstrained optimum (b / a). It is used to improve numerical stability and to encode prior knowledge about plausible minimum template amplitude relative to the template baseline.
        scale_max (float): Upper bound for the allowed amplitude scaling factor. This hard clip prevents arbitrarily large scaling values and is applied elementwise to the unconstrained optimum (b / a). It is used to improve numerical stability and to encode prior knowledge about plausible maximum template amplitude relative to the template baseline.
        amplitude_variance (float): Prior variance for the multiplicative amplitude scaling of templates (the model assumes amplitude scaling ~ N(1, amplitude_variance)). This scalar controls the strength of the Gaussian prior (smaller values give stronger regularization toward 1). It must be a positive finite float; amplitude_variance <= 0 will cause division by zero or invalid calculations and is therefore unsupported.
    
    Behavior, side effects, and failure modes:
        The function computes intermediate arrays b = high_resolution_conv + 1 / amplitude_variance and a = norm_peaks[numpy.newaxis, :] + 1 / amplitude_variance, then computes the unconstrained optimal scaling as b / a. To avoid numerical overflow from squaring large values, the unconstrained scaling is clipped to the interval [scale_min, scale_max] via numpy.clip; if scale_min > scale_max numpy.clip will raise a ValueError. The final high-resolution objective is computed as 2 * b * scaling - (scaling**2 * a) - 1 / amplitude_variance, which is equivalent to the regularized quadratic objective evaluated at the clipped scaling. There are no in-place modifications of inputs; the function returns new arrays. Typical failure modes include: amplitude_variance <= 0 (division by zero or invalid prior), incompatible shapes between high_resolution_conv and norm_peaks that prevent broadcasting (raises a broadcasting error), or invalid clip bounds (scale_min > scale_max leads to a numpy ValueError). The function assumes numeric (floating) inputs; non-numeric types will raise type errors from numpy operations.
    
    Returns:
        high_res_objective (numpy.ndarray): Super-resolution upsampled objective values computed for the same small time window around each spike peak. Shape and alignment match high_resolution_conv (for typical usage shape is (n_timepoints, num_spikes)). These objective values quantify the fit (score) of each scaled template to the data at each super-resolved timepoint under the Gaussian amplitude prior and are used by downstream matching/ranking code in the spike sorting pipeline.
        scalings (numpy.ndarray): Per-spike amplitude scaling factors actually used after clipping. This is a 1-D array of shape (num_spikes,) containing the amplitude multiplier applied to each template for all timepoints in the returned high_res_objective. These values are the clipped form of the closed-form optimum (b / a) and represent the posterior-regularized amplitude estimate for each spike.
    """
    from spikeinterface.sortingcomponents.matching.wobble import compute_scale_amplitudes
    return compute_scale_amplitudes(
        high_resolution_conv,
        norm_peaks,
        scale_min,
        scale_max,
        amplitude_variance
    )


################################################################################
# Source: spikeinterface.sortingcomponents.matching.wobble.compute_template_norm
# File: spikeinterface/sortingcomponents/matching/wobble.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_sortingcomponents_matching_wobble_compute_template_norm(
    visible_channels: numpy.ndarray,
    templates: numpy.ndarray
):
    """Compute the squared L2 norm of each spike template restricted to its visible channels.
    
    This function is used in the spike matching/wobble components of SpikeInterface to produce a per-template magnitude used for normalization and comparison of templates during matching and post-processing. For each template (unit), only channels marked as visible in visible_channels are included in the squared-norm computation; the result is therefore the sum of squared waveform samples across time and the template-specific subset of channels.
    
    Args:
        visible_channels (ndarray): Boolean mask array of shape (num_templates, num_channels). visible_channels[template, channel] is True if the template for that template index has sufficient amplitude on that channel and should be included in the norm calculation. This mask must have the same number of rows as templates.shape[0] and the same number of columns as templates.shape[2]; if the shapes do not align an IndexError or ValueError will be raised. The practical significance of this argument in the spike-sorting domain is to ignore channels with negligible signal or excessive noise on a per-template basis so that normalization and matching operate on relevant channels only.
        templates (ndarray): Array of spike template waveforms with shape (num_templates, num_samples, num_channels). Each entry templates[t, s, c] is the waveform sample at time index s for template t on channel c. This argument supplies the raw waveform data whose per-template magnitudes are computed; it must be a 3-dimensional numeric array where the first dimension is the template (unit) index.
    
    Returns:
        ndarray (num_templates,): A 1D array of dtype float32 containing the squared L2 norm (sum of squared samples) for each template, restricted to the channels indicated by visible_channels for that template. The returned array is intended to be used as the template magnitude for normalization in matching algorithms. The function has no side effects and does not modify its inputs.
    
    Behavior and failure modes:
        The implementation iterates over templates and computes numpy.sum(numpy.square(templates[i, :, visible_channels[i, :]])) for each template index i. If visible_channels contains non-boolean values, NumPy will interpret them as an index array which may produce unexpected behavior; to ensure correct semantics provide a boolean mask. If templates contains NaN or Inf values, the corresponding entries in the returned norm will be NaN or Inf. Large inputs can incur substantial memory and CPU cost proportional to num_templates * num_samples * num_visible_channels; ensure available resources for large recordings.
    """
    from spikeinterface.sortingcomponents.matching.wobble import compute_template_norm
    return compute_template_norm(visible_channels, templates)


################################################################################
# Source: spikeinterface.sortingcomponents.matching.wobble.get_convolution_len
# File: spikeinterface/sortingcomponents/matching/wobble.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_sortingcomponents_matching_wobble_get_convolution_len(x: int, y: int):
    """Returns the length of the linear discrete convolution of two finite-length vectors.
    
    This function computes the number of output samples produced when two one-dimensional sequences of lengths x and y are convolved using the standard linear discrete convolution formula. In the SpikeInterface sortingcomponents.matching.wobble context this value is commonly used to determine buffer sizes, allocate arrays, and align or index convolved waveforms or templates used for template matching, cross-correlation, or wobble/alignment corrections in spike sorting pipelines.
    
    Args:
        x (int): Length in samples of the first input vector (for example, the number of time samples in a waveform or template segment). This parameter represents a count of samples and therefore should be provided as an integer. In the spike-sorting domain it corresponds to the temporal extent of one sequence involved in convolution.
        y (int): Length in samples of the second input vector (for example, the number of time samples in a kernel or another template). This parameter represents a count of samples and therefore should be provided as an integer. In the spike-sorting domain it corresponds to the temporal extent of the other sequence involved in convolution.
    
    Behavior and failure modes:
        The function returns the arithmetic result x + y - 1 following the standard formula for the length of the linear convolution of two finite discrete sequences. There are no side effects. The function assumes x and y represent non-negative integer sample counts; providing negative integers will produce a numeric result but is semantically invalid for sequence lengths and likely indicates a calling error. While Python integers are arbitrary-precision and will not overflow, extremely large returned values can imply very large memory allocations downstream when creating arrays to hold convolution results, which can lead to MemoryError or degraded performance in practical spike-sorting workflows. Inputs of types other than int are not semantically valid for sequence lengths; passing non-integer types may produce a TypeError elsewhere in consuming code or an unexpected arithmetic result here.
    
    Returns:
        int: The length of the convolution output in samples, equal to x + y - 1. This return value is intended to be used to size buffers and arrays and to compute indexing offsets when performing waveform/template convolution, template matching, or wobble alignment operations in SpikeInterface.
    """
    from spikeinterface.sortingcomponents.matching.wobble import get_convolution_len
    return get_convolution_len(x, y)


################################################################################
# Source: spikeinterface.sortingcomponents.matching.wobble.upsample_and_jitter
# File: spikeinterface/sortingcomponents/matching/wobble.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_sortingcomponents_matching_wobble_upsample_and_jitter(
    temporal: numpy.ndarray,
    jitter_factor: int,
    num_samples: int
):
    """spikeinterface.sortingcomponents.matching.wobble.upsample_and_jitter
    Upsample the temporal components of template SVDs and re-index them to produce jittered,
    super-resolution temporal templates used for sub-sample template matching in spike sorting.
    
    This function is used in SpikeInterface's matching/wobble workflow to generate multiple
    time-shifted ("jittered") versions of each template's temporal components so matchers can
    evaluate alignments at fractional-sample offsets. It performs an FFT-based resampling
    (using scipy.signal.resample) to a higher temporal resolution, selects shifted samples
    that correspond to different sub-sample alignments, and returns a stack of jittered
    temporal components suitable for reconstructing jittered templates during template-matching
    or evaluation in sorting components and custom sorters built with SpikeInterface.
    
    Args:
        temporal (ndarray (num_templates, num_samples, approx_rank)): Temporal components of the templates.
            A 3D array containing, for each template, the temporal basis vectors (e.g., from an SVD
            compression) over the template duration. num_templates is the number of distinct templates,
            num_samples is the template duration in samples/frames (the provided num_samples argument
            should match temporal.shape[1]), and approx_rank is the number of temporal components per
            template. These temporal components are the input used to reconstruct template waveforms
            and to generate jittered (time-shifted) reconstructions for sub-sample alignment.
        jitter_factor (int): Number of upsampled jittered templates to create for each provided template.
            A positive integer factor by which the temporal axis is upsampled (super-resolution).
            The function returns jitter_factor versions of each input template, shifted by 0..(jitter_factor-1)
            sub-sample offsets. If jitter_factor == 1 the function returns the input temporal array unchanged
            (trivial case). The implementation expects an integer >= 1; passing values < 1 or non-integers
            will lead to incorrect behavior or raised exceptions.
        num_samples (int): Template duration in samples/frames.
            The nominal number of samples per template at the original sampling resolution. This value
            should match temporal.shape[1]; it is used to compute the number of samples in the
            output jittered templates and to reshape the upsampled data back to (num_jittered, num_samples, approx_rank).
    
    Returns:
        ndarray (num_jittered, num_samples, approx_rank): Temporal component of the compressed templates
        jittered at super-resolution in time.
        The returned array contains num_jittered == num_templates * jitter_factor entries along the
        first axis: for each original template, jitter_factor shifted versions are produced. Each entry
        has num_samples samples (the same nominal template duration) and approx_rank temporal components.
        The output can be used directly to reconstruct jittered template waveforms for sub-sample
        template matching in SpikeInterface matching/wobble routines.
    
    Behavior, side effects, defaults, and failure modes:
        - Trivial fast-path: if jitter_factor == 1 the function returns the original temporal
          array object immediately (no copy), preserving dtype and shape; callers relying on a copy
          should explicitly copy the returned array.
        - Upsampling method: the implementation flips the temporal axis, uses scipy.signal.resample
          to produce num_samples * jitter_factor super-resolved samples (FFT-based resampling), then
          selects shifted sample indices and flips back to the original orientation. The code includes
          an internal flip operation (present in the implementation) that must be preserved for correct
          alignment; the original implementation contains a TODO about why the flip is required, so
          callers should not remove this flip unless they validate the alignment results.
        - Numerical considerations: scipy.signal.resample performs FFT-based interpolation; this can
          introduce small numerical differences (e.g., ringing or boundary effects) compared to other
          interpolation methods. Users concerned about numerical artifacts should validate results on
          representative templates.
        - Dependency: the function imports scipy.signal at runtime. If SciPy is not available an ImportError
          will be raised when this function is executed.
        - Input validation and shape requirements: the implementation assumes temporal is a 3D ndarray
          with shape (num_templates, num_samples, approx_rank) and that the provided num_samples equals
          temporal.shape[1]. If these assumptions are violated, the indexing and reshape steps can raise
          IndexError or ValueError (e.g., during reshape). The function does not perform explicit type
          coercion; supplying mismatched dtypes, non-array inputs, or incompatible shapes will result in
          Python or NumPy errors.
        - Performance and memory: upsampling by jitter_factor increases memory usage by roughly jitter_factor
          (the returned array has num_templates * jitter_factor entries). For large numbers of templates,
          large approx_rank, or large jitter_factor, memory and compute costs can become significant.
        - Intended usage: the output is intended for use in spike sorting matching components that require
          evaluating template matches at sub-sample offsets (wobble/misalignment compensation) and for
          building custom sorters or post-processing that rely on jittered template reconstructions.
    """
    from spikeinterface.sortingcomponents.matching.wobble import upsample_and_jitter
    return upsample_and_jitter(temporal, jitter_factor, num_samples)


################################################################################
# Source: spikeinterface.sortingcomponents.motion.decentralized.compute_global_displacement
# File: spikeinterface/sortingcomponents/motion/decentralized.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_sortingcomponents_motion_decentralized_compute_global_displacement(
    pairwise_displacement: numpy.ndarray,
    pairwise_displacement_weight: numpy.ndarray = None,
    sparse_mask: numpy.ndarray = None,
    temporal_prior: bool = True,
    spatial_prior: bool = True,
    soft_weights: bool = False,
    convergence_method: str = "lsmr",
    robust_regression_sigma: float = 2,
    lsqr_robust_n_iter: int = 20,
    progress_bar: bool = False
):
    """Compute global displacement across time windows for motion correction in spike sorting.
    
    This function estimates a global temporal displacement signal (motion) from pairwise displacement measurements between timepoints or between time windows. It is used in SpikeInterface's sortingcomponents.motion.decentralized pipeline to convert pairwise relative displacements (for example, offsets estimated between pairs of temporal windows of an extracellular recording) into a consistent global displacement per timepoint (and per window when multiple windows/blocks are provided). The algorithm supports multiple numerical solvers (gradient-based, robust least-squares iterations with LSQR, and sparse LSMR with optional temporal and spatial priors), optional pairwise weighting and sparsity masks, and optional robust trimming of outliers. The returned displacement values are in the same numerical units as pairwise_displacement.
    
    Args:
        pairwise_displacement (numpy.ndarray): Array of pairwise relative displacements provided by a lower-level motion estimation routine. For the simple rigid case this is a 2-D array of shape (T, T) where T is the number of temporal points/windows and entry (i, j) encodes the measured displacement from j to i (i.e., value approximating p[i] - p[j]). For nonrigid/multi-window use, this may be a 3-D array of shape (B, T, T) where B is the number of blocks/windows and each block contains a (T, T) pairwise displacement matrix. The function uses this input as the measurement (likelihood) term and preserves its numerical units in the output. The function asserts that the array dimensions are consistent with any provided weight/mask arrays and will expand a 2-D array to a single-window 3-D representation internally for the LSMR solver.
    
        pairwise_displacement_weight (numpy.ndarray): Optional array of the same shape as pairwise_displacement containing multiplicative weights for each pairwise entry. Weights increase or decrease the influence of individual pairwise measurements in the global estimation. If None (default), equal unit weights are assumed. Both dense numpy arrays and scipy.sparse.csr_matrix are accepted in some branches; when a sparse matrix is provided it may be converted to dense in the LSMR branch, which can increase memory usage. When provided together with sparse_mask, the effective weight is pairwise_displacement_weight * sparse_mask.
    
        sparse_mask (numpy.ndarray): Optional boolean or numeric mask (same shape as pairwise_displacement) indicating which pairwise entries are present/active. Entries with zero in sparse_mask are treated as missing and ignored. If None (default), all entries are considered active. Used together with pairwise_displacement_weight to form an effective weight mask. Supplying a sparse mask is advisable when pairwise measurements are missing or when the pairwise matrix is large and mostly empty; however, some solver branches convert sparse inputs to dense arrays and so memory usage can increase.
    
        temporal_prior (bool): When True (default) and when using the "lsmr" convergence_method, a temporal smoothness prior is added per block/window. This prior penalizes large differences between consecutive timepoints within the same block and is implemented by stacking a finite-difference operator into the least-squares system. Enabling this prior biases the solution toward temporally smoother displacement traces and is useful when motion is expected to vary smoothly in time. It has no effect in solver branches that do not construct the LSMR sparse system.
    
        spatial_prior (bool): When True (default) and when using the "lsmr" convergence_method with multiple blocks (B > 1), a spatial/block smoothness prior is added that penalizes differences between corresponding timepoints across adjacent blocks. This encourages consistency across blocks/windows (spatial smoothness in the block axis) and is implemented as an additional sparse operator in the LSMR system. It has no effect for single-block inputs or in solver branches that do not use the LSMR sparse formulation.
    
        soft_weights (bool): When False (default), the LSMR branch treats all active pairwise entries as equally weighted (aside from pairwise_displacement_weight). When True, and when using the "lsmr" convergence_method, the per-pair entries in the weight array are used directly as the pairwise coefficients (soft weighting) instead of uniform coefficients. This changes how the sparse coefficient matrices are constructed and therefore affects the relative influence of different pairwise measurements on the global solution.
    
        convergence_method (str): String selecting the numerical method used to compute the global displacement. Supported values implemented in this function are "gradient_descent", "lsqr_robust", and "lsmr". Default is "lsmr". "gradient_descent" uses a dense formulation and L-BFGS-B minimization of a nonlinear objective (suitable when T is moderate and a dense pairwise matrix is available). "lsqr_robust" builds an A matrix from observed pairs and performs iterative LSQR solves with robust trimming based on z-score outlier detection. "lsmr" constructs a sparse block-diagonal least-squares system (optionally with temporal and spatial priors and soft weights) and solves it with scipy.sparse.linalg.lsmr, optionally performing robust trimming iterations. If an unsupported string is passed, the function raises a ValueError.
    
        robust_regression_sigma (float): Sigma threshold for robust trimming of outliers when performing robust LSQR/LSMR iterations. During iterative robust fitting, residuals are standardized using a z-score and measurements with |z| > robust_regression_sigma are considered outliers and trimmed from the active set (except for untrimmable prior rows). Default is 2. This parameter affects both the "lsqr_robust" method and the iterative robust loop used inside "lsmr". Choosing a small value leads to more aggressive trimming; choosing a large value reduces trimming.
    
        lsqr_robust_n_iter (int): Number of robust iterations to run for the LSQR-based robust routines. For "lsqr_robust" this is the maximum number of reweighted LSQR iterations (default 20). For "lsmr" this controls the number of trim-and-solve iterations executed on the sparse LSMR system; at least one iteration is always executed. Setting this to 1 disables iterative trimming behavior. Increasing the number of iterations can improve robustness to outliers but increases runtime.
    
        progress_bar (bool): When True, and when the chosen solver supports iterative robust iterations, a progress bar (tqdm) is shown for the outer robust iteration loop. Default is False. Enabling a progress bar requires tqdm to be available at runtime and only affects user-visible progress reporting; it does not alter numerical behavior.
    
    Behavior, side effects, defaults, and failure modes:
        - The function dispatches between three solver strategies: "gradient_descent", "lsqr_robust", and "lsmr". Each strategy expects pairwise_displacement and optional weights/masks arranged as described above; passing inputs with incompatible shapes will trigger assertions or errors.
        - Default behavior (convergence_method="lsmr") treats pairwise_displacement_weight and sparse_mask as None unless provided, constructs a sparse block-diagonal coefficient matrix, optionally augments it with temporal and spatial finite-difference priors, and solves with scipy.sparse.linalg.lsmr. For single-window 2-D inputs the function expands dimensions internally to the single-block 3-D representation used by LSMR.
        - If pairwise_displacement_weight or sparse_mask are scipy.sparse.csr_matrix in some branches, they may be converted to dense arrays (particularly in the "lsmr" branch), which increases memory usage; callers should be aware of potential memory impact for large problems.
        - The "lsqr_robust" and "lsmr" branches implement robust trimming: iterative re-solving followed by excluding measurements with large standardized residuals; the threshold and number of iterations are controlled by robust_regression_sigma and lsqr_robust_n_iter. Prior rows (temporal or spatial priors) are marked untrimmable in the "lsmr" formulation to preserve regularization.
        - The "gradient_descent" branch uses scipy.optimize.minimize with method "L-BFGS-B" on a dense objective. If the minimizer reports res.success == False, the function prints a diagnostic message ("Global displacement gradient descent had an error") and still returns the optimizer result vector (res.x) as the displacement; callers should check solver convergence externally if they need to guarantee optimality.
        - If an unsupported convergence_method string is provided, the function raises a ValueError indicating the method is not implemented.
        - The function imports scipy and related submodules at runtime; missing scipy or optional scipy.sparse functionality will raise ImportError errors when the relevant code path is executed.
    
    Returns:
        numpy.ndarray: Estimated global displacement(s). For the simple rigid case (single block) this is a 1-D array of length T containing the displacement value for each timepoint. For multi-block (nonrigid) inputs provided as a 3-D pairwise_displacement of shape (B, T, T), the returned array will generally have shape (T, B) (time x block) unless B == 1 in which case the result is squeezed to shape (T,). The returned displacement values are in the same numerical units as the supplied pairwise_displacement inputs and represent the additive offsets p such that pairwise_displacement entries approximate p[i] - p[j].
    """
    from spikeinterface.sortingcomponents.motion.decentralized import compute_global_displacement
    return compute_global_displacement(
        pairwise_displacement,
        pairwise_displacement_weight,
        sparse_mask,
        temporal_prior,
        spatial_prior,
        soft_weights,
        convergence_method,
        robust_regression_sigma,
        lsqr_robust_n_iter,
        progress_bar
    )


################################################################################
# Source: spikeinterface.sortingcomponents.motion.dredge.dredge_online_lfp
# File: spikeinterface/sortingcomponents/motion/dredge.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_sortingcomponents_motion_dredge_dredge_online_lfp(
    lfp_recording: int,
    direction: str = "y",
    rigid: bool = True,
    win_shape: str = "gaussian",
    win_step_um: float = 800,
    win_scale_um: float = 850,
    win_margin_um: float = None,
    chunk_len_s: float = 10.0,
    max_disp_um: float = 500,
    time_horizon_s: float = None,
    mincorr: float = 0.8,
    mincorr_percentile: float = None,
    mincorr_percentile_nneighbs: int = 20,
    soft: bool = False,
    thomas_kw: dict = None,
    xcorr_kw: dict = None,
    extra_outputs: bool = False,
    device: str = None,
    progress_bar: bool = True
):
    """Online registration of a preprocessed LFP recording using windowed cross-correlation and an online tridiagonal solver to estimate time-varying spatial motion along a single axis.
    
    Args:
        lfp_recording (RecordingExtractor): A preprocessed LFP recording object (SpikeInterface RecordingExtractor-like) that provides channel geometry and time-series access. The function calls lfp_recording.get_channel_locations(), get_sampling_frequency(), get_num_samples(), get_traces(start_frame, end_frame) and get_times(0). This parameter is the primary data source whose local field potential (LFP) traces are analyzed to estimate motion over time across spatial windows.
        direction (str): Spatial axis along which motion is estimated. Must be one of the coordinate names present in channel locations (typically "x", "y", or "z"). The function selects the column corresponding to this axis from get_channel_locations() and uses those values as bin centers for the LFP spatial bins. Default is "y".
        rigid (bool): If True, use rigid (non-overlapping) spatial windows when constructing windows for local cross-correlation; if False, construct nonrigid/overlapping windows according to win_* parameters. Affects how get_spatial_windows(...) builds windows and thus the spatial resolution and overlap in the motion estimation. Default is True.
        win_shape (str): Kernel shape used to build spatial window weights when nonrigid windows are constructed (passed to get_spatial_windows). Typical values are "gaussian" (the default) or other shapes supported by the underlying window constructor. Controls how channel contributions are weighted inside each spatial window.
        win_step_um (float): Step between adjacent spatial window centers in micrometers. Determines how densely windows are sampled along the chosen direction (smaller values -> more windows). Default is 800. Interacts with win_scale_um to determine window extent.
        win_scale_um (float): Scale parameter in micrometers that controls spatial window width. For example, for Gaussian windows this is the Gaussian sigma. This determines the spatial smoothing applied when computing local cross-correlations. Default is 850.
        win_margin_um (float or None): Optional margin (in micrometers) to extend windows beyond computed centers. If None, the window constructor uses an internal default behavior (see get_spatial_windows). This affects which channels contribute to each spatial window and therefore the effective spatial support of local cross-correlations.
        chunk_len_s (float): Duration in seconds of each online processing chunk. The recording is processed sequentially in chunks of up to chunk_len_s seconds to limit memory usage and to produce an online estimate. The first chunk initializes the solver; subsequent chunks are correlated with the previous chunk to propagate continuity. Default is 10.0.
        max_disp_um (float): Maximum absolute spatial displacement in micrometers considered by the cross-correlation routine; used to limit search range and computational cost. The function may update this value based on data-driven estimates returned by the first cross-correlation call. Default is 500.
        time_horizon_s (float or None): Optional time horizon in seconds used by the threshold/weighting logic (passed into the thresholding kwargs). If provided, it constrains how much temporal separation is allowed when computing weights; if None no explicit time horizon is enforced. Default is None.
        mincorr (float): Absolute correlation threshold (0..1) used by threshold_correlation_matrix to filter cross-correlation peaks. A high value (e.g., 0.8) enforces that only strong local correlations contribute to the tridiagonal solve. Default is 0.8.
        mincorr_percentile (float or None): Alternative percentile-based threshold for correlations. If set, thresholding can adapt to data by selecting a percentile of observed correlations instead of using the fixed mincorr. If None, percentile-based adjustment is not applied. Default is None.
        mincorr_percentile_nneighbs (int): Number of spatial neighbors considered when computing percentile-based local thresholds for correlations. Used only when mincorr_percentile is provided or when the thresholding procedure requires a local neighborhood estimate. Default is 20.
        soft (bool): If True, use a "soft" (in-place, softer) thresholding behavior in threshold_correlation_matrix; if False use hard thresholding. Soft thresholding can allow more gradual weighting of weak correlations. Default is False.
        thomas_kw (dict or None): Keyword arguments forwarded to thomas_solve (the tridiagonal solver). Typical keys control solver regularization or matrix construction specifics. If None an empty dict is used. These kwargs affect how the per-window tridiagonal linear system is solved to produce the online motion estimate.
        xcorr_kw (dict or None): Keyword arguments forwarded to xcorr_windows, the low-level cross-correlation routine. Common keys include device selection, progress control, or algorithm-specific parameters. If None an empty dict is used. The function merges these with internally computed defaults before calling cross-correlation functions.
        extra_outputs (bool): If True, the function returns a second output dict with debugging and intermediate arrays useful for inspection and testing: window_centers, windows, lists of per-chunk D (displacement matrices), C (correlation matrices), S (thresholded correlation masks), D01/C01/S01 (cross-chunk matrices), mincorrs (per-chunk thresholds), and max_disp_um values. When False (default) only the Motion is returned.
        device (str or None): Device identifier forwarded to cross-correlation computation (e.g., used by xcorr_windows). If None the implementation chooses a default device. The string is passed through to lower-level routines; valid device strings depend on those routines and are not expanded here.
        progress_bar (bool): If True, show a progress bar (via trange) over online chunks; if False, run silently. This only affects user feedback and not the computed motion. Default is True.
    
    Behavior and side effects:
        This function performs an online (chunked) estimation of spatial motion along a single axis from a preprocessed LFP recording. It builds spatial windows centered on channel positions along the specified direction, computes local cross-correlations per window (and cross-chunk correlations between consecutive chunks), thresholds the correlation matrices to form sparse coupling terms, and solves a tridiagonal linear system (Thomas algorithm) in an online fashion to produce per-window displacement estimates over time. Processing is done sequentially in temporal chunks of length chunk_len_s seconds to limit memory consumption. The function allocates a floating array P_online of shape (number_of_windows, total_time_samples) to accumulate the estimated per-window motion scores and then constructs and returns a Motion object from that array. If extra_outputs is True, intermediate matrices (D, C, S and cross-chunk versions) and threshold values are returned for debugging.
    
    Defaults:
        Default windowing and chunking parameters (win_shape="gaussian", win_step_um=800, win_scale_um=850, chunk_len_s=10.0, max_disp_um=500) favor coarse spatial sampling and moderate temporal chunk sizes for LFP data; adjust those depending on probe geometry and desired spatial/temporal resolution. thomas_kw and xcorr_kw default to empty dicts if not provided.
    
    Failure modes and errors:
        The function raises a ValueError if the channel positions along the chosen direction are not unique or are not strictly ordered (the algorithm expects unique, ordered channel depths along the selected axis). Specifically, if get_channel_locations() yields duplicate values or decreases along the chosen axis, a ValueError is raised with guidance to reorder channels (for example using spikeinterface.preprocessing.depth_order). The function also relies on lfp_recording exposing the required methods; missing methods or incompatible types will raise attribute errors. Cross-correlation or solver routines may raise their own exceptions if numerical issues occur or if provided keyword arguments are invalid.
    
    Returns:
        Motion: A Motion object that encapsulates the estimated motion over time. The Motion is constructed from the transposed P_online array (per-timepoint estimates derived from per-window scores), the timebase lfp_recording.get_times(0) for the first sample, the spatial window_centers computed from channel positions, and the specified direction. The Motion object is the primary output used downstream in SpikeInterface motion-aware processing and visualization.
        tuple(Motion, dict): If extra_outputs is True, returns a tuple where the first element is the Motion as described above and the second element is a dictionary containing intermediate arrays and metadata useful for debugging and inspection (keys include "window_centers", "windows", "D", "C", "S", "D01", "C01", "S01", "mincorrs", and "max_disp_um").
    """
    from spikeinterface.sortingcomponents.motion.dredge import dredge_online_lfp
    return dredge_online_lfp(
        lfp_recording,
        direction,
        rigid,
        win_shape,
        win_step_um,
        win_scale_um,
        win_margin_um,
        chunk_len_s,
        max_disp_um,
        time_horizon_s,
        mincorr,
        mincorr_percentile,
        mincorr_percentile_nneighbs,
        soft,
        thomas_kw,
        xcorr_kw,
        extra_outputs,
        device,
        progress_bar
    )


################################################################################
# Source: spikeinterface.sortingcomponents.motion.dredge.laplacian
# File: spikeinterface/sortingcomponents/motion/dredge.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_sortingcomponents_motion_dredge_laplacian(
    n: int,
    wink: bool = True,
    eps: float = 0.001,
    lambd: float = 1.0,
    ridge_mask: numpy.ndarray = None
):
    """spikeinterface.sortingcomponents.motion.dredge.laplacian constructs a dense discrete 1D Laplacian matrix (with an added small ridge/identity term) used as a regularization / smoothing operator in SpikeInterface sorting components (for example, within motion or drift regularization steps of custom sorters built with the sortingcomponents). The returned matrix is a square numpy.ndarray of shape (n, n) whose main diagonal and immediate off-diagonals encode a second-difference operator scaled by lambd, with an additional eps-scaled identity term to improve numerical stability; an optional ridge_mask allows spatially varying small diagonal augmentation.
    
    Args:
        n (int): The size of the square matrix to construct; the number of rows and columns of the returned Laplacian operator. This integer specifies the number of discrete points over which the second-difference (discrete Laplacian) is defined. The function allocates and returns a dense (n, n) numpy.ndarray; very large n will increase memory and time quadratically and may raise MemoryError from numpy.
        wink (bool): If True (default True), adjust the first and last diagonal entries by subtracting 0.5 * lambd so that endpoint diagonal values are halved relative to interior diagonals. Practically, this modifies the boundary behavior of the discrete Laplacian (commonly used to approximate natural/Neumann-like boundary conditions in 1D regularization). If False, endpoints are treated the same as interior points.
        eps (float): Small nonnegative scalar added (times ridge_mask if provided) to the diagonal for numerical stability (default 0.001). This acts as a ridge (Tikhonov) term that prevents the matrix from being singular or ill-conditioned when used as a penalty/precision matrix in optimization steps related to motion/drift estimation.
        lambd (float): Scaling for the discrete Laplacian operator (default 1.0). lambd sets the strength of the second-difference penalty: interior diagonal entries are initialized as lambd (plus the eps term), and each immediate off-diagonal entry between adjacent indices is -0.5 * lambd, producing the standard tridiagonal Laplacian stencil for a 1D chain of n points.
        ridge_mask (numpy.ndarray): Optional 1D numpy array used to modulate the eps term elementwise across the diagonal. When provided, the code computes per-index diagonal values as lambd + eps * ridge_mask[i] and fills the main diagonal with this array. ridge_mask must therefore be broadcastable to length n (typically a 1D array of shape (n,)); if its length does not match n, numpy.fill_diagonal will raise a ValueError. If None (default), the eps term is a scalar added uniformly so the diagonal is lambd + eps.
    
    Returns:
        numpy.ndarray: A dense square matrix of shape (n, n) representing the discrete Laplacian plus eps*identity (or eps*ridge_mask on the diagonal when ridge_mask is provided). The matrix is populated as follows: the main diagonal entries are lambd + eps (or lambd + eps * ridge_mask[i]); each pair of adjacent off-diagonal entries (i, i+1) and (i+1, i) is set to -0.5 * lambd. If wink is True, the [0,0] and [-1,-1] diagonal entries are reduced by 0.5 * lambd (equivalently, endpoints become 0.5 * lambd + eps or 0.5 * lambd + eps * ridge_mask[end]). No in-place side effects occur beyond allocating and returning this array; the function does not modify inputs. Failure modes include numpy memory errors for very large n and ValueError from numpy.fill_diagonal if ridge_mask has an incompatible length or shape.
    """
    from spikeinterface.sortingcomponents.motion.dredge import laplacian
    return laplacian(n, wink, eps, lambd, ridge_mask)


################################################################################
# Source: spikeinterface.sortingcomponents.motion.dredge.neg_hessian_likelihood_term
# File: spikeinterface/sortingcomponents/motion/dredge.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_sortingcomponents_motion_dredge_neg_hessian_likelihood_term(
    Ub: numpy.ndarray,
    Ub_prevcur: numpy.ndarray = None,
    Ub_curprev: numpy.ndarray = None
):
    """spikeinterface.sortingcomponents.motion.dredge.neg_hessian_likelihood_term computes the negative-Hessian contribution of the non-regularized likelihood term for a nonrigid block and prepares coefficients used in a Newton-step linear system inside SpikeInterface's nonrigid motion estimation components. In the spike sorting domain (see README: SpikeInterface provides sorting components to build custom sorters), this function forms the likelihood part of the coefficient matrix that, together with separate regularization terms, is used to solve for block-wise motion or deformation corrections via a linear solver or Newton update.
    
    This function computes the matrix -Ub - Ub.T and then replaces its diagonal with per-row and per-column sums adjusted by optional neighboring-block contributions. The implementation uses a single temporary array (negHUb) to avoid creating an extra full copy of -Ub + -Ub.T; the input Ub itself is not modified.
    
    Args:
        Ub (numpy.ndarray): A 2-D array representing the within-block pairwise coupling (the block-specific matrix used to build the Hessian of the non-regularized likelihood). This array is expected to be a numeric NumPy array compatible with .T, .sum(0) and .sum(1). Typical use in the SpikeInterface nonrigid motion code is a square matrix of pairwise interactions within a block; the returned array has the same shape as Ub.
        Ub_prevcur (numpy.ndarray): Optional 2-D numeric NumPy array containing coupling terms from the previous block to the current block. If provided, Ub_curprev must also be provided. When not None, its column-wise sums (Ub_prevcur.sum(0)) are added to the diagonal correction so that cross-block interactions from the previous->current direction are included in the diagonal of the negative Hessian term. If omitted (None), the diagonal is computed without these neighboring-block contributions.
        Ub_curprev (numpy.ndarray): Optional 2-D numeric NumPy array containing coupling terms from the current block to the previous block. If provided, Ub_prevcur must also be provided. When not None, its row-wise sums (Ub_curprev.sum(1)) are added to the diagonal correction so that cross-block interactions from the current->previous direction are included in the diagonal of the negative Hessian term.
    
    Behavior and side effects:
        The function computes negHUb = -Ub - Ub.T using a single temporary array to reduce memory allocations. It then computes diagonal_terms as numpy.diagonal(negHUb) + Ub.sum(1) + Ub.sum(0). If both Ub_prevcur and Ub_curprev are provided (neither is None), diagonal_terms is further incremented by Ub_prevcur.sum(0) + Ub_curprev.sum(1). The computed diagonal_terms are written into the diagonal of negHUb via numpy.fill_diagonal. The original input array Ub is not modified by this function. There are no external side effects beyond returning the new array.
    
    Defaults and failure modes:
        If Ub_prevcur is None, the function assumes no neighboring-block corrections and uses the simpler diagonal formula. If Ub_prevcur is provided but Ub_curprev is None, the code will attempt to access Ub_curprev.sum(1) and will raise an AttributeError or TypeError; therefore Ub_prevcur and Ub_curprev must either both be None or both be valid numpy.ndarray objects. If input arrays have incompatible shapes for the implicit elementwise and sum operations, NumPy will raise exceptions (for example ValueError due to shape mismatch). Inputs must be numeric arrays; non-numeric dtypes may lead to TypeError during arithmetic.
    
    Returns:
        numpy.ndarray: A 2-D NumPy array with the same shape as Ub containing the negative-Hessian-like matrix for the non-regularized likelihood term. The returned matrix equals -Ub - Ub.T with its diagonal replaced by the computed diagonal_terms described above. This matrix is intended to be combined with regularization-derived terms to form the final coefficient matrix used in the Newton-step linear problem for nonrigid block motion estimation in SpikeInterface.
    """
    from spikeinterface.sortingcomponents.motion.dredge import neg_hessian_likelihood_term
    return neg_hessian_likelihood_term(Ub, Ub_prevcur, Ub_curprev)


################################################################################
# Source: spikeinterface.sortingcomponents.motion.dredge.newton_rhs
# File: spikeinterface/sortingcomponents/motion/dredge.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_sortingcomponents_motion_dredge_newton_rhs(
    Db: numpy.ndarray,
    Ub: numpy.ndarray,
    Pb_prev: numpy.ndarray = None,
    Db_prevcur: numpy.ndarray = None,
    Ub_prevcur: numpy.ndarray = None,
    Db_curprev: numpy.ndarray = None,
    Ub_curprev: numpy.ndarray = None
):
    """spikeinterface.sortingcomponents.motion.dredge.newton_rhs computes the right-hand side vector for a Newton update in the "dredge" alignment routine used by SpikeInterface sorting components. This function returns the gradient of the cost function evaluated at P=0 for the batch case, or the gradient augmented with alignment terms for the online (sequential) case where previous/current batch coupling is provided. It is typically used inside a Newton solver that updates a permutation-like matrix P to align clusters or templates across recording batches during spike sorting and motion correction.
    
    Args:
        Db (numpy.ndarray): A 2-D numeric array representing multiplicative factors for the pairwise term of the cost function for the current batch. In practice within the dredge alignment context, Db contains values that weight the contribution between pairs of units (for example, pairwise costs or affinities). Must be a numpy.ndarray with shape compatible with Ub for elementwise multiplication (commonly a square matrix of size n_units x n_units). This argument is required for both batch and online modes.
        Ub (numpy.ndarray): A 2-D numeric array of the same shape as Db representing the pairwise interaction coefficients (for example, pairwise potentials or statistical weights) used to compute the gradient at P=0. Ub is elementwise-multiplied with Db inside the function; therefore Ub must be provided as a numpy.ndarray and have a shape broadcast-compatible with Db (typically identical shape).
        Pb_prev (numpy.ndarray): Optional. When None (default), the function runs in batch mode and returns only the gradient at P=0. When provided (online mode), Pb_prev is a numpy.ndarray representing the previous estimate of the permutation or alignment matrix that maps units from the previous batch to the current batch; it is used to compute the alignment term that modifies the gradient. If Pb_prev is not None, the function expects the *_prevcur and _curprev arrays to also be provided and shape-compatible; otherwise a runtime exception will occur.
        Db_prevcur (numpy.ndarray): Optional. Used only in online mode (Pb_prev is not None). A numpy.ndarray representing the Db-like multiplicative factors coupling the previous batch (rows) to the current batch (columns). It participates in a correction term subtracted from the right-hand side; it must have shapes compatible with Ub_prevcur for elementwise multiplication and with Ub_prevcur.sum operations used in the online formula.
        Ub_prevcur (numpy.ndarray): Optional. Used only in online mode. A numpy.ndarray representing the pairwise interaction coefficients from previous batch units to current batch units. In the online formula, Ub_prevcur contributes both via its transpose in an alignment matrix multiplication and via elementwise products with Db_prevcur. Ub_prevcur must be provided and shape-compatible with Pb_prev and Db_prevcur when Pb_prev is not None.
        Db_curprev (numpy.ndarray): Optional. Used only in online mode. A numpy.ndarray representing the Db-like multiplicative factors coupling the current batch (rows) to the previous batch (columns). It is used together with Ub_curprev to form a correction term added to the right-hand side. Must be provided and shape-compatible with Ub_curprev when Pb_prev is not None.
        Ub_curprev (numpy.ndarray): Optional. Used only in online mode. A numpy.ndarray representing the pairwise interaction coefficients from current batch units to previous batch units. Ub_curprev is used in computing an alignment matrix ((Ub_prevcur.T + Ub_curprev) @ Pb_prev) and in elementwise products with Db_curprev; it must be provided and shape-compatible with Pb_prev, Db_curprev, and Ub_prevcur when Pb_prev is not None.
    
    Returns:
        numpy.ndarray: A 1-D numpy array containing the right-hand side vector for the Newton step. In batch mode (Pb_prev is None) this is the gradient of the cost function at P=0, computed as the row-wise sum of Ub * Db minus the column-wise sum of Ub * Db. In online mode (Pb_prev is provided), the returned vector is the batch gradient augmented by the alignment term (Ub_prevcur.T + Ub_curprev) @ Pb_prev plus the appropriate correction terms from elementwise products (Ub_curprev * Db_curprev).sum(1) and (Ub_prevcur * Db_prevcur).sum(0). The returned array is intended to be used directly as the right-hand side in a Newton solver that updates permutation/alignment variables.
    
    Raises:
        TypeError: If required numpy.ndarray inputs for the selected mode are None (for example, Pb_prev provided but one of Ub_prevcur, Ub_curprev, Db_prevcur, Db_curprev is None) or if non-array types are passed that do not support the used numpy operations.
        ValueError: If input arrays have incompatible shapes for elementwise multiplication, summation, transposition, or matrix multiplication required by the batch or online formulas.
    """
    from spikeinterface.sortingcomponents.motion.dredge import newton_rhs
    return newton_rhs(Db, Ub, Pb_prev, Db_prevcur, Ub_prevcur, Db_curprev, Ub_curprev)


################################################################################
# Source: spikeinterface.sortingcomponents.motion.dredge.newton_solve_rigid
# File: spikeinterface/sortingcomponents/motion/dredge.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_sortingcomponents_motion_dredge_newton_solve_rigid(
    D: numpy.ndarray,
    U: numpy.ndarray,
    Sigma0inv: numpy.ndarray,
    Pb_prev: numpy.ndarray = None,
    Db_prevcur: numpy.ndarray = None,
    Ub_prevcur: numpy.ndarray = None,
    Db_curprev: numpy.ndarray = None,
    Ub_curprev: numpy.ndarray = None
):
    """Solve the rigid Newton step used in the motion-correction dredge component of SpikeInterface.
    
    This function builds and solves the linear Newton system for a rigid motion update p used in SpikeInterface's sortingcomponents.motion.dredge workflow. It combines a prior inverse covariance term Sigma0inv with a data-derived negative Hessian of the likelihood (negHU) computed from subsampling/soft-weights, and a right-hand side assembled from the displacement matrix D and optional boundary/adjacent-block contributions. The result p is the Newton update that would be applied to rigid motion parameters in downstream motion-correction steps of the SpikeInterface pipeline.
    
    Args:
        D (numpy.ndarray): T x T displacement matrix. In the SpikeInterface motion-dredge context, D encodes observed displacements (differences) between time points or blocks; it is used to form the right-hand side of the Newton linear system. The function expects a square matrix with the same time dimension T used by U and Sigma0inv.
        U (numpy.ndarray): T x T subsampling or soft-weights matrix. In practice within SpikeInterface this matrix weights contributions from different time points or samples (e.g., subsampling patterns or soft assignment weights) when computing both the negative Hessian of the likelihood and the Newton right-hand side. U must have the same square shape T x T as D.
        Sigma0inv (numpy.ndarray): Prior inverse covariance matrix (numpy.ndarray). This matrix encodes the Gaussian prior precision on the rigid parameters and is added to the data-derived negative Hessian to form the coefficient matrix in the Newton linear system. It must be conformable with negHU (typically T x T).
        Pb_prev (numpy.ndarray): Optional boundary/previous-block prior term. When provided, this array supplies prior or previous-block parameter information that newton_rhs uses to modify the right-hand side targ for continuity or regularization across adjacent blocks in blockwise motion correction. Pass None when no previous-block prior information is available.
        Db_prevcur (numpy.ndarray): Optional displacement term between previous and current blocks. When supplied, this array provides observed displacements specifically across the boundary from a previous block to the current block; newton_rhs uses it to augment the Newton right-hand side to account for inter-block motion. Pass None if no previous→current boundary displacement is used.
        Ub_prevcur (numpy.ndarray): Optional subsampling/weight matrix for previous→current boundary. This optional T x T (or block-conformable) weights matrix is passed to neg_hessian_likelihood_term and newton_rhs so boundary-weighted likelihood/Hessian contributions are included. Pass None when there are no boundary weighting contributions from previous to current block.
        Db_curprev (numpy.ndarray): Optional displacement term between current and previous blocks (current→previous). When provided, this array supplies observed displacements across the boundary in the reverse direction; newton_rhs uses it to form the right-hand side to enforce consistency across blocks. Use None if not applicable.
        Ub_curprev (numpy.ndarray): Optional subsampling/weight matrix for current→previous boundary. This optional weights matrix is used by neg_hessian_likelihood_term and newton_rhs to incorporate boundary-weighted likelihood/Hessian contributions from the current-to-previous direction. Pass None if no such contribution is present.
    
    Behavior and side effects:
        The function computes negHU by calling an internal routine neg_hessian_likelihood_term with Ub_prevcur and Ub_curprev, producing the data-derived negative Hessian contribution to the Newton system. It computes the right-hand side targ by calling newton_rhs with D, U, and any provided boundary arrays (Pb_prev, Db_prevcur, Ub_prevcur, Db_curprev, Ub_curprev). It then attempts to solve the linear system (Sigma0inv + negHU) p = targ using scipy.linalg.solve with assume_a="pos", which assumes the coefficient matrix is symmetric positive definite (common when Sigma0inv encodes a positive-definite prior and negHU is a well-behaved negative Hessian). If a linear-algebra error occurs because the matrix is singular or otherwise unsuitable for the direct solver, the function emits a Python warning ("Singular problem, using least squares.") and falls back to scipy.linalg.lstsq to compute a least-squares solution. No in-place modification of the input arrays is performed by this function.
    
    Failure modes and defaults:
        If the combined matrix Sigma0inv + negHU is singular or ill-conditioned, the direct solver may raise numpy.linalg.LinAlgError; this is caught and a least-squares solution is used instead. If both the direct solve and least-squares fail (rare and typically due to invalid inputs or shapes), the underlying SciPy routines will propagate their exceptions. Optional boundary arguments default to None to indicate absence of boundary contributions; newton_rhs and neg_hessian_likelihood_term are expected to handle None appropriately.
    
    Returns:
        tuple:
            p (numpy.ndarray): The computed Newton update vector/matrix for rigid motion parameters. This is the solution of (Sigma0inv + negHU) p = targ computed either by a direct positive-definite solver or, on singular/degenerate problems, a least-squares solver. The shape matches the dimensionality of targ and is compatible with the time dimension T used in D and U.
            negHU (numpy.ndarray): The negative Hessian matrix contributed by the likelihood term as computed from U and optional boundary weight matrices (Ub_prevcur, Ub_curprev). This matrix has the same shape as Sigma0inv and represents the data-derived curvature used to form the Newton coefficient matrix.
    """
    from spikeinterface.sortingcomponents.motion.dredge import newton_solve_rigid
    return newton_solve_rigid(
        D,
        U,
        Sigma0inv,
        Pb_prev,
        Db_prevcur,
        Ub_prevcur,
        Db_curprev,
        Ub_curprev
    )


################################################################################
# Source: spikeinterface.sortingcomponents.motion.dredge.thomas_solve
# File: spikeinterface/sortingcomponents/motion/dredge.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_sortingcomponents_motion_dredge_thomas_solve(
    Ds: numpy.ndarray,
    Us: numpy.ndarray,
    lambda_t: float = 1.0,
    lambda_s: float = 1.0,
    eps: float = 0.001,
    P_prev: numpy.ndarray = None,
    Ds_prevcur: numpy.ndarray = None,
    Us_prevcur: numpy.ndarray = None,
    Ds_curprev: numpy.ndarray = None,
    Us_curprev: numpy.ndarray = None,
    progress_bar: bool = False,
    bandwidth: int = None
):
    """spikeinterface.sortingcomponents.motion.dredge.thomas_solve solves a block-tridiagonal linear system for nonrigid motion displacement estimation used in SpikeInterface's nonrigid registration component. It computes displacement estimates for B spatial blocks (nonrigid windows) across T temporal bins by combining blockwise pairwise displacement measurements and their weights with temporal and spatial priors. This function is used during motion correction/registration in spike sorting pipelines to produce per-window, per-time-bin displacements that align recording segments.
    
    Args:
        Ds (numpy.ndarray): Blockwise pairwise displacement measurements. Expected shape is (B, T, T) where B is the number of nonrigid spatial windows (blocks) and T is the number of temporal bins. For each block b, Ds[b][i,j] is the displacement estimate relating time bin i to time bin j within that block. These measurements provide the data term (residual displacements) used by the solver to compute a consistent per-block displacement P. Ds is converted to float64 internally; mismatched shapes will trigger an AssertionError.
        Us (numpy.ndarray): Blockwise pairwise weight (uncertainty) matrices with the same shape as Ds, i.e., (B, T, T). For each block b, Us[b][i,j] is the weight associated with Ds[b][i,j] (higher values indicate more confident measurements). Us is used to form the negative Hessian and the right-hand side for each block. Us is converted to float64 internally; shape mismatch with Ds will raise an AssertionError.
        lambda_t (float): Temporal regularization strength (prior) applied along each block's temporal dimension to "fill gaps" and stabilize estimates across time. Typical default from the module is 1.0. A positive lambda_t causes construction of a temporal Laplacian (T x T) per block and avoids interpolation artifacts in low-signal regions. Setting lambda_t to 0 removes the explicit temporal prior and can lead to numerical warnings or instability; use with care.
        lambda_s (float): Spatial regularization strength coupling neighboring spatial blocks (nonrigid windows). Default 1.0. When lambda_s > 0, a block-tridiagonal spatial prior (Kronecker structure across blocks and time) is added, enabling information sharing between adjacent spatial windows to fill untrusted regions. If lambda_s == 0 or B == 1, the function solves each block independently (no spatial coupling).
        eps (float): Small ridge/epsilon value for numerical stability when building Laplacian matrices (used as a ridge term to avoid singular matrices). Default 0.001. eps is added only where appropriate (based on which temporal bins had any weights) so the solution is not changed except to improve numerical conditioning.
        P_prev (numpy.ndarray): Previous displacement estimates used for online processing. When provided (not None), the function runs in "online" mode and integrates the previous chunk's solution into the current block-wise solve. P_prev should be a numpy.ndarray indexed by block (matching the first dimension B) and will be used to construct online RHS/Hessian contributions. If P_prev is given, the corresponding Ds_prevcur/Us_prevcur (and optionally Ds_curprev/Us_curprev) arguments must also be provided; otherwise an AssertionError is raised.
        Ds_prevcur (numpy.ndarray): Pairwise displacement terms between the previous chunk and the current chunk, used only in online mode. When P_prev is supplied, Ds_prevcur[b] is expected for each block b and is converted to float64 internally; it provides cross-chunk data terms that modify the right-hand side of the linear system so the returned solution respects continuity with prior chunks.
        Us_prevcur (numpy.ndarray): Pairwise weight terms between the previous chunk and the current chunk, used only in online mode. When P_prev is supplied, Us_prevcur[b] is required for each block b and is converted to float64 internally; it contributes to the negative Hessian and RHS in the online formulation.
        Ds_curprev (numpy.ndarray): Pairwise displacement terms from the current chunk to the previous chunk, used only in online mode when cross-chunk asymmetry is modeled. If provided alongside P_prev and Ds_prevcur/Us_prevcur, these terms are incorporated into the RHS construction to correctly account for directed pairwise measurements.
        Us_curprev (numpy.ndarray): Pairwise weight terms from the current chunk to the previous chunk, used only in online mode. If provided, Us_curprev contributes to the Hessian/RHS in the online update similarly to Us_prevcur.
        progress_bar (bool): If True, show a progress bar (uses trange) during the forward pass over spatial blocks. Default False. This only affects user feedback; it does not change the algorithmic result.
        bandwidth (int): Unused in the current implementation; accepted for API compatibility. Default None. Passing a non-None value has no effect on computation in this version and is ignored.
    
    Behavior, side effects, defaults, and failure modes:
        The function first converts Ds and Us to numpy.float64 and asserts that they have identical shapes and that the second and third dimensions are equal (T == T_). It constructs a temporal Laplacian matrix L_t per block using lambda_t and eps, masking time bins that had no weights to avoid changing the intended solution while preserving numerical stability. If B == 1 or lambda_s == 0 the solver treats each spatial block independently and calls an internal Newton solver that returns per-block displacements and a diagnostic "HU" matrix; the returned extra dictionary will include "HU" in this branch. For B > 1 and lambda_s > 0 a spatial prior is assembled as a block-tridiagonal system (with half-weighting on boundary blocks) and the function performs a block LU-like forward/backward pass: in the forward pass it solves a sequence of T x T linear systems (using scipy.linalg.solve), stores intermediate gamma and y factors, and in the backward pass it performs back-substitution to obtain per-block solutions. When online mode is enabled (P_prev is not None), additional RHS and Hessian contributions are built from the provided Ds_prevcur/Us_prevcur and Ds_curprev/Us_curprev to enforce continuity with previous chunks; P_prev and the corresponding cross-chunk arrays must be present or an AssertionError is raised. The function may raise AssertionError for shape mismatches or missing online arrays, and numerical warnings or linear solver failures may occur if lambda_t is set to 0 or eps is too small. The function uses scipy.linalg.solve with assume_a="pos" in some branches; if the matrix is not positive definite the solver may raise a LinAlgError. The progress_bar option only toggles a tqdm-like progress display and does not affect numerical results. The bandwidth parameter is currently ignored and provided for API compatibility.
    
    Returns:
        tuple: A tuple (P, extra) where:
            P (numpy.ndarray): The per-block, per-time-bin displacement estimates computed by the solver. Shape is (B, T). P[b, t] is the estimated displacement for spatial block b at temporal bin t. In online mode this corresponds to the updated displacement estimate that is consistent with the provided previous-chunk information.
            extra (dict): A dictionary of diagnostic and intermediate quantities useful for inspection or further processing. At minimum this contains "L_t", a list of T x T temporal Laplacian matrices (one per block) constructed with the provided lambda_t and eps; when the solver takes the independent-blocks branch (B == 1 or lambda_s == 0) it additionally contains "HU", a (B, T, T) array with per-block negative-Hessian times U diagnostic matrices produced by the internal Newton step. Additional keys may appear depending on internal branches (these are intended for debugging and downstream diagnostics, not as an algorithmic contract).
    """
    from spikeinterface.sortingcomponents.motion.dredge import thomas_solve
    return thomas_solve(
        Ds,
        Us,
        lambda_t,
        lambda_s,
        eps,
        P_prev,
        Ds_prevcur,
        Us_prevcur,
        Ds_curprev,
        Us_curprev,
        progress_bar,
        bandwidth
    )


################################################################################
# Source: spikeinterface.sortingcomponents.motion.iterative_template.iterative_template_registration
# File: spikeinterface/sortingcomponents/motion/iterative_template.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_sortingcomponents_motion_iterative_template_iterative_template_registration(
    spikecounts_hist_images: numpy.ndarray,
    non_rigid_windows: list = None,
    num_shifts_global: int = 15,
    num_iterations: int = 10,
    num_shifts_block: int = 5,
    smoothing_sigma: float = 0.5,
    kriging_sigma: float = 1,
    kriging_p: float = 2,
    kriging_d: float = 2
):
    """spikeinterface.sortingcomponents.motion.iterative_template.iterative_template_registration computes an optimal per-time-bin spatial registration (rigid then non-rigid) of spike count histogram images used in spike sorting motion correction. The function performs an initial iterative rigid integer shift alignment across temporal bins, builds a target spatial histogram, and then computes non-rigid per-block shifts with sub-integer resolution by upsampling blockwise cross-covariance using a kriging kernel. This routine is used in SpikeInterface to estimate and correct probe motion by aligning spike-count histograms across time and spatial bins, producing per-time-block spatial shifts and a target histogram used for downstream template alignment or motion-correction steps.
    
    Args:
        spikecounts_hist_images (numpy.ndarray):
            Spike count histogram images provided by the caller. Expected input to represent spike count histograms organized along three axes corresponding to temporal bins, spatial (y) bins, and amplitude bins in some order. In the original code this array is interpreted and internally reordered to have shape (num_spatial_bins, num_amps_bins, num_temporal_bins) by swapping axes. Each element is treated as a count or density for a given spatial bin and amplitude bin at a given temporal bin. This argument is required. The function does not modify the caller's array in place but will reassign the local variable to an axis-permuted view.
        non_rigid_windows (list):
            Default: None. When performing non-rigid registration the function requires a list of taper windows that define spatial blocks to process independently. Each element of this list is treated as a per-spatial-bin window that is applied (tapered) to the corresponding spatial region before computing blockwise cross-covariances. If len(non_rigid_windows) == 1 the function effectively performs rigid registration only. If non_rigid_windows is None, the function will raise an exception when it attempts to use its length; therefore callers intending rigid-only registration should still provide a single-window list or ensure upstream code prepares an appropriate list. The list length determines num_non_rigid_windows.
        num_shifts_global (int):
            Default: 15. Number of integer spatial-bin shifts (in each direction) to consider during the initial global rigid alignment phase. The function searches integer shifts in the range [-num_shifts_global, +num_shifts_global] to maximize cross-covariance against the current target frame. Larger values increase the search range but also increase computation and memory for computing shift covariance curves.
        num_iterations (int):
            Default: 10. Number of iterative passes for the global (rigid) alignment loop. On each iteration the function recomputes per-shift cross-covariances, selects the best integer shift per temporal bin, rolls data accordingly, and updates the target frame as the mean across aligned temporal bins. The routine accumulates per-iteration integer shifts in an internal best_shifts matrix; the final non-rigid shifts are formed by summing these integer shifts with blockwise sub-integer corrections. If set too small, the rigid alignment may not converge; excessive values waste computation.
        num_shifts_block (int):
            Default: 5. Number of integer spatial-bin shifts (in each direction) to consider for local per-block (non-rigid) alignment. The local search range is [-num_shifts_block, +num_shifts_block]. This parameter controls the resolution and local search radius used before upsampling with the kriging kernel.
        smoothing_sigma (float):
            Default: 0.5. Standard deviation (sigma) of the Gaussian smoothing applied to the blockwise shift covariance volumes for robustness. The function applies a 2-D Gaussian filter across time and block dimensions and a 1-D Gaussian filter along the shift dimension. Higher values yield smoother covariance curves (more robust but potentially less sensitive to sharp local changes); zero disables smoothing.
        kriging_sigma (float):
            Default: 1. Sigma parameter passed to the internal kriging_kernel used to compute the upsampling kernel that converts integer-shift covariance curves into higher-resolution shift estimates. This parameter affects the shape/width of the kriging interpolation kernel and therefore the sub-integer tuning of shift estimates.
        kriging_p (float):
            Default: 2. p parameter passed to the kriging_kernel controlling the power term used by the kriging interpolation; it participates in the kernel computation that produces the upsampling matrix transforming integer-shift covariance to sub-integer resolution.
        kriging_d (float):
            Default: 2. d parameter passed to the kriging_kernel controlling an additional kernel shape parameter used during kriging-based upsampling of the blockwise covariance curves.
    
    Behavior and side effects:
        The function first permutes the input spikecounts_hist_images to shape (num_spatial_bins, num_amps_bins, num_temporal_bins) via axis swaps, then computes mean-subtracted data and performs an iterative rigid integer shift registration. On each rigid iteration it computes cross-covariances for all candidate integer shifts in the global search range, selects the best shift per temporal bin, applies those integer rolls to the data, and updates the target frame (the mean aligned histogram). After completing the rigid iterations it computes blockwise (non-rigid) cross-covariances by tapering the data with each provided non_rigid_windows entry, evaluating covariance across candidate integer shifts within num_shifts_block, and then smoothing these covariance volumes using Gaussian filters with smoothing_sigma. The smoothed integer-shift covariance curves for each block are upsampled with a kriging kernel constructed using kriging_sigma, kriging_p, and kriging_d to estimate sub-integer shifts. The final per-temporal-bin per-block shifts are the sum of integer shifts accumulated during rigid iterations and the blockwise sub-integer corrections. The function does not write to external files or global state; it returns computed arrays.
    
    Failure modes and important notes:
        If non_rigid_windows is None the function will raise a TypeError when attempting to take its length; callers must supply a list when expecting any non-rigid processing. The list length must match the intended number of spatial blocks and each element must contain taper values that index into the spatial dimension of spikecounts_hist_images; incorrect sizing will lead to indexing errors. The kriging_kernel call is expected to exist in the module namespace; if that function is missing or misconfigured, a NameError or related exception will be raised. The function assumes numerical inputs (counts/densities) and uses numpy roll and average operations; non-numeric or badly shaped arrays will cause numpy exceptions. Performance and memory use grow with num_temporal_bins, num_spatial_bins, num_shifts_global, and num_shifts_block.
    
    Returns:
        optimal_shift_indices (numpy.ndarray):
            Array of estimated shifts per temporal bin and per non-rigid window with shape (num_temporal_bins, num_non_rigid_windows). Each value represents the total shift (in spatial-bin units) to apply to the corresponding temporal bin and block. The values combine integer shifts found during the global iterative rigid stage and sub-integer corrections obtained from blockwise upsampled covariance curves.
        target_spikecount_hist (numpy.ndarray):
            The target spatial histogram used for alignment, returned as a 2-D array with shape (num_spatial_bins, num_amps_bins). This is the mean-aligned frame produced by the final iteration of the rigid alignment loop and is suitable for use as a reference template for further alignment or motion-correction downstream in the spike sorting pipeline.
        shift_covs_block (numpy.ndarray):
            The raw blockwise cross-covariance values computed prior to smoothing/upsampling, with shape (2 * num_shifts_block + 1, num_temporal_bins, num_non_rigid_windows). The first axis indexes integer shift candidates in the range [-num_shifts_block, +num_shifts_block]. This output can be inspected for diagnostic purposes to evaluate the raw per-shift covariance evidence used to derive the final upsampled shift estimates.
    """
    from spikeinterface.sortingcomponents.motion.iterative_template import iterative_template_registration
    return iterative_template_registration(
        spikecounts_hist_images,
        non_rigid_windows,
        num_shifts_global,
        num_iterations,
        num_shifts_block,
        smoothing_sigma,
        kriging_sigma,
        kriging_p,
        kriging_d
    )


################################################################################
# Source: spikeinterface.sortingcomponents.motion.motion_cleaner.clean_motion_vector
# File: spikeinterface/sortingcomponents/motion/motion_cleaner.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_sortingcomponents_motion_motion_cleaner_clean_motion_vector(
    motion: numpy.ndarray,
    temporal_bins: numpy.ndarray,
    bin_duration_s: float,
    speed_threshold: float = 30,
    sigma_smooth_s: float = None
):
    """Clean a 2-D motion estimate time series by removing spurious rapid changes ("bumps") and optionally applying a Gaussian temporal smoothing. This function is used in SpikeInterface sorting components to stabilize probe motion estimates (units: micrometers) over time prior to downstream steps such as drift correction, waveform extraction, quality metrics, and visualization.
    
    Args:
        motion (numpy.ndarray): 2-D array of motion estimates in micrometers (um). The first dimension corresponds to temporal bins and the second dimension typically corresponds to spatial channels or independent motion traces. This array is not modified in-place; the function returns a cleaned copy.
        temporal_bins (numpy.ndarray): 1-D array of temporal bin centers corresponding to the rows of motion. Values are used as the x-axis for interpolation and must be monotonic (strictly increasing or decreasing) and have the same length as motion.shape[0]. Units are seconds and this array determines where masked segments are interpolated.
        bin_duration_s (float): Duration of each temporal bin in seconds. This value is used to convert discrete differences in motion into a speed in units of um/s for thresholding, and to build the Gaussian smoothing kernel if smoothing is requested. Must be positive.
        speed_threshold (float): Maximum allowed speed (in um/s) between two adjacent temporal bins before that interval is considered a spurious jump. Default is 30 (um/s). Intervals where the absolute discrete speed (numpy.diff(motion, axis=0) / bin_duration_s) exceeds this threshold are marked as offending regions and replaced by linear interpolation across the surrounding non-offending samples.
        sigma_smooth_s (float or None): Standard deviation of the Gaussian smoothing kernel in seconds. If None (the default), no smoothing is applied. If provided (a positive float), a Gaussian kernel is built using bin_duration_s and applied with FFT convolution along the temporal axis to produce a temporally smoothed motion trace. The kernel construction accounts for even/odd number of bins to avoid a fixed shift.
    
    Behavior and side effects:
        The function first creates a copy of motion to avoid mutating the input. For each column (motion trace) it computes discrete speeds between adjacent temporal bins, identifies indices where absolute speed > speed_threshold, and constructs masking intervals that cover pairs of such indices. If the number of detected indices is odd, the function attempts to resolve the ambiguity by dropping either the first or the last index to minimize the total masked duration (this is an automatic heuristic used to handle edge cases). All masked samples are replaced by values obtained from scipy.interpolate.interp1d using temporal_bins and the unmasked motion samples. If sigma_smooth_s is provided, the function then convolves each motion trace with a normalized Gaussian kernel using scipy.signal.fftconvolve with mode="same" along the temporal axis. The function imports scipy.interpolate and scipy.signal internally; these runtime imports mean a scipy installation is required for interpolation and smoothing. The function returns a new numpy.ndarray with the same shape as the input motion array containing the cleaned motion estimates (units: um).
    
    Failure modes and important notes:
        If temporal_bins[mask] contains fewer than two points for any trace (for example, if the mask filters out all or all-but-one samples), scipy.interpolate.interp1d will raise a ValueError because interpolation is not possible with fewer than two points. The function does not perform explicit checks for monotonic temporal_bins; passing non-monotonic temporal_bins may cause interp1d to raise an error. Passing sigma_smooth_s equal to zero will lead to division by zero when constructing the Gaussian kernel; sigma_smooth_s must be None or a positive float. Inputs with NaNs in motion or temporal_bins may propagate or cause failures in interpolation/convolution. The function assumes motion is a 2-D numpy.ndarray; passing arrays of different dimensionality will raise indexing or shape errors.
    
    Returns:
        numpy.ndarray: A 2-D numpy array with the same shape as the input motion array containing the cleaned motion estimates (units: micrometers). This array is a copy and the original motion argument is not modified.
    """
    from spikeinterface.sortingcomponents.motion.motion_cleaner import clean_motion_vector
    return clean_motion_vector(
        motion,
        temporal_bins,
        bin_duration_s,
        speed_threshold,
        sigma_smooth_s
    )


################################################################################
# Source: spikeinterface.sortingcomponents.motion.motion_utils.get_rigid_windows
# File: spikeinterface/sortingcomponents/motion/motion_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_sortingcomponents_motion_motion_utils_get_rigid_windows(
    spatial_bin_centers: numpy.ndarray
):
    """spikeinterface.sortingcomponents.motion.motion_utils.get_rigid_windows generates a single rectangular (all-ones) spatial window that covers the full set of provided spatial bins and computes the spatial center of that window. In the SpikeInterface motion-correction and sorting-components context, this function is used for rigid motion models where the entire probe is treated as a single region: the returned window can be used to weight or aggregate signals across all spatial bins when estimating or applying rigid motion corrections.
    
    Args:
        spatial_bin_centers (numpy.ndarray): 1D numpy array of spatial bin center positions along the probe (numeric, e.g., in micrometers). This array defines the spatial locations of the bins that the window will cover. The function expects at least one element; spatial_bin_centers[0] and spatial_bin_centers[-1] are used to compute the window center. The array should represent positions ordered along the spatial axis (increasing or decreasing); if it is unsorted, the computed center will be the midpoint of the first and last entries rather than the true span center.
    
    Returns:
        tuple: A pair containing (windows, window_centers).
            windows (numpy.ndarray): 2D array of shape (1, N) with dtype float64 where N == spatial_bin_centers.size. This array is filled with ones and represents a single rectangular weighting window that spans all provided spatial bins. In practical use within SpikeInterface, this window signifies that all spatial bins are grouped together with equal weight for rigid-motion computations.
            window_centers (numpy.ndarray): 1D array of shape (1,) containing the center position of the rectangular window computed as (spatial_bin_centers[0] + spatial_bin_centers[-1]) / 2.0. This value gives the representative spatial location of the single rigid window and can be used downstream to label or position the window in visualization or motion-estimation algorithms.
    
    Behavior and failure modes:
        The function constructs windows with numpy.ones and computes the center as the midpoint of the first and last entries of spatial_bin_centers. There are no side effects (no in-place modification of the input). If spatial_bin_centers has size 0, indexing spatial_bin_centers[0] or spatial_bin_centers[-1] will raise an IndexError; the function does not perform explicit validation beyond relying on numpy indexing. If spatial_bin_centers is not a 1D numpy.ndarray, the shape of the returned arrays may not match the documented shapes and downstream code expecting a 1D input may fail.
    """
    from spikeinterface.sortingcomponents.motion.motion_utils import get_rigid_windows
    return get_rigid_windows(spatial_bin_centers)


################################################################################
# Source: spikeinterface.sortingcomponents.motion.motion_utils.get_spatial_windows
# File: spikeinterface/sortingcomponents/motion/motion_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_sortingcomponents_motion_motion_utils_get_spatial_windows(
    contact_depths: numpy.ndarray,
    spatial_bin_centers: numpy.ndarray,
    rigid: bool = False,
    win_shape: str = "gaussian",
    win_step_um: float = 50.0,
    win_scale_um: float = 150.0,
    win_margin_um: float = None,
    zero_threshold: float = None
):
    """spikeinterface.sortingcomponents.motion.motion_utils.get_spatial_windows
    Generate spatial tapering windows used for estimating non-rigid motion along a probe depth axis. In the SpikeInterface motion-correction workflow this function converts electrode/contact depths and precomputed spatial bin centers into a set of overlapping spatial windows (tapers) that are used to weight signals or drift estimates at different depths. For rigid motion this function returns a single rectangular window that effectively covers the whole probe (equivalent to estimating only a global shift). Windows may be shaped as Gaussian, rectangular, or triangular and are placed regularly across the probe depth range; window placement, spread, and margins are controlled by the win_step_um, win_scale_um and win_margin_um parameters. When a zero_threshold is provided, values below this threshold are set to zero and each window row is normalized to sum to 1 (see failure modes below).
    
    Args:
        contact_depths (numpy.ndarray): Position of electrodes along the correction direction (depth). This is a 1D array with shape (num_channels,). The function uses the minimum and maximum of these depths to determine the probe span and to center the spatial windows between the probe borders. Accurate contact_depths are required for meaningful spatial window placement in the non-rigid motion estimation pipeline.
        spatial_bin_centers (numpy.ndarray): Pre-computed centers of spatial bins along the same depth axis. This is a 1D array with length equal to the number of spatial bins used by downstream motion estimation code. Each returned window is an array of scaling coefficients evaluated at these bin centers.
        rigid (bool): If True, compute and return a single rectangular window that covers the entire probe (rigid motion model). In the SpikeInterface pipeline this corresponds to estimating one global shift for the whole probe instead of multiple local shifts. Default is False (compute multiple overlapping windows for non-rigid motion).
        win_shape (str): Shape of each spatial window. Accepted values (as implemented) are "gaussian", "rect", and "triangle". "gaussian" builds a Gaussian taper centered at the window center with standard deviation win_scale_um. "rect" builds a binary rectangular window of width win_scale_um. "triangle" builds a triangular (piecewise linear) window of half-width win_scale_um/2 scaled to lie between 0 and 1. Default is "gaussian".
        win_step_um (float): Spacing (in micrometers) between adjacent window centers along the depth axis. Windows are placed regularly every win_step_um across the probe span. Smaller values produce more windows and more overlap; larger values produce fewer windows. Default is 50.0.
        win_scale_um (float): Scale parameter (in micrometers) controlling window width. For win_shape == "gaussian" this is the Gaussian sigma. For win_shape == "rect" this is the rectangle width. For win_shape == "triangle" this controls the half-width used to build the triangular shape. Default is 150.0.
        win_margin_um (None | float): Margin (in micrometers) to extend (if positive) or shrink (if negative) the probe depth range before computing window centers. When None (default) the function sets win_margin_um = -win_scale_um / 2.0 to ensure first and last windows do not overflow the probe borders. Explicitly setting this value shifts how close the outermost windows are placed relative to the min/max contact_depths.
        zero_threshold (None | float): If provided, values in each window below this lower threshold are set to zero, and then each window row is normalized by its row sum so that rows sum to 1. This is useful to sparsify or truncate small window weights. Default is None (no thresholding or row normalization).
    
    Returns:
        tuple:
            windows (numpy.ndarray): 2D array of scaling coefficients for each window evaluated at spatial_bin_centers. Shape is (num_windows, num_spatial_bins). num_windows equals the number of window centers returned. For rigid=True this will usually be a single-row array representing one rectangular window; for non-rigid it will be multiple rows corresponding to regularly spaced window centers.
            window_centers (numpy.ndarray): 1D array of window center positions (in the same units as contact_depths and win_step_um), length equal to the number of windows (i.e., windows.shape[0]). Centers are computed to be regularly spaced with spacing win_step_um and placed between the min and max contact_depths taking win_margin_um into account.
    
    Behavior, defaults and side effects:
        - For rigid=True the function delegates to a rigid-window generator (returning a single rectangular window that covers the probe) so downstream motion estimation will act as a single global shift estimator.
        - For non-rigid mode the function:
            - if win_margin_um is None sets win_margin_um = -win_scale_um / 2.0 to avoid window overflow at probe edges.
            - computes the probe span from numpy.min(contact_depths) - win_margin_um to numpy.max(contact_depths) + win_margin_um, computes an integer number of steps num_windows = int((max_ - min_) // win_step_um), and then places window_centers = numpy.arange(num_windows + 1) * win_step_um + min_ + border where border = ((max_ - min_) % win_step_um) / 2.
            - builds each window according to win_shape:
                - "gaussian": win = exp(-((spatial_bin_centers - center)**2) / (2 * win_scale_um**2))
                - "rect": win = 1.0 where abs(spatial_bin_centers - center) < (win_scale_um / 2.0) else 0.0
                - "triangle": a triangular profile centered at center, scaled to span 0..1 within half-width win_scale_um/2
        - If win_scale_um <= win_step_um / 5.0, the function will emit a warning indicating windows are likely non-overlapping; this is a caution for the user that non-overlap may degrade non-rigid motion estimation quality.
        - If the computed num_windows < 1 (the probe depth range is too short relative to win_step_um and win_scale_um), the function forces num_windows = 1 and emits a warning recommending switching to rigid motion; in such cases the function effectively falls back to a single-window solution.
        - If zero_threshold is not None then after thresholding values below zero_threshold are set to zero and each window row is normalized by its row sum. Note that if a row becomes all zeros due to thresholding, the subsequent normalization will attempt to divide by zero and can produce NaNs or Infs; the caller must choose zero_threshold appropriately or post-process results to handle such degenerate rows.
        - Unless zero_threshold is provided, the function does not normalize windows to sum to 1. Users relying on normalized windows should explicitly apply normalization if needed.
    
    Failure modes and warnings:
        - Supplying win_scale_um that is too small relative to win_step_um can produce non-overlapping windows and a warning is emitted.
        - If the probe depth range (numpy.ptp(contact_depths)) is smaller than the spacing/margin parameters the function will warn and fall back to a single-window (rigid-like) solution.
        - Providing a zero_threshold that zeroes entire rows will lead to division-by-zero during the internal normalization step and result in NaNs/Infs in the returned array; the function does not internally handle this case beyond performing the division.
    
    Notes:
        - The returned windows are intended for use as spatial tapers in the SpikeInterface non-rigid motion estimation pipeline and are analogous to the overlapping windows used by other tools (for example, kilosort2.5 uses overlapping rectangular windows). By default this implementation uses Gaussian windows for smooth tapering unless another shape is requested.
    """
    from spikeinterface.sortingcomponents.motion.motion_utils import get_spatial_windows
    return get_spatial_windows(
        contact_depths,
        spatial_bin_centers,
        rigid,
        win_shape,
        win_step_um,
        win_scale_um,
        win_margin_um,
        zero_threshold
    )


################################################################################
# Source: spikeinterface.sortingcomponents.peak_selection.select_peak_indices
# File: spikeinterface/sortingcomponents/peak_selection.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_sortingcomponents_peak_selection_select_peak_indices(
    peaks: numpy.ndarray,
    method: str,
    seed: int,
    **method_kwargs
):
    """Select indices of detected peaks to subsample before downstream clustering.
    
    This function is used within the SpikeInterface sorting-components pipeline to subsample the set of detected peaks (spike candidates) prior to feature extraction and clustering. It is wrapped by spikeinterface.sortingcomponents.peak_selection.select_peaks and implements several strategies to reduce the number of peaks passed to clustering: uniform random sampling across all peaks or per-channel, and three "smart" sampling strategies that attempt to produce a more homogeneous sampling across amplitudes, spatial locations, or spatial locations combined with time. The function uses numpy's Generator for randomized selection and sklearn.preprocessing.QuantileTransformer to remap distributions when required.
    
    Args:
        peaks (numpy.ndarray): A 1-D numpy structured array of detected peaks where each element represents a detected peak (spike candidate). The code expects the array to expose at least the structured fields "channel_index" (integer channel id for the peak), "amplitude" (numeric amplitude used for SNR/selection), "sample_index" (temporal sample index) and "segment_index" (segment id). Each entry is a candidate that can be selected; indices returned by this function index into this array.
        method (str): Selection method to use. Supported values (implemented in this function) are "uniform", "smart_sampling_amplitudes", "smart_sampling_locations", and "smart_sampling_locations_and_time". "uniform" performs pure random subsampling (optionally per channel). The "smart_sampling_*" methods attempt to produce a more uniform coverage of the amplitude distribution ("smart_sampling_amplitudes"), the spatial locations ("smart_sampling_locations"), or spatial locations together with time ("smart_sampling_locations_and_time") by using quantile preprocessing and a rejection-sampling-like procedure described in the source. If an unknown method string is provided, the function raises NotImplementedError.
        seed (int): Integer seed for numpy.random.default_rng to make random choices reproducible. The implementation treats any falsy value (for example 0 or None) as None, which produces a non-deterministic Generator; to obtain deterministic behavior supply a non-zero integer seed. This value controls the RNG used for sampling and permutations and therefore affects reproducibility of the selected indices.
        method_kwargs (dict): Additional keyword arguments that control parameters required by the chosen method. The function updates a method-specific params dictionary with values from method_kwargs and asserts presence of required keys. Known parameters used by methods (the caller should supply them when appropriate) include:
            For "uniform": "n_peaks" (int, required) and "select_per_channel" (bool, optional, default False). If "select_per_channel" is True, up to n_peaks are selected per channel; otherwise up to n_peaks are selected in total.
            For "smart_sampling_amplitudes": "n_peaks" (int, required), "noise_levels" (array-like indexed by channel, required), and "select_per_channel" (bool, optional, default False). This method computes per-peak SNR = amplitude / noise_levels[channel], applies sklearn.preprocessing.QuantileTransformer(output_distribution="uniform") to the SNRs and performs a randomized selection to flatten the SNR distribution among the selected peaks.
            For "smart_sampling_locations": "n_peaks" (int, required) and "peaks_locations" (dict with keys "x" and "y", both array-like, required). The function uses the locations to perform quantile preprocessing and randomized selection to obtain spatially uniform sampling.
            For "smart_sampling_locations_and_time": "n_peaks" (int, required) and "peaks_locations" (dict with keys "x" and "y", both array-like, required). This method additionally uses peaks["sample_index"] combined with x and y and applies quantile preprocessing to the triplet (x, y, t) before randomized selection.
            The function asserts presence of required parameters and will raise AssertionError with explanatory text if a required parameter is missing. Do not pass unexpected keys and ensure array-like parameters have lengths consistent with peaks when appropriate.
    
    Behavior, defaults, and failure modes:
        The function constructs an rng via numpy.random.default_rng(seed if seed else None). If seed is falsy, the RNG is non-deterministic. For uniform selection, if n_peaks is larger than the available number of peaks (or peaks per channel when selecting per channel), all available peaks are returned for that scope. For smart sampling methods, QuantileTransformer is used with n_quantiles = min(100, n_samples) where n_samples is the number of values being transformed; sklearn must be available. The selection process uses repeated candidate drawing with random probabilities until at least n_peaks candidates are accumulated, then a random permutation is applied and the first n_peaks are kept. The function raises AssertionError when required method_kwargs entries are missing (for example missing "n_peaks", "noise_levels", or "peaks_locations" as appropriate). If method is not one of the implemented strings, NotImplementedError is raised listing that the method does not exist for peaks selection.
    
    Returns:
        numpy.ndarray: A 1-D numpy array of integer indices into the input peaks array. The returned indices represent the subsampled set of peaks chosen according to the requested method. The array is sorted by (sample_index, segment_index) using numpy.lexsort so that returned indices are ordered temporally within segments. If n_peaks is zero or there are no peaks, an empty numpy.ndarray is returned. The function does not modify the input peaks array; it only computes and returns index positions.
    """
    from spikeinterface.sortingcomponents.peak_selection import select_peak_indices
    return select_peak_indices(peaks, method, seed, **method_kwargs)


################################################################################
# Source: spikeinterface.sortingcomponents.waveforms.waveform_utils.from_temporal_representation
# File: spikeinterface/sortingcomponents/waveforms/waveform_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_sortingcomponents_waveforms_waveform_utils_from_temporal_representation(
    temporal_waveforms: numpy.ndarray,
    num_channels: int
):
    """spikeinterface.sortingcomponents.waveforms.waveform_utils.from_temporal_representation converts a temporal (flattened) waveform representation back into a per-waveform, per-time, per-channel numpy array. This function is the inverse operation of to_temporal_representation used in the SpikeInterface sorting components to move between a stacked temporal format (channels concatenated along the first axis) and a structured waveform array used for visualization, feature extraction, quality metrics, and other post-processing steps in spike sorting pipelines.
    
    Args:
        temporal_waveforms (numpy.ndarray): A 2-D NumPy array containing waveforms in temporal representation. The array shape must be (num_temporal_waveforms, num_time_samples), where num_temporal_waveforms is the product of the original number of waveforms and num_channels. This argument is typically produced by to_temporal_representation and therefore is expected to have channels stacked along the first axis for each time sample block. The function reads the shape directly and does not modify the input in place.
        num_channels (int): The integer number of channels that were present when the temporal representation was created. This value must be a positive integer and must evenly divide the first dimension of temporal_waveforms (num_temporal_waveforms). It controls how the first axis is unstacked into separate channels per waveform.
    
    Returns:
        numpy.ndarray: A NumPy array containing the reconstructed waveforms. The returned array has shape (num_waveforms, num_time_samples, num_channels), where num_waveforms is computed as num_temporal_waveforms // num_channels. The function achieves this by reshaping the input to (num_waveforms, num_channels, num_time_samples) and then swapping the channel and time axes, so the final axis order is (waveform index, time sample, channel index). The returned array may be a view or a copy depending on NumPy's internal memory layout.
    
    Behavior and failure modes:
        The function expects a 2-D numpy.ndarray for temporal_waveforms and a positive integer for num_channels. If temporal_waveforms.ndim != 2, or if num_channels does not divide the first dimension of temporal_waveforms evenly, NumPy will raise a ValueError during the reshape operation. If a non-NumPy array or an object lacking a shape attribute is passed, the function may raise an AttributeError or TypeError. There are no side effects: the input array is not modified in place. This function is intended to be used in SpikeInterface workflows to reconstruct per-spike waveforms from the compact temporal representation for downstream analysis, plotting, and metric computations.
    """
    from spikeinterface.sortingcomponents.waveforms.waveform_utils import from_temporal_representation
    return from_temporal_representation(temporal_waveforms, num_channels)


################################################################################
# Source: spikeinterface.sortingcomponents.waveforms.waveform_utils.to_temporal_representation
# File: spikeinterface/sortingcomponents/waveforms/waveform_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_sortingcomponents_waveforms_waveform_utils_to_temporal_representation(
    waveforms: numpy.ndarray
):
    """spikeinterface.sortingcomponents.waveforms.waveform_utils.to_temporal_representation: Convert a 3D array of extracted spike waveforms into a 2D temporal-only representation by collapsing the channel (spatial) dimension so each output row is the time-series from a single channel of a single waveform. This is used in the SpikeInterface waveform processing and sorting components when downstream algorithms require temporal-only inputs (for example, time-domain feature extraction, PCA over time, or classifier inputs) rather than a per-waveform multichannel spatial representation.
    
    Args:
        waveforms (numpy.ndarray): A 3-dimensional NumPy array containing extracted spike waveforms with shape (num_waveforms, num_time_samples, num_channels). Each entry corresponds to the recorded voltage (or signal) at a given time sample for a given channel of a given waveform. The function expects this exact layout; providing an array with a different number of dimensions will raise an error. The array's dtype is preserved in the output when possible.
    
    Returns:
        numpy.ndarray: A 2-dimensional NumPy array named temporal_waveforms with shape (num_waveforms * num_channels, num_time_samples). Each row is a temporal waveform corresponding to one channel of one original waveform (channels are interleaved per waveform). The returned array contains the same numeric values as the input arranged in a temporal-only format. The function does not modify the input array in-place; the result may be a view or a copy depending on the input memory layout. If the input is not a 3D NumPy array, the function will fail (for example, unpacking the shape will raise a ValueError or a TypeError if the input lacks a shape attribute).
    """
    from spikeinterface.sortingcomponents.waveforms.waveform_utils import to_temporal_representation
    return to_temporal_representation(waveforms)


################################################################################
# Source: spikeinterface.widgets.unit_waveforms.get_waveforms_scales
# File: spikeinterface/widgets/unit_waveforms.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_widgets_unit_waveforms_get_waveforms_scales(
    templates: numpy.ndarray,
    channel_locations: numpy.ndarray,
    nbefore: int,
    x_offset_units: bool = False,
    widen_narrow_scale: float = 1.0
):
    """Return x and amplitude scales used to plot waveform templates arranged by channel
    location for spike-sorting visualization.
    
    This function computes a horizontal coordinate matrix (xvectors) that places each
    sample of template waveforms along an x-axis offset by channel x-positions,
    a scalar y_scale that maps waveform amplitude units to spatial vertical units,
    a per-channel vertical offset (y_offset) derived from channel y-positions, and
    an estimated horizontal inter-channel interval (delta_x). These outputs are
    intended for use when drawing template waveforms (e.g., template stacks or
    unit waveform overlays) on a 2D layout determined by electrode/contact
    locations, a common step in SpikeInterface visualization and waveform analysis
    workflows.
    
    Args:
        templates (numpy.ndarray): Array of template waveforms used to determine
            amplitude range and number of samples. The code expects templates to
            have samples along axis 1 (for example shape (n_templates, nsamples,
            n_channels) as used in SpikeInterface waveform extractors). The global
            maximum and minimum of this array are used to scale amplitudes so that
            plotted waveforms fit the spatial layout. The length of the first axis
            (len(templates)) is used only when x_offset_units is True to scale x
            channel locations.
        channel_locations (numpy.ndarray): 2D array of channel coordinates with
            shape (n_channels, 2). Axis 1 must contain [x, y] positions for each
            channel in the same spatial units (e.g., micrometers or electrode
            grid units). These coordinates define the spatial layout: x values are
            used as horizontal offsets and y values as vertical offsets (y_offset)
            for plotting. Inconsistent shapes or improper ordering will raise
            numpy broadcasting or indexing errors.
        nbefore (int): Number of samples before the event/time-lock point in the
            templates. Used to center the x vector so that sample index nbefore maps
            to x = 0. This determines the horizontal placement of the time-lock
            point within each plotted waveform.
        x_offset_units (bool): If False (default), channel x positions from
            channel_locations are used directly as horizontal offsets (spatial
            units). If True, channel x positions are multiplied by len(templates)
            (the number of templates along axis 0) before being used as offsets.
            This provides an alternate unit convention where x offsets are scaled
            by the number of templates; it is retained for backward compatibility
            with previous plotting conventions in SpikeInterface.
        widen_narrow_scale (float): Multiplicative factor applied to the computed
            horizontal sample spacing (delta_x). Values > 1 widen the horizontal
            spacing between sample points (useful to emphasize waveform shape),
            values < 1 compress it. Default is 1.0. This parameter only affects
            the computed x vector scale and does not mutate the input arrays.
    
    Returns:
        xvectors (numpy.ndarray): 2D array of horizontal coordinates for plotting
            with shape (nsamples, n_channels). Each column corresponds to the x
            coordinates to use for plotting the waveform samples of the associated
            channel. The last row is set to numpy.nan to provide a plotting
            discontinuity between successive waveforms when drawing lines.
        y_scale (float): Scalar factor that maps waveform amplitude units (same as
            templates) to vertical spatial units derived from channel_locations.
            It is computed from the weighted average inter-channel vertical
            spacing and the maximum absolute template amplitude so that waveform
            peaks are scaled to fit the electrode layout for visualization.
        y_offset (numpy.ndarray): 2D row vector with shape (1, n_channels)
            containing the vertical offsets (channel_locations[:, 1]) for each
            channel. When plotting, each channel's waveform should be shifted by
            this offset and multiplied by y_scale to place it at the correct
            electrode y-position.
        delta_x (float): Estimated horizontal interval between channels computed
            from the channel_locations using a Gaussian-weighted, angular-penalized
            average of pairwise distances. This value represents the characteristic
            inter-channel x spacing used to scale the x coordinate vector before
            applying widen_narrow_scale.
    
    Behavior and failure modes:
        - If channel layout is a single column (all x positions identical) the
          horizontal weight sum becomes zero and the function falls back to
          delta_x = 10. Similarly, if layout is a single row (all y positions
          identical) it falls back to delta_y = 10. These are fallback defaults to
          avoid division-by-zero in degenerate layouts.
        - If two or more channels share identical coordinates, eucl.min() may be
          zero; this can produce runtime warnings or invalid intermediate values in
          the Gaussian distance penalty calculation. Ensure channel_locations
          contain distinct coordinates when possible.
        - The function does not modify the input arrays except when x_offset_units
          is True the code multiplies a temporary copy of the channel x positions
          by len(templates); the original channel_locations passed by the caller is
          not mutated by this function (no in-place changes are performed).
        - Incompatible shapes (for example, channel_locations not shaped as
          (n_channels, 2) or templates with samples not on axis 1) will raise
          numpy indexing/broadcasting errors; the caller must provide arrays in the
          expected layout used throughout SpikeInterface waveform visualization.
    """
    from spikeinterface.widgets.unit_waveforms import get_waveforms_scales
    return get_waveforms_scales(
        templates,
        channel_locations,
        nbefore,
        x_offset_units,
        widen_narrow_scale
    )


################################################################################
# Source: spikeinterface.widgets.utils.array_to_image
# File: spikeinterface/widgets/utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_widgets_utils_array_to_image(
    data: numpy.ndarray,
    colormap: str = "RdGy",
    clim: tuple = None,
    spatial_zoom: tuple = (0.75, 1.25),
    num_timepoints_per_row: int = 30000,
    row_spacing: float = 0.25,
    scalebar: bool = False,
    sampling_frequency: float = None
):
    """spikeinterface.widgets.utils.array_to_image converts a 2D numpy array of time-series data (time x channels) into a 3-channel RGB image array suitable for visualization in SpikeInterface widgets and other plotting/export workflows. This function is intended for visualizing extracellular recording data or preprocessed traces (for example, raw or filtered traces arranged as samples-by-channels produced during spike sorting workflows described in the SpikeInterface README). It applies spatial scaling, color mapping, clamping to color limits, row-wise wrapping (to limit samples per row), and optional scalebar drawing (requires Pillow and a provided sampling frequency).
    
    Args:
        data (numpy.ndarray): A 2D numpy array with shape (num_timepoints, num_channels). In the SpikeInterface domain this typically represents recorded or preprocessed extracellular traces where rows correspond to time samples and columns correspond to recording channels. The function expects a numeric array; a non-2D array will cause indexing errors. Values are used to compute color mapping and clamping.
        colormap (str): Identifier for a Matplotlib colormap to map normalized data values to RGB colors. The default is "RdGy". The string must be a valid key in matplotlib.pyplot.colormaps; otherwise a KeyError will propagate. The colormap is applied after data is normalized to the [0,1] range determined by clim.
        clim (tuple or None): A (min, max) tuple specifying color limits used to clamp and scale data before color mapping. If None (default), clim is set to (data.min(), data.max()). The function asserts that a provided clim has length 2. If clim[0] == clim[1] a division by zero will occur when normalizing; callers should avoid equal limits.
        spatial_zoom (tuple): A 2-tuple (time_scale, channel_scale) passed to scipy.ndimage.zoom to scale the input array spatially along the time and channel axes respectively. Default is (0.75, 1.25). Values less than 1 shrink the corresponding dimension and values greater than 1 enlarge it. The zoom operation changes the number of pixels along each axis used for the RGB image and thus affects the final image resolution.
        num_timepoints_per_row (int): Maximum number of time samples (before scaling) to place on a single horizontal row in the output image; if the recording has more samples, the data is wrapped to additional rows. Default is 30000. After spatial_zoom is applied the effective per-row pixel width is computed from this value and spatial_zoom[0]. The function computes the number of rows as ceil(num_timepoints / num_timepoints_per_row).
        row_spacing (float): Ratio of vertical spacing (gap) between consecutive rows relative to overall channel pixel height. Default is 0.25. This value is used with spatial_zoom[1] and the number of channels to compute the integer spacing in pixels inserted between rows; spacing increases the vertical separation so multiple wrapped rows do not overlap.
        scalebar (bool): If True, draw a time scalebar (and text) onto the output image. Default is False. Enabling this option requires Pillow (PIL) to be installed; otherwise an ImportError with instructions to install pillow is raised. Also requires sampling_frequency to be provided (not None); otherwise an assertion error is raised.
        sampling_frequency (float or None): Sampling frequency in Hz used to compute the time length (in milliseconds) of the per-row segment for the scalebar. Required (must be non-None) when scalebar is True. If provided but a font cannot be loaded from common system paths the function will print a warning and omit the scalebar drawing.
    
    Returns:
        output_image (numpy.ndarray): A 3D uint8 numpy array with shape (vertical_pixels, horizontal_pixels, 3) representing an RGB image (values 0-255). The image background is white (255) by default and the visualized traces occupy the left portion of each row up to the scaled per-row width; unused right-side columns remain white. The returned array is suitable for display in GUI widgets or writing to image files.
    
    Additional behavior, side effects, and failure modes:
        - The function clamps input data to the provided clim range, then linearly normalizes to [0,1] using numpy.ptp(clim) before applying the Matplotlib colormap. If clim is None the function uses the data minimum and maximum.
        - Colorization is performed by cmap(normalized_data) and converted to uint8 in the 0-255 RGB range. The channel ordering is flipped vertically (numpy.flip along axis=0) so higher-channel indices appear at the top of the output image.
        - The function uses scipy.ndimage.zoom for spatial scaling; SciPy must be available. Errors from scipy.ndimage.zoom will propagate to the caller.
        - num_timepoints_per_row is applied to the original number of timepoints to compute the number of rows; after scaling the last row may be partially filled and remaining horizontal pixels are left white.
        - If scalebar is True and Pillow (PIL) is missing, the function raises ImportError with the message "To add a scalebar, you need pillow: >>> pip install pillow". If a font file cannot be loaded from common system paths the function prints "Could not load font to use in scalebar. Scalebar will not be drawn." and will skip drawing the scalebar without raising.
        - If colormap is not found in Matplotlib, a KeyError will be raised. If clim has length different from 2 an AssertionError is raised. If clim has zero range normalization will produce a division by zero.
        - The function returns an RGB image as uint8; callers should not assume any additional metadata (such as axis labels or sampling rate) is embedded in the array. For integration into SpikeInterface visualization pipelines, pass the returned array directly to GUI image widgets or save it using standard image libraries.
    """
    from spikeinterface.widgets.utils import array_to_image
    return array_to_image(
        data,
        colormap,
        clim,
        spatial_zoom,
        num_timepoints_per_row,
        row_spacing,
        scalebar,
        sampling_frequency
    )


################################################################################
# Source: spikeinterface.widgets.utils.get_some_colors
# File: spikeinterface/widgets/utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def spikeinterface_widgets_utils_get_some_colors(
    keys: list,
    color_engine: str = "auto",
    map_name: str = "gist_ncar",
    format: str = "RGBA",
    shuffle: bool = None,
    seed: int = None,
    margin: int = None,
    resample: bool = True
):
    """get_some_colors returns a dictionary that maps the provided keys to RGBA color tuples suitable for use in SpikeInterface widget visualizations (for example coloring units, channels, or clusters in plots). This function selects a color-generation engine, produces one color per key, optionally resamples a matplotlib colormap, and can shuffle the colors deterministically using a numpy random generator. It is used by spikeinterface.widgets to assign consistent, visually distinct colors when displaying spike-sorting results and recordings.
    
    Args:
        keys (list): An ordered list of identifiers for which colors are required. Each element in this list will be a key in the returned dictionary. Typical domain usage in SpikeInterface is a list of unit IDs, channel names, or categorical labels used by visualization widgets. The function preserves the input order when associating colors with keys (the mapping is created with dict(zip(keys, colors))).
        color_engine (str): "auto" | "matplotlib" | "colorsys" | "distinctipy", default: "auto". Chooses the backend used to generate colors. "auto" selects the first available engine in this priority: matplotlib (preferred when installed), distinctipy (fallback, slower), then the built-in colorsys. If you request a specific engine (for example "matplotlib"), that engine must be importable in the Python environment; otherwise execution will fail when the code later attempts to use the missing module. The choice affects color aesthetics and performance: matplotlib uses colormaps, colorsys generates HSV-derived colors, and distinctipy produces visually distinct colors (but may be slower).
        map_name (str): The matplotlib colormap name to use when color_engine is "matplotlib". This string is passed to matplotlib's colormap API (plt.colormaps[map_name]). In the SpikeInterface context this controls the color palette used for widgets that rely on matplotlib colormaps.
        format (str): Output color format, default: "RGBA". Only the string "RGBA" is accepted by the implementation (an assertion enforces this). The returned color tuples are therefore 4-tuples representing red, green, blue, and alpha channels, with channel values produced by the selected engine.
        shuffle (bool or None): If True, the generated colors are shuffled with a deterministic random order determined by seed. If False, the colors are left in the order produced by the engine. If None (default behavior), the function sets shuffle automatically: True for "matplotlib" and "colorsys", and False for "distinctipy". When shuffle is None and the automatic decision is applied, the function also sets seed to 91 (see seed description).
        seed (int or None): Integer seed for the numpy random generator used when shuffle is True. If None and shuffle is left to automatic selection (shuffle is None on call), the function sets seed to 91. The implementation uses numpy.random.default_rng(seed=seed) to shuffle indices, so the seed controls reproducible shuffling across runs for widget color assignments.
        margin (int or None): Margin applied only when using the "matplotlib" engine with resample=True. If None and resample=True, the function computes margin = max(4, int(N * 0.08)) where N is the number of keys; this avoids sampling from potentially undesirable border colors of some matplotlib colormaps. If an integer is provided, it is used directly as the margin (the code will index the resampled colormap with offsets of +margin). For non-matplotlib engines, margin is ignored.
        resample (bool): For the "matplotlib" engine, when True the function will resample the specified colormap to exactly N + 2*margin colors (where N is number of keys) and then pick colors offset by margin to avoid border colors. If False, the function will sample the colormap directly without resampling. This parameter has no effect for "colorsys" or "distinctipy" engines.
    
    Behavior, defaults, and failure modes:
        - The function first attempts to import matplotlib.pyplot and distinctipy to detect available engines. If color_engine is "auto", it prefers matplotlib (if importable), otherwise distinctipy, otherwise falls back to the built-in colorsys implementation.
        - The function asserts that color_engine is one of ("auto", "distinctipy", "matplotlib", "colorsys") and that format equals "RGBA"; providing unsupported values will raise an AssertionError.
        - If a specific engine is requested (for example "matplotlib") but that package is not importable in the current environment, subsequent use of that engine will raise an ImportError or NameError when the code tries to access the engine-specific API. Prefer "auto" when package availability is uncertain.
        - For N = len(keys), the function produces exactly N RGBA color tuples. distinctipy colors are converted from RGB to RGBA by appending alpha=1.0. colorsys colors are produced using HSV evenly spaced hues with fixed saturation and value (s=0.5, v=0.5) and then appended with alpha=1.0, producing moderately saturated/darker colors. Matplotlib colormap sampling returns RGBA tuples as provided by the selected colormap.
        - If shuffle is True the function uses numpy.random.default_rng(seed=seed) to deterministically permute the color ordering; seed controls reproducibility. If shuffle is False, the colors remain in the generation order which reflects either the colormap sequence or the algorithmic order from distinctipy/colorsys.
        - The function relies on numpy being available for shuffling. If numpy is not available and shuffle is requested, runtime errors will occur.
        - The returned color values are floating-point channel values produced by the respective engines (typically in the 0.0–1.0 range as generated by matplotlib, colorsys, or distinctipy).
    
    Returns:
        dict: A dictionary mapping each element of the input keys list to a color tuple in RGBA format (4-tuple of floats). The mapping preserves the correspondence to the input keys order (constructed via dict(zip(keys, colors))). These color tuples are intended for use in SpikeInterface widgets and plotting routines to color units, channels, clusters, or other categorical elements consistently across visualizations.
    """
    from spikeinterface.widgets.utils import get_some_colors
    return get_some_colors(
        keys,
        color_engine,
        map_name,
        format,
        shuffle,
        seed,
        margin,
        resample
    )


from typing import Dict, Any


def get_tools() -> Dict[str, Dict[str, Any]]:
    """Extract JSON schemas for all functions in this module."""
    import sys
    import os
    
    # Add project root to path to import our json_schema module
    # Try multiple possible paths
    possible_paths = [
        os.path.join(os.path.dirname(__file__), '..', '..', 'utils'),
        '/app/utils',
        '/app/project/utils',
    ]
    
    json_schema_path = None
    for path in possible_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(os.path.join(abs_path, 'json_schema.py')):
            if abs_path not in sys.path:
                sys.path.insert(0, abs_path)
            json_schema_path = abs_path
            break
    
    if json_schema_path:
        from json_schema import get_json_schema
    else:
        # Fallback to transformers if our module not found
        from transformers.utils import get_json_schema
    
    tools = {}
    failed_count = 0
    
    for name, func in get_lib().items():
        try:
            tools[name] = get_json_schema(func)
        except Exception as e:
            failed_count += 1
            # Only print first few errors to avoid spam
            if failed_count <= 3:
                print(f"Failed to get schema for {name}: {type(e).__name__}: {e}", file=sys.stderr)
            continue
    
    if failed_count > 0:
        print(f"Warning: Failed to extract schemas for {failed_count} out of {len(get_lib())} functions", file=sys.stderr)
    
    return tools


def get_lib():
    """Get all functions defined in this module."""
    import inspect
    global_vars = inspect.currentframe().f_globals
    
    functions = {
        name: obj for name, obj in global_vars.items()
        if inspect.isfunction(obj) and obj.__module__ == __name__
    }
    functions.pop("get_lib", None)
    functions.pop("get_tools", None)
    return functions
