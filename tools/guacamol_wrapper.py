"""
Regenerated Google-style docstrings for module 'guacamol'.
README source: others/readme/guacamol/README.md
Generated at: 2025-12-02T00:45:33.311754Z

Total functions: 5
"""


from typing import List

################################################################################
# Source: guacamol.goal_directed_score_contributions.uniform_specification
# File: guacamol/goal_directed_score_contributions.py
# Category: valid
################################################################################

def guacamol_goal_directed_score_contributions_uniform_specification(*top_counts: int):
    """guacamol.goal_directed_score_contributions.uniform_specification creates a ScoreContributionSpecification that assigns equal weight to each specified top-x contribution used in goal-directed benchmark scoring.
    
    This function is part of the GuacaMol benchmarking suite for de novo molecular design (see README). In the goal-directed benchmarks, scoring functions may compute contributions from the top-x matching subcomponents or predictions; this helper builds a specification listing those top-x values each paired with the same weight (1.0). The returned ScoreContributionSpecification can be passed to the goal-directed scoring machinery to indicate that all listed top-x contributions should be treated with equal importance.
    
    Args:
        top_counts (int): Variable number of integer arguments. Each integer x represents a "top-x" contribution to include in the specification. For every x provided, the function creates a tuple (x, 1.0) and includes it in the resulting ScoreContributionSpecification in the same order as the arguments. If no arguments are provided, the function returns a specification with an empty contributions list. The function does not validate values: non-integer inputs or nonsensical integers (e.g., negative values or zero) are not checked here and may lead to unexpected behavior downstream in scoring code. Duplicate integers are preserved and will produce duplicate contribution entries.
    
    Returns:
        guacamol.goal_directed_score_contributions.ScoreContributionSpecification: An instance whose contributions attribute is a list of (top_count, weight) tuples constructed from the provided top_counts arguments, where each weight is set to 1.0. This object is intended for use in goal-directed benchmark scoring to indicate equal weighting of the specified top-x contributions. The function has no side effects and is deterministic.
    """
    from guacamol.goal_directed_score_contributions import uniform_specification
    return uniform_specification(*top_counts)


################################################################################
# Source: guacamol.utils.data.download_if_not_present
# File: guacamol/utils/data.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def guacamol_utils_data_download_if_not_present(filename: str, uri: str):
    """Download a file from a URI to a local path if the local file does not already exist.
    
    This utility is used by guacamol data ingestion workflows (for example the get_data script) to ensure required dataset files used in GuacaMol benchmarking (training/validation/test splits derived from ChEMBL and published on Figshare) are present on disk before further processing. The function checks for an existing file at the provided local path and, if absent, streams the remote resource to that path while displaying a progress bar. It does not perform content validation (for example, MD5 checksum verification) or atomic move semantics; callers should verify file integrity separately if required for reproducibility.
    
    Args:
        filename (str): Local filesystem path where the downloaded file will be stored. In the GuacaMol context this is typically the path to one of the standardized dataset files (e.g., training, validation, test .smiles or .csv files). If a file already exists at this path, the function will not re-download or overwrite it; it will print a message and return, leaving the existing file intact.
        uri (str): Source URI of the file to download (for example an https:// or http:// link to a Figshare resource). This string is passed to urllib.request.urlretrieve to perform the transfer and must point to a retrievable resource. The function streams the content to disk using an open file handle in binary mode ("wb") and updates a ProgressBarUpTo instance to display progress.
    
    Behavior and side effects:
        If the file at filename already exists (os.path.isfile(filename) is True), the function prints a reuse message and does nothing else. If the file does not exist, the function opens filename for writing in binary mode and calls urlretrieve(uri, filename, reporthook=...), which writes the retrieved bytes directly to the target path while updating a console progress bar. Progress and start/finish messages are printed to standard output. The function does not validate the downloaded content (no checksum comparison), does not perform any post-download verification, and does not implement atomic temporary-file-and-rename behavior; an interrupted download may leave a partially written file at filename.
    
    Failure modes and exceptions:
        Network errors, HTTP errors, permission errors (unable to create or write filename), insufficient disk space, or exceptions raised by urllib.request.urlretrieve or the progress bar implementation will propagate to the caller. Because the function does not catch these exceptions, callers should handle them if they require retry logic, integrity guarantees, or cleanup of partial files.
    
    Returns:
        None: This function does not return a value. Its effect is to ensure that a file exists at the given local path on successful completion (either by reusing an existing file or by downloading and writing the file).
    """
    from guacamol.utils.data import download_if_not_present
    return download_if_not_present(filename, uri)


################################################################################
# Source: guacamol.utils.data.remove_duplicates
# File: guacamol/utils/data.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def guacamol_utils_data_remove_duplicates(list_with_duplicates: list):
    """Removes duplicate elements from a list while preserving the original ordering of the first occurrences.
    
    This function is used in the GuacaMol data-processing pipeline (for example when preparing standardized SMILES training/validation/test sets) to ensure that each molecule or entry appears only once while keeping the original ordering used in the source file. For duplicates, the first occurrence is kept and any later occurrences are ignored. The operation is non-destructive with respect to the input: the input list is not modified and a new list is returned. The function relies on a Python set for membership checks, so elements are expected to be hashable (see Failure modes below). The implementation provides O(n) average-time complexity and O(n) additional memory where n is the length of the input list.
    
    Args:
        list_with_duplicates (list): A list that possibly contains duplicate entries. In the GuacaMol context this is typically a list of molecular identifiers such as SMILES strings read from a dataset file. Each element in the list is treated as a distinct item according to Python equality semantics; the first element that compares equal to a later one is the one that is preserved. Elements are expected to be hashable (for example, strings, integers, tuples). If elements are unhashable (for example, lists or dicts), the function will raise a TypeError when attempting to add them to the internal set.
    
    Returns:
        list: A new list containing the elements from list_with_duplicates with duplicates removed. The order of elements is the same as the order of their first occurrence in the input list. The returned list contains the same element objects (no deep copies are made). No side effects occur on the input list.
    
    Failure modes and notes:
        - If the input contains unhashable elements (e.g., nested lists or dicts), a TypeError will be raised during set insertion; convert or transform such elements to hashable representations before calling this function.
        - Equality semantics follow Python's standard semantics for the element types; objects that compare equal will be treated as duplicates.
        - This function does not perform canonicalization or normalization of molecular representations (for example, it will not convert equivalent SMILES to a canonical form); use a molecule-normalization step before deduplication if canonical equivalence is required for benchmarking reproducibility.
    """
    from guacamol.utils.data import remove_duplicates
    return remove_duplicates(list_with_duplicates)


################################################################################
# Source: guacamol.utils.math.arithmetic_mean
# File: guacamol/utils/math.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for arithmetic_mean because the docstring has no description for the argument 'values'
################################################################################

def guacamol_utils_math_arithmetic_mean(values: List[float]):
    """guacamol.utils.math.arithmetic_mean computes the arithmetic mean of a list of float values. This function is a small pure utility used in the GuacaMol benchmarking toolkit for de novo molecular design to aggregate numeric results such as per-molecule scores, per-run metrics, or other scalar quantities that must be averaged when evaluating distribution-learning or goal-directed generation methods.
    
    Args:
        values (List[float]): A list of floating-point values to average. Each element represents a scalar measurement used in GuacaMol benchmarks (for example, a model score, similarity metric, or other numeric per-molecule quantity). The function uses the implementation sum(values) / len(values), so values must be a non-empty list of floats as required by the calling benchmark code; there are no defaults or in-place side effects.
    
    Returns:
        float: The arithmetic mean (simple average) of the numbers in values. In the GuacaMol context this return value typically represents an aggregated benchmark quantity (for example, mean score across generated molecules) and is used downstream for reporting, comparison, and ranking of generative models.
    
    Raises:
        ZeroDivisionError: If values is an empty list, division by zero will occur because the implementation divides by len(values).
        Exception: If elements of values cannot be summed (for example, because they are of incompatible types), the underlying sum(...) operation may raise a TypeError or another exception; callers should ensure elements conform to List[float] before calling.
    """
    from guacamol.utils.math import arithmetic_mean
    return arithmetic_mean(values)


################################################################################
# Source: guacamol.utils.math.geometric_mean
# File: guacamol/utils/math.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for geometric_mean because the docstring has no description for the argument 'values'
################################################################################

def guacamol_utils_math_geometric_mean(values: List[float]):
    """Compute the geometric mean of a list of numeric values.
    
    This function is a small numeric utility used in the GuacaMol benchmarking codebase to aggregate multiplicative scores (for example, combining per-objective or per-component scores in goal-directed benchmark scoring). It converts the input sequence to a NumPy array, computes the product of all elements, and returns the n-th root of that product (product ** (1/len(values))). The operation is commutative with respect to the input order and has no side effects on external state.
    
    Args:
        values (List[float]): A Python list of floating-point numbers representing the values to be aggregated by geometric mean. In the GuacaMol context, these are typically scalar scores for a molecule or component scores that should be combined multiplicatively. Each element is treated as a factor in the product; the function performs no validation or clipping of elements before aggregation.
    
    Behavior, defaults, and failure modes:
        The implementation converts values to a NumPy array, computes a.prod() to obtain the product, and then raises that product to the power 1.0 / len(values) to obtain the geometric mean. There are no side effects and no mutation of the input list.
        - If values is empty (len(values) == 0), a division by zero occurs when computing 1.0 / len(values), which will raise a ZeroDivisionError; callers must ensure the list is non-empty.
        - If any element is zero, the product is zero and the geometric mean is zero.
        - If the list contains negative numbers and its length is greater than 1, raising a negative product to a fractional power may produce a complex-valued NumPy scalar or propagate NaNs or runtime warnings. For a guaranteed real-valued result, provide non-negative inputs (typically positive scores in the GuacaMol scoring context).
        - For long lists or values with large magnitude, the intermediate product may overflow or underflow, causing inf/0 results or loss of precision; use numerically stable alternatives (e.g., computing the mean of logs) if this is a concern.
        - The function performs no input type checking beyond the conversion to a NumPy array; supplying non-numeric items will raise appropriate TypeError/ValueError from NumPy operations.
    
    Returns:
        float: The geometric mean computed as (product of values) ** (1.0 / n) where n is the number of entries in values. In the common GuacaMol use case with strictly non-negative floating-point scores, this is a Python float giving the combined multiplicative score for a molecule or set of components. If inputs lead to complex-valued results (see failure modes), a NumPy complex scalar may be produced at runtime despite the float annotation.
    """
    from guacamol.utils.math import geometric_mean
    return geometric_mean(values)


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
