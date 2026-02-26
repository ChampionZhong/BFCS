"""
Regenerated Google-style docstrings for module 'aizynthfinder'.
README source: others/readme/aizynthfinder/README.md
Generated at: 2025-12-02T00:44:02.916529Z

Total functions: 3
"""


import numpy

################################################################################
# Source: aizynthfinder.utils.math.rectified_linear_unit
# File: aizynthfinder/utils/math.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for rectified_linear_unit because the docstring has no description for the argument 'x'
################################################################################

def aizynthfinder_utils_math_rectified_linear_unit(x: numpy.ndarray):
    """aizynthfinder.utils.math.rectified_linear_unit returns the element-wise Rectified Linear Unit (ReLU) activation of a NumPy array. In the AiZynthFinder codebase this function is used to introduce a simple, computationally efficient non-linearity in neural network computations (for example in expansion policy or filter policy networks that guide the retrosynthetic Monte Carlo tree search). The function maps each input element to itself when it is greater than zero and to zero otherwise, producing a non-negative array that can induce sparsity and improve training/stability of downstream policy networks.
    
    Args:
        x (numpy.ndarray): Input numeric array containing the pre-activation values (for example, outputs from a linear layer of a neural network used by AiZynthFinder policies). The function applies the ReLU activation element-wise. The input shape is preserved and the operation is vectorized over all elements. The array must support comparison with zero (x > 0); passing an array with object dtype or elements that cannot be compared to 0 may raise a TypeError.
    
    Returns:
        numpy.ndarray: A new NumPy array with the same shape as x containing the element-wise ReLU activations. Each element in the returned array is equal to the original element when that element is strictly greater than zero, and equal to zero otherwise. The returned array is produced by the expression x * (x > 0) and therefore for typical numeric dtypes the numeric dtype and shape of x are preserved. The function does not modify the input array in-place. Note that special values follow NumPy comparison semantics (for example, NaN > 0 is False, so NaN entries become 0 in the output).
    """
    from aizynthfinder.utils.math import rectified_linear_unit
    return rectified_linear_unit(x)


################################################################################
# Source: aizynthfinder.utils.math.sigmoid
# File: aizynthfinder/utils/math.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for sigmoid because the docstring has no description for the argument 'x'
################################################################################

def aizynthfinder_utils_math_sigmoid(x: numpy.ndarray):
    """aizynthfinder.utils.math.sigmoid computes the logistic (sigmoid) activation function used to squash real-valued scores into a bounded range for downstream decision making in AiZynthFinder's neural-network-driven policy components.
    
    This function implements the element-wise logistic sigmoid 1 / (1 + exp(-x)). In the AiZynthFinder retrosynthetic planning workflow (see README), policy networks produce raw scores for candidate reaction templates; applying this sigmoid converts those raw scores into normalized confidence-like values that can be used to rank or weight suggestions during the Monte Carlo tree search.
    
    Args:
        x (numpy.ndarray): Input array of real-valued scores. In practice this is typically the raw output from a neural network policy used by AiZynthFinder to propose precursors. The function expects a numpy.ndarray and applies the logistic sigmoid element-wise to every element of x. Passing a value that is not a numpy.ndarray is not supported and may raise an exception.
    
    Returns:
        numpy.ndarray: A numpy.ndarray containing the sigmoid-transformed values computed as 1 / (1 + exp(-x_i)) for each element x_i in the input. These outputs are floating-point values that serve as normalized scores or probabilities for policy decisions in the retrosynthetic planning algorithm. The function is purely computational with no side effects.
    
    Behavior, defaults, and failure modes:
        The computation is vectorized via numpy and operates element-wise. For very large magnitude inputs, numpy.exp may overflow or underflow leading to runtime warnings and extreme outputs (values approaching 0.0 or 1.0); such numerical issues stem from the underlying exponential and are not caught or altered by this function. The function does not modify its input in place.
    """
    from aizynthfinder.utils.math import sigmoid
    return sigmoid(x)


################################################################################
# Source: aizynthfinder.utils.math.softmax
# File: aizynthfinder/utils/math.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for softmax because the docstring has no description for the argument 'x'
################################################################################

def aizynthfinder_utils_math_softmax(x: numpy.ndarray):
    """Compute column-wise softmax of the input scores and return normalized probabilities.
    
    This function converts raw scores (logits) into non-negative values normalized per column by applying the exponential function and dividing by the column-wise sum. In the AiZynthFinder retrosynthetic planning workflow, it is typically used to transform the output scores from an expansion policy neural network into probabilities that guide the Monte Carlo tree search when selecting precursor reactions. The implementation performs the operation exp(x) / sum(exp(x), axis=0) using NumPy without additional numerical stabilization.
    
    Args:
        x (numpy.ndarray): Array of numeric scores (logits). Each column (summation along axis 0) is treated as an independent set of scores to be normalized into a probability distribution. In practice within AiZynthFinder, columns can represent scores for different candidate reactions or actions produced by a policy network for one or more input molecules. The function accepts any ndarray shape but always normalizes along axis 0. The array should contain finite numeric values for stable, meaningful output.
    
    Returns:
        numpy.ndarray: Array of the same shape as x containing the column-wise softmax values computed as exp(x) / sum(exp(x), axis=0). For each column, the returned values are non-negative and will sum to 1 when the exponentials are finite and the column-wise sum is non-zero; these values are intended to be interpreted as probabilities over the corresponding set of scores (e.g., probabilities of selecting particular precursor suggestions in retrosynthetic planning). Note that if x contains very large magnitude values, NaNs, or infinities, or if a column's exponential values sum to zero, numerical issues (overflow, division by zero, or NaNs) can occur because this implementation does not perform numerical stabilization (such as subtracting the column maximum before exponentiation). The function has no side effects.
    """
    from aizynthfinder.utils.math import softmax
    return softmax(x)


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
