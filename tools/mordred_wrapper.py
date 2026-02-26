"""
Regenerated Google-style docstrings for module 'mordred'.
README source: others/readme/mordred/README.rst
Generated at: 2025-12-02T00:44:12.616627Z

Total functions: 1
"""


################################################################################
# Source: mordred._util.to_ordinal
# File: mordred/_util.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def mordred__util_to_ordinal(n: int):
    """Convert an integer to a short English ordinal string used for human-readable labels.
    
    This utility function in mordred._util is a small pure helper used by the mordred molecular descriptor calculator to produce human-friendly ordinal labels (for example when formatting descriptor positions, log messages, CLI output, or report fields). It maps the integers 1, 2 and 3 to the English words "first", "second" and "third" respectively; for any other integer it returns a numeric ordinal using the pattern "<n>-th" (for example "4-th" or "104-th"). The function has no side effects and does not modify external state.
    
    Args:
        n (int): The integer to convert to an ordinal string. This parameter is the positional index or rank that will be rendered as an English ordinal for presentation purposes (e.g., descriptor number or step index). The function expects an int as specified by the signature; passing non-int values is not supported by the documented API and may produce unintended results.
    
    Returns:
        str: A short English ordinal representation of n. If n == 1 the return value is "first"; if n == 2 it is "second"; if n == 3 it is "third"; otherwise the return value is the string "{}-th".format(n), i.e. the decimal representation of n followed by "-th". No exceptions are raised by the implementation for integer inputs; incorrect types or unexpected inputs are outside the documented contract and may lead to undefined or surprising output.
    """
    from mordred._util import to_ordinal
    return to_ordinal(n)


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
