"""
Regenerated Google-style docstrings for module 'pubchempy'.
README source: others/readme/pubchempy/README.md
Generated at: 2025-12-02T00:43:35.871770Z

Total functions: 2
"""


################################################################################
# Source: pubchempy.deprecated
# File: pubchempy.py
# Category: fix_docstring
# Reason: Schema parsing failed: ("Couldn't parse this type hint, likely due to a custom class or object: ", typing.Callable[[typing.Callable], typing.Callable])
################################################################################

def pubchempy_deprecated(message: str):
    """pubchempy.deprecated: Factory for a decorator that marks a function as deprecated in PubChemPy and emits a runtime warning when the deprecated function is called. This is intended for use by PubChemPy maintainers and contributors to signal that a public API (for example a function that performs PubChem lookups, conversions, or property retrievals) is obsolete and that callers should migrate to an alternative. The returned decorator wraps the original function, preserves its metadata, issues a PubChemPyDeprecationWarning via warnings.warn with stacklevel=2 at each call, and then invokes the original function so normal behavior and return values are preserved.
    
    Args:
        message (str): A human-readable deprecation message describing why the function is deprecated and, optionally, what to use instead. This string is incorporated into the warning text via f"{func.__name__} is deprecated: {message}" when a decorated function is called. The value is expected to be a str (or convertible to str) and should be concise but informative so users of the PubChemPy library can understand the recommended migration or the reason for deprecation.
    
    Returns:
        Callable[[Callable], Callable]: A decorator that accepts a single callable (the function to deprecate) and returns a wrapped callable. The wrapped callable:
            - Emits a PubChemPyDeprecationWarning using warnings.warn with category=PubChemPyDeprecationWarning and stacklevel=2, so the warning points to the caller site in user code.
            - Preserves the wrapped function's metadata (such as __name__, __doc__, and module) using functools.wraps.
            - Forwards all positional and keyword arguments to the original function and returns the original function's return value unchanged.
            - Does not suppress exceptions raised by the original function; such exceptions propagate to the caller.
            - May not result in visible output if Python's warnings are filtered or redirected; in that case the side effect is still the warning emission but it will not be shown to the user.
    """
    from pubchempy import deprecated
    return deprecated(message)


################################################################################
# Source: pubchempy.get_all_sources
# File: pubchempy.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for get_all_sources because the docstring has no description for the argument 'domain'
################################################################################

def pubchempy_get_all_sources(domain: str = "substance"):
    """Return a list of all current depositors (source names) for the specified PubChem domain (for example, substances or assays). This function is a thin helper used by PubChemPy to query the PubChem REST API "sources" endpoint for metadata about who submitted records; it is useful in workflows that need to audit, filter, or cross-reference contributors to PubChem datasets.
    
    Args:
        domain (str): The PubChem data domain to query for depositors. In practice this identifies which set of depositors to return (the original implementation uses this to request either substance-related depositors or assay-related depositors). The default value is "substance", which returns depositors of substance records. The value is passed verbatim to the internal PubChem API helper and is used to construct the "sources" endpoint request; do not rely on automatic validation of domain values by this function.
    
    Returns:
        list[str]: A list of depositor names (strings) as returned by the PubChem "sources" endpoint. Concretely, the function decodes the bytes response with the default text encoding (UTF-8), parses the JSON payload, and returns results["InformationList"]["SourceName"], so the returned list contains the source names in the same order provided by the API.
    
    Behavior, side effects, defaults, and failure modes:
    - The function performs a network request to PubChem via the package's internal get(...) helper to fetch the "sources" endpoint for the given domain; this is a live HTTP call and has the usual network side effects (latency, possible transient failures).
    - The default domain is "substance". If another domain is required (for example, to list assay depositors), pass the appropriate domain string; this function does not validate or restrict domain values beyond forwarding them to the API.
    - The implementation decodes the raw bytes response using bytes.decode() (UTF-8 by default) and parses it with json.loads(); it then extracts the list of names at results["InformationList"]["SourceName"].
    - Errors raised by the underlying HTTP helper (network errors, HTTP error handling behavior implemented by get(...)), json.JSONDecodeError if the response is not valid JSON, UnicodeDecodeError if decoding fails, or KeyError if the expected JSON keys are absent will propagate to the caller. Callers should catch these exceptions when using this function in production code.
    - There is no caching or persistence performed by this function; it always issues a fresh request to PubChem when called.
    """
    from pubchempy import get_all_sources
    return get_all_sources(domain)


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
