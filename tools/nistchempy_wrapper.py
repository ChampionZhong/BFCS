"""
Regenerated Google-style docstrings for module 'nistchempy'.
README source: others/readme/nistchempy/README.md
Generated at: 2025-12-02T00:42:43.042946Z

Total functions: 1
"""


################################################################################
# Source: nistchempy.requests.fix_html
# File: nistchempy/requests.py
# Category: valid
################################################################################

def nistchempy_requests_fix_html(html: str):
    """Fixes common, known typos in HTML source returned by the NIST Chemistry WebBook and returns a cleaned HTML string for use by NistChemPy parsing routines.
    
    Args:
        html (str): Raw HTML text of a NIST Chemistry WebBook page. In the NistChemPy workflow this is the HTML content downloaded from the WebBook for a compound or search result. The function expects a Python str containing the full HTML document; it is not designed to accept bytes or file-like objects. Practical significance: passing the raw HTML produced by the WebBook to this function corrects small, repeatable defects that otherwise cause downstream HTML/XML parsers or CSS-based extractors in NistChemPy to miss elements (for example, mis-typed attributes or non-breaking spaces that break tokenization).
    
    Returns:
        str: A new string containing the corrected HTML. Specifically, the function performs two deterministic textual substitutions: all occurrences of the substring "clss=" are replaced with the correct attribute "class=", and all non-breaking space characters (U+00A0, represented as '\xa0') are replaced with ordinary space characters (U+0020). The original input string is not modified (strings are immutable); the returned value is safe to pass to NistChemPy routines that extract compound properties, coordinates, or spectra.
    
    Behavior and side effects:
        The function is a pure, deterministic text transformer with no external side effects (it does not perform I/O, network requests, or in-place mutation). It is idempotent: applying it multiple times yields the same result as applying it once. It creates and returns a new str object; callers should be aware of the temporary memory copy for very large HTML documents.
    
    Failure modes and notes for callers:
        The function assumes a valid Python str as input. If a non-str object is passed, a runtime exception will occur when attempting to call str.replace (for example, AttributeError); callers should ensure they pass decoded text rather than bytes. The function only performs the two substitutions described above and will not correct other HTML errors; more extensive cleaning or robust HTML parsing should be performed by the caller if needed. Performance cost is proportional to the length of the input string due to the underlying replacements.
    """
    from nistchempy.requests import fix_html
    return fix_html(html)


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
