"""
Regenerated Google-style docstrings for module 'drugbank_downloader'.
README source: others/readme/drugbank_downloader/README.md
Generated at: 2025-12-02T00:42:45.536992Z

Total functions: 1
"""


################################################################################
# Source: drugbank_downloader.version.get_version
# File: drugbank_downloader/version.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for get_version because the docstring has no description for the argument 'with_git_hash'
################################################################################

def drugbank_downloader_version_get_version(with_git_hash: bool = False):
    """drugbank_downloader.version.get_version returns the current drugbank_downloader package version string and, optionally, a git commit hash appended to that version to provide more precise provenance.
    
    Args:
        with_git_hash (bool): If True, append the current git commit hash to the package VERSION constant, separated by a single hyphen, producing a string of the form "VERSION-GITHASH" (for example, "5.1.7-abcdef1"). If False (the default), return the plain VERSION string (for example, "5.1.7"). This flag is useful when you need an exact identifier for reproducible workflows, release artifacts, or to tag downloads and cache directories (the README shows the VERSION being used to name storage paths such as "~/.data/drugbank/5.1.7/..."). The function obtains the base version from the module-level VERSION constant and the git hash by calling get_git_hash(); callers should be aware that the behavior when git metadata is missing depends on get_git_hash(): it may return an empty string (resulting in "VERSION-") or raise an exception, which will propagate to the caller.
    
    Returns:
        str: The version identifier. When with_git_hash is False, this is exactly the module VERSION constant (used throughout the package for release and storage naming). When with_git_hash is True, this is the formatted string f"{VERSION}-{get_git_hash()}" as produced by the implementation. The function performs no network I/O and has no persistent side effects; it is intended only to report version information for logging, cache-keying, packaging, and reproducibility purposes.
    """
    from drugbank_downloader.version import get_version
    return get_version(with_git_hash)


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
