"""
Regenerated Google-style docstrings for module 'descriptastorus'.
README source: others/readme/descriptastorus/README.md
Generated at: 2025-12-02T00:48:14.402709Z

Total functions: 3
"""


################################################################################
# Source: descriptastorus.MolFileIndex.MakeSDFIndex
# File: descriptastorus/MolFileIndex.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def descriptastorus_MolFileIndex_MakeSDFIndex(filename: str, dbdir: str):
    """descriptastorus.MolFileIndex.MakeSDFIndex creates an on-disk index for an SDF (Structure-Data File) so that individual molecules in the SDF can be accessed randomly by row number. This function is used by the DescriptaStorus project to provide fast random access to indexed molecule files (see README: "fast random access to indexed molecule files"). The index maps sequential row numbers to byte offsets in the SDF file and stores those offsets in a raw DescriptaStorus store located at the provided dbdir. The returned MolFileIndex object uses SDFNameGetter as the name extraction function so the index can be used with the rest of the DescriptaStorus API (for example, to call getMol, getName, or to iterate descriptors).
    
    Args:
        filename (str): Path to the input SDF file to index. This file must be readable by the running process. The function determines the file size (via os.path.getsize), counts the number of molecules using simplecount(filename), and locates the per-molecule byte positions by scanning for the SDF record separator b"$$$$\n" (via index(filename, b"$$$$\n")). The SDF file must use "$$$$\n" as the molecule delimiter for the index to be correct.
        dbdir (str): Directory path where the on-disk raw store (index) will be created. The function calls raw.MakeStore([("index", dtype)], N+1, dbdir) to create a raw store with a single column named "index" and N+1 rows (N is the molecule count). Files will be created under this directory as a side effect; the caller must have write permission to create or modify files in dbdir.
    
    Behavior and side effects:
        The function inspects the size of filename and selects a NumPy unsigned integer dtype to store byte offsets: numpy.uint8 if size < 2**8, numpy.uint16 if size < 2**16, numpy.uint32 if size < 2**32, otherwise numpy.uint64. This dtype selection ensures the index column can hold file byte offsets for the file size observed.
        The molecule count N is computed with simplecount(filename). The list of molecule byte offsets is produced by list(index(filename, b"$$$$\n")), where each element gives a byte position (0-based) at which a molecule record ends; these positions are used to compute start offsets for random access.
        The function creates a raw store in dbdir with N+1 rows. It writes a first sentinel row with value 0 (db.putRow(0, [0])), then for each found index position it writes a row i+1 containing pos+1 (db.putRow(i+1, [pos+1])). Storing pos+1 produces a non-zero stored offset for molecules beginning at the start of the file; callers should be aware the stored offsets are offset by +1 relative to the original 0-based byte positions.
        The function returns a MolFileIndex object constructed as MolFileIndex(filename, dbdir, nameFxn=SDFNameGetter). This object provides the standard MolFileIndex API described in the README for random-access retrieval of molecule strings and names.
    
    Failure modes and important notes:
        If filename does not exist or is not readable, os.path.getsize(filename) or the subsequent scanning/counting will raise an OSError (or the underlying functions may raise other I/O-related exceptions).
        If dbdir is not writable or raw.MakeStore cannot create the storage there, an error will be raised by raw.MakeStore; this function does not suppress such errors.
        The correctness of the index depends on detecting the SDF record separator b"$$$$\n"; if the SDF file uses a different line ending convention or a different separator, the index will be incorrect.
        The function relies on the helper functions simplecount and index being available and operating correctly on the provided filename.
        The function will create files under dbdir as a side effect. Behavior if dbdir already contains different or conflicting store files depends on raw.MakeStore and raw.MakeStore's handling of existing directories (the caller should consult raw.MakeStore semantics before calling).
    
    Returns:
        MolFileIndex: A MolFileIndex instance bound to the input filename and the created index in dbdir, using SDFNameGetter to extract names. The MolFileIndex supports random access methods (for example get, getMol, getName) so callers can retrieve the molecule string and associated metadata by row number. The returned object is the primary programmatic result; the persistent index data are stored on disk in dbdir as described above.
    """
    from descriptastorus.MolFileIndex import MakeSDFIndex
    return MakeSDFIndex(filename, dbdir)


################################################################################
# Source: descriptastorus.descriptors.DescriptorGenerator.MakeGenerator
# File: descriptastorus/descriptors/DescriptorGenerator.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def descriptastorus_descriptors_DescriptorGenerator_MakeGenerator(generator_names: list):
    """descriptastorus.descriptors.DescriptorGenerator.MakeGenerator creates a combined descriptor generator by looking up one or more named descriptor generator factories in the DescriptorGenerator.REGISTRY and returning either a single generator or a Container that composes multiple generators. This function is used by consumers of the descriptastorus library (for example, code that needs RDKit2D, Morgan3Counts, or combinations thereof) to obtain a callable descriptor generator that will produce the feature vectors described in the README (the generator.process(smiles) convention where the first element is a boolean success flag and the remaining elements are the descriptor values).
    
    Args:
        generator_names (list): An ordered list of descriptor generator names to include in the created generator. Each entry should be a string matching a key in DescriptorGenerator.REGISTRY (the lookup is performed case-insensitively via name.lower()). Typical practical values are names documented in the README such as "RDKit2D", "Morgan3Counts", "morgan3counts", etc. The order in this list determines the order in which the underlying generators will be combined when multiple generators are requested.
    
    Returns:
        DescriptorGenerator: If a single name is provided and found in the registry, the corresponding DescriptorGenerator instance (or factory object) referenced by DescriptorGenerator.REGISTRY[name.lower()] is returned directly. If multiple names are provided and resolved, a Container instance that composes the resolved generators is returned; this Container produces combined descriptor output (maintaining the per-generator success flag and concatenated feature vectors as described in README usage examples). The returned object implements the descriptor generator interface used elsewhere in descriptastorus (for example, .process(smiles) and .GetColumns()).
    
    Raises:
        ValueError: If generator_names is empty, a ValueError is raised after logging a warning. This enforces that callers must request at least one generator.
        KeyError: If any requested name is not present in DescriptorGenerator.REGISTRY, the dictionary lookup will raise KeyError; the function logs the exception and the set of currently registered descriptor keys before re-raising the original exception. Note that the implementation catches all exceptions during lookup and re-raises them, so callers should be prepared to receive the original exception type (commonly KeyError for unknown names).
        Exception: Any exception raised while resolving a named generator (for example, if a registry entry is present but its construction raises) will be logged with logger.exception (including a message showing the requested name and the sorted registry keys) and then re-raised to the caller.
    
    Behavior and side effects:
        The function does not modify DescriptorGenerator.REGISTRY; it only reads from it. Lookups are performed case-insensitively by calling name.lower() on each provided name. If multiple generators are resolved, a Container wrapper (as implemented in the descriptors package) is returned to combine their outputs. The function logs a warning if called with an empty list and logs full exception and registry information when a lookup fails; callers should consult the library logger output for diagnostics. The function preserves the order of generator_names when composing multiple generators, which determines the order of columns in the combined descriptor output.
    """
    from descriptastorus.descriptors.DescriptorGenerator import MakeGenerator
    return MakeGenerator(generator_names)


################################################################################
# Source: descriptastorus.descriptors.QED.ads
# File: descriptastorus/descriptors/QED.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def descriptastorus_descriptors_QED_ads(
    x: float,
    a: float,
    b: float,
    c: float,
    d: float,
    e: float,
    f: float,
    dmax: float
):
    """descriptastorus.descriptors.QED.ads computes an asymmetric double-sigmoid (ADS) mathematical transform used in descriptor generation to convert a single raw scalar property into a scaled contribution value for descriptor scoring and storage in DescriptaStorus.
    
    This function implements the ADS expression used by descriptor modules (for example, QED-style component scoring) to produce a smoothly varying, bounded-like contribution from an input property. In the DescriptaStorus project this kind of transform is applied when building descriptor rows for storage and subsequent machine learning tasks: it maps a raw property value x through two logistic (sigmoid) components combined with amplitude and offset parameters, and then normalizes by dmax to produce the final scalar that is stored in descriptor arrays.
    
    Args:
        x (float): The raw scalar property value to be transformed. In the descriptor generation workflow this is a measured or calculated molecular property (for example a physicochemical value) supplied to the ADS transform. This value is used as the independent variable in the two sigmoid terms of the formula.
        a (float): Baseline offset added to the ADS numerator. Practically, this shifts the transformed contribution up or down before normalization and is part of the per-component scoring term used by descriptor generators.
        b (float): Amplitude or scale factor that modulates the magnitude of the sigmoid-derived contribution in the numerator. In descriptor construction this controls how strongly the sigmoid product contributes relative to the baseline a.
        c (float): Center parameter that locates the central position of the sigmoid features on the x axis. Descriptor generators use c to align the ADS response to the property value ranges that are considered optimal or reference points.
        d (float): Width/offset that separates the two internal sigmoid centers by ±d/2 in the formula. In practice d sets the separation between the two logistic transitions that form the ADS shape and therefore controls the breadth of the response region.
        e (float): Slope/scaling parameter for the first sigmoid (the term with (x - c + d/2)). This float controls the steepness of the first logistic transition; small magnitudes yield sharper transitions. In descriptor pipelines e is chosen to tune sensitivity on the left side of the ADS shape.
        f (float): Slope/scaling parameter for the second sigmoid (the term with (x - c - d/2)). Analogous to e, this float controls the steepness of the second logistic transition and tunes sensitivity on the right side of the ADS shape.
        dmax (float): Final normalization divisor applied to the entire numerator. In the DescriptaStorus descriptor workflow dmax is used to scale the ADS output into the desired numerical range for storage and machine learning. Because the function divides by dmax, it must be non-zero to avoid a division-by-zero error.
    
    Behavior and implementation details:
        The function computes the numerator exactly as in the source expression:
            numerator = a + (b / (1 + exp(-1 * (x - c + d / 2) / e)) * (1 - 1 / (1 + exp(-1 * (x - c - d / 2) / f)))))
        and returns numerator divided by dmax.
        The implementation uses the exponential function exp in the two logistic denominators. For inputs where the exponent arguments are very large in magnitude, the underlying exp call may overflow or underflow according to the Python math exponential behavior; callers should choose e and f to avoid excessively large exponent values for the expected x and c ranges.
        There are no side effects: the function performs a pure numerical computation and does not modify external state or data structures.
        Default values: the function signature requires all eight parameters to be provided; there are no implicit defaults inside the function.
        Failure modes: if e or f is zero, the expressions (x - c ± d/2) / e or / f will raise a ZeroDivisionError. If dmax is zero, the final division will raise a ZeroDivisionError. If non-numeric types are supplied (not convertible to float), a TypeError or other numeric conversion error may be raised by the arithmetic operations or by exp. Extremely large exponent magnitudes may raise an OverflowError from the exp implementation.
    
    Returns:
        float: The ADS-transformed scalar value computed as the expression above divided by dmax. In the DescriptaStorus descriptor context this returned float is the per-component contribution that will be combined or stored as part of a descriptor row for machine learning and indexing.
    """
    from descriptastorus.descriptors.QED import ads
    return ads(x, a, b, c, d, e, f, dmax)


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
