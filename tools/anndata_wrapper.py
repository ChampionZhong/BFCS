"""
Regenerated Google-style docstrings for module 'anndata'.
README source: others/readme/anndata/README.md
Generated at: 2025-12-02T00:22:33.759282Z

Total functions: 8
"""


################################################################################
# Source: anndata._core.merge.default_fill_value
# File: anndata/_core/merge.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def anndata__core_merge_default_fill_value(els: list):
    """Given some arrays, return the default fill value that should be used when
    merging them for anndata operations.
    
    This function exists for backwards compatibility in anndata's merging logic for
    annotated data matrices (used for single-cell and other omics data). It inspects
    the provided elements to decide whether the merged result should use a sparse-
    friendly fill value (0) or a dense/missing-value sentinel (numpy.nan). In
    practice this affects how missing entries are represented when combining layers,
    observations, variables, or other array-like components inside an AnnData object,
    and therefore influences memory use and downstream numeric semantics (e.g.,
    treating missing entries as zeros for sparse data versus NaN for dense data).
    
    Args:
        els (list): A list of array-like objects to inspect. Each element is tested
            in order to determine whether it should be treated as sparse-like. The
            code checks whether an element is an instance of the internal sparse
            types CSMatrix or CSArray, or whether it is a DaskArray whose _meta
            attribute is an instance of CSMatrix or CSArray. Typical callers pass
            a list of arrays or array wrappers drawn from AnnData components (for
            example, data layers, X, obsm/varm entries) when deciding a merge fill
            value.
    
    Behavior and notable details:
        - If any element in els is a CSMatrix or CSArray, the function returns 0.
          This choice aligns with sparse-matrix conventions used in anndata and
          Scanpy, where absent entries are represented by zeros to preserve sparsity
          and memory efficiency.
        - If any element is a DaskArray and its _meta attribute is a CSMatrix or
          CSArray, the function also returns 0. This covers lazy/dask-backed sparse
          arrays used by anndata's lazy operations.
        - If none of the elements match the sparse-like checks above, the function
          returns numpy.nan, which is a conventional missing-value sentinel for
          dense numeric arrays.
        - The function uses isinstance checks and therefore treats subclasses of the
          checked types according to normal Python isinstance semantics.
        - There are no side effects: the function is pure and does not modify the
          input elements or global state. It only inspects types and attributes.
        - The implementation exists for backwards compatibility and "might not be
          the ideal solution" for all datasets; callers that require explicit fill
          values for domain-specific reasons should set them directly rather than
          relying solely on this heuristic.
    
    Failure modes and edge cases:
        - If an element is an instance of the DaskArray type but lacks an expected
          _meta attribute, attribute access may raise AttributeError; callers should
          ensure DaskArray-like objects used here conform to the expected interface.
        - The function does not validate array contents or shapes; it only inspects
          types and the _meta attribute for DaskArray, so it will not detect cases
          where an array is logically sparse but not represented by the checked
          types.
        - The function does not attempt to coerce or convert input types; unexpected
          or custom array wrappers that are not instances of the handled types will
          be treated as non-sparse and cause numpy.nan to be returned.
    
    Returns:
        int or float: Returns 0 when any element is detected as sparse-like
        (CSMatrix or CSArray) or is a DaskArray whose _meta is one of those sparse
        types; otherwise returns numpy.nan. The return value is intended to be used
        as the default fill value for merge operations in anndata, influencing how
        missing entries are represented (zero for sparse-backed data, NaN for dense
        data).
    """
    from anndata._core.merge import default_fill_value
    return default_fill_value(els)


################################################################################
# Source: anndata._io.utils.check_key
# File: anndata/_io/utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def anndata__io_utils_check_key(key: str):
    """anndata._io.utils.check_key validates and normalizes a candidate HDF5 key used by anndata's I/O routines.
    
    Checks that the provided key is a valid h5py key for use as a group or dataset name in on-disk AnnData (HDF5) storage. In the anndata codebase this function is used by the file I/O pipeline when creating or accessing HDF5 groups and datasets so that keys passed into h5py are guaranteed to be Python built-in str objects. The function accepts values that are instances of str or of subclasses of str and returns a built-in str; it does not perform any other conversions (for example, it does not decode bytes) and has no side effects.
    
    Args:
        key (str): Candidate key for an h5py group or dataset name. In the anndata HDF5 I/O domain this represents the name under which a piece of annotated data (for example, a dataset or subgroup inside an AnnData file) will be stored or accessed. The function expects a value that is already a Python str or a subclass of str; if a subclass is provided (for example, a custom string subclass used elsewhere in the codebase), the function will convert it to a built-in str via str(key) so downstream h5py calls receive a plain str. Although h5py can accept bytes by decoding them to str, this function intentionally does not accept or decode bytes (see the commented TODO in the source).
    
    Returns:
        str: The validated key as a built-in Python str suitable for use with h5py group/dataset names in anndata's on-disk representation. The return value preserves the textual content of the input but guarantees the exact type is built-in str, which prevents type-related errors when interacting with h5py.
    
    Raises:
        TypeError: If key is not an instance or subclass of str. The raised exception message matches the implementation: "{key} of type {typ} is an invalid key. Should be str." This signals to callers of the anndata I/O functions that the provided key must be a string and that no implicit binary-to-text decoding or other coercion was performed.
    """
    from anndata._io.utils import check_key
    return check_key(key)


################################################################################
# Source: anndata._io.utils.convert_bool
# File: anndata/_io/utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def anndata__io_utils_convert_bool(string: str):
    """anndata._io.utils.convert_bool: Determine whether a text token represents a boolean literal and return a normalized boolean indicator and value.
    
    This helper is used in the anndata I/O utilities when parsing textual metadata or serialized fields (for example, when reading annotation columns from CSV or other plain-text representations of AnnData objects used in single-cell workflows). The function implements a strict, deterministic conversion: it recognizes only the exact, case-sensitive string literals "True" and "False" and maps them to Python boolean values. Callers (code that constructs or populates AnnData.obs / AnnData.var or other metadata fields from text) should use the first element of the returned tuple to decide whether the input was recognized as a boolean literal before trusting the second element as the boolean value.
    
    Args:
        string (str): The input text token to evaluate. In the anndata I/O context this is typically a value read from a textual metadata column or serialized attribute. The function performs exact, case-sensitive equality checks against the two accepted literal values "True" and "False". The parameter is annotated as str and the intended use is with string values produced by file parsing or serialization routines.
    
    Returns:
        tuple[bool, bool]: A pair (is_boolean_literal, boolean_value).
            is_boolean_literal is True if and only if the input string exactly matched "True" or "False".
            boolean_value is the corresponding Python bool: True when string == "True", False when string == "False".
            If the input string is not one of the two accepted literals, the function returns (False, False) to indicate that the token was not recognized as a boolean literal; callers should therefore check is_boolean_literal before using boolean_value.
    
    Behavior and failure modes:
        The function is pure and has no side effects. It only performs equality comparisons and will not raise an exception for ordinary string inputs; non-matching strings simply yield (False, False). The conversion is strict and case-sensitive: alternative representations such as "true", "FALSE", "1", "0", "T", or "F" are not recognized and will result in (False, False). This strictness ensures deterministic parsing of metadata during AnnData read/write operations.
    """
    from anndata._io.utils import convert_bool
    return convert_bool(string)


################################################################################
# Source: anndata._io.utils.convert_string
# File: anndata/_io/utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def anndata__io_utils_convert_string(string: str):
    """anndata._io.utils.convert_string converts a textual token (Python str) into an appropriate Python scalar type used by anndata I/O utilities: it will attempt to interpret the input as an int, then a float, then a boolean, then the Python None singleton, and if none of those match it returns the original string. This function is used when parsing annotation/metadata fields and other text-based values read by anndata (for example when reading from text-based files or serialisations) so that numeric and boolean values become native Python types suitable for numeric computation, indexing, and logical masking rather than remaining untyped strings.
    
    Args:
        string (str): The input text token to convert. This parameter is the raw textual representation of a single value obtained from I/O operations (CSV, TSV, plain-text fields, etc.) within the anndata I/O utilities. The function expects a Python str and will attempt to parse it to a more specific Python scalar type following the precedence described below. Passing non-str types is outside the intended use and may lead to undefined behavior or exceptions depending on the helper routines used internally.
    
    Behavior and precedence:
        The conversion is performed in a fixed order to avoid ambiguous interpretations: first is_int(string) is evaluated and, if true, the function returns int(string). If not, is_float(string) is evaluated and, if true, the function returns float(string). If neither numeric check succeeds, the function calls convert_bool(string) (a helper that follows the project's boolean parsing conventions) and inspects its boolean indicator; if that indicates a valid boolean representation, the function returns the parsed bool value. If the literal string "None" is encountered (exact match), the function returns the Python None singleton. If none of these checks match, the original input string is returned unchanged. This deterministic precedence ensures, for example, that integer-looking tokens are not accidentally parsed as floats or booleans.
    
    Side effects and defaults:
        The function has no side effects: it does not modify its input, global state, files, or external resources. It performs pure computation and returns a new Python object (one of int, float, bool, None, or str). There are no configurable defaults for parsing behavior in this function; it relies on the project's helper predicates and parsers (is_int, is_float, convert_bool) for the precise syntactic rules used to recognise ints, floats, and booleans.
    
    Failure modes and error handling:
        The function is designed to avoid raising exceptions for ordinary string inputs by returning the original string when no conversion applies. However, it expects a str; supplying other types may produce errors depending on the helper functions' implementations (for example, TypeError or ValueError raised by those helpers). The function does not perform locale-aware or context-dependent parsing beyond what the helper routines implement.
    
    Returns:
        int, float, bool, None, or str: The converted Python value. Returned type is one of:
            - int: when the input is recognised by is_int and converted via int(string); useful for integer-valued annotation fields that will be used in numeric computations or indexing.
            - float: when the input is recognised by is_float and converted via float(string); useful for continuous-valued annotations.
            - bool: when convert_bool recognises the input as a boolean according to the project's conventions; useful for logical masks and filtering.
            - None: when the input is the exact string "None", returning the Python None singleton to represent missing or null values.
            - str: when none of the above applies, the original string is returned unchanged so that arbitrary textual annotations are preserved.
    
    Practical significance in the anndata domain:
        Using this function during I/O ensures that metadata and annotation fields imported into an AnnData object have appropriate Python scalar types, enabling correct downstream use in numerical analyses, plotting, indexing, and boolean operations without requiring ad-hoc post-processing conversions.
    """
    from anndata._io.utils import convert_string
    return convert_string(string)


################################################################################
# Source: anndata._io.utils.idx_chunks_along_axis
# File: anndata/_io/utils.py
# Category: valid
################################################################################

def anndata__io_utils_idx_chunks_along_axis(shape: tuple, axis: int, chunk_size: int):
    """anndata._io.utils.idx_chunks_along_axis
    Gives an iterator that yields indexer tuples which partition (chunk) an array shape along a single axis.
    
    This utility is used in anndata I/O and array processing code to iterate over contiguous blocks (batches) along a specified axis of an array-like object (for example, a 2-D expression matrix stored in memory or on disk via HDF5). By producing tuples of slice objects that can be passed directly to NumPy-like indexing (array[tuple_of_slices]), callers can read or process the array in fixed-size chunks to limit memory use, enable streaming I/O, or apply batch computations.
    
    Args:
        shape (tuple): A tuple of non-negative integers describing the shape of the target array to be indexed. Each element corresponds to the length of the array along that dimension. This must match the array layout used by the caller (for example, annData.X.shape). The function uses shape[axis] to determine how many elements exist along the chosen axis.
        axis (int): The axis index along which to chunk. This follows Python indexing semantics: 0 is the first axis; negative values are allowed and interpreted relative to the end of the shape tuple (for example, -1 refers to the last axis). The axis must be a valid index into the shape tuple, otherwise an IndexError will be raised.
        chunk_size (int): The desired size of each chunk along the specified axis. This must be a positive integer. The function yields successive slices of length chunk_size along the axis until fewer than chunk_size elements remain, at which point the final yielded slice covers the remainder. If chunk_size is not positive (for example, zero or negative) the behavior is undefined (in practice this may cause an infinite loop or other runtime errors), so callers must validate this value before calling.
    
    Behavior and side effects:
        The function does not modify the input shape or any external state; it is a generator that yields tuples of slice objects. Each yielded tuple has length equal to len(shape). For axes other than the specified axis, the tuple contains slice(None) (full-span slices), and for the specified axis the tuple contains a slice representing the contiguous chunk range. Chunks are non-overlapping and collectively cover the entire axis without gaps. The final chunk is created with slice(cur, None) to include all remaining elements when the remaining count is less than chunk_size.
        Typical use is for iterating over large datasets in batches, e.g., for reading subsets of rows or columns from disk-backed arrays to avoid loading the entire array into memory. The yielded tuples are directly suitable for indexing NumPy arrays, dask arrays, or h5py datasets that accept tuple-of-slices indexing.
    
    Failure modes and errors:
        An IndexError will be raised if axis is not a valid index for the shape tuple. A TypeError or ValueError can occur if shape does not contain indexable integer lengths (for example, if shape[axis] is not an integer). If chunk_size is zero or negative, the function's loop logic will not progress and may hang (infinite loop) or otherwise behave incorrectly; therefore chunk_size must be a positive int. No input types are coerced by this function — callers should pass the exact expected types.
    
    Returns:
        An iterator of tuples: A generator that yields tuples of slice objects suitable for indexing into an array of the given shape. Each tuple has length len(shape); slices on the specified axis represent contiguous chunks of size chunk_size (except the final chunk which may be smaller), and slices on other axes are slice(None). The caller can iterate over this generator to perform batch reads or writes.
    """
    from anndata._io.utils import idx_chunks_along_axis
    return idx_chunks_along_axis(shape, axis, chunk_size)


################################################################################
# Source: anndata._io.utils.is_float
# File: anndata/_io/utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def anndata__io_utils_is_float(string: str):
    """anndata._io.utils.is_float determines whether the provided string can be converted to a floating-point number using Python's built-in float() conversion. This function is used in anndata's I/O utilities when parsing textual representations of data (for example, when reading numeric annotations, matrix entries, or metadata from files) to decide whether a token should be interpreted as a numeric value or treated as non-numeric text.
    
    The implementation attempts float(string) and returns a Boolean result: True when conversion succeeds, False when conversion raises ValueError. The function has no side effects and does not mutate its input. It is a small, local utility intended to help with robust parsing of annotated data matrices in the anndata codebase; see the original reference used when implementing this approach (a StackOverflow discussion on checking float convertibility).
    
    Args:
        string (str): The input text token to test for float convertibility. In the anndata I/O context, this is typically a single field read from a text file or other serialized representation of annotated data (for example, a cell or gene attribute). The function expects a Python str; passing a non-str value may raise a TypeError from the underlying float() call, because only ValueError is caught by this function.
    
    Returns:
        bool: True if float(string) succeeds (meaning the string represents a floating-point literal according to Python's float() rules), False if float(string) raises ValueError (meaning the string cannot be interpreted as a float). No other side effects occur; other exceptions raised by float() (such as TypeError for non-str inputs) will propagate.
    """
    from anndata._io.utils import is_float
    return is_float(string)


################################################################################
# Source: anndata._io.utils.is_int
# File: anndata/_io/utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def anndata__io_utils_is_int(string: str):
    """anndata._io.utils.is_int: Determine whether a given string can be converted to an integer using Python's built-in int() conversion.
    
    Checks whether the input text token commonly produced by I/O and parsing routines in the anndata codebase represents an integer value. This helper is used when reading or interpreting textual fields (for example, column or index values from files or metadata strings) to decide whether a token should be treated as an integer. The function performs no I/O, has no side effects, and is intended to be a small, fast utility called by higher-level parsers in the anndata I/O utilities.
    
    Args:
        string (str): The text to test for integer representation. This argument is expected to be a Python str containing the characters to check (for example, a token extracted from a CSV or HDF5 attribute when parsing AnnData-related files). The function will pass this value to Python's int() conversion. If a non-str value is supplied, the underlying int() call may raise a TypeError (this function only catches ValueError and will not suppress TypeError), so callers should ensure they pass strings when relying on the documented behavior.
    
    Returns:
        bool: True if calling int(string) succeeds without raising ValueError (indicating the string represents an integer according to Python's int() semantics); False if int(string) raises ValueError (indicating the string does not represent a valid integer token). No other side effects occur.
    """
    from anndata._io.utils import is_int
    return is_int(string)


################################################################################
# Source: anndata.logging.get_logger
# File: anndata/logging.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for get_logger because the docstring has no description for the argument 'name'
################################################################################

def anndata_logging_get_logger(name: str):
    """Creates and returns a named child logger that is attached to the anndata library's central logger manager so that messages from library submodules and user code using anndata integrate with the package-wide logging configuration.
    
    This function is used across the anndata codebase and by downstream packages (for example, Scanpy and other scverse projects) to obtain a logger that delegates to the module-level anndata_logger manager instead of the global logging.root. The practical significance is that callers who obtain a logger with this function will inherit handlers, levels, and formatting configured for anndata, ensuring consistent logging behavior for annotated-data workflows (single-cell omics and related analyses) that rely on anndata for in-memory and on-disk annotated data handling.
    
    Args:
        name (str): The name of the logger to fetch or create. This string identifies the logger (commonly a module-level __name__) and determines the logger hierarchy used by the logging.Manager owned by the module-level anndata_logger. Calling this function with the same name repeatedly returns the same logging.Logger instance managed by anndata_logger.manager. The parameter must be a Python string; passing a non-string value may result in unexpected behavior or a TypeError raised by the underlying logging.Manager.
    
    Returns:
        logging.Logger: A logging.Logger instance retrieved from anndata_logger.manager for the given name. The returned logger delegates to the anndata package's central logger manager rather than logging.root, inherits handlers and level from the anndata logging configuration, and may be created as a side effect if no logger with the given name previously existed. Modifying the returned logger's configuration (for example, adding handlers or changing levels) affects how messages from that named logger are emitted within the broader anndata-managed logging hierarchy.
    """
    from anndata.logging import get_logger
    return get_logger(name)


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
