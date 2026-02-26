"""
Regenerated Google-style docstrings for module 'dscribe'.
README source: others/readme/dscribe/README.md
Generated at: 2025-12-02T00:49:02.995726Z

Total functions: 7
"""


import numpy

################################################################################
# Source: dscribe.descriptors.mbtr.check_geometry
# File: dscribe/descriptors/mbtr.py
# Category: valid
################################################################################

def dscribe_descriptors_mbtr_check_geometry(geometry: dict):
    """Used to validate MBTR geometry settings before computing Many-Body Tensor Representation (MBTR) descriptors.
    
    This function checks that the provided geometry configuration dictionary contains a "function" key and that its value is one of the allowed geometry functions used by MBTR k-body terms. In the context of the DScribe library, MBTR (Many-body Tensor Representation) converts atomic structures into fixed-size numerical fingerprints for machine learning and similarity analysis in materials science. Validating the geometry function here ensures that the MBTR descriptor will compute the intended k-body contribution (k = 1, 2, or 3) and prevents silent misconfiguration that would produce incorrect descriptors or runtime errors later in the descriptor pipeline.
    
    Args:
        geometry (dict): Dictionary containing the geometry setup for MBTR. The dictionary must include the key "function" whose value is a string (or other hashable identifier) naming the geometry function to use. The set of accepted names is the union of k1_geometry_functions, k2_geometry_functions, and k3_geometry_functions defined in the mbtr implementation; these represent the valid k = 1, k = 2, and k = 3 geometry functions respectively. This parameter is required and typically comes from the MBTR descriptor constructor or user configuration that specifies how distances, angles, or other geometric quantities are mapped into the MBTR spectrum.
    
    Returns:
        None: This function does not return a value. Its purpose is validation and it has no side effects other than raising an exception on invalid input. When validation succeeds, execution continues and the provided geometry dictionary is left unmodified.
    
    Raises:
        ValueError: If the "function" key is missing from the geometry dictionary, a ValueError is raised with the message "Please specify a geometry function." If the "function" value is present but not a member of the combined valid function sets (k1_geometry_functions | k2_geometry_functions | k3_geometry_functions), a ValueError is raised listing the allowed function names to guide the user.
        TypeError: If a non-dictionary is passed as geometry or the provided mapping does not support membership testing and item access as expected, Python may raise a TypeError before validation completes; callers should pass a dict-like object following the MBTR configuration conventions.
    """
    from dscribe.descriptors.mbtr import check_geometry
    return check_geometry(geometry)


################################################################################
# Source: dscribe.descriptors.mbtr.check_grid
# File: dscribe/descriptors/mbtr.py
# Category: valid
################################################################################

def dscribe_descriptors_mbtr_check_grid(grid: dict):
    """dscribe.descriptors.mbtr.check_grid validates MBTR grid settings and enforces basic consistency rules used by the Many-Body Tensor Representation (MBTR) descriptor in DScribe. This function is used before constructing MBTR fingerprints (fixed-size numerical descriptors for atomic structures) to ensure the provided grid dictionary contains the required entries that define the discretization range and resolution for the descriptor.
    
    Args:
        grid (dict): Dictionary containing the grid setup required by the MBTR descriptor. The dictionary must contain the keys "min", "max", "sigma", and "n". "min" and "max" represent the lower and upper bounds of the grid range used to discretize a continuous geometric or chemical quantity into a fixed-size vector; they must be comparable numeric values and satisfy min < max. "sigma" is expected to be a numeric broadening/width parameter associated with the grid (keeps its value unchanged by this function). "n" is the number of grid points (resolution) and will be converted in-place to an integer via int(grid["n"]); therefore "n" may be a numeric type or a string/number convertible to int. This function mutates the provided dictionary by replacing the original "n" value with its integer conversion.
    
    Returns:
        None: This function does not return a value. Instead, it performs in-place validation and mutation of the input grid dictionary (converts grid["n"] to int). Failure modes include KeyError raised when any required key ("min", "max", "sigma", or "n") is missing (the error message is "The grid information is missing the value for {key}"), ValueError raised when the provided min is greater than or equal to max, and ValueError or TypeError may be raised by the int(...) conversion if grid["n"] cannot be converted to an integer. These exceptions surface to the caller to indicate invalid grid configuration for MBTR descriptor construction.
    """
    from dscribe.descriptors.mbtr import check_grid
    return check_grid(grid)


################################################################################
# Source: dscribe.descriptors.mbtr.check_weighting
# File: dscribe/descriptors/mbtr.py
# Category: valid
################################################################################

def dscribe_descriptors_mbtr_check_weighting(k: int, weighting: dict, periodic: bool):
    """dscribe.descriptors.mbtr.check_weighting validates weighting settings for the Many-Body Tensor Representation (MBTR) descriptor in DScribe. It checks that the provided weighting dictionary contains a supported weighting function for the requested MBTR degree k (1, 2, or 3), that all required additional parameters for that function are present and not contradictory (for example, not providing both 'scale' and 'r_cut'), and that periodic systems have an appropriate non-unity weighting when required. This validation is used before constructing MBTR fingerprints (fixed-size numerical descriptors of atomic structures) so that downstream descriptor creation, machine learning, or analysis tasks receive consistent and well-specified weighting behavior.
    
    Args:
        k (int): The MBTR degree to validate. In the MBTR context, k corresponds to the body-order term: k=1 (one-body term), k=2 (two-body term), and k=3 (three-body term). The function uses k to determine which weighting functions are acceptable for that degree and to enforce periodic-system constraints for k>1.
        weighting (dict): Dictionary containing the weighting setup for the MBTR term or None. This dictionary is expected to contain at least the key "function" whose value is a string naming the weighting function. Allowed functions depend on k: for k=1 only "unity" is allowed; for k=2 allowed functions are "unity", "exp", and "inverse_square"; for k=3 allowed functions are "unity", "exp", and "smooth_cutoff". For the "exp" function, the dictionary must include a numeric "threshold" entry and either "scale" or "r_cut" but not both. For the "inverse_square" and "smooth_cutoff" functions, the dictionary must include a numeric "r_cut" entry. The function treats weighting==None as absence of a weighting specification (this is permitted by the code path but may fail validation for periodic systems).
        periodic (bool): Whether the MBTR descriptor is being computed for a periodic system. In periodic systems and for k>1, a weighting function must be provided and it must not be the "unity" function; this parameter enables the check that enforces that requirement.
    
    Returns:
        None: This function performs validation only and does not return a value. Side effects: it raises a ValueError when validation fails. Specific failure modes that raise ValueError include: an unknown weighting "function" string for the given k; missing required parameters such as "threshold" or "r_cut" for the selected function; supplying both "scale" and "r_cut" for the "exp" function (mutually exclusive); and the absence of a non-"unity" weighting function for periodic systems when k>1. The function does not mutate the input weighting dictionary.
    """
    from dscribe.descriptors.mbtr import check_weighting
    return check_weighting(k, weighting, periodic)


################################################################################
# Source: dscribe.utils.geometry.get_adjacency_matrix
# File: dscribe/utils/geometry.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def dscribe_utils_geometry_get_adjacency_matrix(
    radius: float,
    pos1: numpy.ndarray,
    pos2: numpy.ndarray = None,
    output_type: str = "coo_matrix"
):
    """Calculates a sparse adjacency matrix of pairwise Euclidean distances for points within a specified cutoff radius using a k-d tree for efficient neighbor search.
    
    This function is used throughout DScribe to build neighbor lists and distance-based adjacency information for atomic-structure descriptors (for example SOAP, ACSF, MBTR), where only interatomic pairs within a cutoff radius contribute to the descriptor. By delegating neighbor search to scipy.spatial.cKDTree.sparse_distance_matrix the function attains approximately O(n log n) scaling for large point sets, making it suitable for datasets of atoms or centers in high-throughput materials-science workflows.
    
    Args:
        radius (float): The cutoff radius within which pairwise distances are reported. Only pairs whose Euclidean distance is within this radius are included in the returned adjacency container. This parameter controls the physical neighborhood used by downstream descriptor calculations: a larger radius includes more distant neighbors (increasing computational cost and descriptor size), a smaller radius restricts the neighborhood (reducing cost and possibly losing information).
        pos1 (np.ndarray): An array of shape (N, D) representing N points in D-dimensional space (for atomic structures, D is typically 3 and rows are atomic positions). Each row is treated as a coordinate for which neighbors will be sought in pos2. The array must be numeric and finite; it is passed directly to scipy.spatial.cKDTree for indexing. The function does not modify pos1.
        pos2 (np.ndarray): An optional array of shape (M, D) representing M query points in the same D-dimensional space as pos1. If provided, the function computes pairwise distances between each row of pos1 and each row of pos2 and returns a sparse container of shape (N, M). If pos2 is None (the default), pos2 is assumed equal to pos1 and the returned container corresponds to pairwise distances within the same set (square, and symmetric in the sense that distance(i, j) == distance(j, i) for identical inputs). The array must be numeric and have the same number of columns D as pos1. The function does not modify pos2.
        output_type (str): Specifies which container/format to use for the returned adjacency data. Accepted options are "dok_matrix", "coo_matrix", "dict", or "ndarray". These map to the formats accepted by scipy.spatial.cKDTree.sparse_distance_matrix and represent: a scipy.sparse.dok_matrix ("dok_matrix"), a scipy.sparse.coo_matrix ("coo_matrix"), a Python dict-of-keys mapping (i, j) -> distance ("dict"), or a dense NumPy array ("ndarray"). The default value in the function signature is "coo_matrix". Passing an unsupported string will result in an error from the underlying scipy call. Choosing a sparse output (dok or coo) is recommended for large systems where the number of neighbor pairs within radius is much smaller than N*M.
    
    Returns:
        dok_matrix | np.array | coo_matrix | dict: A container holding pairwise Euclidean distances for point pairs with distance within the specified radius. When pos2 is None, the result corresponds to the square pairwise distance matrix for pos1 and will be symmetric (distance(i, j) == distance(j, i)) up to numerical precision. When pos2 is provided and differs from pos1, the returned matrix has shape (len(pos1), len(pos2)) and is not necessarily symmetric. Entries for pairs with distance greater than the cutoff are omitted in sparse containers (dok_matrix or coo_matrix) or absent from the dict; in a dense ndarray those entries will typically be zero or a value provided by scipy (depending on requested output_type). The concrete returned type corresponds to output_type: "dok_matrix" returns a scipy.sparse.dok_matrix, "coo_matrix" returns a scipy.sparse.coo_matrix, "dict" returns a Python dict mapping (i, j) -> distance, and "ndarray" returns a NumPy array. The distances are the same Euclidean distances used by DScribe descriptors to determine neighbor contributions.
    
    Behavior and failure modes:
        - The function builds scipy.spatial.cKDTree objects for pos1 and pos2; therefore pos1 and pos2 must be 2D numeric arrays with matching dimensionality (same number of columns). Supplying arrays of incompatible shapes, non-numeric data, NaNs, or infinities may raise exceptions from numpy or scipy.
        - If pos2 is None, pos2 is set to pos1 internally; in that case the returned container corresponds to intra-set distances and is symmetric.
        - The function does not modify the input arrays, but cKDTree is constructed with copy_data=False so arrays may be referenced by the tree object.
        - The runtime is dominated by the k-d tree neighbor search; performance degrades for extremely high-dimensional inputs or degenerate data distributions.
        - Invalid output_type values are forwarded to scipy and will raise an error; choose one of the documented options.
        - There are no other external side effects (no IO).
    """
    from dscribe.utils.geometry import get_adjacency_matrix
    return get_adjacency_matrix(radius, pos1, pos2, output_type)


################################################################################
# Source: dscribe.utils.species.get_atomic_numbers
# File: dscribe/utils/species.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def dscribe_utils_species_get_atomic_numbers(species: list):
    """Return ordered unique atomic numbers for a list of chemical species.
    
    This utility is used throughout DScribe to normalize user-provided species lists into a canonical form of atomic numbers that descriptors (for example SOAP, CoulombMatrix, MBTR) expect when constructing fixed-size fingerprints of atomic structures. The function accepts either a sequence of atomic numbers or a sequence of chemical element symbols and returns a sorted one-dimensional numpy array containing the unique atomic numbers present in the input. The function performs input validation, rejects non-iterable inputs and single strings, converts symbols to numbers when needed, checks for negative integers, and raises informative ValueError exceptions for malformed input.
    
    Args:
        species (iterable of ints or strings): An iterable container of chemical species provided either as non-negative integer atomic numbers (for example 1 for hydrogen, 8 for oxygen) or as chemical element symbols given as strings (for example "H", "O"). The iterable may be a list, tuple, numpy array, or any object implementing the iterator protocol, but it must not be a single string. The function requires that all elements of the iterable are of the same kind: either all integers or all strings. If integers are provided, they must be non-negative. If strings are provided, they must be valid chemical element symbols that can be converted by the internal symbols_to_numbers mapping. This parameter represents the set of species that will define the atomic-number-based species ordering used by descriptor calculations and indexing in DScribe.
    
    Returns:
        np.ndarray: A one-dimensional numpy array of integer atomic numbers corresponding to the provided species. The returned array contains unique atomic numbers (duplicates removed) and is sorted in ascending order. This array is intended for use by DScribe descriptors to define the species dimension, ensuring a stable and reproducible ordering of species when constructing feature vectors.
    
    Raises:
        ValueError: If species is not an iterable or if it is a single string, a ValueError is raised instructing the caller to provide an iterable (for example a list). ValueError is raised if the iterable contains negative integers. ValueError is raised if the iterable contains a mixture of types (some integers and some strings) or contains elements that are neither integers nor strings; in this case the caller should provide either all atomic numbers or all chemical symbols. If symbol conversion is attempted, errors from the underlying symbols_to_numbers routine (for example due to unknown element symbols) will propagate as exceptions indicating invalid chemical symbols.
    
    Side effects:
        None. The function does not modify the input iterable; it returns a new numpy array. It is a pure conversion/validation utility used to prepare species information for descriptor computation.
    """
    from dscribe.utils.species import get_atomic_numbers
    return get_atomic_numbers(species)


################################################################################
# Source: dscribe.utils.species.symbols_to_numbers
# File: dscribe/utils/species.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def dscribe_utils_species_symbols_to_numbers(symbols: list):
    """Convert a sequence of chemical element symbols into their corresponding atomic numbers for use in DScribe descriptors.
    
    This function is used throughout DScribe to prepare atomic species input for numerical descriptor generation (for example, SOAP, CoulombMatrix, ACSF, MBTR). It looks up each chemical symbol in the ASE atomic_numbers mapping (ase.data.atomic_numbers) and returns a numpy integer array where each element is the atomic number corresponding to the symbol at the same position in the input. The order of the output matches the input order and the resulting array can be passed directly to descriptor constructors and creation routines that expect atomic numbers.
    
    Args:
        symbols (list): A list of chemical element symbols (strings) in the order corresponding to atoms or species as used by DScribe. Each entry should be a valid chemical symbol recognized by ASE's ase.data.atomic_numbers mapping (for example, ['H', 'O', 'C']). The function does not modify the input list. Passing a non-iterable object will raise a TypeError from Python's iteration protocol; passing symbols that are not valid keys in ASE's atomic_numbers will trigger a ValueError as described below.
    
    Returns:
        np.ndarray: A one-dimensional numpy array of integers (dtype=int) of shape (len(symbols),) containing the atomic numbers corresponding to each input symbol. This array is intended for downstream numeric processing and for supplying atomic numbers to DScribe descriptor routines.
    
    Behavior and failure modes:
    - The function performs a lookup for each symbol using ase.data.atomic_numbers.get(symbol). If the lookup returns None for any symbol (meaning ASE does not recognize the symbol), the function raises a ValueError identifying the invalid symbol and stating that it has no associated atomic number.
    - The input is expected to be a list of strings; while the implementation will iterate over other iterable types, behavior for non-string entries depends on ASE's lookup and may result in a ValueError.
    - There are no side effects: the function does not modify external state or the input list. The only observable effects are the returned numpy array or an exception raised for invalid input.
    """
    from dscribe.utils.species import symbols_to_numbers
    return symbols_to_numbers(symbols)


################################################################################
# Source: dscribe.utils.stats.system_stats
# File: dscribe/utils/stats.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def dscribe_utils_stats_system_stats(system_iterator: list):
    """dscribe.utils.stats.system_stats: Compute aggregated statistics over a collection of atomic systems for use in DScribe descriptor construction and dataset analysis.
    
    Args:
        system_iterator (list): A list of atomic systems to analyze. Each element must be either an ASE.Atoms instance or a dscribe System object (as used elsewhere in this repository). Each element is expected to implement/allow the following operations which the function uses: len() to obtain the number of atoms, get_atomic_numbers() to obtain atomic numbers, get_chemical_symbols() to obtain element symbols, get_distance_matrix() to obtain interatomic distances, and get_pbc() to query periodic boundary conditions. Typical practical use in the DScribe workflow is to pass a list of structures (e.g., ASE Atoms objects or preconstructed dscribe System objects) so that global dataset statistics such as maximum system size and element ranges can be inferred automatically (for example, to choose n_atoms_max for CoulombMatrix or to list species for SOAP). The function does not accept additional parameter types beyond a list as given in the signature; supplying other container types or objects that do not implement the required methods will lead to exceptions.
    
    Returns:
        Dict: A dictionary containing aggregated statistics computed across all systems in system_iterator. The dictionary contains the following keys and associated meanings, types, and practical significance in materials-descriptor workflows:
        n_atoms_max: The maximum number of atoms found in any system (int). This value is useful when configuring fixed-size descriptors (for example, the n_atoms_max parameter of CoulombMatrix) and for memory/shape planning when creating descriptor arrays for a dataset.
        max_atomic_number: The largest atomic number present across all systems (int). This is used to understand the chemical range in the dataset and may be used to size species-dependent arrays or validate species lists.
        min_atomic_number: The smallest atomic number present across all systems (int). As with max_atomic_number, this describes the lower bound of the chemical elements present and helps in dataset validation and descriptor configuration.
        atomic_numbers: A list of all distinct atomic numbers present across the input systems (list of int). The list is produced by converting an internal set to a list; therefore the order is not guaranteed. This list is practically used to determine which element channels to include in species-aware descriptors.
        element_symbols: A list of distinct element symbols present across the input systems (list of str). Like atomic_numbers the order is not guaranteed. This list is useful for human-readable summaries and for mapping element symbols to atomic numbers when preparing inputs for descriptors.
        min_distance: The minimum interatomic distance found across all systems (float or numpy scalar). For non-periodic systems this is the smallest pairwise distance excluding self-distances; for periodic systems the function includes distances to periodic images (it intentionally includes the diagonal entries produced by get_distance_matrix() for periodic systems, since those diagonal entries represent shortest periodic-image distances). min_distance is important for checking unphysically short atomic separations, choosing cutoffs (r_cut) for local descriptors, and validating structures prior to descriptor computation.
    
    Behavior, side effects, defaults, and failure modes:
        - The function iterates over the provided list and, for each element that is an ASE.Atoms instance, constructs a dscribe System object using System.from_atoms(system). This construction is performed internally and does not mutate the original ASE.Atoms objects passed by the caller.
        - The function does not modify the input list or the original atomic objects; it only reads data from them and may construct transient System objects.
        - For systems where any component of get_pbc() is True (periodic boundary conditions present), diagonal entries of get_distance_matrix() are considered when computing the minimum distance; for fully non-periodic systems the function excludes self-distances by considering only strictly off-diagonal distances.
        - If system_iterator is empty or no atomic numbers are discovered across all entries, the function will fail when computing max(list(atomic_numbers)) and min(list(atomic_numbers)) and raise a ValueError. Therefore at least one system containing at least one atom and at least one atomic number must be present.
        - If any entry in system_iterator does not implement the required methods (len, get_atomic_numbers, get_chemical_symbols, get_distance_matrix, get_pbc) a TypeError or AttributeError (or other exceptions raised by those missing calls) will be propagated.
        - If a system contains zero atoms, computing distances or their minimum will raise an error (for example when taking the minimum of an empty distance array); callers should ensure systems contain one or more atoms.
        - The returned atomic_numbers and element_symbols lists are derived from sets and therefore do not have a guaranteed order; if deterministic ordering is required, the caller should sort these lists explicitly after receiving the result.
    
    No other side effects occur; the function returns the described dictionary for downstream use in DScribe descriptor setup, dataset validation, and selection of numerical parameters (such as n_atoms_max and cutoff distances).
    """
    from dscribe.utils.stats import system_stats
    return system_stats(system_iterator)


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
