"""
Regenerated Google-style docstrings for module 'stk'.
README source: others/readme/stk/README.rst
Generated at: 2025-12-02T01:02:09.548248Z

Total functions: 12
"""


import numpy

################################################################################
# Source: stk.molecular.topology_graphs.topology_graph.optimizers.utilities.get_subunits
# File: stk/molecular/topology_graphs/topology_graph/optimizers/utilities.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def stk_molecular_topology_graphs_topology_graph_optimizers_utilities_get_subunits(
    state: str
):
    """stk.molecular.topology_graphs.topology_graph.optimizers.utilities.get_subunits: Return connected subunit graphs grouped by building block identifier from a ConstructionState.
    
    Args:
        state (.ConstructionState): The ConstructionState representing the molecule under construction in stk. In the stk domain, ConstructionState encapsulates the current assembled molecular structure and provides atom-level metadata via its get_atom_infos() method. This function reads state.get_atom_infos(), expecting each atom info object to implement get_building_block_id() (the identifier of the building block the atom belongs to) and get_atom().get_id() (the atomic identifier used by stk). The state is not modified; it is used read-only to derive grouping information needed by topology-graph optimizers that operate on connected subunits defined by building blocks.
    
    Returns:
        dict: A mapping from subunit identifier to the atoms comprising that subunit. Each key is the building block id returned by atom_info.get_building_block_id() (the identifier used by stk to associate atoms with their originating building block). Each value is a list of atom ids (the values returned by atom_info.get_atom().get_id()) that belong to that building block. Practical significance: topology-graph optimizers use this dictionary to treat each building-block-derived subunit as a connected component when performing operations such as placement, rotation, or connectivity checks during automated molecular assembly.
    
    Notes:
        Behavior: Iterates over all atom info objects returned by state.get_atom_infos() and appends each atom's id to the list for its building block id, using the insertion order provided by get_atom_infos(). If get_atom_infos() yields no atom infos, an empty dict is returned. The function constructs the mapping using a list for each key, preserving duplicate atom ids if present in the input.
    
        Side effects: None on the provided state; the function only reads data from state and returns a new dictionary.
    
        Failure modes: Raises AttributeError if state does not implement get_atom_infos(), or if atom info objects do not implement get_building_block_id() or get_atom(). Raises whatever exceptions are raised by get_atom().get_id() if that method fails. The function does not validate building block ids or atom ids beyond using the returned values as dict keys and list elements.
    """
    from stk.molecular.topology_graphs.topology_graph.optimizers.utilities import get_subunits
    return get_subunits(state)


################################################################################
# Source: stk.utilities.utilities.cap_absolute_value
# File: stk/utilities/utilities.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def stk_utilities_utilities_cap_absolute_value(value: float, max_absolute_value: float = 1):
    """stk.utilities.utilities.cap_absolute_value returns `value` with its absolute magnitude limited to `max_absolute_value`. This utility is used in stk's computational routines (for example when preparing arguments for trigonometric functions or computing cosines of angles in molecular geometry), where numerical round-off or accumulated errors can yield inputs slightly outside the mathematically valid range; capping prevents downstream domain errors in such functions. The implementation is adapted from pymatgen.
    
    Args:
        value (float): Value to cap. This is the floating-point number whose magnitude will be limited. In practical stk usage this often is a computed cosine or other scalar derived from molecular geometry; capping ensures the value lies within the allowed range for trigonometric or other domain-restricted functions. The function performs no in-place modification; it returns a new float.
        max_absolute_value (float): Absolute value to cap `value` at. If `value` is greater than `max_absolute_value`, the function returns `max_absolute_value`; if `value` is less than `-max_absolute_value`, the function returns `-max_absolute_value`. Defaults to 1. Use this parameter to change the allowed magnitude for your domain (for example, keep within [-1, 1] for safe arccos/arcsin inputs).
    
    Behavior, side effects, defaults, and failure modes:
        The function is pure and has no side effects. It returns a float whose sign matches the original `value` but whose absolute magnitude does not exceed `max_absolute_value`. The default `max_absolute_value` is 1 to support typical use cases where values are passed to inverse trigonometric functions. If `value` already lies within [-max_absolute_value, max_absolute_value], it is returned unchanged. If non-finite values (infinities, NaN) or non-numeric types are provided, behavior follows Python's built-in numeric comparisons and may propagate NaN or raise a TypeError for types that do not support ordering; callers should validate inputs if deterministic handling of such cases is required.
    
    Returns:
        float: A floating-point number equal to `value` but with its absolute value limited to `max_absolute_value`, preserving the original sign. This returned value can be safely used in downstream computations that require inputs within a bounded range (e.g., inverse trigonometric functions).
    """
    from stk.utilities.utilities import cap_absolute_value
    return cap_absolute_value(value, max_absolute_value)


################################################################################
# Source: stk.utilities.utilities.get_projection
# File: stk/utilities/utilities.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def stk_utilities_utilities_get_projection(start: numpy.ndarray, target: numpy.ndarray):
    """Get the projection of start onto target.
    
    Compute the orthogonal projection of the vector represented by start onto the vector represented by target using the linear-algebra formula
    target * (np.dot(start, target) / np.dot(target, target)).
    In the stk domain this routine is used when manipulating molecular geometry or vector quantities (for example, projecting an atomic displacement, an orientation vector, or a placement direction onto another direction when aligning building blocks, positioning functional groups, or extracting directional components of properties during molecular construction and analysis).
    
    Args:
        start (numpy.ndarray): Numeric array holding the vector to be projected. In practical stk use this commonly represents a coordinate difference, displacement vector, or other per-atom/per-feature vector whose component along target is required. Must be a numpy.ndarray for which np.dot(start, target) is defined; if it is not (for example, due to incompatible shapes or non-numeric contents) numpy will raise an error.
        target (numpy.ndarray): Numeric array holding the vector onto which start is projected. In stk workflows this typically represents an axis, bond direction, or placement/orientation vector used to align or decompose other vectors. Must be a numpy.ndarray for which np.dot(start, target) and np.dot(target, target) are defined. If target is the zero vector (so np.dot(target, target) == 0.0) the computation divides by zero and will produce invalid values (NaNs or Infs) or trigger a runtime warning/error from numpy; callers should avoid zero-length target vectors or check for this condition prior to calling.
    
    Returns:
        numpy.ndarray: The projection of start onto target, computed as target * (np.dot(start, target) / np.dot(target, target)). The returned numpy.ndarray represents the component of start along target and can be directly used to translate or align molecular fragments in stk construction routines. No side effects occur (the function is pure and does not modify its inputs). Failure modes include numpy errors for incompatible shapes passed to np.dot, and invalid numeric results if target is the zero vector or if inputs contain non-finite values; numerical precision is subject to standard floating-point rounding behavior.
    """
    from stk.utilities.utilities import get_projection
    return get_projection(start, target)


################################################################################
# Source: stk.utilities.utilities.kabsch
# File: stk/utilities/utilities.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def stk_utilities_utilities_kabsch(coords1: numpy.ndarray, coords2: numpy.ndarray):
    """stk.utilities.utilities.kabsch
    Return a rotation matrix that minimizes the root-mean-square (RMS) distance between two sets of 3D coordinates using the Kabsch algorithm. This function is used in the stk library for molecular alignment tasks (for example, aligning atomic coordinates of one molecule to another during construction or comparison). The returned 3x3 matrix, when applied as a right-multiplication to coords1 (which are treated as row vectors), produces coordinates whose RMS distance to coords2 is minimized, provided the input coordinate sets are prepared as described below.
    
    Args:
        coords1 (numpy.ndarray): An n x 3 array of Cartesian coordinates representing the points to be rotated. Each row corresponds to the x, y, z coordinates of one point (atom). In the molecular assembly and alignment domain of stk, coords1 typically holds the positions of atoms in the structure to be rotated. Practical significance: to obtain the intended minimizing rotation, coords1 should be expressed as row vectors and should be translated so that its centroid coincides with the centroid of coords2 (see behavior and failure modes below).
        coords2 (numpy.ndarray): An n x 3 array of Cartesian coordinates representing the target points to which coords1 should be aligned. Each row corresponds to the x, y, z coordinates of one point (atom). In stk workflows this typically holds the reference atomic positions. Practical significance: coords2 must have the same ordering and one-to-one correspondence of points with coords1, and should be centered (centroid at the origin) if the user expects the classical Kabsch solution that minimizes RMS distance about centroids.
    
    Returns:
        numpy.ndarray: A 3 x 3 rotation matrix (dtype and exact shape follow numpy.ndarray conventions). This matrix is an orthogonal matrix implementing a proper rotation (determinant corrected to +1 where possible by the algorithm). To apply the rotation to coords1 (treated as row vectors), use right-multiplication: rotated_coords = coords1.dot(rotation_matrix). Practical significance: after applying this rotation to coords1 (and assuming both coordinate sets share the same centroid), the RMS distance between the rotated coords1 and coords2 is minimized.
    
    Detailed behavior, side effects, defaults, and failure modes:
    - Algorithmic behavior: The implementation computes the cross-covariance of the two coordinate sets and performs a singular value decomposition (SVD) to obtain orthogonal matrices. If an improper rotation (reflection) is detected via the sign of the determinant, the implementation flips the sign of the third column of the relevant SVD matrix to enforce a proper rotation (determinant +1). The returned matrix is the product of these orthogonal components and represents the minimizing rotation under the Kabsch formulation.
    - Preconditioning required: The classical Kabsch algorithm minimizes RMS distance between point sets about their centroids. This implementation does not perform explicit centroid subtraction internally; callers should translate coords1 and coords2 so that they share the same origin (for example, subtract each set's centroid) before calling kabsch, otherwise the computed rotation will not, in general, minimize the RMS distance about centroids.
    - Input expectations and shape: Both coords1 and coords2 must be numpy.ndarray objects with shape (n, 3) and the same n, where rows are x, y, z coordinates. The function assumes a one-to-one correspondence between rows (points) in coords1 and coords2; incorrect ordering or mismatched correspondences will produce incorrect alignments.
    - Numerical and library failure modes: If the input arrays have incompatible shapes or ranks, numpy.dot will raise an exception (for example, ValueError). If the SVD fails for numerical reasons, numpy.linalg.svd may raise a LinAlgError. The determinant correction relies on the numeric sign of the determinant; for nearly singular or ill-conditioned inputs numerical precision can affect the result.
    - Side effects: The function is pure with respect to its inputs (it does not modify coords1 or coords2) and has no external side effects. It returns a new numpy.ndarray containing the rotation matrix.
    - Practical notes for use in stk: Use this function when you need a rigid-body rotation to optimally align atomic coordinates (e.g., aligning fragments, comparing conformers, or preparing structures for assembly). Ensure consistent ordering and centering of coordinates before calling, and apply the returned matrix via right-multiplication to row-oriented coordinate arrays.
    """
    from stk.utilities.utilities import kabsch
    return kabsch(coords1, coords2)


################################################################################
# Source: stk.utilities.utilities.matrix_centroid
# File: stk/utilities/utilities.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def stk_utilities_utilities_matrix_centroid(matrix: numpy.ndarray):
    """stk.utilities.utilities.matrix_centroid computes and returns the geometric centroid (arithmetic mean position) of a set of 3D coordinates stored in a matrix. In the context of the stk library (used for automated molecular and supramolecular construction and manipulation), this function is used to obtain the geometric center of atom coordinates for tasks such as centering a molecule, aligning fragments, or determining placement points in assembly procedures.
    
    Args:
        matrix (numpy.ndarray): An n x 3 matrix where each row holds the x, y and z coordinates of a single point (for example, atomic coordinates of a molecule) in that order. The matrix is interpreted as a collection of 3D points and the function computes the arithmetic mean of these points along the row axis. The function expects numeric entries; integer arrays will be converted to floating-point results by the division. The input is not modified by the function.
    
    Returns:
        numpy.ndarray: A length-3 numpy.ndarray containing the x, y and z coordinates of the centroid (arithmetic mean) of the rows in matrix. The returned array represents the geometric center (not a mass-weighted center) and is suitable for use in subsequent molecular placement or alignment operations.
    
    Behavior and failure modes:
        The centroid is computed as the sum of the rows (axis=0) divided by len(matrix), i.e., the arithmetic mean of the coordinates, with O(n) time complexity in the number of rows. If matrix is empty (zero rows) the division by zero will raise a ZeroDivisionError. The function assumes an n x 3 shape; providing an array of a different shape may produce an unexpected result or a NumPy error (for example, shape-related exceptions) and is therefore not supported. The function has no side effects and does not perform mass-weighting or any chemistry-specific weighting; if a mass-weighted center (center of mass) is required, coordinates must be weighted externally before calling this function.
    """
    from stk.utilities.utilities import matrix_centroid
    return matrix_centroid(matrix)


################################################################################
# Source: stk.utilities.utilities.mol_from_mae_file
# File: stk/utilities/utilities.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def stk_utilities_utilities_mol_from_mae_file(mae_path: str):
    """stk.utilities.utilities.mol_from_mae_file: Create an rdkit molecule (rdkit.Mol) from a .mae file for use in stk molecule construction, manipulation, automated molecular design, and molecular/databasing workflows.
    
    This function parses a Maestro (.mae) file at the given filesystem path, extracts the atom and bond blocks, and constructs an rdkit.Mol with a single rdkit.Conformer containing the 3D coordinates from the file. It is intended for use within the stk library when converting .mae-format molecular files into rdkit objects for further processing (e.g., building, editing, property calculation, and storage in molecular databases described in the project README).
    
    Args:
        mae_path (str): The full filesystem path to the .mae file to read. This path must point to a readable text .mae file encoded in the Maestro format expected by this parser. The function opens this path with Python's built-in open(), so standard OS errors such as FileNotFoundError or PermissionError may be raised if the file does not exist or is inaccessible. The string is used only as the source of the .mae content; no write or modification of the input file occurs.
    
    Behavior and parsing details:
        The parser splits the file content on curly braces to locate blocks labelled with 'm_atom[' and 'm_bond['. The atom block is expected to contain a header of labels and a data block, separated by ':::'. Labels are parsed by splitting on newlines and filtering empty lines; data rows are split on whitespace. For each atom row the function reads the atomic_number (parsed as int) and x_coord, y_coord, z_coord (parsed as float) labels to determine the element and 3D position. The element symbol is looked up using the module-level periodic_table mapping (periodic_table[atomic_number]) and added to an rdkit.EditableMol as an rdkit.Atom. The atomic 3D coordinates are stored in an rdkit.Conformer created inside the function and assigned to each atom by index.
    
        The bond block is parsed similarly. Bond rows are expected to include columns labelled 'from', 'to', and 'order'. The 'from' and 'to' fields are interpreted as 1-based atom indices in the .mae file and are converted to 0-based indices before adding a bond to the editable molecule. The bond order value is converted to a string of an integer and mapped to an rdkit bond type using the module-level bond_dict mapping (bond_dict[bond_order]) before adding the bond.
    
    Return and side effects:
        Returns:
            rdkit.Mol: An rdkit molecule instance representing the structure encoded in the .mae file. The returned mol contains atoms and bonds corresponding to the parsed .mae data and has a single conformer attached that stores the 3D coordinates from the file. The function does not write to disk or modify global program state beyond reading the input file and using module-level mappings (periodic_table and bond_dict) to resolve element symbols and bond types.
    
    Failure modes and exceptions:
        The function performs minimal validation and will raise exceptions in several cases:
        - RuntimeError is raised explicitly when the number of labels parsed for a block does not match the number of columns in a data row (either for atoms or bonds). This indicates malformed or unexpected .mae formatting.
        - FileNotFoundError, PermissionError, or other IOErrors may be raised by open(mae_path, 'r') if the path is invalid or inaccessible.
        - ValueError may be raised when converting coordinate or atomic number strings to float/int if the data is not numeric.
        - KeyError may be raised if the module-level periodic_table or bond_dict mappings do not contain entries for a parsed atomic number or bond order.
        - IndexError may occur if bond indices in the bond block refer to atom indices outside the range of parsed atoms.
        - rdkit-related exceptions may be raised during atom, bond, or conformer operations if rdkit rejects constructed elements or topologies.
    
    Notes and practical significance:
        - The function assumes the .mae file follows the specific block and label conventions used by the upstream data this stk utility targets; it is not a general-purpose Maestro parser. It is used in stk workflows to convert .mae source files into rdkit.Mol objects for downstream assembly, analysis, and database ingestion as described in the stk README.
        - Atom indices in bond records are converted from the 1-based convention in many molecular file formats to the 0-based indexing used by rdkit.
        - The returned rdkit.Mol contains spatial information via the attached conformer; downstream stk components can rely on these coordinates for geometry-based assembly and property calculations.
    """
    from stk.utilities.utilities import mol_from_mae_file
    return mol_from_mae_file(mae_path)


################################################################################
# Source: stk.utilities.utilities.normalize_vector
# File: stk/utilities/utilities.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def stk_utilities_utilities_normalize_vector(vector: numpy.ndarray):
    """stk.utilities.utilities.normalize_vector
    Normalizes a numeric vector using its Euclidean (L2) norm and returns a new array containing the unit vector. This utility is used throughout stk for geometric calculations in molecular construction and manipulation (for example, normalizing coordinate difference vectors, orientation axes for placing building blocks, and direction vectors used in assembly algorithms).
    
    Args:
        vector (numpy.ndarray): The input vector to normalize. This must be a one-dimensional NumPy array (dtype and shape are preserved where possible). The function computes the Euclidean norm with numpy.linalg.norm and divides the input by that scalar. The original array is not modified; a new NumPy array is returned.
    
    Returns:
        numpy.ndarray: A new NumPy array with the same shape as the input containing the normalized (unit) vector equal to vector / ||vector|| where ||vector|| is the Euclidean norm computed by numpy.linalg.norm. If the input is the zero vector (norm == 0), division by zero will occur: NumPy will produce inf or NaN values in the result and emit a RuntimeWarning. Callers in stk code that depend on valid unit vectors (for example, placing or orienting molecular fragments) should check for zero norms before calling this function or handle the resulting NaNs/Infs appropriately.
    
    Behavior and side effects:
        - The operation is performed elementwise via numpy.divide and uses numpy.linalg.norm for the denominator.
        - No in-place modification of the input occurs; a new array is returned.
        - The function preserves the input array's shape and generally preserves numeric dtype subject to NumPy's promotion rules.
        - Numerical issues such as underflow/overflow or loss of precision are governed by NumPy's floating-point behavior; this function itself performs no additional safety checks.
        - Intended for use in stk's molecular geometry and assembly context to produce unit direction vectors for placement, orientation, and geometric computations.
    """
    from stk.utilities.utilities import normalize_vector
    return normalize_vector(vector)


################################################################################
# Source: stk.utilities.utilities.quaternion
# File: stk/utilities/utilities.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def stk_utilities_utilities_quaternion(u: list):
    """stk.utilities.utilities.quaternion: Compute a translation + rotation quaternion from three scalar parameters.
    
    Returns a quaternion suitable for encoding a 3D rotation (and used in stk for applying orientations to molecular building blocks during construction and assembly). The implementation follows the Shoemake algorithm for generating uniform random rotations (see K. Shoemake, Uniform random rotations, Graphics Gems III, 1992) by mapping three scalar parameters into a four-component quaternion. The function is used in the stk library to sample or apply rotations when orienting molecules, constructing supramolecular assemblies, and performing random rotational perturbations during automated molecular design workflows.
    
    Args:
        u (list of float): A list of three floating-point parameters [a, b, c] used to construct the quaternion. Each element corresponds to one of the random parameters in Shoemake's method: a controls the distribution between components (used inside square roots), and b and c enter as angular phases through 2 * pi * b and 2 * pi * c. In typical use within stk these three values are sampled uniformly from the interval [0, 1] to produce a uniformly distributed rotation in SO(3). The function requires that u is iterable with exactly three elements; incorrect length will raise a ValueError from the attempt to unpack. Providing non-numeric entries may raise a TypeError or result in propagation of invalid numeric values. Supplying values outside the range [0, 1] is not supported by the Shoemake sampling assumption and will result in mathematically undefined or NaN components when square roots of negative numbers occur.
    
    Returns:
        numpy.ndarray: A one-dimensional NumPy array of dtype numpy.float64 and length 4 containing the quaternion components q such that
            q[0] = sqrt(1 - a) * sin(2 * pi * b)
            q[1] = sqrt(1 - a) * cos(2 * pi * b)
            q[2] = sqrt(a) * sin(2 * pi * c)
            q[3] = sqrt(a) * cos(2 * pi * c)
        The returned quaternion encodes a rotation in 3D and can be used with quaternion-to-matrix conversions or direct quaternion rotation routines to rotate coordinates of atoms or molecular fragments in stk. The function has no side effects and is deterministic for a given input list u.
    """
    from stk.utilities.utilities import quaternion
    return quaternion(u)


################################################################################
# Source: stk.utilities.utilities.rotation_matrix
# File: stk/utilities/utilities.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def stk_utilities_utilities_rotation_matrix(vector1: numpy.ndarray, vector2: numpy.ndarray):
    """stk.utilities.utilities.rotation_matrix
    Returns a 3x3 rotation matrix that rotates one 3D Cartesian vector onto another.
    
    This function is used throughout stk when orienting molecular fragments and building blocks so that one direction (for example, a bond vector or connector direction) in a constructed component is aligned with a target direction in the assembled structure. The implementation normalizes both input vectors, handles special cases where the vectors are already equal or exactly opposite (180° rotation), and otherwise constructs a rotation matrix using the cross product and a Rodrigues-like formula. The resulting matrix is orthonormal and has determinant +1; it is produced via scipy.spatial.transform.Rotation.from_matrix to enforce numerical normalization before being returned as a numpy.ndarray. Numerically, the function treats vectors as 3-element numpy arrays (3D Cartesian coordinates) and uses an absolute tolerance of 1e-8 to detect equality or opposition of the input vectors.
    
    Args:
        vector1 (numpy.ndarray): The source 3D vector to be rotated. In the stk context this typically represents a direction associated with a molecular fragment (for example the direction of a functional group or connector). The function normalizes this vector internally; if it has zero length normalization will fail (the underlying normalize_vector call will raise an error). The caller should provide a length-3 array of Cartesian coordinates.
        vector2 (numpy.ndarray): The target 3D vector onto which vector1 must be rotated. In stk this typically represents the desired orientation in the assembled molecule (for example the direction of a target bond). The function normalizes this vector internally; if it has zero length normalization will fail. The caller should provide a length-3 array of Cartesian coordinates.
    
    Returns:
        numpy.ndarray: A 3x3 rotation matrix R (dtype float) such that applying the matrix to vector1 yields vector2 within numerical tolerance. In numpy terms, np.allclose(np.dot(R, vector1_normalized), vector2_normalized, atol=1e-8) will hold, where vector1_normalized and vector2_normalized are the unit vectors obtained by normalizing the inputs. The matrix is orthonormal and has determinant +1. Special-case behavior: if the inputs are equal within atol=1e-8 the identity matrix is returned; if they are opposite within atol=1e-8 a rotation of pi around an arbitrary axis orthogonal to vector1 is returned (the axis is chosen by orthogonal_vector). No other side effects occur.
    
    References:
        The algorithm follows standard constructions using cross products and a Rodrigues-type formula; see http://tinyurl.com/kybj9ox and http://tinyurl.com/gn6e8mz for derivations and background.
    """
    from stk.utilities.utilities import rotation_matrix
    return rotation_matrix(vector1, vector2)


################################################################################
# Source: stk.utilities.utilities.rotation_matrix_arbitrary_axis
# File: stk/utilities/utilities.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def stk_utilities_utilities_rotation_matrix_arbitrary_axis(
    angle: float,
    axis: numpy.ndarray
):
    """stk.utilities.utilities.rotation_matrix_arbitrary_axis generates a 3x3 rotation matrix that rotates vectors by a specified angle (in radians) about a specified 3D axis vector. In the stk codebase this is used when assembling and manipulating molecular fragments and coordinates (for example to rotate building blocks or molecular subunits during construction and alignment).
    
    Args:
        angle (float): The size of the rotation in radians. This scalar controls the magnitude of the rotation applied about the axis. The implementation uses the quaternion half-angle representation (a = cos(angle/2)) to compute the matrix elements, so values are interpreted as radians and there are no defaults; pass a Python float (or compatible numeric) representing the desired rotation angle.
        axis (numpy.ndarray): A 3 element array which represents a vector; the vector is the axis about which the rotation is carried out. Must be of unit magnitude (length 1) for the produced rotation matrix to correspond exactly to a rotation by the given angle. The function does not internally normalize this input axis before forming the quaternion components, so callers should supply a normalized numpy.ndarray of shape (3,) containing finite numeric values. In the practical domain of stk, this axis is typically a direction vector derived from atomic coordinates or geometric features of molecular fragments.
    
    Returns:
        numpy.ndarray: A 3x3 array representing an orthonormal rotation matrix corresponding to a rotation of angle radians about the provided axis. The matrix is produced by computing quaternion-derived matrix elements from the half-angle values and then passing the assembled 3x3 array through scipy.spatial.transform.Rotation.from_matrix(...).as_matrix(), which normalizes the matrix and returns it in standard numpy.ndarray form. The returned matrix can be applied to 3D column vectors x by matrix multiplication (x_rotated = R @ x) to rotate coordinates in the stk molecular construction/manipulation workflow.
    
    Behavior and side effects:
        This function is pure (no external side effects) and returns a new numpy.ndarray. It computes intermediate quaternion components a, b, c, d where a = cos(angle/2) and b,c,d = axis * sin(angle/2), constructs the 3x3 matrix elements (e11..e33) from those components, and then converts and normalizes the matrix via scipy.spatial.transform.Rotation. There are no hidden state changes or I/O operations.
    
    Failure modes and input validation notes:
        The function expects axis to be a length-3 numpy.ndarray and angle to be a float-like scalar. If axis does not have exactly three elements, broadcasting or indexing operations will raise a ValueError or IndexError (depending on the provided shape), and the function will not produce a valid rotation matrix. If axis is not unit magnitude, the computed quaternion will not represent a unit rotation and the resulting matrix will be incorrect for the intended rotation (although Rotation.from_matrix will normalize the matrix, callers should not rely on implicit correction and should normalize the axis themselves). If axis contains non-finite values (NaN or Inf) or non-numeric types, the arithmetic will propagate invalid values and the function may raise exceptions or return an array containing NaNs. The function does not accept other input types in place of numpy.ndarray for axis or non-float types for angle unless they are implicitly convertible to the documented types.
    """
    from stk.utilities.utilities import rotation_matrix_arbitrary_axis
    return rotation_matrix_arbitrary_axis(angle, axis)


################################################################################
# Source: stk.utilities.utilities.translation_component
# File: stk/utilities/utilities.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def stk_utilities_utilities_translation_component(q: numpy.ndarray):
    """stk.utilities.utilities.translation_component extracts the translation vector encoded in a length-4 quaternion used by stk for placing and orienting molecular building blocks.
    
    Args:
        q (numpy.ndarray): A length 4 quaternion provided as a 1-D numpy array using the scalar-first convention (q[0] is the scalar component and q[1:4] are the vector components). In the stk codebase quaternions are used to encode rigid-body transformations (rotation information that can be converted to an axis-angle / translation-like 3-vector for downstream placement of molecular fragments). This function treats q as input only and makes an internal copy so the original array is not modified. If q is not a numeric 1-D array of length 4, the function will raise indexing or arithmetic errors from numpy operations.
    
    Returns:
        numpy.ndarray: A length-3 numpy array containing the translation-like vector decoded from the input quaternion. Concretely, the function computes the rotation angle theta = 2 * arccos(q[0]) (with q[0] clamped/sign-adjusted as described below) and returns the axis-angle style vector p = axis * theta, where axis is derived from q[1:4]. Implementation details affecting the returned value and numerical behavior:
        - The function first copies q to avoid mutating the caller's array.
        - If q[0] < 0.0 the quaternion is negated (q = -q) so the scalar component is non-negative; this enforces a consistent shortest-rotation convention used across stk.
        - If q[0] > 1.0 the quaternion is normalized (q /= sqrt(dot(q, q))) to correct for small numerical drift while preserving direction; no other input rescaling is performed.
        - A numerical threshold rot_epsilon = 1e-6 is used to detect when s = sqrt(1 - q[0]**2) is effectively zero. If s < rot_epsilon a first-order approximation p = 2 * q[1:4] is returned to avoid division by a near-zero value; otherwise the exact conversion p = q[1:4] / s * theta is returned.
        - No in-place changes are made to the caller's q; the function returns a new numpy.ndarray. Exceptions from improper input shapes, non-numeric types, or other numpy arithmetic errors are propagated to the caller.
    """
    from stk.utilities.utilities import translation_component
    return translation_component(q)


################################################################################
# Source: stk.utilities.utilities.vector_angle
# File: stk/utilities/utilities.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def stk_utilities_utilities_vector_angle(vector1: numpy.ndarray, vector2: numpy.ndarray):
    """stk.utilities.utilities.vector_angle
    Returns the angle between two vectors in radians. This utility function is part of the stk library's geometry helpers used when constructing and manipulating molecular and supramolecular structures (for example, measuring the angle between bond displacement vectors, normals of molecular fragments, or orientation vectors of building blocks during assembly and alignment operations).
    
    Args:
        vector1 (numpy.ndarray): The first vector. This is interpreted as a numeric 1-D array of Cartesian components (e.g., a displacement or orientation vector in 3D space) used in molecular-geometry calculations inside stk. The function does not modify this array. If this array is exactly equal element-wise to vector2 (checked with numpy.equal), the function returns 0.0 immediately.
        vector2 (numpy.ndarray): The second vector. This is interpreted as a numeric 1-D array of Cartesian components analogous to vector1 and used to compute the geometric angle between the two. The function does not modify this array.
    
    Behavior and numerical details:
        The function computes the standard angle between two vectors using the dot product formula: angle = arccos((v1 . v2) / (|v1| |v2|)). The returned angle is in radians and lies in the closed interval [0, pi], where 0 indicates parallel vectors pointing in the same direction and pi indicates vectors pointing in opposite directions.
        To reduce domain errors from floating-point inaccuracy, the intermediate cosine-like term is compared to 1.0 and -1.0: values >= 1.0 yield a returned angle of 0.0; values <= -1.0 yield a returned angle of pi. The function first checks exact element-wise equality of vector1 and vector2 and returns 0.0 without further computation if they are identical.
        This function uses numpy.dot and numpy.linalg.norm internally. It does not perform tolerance-based comparisons (for near-equality use numpy.allclose externally before calling if required).
    
    Failure modes and edge cases:
        If either input has zero magnitude (a zero-length vector), the denominator in the cosine calculation is zero and the result is undefined; depending on NumPy's floating-point behavior this can produce NaN or, in some pathological division-by-zero cases, an arithmetic value that triggers the >=1 or <=-1 clamps. Therefore, callers should ensure vectors have non-zero norm when a defined finite angle is required.
        The identical-vector check uses exact equality (numpy.equal) and therefore will not detect vectors that are effectively equal within floating-point tolerance; use numpy.allclose externally to handle almost-equal vectors if needed.
    
    Side effects:
        None. The function does not mutate its inputs and has no external side effects.
    
    Returns:
        float: The angle between vector1 and vector2 in radians. The value is in [0, pi]. If inputs are invalid (for example, include zero-length vectors) the function may return numpy.nan or a floating value determined by NumPy's arithmetic rules; callers should validate inputs when a finite numeric result is required.
    """
    from stk.utilities.utilities import vector_angle
    return vector_angle(vector1, vector2)


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
