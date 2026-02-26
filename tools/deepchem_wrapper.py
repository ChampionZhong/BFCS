"""
Regenerated Google-style docstrings for module 'deepchem'.
README source: others/readme/deepchem/README.md
Generated at: 2025-12-02T02:48:23.065010Z

Total functions: 128
"""


import numpy
import torch
from typing import List

################################################################################
# Source: deepchem.data.datasets.densify_features
# File: deepchem/data/datasets.py
# Category: valid
################################################################################

def deepchem_data_datasets_densify_features(X_sparse: numpy.ndarray, num_features: int):
    """Expands a sparse feature representation into a dense 2-D numpy array suitable for model input.
    
    This function is used in DeepChem data pipelines to reconstruct a dense feature matrix from a compact sparse representation that stores only nonzero feature indices and values per sample. In the context of molecular machine learning (see DeepChem README), this is typically used to convert sparse fingerprints, bag-of-words style encodings, or other sparse per-example feature encodings into a dense (n_samples, num_features) array that can be passed to models implemented with TensorFlow, PyTorch, or JAX. The function assumes the sparse representation came from a 2-D array of shape (n_samples, num_features) and therefore does not support reconstructing higher-dimensional dense arrays.
    
    Args:
        X_sparse (numpy.ndarray): 1-D numpy array of length n_samples with dtype=object. Each element X_sparse[i] must be a tuple (nonzero_indices, nonzero_values) describing the nonzero entries for sample i. nonzero_indices is an array-like of integer indices into the feature axis (0 .. num_features-1) and nonzero_values is an array-like of numeric values for those indices. The tuple contents are used directly to set values in the returned dense array; X_sparse itself is not modified. This parameter represents the compact sparse per-sample feature encoding produced earlier in a DeepChem data preprocessing step.
        num_features (int): The number of feature columns in the reconstructed dense array. This must be the same number that was used when the sparse representation was created (the original dense array had shape (n_samples, num_features)). This integer controls the width of the returned array and must be a non-negative integer; negative values are invalid.
    
    Returns:
        numpy.ndarray: A new 2-D numpy array of shape (n_samples, num_features) containing the dense feature values for all samples. The array is created with numpy.zeros and thus will have the default numpy floating dtype (e.g., float64) unless num_features is zero, in which case the array will have shape (n_samples, 0). Nonzero entries are filled by assigning nonzero_values at positions given by nonzero_indices for each sample. The returned array is a fresh copy (no in-place modification of X_sparse occurs) and is ready to be consumed by DeepChem models or downstream preprocessing.
    
    Behavior, side effects, and failure modes:
        - Memory: This function allocates O(n_samples * num_features) memory to store the dense array. For large n_samples or large num_features this allocation can be large and may lead to MemoryError.
        - Input validation: The function expects X_sparse to be a numpy.ndarray with dtype=object and each element to be a tuple (indices, values). If X_sparse does not meet these expectations, operations such as tuple unpacking or index-based assignment may raise TypeError or ValueError. The function does not perform deep validation of tuple contents.
        - Index bounds: If any index in nonzero_indices is outside the range [0, num_features-1], numpy will raise an IndexError during assignment.
        - Types: num_features must be an integer type compatible with numpy array shape semantics. Passing non-integer or negative values may raise TypeError or ValueError when constructing the output array.
        - Multidimensional features: The function does not support reconstructing original dense arrays that had per-feature multidimensional values (e.g., per-feature vectors). It assumes each feature is a scalar and the original dense array shape was (n_samples, num_features).
        - Determinism: Given the same X_sparse and num_features inputs, the function deterministically returns the same dense array.
    
    Practical significance:
        - Use this function when you have a sparse representation (per-sample index/value pairs) produced by DeepChem featurizers or custom preprocessing and you need a dense matrix for model training, evaluation, or exporting datasets to libraries that require dense arrays.
        - Be mindful of memory and index correctness when converting large sparse datasets back to dense form.
    """
    from deepchem.data.datasets import densify_features
    return densify_features(X_sparse, num_features)


################################################################################
# Source: deepchem.data.datasets.pad_batch
# File: deepchem/data/datasets.py
# Category: valid
################################################################################

def deepchem_data_datasets_pad_batch(
    batch_size: int,
    X_b: numpy.ndarray,
    y_b: numpy.ndarray,
    w_b: numpy.ndarray,
    ids_b: numpy.ndarray
):
    """Pads a minibatch of examples to exactly batch_size by repeating the provided examples in tiled fashion.
    
    This function is used by DeepChem data-loading and training code to ensure each minibatch passed to a model has a fixed size (batch_size) even when the dataset size or the last batch of an epoch is smaller. It takes arrays of features, labels, sample weights, and identifiers that all represent the same short batch (length <= batch_size) and produces new arrays of length batch_size suitable for input to neural network training or evaluation routines in molecular machine learning, drug discovery, materials science, and related life-science applications.
    
    Args:
        batch_size (int): The target number of datapoints required in the output batch. In training and evaluation loops this is the minibatch dimension expected by model code and loss computations. Must be a positive integer.
        X_b (numpy.ndarray): Array of input features for the current (possibly short) batch. Must have length len(X_b) <= batch_size and must share a consistent feature shape across examples. If X_b.ndim > 1 then the output feature array will have shape (batch_size,) + X_b.shape[1:], preserving the per-example feature shape and the numpy dtype of X_b. If X_b.ndim == 1 then the output feature array will have shape (batch_size,) and the same dtype as X_b. X_b provides the concrete example tensors that will be tiled into X_out to reach batch_size.
        y_b (numpy.ndarray): Array of labels/targets corresponding to X_b; must have length len(y_b) == len(X_b) and len(y_b) <= batch_size. If y_b is None, no label array will be produced (y_out will be None). If y_b is a 1-D array the returned y_out will have shape (batch_size,); if y_b has additional per-example dimensions (y_b.ndim >= 2) the returned y_out will have shape (batch_size,) + y_b.shape[1:], and dtype preserved. y_b supplies supervised targets used by loss functions and evaluation.
        w_b (numpy.ndarray): Array of sample weights corresponding to X_b; must have length len(w_b) == len(X_b) and len(w_b) <= batch_size. If w_b is None, w_out will be None. If w_b is 1-D the returned w_out will have shape (batch_size,); if w_b has extra per-example dimensions (w_b.ndim >= 2) the returned w_out will have shape (batch_size,) + w_b.shape[1:], and dtype preserved. Note: when padding occurs this implementation assigns the original w_b values only to the first len(X_b) positions of w_out (see behavior below); remaining weight entries remain zero.
        ids_b (numpy.ndarray): 1-D array of identifiers (e.g., example IDs, indices, or string/object dtype allowed by numpy) for the examples in X_b; must have length len(ids_b) == len(X_b) and len(ids_b) <= batch_size. ids_out will be a numpy array of length batch_size with the same dtype as ids_b; ids from ids_b are copied repeatedly to fill ids_out.
    
    Behavior and side effects:
        The function first inspects num_samples = len(X_b). If num_samples == batch_size it returns the inputs unchanged as a tuple (X_b, y_b, w_b, ids_b). Otherwise it constructs new zero-initialized output arrays with length batch_size and dtypes matching the inputs, preserving per-example shapes as described above. It fills X_out, y_out (if not None), and ids_out by repeatedly copying entries from the original arrays in a tiled fashion until batch_size entries are written. Sample weights are treated specially: if w_b is not None the original w_b values are copied into the first num_samples entries of w_out before tiling; w_out is not updated during the tiling loop, so padded positions retain their initial zero weight. This design implies that only the original examples contribute nonzero weight to losses computed with w_out, while the padded (repeated) examples have zero weight unless the caller uses a different weighting convention.
        The function preserves numpy dtypes from inputs when allocating outputs. The function does not perform extensive input validation: it assumes len(X_b) == len(y_b) == len(w_b) == len(ids_b) when arrays are provided, that len(X_b) > 0, and that all lengths are <= batch_size. It also assumes X_b has a consistent per-example feature shape. No additional type conversions are performed.
    
    Failure modes and caller responsibilities:
        If the input lengths are inconsistent (for example len(y_b) != len(X_b)) or any input length exceeds batch_size, the function’s behavior is undefined and will likely produce incorrect results. If len(X_b) == 0 (an empty batch) the function will enter an infinite loop because the implementation uses num_samples to advance the fill pointer; therefore callers must ensure num_samples > 0. Callers should ensure inputs are valid and non-empty before calling pad_batch.
    
    Returns:
        tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray): A 4-tuple (X_out, y_out, w_out, ids_out) where each element is either a numpy.ndarray of length batch_size or None (for y_out or w_out when the corresponding input was None). X_out contains features tiled to length batch_size and has shape and dtype as described in Args. y_out and w_out (when provided) contain labels and weights tiled or zero-padded according to the rules above; w_out will have original weights in its first len(X_b) positions and zeros elsewhere. ids_out contains identifiers tiled to length batch_size and has the same dtype as ids_b. These returned arrays are ready to be passed into DeepChem model training/evaluation pipelines that expect fixed-size minibatches.
    """
    from deepchem.data.datasets import pad_batch
    return pad_batch(batch_size, X_b, y_b, w_b, ids_b)


################################################################################
# Source: deepchem.data.datasets.pad_features
# File: deepchem/data/datasets.py
# Category: valid
################################################################################

def deepchem_data_datasets_pad_features(batch_size: int, X_b: numpy.ndarray):
    """Pads a batch of features to exactly the requested batch size by repeating (tiling) the input examples.
    
    This function is used in DeepChem's data pipeline for inference-time query processing when a model or runtime requires a fixed batch size. Given an input array of feature vectors X_b whose length is less than or equal to batch_size, pad_features constructs and returns a new numpy.ndarray of length exactly batch_size by tiling the rows (or elements) of X_b in order until the batch is full. The output preserves the dtype of X_b and the per-sample feature shape: if X_b is 1-D with shape (N,), the result has shape (batch_size,); if X_b is multi-dimensional with shape (N, ...) the result has shape (batch_size, ...). This function performs no operation on labels or weights and is intended for features-only padding (see similar utilities such as pad_batch for label/weight handling).
    
    Args:
        batch_size (int): The target number of datapoints in the returned batch. This controls the first dimension of the returned array. Must be a positive integer and must be greater than or equal to the number of samples in X_b. If batch_size is smaller than the number of input samples, the function will raise a ValueError (see Raises:).
        X_b (numpy.ndarray): A numpy array of feature examples to be padded. X_b must satisfy 1 <= len(X_b) <= batch_size. Each element/row of X_b represents a single example's features in the domain (for example, molecular descriptors or fingerprint vectors in DeepChem workflows). The function preserves X_b.dtype and the per-example feature shape (the trailing dimensions of X_b). X_b is not modified in-place; a new array is returned.
    
    Returns:
        X_out (numpy.ndarray): A numpy array with length exactly batch_size along the first dimension. If X_b is 1-D, X_out.shape == (batch_size,); if X_b is multi-dimensional with shape (N, ...), then X_out.shape == (batch_size, ...). X_out is produced by repeating the elements/rows of X_b in order (tiling) until batch_size entries are filled; when the remaining space is smaller than len(X_b), a prefix of X_b is copied to finish the batch. The dtype of X_out matches X_b.dtype. This returned array is suitable for immediate use as an input feature batch to DeepChem models that require fixed-size batches during inference.
    
    Raises:
        ValueError: If len(X_b) > batch_size, since the function cannot pad to a smaller target and the implementation explicitly checks for this condition and raises a ValueError.
        RuntimeError: The implementation assumes len(X_b) >= 1. If len(X_b) == 0 the function's loop will not progress (division of work by zero-length sample blocks) and the behavior is undefined; callers must ensure X_b contains at least one sample prior to calling pad_features.
    
    Notes:
        - This function is intended for inference-time feature padding in DeepChem workflows (e.g., preparing a small set of molecular features to match a model's expected minibatch size for GPU inference).
        - The function does not attempt to pad labels or sample weights; use higher-level utilities if labels/weights need synchronized padding.
        - No external side effects occur: X_b is not modified and a new numpy.ndarray is returned.
    """
    from deepchem.data.datasets import pad_features
    return pad_features(batch_size, X_b)


################################################################################
# Source: deepchem.data.datasets.sparsify_features
# File: deepchem/data/datasets.py
# Category: valid
################################################################################

def deepchem_data_datasets_sparsify_features(X: numpy.ndarray):
    """Extracts a sparse feature representation from a dense feature array used in DeepChem preprocessing.
    
    This function is used in DeepChem to convert dense per-sample feature vectors (for example, molecular fingerprints, descriptor vectors, or other per-compound feature arrays commonly encountered in drug discovery and materials science workflows) into a compact sparse representation. The sparse representation stores, for each sample, the indices of nonzero features and the corresponding nonzero values. This reduces memory and computational overhead when many features are zero and downstream code expects or can exploit sparse inputs.
    
    Args:
        X (numpy.ndarray): A numpy array of shape `(n_samples, ...)` containing dense per-sample features. In typical DeepChem usage this is a 2-D array with shape `(n_samples, n_features)` where each row is a feature vector for one sample (molecule, material, or biological example). The function iterates over the first axis (samples) and for each sample `X[i]` finds the nonzero entries using NumPy's nonzero semantics. If `X` is empty (length zero) the function returns an empty object-dtype array. If `X` is not a NumPy array, a TypeError or an attribute error may be raised by NumPy operations; if per-sample entries are higher-dimensional arrays the function uses `np.nonzero(X[i])[0]` and then indexes `X[i]` with those indices (behavior for multi-dimensional per-sample arrays is therefore the same as NumPy indexing along the first feature axis and may produce arrays with additional dimensions).
    
    Returns:
        numpy.ndarray: A 1-D numpy array with dtype=object and length `n_samples`. Each element `X_sparse[i]` is a tuple `(nonzero_inds, nonzero_vals)` where `nonzero_inds` is a 1-D numpy.ndarray of integer indices corresponding to positions along the sample's first feature axis that have nonzero values, and `nonzero_vals` is a 1-D numpy.ndarray of the same length containing the values of `X[i]` at those indices. This object-dtype array is produced by collecting per-sample tuples in a Python list and converting to `np.array(..., dtype=object)`. There are no in-place side effects on the input `X`; the function returns a new array. Possible failure modes include exceptions raised by NumPy if `X` is not array-like, if indexing semantics are incompatible with the per-sample array shapes, or if memory allocation for the returned object array fails for very large `n_samples`. Computational cost is proportional to scanning all elements of `X` (approximately O(total number of elements) in practice).
    """
    from deepchem.data.datasets import sparsify_features
    return sparsify_features(X)


################################################################################
# Source: deepchem.dock.binding_pocket.extract_active_site
# File: deepchem/dock/binding_pocket.py
# Category: valid
################################################################################

def deepchem_dock_binding_pocket_extract_active_site(
    protein_file: str,
    ligand_file: str,
    cutoff: float = 4.0
):
    """Extracts an axis-aligned integer bounding box that encloses the protein active site (binding pocket)
    and returns the coordinates of the protein atoms that define that pocket. This function is used in
    the docking/featurization workflow in DeepChem to identify the region of a protein near a bound
    ligand that should be considered for grid-based featurization, docking, or other local analyses.
    
    Behavior: The function loads the protein and ligand from the provided file paths, loads the ligand
    with added hydrogens and computed partial charges, finds protein atoms that are within the given
    cutoff distance (in angstroms) of the ligand, and computes integer axis bounds by taking the floor
    of minima and the ceil of maxima of the pocket atom coordinates. It returns a CoordinateBox that
    covers the pocket region and a numpy.ndarray of the pocket atom coordinates in angstroms. The
    returned CoordinateBox uses integer bounding values computed from the floating point coordinates
    so it is suitable for creating voxel grids or integer-grid featurizers.
    
    Side effects and processing details: Calls load_molecule on each input file; ligand_file is loaded
    with add_hydrogens=True and calc_charges=True, while protein_file is loaded with add_hydrogens=False.
    Then calls get_contact_atom_indices with the specified cutoff to determine which protein atoms contact
    the ligand. The coordinate arrays used are taken directly from the loaded protein coordinates (units:
    angstroms). The box bounds are computed with numpy floor/ceil and converted to Python ints.
    
    Failure modes and errors: If the input files cannot be read or parsed (e.g., missing file, invalid PDB/
    ligand format, or RDKit parsing errors), load_molecule will raise an I/O or parsing-related exception.
    If no protein atoms are found within the cutoff distance, pocket coordinate operations (np.amin or
    np.amax) will raise a ValueError; callers should catch this case if an empty pocket is possible.
    The function does not validate that the ligand is actually bound; it only finds protein atoms within
    cutoff angstroms of ligand atoms.
    
    Args:
        protein_file (str): Path to the protein structure file (typically a PDB). This argument specifies
            the file location from which the protein is parsed for atomic coordinates. In DeepChem's
            docking workflow, this protein is treated without automatically adding hydrogens (add_hydrogens=False)
            so input protonation should be provided if required by downstream steps.
        ligand_file (str): Path to the ligand structure file (e.g., SDF or MOL2). This ligand is loaded
            with hydrogens added (add_hydrogens=True) and partial charges computed (calc_charges=True),
            because the ligand's explicit hydrogens and charges are often required to determine contact
            atoms and subsequent featurization in docking pipelines.
        cutoff (float): Distance threshold in angstroms (Å) used to define contact between protein and
            ligand atoms. Protein atoms whose Euclidean distance to any ligand atom is less than or equal
            to this cutoff are considered part of the active site/pocket. Default is 4.0 Å, a typical
            radius used in featurization to capture first-shell interactions.
    
    Returns:
        Tuple[deepchem.utils.coordinate_box_utils.CoordinateBox, numpy.ndarray]: A two-element tuple where
        the first element is a deepchem.utils.coordinate_box_utils.CoordinateBox describing integer
        axis-aligned bounds that enclose the active site. The bounds are computed by taking the floor of
        the minimum x, y, z coordinates and the ceil of the maximum x, y, z coordinates of the pocket
        atoms, then converting those values to integers; this makes the box suitable for integer-grid
        featurizers and voxel generation. The second element is a numpy.ndarray of shape (N, 3) containing
        the floating-point (x, y, z) coordinates in angstroms of the N protein atoms identified as the
        active site (i.e., those within the cutoff of the ligand). N is the number of contacting protein atoms;
        if N == 0, coordinate reduction operations will fail and an exception is raised.
    """
    from deepchem.dock.binding_pocket import extract_active_site
    return extract_active_site(protein_file, ligand_file, cutoff)


################################################################################
# Source: deepchem.dock.pose_scoring.cutoff_filter
# File: deepchem/dock/pose_scoring.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def deepchem_dock_pose_scoring_cutoff_filter(
    d: numpy.ndarray,
    x: numpy.ndarray,
    cutoff: float = 8.0
):
    """deepchem.dock.pose_scoring.cutoff_filter applies a distance-based cutoff to a pairwise feature matrix used in molecular docking and pose scoring. In docking workflows within DeepChem, d typically encodes pairwise distances (for example atom–atom distances in Angstroms) and x encodes the corresponding per-pair contributions (for example interaction energies, contact indicators, or other features). This function zeroes out entries in x whose corresponding distance in d is greater than or equal to the cutoff, producing a filtered (N, M) array suitable for downstream scoring or featurization.
    
    Args:
        d (numpy.ndarray): Pairwise distances matrix. A numpy array of shape (N, M) containing distances in Angstroms between two sets of atomic or molecular elements. Each element d[i, j] is the distance used to decide whether the corresponding value x[i, j] is kept. The function expects d to be a numpy.ndarray; if d does not have a shape compatible with x, numpy broadcasting rules apply and a ValueError may be raised.
        x (numpy.ndarray): Matrix of shape (N, M) containing values associated with each pairwise distance in d (for example interaction scores, weights, or binary contact indicators). Values in x are retained only when the corresponding entry in d is strictly less than the cutoff. The function does not modify x in place; it constructs and returns a new numpy.ndarray with the same shape and dtype as x.
        cutoff (float): Cutoff distance in Angstroms (default 8.0). Pairs with d[i, j] < cutoff are considered proximal and their corresponding x[i, j] values are preserved; pairs with d[i, j] >= cutoff are considered too distant and will be replaced with zeros. The comparison is strict (<), so distances exactly equal to cutoff are treated as too large and thresholded to zero.
    
    Returns:
        numpy.ndarray: A new numpy.ndarray of shape (N, M) with the same dtype as x where entries x[i, j] are retained when d[i, j] < cutoff and set to 0 where d[i, j] >= cutoff. This returned array is safe to use in downstream docking pose scoring, aggregation, or featurization steps. No in-place side effects occur on the input arrays.
    
    Behavior, side effects, and failure modes: The function performs a vectorized numpy.where operation and is O(N*M) in time with respect to the input size. It preserves the dtype and shape of x by using np.zeros_like(x) for the masked values. If d and x have incompatible shapes that cannot be broadcast together, numpy will raise a ValueError. If non-numeric types are passed for cutoff or the arrays cannot be interpreted as numeric numpy arrays, a TypeError or other numpy-related exception may be raised. The function is deterministic and has no other side effects.
    """
    from deepchem.dock.pose_scoring import cutoff_filter
    return cutoff_filter(d, x, cutoff)


################################################################################
# Source: deepchem.dock.pose_scoring.pairwise_distances
# File: deepchem/dock/pose_scoring.py
# Category: valid
################################################################################

def deepchem_dock_pose_scoring_pairwise_distances(
    coords1: numpy.ndarray,
    coords2: numpy.ndarray
):
    """Compute the matrix of pairwise Euclidean distances between two sets of 3D coordinates. This function is used in DeepChem's docking and pose-scoring workflows to measure inter-point distances (for example, distances between atoms in a ligand and atoms in a receptor), to build contact maps, apply distance-based cutoffs, or supply distance features to scoring functions.
    
    Args:
        coords1 (np.ndarray): A NumPy array of shape (N, 3) representing N points in 3D Cartesian coordinates. In the docking/pose-scoring domain, this typically represents one set of atomic coordinates (for example, all atoms of a ligand or a receptor region). The function treats each row coords1[i] as the i-th 3D point and computes distances from that point to every point in coords2.
        coords2 (np.ndarray): A NumPy array of shape (M, 3) representing M points in 3D Cartesian coordinates. In the docking/pose-scoring domain, this typically represents a second set of atomic coordinates (for example, atoms of the complementary molecule or binding site). The function treats each row coords2[j] as the j-th 3D point.
    
    Returns:
        np.ndarray: A NumPy array of shape (N, M) whose element [i, j] is the Euclidean distance between coords1[i] and coords2[j]. Distances are computed as the square root of the summed squared differences along the last coordinate axis and therefore are floating-point values. The returned array is allocated in memory and can be large (O(N*M) memory); callers should be aware of this for large N and M.
    
    Behavior and failure modes:
        The implementation is vectorized using NumPy broadcasting and does not modify coords1 or coords2 in-place. The function assumes both inputs are two-dimensional numpy.ndarray objects with final dimension size 3; supplying arrays with incompatible shapes (for example wrong number of dimensions or a last dimension not equal to 3) will result in incorrect results or a NumPy broadcasting/indexing error at runtime. The function does not perform explicit type coercion or validation beyond what NumPy naturally enforces; passing non-numpy inputs may raise exceptions from NumPy operations. There are no side effects other than allocating and returning the distance matrix.
    """
    from deepchem.dock.pose_scoring import pairwise_distances
    return pairwise_distances(coords1, coords2)


################################################################################
# Source: deepchem.dock.pose_scoring.vina_energy_term
# File: deepchem/dock/pose_scoring.py
# Category: valid
################################################################################

def deepchem_dock_pose_scoring_vina_energy_term(
    coords1: numpy.ndarray,
    coords2: numpy.ndarray,
    weights: numpy.ndarray,
    wrot: float,
    Nrot: int
):
    """Computes the AutoDock Vina-style energy score for two molecular conformations and returns the summed free energy used to rank docking poses in DeepChem's docking/pose-scoring workflow.
    
    This function is used in DeepChem's docking pipeline to evaluate the energetic compatibility between two sets of 3D coordinates (for example, a ligand pose and a receptor pocket or two molecular conformations). It computes pairwise inter-atomic distances, evaluates five interaction kernels (repulsion, hydrophobic, hydrogen-bonding, and two Gaussian attraction terms), forms a weighted linear combination of these kernels using the provided weights array, applies a distance cutoff filter, and then applies a nonlinearity that depends on the provided rotatable-bond penalty parameters. The final output is the sum of per-pair free-energy contributions computed by vina_nonlinearity and is typically used to rank or score docking poses.
    
    Args:
        coords1 (numpy.ndarray): Molecular coordinates for the first conformation. Must be a 2-D array of shape (N, 3) where N is the number of atoms/points in the first structure and each row is a 3D Cartesian coordinate. These coordinates are used to compute pairwise distances to coords2.
        coords2 (numpy.ndarray): Molecular coordinates for the second conformation. Must be a 2-D array of shape (M, 3) where M is the number of atoms/points in the second structure and each row is a 3D Cartesian coordinate. These coordinates are used to compute pairwise distances to coords1.
        weights (numpy.ndarray): A 1-D numpy array of shape (5,) containing the linear weights applied to the five interaction components in this order: repulsion interaction term, hydrophobic interaction term, hydrogen-bond interaction term, first Gaussian interaction term, and second Gaussian interaction term. The numeric values of these five weights control the relative contribution of each interaction kernel to the final energy; supplying an incorrectly sized array will lead to an error.
        wrot (float): A scalar scaling factor passed to the vina_nonlinearity stage that controls the strength of the nonlinear rotatable-bond penalty. In practice this modifies how per-pair interaction contributions are transformed prior to summation and is used to approximate entropic/rotational penalties in pose scoring.
        Nrot (int): The number of rotatable bonds considered in this scoring calculation. This integer is forwarded to vina_nonlinearity and influences the rotatable-bond penalty applied to the computed per-pair interaction energies.
    
    Returns:
        numpy.ndarray: A scalar numpy.ndarray containing the summed free energy computed for the two conformations. The value is computed by summing the per-pair energies after applying the weighted sum of interaction kernels, a distance cutoff filter, and the vina_nonlinearity transformation using wrot and Nrot. If input shapes are incompatible (for example, coords1 does not have shape (N, 3) or coords2 does not have shape (M, 3), or weights does not have shape (5,)), the function will raise an error originating from the downstream distance or array operations. There are no external side effects; the function is deterministic for given inputs.
    """
    from deepchem.dock.pose_scoring import vina_energy_term
    return vina_energy_term(coords1, coords2, weights, wrot, Nrot)


################################################################################
# Source: deepchem.dock.pose_scoring.vina_gaussian_first
# File: deepchem/dock/pose_scoring.py
# Category: valid
################################################################################

def deepchem_dock_pose_scoring_vina_gaussian_first(d: numpy.ndarray):
    """Computes Autodock Vina's first Gaussian interaction term.
    
    This function implements the first Gaussian term from the Autodock Vina scoring function (Jain, 1996) that is commonly used in molecular docking to estimate a component of protein–ligand binding affinity. The computation is performed elementwise as exp(-(d / 0.5)**2), where the constant 0.5 is the Gaussian width parameter used by Autodock Vina. In practical workflows within DeepChem, this function converts a matrix of surface distances between ligand and protein atoms or grid points into a matrix of Gaussian-shaped interaction contributions that can be summed or combined with other terms to produce a docking score.
    
    Args:
        d (np.ndarray): A numpy array of shape `(N, M)` containing surface distances as defined in Jain 1996. Each element represents a distance measure used by the Vina scoring model (typically non-negative distances between ligand and protein surface points). The array should have a numeric dtype (e.g., float32 or float64). The function expects a full (N, M) array; providing inputs of a different shape will result in an output whose shape follows numpy's broadcasting rules or may raise an exception if broadcasting is not possible.
    
    Returns:
        np.ndarray: A numpy array of shape `(N, M)` containing the Gaussian interaction terms computed elementwise as exp(-(d / 0.5)**2). The returned array contains the same number of rows and columns as the input and typically has a floating-point dtype. This array represents the per-distance contribution of the first Gaussian term to the overall Autodock Vina score and can be summed across atoms or grid points in downstream scoring calculations.
    
    Behavior, side effects, and failure modes:
        This function is pure and has no side effects; it does not modify the input array in-place. If the input `d` is not a numpy ndarray, numpy operations will raise a TypeError or cause implicit conversion. If `d` contains NaN or infinite values, those values will propagate through the exponential and appear in the output. Extremely large magnitude values in `d` can underflow to zero or result in floating-point behavior consistent with numpy's exp. The function uses the fixed width parameter 0.5 from the original Vina formulation; changing that parameter requires modifying the implementation. Reference: Jain, A. N., "Scoring noncovalent protein-ligand interactions: a continuous differentiable function tuned to compute binding affinities." Journal of computer-aided molecular design 10.5 (1996): 427-440.
    """
    from deepchem.dock.pose_scoring import vina_gaussian_first
    return vina_gaussian_first(d)


################################################################################
# Source: deepchem.dock.pose_scoring.vina_gaussian_second
# File: deepchem/dock/pose_scoring.py
# Category: valid
################################################################################

def deepchem_dock_pose_scoring_vina_gaussian_second(d: numpy.ndarray):
    """Computes Autodock Vina's second Gaussian interaction term used in docking pose scoring.
    
    This function implements the second Gaussian term from the Autodock Vina scoring function (Jain 1996) and is intended for use within DeepChem's docking and pose_scoring utilities to model a short-range attractive component of protein-ligand noncovalent interactions. Given an array of surface distances d (as defined in the Vina formulation), the function applies the elementwise transformation exp(-((d - 3) / 2)**2) to produce the corresponding Gaussian interaction contributions that are later combined with other scoring terms to estimate binding affinity.
    
    Args:
        d (numpy.ndarray): A numpy array of shape `(N, M)` containing the set of surface distances for which the second Gaussian interaction term should be computed. Each element represents a distance value used in the Autodock Vina continuous scoring function (see Jain, 1996). The array must be a numpy.ndarray as required by the implementation; the function performs the calculation elementwise and does not perform explicit validation of contents or shape beyond relying on numpy operations.
    
    Returns:
        numpy.ndarray: A numpy array of shape `(N, M)` containing the Gaussian interaction terms computed elementwise as exp(-((d - 3) / 2)**2). The output array has the same shape as the input d and is intended to be used as the second Gaussian term in Autodock Vina-style scoring for protein-ligand binding affinity estimation.
    
    Behavior and failure modes:
        This function is a pure, side-effect-free computation implemented with NumPy operations. It does not modify its input in-place. If d is not a numpy.ndarray, NumPy may attempt to coerce it and may raise a TypeError or produce unexpected results; callers should provide a numpy.ndarray to match the documented signature. If d contains non-finite values (NaN or infinity), the corresponding output entries will follow NumPy semantics and propagate NaN or infinity. This routine does not perform clipping or additional numerical checks; it solely evaluates the mathematical expression used in Autodock Vina's second Gaussian term.
    
    References:
        Jain, Ajay N. "Scoring noncovalent protein-ligand interactions: a continuous differentiable function tuned to compute binding affinities." Journal of computer-aided molecular design 10.5 (1996): 427-440.
    """
    from deepchem.dock.pose_scoring import vina_gaussian_second
    return vina_gaussian_second(d)


################################################################################
# Source: deepchem.dock.pose_scoring.vina_hbond
# File: deepchem/dock/pose_scoring.py
# Category: valid
################################################################################

def deepchem_dock_pose_scoring_vina_hbond(d: numpy.ndarray):
    """Computes AutoDock Vina's hydrogen bond interaction term for docking pose scoring.
    
    This function implements the piecewise linear hydrogen-bond term used by AutoDock Vina (Jain 1996) to convert inter-surface distances into a normalized interaction contribution used in scoring protein-ligand poses. In DeepChem's docking/pose-scoring pipeline this term is one component of the overall Vina-style scoring function and is applied to the set of surface distances computed between interacting atoms or molecular surfaces. The input d is expected to contain the surface distances as defined in the cited reference: negative values indicate overlapping/close contact (favorable for hydrogen bonding), and non-negative values indicate separated surfaces (no hydrogen-bond contribution).
    
    Behavior: for each element of d the function returns a value in the interval [0, 1] according to the following piecewise definition implemented in the source code:
    - If d < -0.7, the interaction term is saturated at 1.
    - If -0.7 <= d < 0, the interaction term increases linearly as (1.0 / 0.7) * (0 - d) (equivalently -d / 0.7), producing a smooth ramp from 1 at d = -0.7 to 0 at d = 0.
    - If d >= 0, the interaction term is 0.
    
    The function performs no in-place modification of its input and has no side effects beyond returning the computed array.
    
    Args:
        d (numpy.ndarray): A numpy array of surface distances with shape (N, M). Each element is a numeric surface distance as used in AutoDock Vina-style scoring (negative values denote close contact/overlap, non-negative values denote separation). The array must be a numpy.ndarray; values should be numeric (float or integer). The function expects the distances arranged so that corresponding rows/columns map to the pairs of interacting entities used elsewhere in the DeepChem docking pipeline.
    
    Returns:
        numpy.ndarray: A numpy array of shape (N, M) with the hydrogen-bond interaction term for each corresponding entry of d. Values are in the closed interval [0, 1], following the piecewise linear mapping described above. The returned array has the same shape as the input and is intended to be combined with other Vina-style terms when computing a final docking score.
    
    Failure modes and notes:
    - If d is not a numpy.ndarray, NumPy operations may raise a TypeError or cause implicit conversion; callers in the DeepChem docking pipeline should pass a numpy.ndarray to avoid such errors.
    - If d contains NaN or infinite values, NumPy comparison semantics apply and the output may contain unexpected values (e.g., NaNs can cause comparisons to be False, leading to 0 in the >=0 branch); callers should sanitize inputs if NaNs/Infs are possible.
    - Boundary behavior is continuous: d == -0.7 yields 1.0 (via the linear branch) and d == 0 yields 0.0.
    - This function implements only the hydrogen-bond term from the Vina scoring functional (Jain 1996) and should be used as part of the full scoring pipeline to estimate binding affinity contributions from hydrogen-bond-like contacts.
    
    Reference: Jain, A. N. "Scoring noncovalent protein-ligand interactions: a continuous differentiable function tuned to compute binding affinities." Journal of Computer-Aided Molecular Design 10.5 (1996): 427-440.
    """
    from deepchem.dock.pose_scoring import vina_hbond
    return vina_hbond(d)


################################################################################
# Source: deepchem.dock.pose_scoring.vina_hydrophobic
# File: deepchem/dock/pose_scoring.py
# Category: valid
################################################################################

def deepchem_dock_pose_scoring_vina_hydrophobic(d: numpy.ndarray):
    """deepchem.dock.pose_scoring.vina_hydrophobic computes the Autodock Vina hydrophobic interaction term for a batch of surface-distance values used in molecular docking pose scoring.
    
    This function implements the piecewise linear hydrophobic term used in Vina-style scoring functions to approximate the contribution of close hydrophobic contacts between a ligand and a receptor to binding affinity. In the DeepChem docking/pose-scoring context (see the project README for the library's application to drug discovery and virtual screening), this term is applied elementwise to a matrix of surface distances to produce a matrix of hydrophobic interaction scores that can be combined with other terms for pose ranking. The distance matrix d is the set of surface distances as defined in the original Vina/Jain formulation [1].
    
    Args:
        d (numpy.ndarray): A numpy array of shape `(N, M)` containing surface-distance values for N examples and M distance measurements per example. Each element represents a surface distance (numeric value) as defined in Jain [1]. The function expects a numpy.ndarray; the implementation uses vectorized comparisons and returns an array of the same shape. The input is not modified in-place.
    
    Behavior:
        For each element x in d the output value y is computed by the following piecewise linear rule:
        if x < 0.5 then y = 1
        else if 0.5 <= x < 1.5 then y = 1.5 - x
        else y = 0
        The implementation uses numpy.where to apply these conditions elementwise, matching the original Vina hydrophobic term. For boundary values, x == 0.5 yields y = 1.0 and x == 1.5 yields y = 0. The function is fully vectorized and preserves the input array's shape. The output dtype is determined by numpy.ones_like / zeros_like behavior and will typically match or be compatible with the input dtype.
    
    Side effects and defaults:
        The function has no side effects: it does not modify the input array and produces a new numpy.ndarray. There are no optional parameters or defaults beyond the single required input array.
    
    Failure modes and special cases:
        If d is not a numpy.ndarray, numpy operations may raise a TypeError or produce unexpected results; callers should supply a numpy.ndarray as required. Non-numeric or object-dtype arrays may lead to exceptions during the numeric comparisons. NaN values in d cause both comparisons (d < 0.5 and d < 1.5) to evaluate as False and therefore yield 0 at those positions in the output. Infinite values are handled according to numpy comparison semantics (e.g., large positive infinities will map to 0).
    
    Returns:
        numpy.ndarray: A `(N, M)` numpy array containing the computed hydrophobic interaction term for each corresponding element of d, following the Vina piecewise linear formula above. The returned array is a new array (input is not modified) and is intended to be used as one component of a docking scoring function for pose ranking and virtual screening, as described in Jain [1].
    
    References:
        [1] Jain, Ajay N. "Scoring noncovalent protein-ligand interactions: a continuous differentiable function tuned to compute binding affinities." Journal of computer-aided molecular design 10.5 (1996): 427-440.
    """
    from deepchem.dock.pose_scoring import vina_hydrophobic
    return vina_hydrophobic(d)


################################################################################
# Source: deepchem.dock.pose_scoring.vina_nonlinearity
# File: deepchem/dock/pose_scoring.py
# Category: valid
################################################################################

def deepchem_dock_pose_scoring_vina_nonlinearity(c: numpy.ndarray, w: float, Nrot: int):
    """deepchem.dock.pose_scoring.vina_nonlinearity: Compute a Vina-inspired nonlinearity used in docking pose scoring to attenuate per-activation values according to molecular flexibility.
    
    This function implements a simple, elementwise nonlinearity used in DeepChem's docking/pose-scoring utilities. It divides every element of an input activation matrix by a scalar factor (1 + w * Nrot). In the context of molecular docking and drug-discovery workflows (see DeepChem README), this nonlinearity is used to penalize or scale pose scoring activations based on the number of rotatable bonds in a ligand: as the number of rotatable bonds increases, the denominator increases (for positive w) and activations are correspondingly reduced, reflecting increased conformational entropy and flexibility.
    
    Args:
        c (numpy.ndarray): A numpy array of shape (N, M) containing input activations or scores for N examples and M channels/features. Each element is treated as a numeric activation value and will be divided by the computed scalar factor. The input array is not modified in-place; a new array is returned.
        w (float): Weighting term used to scale the contribution of Nrot in the denominator. The scalar factor applied to c is (1 + w * Nrot). Practically, w controls how strongly the number of rotatable bonds attenuates the activations: larger positive w increases attenuation. The function accepts any Python float; incorrect (non-numeric) values will result in numpy type errors when dividing.
        Nrot (int): Number of rotatable bonds in the molecule (an integer count, typically >= 0). This integer is multiplied by w to form the denominator adjustment term. Passing a non-integer value is not recommended and may produce unexpected results or numpy type coercion; the function signature and intended domain semantics expect an integer count.
    
    Returns:
        numpy.ndarray: A numpy.ndarray of shape (N, M) containing the transformed activations computed elementwise as c / (1 + w * Nrot). The returned array preserves the input shape and dtype semantics where possible; the operation is elementwise division by a scalar denominator.
    
    Behavior and failure modes:
        - The computation performed is out_tensor = c / (1 + w * Nrot). Because the denominator is a scalar, the division broadcasts across the entire array.
        - The function has no side effects: it does not modify the input array in-place and returns a new numpy.ndarray.
        - If 1 + w * Nrot == 0 (for example, w is negative and exactly equals -1/Nrot), numpy will produce infinities and runtime warnings for division by zero; results may contain inf or nan values depending on c. Users should ensure the denominator is non-zero for stable numeric results.
        - If c contains non-numeric dtypes or if w/Nrot are incompatible with numeric operations, numpy will raise appropriate TypeError or ValueError exceptions.
        - This function is deterministic and stateless; it is intended to be used as a lightweight post-processing/nonlinearity step in pose scoring pipelines within DeepChem's docking utilities to incorporate a simple flexibility-based penalty.
    """
    from deepchem.dock.pose_scoring import vina_nonlinearity
    return vina_nonlinearity(c, w, Nrot)


################################################################################
# Source: deepchem.dock.pose_scoring.vina_repulsion
# File: deepchem/dock/pose_scoring.py
# Category: valid
################################################################################

def deepchem_dock_pose_scoring_vina_repulsion(d: numpy.ndarray):
    """deepchem.dock.pose_scoring.vina_repulsion computes the Autodock Vina repulsion interaction term used in molecular docking pose scoring. In the docking domain this term penalizes steric overlaps between atoms: negative values in the input indicate overlap (distance deficit) and produce a positive quadratic repulsion penalty, whereas non-overlapping separations produce no repulsion contribution.
    
    Args:
        d (numpy.ndarray): A numeric numpy array of shape `(N, M)` containing elementwise distance-like values used by the Vina scoring pipeline. In typical use within DeepChem's pose-scoring routines, each element represents a signed distance measure or deviation for an atom pair or grid point where negative values denote overlap/penetration and non-negative values denote non-overlap. The function performs elementwise operations and therefore expects a numeric dtype; supplying non-numeric data or incompatible objects may raise an exception during computation.
    
    Returns:
        numpy.ndarray: A `(N, M)` array of repulsion terms with the same shape as the input. For each element of `d`, the returned value is `d**2` when the input element is negative (representing a repulsive penalty for overlap) and `0` when the input element is non-negative. The dtype of the result is determined by NumPy arithmetic on the input. This function is pure (no side effects) and does not modify the input array; it raises standard NumPy errors if the input is not a numeric numpy.ndarray or contains values that cannot be squared.
    """
    from deepchem.dock.pose_scoring import vina_repulsion
    return vina_repulsion(d)


################################################################################
# Source: deepchem.dock.pose_scoring.weighted_linear_sum
# File: deepchem/dock/pose_scoring.py
# Category: valid
################################################################################

def deepchem_dock_pose_scoring_weighted_linear_sum(w: numpy.ndarray, x: numpy.ndarray):
    """deepchem.dock.pose_scoring.weighted_linear_sum computes a weighted linear sum by contracting a one-dimensional weight vector with the first axis of a three-dimensional feature tensor. In the DeepChem docking/pose-scoring context (drug discovery and molecular docking workflows described in the repository README), this function is used to combine N individual component scores or feature channels (for example, per-term energy contributions or per-feature model outputs) into final per-pose/per-output values that can be used for ranking or downstream evaluation.
    
    Args:
        w (np.ndarray): A 1-D numpy array of shape (N,) containing the weight for each of the N components or feature channels. In docking pipelines this represents the importance or coefficient applied to each score component. The function expects w to have length N that matches the first dimension of x; if shapes are incompatible a ValueError will be raised by the underlying numpy operation.
        x (np.ndarray): A 3-D numpy array of shape (N, M, L) containing N channels of values to be combined across M items (for example, M docking poses or M molecules) and L outputs per item (for example, L different target properties or replicate outputs). Each slice x[i, :, :] corresponds to the values contributed by the i-th component whose weight is given by w[i]. The function assumes this layout and will contract w with the first axis of x.
    
    Returns:
        np.ndarray: A 2-D numpy array of shape (M, L) produced by contracting the weight vector w with the first axis of x (implemented via numpy.tensordot with axes=1). The returned array contains the weighted sum across the N components for each of the M items and L outputs and is suitable for use as final pose scores or aggregated feature outputs in docking/scoring pipelines. The function has no side effects: it does not modify w or x and returns a newly allocated numpy array. Numerical dtype follows numpy's type promotion rules based on the dtypes of w and x. Failure modes include shape mismatches (e.g., w.shape[0] != x.shape[0]) or non-numpy inputs, which will raise exceptions from numpy (commonly ValueError or TypeError).
    """
    from deepchem.dock.pose_scoring import weighted_linear_sum
    return weighted_linear_sum(w, x)


################################################################################
# Source: deepchem.feat.complex_featurizers.complex_atomic_coordinates.compute_neighbor_list
# File: deepchem/feat/complex_featurizers/complex_atomic_coordinates.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def deepchem_feat_complex_featurizers_complex_atomic_coordinates_compute_neighbor_list(
    coords: numpy.ndarray,
    neighbor_cutoff: float,
    max_num_neighbors: int,
    periodic_box_size: list
):
    """Computes a neighbor list from atom coordinates.
    
    This function constructs a mapping from each atom index to a list of neighbor atom indices that lie within a spherical cutoff radius. It is used in DeepChem complex featurizers to identify local atomic environments for tasks in drug discovery, materials science, and computational chemistry. The implementation relies on the mdtraj library to compute neighbor relationships and applies an optional periodic minimum-image correction when a periodic_box_size is provided. The function reshapes the provided coordinate array into an mdtraj Trajectory of shape (1, N, 3) where N is the number of atoms and returns integer indices that refer to the original order of atoms in coords.
    
    Args:
        coords (numpy.ndarray): Array of atomic coordinates with shape (N, 3) where N is the number of atoms. Each row is the 3D Cartesian coordinates of one atom in the same units used for neighbor_cutoff. The function reshapes coords to (1, N, 3) internally and will raise an error if coords cannot be reshaped to match that layout. The ordering of coords defines the integer indices used in the returned neighbor lists.
        neighbor_cutoff (float): Cutoff radius (in the same distance units as coords) used to determine whether two atoms are neighbors. Atoms separated by a Euclidean distance less than or equal to neighbor_cutoff are considered neighbors by mdtraj. Values that are not finite or are negative will produce behavior or errors from the underlying mdtraj/numpy computations.
        max_num_neighbors (int): Maximum number of neighbors to keep for each atom. If set to an integer, when an atom has more neighbors within neighbor_cutoff than this value, the function selects the nearest max_num_neighbors neighbors by Euclidean distance (after applying periodic minimum-image correction if periodic_box_size is provided). If max_num_neighbors is None (the implementation checks for None), no truncation is applied and all neighbors within neighbor_cutoff are returned.
        periodic_box_size (list): Periodic box lengths provided as a length-3 sequence [Lx, Ly, Lz] (numeric values). If not None, the function configures the mdtraj Trajectory unit cell vectors from periodic_box_size and applies a minimum-image correction to distances when sorting/truncating neighbors. If periodic_box_size is None, periodic boundary conditions are not applied and distances are computed in free space. The provided list must have three elements corresponding to box lengths along the x, y, and z axes.
    
    Returns:
        dict: A dictionary mapping each atom index i (int, 0 <= i < N) to a list of neighbor atom indices (list of int). Each neighbor index refers to a row index in the input coords array. If max_num_neighbors is None, the list contains all neighbors within neighbor_cutoff found by mdtraj; if max_num_neighbors is an int and the number of raw neighbors exceeds it, the returned list contains the nearest max_num_neighbors indices sorted by (periodically-corrected) Euclidean distance.
    
    Raises and side effects:
        ImportError: Raised if the mdtraj library is not installed; this function requires mdtraj to compute neighbor lists.
        ValueError / IndexError: Errors may be raised by numpy or mdtraj if coords does not have a compatible shape (it must be reshaped to (1, N, 3)), if periodic_box_size is not length 3 when not None, or if invalid numeric values are supplied.
        Side effects: The function creates an mdtraj.Trajectory object and, when periodic_box_size is provided, sets traj.unitcell_vectors internally to represent the simulation cell. No global state outside the function is modified.
    
    Behavior notes:
        - Neighbor indices and distances are determined using Euclidean norm. When truncation is required, neighbors are chosen by ascending distance; ties are resolved by the sort order produced by Python's tuple sorting of (distance, index).
        - Units of neighbor_cutoff must match the units of coords.
        - Performance depends on mdtraj's neighbor search implementation; for large systems, runtime and memory usage may increase substantially.
    """
    from deepchem.feat.complex_featurizers.complex_atomic_coordinates import compute_neighbor_list
    return compute_neighbor_list(
        coords,
        neighbor_cutoff,
        max_num_neighbors,
        periodic_box_size
    )


################################################################################
# Source: deepchem.feat.complex_featurizers.splif_fingerprints.compute_splif_features_in_range
# File: deepchem/feat/complex_featurizers/splif_fingerprints.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for compute_splif_features_in_range because the docstring has no description for the argument 'contact_bin'
################################################################################

def deepchem_feat_complex_featurizers_splif_fingerprints_compute_splif_features_in_range(
    frag1: Tuple,
    frag2: Tuple,
    pairwise_distances: numpy.ndarray,
    contact_bin: List,
    ecfp_degree: int = 2
):
    """deepchem.feat.complex_featurizers.splif_fingerprints.compute_splif_features_in_range computes Structure-based Protein-Ligand Interaction Fingerprints (SPLIF) features for atom pairs in two molecular fragments whose pairwise distances fall strictly within a given range. This function is used in DeepChem to convert geometric proximity between two molecules (for example, a ligand and a protein residue or two fragments of a complex) into atom-level ECFP descriptors that serve as inputs for machine learning models in drug discovery and related molecular sciences.
    
    Args:
        frag1 (Tuple): A tuple of (coords, mol) returned by load_molecule. In practice, coords is the atomic coordinate array for the first fragment and mol is the corresponding RDKit molecule object. frag1 provides the source atoms whose local circular ECFP descriptors will be computed when they contact atoms of frag2.
        frag2 (Tuple): A tuple of (coords, mol) returned by load_molecule. coords is the atomic coordinate array for the second fragment and mol is the corresponding RDKit molecule object. frag2 provides the target atoms whose ECFP descriptors will be paired with frag1 descriptors for contacting atom pairs.
        pairwise_distances (numpy.ndarray): Array of pairwise fragment-fragment distances (Angstroms). This array encodes distances between atoms in frag1 and atoms in frag2; entries corresponding to a given atom pair are tested against contact_bin to decide contact membership. The function treats values in this array as distances in Angstrom units.
        contact_bin (List): Two-element list-like object specifying a strict distance range [lower, upper] where lower = contact_bin[0] and upper = contact_bin[1]. The function selects atom pairs where pairwise_distances > lower and pairwise_distances < upper (exclusive bounds). contact_bin therefore controls which atom pairs are considered “in contact” for SPLIF feature generation.
        ecfp_degree (int): ECFP radius used when computing circular fingerprints for atoms (default = 2). This integer is passed to compute_all_ecfp and controls the neighborhood radius considered when constructing each atom’s ECFP descriptor; larger values capture larger local chemical environments.
    
    Behavior, side effects, defaults, and failure modes:
        - The function identifies all atom-pair indices (i, j) for which the corresponding entry in pairwise_distances satisfies the strict inequality contact_bin[0] < distance < contact_bin[1]. The selection is exclusive of the endpoints.
        - For frag1, the function collects the unique frag1 atom indices that appear in any selected pair and computes ECFP descriptors only for those indices by calling compute_all_ecfp(frag1[1], indices=frag1_atoms, degree=ecfp_degree). For frag2, it computes ECFP descriptors for the provided frag2 molecule by calling compute_all_ecfp(frag2[1], degree=ecfp_degree). compute_all_ecfp is an external utility in the same module responsible for producing atom-level ECFP representations.
        - The function does not modify frag1, frag2, or pairwise_distances in place; it only reads them and returns a new dictionary.
        - Default behavior: ecfp_degree defaults to 2 if not provided, which is a commonly used local neighborhood radius in cheminformatics.
        - Error conditions: if contact_bin does not contain at least two elements, accessing contact_bin[0] or contact_bin[1] will raise an IndexError. If pairwise_distances is not shaped or indexed consistently with frag1 and frag2 atom indices, indexing or logical operations may raise exceptions (ValueError, IndexError, or TypeError). If compute_all_ecfp fails for a given molecule or index set, the exception from that function will propagate. If no atom pairs satisfy the distance test, the function will return an empty dictionary.
        - Units and domain: distances are interpreted in Angstroms; the produced ECFP descriptors are intended as molecular descriptors for downstream machine learning tasks in drug discovery, materials science, and related fields supported by DeepChem.
    
    Returns:
        Dict: A dictionary mapping tuples of integer atom indices to tuples of ECFP descriptors. Each key is (frag1_index_i, frag2_index_j) corresponding to an atom in frag1 and an atom in frag2 whose inter-atomic distance satisfied contact_bin[0] < distance < contact_bin[1]. Each value is a tuple (frag1_ecfp_i, frag2_ecfp_j) where frag1_ecfp_i and frag2_ecfp_j are the ECFP representations returned by compute_all_ecfp for the corresponding atoms. The returned mapping is suitable for constructing SPLIF features that pair local chemical environments of contacting atoms for use as model features.
    """
    from deepchem.feat.complex_featurizers.splif_fingerprints import compute_splif_features_in_range
    return compute_splif_features_in_range(
        frag1,
        frag2,
        pairwise_distances,
        contact_bin,
        ecfp_degree
    )


################################################################################
# Source: deepchem.feat.complex_featurizers.splif_fingerprints.featurize_splif
# File: deepchem/feat/complex_featurizers/splif_fingerprints.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def deepchem_feat_complex_featurizers_splif_fingerprints_featurize_splif(
    frag1: tuple,
    frag2: tuple,
    contact_bins: numpy.ndarray,
    pairwise_distances: numpy.ndarray,
    ecfp_degree: int
):
    """deepchem.feat.complex_featurizers.splif_fingerprints.featurize_splif computes Structural Protein-Ligand Interaction Fingerprint (SPLIF) features for fragment–fragment contacts in a binding pocket, returning one SPLIF dictionary per contact distance bin.
    
    This function iterates over the provided contact distance bins and, for each bin, collects fragment pairs whose pairwise distances fall into that bin and produces a mapping from fragment index pairs to the corresponding ECFP-derived fragment descriptors. The result is a list of SPLIF dictionaries, one dictionary per contact bin, preserving the order of contact_bins. This featurization is intended for downstream use by vectorize or voxelize routines in DeepChem that convert SPLIF dictionaries into fixed-size numerical inputs for machine learning models in molecular modeling and drug discovery workflows.
    
    Args:
        frag1 (tuple): A tuple of (coords, mol) as returned by load_molecule for the first fragment set. In DeepChem workflows this represents one side of the interaction (for example, ligand fragments or pocket residues). coords are the spatial coordinates of atoms in the fragment set (typically in Angstroms) and mol is the molecular graph object (commonly an RDKit Mol) used to compute ECFP-style fragment descriptors. The tuple is used to index fragment positions and to extract topological information needed to compute per-fragment ECFP features.
        frag2 (tuple): A tuple of (coords, mol) as returned by load_molecule for the second fragment set. This represents the other side of the interaction (for example, protein pocket fragments or another ligand fragment set). The structure and role mirror frag1 but refer to the second partner in pairwise contacts.
        contact_bins (numpy.ndarray): An array defining the contact distance ranges (in Angstroms) used to separate interactions into distance-resolved bins. Each element in contact_bins specifies a range of pair distances that will be treated as one category when computing SPLIF features. The order of contact_bins determines the order of dictionaries in the returned list (i.e., the i-th returned dictionary corresponds to contact_bins[i]).
        pairwise_distances (numpy.ndarray): A precomputed array of pairwise fragment–fragment distances (in Angstroms) between items in frag1 and frag2. This array is used to decide which fragment pairs fall into each contact bin. pairwise_distances must be consistent with the fragment ordering in frag1 and frag2; mismatched shapes or ordering can lead to incorrect assignments or errors propagated from lower-level utilities.
        ecfp_degree (int): The ECFP radius (graph distance) used to compute per-fragment circular fingerprints (Extended-Connectivity Fingerprints). This integer controls how much local graph context around each fragment atom is included when producing the fragment descriptor pair stored in each SPLIF entry.
    
    Returns:
        list: A list of dictionaries, one dictionary per contact bin in contact_bins. Each dictionary maps 2-tuples of fragment indices (frag1_index_i, frag2_index_j) to 2-tuples of fragment ECFP descriptors (frag1_ecfp_i, frag2_ecfp_j) computed using the provided ecfp_degree. The length of the returned list equals len(contact_bins). The dictionaries are suitable for downstream conversion by DeepChem's vectorize or voxelize utilities which expect SPLIF-style mappings of index pairs to fingerprint pairs.
    
    Behavior, side effects, and failure modes:
        This function is purely computational and has no side effects such as file or global state modification. It delegates per-bin feature computation to compute_splif_features_in_range; any exceptions raised by that helper (for example due to incompatible pairwise_distances shape, invalid contact bin definitions, or an invalid ecfp_degree) will propagate to the caller. The function preserves the ordering of contact_bins in the returned list. Inputs must be consistent: frag1 and frag2 should follow the same fragment indexing convention used to produce pairwise_distances; contact_bins should define non-overlapping or intentionally ordered ranges if overlap is not desired. Invalid or nonsensical numeric inputs (negative distances, negative ecfp_degree) are not validated here and may cause downstream errors.
    """
    from deepchem.feat.complex_featurizers.splif_fingerprints import featurize_splif
    return featurize_splif(frag1, frag2, contact_bins, pairwise_distances, ecfp_degree)


################################################################################
# Source: deepchem.feat.graph_features.features_to_id
# File: deepchem/feat/graph_features.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def deepchem_feat_graph_features_features_to_id(features: list, intervals: list):
    """Convert a list of discrete graph features into a single integer index used by DeepChem graph featurizers.
    
    This function is used in DeepChem's graph feature pipeline to map a vector of discrete, typically integer-valued features (for example atom or bond categorical/binned features returned by get_feature_list()) into a single linear index suitable for indexing into a flattened feature vector or lookup table. The mapping uses the provided spacings in intervals (as returned by get_intervals()) to perform a mixed-radix style encoding: each feature value is multiplied by the corresponding interval spacing and summed. The function then adds 1 to the result so that the integer 0 remains available to represent a reserved "null molecule" or absent feature in DeepChem's downstream data structures.
    
    Args:
        features (list): List of feature values produced by get_feature_list(). In the DeepChem graph-featurization context these are usually discrete, integer-valued feature entries (one per feature dimension). The function expects that features has at least len(intervals) entries because it indexes features by position up to the length of intervals. If features contains non-numeric entries, behavior depends on Python arithmetic semantics and may raise a TypeError.
        intervals (list): List of integer spacings produced by get_intervals(). Each entry defines the multiplicative spacing used for the corresponding feature dimension when constructing the linear index. The function iterates over range(len(intervals)) and uses intervals[k] to weight features[k]. If intervals contains non-integer or negative values the resulting index may be nonstandard for indexing use or reflect an invalid encoding.
    
    Returns:
        int: A positive integer index representing the encoded feature vector for use in DeepChem graph feature vectors and lookup tables. The returned value equals sum_{k=0..len(intervals)-1} features[k] * intervals[k] plus 1. The +1 offset ensures index 0 can be used as a reserved null value. Failure modes: if features is shorter than intervals an IndexError will be raised; if either list contains non-numeric types the operation may raise a TypeError or produce a non-integer result; if inputs are not lists but are list-like the behavior follows Python's sequence indexing and arithmetic rules. There are no external side effects.
    """
    from deepchem.feat.graph_features import features_to_id
    return features_to_id(features, intervals)


################################################################################
# Source: deepchem.feat.graph_features.get_intervals
# File: deepchem/feat/graph_features.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def deepchem_feat_graph_features_get_intervals(l: list):
    """deepchem.feat.graph_features.get_intervals computes multiplicative stride intervals for a list of discrete option lists used in DeepChem graph featurization.
    
    This function is used in DeepChem's graph feature utilities to compute cumulative product "intervals" (mixed-radix strides) when enumerating combinations of categorical choices across multiple feature slots (for example, different atom or edge feature option lists when flattening categorical feature combinations in molecular/graph featurization for drug discovery and related computational chemistry tasks). The implementation adds 1 to every inner list length to ensure an empty option list does not force a zero product; the first interval is initialized to 1 and subsequent intervals are produced by multiplying by (len(inner_list) + 1).
    
    Args:
        l (list): A list of lists. Each element of l is expected to be a sequence (typically a list) representing the set of discrete options for one feature slot. The function treats each inner element by taking its length via len(inner). This parameter is the core input that defines the per-slot radices used to compute cumulative stride intervals for indexing or flattening combinations of options.
    
    Returns:
        list: A list of integers of length len(l). The returned list contains intervals where returned[0] is always 1 (initialized as the base stride), and for k >= 1 returned[k] = returned[k - 1] * (len(l[k]) + 1). These integers serve as multiplicative strides for converting a multi-slot selection (one index per inner list) into a single linear index in a mixed-radix system used by DeepChem graph feature routines.
    
    Behavior, side effects, and failure modes:
        The function performs no mutation of the input list l and has no external side effects. Time and space complexity are O(n) where n = len(l). The function expects len(l) to be at least 1; if l is an empty list the implementation will raise an IndexError when attempting to set the initial interval at index 0. If l is not a list or if elements of l do not support len(inner), a TypeError (or similar exception raised by len()) may occur. Empty inner lists are handled by the +1 rule: len([]) == 0 becomes a multiplier of 1 so that an empty option list does not zero out later intervals.
    """
    from deepchem.feat.graph_features import get_intervals
    return get_intervals(l)


################################################################################
# Source: deepchem.feat.graph_features.id_to_features
# File: deepchem/feat/graph_features.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def deepchem_feat_graph_features_id_to_features(id: int, intervals: list):
    """deepchem.feat.graph_features.id_to_features converts a single flattened feature-vector index into the original set of discrete feature indices used by DeepChem graph featurizers. This function reverses the mixed-radix encoding produced when a multi-feature categorical state was flattened to a single integer index (for example, when enumerating or indexing combinations of atom/bond categorical features during featurization). It is typically used in DeepChem's molecular graph featurization pipeline to map an index back to the list of per-feature category indices as returned by get_feature_list().
    
    Args:
        id (int): The 1-based index in a flattened feature vector that encodes a combination of discrete features. In the featurization workflow, this integer represents a position in the flattened space produced by combining multiple categorical feature dimensions. The function internally subtracts 1 from this value to correct for a null/one-based indexing convention used by the caller; therefore callers should pass the same indexing convention used when the flat index was created. If id is negative after correction or otherwise outside the expected encoded range, the returned feature components may be negative or otherwise invalid.
        intervals (list): List of interval sizes as returned by get_intervals(). Each element in this list is used as the radix (stride) for a corresponding categorical feature when decoding the flattened index and is expected to be an integer-like value that supports integer floor division (//). This list should follow the same ordering and length expected by the featurizer (the implementation assumes six interval entries and will index intervals[0] through intervals[5]). Supplying a shorter list will raise an IndexError; supplying zero-valued entries will raise a ZeroDivisionError when they are used as divisors.
    
    Returns:
        features (list): A length-6 list of integers that correspond to the original per-feature categorical indices (the same structure returned by get_feature_list()). Each element represents the decoded category index for one feature dimension in the order used by the featurizer. The function has no side effects and does not modify its inputs. Failure modes include IndexError if intervals has fewer than six elements, ZeroDivisionError if any interval used as a divisor is zero, and potentially invalid (e.g., negative) feature components if the provided id is outside the valid encoded range.
    """
    from deepchem.feat.graph_features import id_to_features
    return id_to_features(id, intervals)


################################################################################
# Source: deepchem.feat.mol_graphs.cumulative_sum
# File: deepchem/feat/mol_graphs.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def deepchem_feat_mol_graphs_cumulative_sum(l: list, offset: int = 0):
    """deepchem.feat.mol_graphs.cumulative_sum computes cumulative (prefix) sums for a sequence of integer counts and returns a one-dimensional numpy array whose first element corresponds to the zero-based start index (plus any supplied offset). This function is used in DeepChem's molecular graph featurization and batching code to produce reindexing pointers (for example, to compute per-molecule atom or edge start indices when concatenating variable-length segments into flat arrays). The returned array includes the final total (useful for range/search operations) and thus has length len(l) + 1.
    
    Args:
        l (list): List of integers representing counts for consecutive segments. In the DeepChem mol_graphs context these are typically small non-negative counts such as the number of atoms or edges per molecule. The function computes prefix sums over these counts; providing negative values will produce a non-monotonic prefix sequence and can break downstream reindexing logic. Elements that are not integers will be converted by NumPy where possible, but this may raise an error or produce unexpected results.
        offset (int): Integer constant added to every value of the computed cumulative-sum array. Use this to shift the base index when concatenating into a larger buffer or when an existing global offset is required (for example, to start indexing at a previously used length). Defaults to 0. Non-integer inputs may be cast by NumPy or raise an error.
    
    Returns:
        numpy.ndarray: One-dimensional numpy array of length len(l) + 1 containing the cumulative sums with the first element equal to offset and the final element equal to offset + sum(l). Example: l = [3, 2, 4], offset = 0 -> array([0, 3, 5, 9]). No side effects; the function is deterministic and purely functional. Failure modes include TypeError or ValueError if l cannot be interpreted as numeric counts or if offset cannot be used in arithmetic with the computed array.
    """
    from deepchem.feat.mol_graphs import cumulative_sum
    return cumulative_sum(l, offset)


################################################################################
# Source: deepchem.feat.mol_graphs.cumulative_sum_minus_last
# File: deepchem/feat/mol_graphs.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def deepchem_feat_mol_graphs_cumulative_sum_minus_last(l: list, offset: int = 0):
    """cumulative_sum_minus_last returns cumulative sums for a sequence of integer counts, omitting the final total. It is intended for reindexing tasks in molecular graph construction and other DeepChem workflows where per-item counts (for example, number of atoms or bonds per molecule) are converted into starting offsets into a flat concatenated array.
    
    This function computes the cumulative sum of the input list l with numpy dtype np.int32, inserts an initial 0 so that the first returned value is 0, removes the final element (the overall total), and then adds the integer offset value to every element. Practically, the returned values represent the starting index for each block when concatenating blocks of sizes given by l. Example: l = [3, 2, 4] -> cumulative sums [3, 5, 9], insert initial 0 -> [0, 3, 5, 9], drop last element -> [0, 3, 5], then add offset (default 0) -> [0, 3, 5].
    
    Args:
        l (list): List of integers representing counts for consecutive blocks. In DeepChem this is typically a small list of per-molecule or per-component counts such as numbers of atoms, bonds, or features. Each element is cast to numpy.int32 for the cumulative sum. The length of l determines the length of the returned array; the function does not validate that elements are non-negative, but negative values will affect the computed offsets in the usual arithmetic way.
        offset (int): Integer offset to add to every returned cumulative-sum value. Default is 0. In molecular batching workflows this is commonly used to add a base index when building global indexing arrays across multiple molecules or batches.
    
    Returns:
        numpy.ndarray: 1-D numpy.ndarray of dtype numpy.int32 and length equal to len(l). Each element is the starting index (cumulative sum up to but not including that block) plus the provided offset. The final total sum of all elements in l is intentionally omitted from the result.
    
    Behavior and side effects:
        The function is pure and has no side effects beyond allocating and returning a numpy array. It always returns an array of length equal to the input list length. The first returned element is 0 + offset, corresponding to the start index of the first block.
    
    Failure modes and notes:
        If l contains non-numeric or non-integral values that cannot be cast to numpy.int32, numpy will raise an error. Very large sums that exceed the range of numpy.int32 may overflow; if you expect sums larger than the 32-bit integer range, convert inputs externally before calling or handle overflow appropriately. The function does not perform additional type or range validation beyond numpy's casting and arithmetic.
    """
    from deepchem.feat.mol_graphs import cumulative_sum_minus_last
    return cumulative_sum_minus_last(l, offset)


################################################################################
# Source: deepchem.feat.molecule_featurizers.conformer_featurizer.safe_index
# File: deepchem/feat/molecule_featurizers/conformer_featurizer.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def deepchem_feat_molecule_featurizers_conformer_featurizer_safe_index(
    feature_list: list,
    e: int
):
    """deepchem.feat.molecule_featurizers.conformer_featurizer.safe_index returns the index of a requested element within a feature vector used by DeepChem's conformer featurizer; if the requested element is not present, the function returns the index of the last element of the provided list instead of raising an exception. This helper is used in molecular featurization pipelines to robustly map an element identifier to a position in a feature_list without interrupting processing when a lookup fails.
    
    Args:
        feature_list (list): Feature vector used by the conformer featurizer. In the DeepChem conformer featurizer context this is typically a list of per-atom or per-feature entries (e.g., atomic indices, feature identifiers, or precomputed descriptor values). The function searches this list for the element e. Practical significance: passing the featurizer's feature vector ensures a stable fallback index is returned when a requested element is missing.
        e (int): Element index or identifier to find inside feature_list. In molecular featurization this is commonly an integer index corresponding to an atom or a feature key. The function returns the position of the first occurrence of this integer inside feature_list when present.
    
    Returns:
        int: The integer index into feature_list where e first occurs, or len(feature_list) - 1 when e is not present. Behavior details: the implementation calls list.index(e) and catches ValueError; on a missing element it returns the last valid index of feature_list as a safe fallback so downstream code can continue without a ValueError. Edge cases and failure modes: if feature_list is empty, len(feature_list) - 1 evaluates to -1 and this value is returned — using that returned value directly to index into feature_list will raise an IndexError, so callers should validate the returned index or ensure feature_list is non-empty before indexing. There are no side effects and no exceptions are raised by this function for missing elements.
    """
    from deepchem.feat.molecule_featurizers.conformer_featurizer import safe_index
    return safe_index(feature_list, e)


################################################################################
# Source: deepchem.feat.molecule_featurizers.smiles_to_seq.create_char_to_idx
# File: deepchem/feat/molecule_featurizers/smiles_to_seq.py
# Category: valid
################################################################################

def deepchem_feat_molecule_featurizers_smiles_to_seq_create_char_to_idx(
    filename: str,
    max_len: int = 250,
    smiles_field: str = "smiles"
):
    """Creates a character-to-index mapping from SMILES strings stored in a CSV file for use in sequence-based molecular featurization.
    
    This function is used in DeepChem's SMILES-to-sequence featurizers to build a vocabulary of characters that appear in SMILES (Simplified Molecular Input Line Entry System) strings. It reads a CSV file using pandas, extracts the column specified by smiles_field, collects all characters that occur in SMILES strings whose length is less than or equal to max_len, appends two special tokens (PAD_TOKEN and OUT_OF_VOCAB_TOKEN) to the vocabulary, and returns a dictionary that maps each character/token to a unique integer index. The resulting mapping is typically used to convert SMILES strings into integer sequences for embedding layers, one-hot encodings, or other model inputs in molecular machine learning workflows.
    
    Args:
        filename (str): Path or filename of a CSV file containing SMILES strings. The file is read with pandas.read_csv(filename). This parameter specifies where the SMILES data is loaded from on disk; a FileNotFoundError or pandas parser errors will be raised if the file cannot be accessed or parsed.
        max_len (int): Maximum allowed length of SMILES strings to consider when building the character set. Only characters from SMILES entries whose Python len(smile) is less than or equal to max_len are included. Default is 250. Strings longer than max_len are ignored for vocabulary construction.
        smiles_field (str): Name of the column in the CSV file that contains SMILES strings. The function accesses smiles_df[smiles_field]; if this column is missing a KeyError will be raised. Default is "smiles".
    
    Returns:
        Dict[str, int]: A dictionary mapping each character (and two appended special tokens) to a unique integer index. The mapping contains one entry per unique character observed in the selected SMILES strings plus entries for PAD_TOKEN and OUT_OF_VOCAB_TOKEN, which are appended to the character list to support sequence padding and handling of characters not seen during vocabulary construction. The integer indices are assigned by enumerating the list created from a set of characters; therefore the specific numeric indices may vary depending on Python's iteration order over the set and should not be assumed stable across runs unless the caller enforces a deterministic ordering prior to creating the mapping.
    
    Behavior and side effects:
        The function performs I/O by reading filename with pandas.read_csv. It constructs the character set by iterating over smiles_df[smiles_field] and calling len(smile) and set(smile) for each value; non-string or missing values in the column may raise TypeError or other exceptions when len or set are applied. PAD_TOKEN and OUT_OF_VOCAB_TOKEN are appended to the vocabulary list after collecting unique characters; these tokens must be defined in the module namespace where this function is used. No file writes are performed.
    
    Failure modes:
        Raises FileNotFoundError or pandas-related exceptions if filename cannot be read. Raises KeyError if smiles_field is not present in the CSV. May raise TypeError or ValueError when encountering non-string or malformed SMILES entries (for example NaN) during len(smile) or set(smile) operations. If deterministic index ordering is required, callers should sort or otherwise fix the character ordering before relying on index values.
    """
    from deepchem.feat.molecule_featurizers.smiles_to_seq import create_char_to_idx
    return create_char_to_idx(filename, max_len, smiles_field)


################################################################################
# Source: deepchem.feat.sequence_featurizers.position_frequency_matrix_featurizer.PFM_to_PPM
# File: deepchem/feat/sequence_featurizers/position_frequency_matrix_featurizer.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def deepchem_feat_sequence_featurizers_position_frequency_matrix_featurizer_PFM_to_PPM(
    pfm: numpy.ndarray
):
    """deepchem.feat.sequence_featurizers.position_frequency_matrix_featurizer.PFM_to_PPM converts a position frequency matrix (PFM) into a position probability matrix (PPM) by normalizing each column so the entries per column represent relative probabilities. In the DeepChem sequence featurizers context, a PFM is typically a 2-D numpy.ndarray of counts (for example counts of nucleotides or amino acids at each alignment position) and the resulting PPM is used by downstream featurizers and models to represent motif or positional composition as probabilities rather than raw counts.
    
    Args:
        pfm (numpy.ndarray): A 2-D array representing a position frequency matrix. Rows correspond to alphabet symbols (for example nucleotides or amino acids) and columns correspond to sequence positions. Each element pfm[i, j] is the count or weight for symbol i at position j. This function expects pfm to be indexable as pfm[:, col] and to have at least two dimensions; passing a non-2-D array will raise an IndexError when attempting to access shape[1].
    
    Returns:
        numpy.ndarray: A numpy.ndarray of the same shape as pfm where each column has been normalized by the column sum (total counts for that position) so that the entries in columns with positive total count represent the fraction of the total (i.e., probabilities that sum to 1 per column). Columns in pfm whose total count is zero are left unchanged in the returned array (they remain as in the input copy). The returned matrix is typically used as a PPM for downstream sequence-featurization, motif scoring, or probabilistic modeling.
    
    Behavior and side effects:
        This function makes a shallow copy of the input pfm at the start and modifies only that copy; the original pfm passed by the caller is not modified. For each column, if the column sum (total_count) is greater than zero, the column is replaced by its elementwise ratio to total_count. If total_count is zero, the function does not alter that column in the copy, leaving whatever values (typically zeros) were present. Because numpy division is used, integer dtypes in pfm may be converted to floating-point values in the result (or truncated/cast depending on the input dtype and numpy behavior); callers who require a specific dtype should cast explicitly before or after calling this function.
    
    Failure modes and edge cases:
        Passing inputs that are not numpy.ndarray or that are not two-dimensional will result in exceptions (for example, AttributeError or IndexError) when attempting to access shape or slice columns. Negative values in pfm will be normalized as usual and will produce negative entries in the output PPM; the function does not validate that counts are non-negative. Columns with extremely small positive sums are normalized but may lead to floating-point underflow or precision issues consistent with numpy arithmetic.
    """
    from deepchem.feat.sequence_featurizers.position_frequency_matrix_featurizer import PFM_to_PPM
    return PFM_to_PPM(pfm)


################################################################################
# Source: deepchem.metrics.genomic_metrics.get_pssm_scores
# File: deepchem/metrics/genomic_metrics.py
# Category: valid
################################################################################

def deepchem_metrics_genomic_metrics_get_pssm_scores(
    encoded_sequences: numpy.ndarray,
    pssm: numpy.ndarray
):
    """Convolves a position-specific scoring matrix (PSSM) and its reverse complement with one-hot (or otherwise encoded per-base) nucleotide sequences and returns the per-position maximum score between the forward and reverse-complement orientations. This function is used in DeepChem genomic metrics to scan genomic sequences for motif matches (for example, scoring candidate transcription factor binding sites) by cross-correlating a PSSM with encoded sequences and selecting the best orientation at each sequence position.
    
    Args:
        encoded_sequences (numpy.ndarray): A 4D numpy array containing encoded nucleotide sequences with shape `(N_sequences, N_letters, sequence_length, 1)`. The last dimension must be length 1 because the function calls `squeeze(axis=3)` to remove it; if this axis is absent or not equal to 1 a ValueError will be raised. N_letters is the number of distinct alphabet symbols encoded per position (for DNA this is typically 4). The ordering of bases along the N_letters axis must match the row ordering of `pssm` because the implementation indexes `pssm` by base index and uses symmetric indexing to compute reverse-complement scores. The array must contain numeric types (integers or floats); the function returns floating-point scores and will perform numeric cross-correlation per base.
        pssm (numpy.ndarray): A 2D numpy array of shape `(4, pssm_length)` representing a position-specific scoring matrix (PSSM) for a motif of length `pssm_length`. Each row corresponds to the per-base scores across the motif for one alphabet symbol; the number of rows must match `encoded_sequences.shape[1]` (e.g., 4 for standard DNA encodings). The function also uses the column-reversed rows of this matrix to compute reverse-complement scores, so the row ordering must be compatible with the encoding used in `encoded_sequences`.
    
    Behavior and implementation details:
        The function removes the trailing singleton channel dimension of `encoded_sequences` with `squeeze(axis=3)` to obtain a 3D array of shape `(N_sequences, N_letters, sequence_length)`. It initializes separate forward and reverse-complement score arrays to negative infinity (float) with the same shape as the squeezed `encoded_sequences`. For each base index it extracts the corresponding row of `pssm` (shape `(1, pssm_length)`) and its column-reversed version (the reverse complement of the motif row). It computes a 1D cross-correlation between the per-sequence per-base signal and the PSSM row using `correlate2d(..., mode='same')` for both forward and reverse-complement orientations, storing per-base correlation results. After iterating over all bases the function sums per-base correlations to produce a total forward score and a total reverse-complement score for each sequence position, then returns the elementwise maximum of these two arrays so that each output position contains the best orientation score. The returned dtype is floating point because intermediate arrays are initialized with `-np.inf` and correlations produce floats.
    
    Side effects and performance:
        This function has no external side effects (it does not modify inputs in-place and only returns a new numpy array). It performs a loop over bases and calls a 2D correlate operation per base; for large numbers of sequences, long sequences, or long PSSMs this can be computationally and memory intensive. The caller should ensure sufficient memory and may need to batch inputs for very large datasets.
    
    Failure modes and errors:
        A ValueError will be raised if `encoded_sequences` cannot be squeezed at axis 3 (i.e., the array does not have a trailing singleton dimension). An IndexError or related error will occur if the number of rows in `pssm` does not match `encoded_sequences.shape[1]`. A TypeError or numeric error may occur if non-numeric data is provided. The behavior of correlation depends on the presence of a compatible `correlate2d` implementation in scope; if the required function is not available an ImportError/NameError may be raised.
    
    Returns:
        numpy.ndarray: A 2D numpy array of shape `(N_sequences, sequence_length)` containing the maximum motif score at each position for each sequence. Each element is the higher of the forward-orientation sum-of-base correlations and the reverse-complement orientation sum-of-base correlations at that sequence position; values are floating-point and may be -inf if no correlation was computed for a position due to input shapes (see failure modes above).
    """
    from deepchem.metrics.genomic_metrics import get_pssm_scores
    return get_pssm_scores(encoded_sequences, pssm)


################################################################################
# Source: deepchem.metrics.metric.from_one_hot
# File: deepchem/metrics/metric.py
# Category: valid
################################################################################

def deepchem_metrics_metric_from_one_hot(y: numpy.ndarray, axis: int = 1):
    """Converts one-hot encoded label vectors to integer class indices using numpy.argmax. This function, deepchem.metrics.metric.from_one_hot, is used in DeepChem preprocessing and metric computation pipelines (for example when evaluating classification models for molecular property prediction or bioactivity) to transform labels from a one-hot representation into a 1-D array of class IDs that downstream metrics and loss functions expect.
    
    Args:
        y (numpy.ndarray): A numpy array containing one-hot encoded labels. In DeepChem workflows this is typically shaped (n_samples, num_classes) where each row has a single maximal entry marking the true class. The function applies numpy.argmax along the axis specified by the axis parameter to locate the class index for each sample. The array should contain comparable numeric values (e.g., 0/1 one-hot vectors or score/probability vectors) so that a maximum per sample can be determined.
        axis (int): The axis of y that contains the one-hot encoding and over which to take the argmax. The default value 1 corresponds to the common layout (n_samples, num_classes). If your input arranges classes along a different axis, set axis accordingly. This parameter is passed directly to numpy.argmax; invalid axis values will raise the same IndexError from numpy.
    
    Returns:
        numpy.ndarray: A 1-D numpy array of shape (n_samples,) containing integer class indices (positions of the maximum values along the specified axis) for each sample. These indices correspond to class labels used by DeepChem models and metrics (e.g., 0..num_classes-1). The returned array is produced by numpy.argmax and has no side effects.
    
    Behavior, defaults, and failure modes:
        The function is a pure transformation with no side effects; it delegates to numpy.argmax. By default it expects one-hot encodings along axis=1 and will return one index per sample. If a row contains ties for the maximum, numpy.argmax returns the first occurrence (tie-breaking is by lowest index). If y has an unexpected shape (for example not containing a leading sample dimension when using the default axis) or axis is out of bounds, numpy will raise an IndexError. If y contains NaNs or non-numeric entries, the result may be undefined or raise a TypeError/ValueError from numpy. This function does not perform validation beyond what numpy.argmax enforces, so callers should ensure inputs are correctly formatted for classification tasks in DeepChem.
    """
    from deepchem.metrics.metric import from_one_hot
    return from_one_hot(y, axis)


################################################################################
# Source: deepchem.metrics.metric.to_one_hot
# File: deepchem/metrics/metric.py
# Category: valid
################################################################################

def deepchem_metrics_metric_to_one_hot(y: numpy.ndarray, n_classes: int = 2):
    """deepchem.metrics.metric.to_one_hot converts integer class labels into a one-hot encoded 2D numpy array for use in classification tasks in DeepChem (for example, preparing label targets for training or computing classification metrics in molecular machine learning workflows such as drug discovery, materials science, or biology). The function turns a vector of class indices into an (N, n_classes) array where each row is a one-hot representation of the corresponding label. It is intended for integer class indices in the range 0..n_classes-1 and is commonly used by DeepChem components that expect dense one-hot labels for loss computation and metric evaluation.
    
    Args:
        y (numpy.ndarray): A 1-D or 2-D numpy array containing class labels with shape (N,) or (N, 1). Each element is interpreted as a class index and will be cast to numpy.int64 for indexing. Practical significance: in DeepChem classification pipelines this is the observed label vector for a dataset of N examples; typical values are small non-negative integers representing discrete classes. The function will raise a ValueError if y has more than 2 dimensions or if it is 2-D but the second dimension is not 1.
        n_classes (int): The number of target classes to produce (the width of the one-hot encoding). Defaults to 2. Practical significance: sets the number of columns in the output one-hot matrix and must be chosen to match the number of distinct classes the downstream model or metric expects. The function enforces that the number of unique values in y does not exceed n_classes and will raise a ValueError if more unique labels are present than n_classes.
    
    Returns:
        numpy.ndarray: A 2-D numpy array of shape (N, n_classes). Each row i has a 1.0 in the column corresponding to int(y[i]) and 0.0 elsewhere, producing a standard one-hot representation suitable for neural network loss functions and metric calculations in DeepChem. No in-place modification of the input y occurs; the function constructs and returns a new array.
    
    Behavior, defaults, and failure modes:
        - Input shapes allowed: (N,) or (N, 1). If y has more than 2 dimensions, a ValueError is raised with message "y must be a vector of shape (N,) or (N, 1)". If y is 2-D but y.shape[1] != 1, a ValueError is raised with the same message.
        - The function casts y to numpy.int64 for indexing. Therefore y should contain integer-valued class indices; floating values will be truncated when cast to int64 which may lead to unexpected class assignments.
        - The function assumes class indices map to columns 0..n_classes-1. If len(np.unique(y)) > n_classes a ValueError is raised with message "y has more than n_class unique elements." If any cast index is outside the 0..n_classes-1 range, numpy indexing behavior will apply (which may raise an IndexError or produce unintended results for negative indices).
        - The returned array is newly allocated with shape (N, n_classes) and contains numeric entries (zeros and ones). There are no side effects on the input array.
        - Default behavior: when n_classes is not provided by the caller, it defaults to 2. Users should pass an explicit n_classes that matches their dataset when working with more than two classes to avoid ValueError.
    """
    from deepchem.metrics.metric import to_one_hot
    return to_one_hot(y, n_classes)


################################################################################
# Source: deepchem.metrics.score_function.bedroc_score
# File: deepchem/metrics/score_function.py
# Category: valid
################################################################################

def deepchem_metrics_score_function_bedroc_score(
    y_true: numpy.ndarray,
    y_pred: numpy.ndarray,
    alpha: float = 20.0
):
    """Compute the BEDROC (Boltzmann-enhanced discrimination of ROC) score for early recognition in virtual screening and ranked classification tasks.
    
    This function implements the BEDROC metric as described by Truchon and Bayley (2007). In the context of DeepChem and molecular virtual screening (drug discovery and cheminformatics workflows), BEDROC quantifies how well active compounds (positive class) are concentrated at the top of a ranked list produced by a model, with tunable emphasis on early recognition via the alpha parameter. The implementation delegates the final calculation to RDKit's CalcBEDROC and therefore requires RDKit to be installed.
    
    Args:
        y_true (numpy.ndarray): Binary class labels for each example. Values must be 1 for the positive (active) class and 0 for the negative class. This array is flattened internally and validated such that the unique values equal [0, 1]; an assertion error is raised if labels are not binary. In virtual screening, this input represents ground-truth activity annotations for the compounds being ranked.
        y_pred (numpy.ndarray): Predicted scores/probabilities used to rank examples. This function expects model outputs in one-hot/probability format where the positive-class score is in column index 1 (i.e., the implementation indexes y_pred[:, 1]). The array is flattened and paired with y_true to form (label, score) tuples which are then sorted in descending score order for BEDROC computation. Supplying a different shape or encoding for predictions (for example single-column scores) will lead to indexing errors or incorrect results.
        alpha (float): Early recognition tuning parameter (default 20.0). Larger alpha places greater weight on top-of-list enrichment (more emphasis on early recognition). The parameter is passed directly to RDKit's CalcBEDROC and controls how strongly early-ranked actives influence the returned BEDROC value.
    
    Returns:
        float: BEDROC score in the range [0, 1] that indicates the degree of early recognition achieved by the ranking produced from y_pred relative to y_true. Higher values indicate better early enrichment of positive examples. The value is computed by RDKit's CalcBEDROC after constructing and sorting a list of (label, score) pairs.
    
    Raises:
        ImportError: If RDKit is not installed or cannot be imported. This function requires RDKit's ML Scoring utilities (CalcBEDROC) to perform the final calculation.
        AssertionError: If the number of examples in y_true and y_pred do not match, or if y_true does not contain exactly the binary labels 0 and 1. These checks are performed before any computation.
        IndexError / ValueError: If y_pred does not have a column at index 1 (for example, if predictions are provided as a single-column array), indexing y_pred[:, 1] will raise an error. No additional validation is performed for NaN or infinite values; such values will propagate to the RDKit calculation and may produce undefined results.
    
    Behavior and notes:
        The function flattens y_true and extracts the positive-class score from column 1 of y_pred, pairs labels with scores, sorts pairs by descending score, and then calls RDKit's CalcBEDROC with the sorted pairs and the provided alpha. It performs no in-place modification of inputs (purely functional) but will raise an ImportError immediately if RDKit is unavailable. The sort operation is O(n log n) in the number of examples. Users must ensure predictions are provided in the expected one-hot/probability format and that labels are strictly binary (0 and 1) for valid results.
    
    References:
        Truchon, J.-F., & Bayly, C. I. (2007). Evaluating virtual screening methods: good and bad metrics for the "early recognition" problem. Journal of Chemical Information and Modeling, 47(2), 488–508.
    """
    from deepchem.metrics.score_function import bedroc_score
    return bedroc_score(y_true, y_pred, alpha)


################################################################################
# Source: deepchem.metrics.score_function.concordance_index
# File: deepchem/metrics/score_function.py
# Category: valid
################################################################################

def deepchem_metrics_score_function_concordance_index(
    y_true: numpy.ndarray,
    y_pred: numpy.ndarray
):
    """deepchem.metrics.score_function.concordance_index computes the concordance index for a set of true target values and predicted scores, quantifying how well the predicted ranking matches the true ordering. This function is used in DeepChem for evaluating ranking quality in tasks such as survival analysis and regression-based predictions in drug discovery and computational biology: it reports the fraction of comparable instance pairs for which the predicted ordering agrees with the true ordering.
    
    The function sorts inputs by the true values, enumerates all comparable pairs (pairs with different true values), and accumulates a score where correctly ordered prediction pairs contribute 1, tied predictions contribute 0.5, and incorrectly ordered pairs contribute 0. The final concordance index is the ratio of accumulated correct score to the number of comparable pairs.
    
    Args:
        y_true (np.ndarray): 1-D array of ground-truth continuous target values. Each element is the true value used to define the desired ordering (for example, survival times or measured activity). Values that are exactly equal are considered tied and such pairs are ignored for comparison. The function expects y_true to be a one-dimensional numpy array of numeric values; if it is not, NumPy operations may raise an exception.
        y_pred (np.ndarray): 1-D array of predicted scores corresponding to y_true. Each element is a numeric prediction whose relative order is being evaluated (for example, model scores, predicted property values). y_pred must have the same length as y_true; it is not required to be sorted and the function will internally reorder y_pred to match the sort order of y_true.
    
    Behavior and side effects:
        The function does not modify the caller's original arrays in-place; it creates sorted views/reassignments internally. It sorts both arrays by the ascending order of y_true and then examines all index pairs (i, j) with i < j. Only pairs where the true values differ (y_true[i] != y_true[j]) are considered comparable and counted. For each comparable pair:
        - If y_pred[i] == y_pred[j], the pair contributes 0.5 to the numerator (half credit for tied predictions).
        - If y_pred[i] < y_pred[j], the pair contributes 1.0 if y_true[i] < y_true[j], otherwise 0.0.
        - If y_pred[i] > y_pred[j], the pair contributes 1.0 if y_true[i] > y_true[j], otherwise 0.0.
        The function computes correct_pairs / pairs and returns this float.
    
    Complexity and practical considerations:
        The implementation uses a nested loop over all index pairs after sorting, so the time complexity is O(n^2) in the number of examples. For large datasets this can be computationally expensive; consider subsampling or using an alternative O(n log n) concordance implementations for very large n.
    
    Failure modes and errors:
        - If y_true and y_pred have different lengths, NumPy indexing or subsequent comparisons will raise an error.
        - If all y_true values are identical (no comparable pairs), the function raises an AssertionError with the message 'No pairs for comparision'.
        - If inputs are not numeric or not convertible to NumPy arrays with comparable ordering, comparisons may raise TypeError or ValueError.
        - The function assumes 1-D arrays; multi-dimensional arrays will produce behavior determined by NumPy's argsort and indexing semantics and are not supported by this implementation.
    
    Returns:
        float: Concordance index score in the interval [0.0, 1.0], representing the fraction of comparable pairs whose predicted order agrees with the true order (with tied predictions receiving half credit). A value of 1.0 indicates perfect concordance, 0.5 indicates chance-level agreement for random-like predictions on comparable pairs, and 0.0 indicates perfect discordance.
    
    References:
        Steck, Harald, et al. "On ranking in survival analysis: Bounds on the concordance index." Advances in neural information processing systems (2008): 1209-1216.
    """
    from deepchem.metrics.score_function import concordance_index
    return concordance_index(y_true, y_pred)


################################################################################
# Source: deepchem.metrics.score_function.jaccard_index
# File: deepchem/metrics/score_function.py
# Category: valid
################################################################################

def deepchem_metrics_score_function_jaccard_index(y: numpy.ndarray, y_pred: numpy.ndarray):
    """Computes the Jaccard Index (Intersection over Union) between a ground-truth label array and a predicted label array. This metric is commonly used in image segmentation tasks to quantify the overlap between predicted and reference masks, and within DeepChem workflows it can be used when evaluating predicted segmentation or set-like label outputs in applications across drug discovery, materials science, quantum chemistry, and biology.
    
    DEPRECATED: WILL BE REMOVED IN A FUTURE VERSION OF DEEPCHEM. Use jaccard_score instead. This function is a thin wrapper that delegates to jaccard_score and returns the same result.
    
    Args:
        y (numpy.ndarray): Ground truth array containing the reference labels or mask. In typical use this is an elementwise label or boolean mask encoded as a NumPy array where each element represents the true class or membership of the corresponding spatial/location element. In DeepChem pipelines this represents the authoritative annotation produced by experiments, simulations, or preprocessing steps against which predictions are compared.
        y_pred (numpy.ndarray): Predicted array containing model outputs aligned with y. This must be a NumPy array of the same shape as y where each element represents the predicted class or mask membership for the corresponding element in y. In practice this is produced by a model or postprocessing step in a DeepChem evaluation pipeline.
    
    Behavior and side effects:
        This function computes the Jaccard Index (intersection over union) in the same manner as sklearn-like jaccard implementations: it compares y and y_pred elementwise to compute the size of the intersection divided by the size of the union of the predicted and true positive sets. The function performs no in-place modification of inputs and has no external side effects. It is deterministic for a given pair of inputs.
        The implementation delegates directly to jaccard_score; callers should migrate to jaccard_score because this wrapper is deprecated and will be removed in a future DeepChem release.
        Both y and y_pred are expected to be numpy.ndarray instances and to have the same shape; providing arrays of differing shapes or non-ndarray types may cause a TypeError or ValueError from the underlying implementation. The exact interpretation of array contents (binary mask, integer class labels, or multi-label encoding) follows the semantics of the delegated jaccard_score function used in this codebase.
    
    Failure modes:
        If y and y_pred have mismatched shapes a runtime exception will be raised. If inputs are not numpy.ndarray objects, behavior is undefined and will likely raise a TypeError. If the arrays contain unsupported label encodings for the delegated jaccard_score implementation, that function will raise an appropriate error. Because this wrapper is deprecated, new code should call jaccard_score directly to avoid future breakage.
    
    Returns:
        float: The computed Jaccard Index (intersection over union) as a floating-point number between 0 and 1 inclusive, where 0 indicates no overlap between predicted and true sets and 1 indicates perfect agreement.
    """
    from deepchem.metrics.score_function import jaccard_index
    return jaccard_index(y, y_pred)


################################################################################
# Source: deepchem.metrics.score_function.mae_score
# File: deepchem/metrics/score_function.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for mae_score because the docstring has no description for the argument 'y_true'
################################################################################

def deepchem_metrics_score_function_mae_score(y_true: numpy.ndarray, y_pred: numpy.ndarray):
    """deepchem.metrics.score_function.mae_score computes the mean absolute error (MAE) between provided ground-truth and predicted numeric arrays. This function is a small wrapper around sklearn.metrics.mean_absolute_error and is intended for use in DeepChem regression workflows (for example, evaluating models that predict molecular properties such as binding affinities, formation energies, or other continuous targets in drug discovery, materials science, quantum chemistry, and biology). The returned MAE is the average absolute difference between corresponding elements of y_true and y_pred and has the same units as the target values; lower values indicate better agreement between predictions and observations.
    
    Args:
        y_true (numpy.ndarray): Ground-truth target values for a regression task. This array contains the observed property values (for example, experimental measurements) and must be a numeric NumPy array. Its shape must match y_pred so that elements align elementwise for absolute-difference computation. The function does not modify y_true in place.
        y_pred (numpy.ndarray): Predicted target values produced by a model. This array must be a numeric NumPy array with the same shape as y_true so that each prediction corresponds to the correct ground-truth value. The function does not modify y_pred in place.
    
    Returns:
        float: The mean absolute error computed as the arithmetic mean of absolute differences |y_true - y_pred| across all elements. This value is non-negative and expressed in the same units as the inputs; it summarizes the average per-sample absolute deviation of predictions from observations and is commonly used in DeepChem to compare and rank regression models.
    
    Behavior and failure modes:
        This function is deterministic and has no side effects beyond reading the input arrays. It delegates computation to sklearn.metrics.mean_absolute_error; therefore, errors raised by that routine (for example, shape mismatch leading to a ValueError, or passing non-numeric dtypes causing a TypeError) will propagate to the caller. If y_true or y_pred contain NaN values, the resulting MAE will be NaN (NaNs propagate through the arithmetic mean). The function does not perform input coercion beyond what NumPy and scikit-learn perform, and it does not validate domain-specific constraints beyond requiring numeric NumPy arrays of matching shape.
    """
    from deepchem.metrics.score_function import mae_score
    return mae_score(y_true, y_pred)


################################################################################
# Source: deepchem.metrics.score_function.pearson_r2_score
# File: deepchem/metrics/score_function.py
# Category: valid
################################################################################

def deepchem_metrics_score_function_pearson_r2_score(
    y: numpy.ndarray,
    y_pred: numpy.ndarray
):
    """Computes the Pearson R^2 score: the square of the Pearson correlation coefficient between ground-truth and predicted continuous values. In the DeepChem context this is used as a simple regression-quality metric for tasks such as predicting molecular properties in drug discovery, materials science, quantum chemistry, and biology; it quantifies the strength of a linear relationship between true labels and model predictions and (under the usual assumptions) corresponds to the fraction of variance in y linearly explained by y_pred.
    
    Args:
        y (numpy.ndarray): Ground truth array of continuous target values. This should contain the experimentally observed or reference scalar property values for each example in a regression task (for example, measured binding affinities or physical properties). The array is expected to be numeric and aligned elementwise with y_pred; mismatched lengths or incompatible shapes will cause scipy.stats.pearsonr to raise an error.
        y_pred (numpy.ndarray): Predicted array of continuous values from a model. This should contain the model-predicted scalar property values corresponding one-to-one with entries in y. The practical role of y_pred is to be compared to y to evaluate linear agreement; poor alignment indicates a model that fails to capture linear trends in the target property.
    
    Returns:
        float: The Pearson-R^2 score, computed as (pearsonr(y, y_pred)[0])**2 using scipy.stats.pearsonr. This value is the square of the Pearson correlation coefficient between y and y_pred and (when finite) is in the range [0, 1], where values closer to 1 indicate stronger linear correlation and more explained variance. If one or both inputs are constant, scipy may return NaN for the correlation (and the squared result will be NaN); if inputs have length < 2 or other invalid shapes, scipy.stats.pearsonr may raise an exception (for example ValueError). There are no side effects; the function is deterministic and simply delegates computation to SciPy.
    """
    from deepchem.metrics.score_function import pearson_r2_score
    return pearson_r2_score(y, y_pred)


################################################################################
# Source: deepchem.metrics.score_function.pearsonr
# File: deepchem/metrics/score_function.py
# Category: valid
################################################################################

def deepchem_metrics_score_function_pearsonr(y: numpy.ndarray, y_pred: numpy.ndarray):
    """deepchem.metrics.score_function.pearsonr computes the Pearson correlation coefficient between a ground-truth array and a predicted array. This metric is commonly used in DeepChem to evaluate the linear agreement between model predictions and experimental or reference values in domains such as drug discovery, materials science, quantum chemistry, and biology. The function delegates computation to scipy.stats.pearsonr and returns only the correlation coefficient (the first element of scipy.stats.pearsonr), not the associated p-value.
    
    Args:
        y (numpy.ndarray): Ground truth array containing observed target values used to evaluate model performance. In DeepChem workflows this typically represents experimental measurements or reference labels for molecular properties or other continuous targets. The array should be a one-dimensional numeric NumPy array whose length matches y_pred; if lengths differ or the input is otherwise invalid, scipy.stats.pearsonr will raise an exception.
        y_pred (numpy.ndarray): Predicted array containing model output values to be compared against y. This is the array of continuous predictions produced by a regression model in DeepChem and must be a one-dimensional numeric NumPy array of the same length as y. The function will pass this array directly to scipy.stats.pearsonr.
    
    Returns:
        float: The Pearson correlation coefficient, a scalar in the range [-1.0, 1.0] that quantifies linear correlation: 1.0 indicates perfect positive linear correlation, -1.0 indicates perfect negative linear correlation, and values near 0 indicate little or no linear correlation. Note that the function returns only the correlation coefficient and does not return the p-value. If inputs contain constant values, non-finite entries, insufficient observations, or otherwise violate scipy.stats.pearsonr's requirements, the underlying SciPy call may issue warnings (for example, ConstantInputWarning) or raise exceptions (for example, ValueError); such warnings/exceptions are propagated to the caller. There are no other side effects.
    """
    from deepchem.metrics.score_function import pearsonr
    return pearsonr(y, y_pred)


################################################################################
# Source: deepchem.metrics.score_function.pixel_error
# File: deepchem/metrics/score_function.py
# Category: valid
################################################################################

def deepchem_metrics_score_function_pixel_error(y: numpy.ndarray, y_pred: numpy.ndarray):
    """deepchem.metrics.score_function.pixel_error computes a pixel-wise error metric for image-like arrays used in DeepChem image/segmentation evaluation tasks (for example comparing predicted segmentation masks from a model to ground-truth masks in biological microscopy or molecular imaging workflows). The metric is defined as 1 - f1_score(y, y_pred), where f1_score denotes the maximal pixel-wise F-score of similarity between the ground truth and prediction (the original implementation describes this as "1 - the maximal F-score of pixel similarity, or squared Euclidean distance between the original and the result labels"). This function is intended to quantify disagreement between two image-label arrays on a scale from 0 to 1, with lower values indicating better agreement.
    
    Args:
        y (numpy.ndarray): Ground truth array of pixel labels used as the reference in image comparison tasks. In DeepChem workflows this typically represents a labeled segmentation or mask produced by experimental annotation or a high-fidelity simulation. The array must be a numeric numpy.ndarray compatible with the underlying f1_score routine; values are interpreted in a pixel-wise manner to compute true positives/false positives/false negatives. The function does not modify this array.
        y_pred (numpy.ndarray): Predicted array of pixel labels produced by a model or post-processing pipeline in DeepChem image tasks. This should correspond elementwise to y (i.e., represent the same pixel grid and label semantics); the array must be a numeric numpy.ndarray compatible with the underlying f1_score routine. If y_pred contains continuous scores rather than discrete labels, callers should apply any required thresholding before calling this function because the returned metric reflects the interpretation used by f1_score. The function does not modify this array.
    
    Returns:
        float: The pixel-error value computed as 1 - f1_score(y, y_pred). By design this value ranges between 0 and 1 where 0 indicates a perfect pixel-wise match between y and y_pred and values closer to 1 indicate greater disagreement. There are no side effects: inputs are not mutated and no global state is changed.
    
    Behavior, defaults, and failure modes:
        This is a pure evaluation function with no internal state or defaults. It delegates the core similarity computation to the f1_score implementation available in the module's context and returns one minus that score. If y and y_pred are not compatible for pixel-wise comparison (for example differing shapes, incompatible dtypes, or containing NaNs), the underlying f1_score or numpy operations may raise an exception; callers should ensure inputs are preprocessed into comparable, valid numpy.ndarrays (e.g., same grid dimensions and appropriate label encoding). If y and y_pred encode continuous probabilities instead of discrete labels, the metric may be misleading unless the user thresholds or otherwise converts predictions to the expected label format prior to calling this function.
    """
    from deepchem.metrics.score_function import pixel_error
    return pixel_error(y, y_pred)


################################################################################
# Source: deepchem.metrics.score_function.prc_auc_score
# File: deepchem/metrics/score_function.py
# Category: valid
################################################################################

def deepchem_metrics_score_function_prc_auc_score(y: numpy.ndarray, y_pred: numpy.ndarray):
    """deepchem.metrics.score_function.prc_auc_score computes the area under the precision-recall curve (AUPRC) for the positive class (class index 1) from true labels and predicted class probabilities. This function is used in DeepChem evaluation pipelines (for example in molecular machine learning and drug-discovery model assessment) to quantify classifier performance on imbalanced binary or multi-class tasks by summarizing precision and recall trade-offs into a single scalar between 0 and 1.
    
    Args:
        y (numpy.ndarray): A numpy array containing true labels. The implementation slices the second column (y[:, 1]) to obtain the positive-class labels, so the expected and supported layout is a 2-D array of shape (N, n_classes) where the second column encodes the positive-class ground truth for each of N examples. The original documentation also allowed a 1-D array of shape (N,), but the current implementation will attempt to index y[:, 1] and will raise IndexError if y is strictly one-dimensional; in that case callers should convert binary labels to a 2-D one-hot or two-column format before calling this function. Values should represent class membership (e.g., 0/1 indicators for the positive class) appropriate for precision-recall computation in classification tasks common to DeepChem (drug discovery, materials, bioinformatics).
        y_pred (numpy.ndarray): A numpy array of predicted class probabilities with shape (N, n_classes). The function uses the second column (y_pred[:, 1]) as the predicted probability for the positive class and computes the precision-recall curve from those probabilities. Each row corresponds to a single example and the second column should represent the model's estimated probability for the positive class. Probabilities are expected as floating-point values typically in the [0, 1] range produced by softmax/sigmoid outputs; mismatched shapes between y and y_pred or non-probabilistic scores may yield misleading results.
    
    Returns:
        float: The computed area under the precision-recall curve (AUPRC) for the positive class (class index 1). This value is computed by first calling sklearn.metrics.precision_recall_curve on y[:, 1] and y_pred[:, 1], and then integrating recall vs. precision with sklearn.metrics.auc. The return value is a scalar between 0 and 1, where higher values indicate better precision-recall trade-off for the positive class. If input shapes are incompatible (for example, y is 1-D or either array has fewer than two columns, or N differs between y and y_pred), the function may raise IndexError or ValueError arising from attempted indexing or shape mismatches; there are no side effects.
    """
    from deepchem.metrics.score_function import prc_auc_score
    return prc_auc_score(y, y_pred)


################################################################################
# Source: deepchem.metrics.score_function.rms_score
# File: deepchem/metrics/score_function.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for rms_score because the docstring has no description for the argument 'y_true'
################################################################################

def deepchem_metrics_score_function_rms_score(y_true: numpy.ndarray, y_pred: numpy.ndarray):
    """deepchem.metrics.score_function.rms_score computes the root-mean-square (RMS) error between true values and predictions.
    
    This function returns the square root of the mean squared error between y_true and y_pred and is typically used in DeepChem workflows to assess regression model performance in domains such as drug discovery, materials science, quantum chemistry, and biology. The RMS error is a single non-negative float that summarizes the typical magnitude of prediction errors in the same units as the target values; lower values indicate better agreement between predictions and observations. Implementation-wise, the function computes np.sqrt(mean_squared_error(y_true, y_pred)).
    
    Args:
        y_true (numpy.ndarray): Ground-truth target values provided as a NumPy array. In DeepChem regression tasks this represents experimental or reference quantities (for example, binding affinities, formation energies, or other molecular properties) against which model predictions are compared. The array values must be numeric; if they contain NaNs the result may be NaN. y_true must be shape-compatible with y_pred for elementwise comparison; if shapes are incompatible, the underlying mean-squared-error computation will raise an error.
        y_pred (numpy.ndarray): Predicted target values produced by a model, provided as a NumPy array with the same shape and numeric dtype as y_true. In practice this is the model output you want to evaluate (for example predicted energies or activity scores). As with y_true, presence of NaNs will propagate and incompatible shapes will cause an error.
    
    Returns:
        float: The RMS error as a Python float. This is the non-negative square root of the mean squared error between y_true and y_pred. The returned value has the same physical units as the target variable and provides a scalar summary of prediction accuracy: values closer to 0 indicate better predictions. There are no side effects. Failure modes include exceptions raised by the underlying mean-squared-error calculation when inputs are non-numeric or have incompatible shapes; NaN values in the inputs will generally produce NaN output.
    """
    from deepchem.metrics.score_function import rms_score
    return rms_score(y_true, y_pred)


################################################################################
# Source: deepchem.models.losses.get_negative_expectation
# File: deepchem/models/losses.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def deepchem_models_losses_get_negative_expectation(
    q_samples: torch.Tensor,
    measure: str = "JSD",
    average_loss: bool = True
):
    """Compute the negative component of a divergence or difference from per-sample critic scores.
    
    This function implements the "negative expectation" terms used in variational estimators of divergences and adversarial losses. It transforms a tensor of negative-sample critic outputs (q_samples) into the negative part of a chosen divergence/difference according to standard formulas used for GANs, f-divergences, and related objectives. In DeepChem, these terms are used as components of loss functions when training models (for example, adversarial discriminators or mutual-information estimators) in molecular machine learning, drug discovery, and other life-science deep-learning tasks. The function is pure (no side effects) and returns either a per-sample tensor of negative terms or their mean, depending on average_loss.
    
    Args:
        q_samples (torch.Tensor): Negative samples. A torch.Tensor containing critic scores or log-density-ratio estimates computed for negative examples. Each element of this tensor is interpreted independently as the scalar score for one negative sample; the function applies elementwise or reduction operations on this tensor to produce the negative part of the divergence/difference. The input must be a torch.Tensor; passing another type may raise a TypeError when torch operations are executed.
        measure (str): The divergence/difference measure to use. This string selects the functional form applied to q_samples. Supported, case-sensitive values and their behaviors implemented in the function are:
            'GAN' : softplus(-q) + q (standard GAN negative term),
            'JSD' : softplus(-q) + q - log(2) (Jensen-Shannon-type term; default),
            'X2'  : -0.5 * ((sqrt(q^2) + 1)^2) (Pearson X^2 style term),
            'KL'  : exp(q) (Kullback-Leibler style),
            'RKL' : q - 1 (reverse KL style),
            'DV'  : log_sum_exp(q_samples, 0) - log(N) (Donsker-Varadhan style; uses a log-sum-exp across the first dimension and subtracts log of the number of samples),
            'H2'  : exp(q) - 1 (squared Hellinger style),
            'W1'  : q (Wasserstein-1 linear term).
          The default value is 'JSD'. If an unsupported value is provided, the function raises a ValueError identifying the unknown measure.
        average_loss (bool): Average the result over samples when True. If True (the default), the function returns the mean of the computed negative-term tensor as a scalar torch.Tensor. If False, the function returns the elementwise per-sample tensor of negative terms with the same shape as q_samples. This flag controls whether the caller receives an aggregate scalar loss (typical for optimization steps) or per-sample contributions (useful for debugging, per-example weighting, or custom reductions).
    
    Returns:
        torch.Tensor: The negative part of the divergence/difference computed from q_samples according to the selected measure. If average_loss is True, a scalar torch.Tensor containing the mean negative term across all elements of q_samples is returned. If average_loss is False, a torch.Tensor of the same shape as q_samples is returned containing the per-sample negative terms. This returned tensor is intended to be combined with the corresponding positive expectation term in divergence estimation or used directly as a loss component in training.
    
    Behavior, defaults, and failure modes:
        The function uses standard torch operations and a Python constant log(2.) for the 'JSD' adjustment. It performs no in-place mutations and has no side effects. The default behavior is to compute the Jensen-Shannon (JSD) negative term and return its mean (measure='JSD', average_loss=True). If measure equals 'DV', the function calls a log-sum-exp reduction and subtracts math.log(q_samples.size(0)); passing an empty q_samples (size 0 along the reduction dimension) will cause a math domain error (log(0)) or other runtime error. If measure is not one of the supported strings, a ValueError is raised. If q_samples is not a torch.Tensor or has incompatible dtype/device for the torch operations used, downstream torch operations will raise appropriate TypeError/RuntimeError. The function does not perform device transfers; the caller is responsible for ensuring q_samples resides on the intended device (CPU/GPU) before calling.
    """
    from deepchem.models.losses import get_negative_expectation
    return get_negative_expectation(q_samples, measure, average_loss)


################################################################################
# Source: deepchem.models.losses.get_positive_expectation
# File: deepchem/models/losses.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def deepchem_models_losses_get_positive_expectation(
    p_samples: torch.Tensor,
    measure: str = "JSD",
    average_loss: bool = True
):
    """Computes the positive part of a divergence or difference used in unsupervised losses.
    
    This function is used in DeepChem's unsupervised loss and divergence estimators (for example when implementing GAN losses or mutual-information estimators) to convert model outputs for positive samples into the "positive expectation" term of a chosen divergence measure. The input p_samples is expected to contain scalar scores or critic outputs for positive examples (for instance discriminator logits or critic values produced during training). The function supports several common measures and implements the exact per-measure transformations used in DeepChem's loss routines. By default it returns the mean over samples, which is suitable for use directly as a scalar loss term during optimization.
    
    Args:
        p_samples (torch.Tensor): Positive samples. A PyTorch tensor containing the model’s scores or critic outputs for the positive examples. The tensor is used elementwise by the selected divergence formula; standard PyTorch broadcasting and dtype/device semantics apply. This argument is required.
        measure (str): The divergence measure to use for the unsupervised loss. Options are 'GAN', 'JSD', 'KL', 'RKL', 'X2', 'DV', 'H2', or 'W1'. The default is 'JSD'. Each measure maps p_samples to the positive expectation Ep as implemented in the function:
            - 'GAN': Ep = -softplus(-p_samples)
            - 'JSD': Ep = log(2) - softplus(-p_samples)
            - 'X2': Ep = p_samples**2
            - 'KL': Ep = p_samples + 1.
            - 'RKL': Ep = -exp(-p_samples)
            - 'DV': Ep = p_samples
            - 'H2': Ep = 1. - exp(-p_samples)
            - 'W1': Ep = p_samples
          The measure parameter selects which of these formulas is applied. If an unknown measure string is provided, the function raises a ValueError.
        average_loss (bool): Average the result over samples. If True (the default), the function returns the mean of Ep across all elements in p_samples, producing a scalar tensor suitable as a loss value for optimization. If False, the function returns Ep with the same elementwise shape as p_samples so callers can apply custom reductions or weighting.
    
    Returns:
        torch.Tensor: The positive part of the divergence/difference (Ep) computed from p_samples using the selected measure. If average_loss is True, this is a scalar tensor equal to Ep.mean(); if average_loss is False, this is a tensor of the same elementwise shape as p_samples containing Ep for each input element. The returned tensor is produced by standard PyTorch operations and therefore will follow PyTorch rules for dtype and device.
    
    Raises:
        ValueError: If measure is not one of the supported strings ('GAN', 'JSD', 'KL', 'RKL', 'X2', 'DV', 'H2', 'W1'), a ValueError is raised with a message identifying the unknown measure.
    
    Behavior and side effects:
        This function is pure and has no side effects beyond computing and returning a tensor. It uses torch.nn.functional.softplus and torch.exp internally where specified by the measure. It does not perform in-place modification of p_samples. Typical usage in DeepChem is to compute the positive contribution to a divergence-based unsupervised loss (for example in GAN training or mutual information objectives) that is then combined with a corresponding negative expectation to form a complete loss.
    """
    from deepchem.models.losses import get_positive_expectation
    return get_positive_expectation(p_samples, measure, average_loss)


################################################################################
# Source: deepchem.models.losses.log_sum_exp
# File: deepchem/models/losses.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def deepchem_models_losses_log_sum_exp(x: torch.Tensor, axis: int = None):
    """deepchem.models.losses.log_sum_exp computes a numerically stable log-sum-exp reduction over a specified axis. This function is intended for use in DeepChem loss and model computations where one needs to aggregate scores or logits across a dimension (for example computing log-partition functions, stable denominators for softmax, or reductions used in cross-entropy and likelihood calculations in molecular machine learning tasks).
    
    Args:
        x (torch.Tensor): Input tensor containing raw scores, logits, or real-valued quantities produced by a model or intermediate computation. In the DeepChem context, x is typically a floating-point tensor representing per-example or per-class values (for example, model outputs for each class or energy scores over a set of states). The function performs no in-place modification of x; it reads x to compute a new tensor y.
        axis (int): Axis (dimension) along which to perform the log-sum-exp reduction. This should be an integer indexing a valid dimension of x. The implementation applies a numerically stable algorithm by first taking the maximum along axis, subtracting that maximum from x, exponentiating, summing over axis, taking the logarithm, and finally adding the maximum back. The signature provides a default value of None, but passing axis=None will cause torch.max and the subsequent reduction calls in the implementation to raise an error (TypeError or IndexError) because an integer dimension is required. Supplying an integer outside the valid range for x will similarly raise an IndexError. The caller is responsible for providing an appropriate integer axis.
    
    Returns:
        y (torch.Tensor): A new tensor containing the log-sum-exp of x computed over the specified axis. The returned tensor has the same dtype as the intermediate computations and has one fewer dimension than x (the reduction axis is removed). Numerically, y is computed as log(sum(exp(x))) along axis in a stable way by subtracting the per-axis maximum before exponentiation to avoid overflow and then adding it back. If x is not a torch.Tensor or axis is invalid, the function will raise the underlying torch error (TypeError or IndexError) rather than returning a value.
    """
    from deepchem.models.losses import log_sum_exp
    return log_sum_exp(x, axis)


################################################################################
# Source: deepchem.molnet.dnasim.motif_density
# File: deepchem/molnet/dnasim.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def deepchem_molnet_dnasim_motif_density(
    motif_name: str,
    seq_length: int,
    num_seqs: int,
    min_counts: int,
    max_counts: int,
    GC_fraction: float,
    central_bp: int = None
):
    """deepchem.molnet.dnasim.motif_density generates synthetic DNA sequences containing a specified motif at a controlled density and returns both the generated nucleotide sequences and their corresponding embedding metadata. This function uses the simdna library and the ENCODE motif collection (via simdna.ENCODE_MOTIFS_PATH) to sample motif instances, places a sampled number of motif occurrences into each sequence at positions chosen either uniformly or within a central region, and embeds these motifs into a zero-order background sequence whose nucleotide distribution is determined by the provided GC fraction. This is commonly used in DeepChem workflows for creating labeled synthetic genomic datasets for model training and evaluation in tasks such as motif detection and regulatory sequence modeling.
    
    Args:
        motif_name (str): Name/identifier of the motif to embed. This must match an entry in the ENCODE motif collection loaded by simdna.synthetic.LoadedEncodeMotifs. The function constructs a PwmSamplerFromLoadedMotifs using pseudocountProb=0.001 to sample subsequences matching this motif. If the motif_name is not found, simdna will raise an error when attempting to construct the sampler.
        seq_length (int): Length of each generated DNA sequence in nucleotides. This value is passed to simdna.synthetic.ZeroOrderBackgroundGenerator to create background sequences of this length. If seq_length is incompatible with internal placement constraints or simdna expectations, simdna may raise an error.
        num_seqs (int): Number of sequences to generate. The function calls synthetic.GenerateSequenceNTimes(..., num_seqs).generateSequences() to produce exactly this many sequences; the return arrays have one entry per generated sequence.
        min_counts (int): Lower bound used to construct a UniformIntegerGenerator that determines how many motif instances to embed in each sequence. Each sequence's motif count is randomly drawn from the generator according to simdna's UniformIntegerGenerator behavior.
        max_counts (int): Upper bound used to construct a UniformIntegerGenerator that determines how many motif instances to embed in each sequence. Combined with min_counts, this controls the per-sequence motif density sampled for each generated sequence.
        GC_fraction (float): Fraction of guanine and cytosine bases used to define a discrete nucleotide distribution for the ZeroOrderBackgroundGenerator. The function passes this value into get_distribution(GC_fraction) to obtain the background distribution. Invalid or out-of-range values for GC_fraction may cause the distribution construction or simdna background generator to raise an error.
        central_bp (int): Optional index of the central base pair used to constrain motif placement. If provided (not None), the function uses synthetic.InsideCentralBp(central_bp) to bias motif positions to lie inside a region around this central base; if None, synthetic.UniformPositionGenerator() is used and motif positions are sampled uniformly across the sequence. This parameter controls positional bias of embedded motifs and defaults to None (uniform placement).
    
    Returns:
        tuple: A tuple (sequence_arr, embedding_arr) where:
            sequence_arr (numpy.ndarray): One-dimensional NumPy array with length equal to num_seqs. Each element is the generated nucleotide sequence for a sample produced by simdna (typically a string or sequence-like object as returned by simdna). These sequences include embedded motif instances placed according to the sampled counts and chosen position generator; they are generated against a zero-order background with the requested GC_fraction.
            embedding_arr (list): A list of length num_seqs where each element is the embedding metadata produced by simdna for the corresponding generated sequence (the generated_seq.embeddings values). These embedding objects describe the locations, orientations (including reverse complements when used), and identity of embedded motif occurrences and are intended for downstream use in tasks that require ground-truth motif locations (for example, evaluating motif detection models).
    
    Behavior and side effects:
        - The function loads motif definitions from simdna.ENCODE_MOTIFS_PATH using LoadedEncodeMotifs(..., pseudocountProb=0.001). This reads motif data from the environment configured by simdna and may raise file I/O errors if the path is unavailable.
        - The function constructs samplers and embedders from simdna.synthetic classes: PwmSamplerFromLoadedMotifs, ReverseComplementWrapper, SubstringEmbedder, RepeatedEmbedder, UniformPositionGenerator or InsideCentralBp, UniformIntegerGenerator, ZeroOrderBackgroundGenerator, EmbedInABackground, and GenerateSequenceNTimes. Behavior and exceptions from those classes propagate to the caller.
        - No files are written by this function; it returns generated data in memory.
        - The function relies on simdna and numpy being importable; ImportError will occur if these dependencies are missing.
        - Typical failure modes include: motif_name not present in the loaded motif collection, invalid GC_fraction leading to a malformed distribution, seq_length or count bounds incompatible with simdna placement logic, or other errors raised by simdna when generating sequences.
    
    Notes on practical significance:
        - This routine is intended for creating synthetic datasets used in DeepChem workflows for genomics and regulatory sequence modeling, enabling controlled experiments where the number and positions of motif instances are known and can be used as ground truth for model training or evaluation.
        - Embedding metadata (embedding_arr) is crucial for supervised tasks that require exact motif locations (for example, constructing positive labels for motif detection), while sequence_arr supplies the corresponding nucleotide inputs.
    """
    from deepchem.molnet.dnasim import motif_density
    return motif_density(
        motif_name,
        seq_length,
        num_seqs,
        min_counts,
        max_counts,
        GC_fraction,
        central_bp
    )


################################################################################
# Source: deepchem.molnet.dnasim.simple_motif_embedding
# File: deepchem/molnet/dnasim.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def deepchem_molnet_dnasim_simple_motif_embedding(
    motif_name: str,
    seq_length: int,
    num_seqs: int,
    GC_fraction: float
):
    """deepchem.molnet.dnasim.simple_motif_embedding simulates DNA sequences and returns synthetic sequences together with their embedding metadata by embedding an ENCODE motif (or no motif) anywhere in each sequence. This function is used in DeepChem's molnet dnasim utilities to generate controlled, labeled synthetic datasets for sequence-based model development, benchmarking motif discovery methods, or testing classifiers that consume DNA sequence strings and known motif embeddings.
    
    Args:
        motif_name (str): Name of the motif to embed from the ENCODE motif collection. When a non-None motif_name is provided, the function loads motifs from simdna.ENCODE_MOTIFS_PATH using synthetic.LoadedEncodeMotifs and constructs a PWM sampler (synthetic.PwmSamplerFromLoadedMotifs) for the specified motif. The sampler is wrapped with synthetic.ReverseComplementWrapper so that the motif may be embedded on either strand. If motif_name is None, no motif embedder is created and generated sequences contain only background sequence. Supplying a motif_name that is not present in the loaded ENCODE motifs or if simdna cannot read the ENCODE motif files will raise an error from simdna (for example a KeyError or an I/O-related exception raised by the underlying simdna routines).
        seq_length (int): Length of each simulated DNA sequence in nucleotides. This is passed to simdna.ZeroOrderBackgroundGenerator to produce background sequences of exactly seq_length bases. The returned sequence strings in sequence_arr will each have this length unless simdna raises an exception. Providing an invalid seq_length (for example a non-integer or an out-of-range value) will cause simdna to raise an exception when constructing the background generator.
        num_seqs (int): Number of sequences to generate. This numeric argument is passed to synthetic.GenerateSequenceNTimes to control how many independent sequences (and corresponding embedding metadata objects) are produced. If num_seqs is zero or negative, behavior is determined by simdna (it may return an empty result or raise an error); callers should provide a positive integer to obtain a non-empty dataset.
        GC_fraction (float): Fraction of G+C content used to parameterize the discrete background base distribution. This float is converted (via get_distribution(GC_fraction)) into a discreteDistribution argument for synthetic.ZeroOrderBackgroundGenerator to bias background nucleotide sampling toward the specified GC composition. If get_distribution or simdna rejects the supplied GC_fraction (for example if it is not a float or is outside the domain expected by those utilities), an exception will be raised by those functions.
    
    Returns:
        sequence_arr (1darray): 1darray containing sequence strings. Each element is a Python str representing a simulated DNA sequence of length seq_length generated by simdna. If a motif_name was provided, the motif has been embedded somewhere within each sequence according to the embedding strategy; if motif_name is None, these are pure background sequences. This array is produced by extracting generated_seq.seq for each sequence object returned by simdna's GenerateSequenceNTimes.
        embedding_arr (1darray): 1darray (list-like) of embedding objects corresponding to each sequence in sequence_arr. Each element is the embeddings attribute from simdna's generated sequence object and encodes where and how motifs were embedded (positions, orientation, and any sampler metadata). These embedding objects are produced by simdna and are useful for downstream evaluation (for example, comparing predicted motif positions to the ground-truth embeddings).
    
    Behavior and side effects:
        This function imports and uses the external simdna library and its synthetic submodule and reads motif definitions from simdna.ENCODE_MOTIFS_PATH. Calling this function will trigger file I/O when loading ENCODE motifs. If simdna is not installed or cannot load its motifs, an ImportError or an exception from simdna will be raised. The function constructs a background generator using the provided GC_fraction via get_distribution(GC_fraction) (get_distribution is expected to be available in the same module context). The function may raise exceptions coming from simdna (e.g., when a motif name is invalid, when file I/O fails, or when provided parameter types are incompatible with simdna APIs). No files are written by this function; it only returns in-memory arrays of sequences and embedding metadata.
    """
    from deepchem.molnet.dnasim import simple_motif_embedding
    return simple_motif_embedding(motif_name, seq_length, num_seqs, GC_fraction)


################################################################################
# Source: deepchem.molnet.dnasim.simulate_differential_accessibility
# File: deepchem/molnet/dnasim.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def deepchem_molnet_dnasim_simulate_differential_accessibility(
    pos_motif_names: list,
    neg_motif_names: list,
    seq_length: int,
    min_num_motifs: int,
    max_num_motifs: int,
    num_pos: int,
    num_neg: int,
    GC_fraction: float
):
    """Generate simulated DNA sequences, binary labels, and motif-embedding objects for a differential accessibility
    binary classification task used in DeepChem's MolNet dnasim utilities. This function constructs two sets of
    simulated sequences (positive and negative) by embedding motifs named in pos_motif_names and neg_motif_names,
    respectively, using the helper simulate_multi_motif_embedding. The outputs are concatenated so that all positive
    examples appear first followed by all negative examples. These outputs are intended to be used as synthetic
    training or evaluation data for models studying regulatory DNA accessibility and motif-driven sequence signals.
    
    Args:
        pos_motif_names (list): List of strings. Names of motifs to embed into sequences designated as the
            positive class. Each name should correspond to a motif definition that simulate_multi_motif_embedding
            recognizes. In the DeepChem differential accessibility simulation domain, these motifs represent sequence
            patterns that confer accessibility and thus provide the discriminative signal for positive examples.
        neg_motif_names (list): List of strings. Names of motifs to embed into sequences designated as the
            negative class. These motif names are used in the same manner as pos_motif_names but define signals
            characteristic of the negative class for the simulated task.
        seq_length (int): Integer specifying the length of each simulated DNA sequence in base pairs. This determines
            the dimensionality of the sequence features that downstream models will consume.
        min_num_motifs (int): Integer minimum number of motif instances to insert into each simulated sequence.
            Controls the sparsity/abundance of motif signal per sequence; must be coherent with max_num_motifs.
        max_num_motifs (int): Integer maximum number of motif instances to insert into each simulated sequence.
            Controls the upper bound on motif occurrences per sequence; simulate_multi_motif_embedding will place
            between min_num_motifs and max_num_motifs motifs per sequence.
        num_pos (int): Integer number of positive-class sequences to generate. Determines how many sequences will be
            produced using pos_motif_names and therefore the number of True labels in the returned label array.
        num_neg (int): Integer number of negative-class sequences to generate. Determines how many sequences will be
            produced using neg_motif_names and therefore the number of False labels in the returned label array.
        GC_fraction (float): Float specifying the target GC content used when generating background sequence composition.
            This parameter controls background nucleotide composition during simulation so that generated sequences
            reflect a chosen GC bias (expressed as a fraction).
    
    Behavior and side effects:
        This function internally calls simulate_multi_motif_embedding twice: once to generate num_pos sequences and
        corresponding embedding objects for the positive motif set, and once to generate num_neg sequences and
        embedding objects for the negative motif set. It then concatenates the positive and negative sequence arrays
        (positive sequences first), constructs a label array with True entries for positive examples followed by False
        entries for negative examples, and concatenates the corresponding embedding object arrays. The function does
        not modify global state; all side effects are limited to allocating and returning the generated arrays. Errors
        and exceptions raised by simulate_multi_motif_embedding (for example, due to invalid motif names or invalid
        parameter types) will propagate to the caller. Logical parameter errors such as providing min_num_motifs greater
        than max_num_motifs or negative values for num_pos/num_neg will likely cause downstream failures or exceptions
        inside the embedding simulator.
    
    Returns:
        sequence_arr (1darray): Contains sequence strings. A one-dimensional array whose elements are simulated DNA
            sequence strings. The array is ordered with the num_pos positive sequences first followed by the num_neg
            negative sequences. These sequences serve as the feature inputs for differential accessibility models.
        y (1darray): Contains labels. A one-dimensional array of boolean labels aligned with sequence_arr where each
            element indicates class membership: True for sequences generated with pos_motif_names and False for
            sequences generated with neg_motif_names. The number of labels equals num_pos + num_neg.
        embedding_arr (1darray): Array of embedding objects. A one-dimensional array (or list-like container) of
            embedding objects produced by simulate_multi_motif_embedding for each sequence in sequence_arr. Each
            embedding object encodes the motif insertions (e.g., motif identities and positions) used to generate the
            corresponding sequence and can be used as ground truth for interpretation, attribution, or downstream
            evaluation tasks in the DeepChem MolNet dnasim workflow.
    """
    from deepchem.molnet.dnasim import simulate_differential_accessibility
    return simulate_differential_accessibility(
        pos_motif_names,
        neg_motif_names,
        seq_length,
        min_num_motifs,
        max_num_motifs,
        num_pos,
        num_neg,
        GC_fraction
    )


################################################################################
# Source: deepchem.molnet.dnasim.simulate_heterodimer_grammar
# File: deepchem/molnet/dnasim.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def deepchem_molnet_dnasim_simulate_heterodimer_grammar(
    motif1: str,
    motif2: str,
    seq_length: int,
    min_spacing: int,
    max_spacing: int,
    num_pos: int,
    num_neg: int,
    GC_fraction: float
):
    """deepchem.molnet.dnasim.simulate_heterodimer_grammar simulates a two-class DNA-sequence dataset used in DeepChem benchmarking and model development where the positive class follows a heterodimer grammar (two motifs placed with a constrained spacing) and the negative class contains the same motifs embedded independently (not forming the heterodimer grammar).
    
    This function constructs synthetic DNA sequences by loading ENCODE motifs via simdna, sampling motif instances for motif1 and motif2 (each with reverse-complement sampling enabled), embedding a paired motif arrangement for positive examples with an inter-motif separation drawn from the supplied spacing range, and embedding motif instances independently for negative examples using the helper simulate_multi_motif_embedding. The generated sequences, binary class labels, and per-sequence embedding metadata are returned for downstream training, evaluation, or analysis of sequence models.
    
    Args:
        motif1 (str): Name/identifier of the first motif to embed. This string is passed to simdna.PwmSamplerFromLoadedMotifs via a LoadedEncodeMotifs object that reads simdna.ENCODE_MOTIFS_PATH. The motif name must match an entry in the loaded ENCODE motif collection; if the name is not found, simdna will raise an error. In practice this parameter selects the PWM used to generate instances of the first motif for both positive (heterodimer) and negative (independent) sequences.
        motif2 (str): Name/identifier of the second motif to embed. Semantically identical to motif1 but selects the second motif in the paired heterodimer and the independent embeddings used for negative examples. Must match an entry in the same ENCODE motif collection; mismatches will propagate errors from simdna.
        seq_length (int): Length of each synthetic DNA sequence to generate (number of nucleotides). This integer controls the background sequence generator used by simdna.ZeroOrderBackgroundGenerator and therefore determines where motifs can be placed. Practically, seq_length should be large enough to accommodate both motif lengths plus the chosen inter-motif spacing; otherwise embedding or placement operations by simdna may fail or produce unexpected placement behavior.
        min_spacing (int): Minimum inter-motif spacing (in nucleotides) used when constructing positive-class heterodimer embeddings. The function uses a uniform integer generator to sample a separation value between min_spacing and max_spacing for each positive example. min_spacing should be compatible with seq_length and motif lengths; if min_spacing > max_spacing or values are incompatible with sequence length, simdna or the generator may raise an exception.
        max_spacing (int): Maximum inter-motif spacing (in nucleotides) used for positive-class heterodimer embeddings. Used together with min_spacing to sample the inter-motif separation for each positive example.
        num_pos (int): Number of positive-class sequences to generate. Positive sequences contain an embedded pair (motif1, motif2) arranged according to the heterodimer grammar (separation sampled from the spacing range). num_pos must be an integer; negative or non-integer values will cause errors from the underlying generators or numpy operations.
        num_neg (int): Number of negative-class sequences to generate. Negative sequences are produced by simulate_multi_motif_embedding and contain motif1 and motif2 embedded independently (i.e., without the enforced heterodimer spacing constraint). num_neg must be an integer; negative values or incompatible arguments may raise exceptions.
        GC_fraction (float): Fraction of G+C content to use when constructing the zero-order background nucleotide distribution. This float is passed to a distribution helper (get_distribution) used by simdna.ZeroOrderBackgroundGenerator. GC_fraction controls the overall nucleotide composition of the background sequence and therefore affects the baseline sequence statistics for both positive and negative examples.
    
    Behavior and side effects:
        - Loads ENCODE motifs from the simdna installation path (simdna.ENCODE_MOTIFS_PATH) using simdna.synthetic.LoadedEncodeMotifs with a small pseudocount; this is a side effect that requires the simdna motif resource to be available on disk.
        - Generates num_pos heterodimer-grammar sequences by embedding motif pairs with inter-motif separation sampled between min_spacing and max_spacing, using reverse-complement sampling for each motif.
        - Generates num_neg non-grammar (negative) sequences by calling simulate_multi_motif_embedding([motif1, motif2], seq_length, 2, 2, num_neg, GC_fraction), i.e., embedding the two motifs independently.
        - Concatenates positive then negative sequences, constructs corresponding labels, and aggregates embedding metadata for every generated sequence.
        - Uses stochastic sampling from simdna generators; results are non-deterministic unless simdna's random seed state is controlled externally. No random-seed parameter is provided by this function.
        - May raise exceptions propagated from simdna or numpy in cases such as missing motif names, incompatible spacing versus sequence length, non-integer inputs where integers are required, or other resource/IO errors when loading motif files.
    
    Returns:
        sequence_arr (1darray): Array of sequence strings for all generated examples. The ordering is positive-class sequences first (num_pos entries) followed by negative-class sequences (num_neg entries). Each element is a DNA sequence string produced by simdna (background plus embedded motif instances).
        y (1darray): Array with positive/negative class labels corresponding to sequence_arr. Labels follow the same ordering as sequence_arr (first num_pos positives, then num_neg negatives). The entries indicate class membership for downstream supervised training or evaluation.
        embedding_arr (list): List of embedding objects or embedding metadata corresponding to each sequence in sequence_arr. For positive-class sequences these are the embeddings produced by the paired-embedder; for negative-class sequences these are the embeddings returned by simulate_multi_motif_embedding. The list length equals len(sequence_arr) and can be used to locate motif insertion positions for analysis or visualization.
    
    Failure modes:
        - If motif1 or motif2 are not present in the ENCODE motif resource loaded by simdna, the call to PwmSamplerFromLoadedMotifs or subsequent sampling will raise an error.
        - If min_spacing > max_spacing, or if chosen spacings cannot be satisfied within seq_length given motif lengths, simdna's generator code may raise exceptions.
        - Non-integer or negative values for parameters expected to be integers (seq_length, min_spacing, max_spacing, num_pos, num_neg) may cause TypeError or ValueError from simdna or numpy operations.
        - IO or resource errors can occur if simdna cannot access its motif data files; these are raised by the underlying simdna code.
    
    Usage note:
        This function is intended for creating synthetic datasets for model development, ablation studies, and motif grammar research within DeepChem workflows. The output is suitable for training classifiers that must distinguish sequences following a heterodimer motif grammar from sequences containing the same motifs placed independently.
    """
    from deepchem.molnet.dnasim import simulate_heterodimer_grammar
    return simulate_heterodimer_grammar(
        motif1,
        motif2,
        seq_length,
        min_spacing,
        max_spacing,
        num_pos,
        num_neg,
        GC_fraction
    )


################################################################################
# Source: deepchem.molnet.dnasim.simulate_motif_counting
# File: deepchem/molnet/dnasim.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def deepchem_molnet_dnasim_simulate_motif_counting(
    motif_name: str,
    seq_length: int,
    pos_counts: list,
    neg_counts: list,
    num_pos: int,
    num_neg: int,
    GC_fraction: float
):
    """Generates a synthetic dataset for a DNA motif-counting classification task. This function is used in DeepChem's dnasim utilities to produce sequences, binary labels, and embedding objects suitable for training or evaluating models that count occurrences of a specific DNA motif in sequences (a common task in computational biology and regulatory genomics). The function builds a positive set where sequences contain motif counts sampled between pos_counts[0] and pos_counts[1] and a negative set where sequences contain motif counts sampled between neg_counts[0] and neg_counts[1], uses motif_density to synthesize sequences and associated embedding objects, concatenates the positive and negative sets, and returns the combined arrays for downstream model input or analysis. Note: the current implementation calls motif_density for both positive and negative sets using the same num_pos value for the number of sequences passed to motif_density; as a result, the number of generated negative sequences may not match num_neg unless num_pos == num_neg.
    
    Args:
        motif_name (str): Name or identifier of the DNA motif to be embedded into synthetic sequences. In practice this is a motif string or key recognized by the underlying motif_density routine; it directs which motif pattern will be inserted into sequences during data generation.
        seq_length (int): Length of each synthetic DNA sequence to generate (number of nucleotides). This determines sequence dimension for model inputs and affects how many motif instances can be accommodated per sequence.
        pos_counts (list): Two-element list interpreted as (min_counts, max_counts) specifying the inclusive range of motif counts to sample for each sequence in the positive class. These integers control the density of motif occurrences in the positive set.
        neg_counts (list): Two-element list interpreted as (min_counts, max_counts) specifying the inclusive range of motif counts to sample for each sequence in the negative class. These integers control the density of motif occurrences in the negative set.
        num_pos (int): Number of positive-class sequences requested. This value is passed to motif_density to produce the positive-set sequences and embeddings. It determines how many True labels will appear in the returned label array.
        num_neg (int): Number of negative-class sequences requested. Intended to determine how many False labels will appear in the returned label array. Note that due to the current implementation, motif_density is invoked for the negative set using num_pos (not num_neg), so the actual number of generated negative sequences may differ from num_neg unless they are equal.
        GC_fraction (float): Fractional parameter that controls the background nucleotide composition (proportion of G/C versus A/T) used when synthesizing sequences via motif_density. This adjusts the base composition of background sequence context around motifs and thereby affects motif detectability and model behavior.
    
    Returns:
        sequence_arr : 1darray
            Array of sequence strings produced by motif_density. The array contains all positive-class sequences first (generated from pos_counts and num_pos) followed by all negative-class sequences (generated from neg_counts). These sequences are ready for featurization or direct use in sequence-based models.
        y : 1darray
            Array of labels aligned with sequence_arr. Entries indicate class membership for each sequence: True for sequences generated using pos_counts (positive class) and False for sequences generated using neg_counts (negative class). The label array order matches sequence_arr (positives first, then negatives).
        embedding_arr: 1darray
            Array (or list-like 1d array) of embedding objects returned by motif_density for each generated sequence. Embeddings for the positive set are concatenated with embeddings for the negative set in the same order as sequence_arr, allowing downstream code to use embeddings paired with sequences and labels.
    
    Behavior and side effects:
        The function calls motif_density twice to synthesize sequences and embeddings for positive and negative classes, concatenates the resulting sequence arrays using numpy.concatenate, constructs a boolean label array with num_pos True entries followed by num_neg False entries, and concatenates embedding arrays using list concatenation (positive_embedding_arr + negative_embedding_arr). Because motif_density is invoked for the negative set with num_pos (not num_neg) in the current implementation, the actual lengths of the generated negative sequence and embedding lists may differ from num_neg; however, the returned label array uses num_neg to determine how many False labels to include, which can lead to a mismatch between sequence_arr length and y length if num_pos != num_neg. motif_density may perform its own validation and can raise exceptions; this function propagates those exceptions.
    
    Failure modes and validation notes:
        If pos_counts or neg_counts are not two-element lists of integers interpreted as (min_counts, max_counts), motif_density or downstream code may raise a ValueError or TypeError. If seq_length is too small to accommodate the requested motif counts or motif pattern, motif_density may raise an error or produce sequences that do not contain the requested counts. If num_pos or num_neg are negative or non-integer, behavior is undefined and may raise an exception. Users should ensure motif_name is valid for the underlying motif library used by motif_density and that GC_fraction is provided in a form accepted by motif_density. Because of the implementation detail noted above (negative set generation uses num_pos), callers should verify that the lengths of the returned arrays match expectations and, if necessary, pass equal values for num_pos and num_neg or modify the source to use num_neg for the negative call.
    """
    from deepchem.molnet.dnasim import simulate_motif_counting
    return simulate_motif_counting(
        motif_name,
        seq_length,
        pos_counts,
        neg_counts,
        num_pos,
        num_neg,
        GC_fraction
    )


################################################################################
# Source: deepchem.molnet.dnasim.simulate_motif_density_localization
# File: deepchem/molnet/dnasim.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def deepchem_molnet_dnasim_simulate_motif_density_localization(
    motif_name: str,
    seq_length: int,
    center_size: int,
    min_motif_counts: int,
    max_motif_counts: int,
    num_pos: int,
    num_neg: int,
    GC_fraction: float
):
    """Simulates two classes of DNA sequences used for benchmarking motif-detection and localization tasks. The function produces a set of "positive" sequences that contain multiple instances of a named motif concentrated inside the central region of each sequence, and a set of "negative" sequences that contain multiple instances of the same motif placed anywhere in the sequence. The number of motif instances placed in each sequence is sampled uniformly between the provided minimum and maximum motif counts. This routine is used by DeepChem's molnet dnasim utilities to generate synthetic datasets for model development and evaluation (for example, models that must detect motif presence and learn positional localization).
    
    Args:
        motif_name (str): encode motif name. Identifier for the motif to implant into sequences; passed verbatim to the underlying motif generation routine (motif_density) to select the motif model and its sequence representation. In practice this string names a motif from the motif library used by the simulator.
        seq_length (int): length of sequence. Total number of nucleotides in each output sequence; all generated sequences will have this fixed length.
        center_size (int): length of central part of the sequence where motifs can be positioned. For the positive class, motif instances are restricted to positions inside the central window of this length; for the negative class this parameter is ignored by motif_density and motifs may appear anywhere.
        min_motif_counts (int): minimum number of motif instances. Lower bound (inclusive) of motif instances to place in each simulated sequence; the actual count per sequence is drawn uniformly between this value and max_motif_counts.
        max_motif_counts (int): maximum number of motif instances. Upper bound (inclusive) of motif instances to place in each simulated sequence; the actual count per sequence is drawn uniformly between min_motif_counts and this value.
        num_pos (int): number of positive class sequences. Number of sequences to generate where motif instances are localized to the central region (positive examples).
        num_neg (int): number of negative class sequences. Number of sequences to generate where motif instances are not localized and may appear anywhere (negative examples).
        GC_fraction (float): GC fraction in background sequence. Fraction of G/C bases in the background sequence composition used when sampling non-motif positions; passed to motif_density to control background nucleotide frequencies.
    
    Behavior and side effects:
        The function calls an internal helper (motif_density) twice: once to generate num_pos sequences with motif instances constrained inside the center window defined by center_size, and once to generate num_neg sequences with motif instances placed without localization constraints. The generated positive sequence array and negative sequence array are concatenated into a single sequence array with the positive sequences first and the negative sequences second. The label array y is constructed with boolean labels aligned to sequence_arr: True for the first num_pos entries (positive class) and False for the next num_neg entries (negative class). embedding_arr is the concatenation of embedding objects returned by the helper for positive then negative sequences. There are no external side effects (no file I/O). Default behavior has no randomness seed argument; motif placement and motif counts are therefore non-deterministic across calls unless controlled externally by the environment or by modifying the underlying helper.
    
    Failure modes and notes:
        The function relies on motif_density for core validation and will propagate exceptions raised by that routine. Invalid inputs such as nonsensical combinations of parameters (for example, center_size greater than seq_length, or min_motif_counts greater than max_motif_counts) will typically cause errors in motif_density or during sequence construction. The function does not itself validate parameter ranges beyond delegating to the helper; callers should ensure parameters are consistent for their use case.
    
    Returns:
        sequence_arr (1darray): Contains sequence strings. A one-dimensional array-like collection of nucleotide sequence strings; the first num_pos entries are the positive (center-localized) sequences and the next num_neg entries are the negatives.
        y (1darray): Contains labels. A one-dimensional array-like collection of boolean labels aligned to sequence_arr where True indicates a positive (localized) sequence and False indicates a negative sequence. The ordering matches sequence_arr (positives first, negatives second).
        embedding_arr (1darray): Array of embedding objects. A one-dimensional array-like collection of embedding objects returned from the motif generation helper, concatenated in the same order as sequence_arr; these objects capture the placement/identity of implanted motif instances for downstream analysis.
    """
    from deepchem.molnet.dnasim import simulate_motif_density_localization
    return simulate_motif_density_localization(
        motif_name,
        seq_length,
        center_size,
        min_motif_counts,
        max_motif_counts,
        num_pos,
        num_neg,
        GC_fraction
    )


################################################################################
# Source: deepchem.molnet.dnasim.simulate_multi_motif_embedding
# File: deepchem/molnet/dnasim.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def deepchem_molnet_dnasim_simulate_multi_motif_embedding(
    motif_names: list,
    seq_length: int,
    min_num_motifs: int,
    max_num_motifs: int,
    num_seqs: int,
    GC_fraction: float
):
    """Generates synthetic DNA sequences containing variable numbers of embedded motifs for use in multi-motif recognition experiments (a common supervised learning task in regulatory genomics). This function constructs sequences of length seq_length with a background nucleotide composition controlled by GC_fraction, embeds between min_num_motifs and max_num_motifs motifs sampled from the ENCODE motif collection via simdna, and returns the sequences, per-motif presence labels, and embedding metadata for downstream model training or evaluation.
    
    Args:
        motif_names (list): List of strings naming motifs to embed. Each name must match an entry in the simdna LoadedEncodeMotifs database (the function loads motifs using synthetic.LoadedEncodeMotifs(simdna.ENCODE_MOTIFS_PATH, ...)). In practice these names identify transcription-factor binding motifs from the ENCODE motif collection and determine which motif PWMs will be sampled and embedded into generated sequences.
        seq_length (int): Total length of each generated DNA sequence (number of nucleotides). The function generates sequences of exactly this length using a zero-order background generator; downstream learners expect sequence strings of this fixed length.
        min_num_motifs (int): Minimum number of motifs to embed per sequence (inclusive). Used by the UniformIntegerGenerator to sample how many motifs to place in a given sequence; controls signal density for detection/recognition tasks.
        max_num_motifs (int): Maximum number of motifs to embed per sequence (inclusive). Must be >= min_num_motifs. Together with min_num_motifs this defines the allowed range for motif counts per sequence and therefore the difficulty and class balance of the simulated dataset.
        num_seqs (int): Number of sequences to generate. The function will produce exactly num_seqs sequences and corresponding labels and embedding metadata; returned arrays have length num_seqs along the sequence axis.
        GC_fraction (float): Fractional GC content used to construct the zero-order background nucleotide distribution (value passed to get_distribution(GC_fraction) in the local module scope). This controls the expected frequencies of G and C vs A and T in background sequence and therefore the signal-to-noise characteristics of motif detection in a genomics context.
    
    Behavior and side effects:
        The function imports and uses the simdna.synthetic utilities to load motif PWMs from simdna.ENCODE_MOTIFS_PATH (this typically reads motif files from disk). It constructs per-sequence backgrounds via synthetic.ZeroOrderBackgroundGenerator and embeds motifs using synthetic.SubstringEmbedder wrapped with ReverseComplementWrapper so that motifs may appear on either strand. The RandomSubsetOfEmbedders with a UniformIntegerGenerator(min_num_motifs, max_num_motifs) controls how many motifs are embedded per sequence. Sampling of motifs and their positions is random; results are nondeterministic unless simdna's RNG or other global seeds are set externally. The function does not modify global state except for reading motif resources from disk via simdna; it returns freshly allocated arrays and Python lists.
    
    Failure modes and validation:
        If simdna is not installed or simdna.ENCODE_MOTIFS_PATH is not accessible, the function will raise ImportError or an I/O exception when attempting to load motifs. If any entry in motif_names is not present in the loaded ENCODE motifs, simdna.synthetic.PwmSamplerFromLoadedMotifs or subsequent embedder construction will raise an error. If max_num_motifs < min_num_motifs, the UniformIntegerGenerator will be constructed with an invalid range and may raise an exception. GC_fraction should be a float value appropriate for the module-level get_distribution function; an invalid value may cause that helper to raise. No input validation is performed beyond what simdna and the underlying generators enforce.
    
    Returns:
        sequence_arr : 1darray
            Contains the generated DNA sequence strings. Expected length is num_seqs; each entry is a sequence string of length seq_length composed of uppercase nucleotide characters (A, C, G, T) sampled from the zero-order background with approximate GC content set by GC_fraction. These sequences are intended as model inputs for sequence-based motif recognition tasks.
        y : ndarray
            Contains boolean labels indicating motif presence. This is a 2D boolean array with shape (num_seqs, len(motif_names)) where each row corresponds to a generated sequence and each column corresponds to one motif from motif_names; True indicates that the motif appears at least once in the corresponding sequence. These labels serve as supervised targets for per-motif classification tasks.
        embedding_arr : 1darray
            Array (list-like) of embedding objects extracted from simdna generated sequences. There is one entry per generated sequence (length num_seqs). Each embedding object provides metadata about the motifs embedded into that sequence (for example, embedded motif name, start/end positions, and orientation/strand as provided by simdna). This embedding metadata is useful for evaluation, visualization, or for constructing more detailed training targets (e.g., localization).
    """
    from deepchem.molnet.dnasim import simulate_multi_motif_embedding
    return simulate_multi_motif_embedding(
        motif_names,
        seq_length,
        min_num_motifs,
        max_num_motifs,
        num_seqs,
        GC_fraction
    )


################################################################################
# Source: deepchem.molnet.dnasim.simulate_single_motif_detection
# File: deepchem/molnet/dnasim.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def deepchem_molnet_dnasim_simulate_single_motif_detection(
    motif_name: str,
    seq_length: int,
    num_pos: int,
    num_neg: int,
    GC_fraction: float
):
    """Simulates two classes of nucleotide sequences for single-motif binary classification experiments used in DNA motif detection benchmarking and synthetic dataset generation. This function produces a positive class of sequences that each contain a single occurrence of the named motif embedded at an arbitrary position in a background sequence, and a negative class of background sequences that do not contain the motif. The function is used to create labeled sequence arrays and corresponding embedding metadata that can be passed to downstream model training, evaluation, or visualization pipelines in DeepChem's motif-simulation utilities.
    
    Args:
        motif_name (str): Name of the motif to embed into positive-class sequences. This string identifies the motif to be placed into each positive sequence; the motif is encoded and embedded anywhere within a background sequence to simulate a binding site or functional element. The function uses this motif_name to generate the positive-class sequences. Supplying an invalid motif identifier or a motif that cannot be encoded by the underlying motif-encoding routine will result in an error from the embedding helper.
        seq_length (int): Length of each simulated nucleotide sequence (number of bases). This integer controls the total length of every generated sequence (both positive and negative). The sequence length must be sufficient to contain the motif; if seq_length is smaller than the motif length, the motif embedding routine will fail or raise an error.
        num_pos (int): Number of positive-class sequences to generate. This integer specifies how many sequences will contain the embedded motif. The function will produce exactly num_pos positive sequences and place them first in the returned arrays. Passing non-integer or negative values may raise an exception.
        num_neg (int): Number of negative-class sequences to generate. This integer specifies how many background sequences without the motif will be produced. The function will produce exactly num_neg negative sequences and place them after the positive sequences in the returned arrays. Passing non-integer or negative values may raise an exception.
        GC_fraction (float): GC fraction used to sample background nucleotide composition. This float controls the relative frequency of G/C versus A/T bases when generating the background sequence context for both positive and negative examples. The value is forwarded to the underlying background-sequence generator; providing an inappropriate value for the embedding routine may produce an error from that routine.
    
    Returns:
        sequence_arr (1darray): 1darray of sequence strings. Each element is a nucleotide sequence (string) of length seq_length. The first num_pos elements are positive-class sequences containing the motif; the following num_neg elements are negative-class background sequences without the motif. This array is suitable for input to sequence featurizers or for direct inspection.
        y (1darray): 1darray of class labels corresponding to sequence_arr. Each element is a boolean label stored in a one-column array format (the function constructs labels as [[True]] for positives and [[False]] for negatives). The ordering matches sequence_arr (positives first, then negatives). These labels are intended for binary classification tasks where True indicates the presence of the motif.
        embedding_arr (1darray): 1darray of embedding objects returned by the embedding helper. Each element corresponds to the embedding metadata for the matching sequence in sequence_arr and encodes information about how and where the motif was embedded (for positive examples) or that no motif was embedded (for negative examples). The order of embedding_arr matches sequence_arr and y.
    
    Behavior and side effects:
        The function internally calls the motif embedding helper to produce positive sequences with the specified motif and a separate call (with no motif) to produce negative background sequences, then concatenates the results into the returned arrays. There is no external I/O performed by this function (no files are read or written). The returned arrays preserve ordering so that downstream code can rely on the correspondence sequence_arr[i], y[i], embedding_arr[i]. Errors from the underlying embedding helper (for example, invalid motif_name, insufficient seq_length for the motif, or invalid GC_fraction) will propagate out of this function. The function does not perform random-seed management; reproducibility depends on the randomness behavior of the underlying embedding routines.
    """
    from deepchem.molnet.dnasim import simulate_single_motif_detection
    return simulate_single_motif_detection(
        motif_name,
        seq_length,
        num_pos,
        num_neg,
        GC_fraction
    )


################################################################################
# Source: deepchem.molnet.load_function.factors_datasets.gen_factors
# File: deepchem/molnet/load_function/factors_datasets.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def deepchem_molnet_load_function_factors_datasets_gen_factors(
    FACTORS_tasks: list,
    data_dir: str,
    train_dir: str,
    valid_dir: str,
    test_dir: str,
    shard_size: int = 2000
):
    """deepchem.molnet.load_function.factors_datasets.gen_factors loads and prepares the FACTORS molecular dataset for downstream modeling in DeepChem: it ensures raw CSV files are present (downloading them if necessary), featurizes molecules using the Merck descriptor set via a UserDefinedFeaturizer, removes missing entries, shuffles the training set, applies dataset transformers derived from the training data, and moves the resulting datasets into the specified target directories. This function is intended for molecular machine-learning workflows in drug discovery and cheminformatics where the FACTORS dataset's per-molecule tasks (labels) are used to train, validate, and test predictive models. It does not perform any train/test splitting because FACTORS is provided with predefined train/validation/test CSV files.
    
    Args:
        FACTORS_tasks (list): A list of task names (strings) corresponding to label columns in the FACTORS CSV files. These names are passed verbatim to deepchem.data.UserCSVLoader as the tasks argument and define which targets will be loaded and featurized for modeling.
        data_dir (str): Path to a local directory where the raw FACTORS CSV files are expected. If the expected TRAIN/VALID/TEST filenames are not present under data_dir, this function will attempt to download them from the repository URLs (TRAIN_URL, VALID_URL, TEST_URL) into this directory. This directory therefore must be writable and have sufficient space for the downloaded CSV files.
        train_dir (str): Destination directory path where the prepared (featurized, transformed, and optionally sharded) training dataset will be moved. The function calls train_dataset.move(train_dir), so this path must be accessible and writable; after return, the training dataset artifacts will reside under this directory.
        valid_dir (str): Destination directory path where the prepared validation dataset will be moved via valid_dataset.move(valid_dir). The validation dataset is featurized and transformed using the same transformers derived from the training data.
        test_dir (str): Destination directory path where the prepared test dataset will be moved via test_dataset.move(test_dir). The test dataset is featurized and transformed using the same transformers derived from the training data.
        shard_size (int): Optional. Shard size passed to loader.featurize for creating disk-backed sharded datasets; default is 2000. This controls how many examples are written per shard when creating the dataset on disk. Use a positive integer; very small values increase the number of files while very large values increase memory pressure during featurization.
    
    Behavior and side effects:
        This function executes the following sequence of operations with observable side effects:
        1) Constructs expected file paths for train/validation/test CSVs using data_dir and internal filename constants (TRAIN_FILENAME, VALID_FILENAME, TEST_FILENAME).
        2) If any expected CSV is missing, downloads that CSV into data_dir from predefined URLs (TRAIN_URL, VALID_URL, TEST_URL) using deepchem.utils.data_utils.download_url. Network failures, permission issues, or missing URLs will raise exceptions from the download utility.
        3) Creates a UserDefinedFeaturizer initialized with the repository's merck_descriptors and a UserCSVLoader configured with id_field="Molecule" and the provided FACTORS_tasks. The featurizer and id_field are specific to the FACTORS dataset and are used to compute numeric descriptors for each molecule row in the CSV files.
        4) Calls loader.featurize on each CSV (train, validation, test) with the provided shard_size to produce dataset objects. These operations may be disk-backed and can be time- and I/O-intensive depending on shard_size and dataset size.
        5) Removes entries with missing labels or features from each dataset via remove_missing_entries.
        6) Calls sparse_shuffle() on the training dataset to randomize training example order; this mutates the train_dataset in-place.
        7) Obtains a list of transformers from get_transformers(train_dataset) (these transformers are derived from training data statistics) and applies each transformer sequentially to train, validation, and test datasets by calling transformer.transform(...). Transformer application may modify datasets in-place or return new dataset objects depending on transformer implementations.
        8) Moves the resulting train, validation, and test datasets into train_dir, valid_dir, and test_dir respectively using each dataset's move method. This writes dataset shards and metadata to those target directories.
        9) Logs timing and progress information to the module logger.
    
    Failure modes and error propagation:
        - Network or URL issues during download will raise exceptions from deepchem.utils.data_utils.download_url.
        - Filesystem permission errors, insufficient disk space, or invalid directory paths for data_dir, train_dir, valid_dir, or test_dir will raise OS-level errors when writing or moving files.
        - Featurization or transformer operations may raise exceptions if merck_descriptors, the featurizer, loader, or transformer code encounters unexpected input (for example, malformed CSV rows or unsupported molecule representations).
        - Invalid shard_size values (non-positive integers) may produce undefined behavior in the loader.featurize call; the function assumes shard_size is a positive int.
        The function does not catch these exceptions internally, so callers should handle or allow them to propagate.
    
    Returns:
        tuple: A 3-tuple (train_dataset, valid_dataset, test_dataset) where each element is a DeepChem Dataset object representing the prepared dataset for its split. These dataset objects have been featurized using the Merck descriptors, had missing entries removed, had the training split shuffled, had transformers (derived from training data) applied, and been moved to their respective directories (train_dir, valid_dir, test_dir). The returned datasets may be disk-backed (sharded) objects depending on shard_size and loader behavior.
    """
    from deepchem.molnet.load_function.factors_datasets import gen_factors
    return gen_factors(
        FACTORS_tasks,
        data_dir,
        train_dir,
        valid_dir,
        test_dir,
        shard_size
    )


################################################################################
# Source: deepchem.molnet.load_function.kaggle_datasets.gen_kaggle
# File: deepchem/molnet/load_function/kaggle_datasets.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def deepchem_molnet_load_function_kaggle_datasets_gen_kaggle(
    KAGGLE_tasks: list,
    train_dir: str,
    valid_dir: str,
    test_dir: str,
    data_dir: str,
    shard_size: int = 2000
):
    """deepchem.molnet.load_function.kaggle_datasets.gen_kaggle loads and prepares the KAGGLE molecular benchmarking datasets used by DeepChem for molecular machine learning (drug discovery, materials science, and related cheminformatics tasks). This function locates or downloads three compressed CSV files containing labeled molecular data, featurizes molecules using a predefined Merck descriptor set, removes missing entries, applies dataset transformers consistent with DeepChem preprocessing, shuffles the training set, and moves the resulting Dataset objects to the specified output directories. The function does not perform any train/test splitting because the KAGGLE files are provided as separate train/validation/test splits.
    
    Args:
        KAGGLE_tasks (list): A list of column names in the CSV files that correspond to prediction targets (task labels). In practice, these are the label columns that the DeepChem UserCSVLoader will load as the supervised tasks for molecular property prediction in drug discovery or related benchmarks.
        train_dir (str): Destination directory path where the prepared training Dataset will be moved. After featurization and transformation, the produced train_dataset is moved to this location; callers should treat this as the on-disk location for the training data artifacts produced by DeepChem.
        valid_dir (str): Destination directory path where the prepared validation Dataset will be moved. After featurization and transformation, the produced valid_dataset is moved to this location; callers should treat this as the on-disk location for the validation data artifacts produced by DeepChem.
        test_dir (str): Destination directory path where the prepared test Dataset will be moved. After featurization and transformation, the produced test_dataset is moved to this location; callers should treat this as the on-disk location for the test data artifacts produced by DeepChem.
        data_dir (str): Directory path that contains or will contain the raw KAGGLE CSV files. The function expects three files with fixed names inside this directory: "KAGGLE_training_disguised_combined_full.csv.gz", "KAGGLE_test1_disguised_combined_full.csv.gz", and "KAGGLE_test2_disguised_combined_full.csv.gz". If these files are not present, the function will attempt to download them from DeepChem's S3 dataset host into data_dir.
        shard_size (int): Number of examples per shard passed to UserCSVLoader.featurize when creating Dataset shards; controls the granularity of on-disk dataset shards and affects memory and I/O behavior during featurization. Default is 2000. Larger shard_size can reduce the number of files written but increase memory usage during featurization; smaller shard_size reduces memory pressure but increases the number of shard files.
    
    Behavior, side effects, defaults, and failure modes:
        This function performs the following steps in order:
        1. Constructs expected full paths for the three KAGGLE CSV files inside data_dir using the fixed filenames mentioned above.
        2. If the training CSV file is not present at the constructed path, the function downloads all three required CSV.GZ files from DeepChem's public dataset S3 URLs into data_dir. Network failures, permission errors, or lack of disk space during download will raise exceptions (e.g., network-related errors or I/O errors) and abort processing.
        3. Creates a featurizer using deepchem.feat.UserDefinedFeaturizer configured with the merck_descriptors descriptor set (a predefined set of molecular descriptors used in DeepChem for the KAGGLE benchmarks). The function relies on merck_descriptors being available in module scope; if merck_descriptors is missing or the featurizer fails, an exception will be raised.
        4. Instantiates a deepchem.data.UserCSVLoader configured with tasks=KAGGLE_tasks and id_field="Molecule" so that each CSV row is identified by the "Molecule" column and the listed KAGGLE_tasks columns are treated as supervised targets. If the CSV files do not contain the expected columns, UserCSVLoader.featurize will raise an error.
        5. Featurizes the three CSV files (train, valid, test) by calling loader.featurize(file_path, shard_size=shard_size). This produces deepchem.data.Dataset objects stored in memory and/or on disk as shards. Featurization may raise exceptions if molecular parsing or descriptor computation fails for certain rows.
        6. Calls remove_missing_entries on each Dataset to drop examples with missing features or labels; this mutates the Dataset objects to exclude invalid examples.
        7. Calls sparse_shuffle() on the training Dataset to randomize example order; this affects downstream training reproducibility unless a random seed is set elsewhere.
        8. Obtains a list of transformer objects by calling get_transformers(train_dataset) and applies each transformer to train, valid, and test Datasets via transformer.transform(...). Transformers typically perform normalization, imputation, or other preprocessing required for model input. Transformer application may alter dataset metadata and the stored on-disk shards.
        9. Moves the final Dataset objects to the provided train_dir, valid_dir, and test_dir using Dataset.move(...). This operation persists the prepared datasets to the specified directories and is a visible side effect; errors during write (insufficient permissions, disk full) will raise I/O exceptions.
        10. Logs timing and progress information throughout the process for profiling and debugging.
    
        Important notes and failure modes:
        - The function does not perform any additional train/validation/test splitting; it uses the three provided KAGGLE files as the splits.
        - Network failures or permission issues during download will prevent dataset preparation and raise exceptions.
        - Featurization depends on the presence and correctness of merck_descriptors and the RDKit toolchain (or other molecule-parsing dependencies required by the featurizer); missing dependencies or malformed SMILES/structures can cause featurization to fail.
        - Dataset.move writes the prepared datasets to disk; callers should ensure the provided directories are writable and have sufficient space.
        - Transformer application assumes compatibility between transformers returned by get_transformers and the datasets produced; transformer failure will raise an exception and abort processing.
        - The shard_size parameter should be chosen with awareness of memory and disk trade-offs; the default 2000 is a balance between file count and memory use.
    
    Returns:
        tuple: A 3-tuple (train_dataset, valid_dataset, test_dataset), where each element is a deepchem.data.Dataset instance containing the featurized, cleaned, transformed, and persisted examples for the corresponding split. These Dataset objects are ready for use in DeepChem model training, validation, and testing pipelines for molecular machine learning tasks.
    """
    from deepchem.molnet.load_function.kaggle_datasets import gen_kaggle
    return gen_kaggle(
        KAGGLE_tasks,
        train_dir,
        valid_dir,
        test_dir,
        data_dir,
        shard_size
    )


################################################################################
# Source: deepchem.molnet.load_function.uv_datasets.gen_uv
# File: deepchem/molnet/load_function/uv_datasets.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def deepchem_molnet_load_function_uv_datasets_gen_uv(
    UV_tasks: list,
    data_dir: str,
    train_dir: str,
    valid_dir: str,
    test_dir: str,
    shard_size: int = 2000
):
    """gen_uv loads and preprocesses the UV dataset for molecular property modeling: it ensures raw CSV splits exist (downloads them if missing), featurizes molecules using a Merck descriptor UserDefinedFeaturizer, removes missing entries, shuffles and transforms datasets, and writes processed shards to the provided output directories. This function is used in DeepChem's MolNet data-loading pipeline for experiments in molecular machine learning (e.g., prediction tasks in drug discovery and chemistry) and does not perform any additional train/validation/test splitting beyond the splits present in the source CSV files.
    
    Args:
        UV_tasks (list): A list of task names (column headers) from the UV CSV files that specify target properties to load. In the MolNet/DeepChem domain these names indicate the labels for supervised learning (for example, spectral or molecular property targets). UV_tasks is passed to deepchem.data.UserCSVLoader to select and order output label arrays in the produced datasets.
        data_dir (str): Path to the directory that should contain or will receive the raw split CSV files. If the expected files (TRAIN_FILENAME, VALID_FILENAME, TEST_FILENAME) are not present under data_dir, the function attempts to download them from the configured TRAIN_URL/VALID_URL/TEST_URL and save them into this directory. This parameter controls where downloads are read from and where unprocessed raw data is expected.
        train_dir (str): Destination directory where the processed/sharded training dataset will be moved. After featurization, cleaning, shuffling, and transformation, the function calls train_dataset.move(train_dir) to write or relocate dataset shards and metadata suitable for later loading by DeepChem, so this directory becomes the canonical stored training dataset for downstream model fitting.
        valid_dir (str): Destination directory where the processed/sharded validation dataset will be moved. Analogous to train_dir, this is where the function writes the processed validation set after featurization and transformations so it can be reloaded for model selection or early stopping.
        test_dir (str): Destination directory where the processed/sharded test dataset will be moved. Analogous to train_dir and valid_dir, this is the canonical location for the processed test set used for final evaluation.
        shard_size (int): Size of the shard chunk used during featurization (passed to UserCSVLoader.featurize). Each dataset (train/valid/test) is featurized in blocks of at most shard_size rows to control memory usage; larger shard_size can reduce overhead but increases peak memory. Default is 2000. This parameter affects runtime memory and disk shard layout but does not change the splitting logic.
    
    Behavior and side effects:
        This function performs these concrete steps in order: (1) computes expected paths for training, validation, and test CSV files under data_dir; (2) if any file is missing, attempts to download the corresponding file(s) from the repository URLs and write them to data_dir (network access and write permissions are required); (3) constructs a UserDefinedFeaturizer using the Merck descriptor set (merck_descriptors) and a UserCSVLoader configured with tasks=UV_tasks and id_field="Molecule" so rows are identified by the "Molecule" column and features are the specified Merck descriptors; (4) featurizes each split with the given shard_size, producing deepchem.data.Dataset objects; (5) removes examples with missing labels or features using remove_missing_entries (this mutates the datasets and decreases example counts); (6) shuffles the training dataset in-place using sparse_shuffle to randomize example order; (7) obtains standard transformers via get_transformers(train_dataset) and applies each transformer in sequence to train, validation, and test datasets (transformers typically perform scaling/normalization appropriate for model training); (8) moves the final processed datasets to train_dir, valid_dir, and test_dir using Dataset.move, which persists shard files and dataset metadata for later loading; (9) logs timing and progress information via the module logger.
    
    Failure modes and error conditions:
        If network access is not available or the configured TRAIN_URL/VALID_URL/TEST_URL are unreachable, the download step will fail and raise an exception from deepchem.utils.data_utils.download_url. If data_dir or the destination move directories (train_dir, valid_dir, test_dir) are not writable, Dataset.move or download operations will raise filesystem errors. Featurization requires the merck_descriptors and any underlying chemistry toolkits used by the UserDefinedFeaturizer (for example RDKit); missing or misconfigured chemistry dependencies can cause featurization to raise an exception. The loader expects an "Molecule" id_field and the labeled columns named in UV_tasks to exist in the CSVs; malformed CSVs or missing columns will cause loading/featurization errors. remove_missing_entries and transformer.transform are applied in-place or return new Dataset objects depending on transformer implementation; callers should expect that returned datasets reflect removed examples and applied scaling.
    
    Returns:
        tuple: A 3-tuple containing (train_dataset, valid_dataset, test_dataset) where each element is a deepchem.data.Dataset representing the processed, featurized, cleaned, and transformed split. These returned Dataset objects are the same objects that have been persisted to train_dir, valid_dir, and test_dir via Dataset.move and are ready for use in DeepChem model training, validation, and testing workflows.
    """
    from deepchem.molnet.load_function.uv_datasets import gen_uv
    return gen_uv(UV_tasks, data_dir, train_dir, valid_dir, test_dir, shard_size)


################################################################################
# Source: deepchem.molnet.run_benchmark.load_dataset
# File: deepchem/molnet/run_benchmark.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def deepchem_molnet_run_benchmark_load_dataset(
    dataset: str,
    featurizer: str,
    split: str = "random"
):
    """deepchem.molnet.run_benchmark.load_dataset: Load a predefined molecular benchmark dataset, its train/validation/test splits, and the data transformers used for preprocessing so they can be used in DeepChem benchmarking and model evaluation workflows.
    
    This convenience function selects and invokes a dataset-specific loader from deepchem.molnet, prints basic progress to standard output, and returns the loader's outputs. It is used in DeepChem benchmarking scripts to obtain standardized datasets and preprocessing pipelines for tasks in drug discovery, materials science, quantum chemistry, and related molecular ML domains. The function may trigger network downloads or local caching when a requested dataset is not already available locally, and it delegates featurization and splitting behavior to the underlying loader function.
    
    Args:
        dataset (str): Identifier of the dataset to load. This string must match one of the supported MolNet loader names such as 'bace_c', 'bace_r', 'bbbp', 'chembl', 'clearance', 'clintox', 'delaney', 'factors', 'hiv', 'hopv', 'hppb', 'kaggle', 'kinase', 'lipo', 'muv', 'nci', 'pcba', 'pcba_128', 'pcba_146', 'pcba_2475', 'pdbbind', 'ppb', 'qm7', 'qm8', 'qm9', 'sampl', 'sider', 'thermosol', 'tox21', 'toxcast', or 'uv'. The chosen identifier determines which file(s) will be loaded, the set of predictive tasks (labels) returned, and what preprocessing the loader applies. Supplying an identifier not present in the internal mapping will raise a KeyError.
        featurizer (str or dc.feat.Featurizer): Featurization specification passed through to the dataset loader. This may be the name of a built-in featurizer (as a string) or an instantiated DeepChem Featurizer object (dc.feat.Featurizer). The featurizer controls how raw molecular representations (SMILES, SDF, etc.) are converted into input features for models; incorrect types or incompatible featurizer choices can cause the underlying loader to raise TypeError or other errors from the featurization code.
        split (str): Name of the splitting strategy to use for creating train/validation/test partitions, passed to the dataset loader. Typical values are splitter names recognized by MolNet (for example 'random', 'scaffold', etc.), and the default used by this function is 'random' when no explicit value is provided. If you pass None, the loader may use its internal default splitter behavior. The chosen splitter affects how molecules are partitioned for benchmarking and therefore has a direct impact on measured model generalization.
    
    Returns:
        tuple: A 3-tuple (tasks, all_dataset, transformers) returned by the selected MolNet loader.
        tasks: A sequence of strings naming the prediction targets (labels) in the dataset. In the context of molecular ML benchmarks these names define what the model is trained to predict (for example toxicity endpoints, activity against a target, or quantum properties).
        all_dataset: The dataset container(s) produced by the loader representing the data splits and features to be used for training and evaluation. Depending on the specific MolNet loader, this value is typically a tuple or other collection of DeepChem Dataset objects (e.g., train, valid, test) or a structure equivalent to what the loader documents. These Dataset objects encapsulate features, labels, and metadata required for model training and scoring.
        transformers: A list of transformer objects applied by the loader to preprocess labels and/or features (for example normalization or imputation). These transformers are returned so they can be applied consistently to predictions during evaluation and to allow reversing transformations as needed.
    
    Behavior and side effects:
        This function prints which dataset and splitting function are being used to standard output. It delegates all heavy lifting to the appropriate deepchem.molnet.load_* function selected by the dataset argument. Many MolNet loaders will download data from the internet and cache files on disk the first time they are invoked; ensure you have network access and write permission to the cache directory if required. The function itself does not modify model state but the loader it calls may create or update local cache files and may require optional dependencies (for example RDKit) that must be installed beforehand.
    
    Failure modes:
        If dataset is not a key in the internal loader mapping, a KeyError is raised. If featurizer is not a supported type for the chosen loader, the underlying featurization code may raise TypeError or other exceptions. Network failures, missing optional dependencies (such as RDKit), or corrupted cache files can cause the underlying loader to raise IO- or parsing-related exceptions. Users should consult the specific MolNet loader documentation for loader-specific constraints and remedies.
    """
    from deepchem.molnet.run_benchmark import load_dataset
    return load_dataset(dataset, featurizer, split)


################################################################################
# Source: deepchem.molnet.run_benchmark.run_benchmark
# File: deepchem/molnet/run_benchmark.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def deepchem_molnet_run_benchmark_run_benchmark(
    datasets: list,
    model: str,
    split: str = None,
    metric: str = None,
    direction: bool = True,
    featurizer: str = None,
    n_features: int = 0,
    out_path: str = ".",
    hyper_parameters: dict = None,
    hyper_param_search: bool = False,
    max_iter: int = 20,
    search_range: float = 2,
    test: bool = False,
    reload: bool = True,
    seed: int = 123
):
    """deepchem.molnet.run_benchmark.run_benchmark runs end-to-end benchmarking for one or more MolNet datasets using either DeepChem-provided model implementations or a user-supplied model. This function automates dataset loading (via deepchem.molnet loaders), featurization assignment, optional hyperparameter optimization using a Gaussian process optimizer, model training/evaluation on train/validation/(optional) test splits, and persistent logging of results to a CSV file. It is intended for practitioners in molecular machine learning and drug discovery who want reproducible comparisons of models and featurizers across standard DeepChem datasets described in the DeepChem README (e.g., bace_c, bbbp, hiv, qm9, tox21, etc.).
    
    Args:
        datasets (list): A list of dataset identifiers to benchmark in this run. Each element must be one of the supported MolNet keys recognized by the internal loading_functions mapping (examples: 'bace_c', 'bace_r', 'bbbp', 'chembl', 'clearance', 'clintox', 'delaney', 'hiv', 'hopv', 'kaggle', 'lipo', 'muv', 'nci', 'pcba', 'pdbbind', 'ppb', 'qm7', 'qm7b', 'qm8', 'qm9', 'sampl', 'sider', 'tox21', 'toxcast', 'uv', 'factors', 'kinase', 'thermosol'). The function iterates over this list; unsupported dataset names will raise a ValueError('Dataset not supported') or be skipped according to internal checks. Providing multiple datasets runs the full benchmark pipeline for each dataset in sequence, appending results for each to the output CSV.
        model (str): Specifies which model implementation to run or a user-defined model object. When a string, it must name one of DeepChem's implemented models (examples in original code: logistic regression, random forest, multitask network, bypass multitask network, irv, graph convolution). When a non-string object is provided, it is treated as a user-defined model and must implement the methods fit(train_dataset) and evaluate(dataset, metric, transformers); these will be called directly. If model is a string the function dispatches to internal benchmark_classification or benchmark_regression helpers depending on dataset mode.
        split (str): Optional splitter name to use when loading a dataset; default None uses the loader's default splitting behavior. Accepted split names are validated against an internal CheckSplit mapping per dataset; if an invalid split is supplied the dataset is skipped. When split is not None, the loader is invoked with split=split and reload=reload so that reproducible train/validation/test splits are created.
        metric (str): Optional metric identifier; default None causes the function to set dataset-appropriate default evaluation metrics (for classification datasets the default is area under ROC (deepchem.metrics.roc_auc_score aggregated by numpy.mean); for regression datasets the default is Pearson R^2 (deepchem.metrics.pearson_r2_score aggregated by numpy.mean)). In practice callers may pass metric names or metric objects compatible with deepchem.metrics.Metric; when None the function replaces it with a list containing the appropriate deepchem.metrics.Metric instance(s).
        direction (bool): Direction for hyperparameter optimization when hyper_param_search is True. True means optimization will attempt to maximize the metric(s) (e.g., maximize ROC AUC); False means minimize the metric(s). This flag is forwarded to the hyperparameter search routine and impacts the Gaussian process acquisition objective during hyperparameter tuning.
        featurizer (str): Optional featurizer specification. Accepts a featurizer name string (for DeepChem-built models) or a deepchem.feat.Featurizer object for custom featurization. If None and model is a DeepChem model string, the function attempts to assign a default featurizer and associated n_features from an internal CheckFeaturizer mapping keyed by (dataset, model). If no mapping exists for the (dataset, model) pair the dataset is skipped. Featurizer choice determines how raw molecular inputs are converted to model inputs and therefore directly affects feature dimensionality and model compatibility.
        n_features (int): Number of features produced by the featurizer. When using DeepChem featurizers this is normally set automatically from the CheckFeaturizer mapping; when using a user-defined featurizer the caller must set this value if the downstream model requires it. This value is forwarded to model training and hyperparameter search routines to inform dimensionality-dependent behavior.
        out_path (str): Filesystem path to which benchmark outputs are written. By default results are appended to a CSV file named 'results.csv' inside out_path. If hyper_param_search is True, the optimized hyperparameters are pickled and written to a file named <dataset><model>.pkl inside out_path. The directory must be writable by the running process; IO errors when opening these files will propagate as exceptions.
        hyper_parameters (dict): Optional dictionary of hyperparameter names and initial values to use for model training or for hyperparameter optimization. If None and hyper_param_search is True the function will attempt to use preset hyperparameter dictionaries (internal hps mapping keyed by model). If provided, these values are used directly for training (when hyper_param_search is False) or as starting points/bounds for the optimizer (when hyper_param_search is True).
        hyper_param_search (bool): Whether to perform automated hyperparameter optimization prior to final training/evaluation. When True the function constructs a deepchem.hyper.GaussianProcessHyperparamOpt optimizer and runs its hyperparam_search method with parameters max_iter and search_range. On success hyper_parameters is updated to the best-found set and optionally written to disk; when False the provided hyper_parameters (or presets) are used without search.
        max_iter (int): Maximum number of hyperparameter optimization iterations/trials to run when hyper_param_search is True. This integer is passed to the underlying GaussianProcessHyperparamOpt.hyperparam_search call and controls optimization budget; larger values increase search time and computational cost.
        search_range (float): Multiplicative factor that determines the search interval for continuous hyperparameters during optimization. The optimizer searches the range [initial_value / search_range, initial_value * search_range]. Typical values > 1 expand the search interval; this value is forwarded unchanged to the hyperparameter search routine.
        test (bool): Whether to evaluate and record metrics on the test split after training. When True the function calls model.evaluate on the test_dataset and includes test results in the output CSV. When False only train and validation metrics are computed and recorded. The test flag does not affect loading or splitting behavior beyond enabling the extra evaluation step.
        reload (bool): Whether to use cached featurized dataset files if present (True) or force re-featurization from raw data (False). This flag is forwarded to the deepchem.molnet loader functions; when True loaders will try to read and reuse precomputed features to save time, which can affect reproducibility if underlying code or dependencies change.
        seed (int): Random seed used for any seeded operations to improve reproducibility (e.g., model initialization, dataset splits where applicable). The seed value is passed to downstream training and benchmarking helpers so runs can be repeated.
    
    Returns:
        None: This function does not return a Python value. Instead it has the following side effects: it appends one row per reported metric to a CSV file named 'results.csv' under out_path (creating or opening the file in append mode); if hyper_param_search is True it persists the best-found hyper_parameters by pickling them to out_path/<dataset><model>.pkl; and it prints progress and dataset/model diagnostics to standard output. Failure modes include ValueError('Dataset not supported') for unrecognized dataset keys, IOErrors when writing files if out_path is not writable, and exceptions raised by user-defined models that do not implement the required fit/evaluate methods. Additionally, certain dataset/model combinations may be skipped internally (the function uses internal mappings CheckFeaturizer and CheckSplit and will continue to the next dataset if a required mapping is missing or a split is invalid).
    """
    from deepchem.molnet.run_benchmark import run_benchmark
    return run_benchmark(
        datasets,
        model,
        split,
        metric,
        direction,
        featurizer,
        n_features,
        out_path,
        hyper_parameters,
        hyper_param_search,
        max_iter,
        search_range,
        test,
        reload,
        seed
    )


################################################################################
# Source: deepchem.molnet.run_benchmark_models.benchmark_classification
# File: deepchem/molnet/run_benchmark_models.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def deepchem_molnet_run_benchmark_models_benchmark_classification(
    train_dataset: str,
    valid_dataset: str,
    test_dataset: str,
    tasks: list,
    transformers: str,
    n_features: int,
    metric: list,
    model: str,
    test: bool = False,
    hyper_parameters: dict = None,
    seed: int = 123
):
    """Calculate performance of a specified classification model on given DeepChem datasets and tasks, fit the model, and return evaluation results for training, validation, and optionally test sets. This function is used in the DeepChem Molnet benchmarking workflow (deepchem.molnet.run_benchmark_models) to compare different model architectures and configurations on tasks relevant to drug discovery, materials science, and computational biology; it constructs the requested model, applies dataset transformations required by some architectures (for example IRV or DAG preprocessing), fits the model, and evaluates performance using provided metrics.
    
    Args:
        train_dataset (dataset struct): Dataset used for model training and evaluation during fitting. In DeepChem this is typically a Dataset or DiskDataset containing input features (X), labels (y), and optionally weights/ids. Practical significance: the model is trained on this dataset and metrics reported for the training set are computed against it. Note that some model branches transform or reshard this object in-place (for example IRVTransformer replaces train_dataset with its transformed output; DAG branch calls train_dataset.reshard), so callers should make copies if they need to preserve the original dataset object.
        valid_dataset (dataset struct): Dataset used only for model evaluation and hyperparameter selection. In DeepChem benchmarking, valid_dataset provides an independent hold-out set for monitoring generalization during fitting and for computing validation metrics. This dataset may also be transformed in-place or replaced with a transformed copy by some model branches (IRVTransformer, DAGTransformer), and the function assumes it contains the same data layout conventions as train_dataset.
        test_dataset (dataset struct): Dataset used only for final model evaluation. When test is True the function evaluates the trained model on this dataset and returns test set metrics. Some model branches require access to test_dataset to compute dataset-wide parameters (for example DAG branch computes maximum number of atoms across train/valid/test); if test is True and test_dataset is invalid or missing, evaluation will fail.
        tasks (list of string): List of target names (task identifiers) that the multitask or singletask-to-multitask wrapper will predict. In Molnet benchmarks, these correspond to assay endpoints or properties (for example binary activity labels). The number and ordering of tasks determines model output dimensionality and mapping of metric values to tasks.
        transformers (dc.trans.Transformer struct): Transformer or list/tuple of transformers applied during evaluation. This argument is passed to deepchem.models.Model.evaluate and is used to reverse any data transformations (for example scaling or label transformations) before computing metrics. It should be compatible with the Dataset objects and with the model.evaluate method; mismatched transformers may cause evaluation to raise exceptions.
        n_features (integer): Number of input features per example (for fingerprint-based or dense-input models) or a structure describing feature sizes where required by the model (for example some graph/MPNN models expect n_features to describe atom and pair feature sizes). This value is used when constructing model input layers or feature-handling components and must match the representation of X in the datasets.
        metric (list of dc.metrics.Metric objects): List of DeepChem Metric objects used to evaluate predictions (for example AUC, accuracy, or others). The function passes this list to model.evaluate for training/validation/test assessments; the returned evaluation dictionaries contain results computed according to these Metric objects. If metrics are improperly specified the evaluate call will raise an error.
        model (string): String identifier selecting which model architecture to build and benchmark. Accepted values (enforced by an assertion) are 'rf', 'tf', 'tf_robust', 'logreg', 'irv', 'graphconv', 'dag', 'xgb', 'weave', 'kernelsvm', 'textcnn', and 'mpnn'. The chosen string determines which branch constructs a DeepChem model object (or a scikit-learn/xgboost wrapped model), what hyper-parameters are required, and which dataset preprocessing steps are applied.
        test (boolean, optional): Whether to calculate and return test_set performance. Default False. If True, the function will evaluate and include test set metrics in the returned test_scores dict. Note that some model branches perform additional preprocessing on the test_dataset only when test is True; if test is False then test_dataset will not be transformed or evaluated.
        hyper_parameters (dict, optional): Hyper-parameter dictionary for the designated model. Default None, which causes the function to use preset hyper-parameters (hps[model]) defined elsewhere in the benchmarking module. When supplied, the dictionary must contain all keys that the selected model branch expects (for example 'layer_sizes', 'batch_size', etc. for the 'tf' branch). Missing keys will raise KeyError during model construction. Values in this dict directly control model architecture and training behavior (learning rates, batch sizes, number of epochs, regularization), and may also override the seed for some model builders (for example xgboost branch reads seed from this dict).
        seed (int, optional): Random seed passed to model constructors to control stochastic behavior (weight initialization, random splits inside models, etc.). Default 123. Practical significance: setting a deterministic seed improves reproducibility of benchmark runs; however external libraries (XGBoost, scikit-learn) or low-level nondeterministic operations may still introduce variability.
    
    Behavior and side effects:
        - The function asserts that model is one of the supported string identifiers and will raise an AssertionError otherwise.
        - If hyper_parameters is None the function uses preset hyper-parameter values (hps[model]) defined in the benchmarking context; if those presets are missing or malformed a KeyError or NameError may be raised.
        - Certain model branches mutate or replace the provided dataset objects: IRVTransformer transforms train/valid/(test) datasets in-place by assigning transformed Dataset objects back to the local variables; DAG branch calls reshard on datasets (train_dataset.reshard) and replaces them with DAGTransformer outputs; textcnn merges datasets to build a character dictionary. Callers should be aware that the dataset objects passed in may be modified by reference or reassigned within the function.
        - Model construction may require external libraries: scikit-learn (logreg, rf, kernelsvm), xgboost (xgb branch), or deepchem-specific classes. If these libraries are not installed, importing or constructing those models will raise ImportError.
        - The function calls model.fit(...) and then model.evaluate(...) on train and validation datasets; if nb_epoch is provided in hyper_parameters the model is trained for that number of epochs, otherwise some scikit-learn or wrapper models are fit without epoch counts (nb_epoch is None for those). Training or evaluation may raise runtime errors if the datasets, hyper-parameters, or model configuration are inconsistent.
        - For the DAG branch the function computes maximum number of atoms across molecules in the provided datasets; if molecular objects in X do not implement get_num_atoms() as expected, an AttributeError will be raised.
        - For xgboost and other external-model branches, hyper_parameters keys such as 'early_stopping_rounds' are used to construct model wrappers; incorrect values may produce training-time errors.
    
    Returns:
        tuple: A 3-tuple (train_scores, valid_scores, test_scores) of dict objects containing evaluation results.
        train_scores (dict): Dictionary keyed by the model name string used (e.g., 'tf', 'rf') with values equal to the result returned by deepchem.models.Model.evaluate on the training dataset. In Molnet benchmarks these results typically include per-metric values (for example AUC for each task) computed from the Metric objects passed in. This return value represents the model's performance on the data it was trained on.
        valid_scores (dict): Dictionary keyed by the model name string with values equal to the result returned by model.evaluate on the validation dataset. This captures generalization behavior during hyperparameter selection and benchmark reporting.
        test_scores (dict): Dictionary keyed by the model name string with values equal to the result returned by model.evaluate on the test dataset, included only if test is True; if test is False this dict will be empty. This captures final model performance on held-out test data.
        Practical significance of returns: these dictionaries are intended to be consumed by benchmarking code that aggregates results across models and datasets, reporting metrics such as AUC per task and enabling comparison of architectures for tasks in drug discovery and related domains.
    """
    from deepchem.molnet.run_benchmark_models import benchmark_classification
    return benchmark_classification(
        train_dataset,
        valid_dataset,
        test_dataset,
        tasks,
        transformers,
        n_features,
        metric,
        model,
        test,
        hyper_parameters,
        seed
    )


################################################################################
# Source: deepchem.molnet.run_benchmark_models.benchmark_regression
# File: deepchem/molnet/run_benchmark_models.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def deepchem_molnet_run_benchmark_models_benchmark_regression(
    train_dataset: str,
    valid_dataset: str,
    test_dataset: str,
    tasks: list,
    transformers: str,
    n_features: int,
    metric: list,
    model: str,
    test: bool = False,
    hyper_parameters: dict = None,
    seed: int = 123
):
    """deepchem.molnet.run_benchmark_models.benchmark_regression
    Calculate regression benchmark performance for a specified DeepChem dataset and a selected model.
    
    This function is used by DeepChem benchmarking and example scripts to construct a model (one of several supported regression architectures), fit it to a provided training dataset, and evaluate performance on training, validation, and optional test datasets. It automates model construction from preset or user-provided hyper-parameters, applies dataset-specific transformers (for e.g., DAG, ANI, or Coulomb fit preprocessing), and returns per-model evaluation scores (commonly R^2 or other regression metrics) for use in model comparison, hyperparameter tuning, or reporting in molecular machine learning workflows such as drug discovery, quantum chemistry, and materials science.
    
    Args:
        train_dataset (str): Dataset used for model training and primary evaluation during fitting. In practice this should be a DeepChem Dataset-like object (contains attributes such as X and y). The function will pass this dataset to model.fit(...). For some model choices (e.g., dag_regression, krr_ft, ani) this dataset will be transformed in-place or replaced by a transformed copy (e.g., via DAGTransformer, CoulombFitTransformer, or ANITransformer) and may be reshared for queue-based training. The training dataset is required to have the features and labels appropriate for the selected model; otherwise calls such as model.fit(...) or internal preparations (e.g., computing max atoms) will fail with AttributeError or KeyError.
        valid_dataset (str): Dataset used only for validation/evaluation and hyperparameter selection. This should be a DeepChem Dataset-like object. The function evaluates the constructed model on this dataset using the provided metric(s) and transformers. For some models the function will transform or reshuffle this dataset in the same manner as the training set (for example DAGTransformer, ANITransformer, or CoulombFitTransformer), replacing the original valid_dataset variable with a transformed dataset object; callers should be aware that the original object may be mutated or superseded.
        test_dataset (str): Dataset used only for final test/evaluation. This should be a DeepChem Dataset-like object. If the boolean parameter test is True, the function evaluates the constructed model on this dataset after applying any dataset-specific transformations that were applied to train/valid. If test is False, the test_dataset is not evaluated. The test dataset may be transformed in-place or replaced when required by the model (for example DAGTransformer or ANITransformer).
        tasks (list): List of strings specifying the target names (tasks) for multi-task regression. This list determines the number of outputs the model must predict and is passed to model constructors (e.g., len(tasks) is used for MultitaskRegressor and other DeepChem models). The order and content of tasks should match the labels in the datasets passed above.
        transformers (str): Transformer(s) used for evaluation and postprocessing of model outputs. In DeepChem workflows this is typically a DeepChem Transformer instance (or list of Transformer instances) that converts raw model outputs back into the domain of the original labels (for example undoing normalization). Although the signature lists a str type, the practical role is to supply DeepChem transformer objects that will be forwarded to model.evaluate(...); supplying an incompatible type will cause evaluate(...) to raise a TypeError.
        n_features (int): Integer specifying the number of input features or fingerprint length for models that expect a flat feature vector. For some models used for quantum-mechanics-like datasets (DTNN, ANI, MPNN), n_features may be a sequence (the code expects a length-2 sequence for DTNN and ANI), and the function will assert or index accordingly; passing an incompatible shape or value will raise an AssertionError or IndexError. This parameter is used to configure model input layer sizes and some model-specific feature counts.
        metric (list): List of dc.metrics.Metric objects (DeepChem metric instances) that define how model performance is quantified during evaluation. These metrics are passed directly to model.evaluate(...) for train/valid/test scoring and typically include regression metrics such as R^2, mean absolute error, or RMSE. The returned score dictionaries map metric names to numeric values computed on each dataset.
        model (str): Name of the model architecture to build and benchmark. Must be one of: 'tf_regression', 'tf_regression_ft', 'rf_regression', 'graphconvreg', 'dtnn', 'dag_regression', 'xgb_regression', 'weave_regression', 'textcnn_regression', 'krr', 'ani', 'krr_ft', 'mpnn'. The function asserts membership in this list and will raise an AssertionError if an unsupported model name is supplied. Each model name selects a different DeepChem or scikit-learn model construction path with distinct expected dataset formats, required hyper-parameters, and potential side effects (e.g., dataset transformations).
        test (bool, optional): Whether to calculate and return test set performance. Default is False. When True, the function will evaluate the constructed model on test_dataset (after any required transformations) and populate the returned test_scores dictionary. When False, test_scores will be returned empty. Note that enabling test evaluation may transform or reshuffle the test_dataset in-place.
        hyper_parameters (dict, optional): Hyper-parameter dictionary for the specified model. If None (default), the function uses a preset hyper-parameter mapping accessed as hps[model]. For each supported model, the function expects specific keys in this dict (for example 'layer_sizes', 'batch_size', 'nb_epoch' for 'tf_regression', or 'n_estimators' for 'rf_regression'); missing keys will raise a KeyError. Values from this dict are used to configure model constructors, set training epochs (nb_epoch), learning rates, and other model-specific settings. For scikit-learn or xgboost based routes, this dict must contain parameters required to instantiate those external models; missing external libraries (e.g., xgboost) will raise ImportError at model construction.
        seed (int, optional): Random seed used to initialize model randomness and reproducible components (passed to DeepChem model constructors using seed or random_seed where available). Default is 123. Different models may use this value differently; for scikit-learn wrappers the seed may instead be provided from hyper_parameters if present.
    
    Behavior and side effects:
        - The function constructs a model based on model and hyper_parameters. If hyper_parameters is None, it loads defaults via hps[model]. It raises AssertionError if model is not in the supported list.
        - For some models, dataset preprocessing is required and applied: DAG models compute a maximum atom count across train/valid/test, create a deepchem.trans.DAGTransformer(max_atoms=...), call reshard(...) on datasets, and replace train/valid/(test) with transformed datasets. The function prints the detected maximum number of atoms and may clamp it to a model default_max_atoms value from hyper_parameters.
        - For ANI ('ani') and DTNN ('dtnn') models the function asserts len(n_features) == 2 and applies an ANITransformer or other specialized preprocessing; these models are intended for quantum-mechanics style datasets and will raise AssertionError if the feature description is incompatible.
        - For krr_ft the function applies a CoulombFitTransformer to train/valid/test datasets before constructing KernelRidge models.
        - For scikit-learn-based models (rf_regression, krr, xgb_regression) the function constructs singletask-to-multitask wrappers. These routes require external libraries (scikit-learn, xgboost) to be installed; otherwise ImportError will be raised during model construction.
        - The function calls model.fit(train_dataset) or model.fit(train_dataset, nb_epoch=nb_epoch) depending on whether nb_epoch is provided by the chosen model path. If nb_epoch is None (typical for non-neural models), model.fit(...) is invoked without an epoch count.
        - The function calls model.evaluate(...) on train and valid datasets and, if test is True, on test_dataset as well, passing the metric list and transformers argument. The returned evaluation dictionaries are stored under a key equal to the model name in the respective return dictionaries.
        - The function may mutate or replace the input datasets (train_dataset, valid_dataset, test_dataset) when applying transformers or resharding; callers should not assume the input dataset objects remain unchanged after calling this function.
        - The function prints progress messages such as the model name being fitted and diagnostic details (e.g., "Maximum number of atoms: N", "Start fitting: model_name"). Logging or stdout capture may observe these prints.
    
    Failure modes and errors:
        - AssertionError if model is not one of the supported model names or if a model-specific assertion fails (for example DTNN/ANI require len(n_features) == 2).
        - KeyError if required keys are missing from hyper_parameters when hyper_parameters is provided but incomplete.
        - AttributeError or TypeError if supplied dataset objects do not conform to expected DeepChem Dataset-like interface (missing X, get_num_atoms on molecule objects, or transform/reshard methods).
        - ImportError if external libraries required by a chosen model are not installed (e.g., xgboost for 'xgb_regression', scikit-learn for KernelRidge/RandomForestRegressor).
        - Any exceptions raised by model.fit(...) or model.evaluate(...) (e.g., numerical errors, shape mismatches) will propagate to the caller.
    
    Returns:
        train_scores (dict): Dictionary mapping the chosen model_name (string) to the evaluation results obtained by model.evaluate(train_dataset, metric, transformers). The evaluation result is the raw output from DeepChem's model.evaluate(...) and typically maps metric names (or metric objects) to numeric regression scores (for example R^2).
        valid_scores (dict): Dictionary mapping the chosen model_name (string) to the evaluation results on the validation dataset (model.evaluate(valid_dataset, metric, transformers)). Useful for model selection and hyperparameter assessment.
        test_scores (dict): Dictionary mapping the chosen model_name (string) to the evaluation results on the test dataset (model.evaluate(test_dataset, metric, transformers)) if the test parameter is True. If test is False, this dictionary will be empty. Note that this result is only produced when test=True and when the model path performs any required test-set transformations.
    
    Practical significance:
        This function centralizes the construction, training, preprocessing, and evaluation of a wide variety of regression models used in DeepChem benchmarks (graph convolutional models, message-passing neural networks, traditional ML models such as RandomForest and KernelRidge, and domain-specific options for QM datasets such as ANI and DTNN). It is intended for reproducible benchmarking workflows in molecular machine learning and expects DeepChem-compatible datasets and metric objects. Use this function to compare architectures, validate hyper-parameter choices, or generate training/validation/test metrics for reporting and downstream analysis.
    """
    from deepchem.molnet.run_benchmark_models import benchmark_regression
    return benchmark_regression(
        train_dataset,
        valid_dataset,
        test_dataset,
        tasks,
        transformers,
        n_features,
        metric,
        model,
        test,
        hyper_parameters,
        seed
    )


################################################################################
# Source: deepchem.splits.task_splitter.merge_fold_datasets
# File: deepchem/splits/task_splitter.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def deepchem_splits_task_splitter_merge_fold_datasets(fold_datasets: list):
    """Merges a sequence of fold datasets produced by k_fold_split into a single NumpyDataset that preserves features and identifiers and concatenates task/label and weight columns from each fold.
    
    This function is intended for use in DeepChem workflows (for example, molecular machine learning, drug discovery, or other chemistry/biology ML pipelines) after performing a k-fold split of a dataset. It assumes each element of fold_datasets was produced by deepchem.splits.k_fold_split and therefore represents the same set of datapoints in the same ordering, but with different label/weight columns for each fold. The returned dataset reuses the feature matrix (X) and identifiers (ids) from the first fold and constructs new label (y) and weight (w) arrays by concatenating the corresponding arrays from each fold along axis 1, so that columns in the returned y and w correspond to the concatenated fold outputs in the order given by fold_datasets.
    
    Args:
        fold_datasets (list): A list of dataset-like objects produced by k_fold_split. Each element must expose attributes X, ids, y, and w. X represents the input features (shared across folds), ids represents sample identifiers (shared across folds), y represents labels/targets, and w represents example weights. The function relies on the practical significance that these fold datasets represent the same samples in identical ordering; if this assumption is violated the merged dataset will be incorrect.
    
    Returns:
        NumpyDataset or None: If fold_datasets is a non-empty list, returns a new deepchem.data.NumpyDataset constructed with X taken from the first fold dataset, ids taken from the first fold dataset, and y and w formed by concatenating the y and w arrays from each fold in the order given (concatenation performed with numpy.concatenate along axis=1). This merged NumpyDataset is useful for downstream evaluation or bookkeeping across folds. If fold_datasets is empty, returns None. No input datasets are modified; a new dataset object is created. Possible failure modes include AttributeError if elements of fold_datasets do not provide the required attributes (X, ids, y, w), and numpy errors (e.g., ValueError) if the y or w arrays across folds cannot be concatenated due to incompatible shapes. The function does not perform validation beyond these assumptions and therefore will silently produce incorrect results if features or identifiers differ between folds.
    """
    from deepchem.splits.task_splitter import merge_fold_datasets
    return merge_fold_datasets(fold_datasets)


################################################################################
# Source: deepchem.trans.transformers.get_cdf_values
# File: deepchem/trans/transformers.py
# Category: valid
################################################################################

def deepchem_trans_transformers_get_cdf_values(array: numpy.ndarray, bins: int):
    """get_cdf_values computes per-column empirical cumulative distribution function (CDF) quantile values for a 1D or 2D numeric array. This helper is used in DeepChem transformer code to map raw feature values (rows = examples, columns = features) to discretized CDF-like values by splitting the sorted data into a fixed number of bins and assigning each sample a bin-based quantile. It is intended for preprocessing and normalization steps in molecular machine-learning workflows where consistent, column-wise percentile mapping is required.
    
    Args:
        array (numpy.ndarray): Input data to be mapped to CDF-like values. Must be either a 1-D array of shape (n_rows,) or a 2-D array of shape (n_rows, n_cols). Rows are interpreted as independent samples (examples) and columns as separate features/channels. If a 1-D array is provided, it is internally reshaped to shape (n_rows, 1) so the output always has two dimensions. Arrays with more than two dimensions are not supported. The function does not modify the input array in place; it returns a new numpy.ndarray.
        bins (int): Number of bins to split the data into when computing the quantile mapping. This integer controls the discretization resolution: larger values produce finer-grained quantile steps. The implementation requires bins to be an integer greater than 1 (bins <= 1 will cause division-by-zero or invalid partitioning). If bins is odd the code normalizes by (bins - 1) producing values that can reach 1.0; if bins is even the code normalizes by bins producing values with maximum (bins - 1) / bins (< 1.0). Non-integer or non-positive inputs for this parameter will raise an error (TypeError for non-int types, or runtime errors such as ZeroDivisionError for invalid values).
    
    Behavior and practical details:
        The function computes an auxiliary 1-D histogram index array of length n_rows named hist_values by partitioning the sorted positions of the n_rows samples into bins equal-width in index space. This is done by computing parts = n_rows / bins and assigning floor(row / parts) to each row index. For odd bins the floor result is divided by (bins - 1); for even bins it is divided by bins. These choices determine whether the maximum returned quantile reaches 1.0 (odd bins) or reaches (bins - 1) / bins (even bins). For each column, the function sorts the column values and replaces each original value with the corresponding entry from the precomputed histogram index array according to the column's sort order. The returned array therefore contains per-column quantile-like values with the same ordering structure as the input column values (smallest input values map to the smallest histogram quantile).
        The function performs an argsort per column; computational complexity is dominated by the per-column sort, approximately O(n_cols * n_rows log n_rows). Memory usage is O(n_rows * n_cols) for the returned array plus temporary arrays of size O(n_rows).
    
    Side effects and defaults:
        The function has no external side effects and does not modify its input. A 1-D input is reshaped to (n_rows, 1) internally so the shape of the returned numpy.ndarray is always (n_rows, n_cols) where n_cols equals the original second dimension or 1 for reshaped 1-D input.
    
    Failure modes and errors:
        Passing bins <= 1 will lead to division-by-zero or invalid partitioning; callers should ensure bins is an integer greater than 1. Supplying an array with zero rows (n_rows == 0) or an array with more than two dimensions is unsupported and may raise indexing or arithmetic errors. Supplying non-numeric data in array will likely cause argsort or numeric operations to fail. Type mismatches (e.g., non-numpy array for array or non-int for bins) will raise TypeError or other runtime exceptions from numpy operations.
    
    Returns:
        numpy.ndarray: A new 2-D array of shape (n_rows, n_cols) containing the mapped CDF-like values for each input element. The i-th row and j-th column of the return corresponds to the quantile-mapped value of the element at the same position in the input (after an internal reshaping if a 1-D array was supplied). The values are produced by the binning logic described above and are suitable for downstream transformer workflows in DeepChem that require column-wise percentile or quantile features.
    """
    from deepchem.trans.transformers import get_cdf_values
    return get_cdf_values(array, bins)


################################################################################
# Source: deepchem.trans.transformers.undo_grad_transforms
# File: deepchem/trans/transformers.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def deepchem_trans_transformers_undo_grad_transforms(
    grad: numpy.ndarray,
    tasks: list,
    transformers: list
):
    """undo_grad_transforms(grad, tasks, transformers)
    Performs the inverse of output-gradient modifications applied by a sequence of DeepChem transformers and returns the restored gradient array.
    This function was historically used to undo per-task output transforms when computing gradients (for example, when converting model output gradients back into physical force contributions in molecular modeling workflows). It walks the provided transformers in reverse application order (last-applied transform undone first) and calls each transformer's untransform_grad method when that transformer declared it modified outputs (transform_y is truthy). NOTE: This function is DEPRECATED and will be removed in a future DeepChem release; callers should manually implement the necessary untransformation logic for force/gradient calculations as recommended by the library.
    
    Args:
        grad (numpy.ndarray): Array of gradients with respect to model outputs. In DeepChem workflows this typically represents dL/dy or dE/dy produced during backpropagation and used to compute derived quantities such as atomic forces in quantum chemistry and molecular mechanics applications. The function treats this array as the current gradient state and passes it through each applicable transformer's untransform_grad method; the returned value is used as the input to the next undo step.
        tasks (list): List of task identifiers or task-specific metadata that parallel the model output dimensions. In DeepChem these are the task names (for example, prediction targets for multiple endpoints in multitask models) and are supplied to transformer.untransform_grad so transformers can apply task-specific inverse operations. The function forwards this list unchanged to each transformer's untransform_grad call.
        transformers (list): Sequence of transformer objects that were previously applied to model outputs during preprocessing or postprocessing. Each object is expected to have a boolean attribute transform_y that indicates whether the transformer modified model outputs, and a callable method untransform_grad(grad, tasks) that accepts the current gradient array and the tasks list and returns a numpy.ndarray with the transform undone. The function iterates over reversed(transformers) so that the last-applied transformer is undone first, matching typical invertible-transform semantics.
    
    Returns:
        numpy.ndarray: The gradient array after all applicable transformers have been undone. This is the same conceptual quantity as the input grad but mapped back through the inverse of each transformer's effect on model outputs. If no transformers have transform_y truthy, the original grad array is returned unchanged.
    
    Behavior, side effects, defaults, and failure modes:
        This function has no implicit defaults beyond using the parameters provided. It calls transformer.untransform_grad(grad, tasks) for each transformer where transformer.transform_y evaluates to True; the function assigns the return value of each call back to grad and uses it for subsequent undo operations. Side effects are limited to whatever side effects the individual transformer.untransform_grad implementations may perform (for example, they may modify internal transformer state or mutate inputs); this function itself does not mutate the transformers list or the tasks list. Failure modes include AttributeError or TypeError if an object in transformers does not implement the required transform_y attribute or untransform_grad method, or if untransform_grad returns a non-numpy.ndarray value incompatible with downstream code. Because this API is deprecated, rely on explicit, well-tested custom untransformation logic for production force calculations and similar sensitive workflows.
    """
    from deepchem.trans.transformers import undo_grad_transforms
    return undo_grad_transforms(grad, tasks, transformers)


################################################################################
# Source: deepchem.utils.batch_utils.batch_coulomb_matrix_features
# File: deepchem/utils/batch_utils.py
# Category: valid
################################################################################

def deepchem_utils_batch_utils_batch_coulomb_matrix_features(
    X_b: numpy.ndarray,
    distance_max: float = -1,
    distance_min: float = 18,
    n_distance: int = 100
):
    """Computes per-batch Coulomb-matrix-derived features used by models such as the DTNN (Deep Tensor Neural Network). This helper converts a batch of Coulomb matrices into atomic identifiers and a Gaussian-expanded pairwise distance representation that downstream models consume as input features. The function expects a 3-D NumPy array of Coulomb matrices (a batch), infers the number of atoms per molecule from nonzero entries, recovers approximate atomic numbers from Coulomb-matrix diagonals, computes interatomic distances by inverting the Coulomb interaction entries, and projects those distances onto a fixed set of Gaussian basis functions (distance bins). The outputs encode atom identity, pairwise distance features at specified granularity, per-atom molecule membership, and flattened pair-index mappings required by graph- or tensor-based molecular models.
    
    Args:
        X_b (np.ndarray): A 3-D NumPy array representing a batch of Coulomb matrices. The expected layout is (batch_size, max_atoms, max_atoms, ...) where the function indexes X_b[:, :, 0] to detect nonzero entries and uses X_b[i, :num_atoms, :num_atoms] as the Coulomb matrix for molecule i. Nonzero entries in X_b[:, :, 0] indicate atom presence and determine per-molecule atom counts. In the molecular machine-learning domain this array is produced by a CoulombMatrix featurizer (for example the CoulombMatrix featurizer used with QM9-like datasets) and its off-diagonal entries are assumed to follow the Coulomb interaction form Z_i * Z_j / r_ij (atomic charges product divided by interatomic distance) and diagonal entries correspond to the diagonal Coulomb-matrix prescription (used to recover approximate atomic number).
        distance_max (float): Upper bound of the distance range (in Angstrom) used to construct the Gaussian basis. This value is used with distance_min and n_distance to compute the step size and the center positions of the Gaussian distance bins. Practical significance: controls the maximum pairwise distance scale covered by the Gaussian expansion (larger values include longer-range interactions). Default in function signature is -1.0. If distance_max equals distance_min or n_distance <= 0, the computed step size will be zero or invalid and will cause division-by-zero errors in the Gaussian computation; callers must ensure distance_max, distance_min, and n_distance together define a valid nonzero step size.
        distance_min (float): Lower bound of the distance range (in Angstrom) used to construct the Gaussian basis. This value is used together with distance_max and n_distance to compute the Gaussian centers and the Gaussian width (step size). Practical significance: controls the minimum pairwise distance scale represented in the Gaussian expansion (smaller values allow resolution near-zero distances). Default in function signature is 18.0. If distance_min equals distance_max or n_distance <= 0, the computed step size will be zero or invalid and will cause division-by-zero errors in the Gaussian computation.
        n_distance (int): Number of Gaussian bins (granularity) used to expand each pairwise distance into a fixed-length vector. The function computes step_size = (distance_max - distance_min) / n_distance and then defines n_distance centers. Practical significance: each pairwise atom distance will be represented by a length-n_distance vector (a radial basis expansion); larger n_distance increases angular resolution and memory cost. Default in function signature is 100. Must be a positive integer; nonpositive values will produce runtime errors (division by zero).
    
    Returns:
        atom_number (np.ndarray): 1-D integer NumPy array of dtype int32 containing an approximate atomic number for every atom across the entire batch concatenated in batch order. For each molecule the function recovers atomic numbers by computing np.round((2 * diag) ** (1 / 2.4)) on the Coulomb-matrix diagonal (undoing the Coulomb-matrix diagonal prescription commonly used in CoulombMatrix featurizers). Practical significance: these integer atom identifiers (e.g., 1 for H, 6 for C, 8 for O) are used as per-atom features in models such as DTNN.
        gaussian_dist (np.ndarray): 2-D floating NumPy array of dtype float64 with shape (total_atom_pair_count, n_distance) where total_atom_pair_count equals the sum over molecules of (num_atoms_in_molecule ** 2). Each row is the Gaussian expansion of a single ordered atom-pair distance computed by inverting the Coulomb interaction (r_ij = Z_i * Z_j / CoulombEntry_ij). The function first flattens per-molecule distance matrices (with diagonal entries overwritten to -100 to exclude self-pairs), constructs a column vector of pairwise distances, then computes exp(- (distance - centers)^2 / (2 * step_size**2)) for each of the n_distance centers. Practical significance: this matrix is the radial-basis representation of pairwise distances that models use instead of raw scalar distances.
        atom_mem (np.ndarray): 1-D integer NumPy array of dtype int64 giving a molecule index for each atom in the concatenated atom list. Length equals the total number of atoms across the batch. For each atom this array indicates which molecule in the original batch the atom belongs to. Practical significance: used to assemble per-atom features back into per-molecule structures or to mask/aggregate contributions within a batch-aware model.
        dist_mem_i (np.ndarray): 1-D integer NumPy array of dtype int64 giving the flattened source-atom indices for each row of gaussian_dist, shifted into a single global indexing across the concatenated atom list. Length equals total_atom_pair_count. Practical significance: when gaussian_dist rows correspond to ordered atom pairs, dist_mem_i locates the row's first atom in the global atom indexing so model code can map pair features to graph edges or build sparse interaction tensors.
        dist_mem_j (np.ndarray): 1-D integer NumPy array of dtype int64 giving the flattened target-atom indices for each row of gaussian_dist, shifted into the same global indexing as dist_mem_i. Length equals total_atom_pair_count. Practical significance: together with dist_mem_i this array maps each Gaussian-expanded distance row to the two atom indices it connects.
    
    Notes on behavior and failure modes:
        - The function does not modify X_b in-place; it constructs and returns new NumPy arrays.
        - The routine infers per-molecule atom counts by counting nonzero entries in X_b[:, :, 0]; therefore X_b must have a meaningful element at index [:, :, 0] and nonzero entries must indicate atom presence. If X_b has an unexpected layout or zeroed indicator plane, atom counts will be incorrect.
        - Atomic numbers are recovered from the Coulomb-matrix diagonal via the inverse of the typical diagonal prescription used in CoulombMatrix featurizers (the code multiplies the diagonal by 2 and takes the 1/2.4 power). This is an approximation: nonstandard diagonal conventions or noisy diagonals will produce incorrect atomic numbers.
        - The interatomic distance calculation divides Z_i * Z_j by the Coulomb matrix off-diagonal entries. If any Coulomb off-diagonal entries are zero, this leads to division-by-zero (infs) and those values will propagate into gaussian_dist; the function does not explicitly sanitize or replace infinities or NaNs. Diagonal self-pairs are set to -100 to exclude them from meaningful Gaussian encoding.
        - If n_distance <= 0 or distance_max == distance_min the computed step_size will be zero or invalid; this will produce division-by-zero in the Gaussian width and raise a runtime warning/error or NaNs. Callers must ensure n_distance is a positive integer and that distance_max and distance_min define a nonzero interval.
        - The function returns a list of feature arrays in the order [atom_number, gaussian_dist, atom_mem, dist_mem_i, dist_mem_j]. These arrays are ready to be consumed by models that expect concatenated-batch atom lists and Gaussian-expanded pairwise distances (for example DTNN).
    """
    from deepchem.utils.batch_utils import batch_coulomb_matrix_features
    return batch_coulomb_matrix_features(X_b, distance_max, distance_min, n_distance)


################################################################################
# Source: deepchem.utils.coordinate_box_utils.get_face_boxes
# File: deepchem/utils/coordinate_box_utils.py
# Category: valid
################################################################################

def deepchem_utils_coordinate_box_utils_get_face_boxes(
    coords: numpy.ndarray,
    pad: float = 5.0
):
    """Get coordinate-space bounding boxes for each triangular face of the convex hull of a molecular coordinate set.
    
    For a macromolecule represented by a set of 3D atomic coordinates, the convex hull procedure identifies exterior triangular faces that describe the outer surface of the structure. For each triangular face, this function computes axis-aligned bounding intervals (x, y, z) that enclose the three vertices of the triangle, expands those intervals by an additive padding value, and returns a CoordinateBox for each face. These per-face boxes serve as crude geometric approximations of local exterior regions of the molecule (for example, to define candidate binding/interacting regions, to generate local grids for featurization, or to filter points by spatial locality). The algorithm uses simple geometry (floor/ceil of vertex coordinates followed by padding) and therefore is fast and interpretable but may be a coarse approximation of the true pocket geometry.
    
    Args:
        coords (numpy.ndarray): A numpy array of shape (N, 3) containing the 3D coordinates of a molecule (typically atomic coordinates in angstroms). This array is treated as an N×3 list of (x, y, z) points and is used as input to compute the convex hull. Practical role: provides the spatial points from which exterior triangular faces are extracted. Requirements and failure modes: coords must be two-dimensional with second dimension size 3 and must contain a sufficient number of non-coplanar points for a 3D convex hull (typically at least four non-coplanar points). If coords has an invalid shape or the point set is degenerate, the underlying convex hull routine (e.g., scipy.spatial.ConvexHull) will raise an error.
        pad (float): The number of angstroms to add outside the min/max extent of each triangular face when forming the coordinate box. Default is 5.0 (angstroms). Behavior detail: for each axis the code first computes the floor of the minimum vertex coordinate and the ceil of the maximum vertex coordinate (casting those to int), then subtracts pad from the floored minimum and adds pad to the ceiled maximum to produce final axis bounds. Because pad is a float and the floor/ceil are cast to int before applying pad, the resulting bounds may be floating-point values. Practical role: controls how much exterior margin is included around each face to capture surrounding interaction volume. Failure modes and notes: extremely large or negative pad values will respectively produce very large boxes or shrink/flip bounds; pad is applied uniformly to all three axes.
    
    Returns:
        List[deepchem.utils.coordinate_box_utils.CoordinateBox]: A list of CoordinateBox objects, one per triangular face of the convex hull. Each CoordinateBox is constructed from a triple of axis bounds (x_bounds, y_bounds, z_bounds), where each bounds value is a (min, max) tuple representing the padded interval along that axis (units consistent with coords, typically angstroms). Practical significance: the list length equals the number of exterior triangular faces (hull.simplices) and can be used downstream to select regions for docking, grid-based featurization, visualization, or other spatial analyses. Side effects: none on the input coords; the function allocates and returns a new list of CoordinateBox instances. Performance: runtime scales with the number of hull faces; computing the convex hull itself is the dominant cost.
    """
    from deepchem.utils.coordinate_box_utils import get_face_boxes
    return get_face_boxes(coords, pad)


################################################################################
# Source: deepchem.utils.coordinate_box_utils.intersect_interval
# File: deepchem/utils/coordinate_box_utils.py
# Category: valid
################################################################################

def deepchem_utils_coordinate_box_utils_intersect_interval(
    interval1: Tuple[float, float],
    interval2: Tuple[float, float]
):
    """deepchem.utils.coordinate_box_utils.intersect_interval computes the intersection of two one-dimensional closed intervals. This function is used in DeepChem codepaths that manipulate coordinate-aligned bounding boxes or 1D coordinate ranges (for example, computing overlap along a single axis when intersecting molecular or atom-centered boxes in drug discovery, materials science, quantum chemistry, and biology workflows). The function returns a tuple representing the overlapping interval or a sentinel representing an empty intersection.
    
    Args:
        interval1 (Tuple[float, float]): The first interval, specified as (x1_min, x1_max). These floats represent the lower and upper bounds along a single coordinate axis. The caller is expected to supply the bounds in increasing order (min then max). The function uses these values directly to determine overlap; it does not reorder or validate that x1_min <= x1_max.
        interval2 (Tuple[float, float]): The second interval, specified as (x2_min, x2_max). These floats represent the lower and upper bounds along the same coordinate axis as interval1. As with interval1, the caller must supply bounds in increasing order. The function treats the intervals as closed (endpoints included) when determining intersection.
    
    Returns:
        Tuple[float, float]: A tuple (x_min, x_max) describing the intersection interval. If the intervals overlap or touch, the returned tuple is (max(x1_min, x2_min), min(x1_max, x2_max)) and represents the closed interval of their overlap; when the intervals meet exactly at a point (for example x1_max == x2_min), the function returns a degenerate interval (v, v) representing that single-point intersection. If the intervals do not overlap, the function returns the sentinel (0, 0) to indicate the empty intersection. Note that (0, 0) can also be a legitimate single-point interval at coordinate zero; callers should be aware of this convention and, if necessary, apply additional checks (for example verifying the input intervals) to distinguish an intended zero-length interval at the origin from the empty-set sentinel.
    
    Behavior and failure modes:
        - The function performs no side effects and is purely functional.
        - It checks non-overlap with strict comparisons x1_max < x2_min and x2_max < x1_min; touching endpoints are treated as overlapping.
        - The function assumes inputs are well-formed as (min, max). If an interval is provided with min > max (i.e., reversed bounds), the result is undefined and may be incorrect because the implementation does not validate or reorder bounds.
        - No exceptions are raised for numeric inputs; invalid types (non-tuple or non-float elements) will raise normal Python type errors when the values are unpacked or compared.
        - The choice to represent the empty intersection as (0, 0) is an implementation decision retained from the original utility; callers in DeepChem code that require an unambiguous empty-set representation should convert or encode emptiness differently if needed.
    """
    from deepchem.utils.coordinate_box_utils import intersect_interval
    return intersect_interval(interval1, interval2)


################################################################################
# Source: deepchem.utils.data_utils.load_dataset_from_disk
# File: deepchem/utils/data_utils.py
# Category: valid
################################################################################

def deepchem_utils_data_utils_load_dataset_from_disk(save_dir: str):
    """deepchem.utils.data_utils.load_dataset_from_disk loads MoleculeNet train/valid/test DiskDataset objects and their saved preprocessing transformers from disk for downstream model training, validation, and testing in DeepChem workflows (e.g., drug discovery, materials, or biology tasks).
    
    This function expects data saved using save_dataset_to_disk and a specific directory layout under save_dir. It checks for the presence of three subdirectories named "train_dir", "valid_dir", and "test_dir" and a serialized transformers file (loaded via load_transformers). If any of the three dataset subdirectories is missing, the function returns immediately with a failure indicator and does not attempt to construct DiskDataset objects. On successful load, it instantiates dc.data.DiskDataset for each split using the corresponding subdirectory, sets a memory cache size on the train DiskDataset as a side effect (train.memory_cache_size = 40 * (1 << 20), i.e., 40 MB) to optimize small-batch access patterns, and loads transformers via load_transformers(save_dir). Errors raised by dc.data.DiskDataset constructors or load_transformers (for example due to corrupted files or permission errors) are not caught by this function and will propagate to the caller.
    
    Args:
        save_dir (str): Path to the top-level directory where the dataset and transformers were saved. This must be a filesystem path (string) that contains the following entries when the save was created by save_dataset_to_disk:
            save_dir/train_dir/  -- directory containing data files for the training split to be loaded by dc.data.DiskDataset
            save_dir/valid_dir/  -- directory containing data files for the validation split to be loaded by dc.data.DiskDataset
            save_dir/test_dir/   -- directory containing data files for the test split to be loaded by dc.data.DiskDataset
            save_dir/transformers.pkl (or other files load_transformers expects) -- serialized preprocessing transformers.
        The practical significance: callers should supply the exact directory produced by save_dataset_to_disk for a MoleculeNet-style dataset so that subsequent model training and evaluation pipelines can reuse identical datasets and preprocessing transforms.
    
    Returns:
        tuple: A 3-tuple containing:
            loaded (bool): Whether the load succeeded. If False, it indicates that one or more of the required subdirectories ("train_dir", "valid_dir", "test_dir") were missing under save_dir; in this case all_dataset will be None and transformers will be an empty list. If True, datasets and transformers were successfully instantiated and returned.
            all_dataset (Optional[Tuple["dc.data.DiskDataset", "dc.data.DiskDataset", "dc.data.DiskDataset"]]): When loaded is True, a tuple (train, valid, test) of dc.data.DiskDataset objects corresponding to the train, validation, and test splits, in that order. These DiskDataset objects are used throughout DeepChem pipelines to provide batched access to features, labels, and metadata. When loaded is False, this value is None.
            transformers (List["dc.trans.Transformer"]): A list of transformer objects loaded from the save directory (via load_transformers). These transformers represent preprocessing steps (normalization, featurization adjustments, etc.) that must be applied to new data to ensure consistency with the saved datasets. When loaded is False this list will be empty.
    
    Behavior and side effects:
        - Verifies existence of save_dir/train_dir, save_dir/valid_dir, and save_dir/test_dir using os.path.exists; if any are missing, returns (False, None, []) without attempting to read files.
        - On success, constructs dc.data.DiskDataset objects for each split by passing the corresponding directory path to dc.data.DiskDataset.
        - Sets train.memory_cache_size = 40 * (1 << 20) (40 MB) as an explicit side effect to configure in-memory caching policy for the training dataset; callers should be aware this mutates the returned train DiskDataset object.
        - Calls load_transformers(save_dir) to obtain preprocessing transformers; the loaded list is returned to enable exact replay of preprocessing during model inference or further data processing.
        - Does not swallow exceptions from disk access, DiskDataset construction, or transformer loading; such exceptions (e.g., I/O errors, corrupted files, or incompatible transformer formats) will propagate to the caller for handling.
    
    Failure modes:
        - Missing required subdirectories -> returns (False, None, []) as a deterministic failure signal.
        - Corrupted or incompatible saved dataset files or transformer serialization may raise exceptions from dc.data.DiskDataset or load_transformers; callers should catch and handle these exceptions as appropriate.
        - If save_dir is not a string path or points to a location without appropriate permissions, underlying I/O operations will raise exceptions propagated to the caller.
    """
    from deepchem.utils.data_utils import load_dataset_from_disk
    return load_dataset_from_disk(save_dir)


################################################################################
# Source: deepchem.utils.data_utils.load_from_disk
# File: deepchem/utils/data_utils.py
# Category: valid
################################################################################

def deepchem_utils_data_utils_load_from_disk(filename: str):
    """deepchem.utils.data_utils.load_from_disk loads a dataset or arbitrary serialized object from a file on disk and returns the in-memory Python object appropriate for use with DeepChem workflows (for example, datasets used in drug discovery, materials science, quantum chemistry, and biology model training and evaluation).
    
    This function inspects the file extension (with special handling for gzip-compressed files ending in ".gz") to select a loader that reconstructs Python objects from common serialization formats used in DeepChem examples and pipelines. It is intended to be used wherever code needs to restore a saved dataset or object from disk for downstream model training, evaluation, or analysis in the DeepChem ecosystem.
    
    Args:
        filename (str): Path to the file to load. This is the filesystem path or filename string pointing to the saved object. The function supports files whose final extension (after optional ".gz" compression suffix) is one of ".pkl", ".joblib", ".csv", or ".npy". The parameter is required and must refer to a readable file on the local filesystem accessible to the process. Example usage in DeepChem pipelines: passing the path to a saved featurized dataset or a serialized model checkpoint to restore it for further training or inference.
    
    Returns:
        Any: The Python object reconstructed from the file. The concrete returned type depends on the file extension:
            - For ".pkl": the object returned by load_pickle_file(filename). This typically restores Python objects serialized with pickle and is used in DeepChem to reload previously pickled datasets or utilities.
            - For ".joblib": the object returned by joblib.load(filename). Joblib is commonly used to persist scikit-learn-style objects or large numpy-backed objects in DeepChem workflows.
            - For ".csv": a pandas.DataFrame created by pd.read_csv(filename, header=0) with the first line of the CSV required to be a header; after loading, any NaN values are replaced with the empty string (df.replace(np.nan, str(""), regex=True)) so downstream DeepChem code that expects string placeholders does not encounter pandas NA values.
            - For ".npy": a numpy.ndarray or other object returned by np.load(filename, allow_pickle=True). The loader sets allow_pickle=True to permit arrays that contain Python objects to be restored (consistent with DeepChem usage where arrays may hold Python objects).
        The function returns the loaded object for immediate use by DeepChem model training, evaluation, or analysis code.
    
    Behavior, side effects, and failure modes:
        - The function treats a filename ending with ".gz" as gzip-compressed and strips the ".gz" suffix before determining the true extension (for example "data.csv.gz" is handled as a CSV).
        - For CSV files, the loader enforces header=0; the first line of the CSV must be column names. After reading, NaN values are replaced with empty strings to match DeepChem's common data handling expectations.
        - For ".pkl" and ".joblib" files and for ".npy" with allow_pickle=True, deserialization may execute code embedded in pickled objects; therefore, loading untrusted files can run arbitrary code. Do not load files from untrusted sources.
        - IO-related exceptions from the underlying libraries may be raised to the caller, for example FileNotFoundError or PermissionError if the file cannot be accessed, pd.errors.ParserError for malformed CSVs, joblib/Pickle errors for corrupted serialized objects, or numpy errors for invalid .npy files. These exceptions are not caught by this function and should be handled by the caller as appropriate.
        - If the file's (post-.gz) extension is not one of the recognized types, the function raises ValueError("Unrecognized filetype for %s" % filename).
        - Loading large datasets will allocate memory for the returned object; callers should ensure sufficient memory is available.
    
    Practical significance in the DeepChem domain:
        - This loader is a convenience used across DeepChem examples and workflows to restore datasets, featurized matrices, model artifacts, and other persisted objects. It standardizes handling of common serialization formats and enforces CSV header semantics and NaN-to-empty-string replacement so downstream DeepChem model code receives consistent input types and values.
    """
    from deepchem.utils.data_utils import load_from_disk
    return load_from_disk(filename)


################################################################################
# Source: deepchem.utils.data_utils.load_image_files
# File: deepchem/utils/data_utils.py
# Category: valid
################################################################################

def deepchem_utils_data_utils_load_image_files(input_files: List[str]):
    """deepchem.utils.data_utils.load_image_files loads a set of image files from disk into a numpy array suitable for downstream model input or preprocessing in DeepChem workflows.
    
    This function reads image files from the filesystem and returns them as a single numpy array with one entry per input file. It is intended for use in DeepChem pipelines that require loading small to moderate collections of PNG or TIF images (for example microscopy or assay images used in machine-learning models in drug discovery, materials science, or biology). The function determines how to read each file by its file extension (case-insensitive) and uses the Pillow (PIL) library to open image files. All files are read into memory; therefore the caller should ensure sufficient RAM is available for the total size of all images. The function makes no preprocessing beyond converting images to numpy arrays (no resizing, normalization, or channel reordering).
    
    Args:
        input_files (List[str]): List of image filenames (filesystem paths) to load. Each entry must be a string path to a file accessible from the running process. Paths are checked only by their extension to choose a reader; files with extensions other than ".png" or ".tif" (case-insensitive) will cause a ValueError. The order of images in the returned array matches the order of filenames in this list.
    
    Returns:
        np.ndarray: A numpy array that contains the loaded images. The nominal shape is (N, ...) where N is len(input_files) and "..." denotes each image's shape as produced by numpy.array(Image.open(...)). If all loaded images share an identical shape and dtype, the returned array will be a regular numeric numpy array with shape (N, height, width, ...) or similar. If images have differing shapes or types, the function falls back to returning a 1-D numpy array with dtype=object where each element is the per-file image array. The caller must handle either form; no automatic broadcasting or padding is performed.
    
    Behavior, side effects, defaults, and failure modes:
        The function requires the Pillow package to be installed. If Pillow is not importable, the function raises ImportError with a message indicating Pillow is required. For each filename in input_files, the function uses os.path.splitext to obtain the extension and compares it in lowercase; supported extensions are ".png" and ".tif". For ".png" and ".tif" files the function opens the file with PIL.Image.open and converts it to a numpy array. If an input filename has an unsupported extension, the function raises ValueError identifying the offending filename. File-system or image-decoding errors raised by Pillow (for example FileNotFoundError, OSError, or other IO-related exceptions) are propagated to the caller. All images are loaded into memory; large or numerous images can cause high memory usage or MemoryError. The function does not perform any validation beyond file reading and extension checking, and it does not modify files on disk.
    """
    from deepchem.utils.data_utils import load_image_files
    return load_image_files(input_files)


################################################################################
# Source: deepchem.utils.data_utils.load_pickle_file
# File: deepchem/utils/data_utils.py
# Category: valid
################################################################################

def deepchem_utils_data_utils_load_pickle_file(input_file: str):
    """deepchem.utils.data_utils.load_pickle_file loads a Python object from a single pickle file on disk, automatically handling files that are gzip-compressed when the filename contains the substring ".gz". This utility is used throughout DeepChem data pipelines and tutorials to restore saved artifacts such as featurized datasets, cached preprocessing outputs, NumPy arrays, model checkpoints, and other Python objects that were serialized with pickle.
    
    Args:
        input_file (str): The filename or path of the pickle file to read. This function opens the file in binary read mode. If the string ".gz" appears anywhere in input_file, the function treats the file as gzip-compressed and opens it with gzip.open(..., "rb"); otherwise it opens the file with built-in open(..., "rb"). In DeepChem workflows (for example when loading cached features, saved Dataset objects, or model parameter files produced in tutorials and examples), provide the exact path to the file you previously wrote with pickle (optionally compressed). The function does not validate that the extension accurately reflects the file contents; detection is purely substring-based, so filenames that include ".gz" but are not gzip-compressed will cause gzip to attempt decompression and likely raise an error.
    
    Returns:
        Any: The Python object reconstructed by pickle.load from the file. Typical objects in DeepChem usage include Dataset-like containers, numpy arrays, dictionaries of metadata, or objects saved by DeepChem model or preprocessing code. The returned object is owned by the caller; large objects will consume memory in the current process.
    
    Raises:
        FileNotFoundError: If the specified input_file does not exist.
        PermissionError: If the file cannot be opened due to filesystem permissions.
        pickle.UnpicklingError: If the file contents are not a valid pickle stream.
        EOFError: If the file ends unexpectedly while unpickling.
        OSError: If gzip.open or the underlying file I/O encounters an OS-level error (for example, attempting to decompress a non-gzip file when ".gz" is present).
        Exception: Other exceptions raised by pickle.load or the I/O layer may propagate to the caller.
    
    Notes:
        - Security: Unpickling executes arbitrary code embedded in the pickle. Do not load pickles from untrusted or unauthenticated sources.
        - Side effects: This function performs synchronous, blocking I/O and may allocate significant memory when reconstructing large objects.
        - Compatibility: Objects pickled with different Python versions, or containing classes not importable in the current environment, may fail to unpickle. Ensure the runtime environment has the definitions required to reconstruct the saved objects.
    """
    from deepchem.utils.data_utils import load_pickle_file
    return load_pickle_file(input_file)


################################################################################
# Source: deepchem.utils.data_utils.load_pickle_files
# File: deepchem/utils/data_utils.py
# Category: fix_docstring
# Reason: Schema parsing failed: ("Couldn't parse this type hint, likely due to a custom class or object: ", typing.Iterator[typing.Any])
################################################################################

def deepchem_utils_data_utils_load_pickle_files(input_files: List[str]):
    """Load objects from a sequence of pickle files and yield them one at a time.
    
    This function is used in DeepChem workflows to load serialized Python objects that represent datasets or other artifacts used in molecular machine learning, drug discovery, materials science, quantum chemistry, and biology. Given a list of file paths, it opens each file (supporting plain pickle files and gzipped pickle files like XXXX.pkl.gz), delegates the actual file deserialization to the module-level helper load_pickle_file, and yields the deserialized top-level Python object for each file in the same order as the input list. Because it yields results as an iterator, this function enables lazy loading of potentially large dataset objects so the caller can avoid loading all files into memory at once.
    
    Args:
        input_files (List[str]): A list of filesystem paths to pickle files to load. Each entry must be a string path to a file containing a single pickled Python object (for example, a DeepChem Dataset, a pandas DataFrame, a NumPy array, or a Python dict/list). Paths may point to gzipped pickles using a .gz extension (e.g., "dataset.pkl.gz"); such files are handled transparently by the underlying loader. The order of paths in this list determines the order in which objects are yielded.
    
    Returns:
        Iterator[Any]: A generator that yields the deserialized Python object for each file in input_files, in the same order as provided. Each yielded value is the top-level object that was serialized into the corresponding pickle file. The generator is lazy: no file is fully deserialized until the caller iterates to that element.
    
    Behavior, side effects, and failure modes:
        This function performs file I/O and will open each path in input_files when the generator advances. It does not accumulate all objects in memory; memory usage is proportional to the largest single object yielded rather than the total size of all files. The function preserves the exact Python types stored in the pickle files; downstream code should handle the concrete types returned (for example, converting a yielded object into a DeepChem Dataset if expected). Because pickle deserialization can execute arbitrary code embedded in the pickle stream, do not use this function on files from untrusted or unauthenticated sources; doing so is a significant security risk. If a file does not exist, is not readable, is corrupted, or cannot be unpickled, the underlying load_pickle_file call will raise an exception (for example, FileNotFoundError, OSError, EOFError, or pickle.UnpicklingError), which will propagate to the caller when that file is accessed during iteration.
    """
    from deepchem.utils.data_utils import load_pickle_files
    return load_pickle_files(input_files)


################################################################################
# Source: deepchem.utils.data_utils.load_transformers
# File: deepchem/utils/data_utils.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for load_transformers because the docstring has no description for the argument 'save_dir'
################################################################################

def deepchem_utils_data_utils_load_transformers(save_dir: str):
    """Load the transformers for a MoleculeNet dataset from disk.
    
    This function reads a pickle file named "transformers.pkl" from the provided directory and returns the list of Transformer objects that were previously saved for a DeepChem MoleculeNet dataset. In DeepChem workflows (used for molecular machine learning in domains such as drug discovery, materials science, quantum chemistry, and biology) transformers implement preprocessing and postprocessing steps (for example feature scaling, normalization, label transforms, or other dataset-specific conversions) that must be reapplied consistently at training and inference time. Calling this function lets users restore the exact preprocessing pipeline that was used when a dataset or model was saved so subsequent model evaluation, retraining, or deployment uses identical transforms.
    
    Args:
        save_dir (str): Path to the directory that contains the saved transformers file. The function expects a file literally named "transformers.pkl" inside this directory; this file must have been created by DeepChem dataset/model saving utilities (for example via DeepChem Dataset or Transformer save helpers). The argument should be a filesystem path string; no interpretation or expansion (such as shell globs) is performed by this function.
    
    Returns:
        List["dc.trans.Transformer"]: A Python list of dc.trans.Transformer objects loaded from the "transformers.pkl" file. Each element is a Transformer instance that implements a specific data transformation used for the MoleculeNet dataset; restoring this list is necessary to reproduce preprocessing and postprocessing behavior across experiments, evaluations, and deployments.
    
    Behavior and side effects:
        The function opens the file os.path.join(save_dir, "transformers.pkl") in binary read mode and uses pickle.load to deserialize the list of transformers. The only filesystem side effect is reading that file; no files are written. The function returns the deserialized list.
    
    Failure modes and risks:
        If the file "transformers.pkl" does not exist in save_dir, a FileNotFoundError will be raised. If the file is not a valid pickle of the expected objects, pickle.UnpicklingError or other exceptions (for example AttributeError or EOFError) may be raised during deserialization. PermissionError may be raised if the file cannot be opened due to filesystem permissions. Because this function uses pickle to deserialize objects, unpickling data from untrusted sources is a security risk and can execute arbitrary code; only load transformers from trusted, integrity-checked sources. Compatibility issues can occur if transformers were saved with a different DeepChem version that changes Transformer class implementations.
    """
    from deepchem.utils.data_utils import load_transformers
    return load_transformers(save_dir)


################################################################################
# Source: deepchem.utils.debug_utils.set_max_print_size
# File: deepchem/utils/debug_utils.py
# Category: valid
################################################################################

def deepchem_utils_debug_utils_set_max_print_size(max_print_size: int):
    """deepchem.utils.debug_utils.set_max_print_size sets the maximum dataset size at which a dataset's ids will be included in its string representation.
    
    This function is used within DeepChem (a library for deep learning in the life sciences) to avoid very slow or extremely verbose string representations of Dataset objects when their self.ids arrays are large. Many DeepChem workflows print or log datasets during data loading, model debugging, or tutorial examples; when a Dataset contains a large number of entries, including the full ids list in repr()/str() can severely degrade performance and readability. Calling this function updates a module-level configuration value that caller code checks to decide whether to include ids in printed representations.
    
    Args:
        max_print_size (int): The maximum length (number of entries) of a Dataset for which ids will be included in the Dataset's string representation. This parameter is required and must be an integer according to the function signature. The function assigns this value to the module-global variable _max_print_size so other DeepChem components that render or log Dataset objects can consult it. The function itself does not validate the value beyond assignment; passing a non-integer or otherwise inappropriate integer may lead to incorrect or unexpected behavior in downstream formatting code that assumes an integer threshold.
    
    Returns:
        None: This function does not return a value. Its effect is purely a side effect: it sets the module-level global variable _max_print_size to the provided integer so that subsequent calls to dataset string/representation logic in DeepChem will use the updated threshold to decide whether to include self.ids in output. There is no built-in error reporting in this function; misuse (for example, supplying a non-integer) will not raise here but can produce unexpected formatting behavior elsewhere.
    """
    from deepchem.utils.debug_utils import set_max_print_size
    return set_max_print_size(max_print_size)


################################################################################
# Source: deepchem.utils.debug_utils.set_print_threshold
# File: deepchem/utils/debug_utils.py
# Category: valid
################################################################################

def deepchem_utils_debug_utils_set_print_threshold(threshold: int):
    """deepchem.utils.debug_utils.set_print_threshold sets the module-level print threshold that controls how many elements of dataset identifiers and task labels are shown when printing or obtaining string representations of DeepChem Dataset objects. This function is used in DeepChem (a library for deep learning in drug discovery, materials science, quantum chemistry, and biology) to limit or expand the amount of dataset metadata shown in logs, REPL sessions, and debugging output so that users can avoid excessively long prints for large datasets or show more detail when inspecting small datasets.
    
    Args:
        threshold (int): Number of elements to include when printing representations of Dataset objects' ids and tasks. This integer is written to the module-level variable _print_threshold and is used by Dataset __repr__/__str__ logic to decide how many entries to display. The parameter is required and must be provided as an int according to the function signature; passing a non-int value is not validated by this function and may produce unexpected or confusing output.
    
    Returns:
        None: This function does not return a value. Instead, it has the side effect of updating the global state in deepchem.utils.debug_utils by assigning the provided integer to the module-global variable _print_threshold. The change affects subsequent prints and string representations of Dataset objects within the same Python process and module import context. There is no persistence across process restarts.
    
    Behavior and side effects:
        - Sets the global variable _print_threshold to the given threshold value; other code that reads _print_threshold (notably Dataset printing routines) will observe the new value immediately.
        - Intended for debugging and interactive use to control the verbosity of Dataset metadata (ids/tasks) printed to console or logs.
        - The function does not perform input validation beyond the type annotation. Supplying values that do not make sense for printing (for example, extremely large integers or non-integer types) may lead to unexpected formatting or confusing output; callers should pass a sensible integer for the desired level of detail.
        - Because the function modifies global state, changing the threshold may have cross-cutting effects in multi-threaded applications or libraries that concurrently print Dataset objects; callers should be cautious about changing the threshold in code paths that run concurrently.
    """
    from deepchem.utils.debug_utils import set_print_threshold
    return set_print_threshold(threshold)


################################################################################
# Source: deepchem.utils.dft_utils.grid.radial_grid.get_xw_integration
# File: deepchem/utils/dft_utils/grid/radial_grid.py
# Category: valid
################################################################################

def deepchem_utils_dft_utils_grid_radial_grid_get_xw_integration(n: int, s0: str):
    """Return n quadrature points and corresponding integration weights on the interval [-1, 1] using a chosen one-dimensional quadrature rule.
    
    This function is used by DeepChem's DFT utilities (deepchem.utils.dft_utils.grid.radial_grid) to build one-dimensional quadrature rules that serve as the radial component of numerical integration grids in quantum chemistry and density functional theory workflows. The produced points and weights are suitable for numerically approximating integrals of functions defined on [-1, 1], for example the radial part of atomic orbitals or density components after a coordinate transformation. Three integrator choices are provided: two Chebyshev-based nonuniform grids (useful for concentrating points near boundaries) and a uniform/trapezoidal grid (simple, evenly spaced points).
    
    Args:
        n (int): Number of grid points to generate. This integer controls the resolution of the quadrature: larger n increases quadrature accuracy at increased computational cost. For the Chebyshev-based integrators ('chebyshev' and 'chebyshev2') values n >= 1 are supported by the implementation. For the 'uniform' integrator n must be at least 2 because the routine computes the spacing using x[1] and will raise an IndexError otherwise. The function does not validate types beyond relying on NumPy operations, so passing a non-integer may produce NumPy/TypeErrors.
        s0 (str): Name of the grid integrator to use; comparison is case-insensitive because the implementation lower-cases this string. Available options are 'chebyshev', 'chebyshev2', and 'uniform'. 'chebyshev' implements the specific Chebyshev-based node and weight formula used in the referenced DFT/radial-grid literature (non-uniform clustering with weights computed from trigonometric expressions). 'chebyshev2' returns Chebyshev-type nodes defined by cos(ipn1) and their associated weights. 'uniform' returns an evenly spaced grid on [-1, 1] with trapezoidal rule weights (endpoints half-weighted). Use the Chebyshev options when endpoint clustering or specific polynomial-based quadrature behavior is required; use 'uniform' for simple, equally spaced sampling.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple (x, w) where x is a NumPy array of length n containing the integration points on the closed interval [-1, 1], and w is a NumPy array of length n containing the corresponding quadrature weights. Both arrays are one-dimensional (shape (n,)) and contain floating-point values. The points x specify where function evaluations should be taken; the weights w are to be multiplied with those evaluations and summed to approximate the integral over [-1, 1]. There are no side effects; the function is pure and does not modify global state or its inputs.
    
    Raises:
        RuntimeError: If s0 does not match one of the available integrators ('chebyshev', 'chebyshev2', 'uniform'), a RuntimeError is raised with an explanatory message listing available options.
        IndexError: For the 'uniform' integrator, if n < 2 the implementation attempts to index x[1] to compute spacing and will raise an IndexError.
    """
    from deepchem.utils.dft_utils.grid.radial_grid import get_xw_integration
    return get_xw_integration(n, s0)


################################################################################
# Source: deepchem.utils.differentiation_utils.misc.assert_runtime
# File: deepchem/utils/differentiation_utils/misc.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def deepchem_utils_differentiation_utils_misc_assert_runtime(cond: bool, msg: str = ""):
    """deepchem.utils.differentiation_utils.misc.assert_runtime asserts a boolean condition at runtime and aborts execution by raising a RuntimeError with a provided message when the condition is not met. This utility is used throughout DeepChem, including in differentiation utilities, model construction, and data processing pipelines, to enforce invariants and signal unrecoverable runtime errors during workflows in drug discovery, materials science, quantum chemistry, and biology.
    
    Args:
        cond (bool): Condition to assert. This boolean represents the runtime invariant that must be true for subsequent code to be valid. In DeepChem code paths (for example, within differentiation utilities), callers pass a computed boolean expression here to verify assumptions about inputs, shapes, types, or runtime state. If cond is True, the function performs no side effects and execution continues; if cond is False, the function raises a RuntimeError with the given msg to halt execution and surface an explanatory message to the user or calling code.
        msg (str): Message to raise if condition is not met. This string defaults to the empty string ("") and is used as the message of the RuntimeError when cond is False. Provide a clear, actionable message describing the violated invariant or the expected state (for example, "Expected non-empty batch" or "Gradient shape mismatch") so that users or upstream error handlers can diagnose and respond to the failure. The function does not modify the message and uses it verbatim in the raised RuntimeError.
    
    Returns:
        None: This function does not return a value. Its effect is purely procedural: if cond is True, it returns None and has no side effects; if cond is False, it raises a RuntimeError with the provided msg to signal a runtime assertion failure and stop further execution.
    
    Raises:
        RuntimeError: Raised when cond is False. The RuntimeError contains msg as its message. This is the documented failure mode and is intended to be caught by higher-level code if recovery is possible; otherwise it will propagate and terminate the current operation.
    """
    from deepchem.utils.differentiation_utils.misc import assert_runtime
    return assert_runtime(cond, msg)


################################################################################
# Source: deepchem.utils.differentiation_utils.misc.get_and_pop_keys
# File: deepchem/utils/differentiation_utils/misc.py
# Category: valid
################################################################################

def deepchem_utils_differentiation_utils_misc_get_and_pop_keys(dct: Dict, keys: List):
    """Get and pop keys from a dictionary.
    
    This utility extracts the values for the specified keys from the provided dictionary and removes those entries from the original dictionary. It is primarily used in DeepChem's differentiation utilities and related preprocessing code to extract and remove selected configuration entries, intermediate tensors, or auxiliary values from a parameter or metadata dictionary before further processing (for example, removing gradient-related entries prior to a numerical routine). The function returns a new dictionary mapping each requested key to its popped value while mutating the input dictionary in-place.
    
    Args:
        dct (Dict): Dictionary to pop from. This is the mutable mapping that will be modified by this function: each key listed in keys will be removed from this dictionary and its value returned in the result. In DeepChem code paths this typically represents model inputs, parameter maps, or metadata dictionaries used during differentiation or preprocessing.
        keys (List): Keys to pop. This is an iterable list of keys that must exist in dct; for each key in this list the function performs dct.pop(key) and stores the popped value in the returned dictionary in the same iteration order as this list. Supplying duplicate keys in this list will cause subsequent attempts to pop an already-removed key to fail.
    
    Returns:
        Dict: A new dictionary containing the popped keys and their corresponding values. The returned dictionary contains exactly one entry for each key in the keys list (in the same order as keys were provided). The original dct no longer contains these keys after the call.
    
    Raises:
        KeyError: If any key in keys is not present in dct at the time it is popped. Because the function uses dct.pop(key) with no default, missing keys (or duplicates that cause a second pop of the same key) will raise KeyError.
    """
    from deepchem.utils.differentiation_utils.misc import get_and_pop_keys
    return get_and_pop_keys(dct, keys)


################################################################################
# Source: deepchem.utils.differentiation_utils.misc.set_default_option
# File: deepchem/utils/differentiation_utils/misc.py
# Category: valid
################################################################################

def deepchem_utils_differentiation_utils_misc_set_default_option(defopt: Dict, opt: Dict):
    """deepchem.utils.differentiation_utils.misc.set_default_option returns a merged options dictionary by taking a shallow copy of a dictionary of default options and updating it with user-provided options so that keys in opt override those in defopt. This function is used in DeepChem (a library for deep learning in drug discovery, materials science, quantum chemistry, and biology) to construct concrete configuration dictionaries for components such as models, training loops, data preprocessors, and differentiation utilities where sensible defaults must be combined with user overrides.
    
    This function makes a shallow copy of defopt to detach the returned mapping object from the original default mapping, then updates that copy in-place with opt. Because the copy is shallow, mutable values stored in defopt (for example lists, dicts, or objects representing hyperparameters or preprocessors) remain the same objects in the returned dictionary unless they are overridden by entries in opt. Keys present in opt will replace the corresponding keys from defopt; keys absent from opt will retain the values from defopt.
    
    Args:
        defopt (dict): Default options. A dictionary containing canonical default configuration values (for example, default hyperparameters like learning rate, batch size, optimizer settings, or preprocessing flags). The function will make a shallow copy of this dictionary to form the basis of the returned options mapping.
        opt (dict): Options. A dictionary of user-specified option values that should override or extend defaults in defopt. Keys in this dictionary take precedence and will replace the corresponding entries from defopt in the returned dictionary.
    
    Returns:
        dict: A new dictionary containing the merged options. The returned dictionary is created by making a shallow copy of defopt and then updating that copy with opt. This means the returned object itself is distinct from defopt (so assigning the return value will not rebind defopt), but any mutable objects referenced as values in defopt and not overridden by opt remain shared between defopt and the returned dictionary.
    
    Raises:
        TypeError: If defopt or opt are not of type dict, the function may raise a TypeError (for example, when dict.update is called) or behave unexpectedly. Callers should pass plain dict objects to guarantee correct and predictable behavior.
    
    Notes:
        - Side effects: The function does not mutate the input dict defopt; it returns a separate dict object. However, because the copy is shallow, nested mutable values from defopt remain shared and can be mutated through the returned dictionary.
        - Failure modes: Supplying non-dict types or dictionaries containing uncopyable values may lead to exceptions from copy.copy or dict.update.
        - Practical significance in DeepChem: Use this helper when assembling component configurations to ensure that library defaults are preserved unless explicitly overridden by the caller, for example when creating model configuration dictionaries in differentiation utilities or setting default preprocessing parameters in data pipelines.
    
    Example:
        res = set_default_option({'a': 1, 'b': [2]}, {'a': 3})
        # res == {'a': 3, 'b': [2]}
        # res is a new dict, but res['b'] is the same list object as in defopt.
    """
    from deepchem.utils.differentiation_utils.misc import set_default_option
    return set_default_option(defopt, opt)


################################################################################
# Source: deepchem.utils.docking_utils.load_docked_ligands
# File: deepchem/utils/docking_utils.py
# Category: valid
################################################################################

def deepchem_utils_docking_utils_load_docked_ligands(pdbqt_output: str):
    """Load ligands docked by AutoDock Vina from a PDBQT output file and return a list of RDKit molecule objects and the associated Vina scores.
    
    This utility is used in docking workflows (for example, in DeepChem-based drug-discovery pipelines) to read the multi-pose PDBQT files that AutoDock Vina writes to disk. AutoDock Vina writes one or more MODEL/ENDMDL blocks into the same PDBQT file, and includes a single "REMARK VINA RESULT:" line per model that contains the numerical score for that pose. This function parses that file, extracts the text block for each MODEL, converts each block to a PDB block using the module-local pdbqt_to_pdb helper, and then converts the PDB block to an RDKit molecule using rdkit.Chem.MolFromPDBBlock. The returned molecules preserve 3D coordinates and retain explicit hydrogens because the conversion is called with sanitize=False and removeHs=False. Callers typically use the returned molecules and scores for downstream tasks such as featurization, pose selection, rescoring, or building machine-learning datasets in ligand-protein docking studies.
    
    Args:
        pdbqt_output (str): Path to the PDBQT file produced by AutoDock Vina. This must be a filename (string) pointing to a file on disk that contains one or more Vina MODEL/ENDMDL blocks and REMARK VINA RESULT lines. The function will open and read this file (side effect: a file read occurs and a FileNotFoundError/OSError will be raised if the file cannot be opened). The function does not modify the file.
    
    Returns:
        Tuple[List[rdkit.Chem.rdchem.Mol], List[float]]: A tuple (molecules, scores). `molecules` is a list of rdkit.Chem.rdchem.Mol objects created from each MODEL block in the input PDBQT file; each molecule retains 3D coordinates and explicit hydrogens because MolFromPDBBlock is invoked with sanitize=False and removeHs=False. Note that RDKit may fail to parse some PDB blocks and return None for those entries, so callers should validate each element before use. `scores` is a list of floating-point values parsed from the corresponding "REMARK VINA RESULT:" lines in the PDBQT file; the i-th score is intended to correspond to the i-th model block as read from the file. There is one score appended per REMARK line encountered; if the input file is malformed (for example, missing REMARK lines or mismatched MODEL/ENDMDL pairing) the lengths and alignment of the two lists may not match or some molecules may be None.
    
    Behavior, side effects, and failure modes:
        This function requires RDKit to be importable. If RDKit is not installed, the function raises ImportError with a message indicating RDKit is required. The function opens and reads the entire file at the given path into memory; for very large PDBQT files this may use significant memory. If the file cannot be opened, Python will raise FileNotFoundError or another OSError. If the PDBQT file is malformed (for example, MODEL/ENDMDL blocks are not present or a REMARK VINA RESULT line is missing), parsing may produce unexpected results: scores may be missing, models may be None, or an AttributeError may occur if lines outside of a MODEL block are encountered. The produced RDKit molecules are returned unsanitized; downstream code should sanitize or validate molecules if required by later processing steps.
    """
    from deepchem.utils.docking_utils import load_docked_ligands
    return load_docked_ligands(pdbqt_output)


################################################################################
# Source: deepchem.utils.docking_utils.read_gnina_log
# File: deepchem/utils/docking_utils.py
# Category: valid
################################################################################

def deepchem_utils_docking_utils_read_gnina_log(log_file: str):
    """Read GNINA logfile and extract per-mode docking scores used in structure-based drug discovery workflows.
    
    This function is used in DeepChem pipelines that consume GNINA docking output to produce numerical labels or features for machine learning models in drug discovery. It opens and reads the GNINA-generated logfile, locates the section containing per-mode score entries, parses numeric values for each binding mode, and returns them as a numpy.ndarray suitable for downstream analysis (for example, training or evaluating models that predict binding affinity or pose quality).
    
    Args:
        log_file (str): Path to the GNINA logfile to read. This is the filename produced by GNINA when run with logging enabled; the function will open this file for reading. The file is expected to contain a delimiter line beginning with '-----+' followed by one or more lines describing individual docking modes. Each mode line is expected to have a leading token (mode identifier) followed by three numeric tokens: binding affinity in kcal/mol, CNN pose score, and CNN affinity. Supplying a non-existent path will raise FileNotFoundError; supplying a file that does not follow the expected GNINA logfile format may raise ValueError during numeric parsing.
    
    Returns:
        numpy.ndarray: A 2D numpy array with one row per docking mode and three columns corresponding to [binding affinity (kcal/mol), CNN pose score, CNN affinity], in the same order the modes appear in the logfile. This array is intended to be used directly as numerical labels or features in DeepChem workflows for docking/modeling tasks. If the logfile contains no mode entries after the expected delimiter, the function will return an empty numpy.ndarray (i.e., no rows). Possible failure modes that callers should handle include FileNotFoundError or PermissionError when opening the file, and ValueError when converting malformed tokens to float. The function has no other side effects (it only reads the given file).
    """
    from deepchem.utils.docking_utils import read_gnina_log
    return read_gnina_log(log_file)


################################################################################
# Source: deepchem.utils.equivariance_utils.so3_generators
# File: deepchem/utils/equivariance_utils.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for so3_generators because the docstring has no description for the argument 'k'
################################################################################

def deepchem_utils_equivariance_utils_so3_generators(k: int):
    """deepchem.utils.equivariance_utils.so3_generators constructs the three matrix generators of the SO(3) Lie algebra for a given representation index k used to represent angular momentum (quantum spin) in rotation-equivariant computations.
    
    This function is used in DeepChem's equivariance utilities when constructing rotation-equivariant operators or representations (for example, in physics-informed or molecular models where three-dimensional rotations and angular momentum operators appear). Internally it obtains SU(2) generators for the representation index k via su2_generators(k), obtains the change-of-basis matrix from real to complex spherical harmonics via change_basis_real_to_complex(k), performs the similarity transform X <- conj(Q.T) @ X @ Q to move the SU(2) generators into the SO(3) (complex spherical harmonic) basis, and then returns the real part of the resulting matrices so the final SO(3) generators are purely real-valued.
    
    Args:
        k (int): The representation index that determines the order (dimension) of the angular-momentum representation used to build the generators. In the context of the SU(2)/SO(3) representations used by the helper functions called here, k indexes the representation returned by su2_generators(k) and the corresponding basis change returned by change_basis_real_to_complex(k). Practically, k selects the matrix size used for the angular-momentum operators (the resulting matrices operate on a (2*k + 1)-dimensional representation space). k must be provided as an integer; if k is not an integer or is an invalid index for the underlying helper functions, those functions will raise an error.
    
    Returns:
        torch.Tensor: A real-valued tensor containing a stack of three SO(3) generator matrices. The tensor has shape (3, 2*k + 1, 2*k + 1) corresponding to the three generators (ordered as J_x, J_z, J_y) acting on the (2*k + 1)-dimensional representation space. Each slice along the first dimension is a real square matrix representing one infinitesimal rotation generator in the returned SO(3) basis. The function guarantees the returned tensor is purely real by applying torch.real to the complex-valued result of the similarity transform.
    
    Raises:
        Exception: Errors raised by the underlying helper functions su2_generators or change_basis_real_to_complex are propagated. Common failure modes include providing an invalid k (e.g., a negative integer or a value not supported by the helper functions) or encountering numerical/shape mismatches during the matrix multiplications. The function performs no in-place mutation of inputs; its side effects are limited to calling the helper functions and allocating the returned torch.Tensor.
    """
    from deepchem.utils.equivariance_utils import so3_generators
    return so3_generators(k)


################################################################################
# Source: deepchem.utils.equivariance_utils.su2_generators
# File: deepchem/utils/equivariance_utils.py
# Category: valid
################################################################################

def deepchem_utils_equivariance_utils_su2_generators(k: int):
    """Generate the three matrix generators of the special unitary group SU(2) for the representation indexed by k.
    
    This function constructs the standard angular-momentum (SU(2)) generators used in quantum mechanics and symmetry analysis for a representation of order k. In physics terms, k corresponds to the spin quantum number j (here restricted to integer values by the function signature), and the representation dimension is 2*k + 1. The function builds the ladder (raising and lowering) operators from closed-form matrix elements sqrt(k*(k+1) - m*(m±1)), forms the J_x, J_y, and J_z operators using the usual linear combinations of ladder operators, and stacks them into a single torch.Tensor. These generators are commonly used to represent infinitesimal rotations, compute commutators, construct representation matrices for rotational symmetry, and test equivariance properties in machine-learning models that exploit SU(2) symmetry (as used elsewhere in DeepChem for physically informed modeling).
    
    Args:
        k (int): The representation index that determines the SU(2) representation (often denoted j in physics). For a non-negative integer k the function produces matrices of size (2*k + 1) x (2*k + 1); the representation describes spin-k angular-momentum operators. The integer k selects the set of magnetic quantum numbers m = -k, -k+1, ..., k and therefore fixes the ladder operator matrix elements via sqrt(k*(k+1) - m*(m±1)). Supplying a value outside the intended domain (for example a negative integer) will produce arithmetic sequences and matrix sizes according to the numpy-style arange semantics used internally and may lead to empty or invalid matrices or NaNs in square roots; the function signature requires an int and the construction assumes the conventional non-negative representation index used in quantum/angular-momentum contexts.
    
    Returns:
        torch.Tensor: A complex-valued tensor of shape (3, 2*k + 1, 2*k + 1) containing the three SU(2) generators stacked along the first dimension in the order [J_x, J_z, J_y]. Each slice is a square matrix representation of the corresponding angular-momentum operator in the chosen k representation. J_x is computed as 0.5*(J_+ + J_-), J_y as -0.5j*(J_+ - J_-), and J_z as 1j * diag(m) with m = -k,...,k. The returned tensor is produced using pure PyTorch operations (no in-place side effects on external state); its device and dtype are those inferred by the torch operations (the presence of complex imaginary unit 1j yields a complex-valued tensor). Numerical issues (for example due to invalid k or extremely large k) can produce NaNs or overflows in the square-root expressions; for conventional, modest non-negative integer k values used in angular-momentum applications this function yields the standard, real/complex-valued matrix generators that satisfy the SU(2) commutation relations.
    """
    from deepchem.utils.equivariance_utils import su2_generators
    return su2_generators(k)


################################################################################
# Source: deepchem.utils.evaluate.output_statistics
# File: deepchem/utils/evaluate.py
# Category: valid
################################################################################

def deepchem_utils_evaluate_output_statistics(scores: Dict[str, float], stats_out: str):
    """deepchem.utils.evaluate.output_statistics writes evaluation metric scores to a plain text file.
    
    This function is a small utility used in DeepChem evaluation pipelines to persist a mapping of metric names to numeric scores produced by model evaluation routines (for example during model validation or testing in drug discovery, materials science, quantum chemistry, or biological modeling workflows). It emits a deprecation warning via the module logger and then opens the specified output file in write mode, truncating any existing contents, and writes the Python string representation of the scores mapping followed by a newline. Because the function writes the raw str() form of the mapping, the output is intended for simple human inspection or quick experiment logging rather than robust machine parsing; for structured, machine-readable outputs consider using experiment tracking integrations (for example, Weights & Biases) or writing JSON externally. Typical failure modes include file system permission errors or other I/O errors raised when opening/writing the file.
    
    Args:
        scores (Dict[str, float]): Dictionary mapping metric names to numeric scores computed by DeepChem evaluation utilities. Each key is the canonical name of a metric (for example "roc_auc", "accuracy", "mae") and each value is the corresponding floating-point score produced by model evaluation. In the domain of molecular machine learning and related applications, these entries represent the quantitative performance measures used to compare models and track experiment progress.
        stats_out (str): Filesystem path (filename) to which the scores mapping will be written. The function opens this path with mode "w", which truncates any existing file at that path and creates the file if it does not exist. Provide an absolute or relative path as appropriate for your environment; insufficient permissions, non-existent directories, or other filesystem errors will raise an exception (e.g., OSError) from the underlying file I/O operations.
    
    Returns:
        None: This function does not return a value. Side effects: it logs a deprecation warning via the module logger and writes the stringified scores mapping followed by a newline to the file specified by stats_out, truncating any previous contents of that file.
    """
    from deepchem.utils.evaluate import output_statistics
    return output_statistics(scores, stats_out)


################################################################################
# Source: deepchem.utils.evaluate.relative_difference
# File: deepchem/utils/evaluate.py
# Category: valid
################################################################################

def deepchem_utils_evaluate_relative_difference(x: numpy.ndarray, y: numpy.ndarray):
    """Compute the elementwise relative difference between two numpy arrays as (x - y) / abs(y).
    
    This function is used in DeepChem for numerical comparison of two arrays, for example to measure relative error between model predictions and reference values in drug discovery, materials science, quantum chemistry, and biology workflows described in the project README. The operation is performed elementwise and returns a numpy.ndarray that contains the relative difference for each corresponding element pair from x and y.
    
    Args:
        x (numpy.ndarray): First input array. This is typically the numerator array (for example, predicted values from a model) and must have the same shape as y for a direct elementwise comparison. The function does not perform any in-place modification of x.
        y (numpy.ndarray): Second input array. This is used as the denominator (for example, ground-truth or reference values). y must have the same shape as x for a direct elementwise comparison. The function uses elementwise absolute value of y to form the denominator.
    
    Behavior, side effects, defaults, and failure modes:
    This function computes z = (x - y) / abs(y) using numpy semantics. It does not enforce explicit shape equality; the docstring and user guidance require x and y to have the same shape for a straightforward elementwise comparison. If shapes differ, numpy broadcasting rules will be applied where possible; if broadcasting is not possible, numpy will raise a ValueError. If any element of abs(y) is zero, division by zero will occur and numpy will produce inf or NaN values for those positions and may emit a RuntimeWarning. The return array's dtype and precision follow numpy's arithmetic and upcasting rules based on the input dtypes. Calling this function emits a FutureWarning advising users to directly use the equivalent expression (x - y) / np.abs(y) or to use numpy functions such as np.isclose or np.allclose when testing for tolerance; this warning is a side effect intended to guide users toward explicit or more robust alternatives.
    
    Returns:
        numpy.ndarray: Elementwise relative difference array with the same shape as the broadcasted shape of x and y. Each element is computed as (x_i - y_i) / abs(y_i). The result may contain infinities or NaNs when elements of y are zero, and numpy may raise runtime warnings in those cases.
    """
    from deepchem.utils.evaluate import relative_difference
    return relative_difference(x, y)


################################################################################
# Source: deepchem.utils.fake_data_generator.generate_edge_index
# File: deepchem/utils/fake_data_generator.py
# Category: valid
################################################################################

def deepchem_utils_fake_data_generator_generate_edge_index(
    n_nodes: int,
    avg_degree: int,
    remove_loops: bool = True
):
    """Generate a random edge index array containing source and destination node indices for synthetic graphs used in DeepChem utilities, tests, and examples. The function draws n_nodes * avg_degree random directed edges (as integer node indices) using NumPy and optionally removes self-loops.
    
    Args:
        n_nodes (int): Number of nodes in the graph. This integer is passed as the exclusive upper bound ("high") to numpy.random.randint and therefore defines the valid node index range [0, n_nodes - 1] for every source and destination entry. In practice within DeepChem this models the number of atoms or graph vertices in a synthetic molecular or material graph used to exercise graph-based models.
        avg_degree (int): Average degree per node. The function computes n_edges = n_nodes * avg_degree and allocates an initial edge index array with shape (2, n_edges). Each of the n_edges columns represents one directed edge as (source_index, destination_index). In practice this parameter controls the total number of sampled edges used to construct a fake graph for testing or benchmarking DeepChem graph algorithms.
        remove_loops (bool): If True (default), self-loops (edges where source_index == destination_index) are removed from the returned edge index by calling remove_self_loops(edge_index). If False, self-loops generated by random sampling are retained. This flag lets callers produce either strictly simple directed graphs (no self-loops) or allow self-edges for algorithms that require them.
    
    Behavior and side effects:
        The function computes n_edges = n_nodes * avg_degree and samples integers with numpy.random.randint(low=0, high=n_nodes, size=(2, n_edges)). Sampling is independent and uniform over node indices; reproducibility depends on NumPy's global random state and can be achieved by seeding numpy.random externally before calling this function. If remove_loops is True, the function invokes remove_self_loops(edge_index) to filter out self-loop columns; this can reduce the number of columns in the returned array. The function does not modify any input in-place and has no side effects other than advancing NumPy's random number generator state. The implementation assumes NumPy is available and that remove_self_loops is present in the module namespace.
    
    Failure modes and notes:
        The function does not perform explicit type coercion or comprehensive validation. Passing non-integer or incompatible types for n_nodes or avg_degree may raise TypeError or ValueError from NumPy. Providing n_nodes <= 0 or avg_degree < 0 will result in numpy.random.randint raising an error or numpy attempting to create arrays with non-positive dimensions; avg_degree == 0 yields an empty edge index with shape (2, 0). After removing self-loops it is possible for the returned edge index to have fewer columns than n_nodes * avg_degree, including being empty, if many or all sampled edges are self-loops.
    
    Returns:
        numpy.ndarray: A 2D integer array of shape (2, M) where each column is a pair (source_index, destination_index). M is initially n_nodes * avg_degree but may be smaller if remove_loops is True and self-loops are removed. Each index is in the integer range [0, n_nodes - 1] as produced by numpy.random.randint. The array is suitable for use as an edge_index in DeepChem graph utilities and downstream graph model inputs.
    """
    from deepchem.utils.fake_data_generator import generate_edge_index
    return generate_edge_index(n_nodes, avg_degree, remove_loops)


################################################################################
# Source: deepchem.utils.fake_data_generator.remove_self_loops
# File: deepchem/utils/fake_data_generator.py
# Category: valid
################################################################################

def deepchem_utils_fake_data_generator_remove_self_loops(edge_index: numpy.ndarray):
    """Removes self-loops from an edge list represented as a 2-row numpy array.
    
    This function is used in DeepChem's graph utilities and fake data generation to filter out self-loop edges (edges that start and end on the same node) from an edge index representation. In the context of molecular and biological graph data used throughout DeepChem, self-loops typically do not represent meaningful chemical or structural relationships, so they are removed prior to downstream processing (e.g., graph neural network input preparation or dataset sanitization). The function scans each column (edge) and retains only those columns where the source node index differs from the target node index.
    
    Args:
        edge_index (numpy.ndarray): A numpy array of shape (2, num_edges) representing a list of edges in a graph. The first row contains source node indices and the second row contains target node indices. Each column corresponds to a single directed edge (source -> target). The array's dtype and numeric indexing semantics are preserved in the output.
    
    Returns:
        numpy.ndarray: A new numpy array with shape (2, num_filtered_edges) containing only the columns from edge_index that are not self-loops (i.e., columns where edge_index[0, i] != edge_index[1, i]). The returned array preserves the input array's ordering of surviving edges and uses the same element dtype as the input.
    
    Behavior and side effects:
        The function performs an element-wise equality check between the first and second rows of edge_index for each column and constructs a mask of columns to keep. It then returns edge_index[:, mask]. The original input array is not modified in-place; a view or copy may be returned depending on numpy's indexing semantics, but callers should not rely on an in-place modification.
    
    Complexity:
        Time complexity is O(num_edges) because the function examines each edge exactly once. Memory overhead is proportional to the number of retained edges for the returned array.
    
    Failure modes and assumptions:
        The function assumes edge_index has two rows (shape[0] == 2) and a non-negative number of columns. If edge_index does not have shape (2, num_edges), behavior is undefined and the function may raise an IndexError or produce unexpected results. The equality comparison used to detect self-loops relies on numpy's element comparison semantics; inputs whose entries are not comparable by == may lead to errors.
    """
    from deepchem.utils.fake_data_generator import remove_self_loops
    return remove_self_loops(edge_index)


################################################################################
# Source: deepchem.utils.genomics_utils.encode_bio_sequence
# File: deepchem/utils/genomics_utils.py
# Category: valid
################################################################################

def deepchem_utils_genomics_utils_encode_bio_sequence(
    fname: str,
    file_type: str = "fasta",
    letters: str = "ATCGN"
):
    """deepchem.utils.genomics_utils.encode_bio_sequence: Load a biological sequence file (FASTA/FASTQ) and return a one-hot encoded numpy array suitable for use in genomics-focused deep learning workflows.
    
    This function is part of DeepChem's genomics utilities and is used to convert sequence files into a numerical representation (one-hot encoding) that downstream models (for example convolutional neural networks or other sequence models) can consume. It uses BioPython's SeqIO.parse to read the sequence file specified by fname and then passes the parsed sequence records to seq_one_hot_encode to produce the final array. The ordering of letters controls the order of channels in the output and the default includes the common DNA alphabet with an ambiguity character ('N'). This function performs file I/O (reads the file at fname) and depends on BioPython being installed; if BioPython is not present an ImportError is raised. File reading or parsing errors from BioPython (for example FileNotFoundError or parsing of a malformed file) will propagate to the caller.
    
    Args:
        fname (str): Path to the input sequence file on disk. This is the filename (or path) passed directly to BioPython.SeqIO.parse; it must point to a readable file containing biological sequences encoded in the format indicated by file_type. In DeepChem workflows this is typically a FASTA or FASTQ file containing DNA or RNA sequences to be converted into model inputs.
        file_type (str): The sequence file format string passed to BioPython.SeqIO.parse to select the parser (for example "fasta" or "fastq"). The default is "fasta". This parameter controls how SeqIO reads and interprets records in the file; it must be a format name supported by the installed BioPython version.
        letters (str): String of characters defining the alphabet and channel ordering used for one-hot encoding (for example "ATCGN"). The length of this string determines the second axis size in the returned array and the characters (in this order) correspond to the channel dimension produced by seq_one_hot_encode. Sequences in the input file are expected to consist of characters from this set (the default includes the common DNA bases plus 'N' for ambiguous positions).
    
    Returns:
        numpy.ndarray: A numpy array containing one-hot encoded sequences with shape (N_sequences, N_letters, sequence_length, 1). N_sequences is the number of sequence records read from fname, N_letters equals len(letters) and corresponds to the alphabet/order provided, sequence_length is the length of the sequences as represented by seq_one_hot_encode, and the final dimension is a single-channel axis (commonly used as a channel dimension for convolutional models). The returned array is intended for direct use as input tensors to DeepChem models or other machine-learning pipelines expecting a one-hot encoded representation.
    
    Notes on behavior and failure modes:
        - This function requires BioPython to be installed; if BioPython cannot be imported, an ImportError is raised before any file I/O occurs.
        - The function reads the file at fname using SeqIO.parse and therefore can raise standard file I/O errors (e.g., FileNotFoundError, PermissionError) or parsing errors produced by BioPython for malformed input.
        - The exact handling of sequences of differing lengths is determined by seq_one_hot_encode; users should ensure sequences are of the expected length for their downstream models or preprocess them accordingly.
    """
    from deepchem.utils.genomics_utils import encode_bio_sequence
    return encode_bio_sequence(fname, file_type, letters)


################################################################################
# Source: deepchem.utils.genomics_utils.seq_one_hot_encode
# File: deepchem/utils/genomics_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def deepchem_utils_genomics_utils_seq_one_hot_encode(
    sequences: numpy.ndarray,
    letters: str = "ATCGN"
):
    """One-hot encodes a collection of genomic sequences into an image-like NumPy array for use in convolutional or other models that expect channeled image inputs in genomics and computational biology workflows (for example, converting DNA sequences to inputs for CNNs used in regulatory element prediction or sequence-based models in drug discovery). The function maps each character in the input sequences to a channel index according to the provided letters string and returns a 4D array where the second axis indexes letters and the final axis is a singleton channel dimension so the result can be treated as an image with one color channel.
    
    Args:
        sequences (numpy.ndarray or Iterator[Bio.SeqRecord]): Iterable container of genetic sequences to encode. If a numpy.ndarray is provided, the function expects an array-like sequence of equal-length sequence objects (e.g., strings, Bio.SeqRecord objects whose string representation yields the sequence) and will index into the array to obtain the first element then process the remainder via slicing (sequences[1:]). If an iterator is provided, the function will call next(sequences) to obtain the first sequence and then iterate the remainder from the same iterator; therefore the iterator will be partially consumed (the first element is removed) as a side effect. All sequences must support len() and iteration over characters and must have the same length; otherwise a ValueError is raised. This parameter is the primary input representing the DNA/RNA sequences to convert to a neural-network-ready numeric representation used in DeepChem genomics utilities.
        letters (str): String containing the ordered set of possible letters (alphabet) present in the sequences. Default is "ATCGN". The order of characters in this string determines the ordering of channels in the output (the second axis of the returned array). The function builds an internal mapping from each character in letters to a channel index; any character in the input sequences that is not present in letters will cause a KeyError during encoding. Use this parameter to control which nucleotides or ambiguous symbols are recognized and their channel ordering for downstream models.
    
    Returns:
        np.ndarray: A NumPy array of shape (N_sequences, N_letters, sequence_length, 1). N_sequences is the number of sequences provided, N_letters is len(letters), and sequence_length is the length of each input sequence. The array contains a one-hot style numeric encoding along the second axis: for each sequence and sequence position, the element corresponding to the present letter is set (typically 1) and other elements are unset (typically 0). The final axis is a singleton channel dimension so the output can be consumed by image-based model components that expect a channel-last tensor.
    
    Raises:
        ValueError: If the provided sequences do not all have the same length. This function checks the length of the first sequence and enforces that all subsequent sequences match that length; mismatch triggers this error.
        KeyError: If an input sequence contains a character not present in letters, the internal mapping cannot encode that character and a KeyError will be raised during encoding.
    
    Notes:
        - The function is intended for preparing fixed-length genomic sequences for machine learning models in DeepChem (e.g., CNNs treating sequences as single-channel images).
        - When passing an iterator for sequences, be aware that the iterator will be advanced (its first element consumed) and cannot generally be reused without reinitialization.
        - The letters parameter controls both which characters are considered valid and the channel ordering in the returned array; ensure it matches the alphabet expected by downstream models.
    """
    from deepchem.utils.genomics_utils import seq_one_hot_encode
    return seq_one_hot_encode(sequences, letters)


################################################################################
# Source: deepchem.utils.geometry_utils.angle_between
# File: deepchem/utils/geometry_utils.py
# Category: valid
################################################################################

def deepchem_utils_geometry_utils_angle_between(
    vector_i: numpy.ndarray,
    vector_j: numpy.ndarray
):
    """deepchem.utils.geometry_utils.angle_between returns the smaller angle (in radians) between two 3-dimensional vectors. This function is used in geometric calculations common in DeepChem workflows (for example, computing bond angles or relative orientations of atomic displacement vectors in molecular and materials modelling) and returns a scalar float value in the range [0, pi].
    
    Args:
        vector_i (numpy.ndarray): A numpy array of shape (3,), where the three elements correspond to the x, y, z components of the first vector. This argument represents one of the two 3D direction vectors whose inter-angle is being measured. The function expects a numeric 3-element array; no explicit shape validation is performed in this function, so supplying arrays of incorrect shape or non-numeric entries will cause downstream numpy operations (unit_vector, np.dot, np.arccos) to raise errors.
        vector_j (numpy.ndarray): A numpy array of shape (3,), where the three elements correspond to the x, y, z components of the second vector. This argument represents the second 3D direction vector. As with vector_i, this should be a numeric numpy array of length 3; improper inputs may result in numpy exceptions.
    
    Returns:
        float: The angle in radians between the two vectors. The returned value is the smaller of the two possible angles between the input directions and therefore lies between 0 and pi (inclusive). Practically, a return value of 0.0 indicates the vectors are (numerically) colinear and pointing in the same direction; pi indicates they are (numerically) colinear but opposite; intermediate values correspond to the geometric angle between directions.
    
    Behavior and failure modes:
        The function normalizes both input vectors using unit_vector and computes the angle via arccos of their dot product. Due to floating-point rounding, the dot product may occasionally fall slightly outside the closed interval [-1, 1], causing np.arccos to produce NaN. In that case the implementation performs a numeric fallback: if the normalized vectors are numerically close (np.allclose), the function returns 0.0; otherwise it returns pi. This fallback handles common numerical instability when vectors are effectively parallel or antiparallel. If inputs are degenerate (for example, zero-length vectors) or contain NaNs/infs, unit_vector and subsequent numpy operations may produce NaNs or raise exceptions; the function does not perform explicit zero-length checks, so callers should validate inputs when necessary. There are no side effects; the function does not modify its inputs.
    """
    from deepchem.utils.geometry_utils import angle_between
    return angle_between(vector_i, vector_j)


################################################################################
# Source: deepchem.utils.geometry_utils.compute_centroid
# File: deepchem/utils/geometry_utils.py
# Category: valid
################################################################################

def deepchem_utils_geometry_utils_compute_centroid(coordinates: numpy.ndarray):
    """Compute the (x, y, z) centroid of the provided atomic coordinates.
    
    This utility computes the arithmetic mean position (centroid) over the set of 3D coordinates supplied. In the DeepChem molecular machine-learning context (drug discovery, materials science, quantum chemistry, biology), this centroid is commonly used as the geometric center of a molecule's atom positions for preprocessing tasks such as translating a molecule to the origin, centering coordinate-based features, or as a reference point for alignment and distance calculations. This function computes the center of geometry (simple mean of coordinates) and is distinct from a mass-weighted center of mass.
    
    Args:
        coordinates (numpy.ndarray): A numpy array of shape (N, 3), where N is the number of atoms and each row is an (x, y, z) coordinate for an atom. This argument represents the 3D positions of atoms in a molecule or structure; providing correctly shaped coordinates is required for the returned centroid to correspond to (x, y, z) geometry used in downstream DeepChem featurizers, alignments, and models. The function does not modify this array in-place.
    
    Returns:
        numpy.ndarray: A numpy array of shape (3,) containing the centroid coordinates (x, y, z) computed as the arithmetic mean across the N input coordinates. The returned array is the geometric center used for centering or referencing molecular coordinates in DeepChem workflows.
    
    Behavior and failure modes:
        The centroid is computed via numpy.mean(coordinates, axis=0). If coordinates is an empty array with shape (0, 3), numpy.mean will produce a length-3 array of NaNs and emit a RuntimeWarning; callers should check for empty inputs before calling if NaNs are not acceptable. If coordinates does not have shape (N, 3) (for example, wrong number of columns), the semantic meaning of the result will not be a 3D centroid and may lead to downstream errors; the function expects a two-dimensional numpy.ndarray with three columns corresponding to x, y, z. If the argument is not a numpy.ndarray, behavior is undefined and a TypeError or AttributeError may occur when numpy.mean is invoked; callers should ensure the input type matches the signature. There are no side effects: the input array is not altered and the function returns a new numpy.ndarray.
    """
    from deepchem.utils.geometry_utils import compute_centroid
    return compute_centroid(coordinates)


################################################################################
# Source: deepchem.utils.geometry_utils.compute_pairwise_distances
# File: deepchem/utils/geometry_utils.py
# Category: valid
################################################################################

def deepchem_utils_geometry_utils_compute_pairwise_distances(
    first_coordinate: numpy.ndarray,
    second_coordinate: numpy.ndarray
):
    """Computes pairwise Euclidean distances between atoms of two molecules.
    
    This function is used in DeepChem workflows (for example in drug discovery, materials
    science, quantum chemistry, and computational biology) to build distance matrices,
    contact maps, and geometric descriptors from 3D atomic coordinates. Given two sets
    of 3D coordinates representing two molecules (or two fragments/ensembles of atoms),
    it returns an m-by-n array where the entry at (i, j) is the Euclidean distance
    (in Angstroms) between the i-th atom in the first set and the j-th atom in the
    second set. The function delegates the numeric computation to an optimized
    pairwise distance routine and performs no in-place modification of its inputs.
    
    Args:
        first_coordinate (numpy.ndarray): A numpy array of shape (m, 3) representing
            the 3D Cartesian coordinates of the first molecule's m atoms. Each row
            corresponds to one atom and the three columns correspond to X, Y, Z
            coordinates in Angstroms. Supplying coordinates in a different unit will
            produce distances in that unit; DeepChem conventions typically use
            Angstroms.
        second_coordinate (numpy.ndarray): A numpy array of shape (n, 3) representing
            the 3D Cartesian coordinates of the second molecule's n atoms. Each row
            corresponds to one atom and the three columns correspond to X, Y, Z
            coordinates in the same units as first_coordinate.
    
    Returns:
        numpy.ndarray: A numpy array of shape (m, n) whose (i, j) element is the
        Euclidean distance between the i-th atom of first_coordinate and the j-th
        atom of second_coordinate. Distances are computed with the standard
        Euclidean (L2) metric and are reported in the same linear units as the input
        coordinates (commonly Angstroms in DeepChem datasets).
    
    Behavior, side effects, and failure modes:
        - The function is pure (no side effects) and does not modify first_coordinate
          or second_coordinate.
        - Both inputs must be two-dimensional numpy.ndarray objects with size-3
          trailing dimensions (shapes (m, 3) and (n, 3)). If an input does not have
          shape (k, 3), the underlying distance routine may raise a ValueError or
          produce undefined results; callers should validate shapes before calling.
        - If either input contains NaN or infinite values those values will propagate
          into the output distances according to IEEE floating-point rules.
        - Memory and compute scale as O(m * n): large values of m and n can cause
          high memory usage and long runtimes; for very large systems consider
          blockwise computation or sparse approximations.
        - The implementation uses an optimized pairwise distance routine (Euclidean
          metric). Any precision/dtype behavior is determined by the underlying
          numeric library used for the computation.
        - Empty inputs are allowed in principle: if m == 0 or n == 0 the function
          returns an array with shape (m, n) and zero elements.
    """
    from deepchem.utils.geometry_utils import compute_pairwise_distances
    return compute_pairwise_distances(first_coordinate, second_coordinate)


################################################################################
# Source: deepchem.utils.geometry_utils.compute_protein_range
# File: deepchem/utils/geometry_utils.py
# Category: valid
################################################################################

def deepchem_utils_geometry_utils_compute_protein_range(coordinates: numpy.ndarray):
    """Compute the protein range of provided coordinates.
    
    This function computes the axis-aligned extent (size) of a protein (or any set of atoms)
    along the three Cartesian axes by taking the element-wise difference between the
    maximum and minimum coordinates across all atoms. In DeepChem workflows (for example,
    preparing inputs for voxelization, bounding-box cropping, or spatial featurizers in
    drug discovery and structural biology), this range vector is used to determine the
    size of the protein along x, y, and z and to guide grid sizing, padding, or scale
    normalization. The returned values are in the same units as the input coordinates
    (typically Angstroms when coordinates are read from PDB files).
    
    Args:
        coordinates (numpy.ndarray): A numpy array of shape (N, 3), where N is the
            number of atoms and the three columns correspond to Cartesian x, y, z
            coordinates. The array must be numeric and finite for meaningful results.
            The function treats rows as independent atom positions and computes the
            per-axis maxima and minima across rows. Providing an empty array, an array
            with an unexpected shape (not 2D with size-3 second dimension), or arrays
            containing only NaNs will result in errors or propagate NaNs from numpy
            operations (see failure modes below).
    
    Returns:
        numpy.ndarray: A 1-D numpy array of shape (3,), containing the protein range
        along each Cartesian axis in the order (x, y, z). The value at each index is
        computed as max(coordinates[:, i]) - min(coordinates[:, i]) for i in {0,1,2}.
        This vector represents the length of the axis-aligned bounding box spanning
        the input coordinates and can be used directly to size spatial grids or to
        compute padding requirements.
    
    Behavior, defaults, and failure modes:
        The function performs a pure, deterministic computation with no side effects.
        It uses numpy's max and min along axis=0 and returns their difference. If
        coordinates has shape (N, 3) with N >= 1 and contains finite numeric values,
        the result is well-defined. If coordinates is empty (N == 0), numpy.max and
        numpy.min will raise a ValueError. If coordinates has an incompatible shape
        (for example not 2-D or with second dimension not equal to 3), numpy will
        raise an AxisError or produce unexpected results. If coordinates contain NaN
        values, numpy's max/min will propagate NaNs and the returned protein range
        will contain NaNs for any axis with NaNs in the inputs. The function does not
        validate units; callers are responsible for ensuring coordinate units are
        consistent with downstream expectations.
    """
    from deepchem.utils.geometry_utils import compute_protein_range
    return compute_protein_range(coordinates)


################################################################################
# Source: deepchem.utils.geometry_utils.is_angle_within_cutoff
# File: deepchem/utils/geometry_utils.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for is_angle_within_cutoff because the docstring has no description for the argument 'angle_cutoff'
################################################################################

def deepchem_utils_geometry_utils_is_angle_within_cutoff(
    vector_i: numpy.ndarray,
    vector_j: numpy.ndarray,
    angle_cutoff: float
):
    """deepchem.utils.geometry_utils.is_angle_within_cutoff: Determine whether two 3-D vectors are within a specified angular deviation (in degrees) of being exactly 180 degrees apart. This function is used in DeepChem geometry utilities to detect anti-parallel or nearly anti-parallel directions (for example, opposing bond directions or opposite normals) by computing the angle between vectors in degrees and testing whether that angle lies strictly within the symmetric interval (180 - angle_cutoff, 180 + angle_cutoff).
    
    Args:
        vector_i (numpy.ndarray): A 1-D numpy array of shape (3,) representing a 3D vector with components (x, y, z). In molecular and materials contexts (per the DeepChem project), this typically encodes a bond direction, displacement, or normal vector. The function expects a real-valued vector of length 3; providing arrays of other shapes may lead to incorrect results or exceptions.
        vector_j (numpy.ndarray): A 1-D numpy array of shape (3,) representing a second 3D vector with components (x, y, z). As with vector_i, this usually represents a geometric direction in molecular structures. The relative orientation of vector_i and vector_j is tested against the 180-degree criterion.
        angle_cutoff (float): The allowed deviation from exactly 180 degrees, expressed in degrees. For example, angle_cutoff = 5.0 means the function returns True when the angle between vector_i and vector_j is strictly greater than 175.0 degrees and strictly less than 185.0 degrees. The comparison is strict (uses > and <), so exact boundary values equal to 180 +/- angle_cutoff do not return True.
    
    Returns:
        bool: True if the angle between vector_i and vector_j, computed as angle_between(vector_i, vector_j) and converted from radians to degrees, lies strictly within the open interval (180 - angle_cutoff, 180 + angle_cutoff). False otherwise. There are no side effects; the function performs no in-place modification of inputs.
    
    Behavior, defaults, and failure modes:
        The function computes the angle in radians via the internal angle_between routine, converts it to degrees by multiplying by 180/pi, and then applies the strict range test described above. It assumes both inputs are finite, non-zero 3D vectors; zero-length vectors or inputs with incorrect shape or non-numeric entries can lead to undefined behavior, exceptions (for example from the underlying angle computation), or NaN results. The function does not clamp or normalize angle_cutoff; callers should ensure angle_cutoff is non-negative and given in degrees. The function is lightweight and has no side effects on the provided numpy arrays.
    """
    from deepchem.utils.geometry_utils import is_angle_within_cutoff
    return is_angle_within_cutoff(vector_i, vector_j, angle_cutoff)


################################################################################
# Source: deepchem.utils.geometry_utils.rotate_molecules
# File: deepchem/utils/geometry_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def deepchem_utils_geometry_utils_rotate_molecules(mol_coordinates_list: list):
    """deepchem.utils.geometry_utils.rotate_molecules rotates a collection of molecular Cartesian coordinate sets by a single random 3D rotation matrix.
    
    Rotates each molecule's 3D atomic coordinates by generating one random rotation matrix via generate_random_rotation_matrix() and applying that same rotation to every molecule in mol_coordinates_list. This is typically used in DeepChem workflows for molecular machine learning and data augmentation in drug discovery, materials science, and computational chemistry to produce rotated variants of molecular geometries while preserving interatomic distances and overall molecular shape. The random rotation matrix is drawn so that applying it to a 3-vector uniformly samples orientations on the sphere of radius equal to the vector norm, as implemented by generate_random_rotation_matrix(). The function performs a deep copy of each input coordinate array before applying the rotation, so the input objects are not modified.
    
    Args:
        mol_coordinates_list (list): A list of molecular coordinate arrays or array-like objects. Each entry represents one molecule's atomic coordinates and is expected to be a 2-D sequence with shape (num_atoms, 3), where the last dimension corresponds to the Cartesian x, y, z coordinates for each atom. The list order is preserved in the output. In practice within DeepChem, these coordinate arrays are numeric (e.g., NumPy arrays) produced by molecular preprocessing pipelines; providing non-numeric items or arrays that do not have a final dimension of size 3 will lead to runtime errors.
    
    Returns:
        list: A list of rotated coordinate arrays with the same length and ordering as mol_coordinates_list. Each returned element contains the coordinates of the corresponding input molecule after application of the same random rotation matrix; the per-molecule shape (num_atoms, 3) is preserved. If mol_coordinates_list is empty, an empty list is returned.
    
    Behavior, side effects, and failure modes:
        This function generates one random rotation matrix and applies it to every molecule in the provided list, providing a consistent orientation change across the batch. The function uses deepcopy internally, so it does not mutate the original input objects. It relies on NumPy linear algebra operations; inputs should be convertible to numeric NumPy arrays. If an element of mol_coordinates_list does not have a compatible shape (for example, not two-dimensional or its last dimension is not size 3), NumPy will raise an error (e.g., ValueError or IndexError) during the transpose or dot product. The randomness depends on the global random state used by generate_random_rotation_matrix() and will produce different rotations across calls unless that random state is controlled externally.
    """
    from deepchem.utils.geometry_utils import rotate_molecules
    return rotate_molecules(mol_coordinates_list)


################################################################################
# Source: deepchem.utils.geometry_utils.subtract_centroid
# File: deepchem/utils/geometry_utils.py
# Category: valid
################################################################################

def deepchem_utils_geometry_utils_subtract_centroid(
    coordinates: numpy.ndarray,
    centroid: numpy.ndarray
):
    """deepchem.utils.geometry_utils.subtract_centroid subtracts a 3-D centroid vector from every atom coordinate in a molecular coordinate array, centering the molecule at the origin for downstream geometric or machine-learning tasks (for example, preparing inputs for models in DeepChem used in drug discovery and computational chemistry).
    
    This function performs an in-place, componentwise subtraction of the provided centroid (x, y, z) from each row of the coordinates array. It is intended for preprocessing molecular geometries represented as NumPy arrays, making coordinate sets translation-invariant for algorithms that assume centered inputs. The operation mutates the input coordinates array and also returns it for convenience; no new array is allocated by this function beyond NumPy's in-place update.
    
    Args:
        coordinates (numpy.ndarray): A NumPy array of shape (N, 3), where N is the number of atoms in the molecule. Each row is an atom coordinate in 3D space in the order (x, y, z). This array is modified in place: after the call, each row equals the original row minus the centroid. The function expects numeric dtype (e.g., float32/float64) compatible with subtraction; if the array has an incompatible shape or dtype, NumPy broadcasting or arithmetic errors will occur.
        centroid (numpy.ndarray): A NumPy array of shape (3,), representing the centroid vector (x, y, z) to subtract from each atom coordinate. The centroid is interpreted componentwise and is broadcast against the coordinates rows. The centroid should be a one-dimensional length-3 numeric array; using a differently shaped array may trigger NumPy broadcasting rules or runtime errors.
    
    Returns:
        numpy.ndarray: The same NumPy array object passed as `coordinates`, now mutated so that each atom coordinate has had `centroid` subtracted. The returned array has shape (N, 3) and contains the centered coordinates. Note that because the update is in place, callers that keep references to the original `coordinates` array will observe the changed values.
    
    Behavior and failure modes:
        The subtraction is performed in place via NumPy broadcasting (coordinates -= np.transpose(centroid)). If `coordinates` is not shaped (N, 3) or `centroid` is not shaped (3,), NumPy may raise a broadcasting or arithmetic error (e.g., ValueError) at runtime. The function does not perform explicit validation of shapes or dtypes before attempting the operation. Users who require non-destructive behavior should pass a copy of `coordinates` (for example, coordinates.copy()) to avoid mutating the original array.
    """
    from deepchem.utils.geometry_utils import subtract_centroid
    return subtract_centroid(coordinates, centroid)


################################################################################
# Source: deepchem.utils.geometry_utils.unit_vector
# File: deepchem/utils/geometry_utils.py
# Category: valid
################################################################################

def deepchem_utils_geometry_utils_unit_vector(vector: numpy.ndarray):
    """deepchem.utils.geometry_utils.unit_vector returns the unit (normalized) vector of a three-dimensional Cartesian vector. This utility is part of DeepChem's geometry utilities used in molecular machine learning and computational chemistry workflows (for example, to normalize bond direction vectors, compute molecular orientation axes, or prepare direction features for models).
    
    Args:
        vector (numpy.ndarray): A numpy array of shape `(3,)` representing a 3D Cartesian vector (x, y, z). This argument is the input vector to be normalized. The function expects exactly three components as used in DeepChem geometry operations; providing a different shape is not supported by this implementation and may lead to incorrect results or runtime errors.
    
    Returns:
        numpy.ndarray: A numpy array of shape `(3,)` representing the unit vector of the input, computed as vector / ||vector|| where ||vector|| is the Euclidean (L2) norm computed with numpy.linalg.norm. The returned array is the normalized direction of the input and is intended for use in downstream geometric calculations in DeepChem (e.g., computing directional features for molecules).
    
    Behavior and failure modes:
        The function performs elementwise division of the input vector by its Euclidean norm and does not modify the input array in-place; it returns a new numpy.ndarray. If the input vector has zero magnitude (norm == 0.0), the division produces undefined values (NaNs or Infs) and numpy may emit a RuntimeWarning for division by zero; callers should check the norm before calling this function if zero vectors are possible in their data. No device or dtype coercion is performed beyond numpy's default arithmetic rules; the function relies on numpy semantics for dtype handling and will preserve numpy's behaviour for floating/integer inputs.
    """
    from deepchem.utils.geometry_utils import unit_vector
    return unit_vector(vector)


################################################################################
# Source: deepchem.utils.graph_utils.aggregate_max
# File: deepchem/utils/graph_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def deepchem_utils_graph_utils_aggregate_max(h: torch.Tensor, **kwargs):
    """aggregate_max Computes the element-wise maximum of a tensor along its second-to-last dimension.
    
    This function is part of DeepChem's graph utilities and is used to perform a max-aggregation (max pooling) operation across the second-to-last axis of a feature tensor. In graph and molecular machine learning workflows (as in DeepChem), this operation is commonly used to aggregate node or neighbor feature information into a summary representation by taking the maximum value across that axis. The function returns only the maximum values for each element across that axis (it does not return the indices of the maxima).
    
    Args:
        h (torch.Tensor): Input tensor whose values will be reduced by taking the maximum along the second-to-last dimension (dimension index -2). The tensor is expected to be at least two-dimensional so that the second-to-last axis exists; typical usages in DeepChem treat h as a batch of per-node or per-neighbor feature tensors, but the function does not enforce any specific shape beyond supporting dim=-2.
        kwargs (dict): Additional keyword arguments accepted for API compatibility with higher-level callers. These keyword arguments are ignored by this implementation and have no effect on the computation or return value. Accepting kwargs allows callers to pass through unused options without raising an error.
    
    Returns:
        torch.Tensor: A tensor containing the maximum values of h taken along the second-to-last dimension. The returned tensor has the same dtype and device as the input h and its rank is one less than the rank of h (the reduced axis is removed). This function returns only the values (not the indices) produced by torch.max(h, dim=-2)[0].
    
    Behavior, defaults, and failure modes:
        This function performs a reduction using torch.max over dim=-2 and selects the values output. There are no side effects: it does not modify the input tensor h in-place. If h has fewer than two dimensions (i.e., the second-to-last axis does not exist), PyTorch will raise an IndexError or RuntimeError; callers should ensure h.ndim >= 2. If the size of the second-to-last dimension is zero (an empty reduction), PyTorch may raise a runtime error depending on the backend. The function preserves the input tensor's device (CPU or GPU) and dtype; ensure that torch is properly configured for the intended device. The kwargs parameter is ignored and will not alter behavior.
    """
    from deepchem.utils.graph_utils import aggregate_max
    return aggregate_max(h, **kwargs)


################################################################################
# Source: deepchem.utils.graph_utils.aggregate_mean
# File: deepchem/utils/graph_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def deepchem_utils_graph_utils_aggregate_mean(h: torch.Tensor, **kwargs):
    """deepchem.utils.graph_utils.aggregate_mean computes the arithmetic mean of an input PyTorch tensor over its second-to-last dimension. This function is provided in DeepChem's graph utilities and is commonly used in graph neural network (GNN) code paths within DeepChem to aggregate features across a neighbor or atom dimension when building models for drug discovery, materials science, quantum chemistry, and biology.
    
    This operation is a pure, functional tensor reduction: it does not modify the input tensor in-place and yields a new torch.Tensor. The operation is differentiable through PyTorch's autograd, so it can be used in model forward passes and participate in gradient-based optimization.
    
    Args:
        h (torch.Tensor): Input tensor to be reduced. The function computes the mean along the tensor's second-to-last dimension (dimension index -2). The returned tensor will have the same rank as h minus one, with the second-to-last dimension removed. Typical usage in DeepChem graph utilities is to aggregate per-neighbor or per-atom feature vectors into a single representation along that axis.
        kwargs (dict): Additional keyword arguments accepted for API compatibility. This function does not use or forward any keyword arguments presently; kwargs are ignored. Providing keyword arguments will not change behavior but may be used by callers that pass through extra parameters for uniformity across aggregation functions.
    
    Returns:
        torch.Tensor: A new tensor containing the arithmetic mean of h along its second-to-last dimension. The output shape is h.shape with the second-to-last dimension removed. The operation preserves device placement (CPU/GPU) consistent with PyTorch semantics and is compatible with autograd for gradient propagation.
    
    Raises:
        IndexError: If h has fewer than two dimensions, indexing dimension -2 is invalid and PyTorch will raise an IndexError.
        TypeError: If h is not a torch.Tensor, PyTorch reduction semantics will raise a TypeError.
    """
    from deepchem.utils.graph_utils import aggregate_mean
    return aggregate_mean(h, **kwargs)


################################################################################
# Source: deepchem.utils.graph_utils.aggregate_min
# File: deepchem/utils/graph_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def deepchem_utils_graph_utils_aggregate_min(h: torch.Tensor, **kwargs):
    """Compute the elementwise minimum of the input tensor along its second-to-last dimension.
    
    Args:
        h (torch.Tensor): Input tensor. In DeepChem's graph utilities this tensor typically represents a batch of per-node or per-edge feature arrays (for example, a tensor shaped (batch_size, num_neighbors, feature_dim) or similar). The function computes the minimum values over the second-to-last axis (dimension index -2). The returned tensor preserves h's dtype and device and has the second-to-last dimension removed (i.e., the reduction collapses that axis). The function requires that h has at least two dimensions; calling this with a tensor of dimensionality less than 2 will raise the underlying PyTorch indexing error.
        kwargs (dict): Additional keyword arguments. These are accepted for API compatibility with other aggregation functions in deepchem.utils.graph_utils but are not used by this implementation; they are ignored. No side effects result from passing keyword arguments.
    
    Returns:
        torch.Tensor: A tensor containing the minimum values computed along h's second-to-last dimension. The shape is the same as h except that the second-to-last axis is removed. The operation is pure (no in-place modification of h). NaN values in h propagate according to PyTorch semantics (i.e., if NaNs are present in the slice being reduced, the corresponding output element will be NaN). This aggregation is commonly used in graph neural network pipelines within DeepChem to pool features across neighbors or variable-size sets (for example, taking the minimum over neighbor feature vectors when constructing node-level or graph-level descriptors for molecular machine learning).
    """
    from deepchem.utils.graph_utils import aggregate_min
    return aggregate_min(h, **kwargs)


################################################################################
# Source: deepchem.utils.graph_utils.aggregate_moment
# File: deepchem/utils/graph_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def deepchem_utils_graph_utils_aggregate_moment(h: torch.Tensor, n: int = 3, **kwargs):
    """Compute the n-th central moment aggregated along the second-to-last dimension of a tensor and return its signed n-th root.
    
    This function is used in DeepChem's graph utilities to aggregate feature tensors (for example, node feature arrays in molecular/graph representations) into higher-order distributional descriptors that capture variance/skewness/kurtosis-like information. Concretely, for an input tensor h the function computes h_mean = mean(h, dim=-2, keepdim=True), then h_n = mean((h - h_mean)**n, dim=-2), and finally returns sign(h_n) * (abs(h_n) + EPS)**(1.0 / n). EPS is a small, module-level constant added to the absolute value before taking the n-th root to improve numerical stability. The second-to-last dimension of h is reduced (aggregated) by this operation; the returned tensor has that dimension removed.
    
    Args:
        h (torch.Tensor): Input tensor to aggregate. In DeepChem graph workflows this is typically a tensor of per-node or per-element features where the second-to-last dimension indexes the elements to be aggregated (for example, nodes in a graph). The tensor must have at least two dimensions so that the second-to-last dimension (dim=-2) exists; if it does not, an exception (IndexError) will be raised. The aggregation computes statistics along this specific axis and returns a tensor with that axis removed.
        n (int): The order of the moment to compute. Default is 3. This should be a positive integer (n > 0). If n is zero or a non-positive integer, the behavior is undefined and may raise an error (for example ZeroDivisionError when computing 1.0 / n) or produce NaNs. Typical use in DeepChem is n=2 for RMS-like aggregation or n=3 for a signed cubic moment emphasizing skew.
        kwargs (dict): Additional keyword arguments accepted for API compatibility. These are not used by this implementation and are ignored. Passing unknown keys will not change behavior; they exist so higher-level code can forward extra options without breaking this function.
    
    Returns:
        torch.Tensor: A tensor containing the signed n-th root of the n-th central moment computed along the second-to-last dimension of h. The returned tensor shape equals the input shape with the second-to-last dimension removed (i.e., the axis aggregated). The sign of the pre-root moment is preserved via sign(h_n), and numerical stability is improved by adding the module-level EPS to the absolute value before taking the n-th root. No in-place modification of the input tensor h is performed.
    
    Failure modes and notes:
        - If h is not a torch.Tensor, a TypeError will be raised by the underlying torch operations.
        - If h has fewer than two dimensions, an IndexError will occur when reducing dim=-2.
        - If n is not a positive integer, the function may raise an error or return invalid values.
        - The function relies on a module-level EPS constant for stability; if EPS is undefined in the module scope, a NameError will be raised.
        - The function performs floating-point operations; integer dtypes for h may be implicitly cast by torch operations.
    """
    from deepchem.utils.graph_utils import aggregate_moment
    return aggregate_moment(h, n, **kwargs)


################################################################################
# Source: deepchem.utils.graph_utils.aggregate_std
# File: deepchem/utils/graph_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def deepchem_utils_graph_utils_aggregate_std(h: torch.Tensor, **kwargs):
    """deepchem.utils.graph_utils.aggregate_std computes the standard deviation of the input tensor along its second-to-last dimension and returns that per-position measure of variability. This function is provided in DeepChem's graph utilities and is typically used in graph neural network feature aggregation (for example, computing per-feature variability across nodes in a molecular graph) in DeepChem workflows for drug discovery, materials science, quantum chemistry, and biology.
    
    Args:
        h (torch.Tensor): Input tensor from which standard deviation is computed. The routine treats the second-to-last axis (axis -2) as the aggregation axis: for every fixed index of the remaining axes, the standard deviation is computed over the elements along that second-to-last axis. In practice within DeepChem graph utilities, h commonly represents batched node or edge feature arrays (for example, a tensor shaped like [batch, nodes, features] where the nodes axis is reduced), but the function accepts any torch.Tensor with at least two dimensions. The tensor should contain numeric values; floating-point dtypes are typical to avoid unintended integer truncation during variance and square-root computations.
        kwargs (dict): Additional keyword arguments accepted for API compatibility with other aggregation functions. This function does not consume or modify these keyword arguments; they are accepted and ignored. Passing keywords has no side effects on the function's computation.
    
    Returns:
        torch.Tensor: A tensor containing the standard deviation computed along the second-to-last dimension of the input h. Concretely, for every fixed index of all axes except the second-to-last one, the returned tensor holds the square root of the variance across the second-to-last axis. Implementation detail: the function computes aggregate_var(h) to obtain variance and then returns torch.sqrt(variance + EPS), where EPS is a small positive constant defined in the module to ensure numerical stability and to avoid taking the square root of exact zero or tiny negative rounding errors. The returned tensor therefore has the aggregated axis removed (it holds one value per position in the remaining axes). No in-place modification of h occurs.
    
    Behavior, defaults, and failure modes:
        This function is deterministic and has no side effects other than returning a new tensor. It relies on aggregate_var(h) to compute variance; aggregate_var is expected to reduce along the same second-to-last axis. If h has fewer than two dimensions, attempting to aggregate along axis -2 will raise an indexing or shape-related exception. If h contains NaNs or infinities, those will propagate through the variance and square-root operations and may produce NaNs or infinities in the output. The EPS addition ensures small negative rounding errors in the variance do not yield NaNs when taking the square root, but it does not sanitize explicit NaNs or infinities in the input.
    """
    from deepchem.utils.graph_utils import aggregate_std
    return aggregate_std(h, **kwargs)


################################################################################
# Source: deepchem.utils.graph_utils.aggregate_sum
# File: deepchem/utils/graph_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def deepchem_utils_graph_utils_aggregate_sum(h: torch.Tensor, **kwargs):
    """Compute the sum of the input tensor along the second-to-last dimension.
    
    This function is a small utility used by DeepChem graph utilities to perform a common aggregation operation in graph neural networks: summing feature vectors across a neighborhood axis (for example, summing messages from neighbors for each node). In the context of DeepChem's application domains (drug discovery, materials science, quantum chemistry, and biology), this operation is typically used to aggregate per-edge or per-neighbor feature tensors into per-node feature representations during message-passing or readout steps. The implementation delegates to torch.sum with dim=-2 and does not modify the input tensor in place.
    
    Args:
        h (torch.Tensor): Input tensor to be reduced. This tensor is expected to contain a neighborhood or grouping dimension in the second-to-last position (index -2). For example, if h has shape (..., N, F) where N indexes neighbors and F is the feature dimension, this function computes the sum over N and returns a tensor of shape (..., F). The dtype and device of the returned tensor follow torch.sum semantics and are typically preserved from h. If h has fewer than two dimensions, torch.sum with dim=-2 will raise an IndexError.
        kwargs (dict): Additional keyword arguments accepted for API compatibility with other aggregation functions. This implementation ignores all entries in kwargs (they have no effect). kwargs exist so callers can pass through optional parameters without changing call sites; do not rely on any keyword argument to alter behavior of this function.
    
    Returns:
        torch.Tensor: A new tensor containing the elementwise sum of h along its second-to-last dimension (dim=-2). The returned tensor does not share storage with the input in a way that modifies h in place. The shape is equal to h.shape with the -2 dimension removed (for example, input shape (..., N, F) -> output shape (..., F)). Behavior, error conditions, and dtype/device handling are those of torch.sum; in particular, an IndexError is raised if h has fewer than two dimensions.
    """
    from deepchem.utils.graph_utils import aggregate_sum
    return aggregate_sum(h, **kwargs)


################################################################################
# Source: deepchem.utils.graph_utils.aggregate_var
# File: deepchem/utils/graph_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def deepchem_utils_graph_utils_aggregate_var(h: torch.Tensor, **kwargs):
    """Compute the variance of the input tensor along its second-to-last dimension.
    
    This function is used in DeepChem's graph utilities (deepchem.utils.graph_utils) to produce a per-feature variance aggregated over an axis that commonly represents neighbors or messages in graph neural network pipelines (for example, computing the variance of node features across neighboring nodes or the variance of message vectors across incoming edges). The implementation computes E[x^2] - (E[x])^2 along dim=-2 and applies a ReLU clamp to guard against small negative values that can arise from floating-point rounding. The operation is implemented with standard PyTorch ops (torch.mean, elementwise multiplication, torch.relu) and preserves autograd flow (except for ReLU's subgradient behavior at zero).
    
    Args:
        h (torch.Tensor): Input tensor containing feature vectors, activations, or other numeric values from which variance should be aggregated. The function computes the variance across the second-to-last dimension (dim=-2), so the tensor must have at least two dimensions; in common graph-use shapes this corresponds to an input shaped like (..., num_items, feature_dim), and the returned tensor will have the same shape with the second-to-last dimension removed (i.e., the variance per feature dimension). If h contains NaNs those will propagate into the result. The operation performs no in-place modifications of h.
        kwargs (dict): Additional keyword arguments accepted for API compatibility with other aggregation functions. Any entries in kwargs are ignored by this function and have no effect or side effects. Keeping this argument allows callers to pass through optional parameters without changing call sites.
    
    Returns:
        torch.Tensor: A tensor of variances computed along the second-to-last dimension of h. The returned tensor has the same number of dimensions as h minus one (the -2 axis is reduced), and its elements are computed as relu(mean(h*h, dim=-2) - mean(h, dim=-2)**2) to ensure non-negative results in the presence of floating-point error. No external state is modified.
    
    Failure modes and notes:
        - If h is not a torch.Tensor, PyTorch operations will raise a TypeError.
        - If h has fewer than two dimensions, indexing dim=-2 will raise an IndexError.
        - NaNs or infinities in h will produce NaN or inf in the output.
        - The ReLU clamp ensures the result is non-negative but introduces a non-differentiable point at zero (standard ReLU subgradient behavior).
    """
    from deepchem.utils.graph_utils import aggregate_var
    return aggregate_var(h, **kwargs)


################################################################################
# Source: deepchem.utils.graph_utils.fourier_encode_dist
# File: deepchem/utils/graph_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def deepchem_utils_graph_utils_fourier_encode_dist(
    x: torch.Tensor,
    num_encodings: int = 4,
    include_self: bool = True
):
    """deepchem.utils.graph_utils.fourier_encode_dist performs a Fourier-style encoding of a tensor of scalar values (typically inter-node or inter-atomic distances used in graph-based molecular models) by applying sinusoidal basis functions at multiple scales. The function divides the input values by powers-of-two scales (2**i for i in range(num_encodings)), computes sine and cosine for each scaled value, and concatenates these trigonometric features. Optionally, the original input values can be appended to the encoded features to preserve the absolute magnitude information. This encoding is commonly used in graph neural networks and other geometric deep learning models within DeepChem to provide models with multi-scale, periodic features that help represent distance-based relationships in molecules and materials.
    
    Args:
        x (torch.Tensor): Input tensor containing scalar values to encode (for example, pairwise distances or edge attributes in a molecular graph). The function uses x.device and x.dtype to allocate intermediate tensors on the same device and with the same dtype; x must therefore be a torch.Tensor. The function will internally unsqueeze x at the final axis before encoding and will squeeze singleton dimensions before returning, so the leading (batch) dimensions are preserved while the final axis is replaced by the encoding features.
        num_encodings (int): Number of Fourier encodings (scales) to apply. For each i in range(num_encodings) the function uses a scale factor 2**i and computes both sin(x / 2**i) and cos(x / 2**i). Defaults to 4. Must be a positive integer; providing zero or a negative value will result in an error during the computation (the implementation assumes at least one encoding).
        include_self (bool): If True (default), the original input values (after the temporary unsqueeze) are concatenated to the trigonometric encodings so that the output retains the raw scalar in addition to the Fourier features. If False, only the sine and cosine features are returned.
    
    Behavior, defaults, side effects, and failure modes:
        The function computes scales as 2**torch.arange(num_encodings, device=x.device, dtype=x.dtype) so scales are created on the same device and with the same dtype as x. It then divides the unsqueezed x by these scales, computes sine and cosine along the last axis, and concatenates results. If include_self is True, the (unsqueezed) input is concatenated as an additional feature channel. The function does not perform in-place modification of the caller's tensor; all operations create new tensors. The default num_encodings is 4, yielding 8 trigonometric features per input scalar (plus the original scalar if include_self is True). The function requires x to be a torch.Tensor; passing other types will raise a TypeError from PyTorch operations. Supplying num_encodings that is not a positive integer (for example 0 or negative) will lead to an error during arithmetic/broadcasting; callers should validate this argument if it may be non-positive. Because the implementation uses torch.squeeze() before returning, singleton dimensions may be removed; however, the intended effect is to preserve leading (batch) dimensions and replace the original scalar feature axis with the encoded feature axis.
    
    Returns:
        torch.Tensor: A tensor containing the Fourier-encoded features computed from x. The returned tensor is on the same device and has the same dtype as the input x. The final feature axis size equals 2 * num_encodings, plus 1 if include_self is True (to account for the appended original values). The leading batch dimensions of x are preserved while the last axis of x is replaced by this feature axis.
    """
    from deepchem.utils.graph_utils import fourier_encode_dist
    return fourier_encode_dist(x, num_encodings, include_self)


################################################################################
# Source: deepchem.utils.graph_utils.scale_amplification
# File: deepchem/utils/graph_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def deepchem_utils_graph_utils_scale_amplification(
    h: torch.Tensor,
    D: torch.Tensor,
    avg_d: dict
):
    """deepchem.utils.graph_utils.scale_amplification: Compute a degree-based amplification scaling factor and apply it to an input tensor h. This function implements the scaling used in graph neural network preprocessing or message-passing layers to normalize or amplify node feature activations according to node degree statistics computed on the training set. It is intended for use in DeepChem workflows for molecular machine learning tasks such as drug discovery, materials science, quantum chemistry, and biology where graph degree statistics are used to stabilize learning across nodes of varying connectivity.
    
    Args:
        h (torch.Tensor): Input feature tensor to be scaled. In the DeepChem graph utilities this represents node feature activations or messages produced by a graph neural network layer. The function multiplies h elementwise by the amplification factor computed from D and avg_d, and does not modify h in-place; a new tensor is returned. Because the implementation uses elementwise arithmetic, h must be shape-compatible with the computed amplification factor (for example, matching elementwise or broadcastable shapes).
        D (torch.Tensor): Degree tensor. This tensor contains node degree values (or another nonnegative integer/float measure of connectivity) for each element in h for which amplification is desired. The implementation computes np.log(D + 1) as the numerator of the amplification factor; therefore D should contain values for which log(D + 1) is defined (typical use is nonnegative degrees). Note that, in the current implementation, numpy's log (np.log) is applied to D; if D is a torch.Tensor on a CUDA device, this may raise an error or trigger an implicit conversion. Users should ensure D is compatible with numpy operations or convert to CPU/numpy before calling this function to avoid device/type mismatches.
        avg_d (dict): Dictionary containing averages computed over the training set. This dictionary must include the key "log" whose value is the scalar average of log(D + 1) computed on the training data (the function uses avg_d["log"] as the denominator). In DeepChem workflows this average is computed once from the training graphs to provide a normalization constant that stabilizes amplification across datasets. If the "log" key is missing, a KeyError will be raised; if avg_d["log"] is zero, a ZeroDivisionError or an infinite amplification factor may occur.
    
    Returns:
        torch.Tensor: The scaled input tensor. The returned tensor equals h * (np.log(D + 1) / avg_d["log"]) computed elementwise: each element of h is multiplied by the corresponding amplification factor derived from D and normalized by the training-set average stored in avg_d["log"]. The function does not perform in-place modification of h; it returns a new tensor representing the scaled activations. Failure modes include KeyError when "log" is absent in avg_d, ZeroDivisionError or infinite/NaN results when avg_d["log"] is zero or not a finite scalar, and TypeError or device-conversion errors if numpy operations are applied to torch tensors on non-CPU devices.
    """
    from deepchem.utils.graph_utils import scale_amplification
    return scale_amplification(h, D, avg_d)


################################################################################
# Source: deepchem.utils.graph_utils.scale_attenuation
# File: deepchem/utils/graph_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def deepchem_utils_graph_utils_scale_attenuation(
    h: torch.Tensor,
    D: torch.Tensor,
    avg_d: dict
):
    """deepchem.utils.graph_utils.scale_attenuation scales node feature tensors by a degree-dependent attenuation factor computed from training-set averages. This function implements the elementwise operation h * (avg_d["log"] / np.log(D + 1)), which realizes the attenuation described in the source as (log(D + 1))^-1 re-scaled by the training-set average of (log(D + 1))^-1. In DeepChem this is used to normalize attention or message-passing updates in graph neural network models for molecular and biological graphs so that nodes with different connectivity (degree) are attenuated according to statistics observed during training.
    
    Args:
        h (torch.Tensor): Input feature tensor representing node features or per-node messages produced by a graph neural network layer. In the DeepChem graph modeling context, h contains the values that will be attenuated by node degree so that high- and low-degree nodes are scaled consistently with training-set statistics.
        D (torch.Tensor): Degree tensor. Each element contains the degree (connectivity) value for the corresponding node or graph element. The function computes np.log(D + 1) from this tensor and uses it in the denominator of the attenuation factor. D is expected to contain values for which np.log(D + 1) is defined (typically non-negative degrees in molecular/graph data). Supplying values that make np.log(D + 1) equal to zero or undefined will lead to infinities or NaNs.
        avg_d (dict): Dictionary containing averages computed over the training set. This function reads the key "log" from avg_d and expects avg_d["log"] to be a numeric scalar (or a tensor/array-like broadcastable with h) that represents the training-set average of (log(D + 1))^-1. The value avg_d["log"] is used as the numerator in the re-scaling factor. If the key "log" is missing, a KeyError will be raised.
    
    Returns:
        torch.Tensor: Scaled input tensor. The returned tensor is the elementwise product h * (avg_d["log"] / np.log(D + 1)). There are no in-place modifications to the inputs performed by this function; it returns a new tensor (or a tensor expression) representing the scaled features. Possible failure modes and behaviors: if avg_d does not contain the "log" key a KeyError is raised; if D contains values that make np.log(D + 1) equal to zero (for example D == 0) or negative/invalid arguments to log, the result will contain infinities or NaNs and runtime warnings or errors may occur; using np.log on a torch.Tensor that is not CPU-backed or not directly convertible to a NumPy array may raise a TypeError or trigger an implicit device/array conversion error. The caller should ensure D contains valid degree values and that avg_d["log"] is provided and numerically stable for the intended DeepChem graph-learning workflow.
    """
    from deepchem.utils.graph_utils import scale_attenuation
    return scale_attenuation(h, D, avg_d)


################################################################################
# Source: deepchem.utils.graph_utils.scale_identity
# File: deepchem/utils/graph_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def deepchem_utils_graph_utils_scale_identity(
    h: torch.Tensor,
    D: torch.Tensor = None,
    avg_d: dict = None
):
    """deepchem.utils.graph_utils.scale_identity returns the input tensor unchanged and is intended as the "no-op" scaling function used in DeepChem graph utilities when node- or graph-level features should not be rescaled. In the DeepChem context (molecular machine learning, graph neural networks for drug discovery, materials science, and biology), this function serves as a placeholder or default scaling strategy in preprocessing or in model forward passes where other scaling functions (which may use node degree information or dataset averages) would normally be applied.
    
    Args:
        h (torch.Tensor): Input tensor containing features to be passed through the scaling step. In graph-based models this is typically a tensor of node features or graph-level features produced by featurization layers. This function returns this same tensor object without modification (no copy), so in-place changes to the returned tensor will also affect the original tensor object.
        D (torch.Tensor, optional): Degree tensor. In DeepChem graph utilities this typically represents per-node degree values used by alternate scaling routines to normalize features by node degree. This parameter is accepted for API compatibility with other scaling functions but is ignored by scale_identity. Default: None.
        avg_d (dict, optional): Dictionary containing averages computed over the training set (for example, average degree statistics or other dataset-level summary statistics). Other scaling implementations may use values from this dictionary to normalize features; scale_identity accepts the parameter for compatibility but does not use it. Default: None.
    
    Returns:
        torch.Tensor: The same tensor object passed as h, returned unchanged and not copied. The returned tensor is intended to be used in downstream preprocessing or model code as the "scaled" feature tensor; because no clone is made, callers should be aware that in-place modifications to the returned tensor will modify the original h. There are no side effects performed by this function aside from returning the reference to h.
    
    Failure modes and notes:
        This function assumes callers supply a torch.Tensor for h as per the signature; behavior is not defined for other types. Supplying D or avg_d has no effect on the output but is allowed for API compatibility with other scaling functions in DeepChem.
    """
    from deepchem.utils.graph_utils import scale_identity
    return scale_identity(h, D, avg_d)


################################################################################
# Source: deepchem.utils.hash_utils.hash_ecfp
# File: deepchem/utils/hash_utils.py
# Category: valid
################################################################################

def deepchem_utils_hash_utils_hash_ecfp(ecfp: str, size: int = 1024):
    """deepchem.utils.hash_utils.hash_ecfp returns a deterministic integer index in the range [0, size) computed from an input ECFP fragment string; it is used by DeepChem to fold arbitrary-length ECFP fragment identifiers into fixed-size integer indices for constructing ECFP-based fingerprint vectors used in molecular machine learning and chemoinformatics pipelines.
    
    This function encodes the input string using UTF-8, computes an MD5 digest, converts that digest to a base-16 integer, and reduces it modulo the provided size to produce a stable, platform-independent index. It is intended to map ECFP fragment identifiers (typically produced by RDKit or similar cheminformatics tools) into bit positions of a fixed-length fingerprint array so that downstream models in DeepChem can consume consistent, bounded-length feature vectors.
    
    Args:
        ecfp (str): The input string to hash. In practice this is usually an ECFP fragment identifier (for example, a string representation of a circular substructure produced when computing Extended-Connectivity Fingerprints with RDKit). This parameter must be a Python string; the function calls ecfp.encode('utf-8') internally so passing non-string types will raise an exception (for example, AttributeError if a bytes object is provided or a TypeError for other incompatible types). The practical role of this parameter is to supply the fragment identity whose presence will be assigned to a fingerprint index.
        size (int): The number of distinct hash bins (the length of the fingerprint vector into which the fragment is folded). Default is 1024. This must be a positive integer; if size is zero a ZeroDivisionError will be raised when computing the modulo, and non-integer types may produce a TypeError. In domain terms, size determines the dimensionality of the folded fingerprint used by DeepChem models: larger sizes reduce collisions at the cost of higher memory and model input dimensionality.
    
    Behavior, side effects, defaults, and failure modes:
        The function is pure and has no side effects beyond local computation; it does not modify inputs or global state. It deterministically maps the same ecfp string to the same integer index across runs and platforms as long as the implementation (UTF-8 encoding and MD5) remains unchanged. Because the output space is limited to size bins, distinct ECFP fragments can collide (map to the same integer); collisions are expected and are a known trade-off when folding high-cardinality fragment sets into a fixed-length fingerprint. The function uses Python's hashlib.md5 and the MD5 hexadecimal digest is converted to an integer with base 16 before applying modulo size. The default size is 1024 to provide a commonly used fingerprint length in cheminformatics, but users may change it to tune the collision rate and fingerprint dimensionality. Errors will occur if ecfp is not a str (encoding call fails) or if size is not a positive integer (ZeroDivisionError for size <= 0 or TypeError for incompatible types).
    
    Returns:
        int: An integer ecfp_hash in the range [0, size) that represents the hashed position for the provided ECFP fragment string. This returned value is intended to be used as an index or bin for setting a bit or incrementing a count in a fixed-length fingerprint vector consumed by DeepChem models and downstream chemoinformatics workflows.
    """
    from deepchem.utils.hash_utils import hash_ecfp
    return hash_ecfp(ecfp, size)


################################################################################
# Source: deepchem.utils.hash_utils.hash_ecfp_pair
# File: deepchem/utils/hash_utils.py
# Category: valid
################################################################################

def deepchem_utils_hash_utils_hash_ecfp_pair(ecfp_pair: Tuple[str, str], size: int = 1024):
    """Compute a deterministic integer hash in the range [0, size) that represents a pair of ECFP (Extended-Connectivity Fingerprint) fragment strings.
    
    This function is used by DeepChem spatial contact featurizers to map a pair of fragment identifiers (for example, a protein fragment and a ligand fragment that are in close spatial contact) to a single integer index. The resulting index can be used as an entry in a fixed-length fingerprint vector or as a bucket identifier in featurization pipelines. The function constructs a single string from the two fragments using the comma separator "%s,%s", encodes it with UTF-8, computes the MD5 digest, interprets the digest as a base-16 integer, and reduces it modulo size to produce the final integer. The default size of 1024 corresponds to a common fingerprint length used in DeepChem examples and featurizers.
    
    Args:
        ecfp_pair (Tuple[str, str]): Pair of ECFP fragment strings. The first element is typically the protein (or receptor) fragment identifier and the second is typically the ligand (or small-molecule) fragment identifier. Both elements must be Python str objects; the function expects exactly two string fragments and will produce a hash representing their ordered pair (order matters: (A, B) hashes differently than (B, A)). The concatenation is performed as "%s,%s" so any commas present in the fragment strings are preserved and affect the hash.
        size (int): Total number of hash buckets (the modulus). The function returns an integer in the half-open interval [0, size). Defaults to 1024. size must be a positive integer; the canonical use is to set size equal to the length of a fixed-size fingerprint vector used by downstream models or featurizers.
    
    Returns:
        int: A non-negative integer strictly less than size that deterministically represents the given ecfp_pair. This integer is suitable for use as an index into a fingerprint or as a bucket identifier in featurization. The mapping is deterministic (same input yields same output across runs) but not collision-free: different ecfp_pair inputs can map to the same output value due to the modulo reduction and finite size, so downstream code should be designed to tolerate or account for collisions.
    
    Raises:
        IndexError: If ecfp_pair does not contain two elements (for example, a sequence of incorrect length), indexing ecfp_pair[0] or ecfp_pair[1] will raise IndexError.
        TypeError: If ecfp_pair is not an indexable sequence of strings (for example, not a tuple or not containing str elements as documented), operations that assume two strings may raise TypeError.
        ZeroDivisionError: If size is zero, the modulo operation will raise ZeroDivisionError. Passing non-positive or otherwise invalid size values may produce undefined or unintended results; callers should ensure size is a positive integer.
    
    Behavior and side effects:
        The function has no external side effects (it does not modify inputs, global state, or files). It encodes the concatenated fragment pair as UTF-8 and uses the MD5 hashing algorithm to produce a hexadecimal digest; that digest is converted to an integer with base 16 and reduced modulo size. The function is deterministic across platforms and runs given identical inputs and Python/encoding behavior. Because MD5 and modulo reduction are used, the function provides a compact, reproducible mapping suitable for featurization but not suitable where collision-free or cryptographically secure hashing is required.
    """
    from deepchem.utils.hash_utils import hash_ecfp_pair
    return hash_ecfp_pair(ecfp_pair, size)


################################################################################
# Source: deepchem.utils.misc_utils.indent
# File: deepchem/utils/misc_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def deepchem_utils_misc_utils_indent(s: str, nspace: int):
    """deepchem.utils.misc_utils.indent indents every line after the first by a fixed number of spaces to produce a multi-line string suitable for readable string representations of DeepChem objects.
    
    This function is used to format the string representation of objects in DeepChem (for example LinearOperator and other composite objects) where the human-readable repr or str contains multiple lines. It preserves the first line exactly and prefixes the second and subsequent lines with a block of spaces so that nested or multi-line contents align visually in logs, model summaries, and debugging output.
    
    Args:
        s (str): The input string to be indented. In DeepChem usage this is typically the result of calling str() or repr() on an object whose textual representation can contain newline characters. The function splits this string on the newline character ("\n"), so any existing newline characters determine the line boundaries that will be considered for indentation. If a non-str value is passed at runtime, a TypeError will be raised when string methods are invoked.
        nspace (int): The number of space characters to prefix to every line after the first. This integer controls the visual indentation width used when rendering multi-line object representations in DeepChem (for example to align nested components of a LinearOperator). If nspace is zero, subsequent lines are left unchanged relative to the first line. If nspace is negative, Python string repetition yields an empty string and therefore no indentation is applied; passing a non-int at runtime will raise a TypeError.
    
    Returns:
        str: A new string where the first line of the input s is unchanged and each subsequent line is prefixed with exactly nspace space characters. The function does not modify the original input string object but returns a newly constructed string. Note that because the implementation splits on "\n" and then rejoins with "\n", a trailing newline present in the input may not be preserved exactly in the output; callers that rely on preserving a terminal newline should handle that explicitly. The operation runs in linear time with respect to the length of s and has no other side effects.
    """
    from deepchem.utils.misc_utils import indent
    return indent(s, nspace)


################################################################################
# Source: deepchem.utils.misc_utils.shape2str
# File: deepchem/utils/misc_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def deepchem_utils_misc_utils_shape2str(shape: tuple):
    """deepchem.utils.misc_utils.shape2str converts a shape tuple into a concise, human-readable string. This function is intended for use in DeepChem logging, error messages, and user-facing text where a compact, readable representation of array/tensor shapes (for example NumPy arrays or model tensors) is required.
    
    Args:
        shape (tuple): A tuple of integers describing the dimensions of an array or tensor used in DeepChem (for example a NumPy array shape or a tensor shape passed between model layers). Each element represents the length of the corresponding axis. The function iterates over this tuple and converts each element to its string form; elements are expected to be integer-like but any object with a sensible str() will be converted. Typical inputs come from array.shape or tensor.shape. Passing a non-iterable for shape will raise a TypeError during iteration. An empty tuple yields the string "()". A single-element tuple such as (3,) will be rendered as "(3)" (note: this function does not include the Python tuple trailing comma in the string representation).
    
    Returns:
        str: A parenthesized, comma-separated string representation of the input shape, suitable for display in logs and messages in DeepChem (for example "(batch_size, features)" or "(10, 64, 64, 3)"). There are no side effects. If the input is not iterable, a TypeError will be raised when attempting to iterate; if elements cannot be converted to string, their str() representations will be used and included in the returned value.
    """
    from deepchem.utils.misc_utils import shape2str
    return shape2str(shape)


################################################################################
# Source: deepchem.utils.noncovalent_utils.compute_hbonds_in_range
# File: deepchem/utils/noncovalent_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def deepchem_utils_noncovalent_utils_compute_hbonds_in_range(
    frag1: tuple,
    frag2: tuple,
    pairwise_distances: numpy.ndarray,
    hbond_dist_bin: tuple,
    hbond_angle_cutoff: list
):
    """Compute all atom-index pairs between two molecular fragments that satisfy
    a hydrogen-bond distance bin and angle cutoff.
    
    This function is part of DeepChem's noncovalent interaction utilities used in
    drug discovery and molecular modeling workflows (see DeepChem README). It
    examines a precomputed matrix of pairwise inter-atomic distances between two
    fragments and returns those atom-index pairs that both fall inside an exclusive
    distance window (hbond_dist_bin) and pass an angular test implemented by the
    helper is_hydrogen_bond. The function treats input fragments as tuples
    (containing coordinates and an RDKit molecule object or a MolecularFragment
    object) and is typically used when featurizing or scoring noncovalent
    interactions (hydrogen bonds) between ligand and receptor fragments for
    machine learning or analysis pipelines.
    
    Args:
        frag1 (tuple): Tuple describing the first fragment. In DeepChem this is
            expected to contain at least coordinates (array-like) and an RDKit
            mol or MolecularFragment instance. The coordinates correspond to the
            axis-0 size N that must match the first dimension of pairwise_distances.
            The fragment provides atom identity and geometry information used by
            is_hydrogen_bond to test angular geometry for candidate contacts.
        frag2 (tuple): Tuple describing the second fragment. Analogous to frag1,
            this should contain coordinates and an RDKit mol or MolecularFragment
            instance. The coordinates correspond to the axis-1 size M that must
            match the second dimension of pairwise_distances.
        pairwise_distances (numpy.ndarray): Numeric 2-D array of shape (N, M)
            giving inter-atomic distances (in the same units used across the
            fragments, typically angstroms) between each atom in frag1 (N atoms)
            and each atom in frag2 (M atoms). The function performs elementwise
            comparisons against hbond_dist_bin and iterates candidate index pairs.
            If this array has an incompatible shape relative to frag1/frag2
            coordinates, or is not a numpy.ndarray, comparisons or indexing may
            raise exceptions.
        hbond_dist_bin (tuple): Tuple of two floats (min_dist, max_dist)
            specifying an exclusive distance window in angstroms for candidate
            hydrogen-bonding atom pairs. The code selects pairs satisfying
            (pairwise_distances > min_dist) and (pairwise_distances < max_dist),
            i.e., both bounds are exclusive. These bounds are applied before the
            angular test to reduce the number of angle evaluations.
        hbond_angle_cutoff (list): List of floats that define the angular
            tolerance(s) used by is_hydrogen_bond to decide whether a candidate
            distance-contact qualifies as a hydrogen bond. The exact interpretation
            of these values (for example, allowed deviations from ideal geometry)
            is delegated to is_hydrogen_bond; this function forwards the list
            unchanged. Providing an invalid list may cause is_hydrogen_bond to
            raise an exception.
    
    Returns:
        list of tuple: A list of 2-tuples (i, j) where i is a zero-based atom
        index into frag1 and j is a zero-based atom index into frag2. Each
        returned pair satisfies the exclusive distance criteria defined by
        hbond_dist_bin and also passed the angular test implemented by
        is_hydrogen_bond(frag1, frag2, (i, j), hbond_angle_cutoff). The list
        contains no side effects on the input fragments or the pairwise_distances
        array. If no contacts meet both distance and angle criteria, an empty
        list is returned.
    
    Failure modes and behavior notes:
        - The function performs elementwise numeric comparisons on pairwise_distances
          using the exclusive operators ">" and "<". Very small numerical errors in
          pairwise_distances near the bin edges can affect inclusion.
        - If pairwise_distances is not a numpy.ndarray or has incompatible
          dimensions relative to frag1/frag2 coordinates, NumPy operations or the
          downstream is_hydrogen_bond call may raise TypeError or IndexError.
        - is_hydrogen_bond is invoked for each candidate distance pair; any
          exceptions raised by that helper (for example due to missing molecular
          metadata in frag tuples) will propagate to the caller.
        - There are no side effects: input tuples and the pairwise_distances array
          are not modified. The function makes no assumptions about device (CPU/GPU)
          and uses standard NumPy operations consistent with DeepChem's CPU-based
          utilities.
    """
    from deepchem.utils.noncovalent_utils import compute_hbonds_in_range
    return compute_hbonds_in_range(
        frag1,
        frag2,
        pairwise_distances,
        hbond_dist_bin,
        hbond_angle_cutoff
    )


################################################################################
# Source: deepchem.utils.noncovalent_utils.compute_hydrogen_bonds
# File: deepchem/utils/noncovalent_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def deepchem_utils_noncovalent_utils_compute_hydrogen_bonds(
    frag1: tuple,
    frag2: tuple,
    pairwise_distances: numpy.ndarray,
    hbond_dist_bins: list,
    hbond_angle_cutoffs: list
):
    """Computes hydrogen bonds between two molecular fragments (typically a protein and a ligand) and groups the detected contacts by distance bin. This function is used in DeepChem molecular featurization and interaction analysis to identify candidate hydrogen-bonding atom pairs for downstream tasks in drug discovery, molecular modeling, and structure-based feature engineering. The routine delegates per-bin detection to compute_hbonds_in_range and returns a list where each element corresponds to the hydrogen-bond contacts found for a single distance range (bin).
    
    Args:
        frag1 (tuple): A tuple (coords, mol_fragment) representing the first molecular fragment. In DeepChem usage this is typically the protein fragment but may be any molecular fragment; coords is the array or sequence of 3D coordinates whose ordering defines atom indices used in the returned contact tuples, and mol_fragment is an RDKit Mol or MolecularFragment object that provides chemical context for angle and donor/acceptor detection.
        frag2 (tuple): A tuple (coords, mol_fragment) representing the second molecular fragment. In DeepChem usage this is typically the ligand fragment but may be any molecular fragment; coords is the array or sequence of 3D coordinates whose ordering defines atom indices used in the returned contact tuples, and mol_fragment is an RDKit Mol or MolecularFragment object used for chemical context.
        pairwise_distances (numpy.ndarray): A 2-D numpy array of shape (N, M) containing precomputed pairwise distances between points in frag1.coords (N rows) and frag2.coords (M columns). This matrix is used to quickly select candidate donor–acceptor pairs that fall into the supplied distance bins. The function assumes the shape is consistent with the coordinate arrays contained in frag1 and frag2.
        hbond_dist_bins (list[tuple]): A list of tuples, each tuple specifying an inclusive distance range (min_dist, max_dist) in the same length units as pairwise_distances. Each tuple defines one bin; the function iterates over these bins and collects hydrogen-bond contacts whose donor–acceptor distances fall within the range. The output list has the same length and ordering as this list of bins.
        hbond_angle_cutoffs (list[float]): A list of floating-point angle cutoff values (in degrees or radians consistent with how compute_hbonds_in_range expects them) that specify the maximum allowed angular deviation for a hydrogen bond in the corresponding distance bin. The i-th cutoff is applied to the i-th distance bin. The lengths of hbond_angle_cutoffs and hbond_dist_bins must match.
    
    Returns:
        list: A list of hydrogen-bond contact lists. The returned object is a list with length equal to len(hbond_dist_bins). Each element is itself a list (possibly empty) of 2-tuples (protein_index_i, ligand_index_j) where protein_index_i is an integer index into frag1.coords and ligand_index_j is an integer index into frag2.coords. These tuples identify atom pairs that satisfy both the distance range and angular cutoff criteria for that bin. The ordering of contacts within each sublist is determined by compute_hbonds_in_range. This return value is intended for use in featurization, contact analysis, or further aggregation in DeepChem workflows.
    
    Behavior, side effects, and failure modes:
        The function is pure (no external side effects) and deterministic given identical inputs; it constructs and returns the grouped contact lists without modifying frag1, frag2, or pairwise_distances. It calls compute_hbonds_in_range once per distance bin, passing the corresponding angle cutoff; therefore the output reflects the behavior and chemical heuristics implemented by compute_hbonds_in_range. Common failure modes include:
        - IndexError if len(hbond_angle_cutoffs) < len(hbond_dist_bins) because the function indexes hbond_angle_cutoffs by the bin loop index.
        - ValueError or incorrect results if pairwise_distances.shape is not (N, M) matching the lengths of frag1.coords and frag2.coords or if coords ordering does not match the intended atom indexing.
        - TypeError if inputs do not match the expected types (frag1/frag2 not tuples, pairwise_distances not a numpy.ndarray, or bin/cutoff lists not iterable of the documented element types).
        The function does not validate units or coordinate systems; callers must ensure distance units and angle units (degrees vs radians) are consistent with compute_hbonds_in_range and with each other.
    """
    from deepchem.utils.noncovalent_utils import compute_hydrogen_bonds
    return compute_hydrogen_bonds(
        frag1,
        frag2,
        pairwise_distances,
        hbond_dist_bins,
        hbond_angle_cutoffs
    )


################################################################################
# Source: deepchem.utils.noncovalent_utils.is_cation_pi
# File: deepchem/utils/noncovalent_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def deepchem_utils_noncovalent_utils_is_cation_pi(
    cation_position: numpy.ndarray,
    ring_center: numpy.ndarray,
    ring_normal: numpy.ndarray,
    dist_cutoff: float = 6.5,
    angle_cutoff: float = 30.0
):
    """Determine whether a cation and an aromatic ring form a cation–π contact using geometric criteria (distance and angular alignment). This function is used in DeepChem's noncovalent interaction utilities to detect cation–π interactions commonly important in protein–ligand recognition, molecular binding, and drug discovery workflows.
    
    Args:
        cation_position (numpy.ndarray): Cartesian coordinates of the cation center (e.g., the position of an ammonium nitrogen, metal ion, or other positively charged atom) in the same coordinate frame and units as ring_center. This vector provides the spatial location of the cation used to compute the vector from the aromatic ring center to the cation and the Euclidean distance between them; typical units are Angstroms when used with molecular structures in DeepChem.
        ring_center (numpy.ndarray): Cartesian coordinates of the aromatic ring center. This value can be computed with DeepChem's compute_ring_center helper and represents the geometric center of the ring; it is used as the origin for the cation-to-ring vector whose length is compared to dist_cutoff.
        ring_normal (numpy.ndarray): A vector normal to the aromatic ring plane (the ring plane's perpendicular direction). This can be obtained from DeepChem's compute_ring_normal helper. The function uses the angle between this vector and the vector from ring_center to cation_position to determine whether the cation lies approximately perpendicular to the ring plane (above or below the ring). The vector does not strictly need to be unit-normalized, but it must be non-zero and expressed in the same coordinate frame as cation_position and ring_center.
        dist_cutoff (float): Distance cutoff in Angstroms (default 6.5). This scalar sets the maximum allowed Euclidean distance between ring_center and cation_position for the pair to be considered in contact. In the DeepChem noncovalent context, typical cation–π contacts are considered when the cation is within this spatial threshold of the ring center.
        angle_cutoff (float): Angular cutoff in degrees (default 30.0). This scalar is the maximum allowed deviation from perfect alignment (0°) between the ring normal and the vector pointing from ring_center to cation_position. The function treats cations located on either side of the ring plane as aligned if the angle is less than angle_cutoff or greater than 180 - angle_cutoff (i.e., within angle_cutoff degrees of 0° or 180°), indicating the cation lies approximately perpendicular to the ring plane.
    
    Returns:
        bool: True if the cation–ring pair meets both geometric criteria for a cation–π contact: the Euclidean distance from ring_center to cation_position is strictly less than dist_cutoff (in Angstroms) and the angle between the ring_normal and the vector from ring_center to cation_position (measured in degrees) is within angle_cutoff degrees of 0° or 180° (i.e., (angle < angle_cutoff or angle > 180.0 - angle_cutoff)). Returns False otherwise.
    
    Notes and failure modes:
        - The function performs no in-place modification of inputs (no side effects).
        - Inputs are expected to be finite numeric numpy.ndarray objects representing 3D Cartesian coordinates and vectors in a consistent coordinate frame and unit system. Mismatched shapes that cannot be broadcast together, non-finite values (NaN or inf), or a zero-length ring_normal may lead to runtime errors (e.g., from numpy.linalg.norm or the angle computation) or undefined behavior. Callers should validate or pre-process coordinates (e.g., ensure ring_normal is non-zero and inputs are finite) before invoking this function.
        - The default cutoff values (dist_cutoff=6.5 Å, angle_cutoff=30°) reflect permissive geometric criteria commonly used to flag potential cation–π interactions in structural analysis; users may tighten or relax these thresholds to match specific datasets or stricter definitions of contact.
    """
    from deepchem.utils.noncovalent_utils import is_cation_pi
    return is_cation_pi(
        cation_position,
        ring_center,
        ring_normal,
        dist_cutoff,
        angle_cutoff
    )


################################################################################
# Source: deepchem.utils.noncovalent_utils.is_hydrogen_bond
# File: deepchem/utils/noncovalent_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def deepchem_utils_noncovalent_utils_is_hydrogen_bond(
    frag1: tuple,
    frag2: tuple,
    contact: tuple,
    hbond_distance_cutoff: float = 4.0,
    hbond_angle_cutoff: float = 40.0
):
    """Determine whether a specified atom pair contact between two molecular fragments represents a hydrogen bond in the context of noncovalent interaction analysis used by DeepChem for molecular machine learning and computational chemistry tasks.
    
    This function inspects the two fragments provided as tuples of (coords, molecule) where coords are the atomic coordinates and molecule is an RDKit Mol or DeepChem MolecularFragment. It checks that the two contact atoms are potential hydrogen-bond participants (nitrogen or oxygen, atomic numbers 7 or 8), collects nearby hydrogen atom positions that appear covalently bonded to those heavy atoms (using a 1.3 Å threshold for O–H / N–H detection), and then evaluates the geometric angle formed by the hydrogen, donor, and acceptor to decide whether the geometry is consistent with a hydrogen bond. The function returns True if a qualifying hydrogen and geometry are found, otherwise False. Note that although a hbond_distance_cutoff parameter exists with a default of 4.0, the current implementation does not use that cutoff; hydrogen detection uses an internal 1.3 Å bond-length filter and angle checking is performed via is_angle_within_cutoff against the hbond_angle_cutoff.
    
    Args:
        frag1 (tuple): Tuple of (coords, rdkit mol / MolecularFragment) for the first fragment. coords is expected to be an indexable collection of 3D atomic coordinates (used as numeric vectors by numpy.linalg.norm), and the molecule is expected to provide RDKit-like atom access via GetAtoms() and atom.GetAtomicNum(). In DeepChem workflows this argument is typically a MolecularFragment paired with its conformation coordinates and is used as the source or target of the tested contact.
        frag2 (tuple): Tuple of (coords, rdkit mol / MolecularFragment) for the second fragment. Same format and role as frag1 but representing the other molecule or fragment participating in the contact. The function treats frag1 and frag2 symmetrically when identifying hydrogens and computing angles.
        contact (tuple): Tuple of indices (atom_i_index, atom_j_index) identifying the specific atom pair to test for a hydrogen bond, where atom_i_index indexes into frag1's coords and atom list and atom_j_index indexes into frag2's coords and atom list. Elements must be convertible to int because the implementation uses int(contact[0]) and int(contact[1]) to index arrays and RDKit atom lists. In DeepChem usage this contact typically comes from a distance-based contact list or a docking/contact map.
        hbond_distance_cutoff (float): Distance cutoff for considering a noncovalent contact as potentially relevant to hydrogen bonding. Default is 4.0. Practical note: in the current implementation this parameter is present for API compatibility but is not referenced; instead, covalently bonded hydrogens are detected using an internal 1.3 Å cutoff and no global interfragment distance threshold is applied by this function.
        hbond_angle_cutoff (float): Angle deviance cutoff in degrees used to decide whether the hydrogen-to-donor and hydrogen-to-acceptor vectors define an angle consistent with a hydrogen bond. Default is 40.0 (degrees). This value is passed to is_angle_within_cutoff to perform the angular test; smaller values impose a stricter linearity requirement for the hydrogen bond geometry.
    
    Returns:
        bool: True if the specified contact is classified as a hydrogen bond according to the implemented geometric tests (both atoms are N or O, a nearby covalently bonded hydrogen is found within ~1.3 Å of a donor, and the hydrogen-donor-acceptor angle is within hbond_angle_cutoff); False otherwise.
    
    Behavioral details, side effects, defaults, and failure modes:
        - The function is pure with respect to its inputs: it does not modify frag1, frag2, or contact and has no external side effects.
        - Hydrogen detection is performed by scanning both fragments for atoms with atomic number 1 (hydrogen) and selecting those whose distance to the candidate heavy atom is less than 1.3 Å. This 1.3 Å threshold is based on typical O–H and N–H covalent bond lengths and is hard-coded in the implementation.
        - The function first requires that both contact atoms be potential hydrogen-bond participants (atomic numbers 7 or 8 for N or O). If this condition is not met, the function immediately returns False.
        - The hbond_distance_cutoff parameter exists for API compatibility and future use but is not used in the current code path; callers should not rely on it affecting the result.
        - The angular test is delegated to is_angle_within_cutoff(hydrogen_to_frag2, hydrogen_to_frag1, hbond_angle_cutoff), so the behavior depends on that helper's implementation and units (degrees are expected for hbond_angle_cutoff).
        - Failure modes include IndexError if contact indices are out of range for the provided coords or atom lists; AttributeError if the molecule objects do not provide GetAtoms() or atoms do not provide GetAtomicNum(); TypeError or ValueError if coords are not numeric arrays or sequences compatible with numpy.linalg.norm; and NameError if is_angle_within_cutoff is not defined or imported in the module. Callers should ensure frag tuples follow the expected structure (coordinates compatible with numpy operations and RDKit-like molecule objects) to avoid exceptions.
        - Typical DeepChem usage: supply fragments extracted from RDKit molecules or DeepChem MolecularFragment objects along with their 3D coordinates to detect hydrogen bonds for feature extraction, contact classification, or noncovalent interaction analysis in drug discovery and molecular modeling pipelines.
    """
    from deepchem.utils.noncovalent_utils import is_hydrogen_bond
    return is_hydrogen_bond(
        frag1,
        frag2,
        contact,
        hbond_distance_cutoff,
        hbond_angle_cutoff
    )


################################################################################
# Source: deepchem.utils.noncovalent_utils.is_pi_parallel
# File: deepchem/utils/noncovalent_utils.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for is_pi_parallel because the docstring has no description for the argument 'ring1_center'
################################################################################

def deepchem_utils_noncovalent_utils_is_pi_parallel(
    ring1_center: numpy.ndarray,
    ring1_normal: numpy.ndarray,
    ring2_center: numpy.ndarray,
    ring2_normal: numpy.ndarray,
    dist_cutoff: float = 8.0,
    angle_cutoff: float = 30.0
):
    """Check whether two aromatic rings form a parallel pi–pi contact used in noncovalent interaction detection for molecular modelling and drug discovery.
    
    This function is used in DeepChem to identify parallel stacking between aromatic rings, a common noncovalent interaction relevant to ligand binding, molecular recognition, and conformational analysis. It compares the Euclidean distance between ring centers and the angle between ring normal vectors against user-provided cutoffs to decide whether the rings are in a parallel pi–pi arrangement. The ring center and normal vectors can be produced with helper utilities such as compute_ring_center and compute_ring_normal; the angle is computed via angle_between and converted from radians to degrees.
    
    Args:
        ring1_center (numpy.ndarray): Cartesian coordinates of the center of the first aromatic ring. Practical use: typically a 3-element array [x, y, z] in Angstroms computed by compute_ring_center; used as one endpoint for the center-to-center distance test.
        ring1_normal (numpy.ndarray): Normal vector of the first aromatic ring. Practical use: typically a 3-element array representing the ring plane normal (preferably normalized) computed by compute_ring_normal; used to compute the angle between ring planes.
        ring2_center (numpy.ndarray): Cartesian coordinates of the center of the second aromatic ring. Practical use: typically a 3-element array [x, y, z] in Angstroms computed by compute_ring_center; compared to ring1_center to obtain the center-to-center distance.
        ring2_normal (numpy.ndarray): Normal vector of the second aromatic ring. Practical use: typically a 3-element array representing the ring plane normal (preferably normalized) computed by compute_ring_normal; used to compute the angle between ring planes.
        dist_cutoff (float): Distance cutoff in Angstroms. Max allowed Euclidean distance between ring centers for the rings to be considered interacting. Default is 8.0. In practice this threshold filters out rings that are too far apart to engage in meaningful pi–pi stacking.
        angle_cutoff (float): Angle cutoff in degrees. Max allowed deviation from perfect parallelism (0 degrees) between ring normals. The function treats normals with an angle less than angle_cutoff or greater than 180.0 - angle_cutoff as effectively parallel (accounting for opposite normal directions); default is 30.0 degrees. In practice this allows some tolerance for non-ideal geometries typical in molecular structures.
    
    Returns:
        bool: True if the two aromatic rings are classified as forming a parallel pi–pi contact according to the following criteria: the Euclidean distance between ring centers is less than dist_cutoff and the angle between ring normals (converted to degrees) is either less than angle_cutoff or greater than 180.0 - angle_cutoff. False otherwise. This boolean result is commonly used downstream in noncovalent interaction detection, feature generation for machine learning, or structural analysis workflows.
    
    Notes on behavior and failure modes:
        The function computes distance using numpy.linalg.norm(ring1_center - ring2_center) and computes the angle via angle_between(ring1_normal, ring2_normal) converted from radians to degrees. There are no side effects. Inputs are expected to be numeric numpy.ndarray objects representing 3D vectors; providing arrays with incompatible shapes, non-numeric entries, or zero-length normal vectors may lead to numpy or angle_between exceptions (e.g., ValueError, TypeError, or runtime warnings/NaNs). The function does not perform input validation beyond what numpy and angle_between enforce, so callers should ensure inputs are well-formed and normals are non-zero (ideally normalized) to avoid undefined behavior.
    """
    from deepchem.utils.noncovalent_utils import is_pi_parallel
    return is_pi_parallel(
        ring1_center,
        ring1_normal,
        ring2_center,
        ring2_normal,
        dist_cutoff,
        angle_cutoff
    )


################################################################################
# Source: deepchem.utils.noncovalent_utils.is_pi_t
# File: deepchem/utils/noncovalent_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def deepchem_utils_noncovalent_utils_is_pi_t(
    ring1_center: numpy.ndarray,
    ring1_normal: numpy.ndarray,
    ring2_center: numpy.ndarray,
    ring2_normal: numpy.ndarray,
    dist_cutoff: float = 5.5,
    angle_cutoff: float = 30.0
):
    """Check whether two aromatic ring planes form a T-shaped (perpendicular) pi–pi contact used in noncovalent interaction analysis.
    
    This function is used in DeepChem's noncovalent interaction utilities to identify T-shaped aromatic pi–pi contacts between two rings, a common motif important for molecular recognition in drug discovery, materials science, and computational chemistry. The function compares the Euclidean distance between the two ring centers and the angle between their plane normals (converted to degrees) against the supplied cutoffs. It expects ring centers and normals to be provided (for example, computed with compute_ring_center and compute_ring_normal utilities in this module). The decision logic uses a strict angle test (angle must lie strictly between the bounds) and a strict distance test (distance must be strictly less than dist_cutoff).
    
    Args:
        ring1_center (numpy.ndarray): 3D coordinates of the center of the first aromatic ring. Represents the geometric centroid of the ring atoms in Angstroms. This array is subtracted from ring2_center to compute the inter-center Euclidean distance; therefore it must be a numeric array of shape compatible with ring2_center (commonly (3,) for x,y,z).
        ring1_normal (numpy.ndarray): Normal vector of the first ring's plane. Represents the orientation of the aromatic ring and is used to compute the angle between ring planes. Typically obtained from compute_ring_normal; must be a numeric array shaped compatibly with ring2_normal (commonly (3,)).
        ring2_center (numpy.ndarray): 3D coordinates of the center of the second aromatic ring. Represents the geometric centroid of the second ring in Angstroms. Used with ring1_center to compute the Euclidean distance between ring centers.
        ring2_normal (numpy.ndarray): Normal vector of the second ring's plane. Represents the orientation of the second aromatic ring and is used to compute the inter-plane angle in conjunction with ring1_normal.
        dist_cutoff (float): Distance cutoff in Angstroms. Default is 5.5. The function considers a T-shaped contact only if the Euclidean distance between ring centers is strictly less than this value. This cutoff reflects a typical maximum center-to-center separation for meaningful pi–pi interactions in molecular modeling.
        angle_cutoff (float): Angle tolerance in degrees around the ideal 90° perpendicular orientation. Default is 30.0. The function computes the angle between ring normals (converted from radians to degrees) and requires it to satisfy 90.0 - angle_cutoff < angle < 90.0 + angle_cutoff (strict inequalities). This parameter controls how close the rings must be to perpendicular to be classified as T-shaped.
    
    Returns:
        bool: True if the rings form a T-shaped pi–pi contact according to the criteria: the inter-center distance is strictly less than dist_cutoff (Angstroms) and the angle between ring normals is strictly within (90.0 - angle_cutoff, 90.0 + angle_cutoff) degrees. False otherwise.
    
    Behavior, defaults, and failure modes:
        The function computes dist = np.linalg.norm(ring1_center - ring2_center) and angle = angle_between(ring1_normal, ring2_normal) * 180 / np.pi, then applies the cutoffs described above. There are no side effects; the function does not modify its inputs. If the input arrays have incompatible shapes or contain non-numeric data, underlying NumPy operations may raise exceptions (for example, ValueError for shape mismatches or TypeError for invalid dtypes). The function does not perform normalization of normal vectors; callers should provide meaningful normal vectors (e.g., unit normals from compute_ring_normal) to ensure the angle computation is correct.
    """
    from deepchem.utils.noncovalent_utils import is_pi_t
    return is_pi_t(
        ring1_center,
        ring1_normal,
        ring2_center,
        ring2_normal,
        dist_cutoff,
        angle_cutoff
    )


################################################################################
# Source: deepchem.utils.periodic_table_utils.get_atom_mass
# File: deepchem/utils/periodic_table_utils.py
# Category: valid
################################################################################

def deepchem_utils_periodic_table_utils_get_atom_mass(atom_z: int):
    """Return the atomic mass for the element with the given atomic number, expressed in atomic units (electron mass units). This function is used in DeepChem's quantum-chemistry and molecular modeling utilities to supply nuclear masses in the unit system expected by certain electronic-structure and differentiable DFT code (for example, code derived from the referenced DQC implementation). It looks up a canonical atomic mass (stored in the module-level atom_masses mapping) and converts that mass from atomic mass units (unified atomic mass unit, u) to atomic units of mass by multiplying by 1822.888486209 (the conversion factor from u to electron masses).
    
    Args:
        atom_z (int): Atomic Number of the element to query. This integer is used as the key into the module-level atom_masses mapping, which contains canonical atomic masses expressed in unified atomic mass units (u). The value must correspond to an element documented in that mapping; passing an integer not present in the mapping will cause a KeyError. The parameter has no default and there are no side effects from providing a valid atom_z.
    
    Returns:
        float: Atomic mass of the element converted to atomic units (electron mass units). Concretely, this is computed as atom_masses[atom_z] * 1822.888486209. The returned float is suitable for use in DeepChem routines and external quantum-chemistry code that expect nuclear masses expressed in electron-mass atomic units.
    
    Raises:
        KeyError: If atom_z is not a key in the module-level atom_masses mapping (for example, if the atomic number is out of range or the element is not documented), a KeyError is raised with the message "Element Does Not Exists or Not Documented: <atom_z>".
    
    Notes:
        This function is pure (no external side effects) and relies on the module-level atom_masses data. The conversion factor 1822.888486209 is the number of electron masses per unified atomic mass unit and is chosen to make the returned mass compatible with code and models that use electron-mass-based atomic units, such as differentiable DFT implementations referenced in the original source.
    """
    from deepchem.utils.periodic_table_utils import get_atom_mass
    return get_atom_mass(atom_z)


################################################################################
# Source: deepchem.utils.periodic_table_utils.get_period
# File: deepchem/utils/periodic_table_utils.py
# Category: valid
################################################################################

def deepchem_utils_periodic_table_utils_get_period(atom_z: int):
    """get_period returns the chemical period (row) in the periodic table for a given atomic number.
    
    This function maps a nuclear charge (atomic number) to the corresponding period index used in periodic-table based features and analyses. In DeepChem this is used when converting elemental identity into simple periodic descriptors for tasks in drug discovery, materials science, quantum chemistry, and molecular machine learning (for example, featurizers or models that rely on period-based grouping of elements). The implementation uses fixed atomic-number cutoffs that correspond to the standard periods 1 through 7 of the modern periodic table.
    
    Args:
        atom_z (int): Atomic number (Z) of the element. This integer represents the number of protons in the nucleus and is the canonical identifier for an element in chemistry and materials science. The function expects a (positive) integer atomic number as used in DeepChem datasets and featurizers. The code maps ranges of atomic numbers to period indices; values less than or equal to 2 are mapped to period 1, values greater than 2 and less than or equal to 10 to period 2, and so on (see Returns section for exact cutoffs). Non-physical or out-of-range atomic numbers (see Failure modes) are handled as described below.
    
    Returns:
        int: The period number (1 through 7) corresponding to atom_z according to the implemented cutoffs. The exact mapping implemented by this function is:
        atom_z <= 2 -> period 1,
        atom_z <= 10 -> period 2,
        atom_z <= 18 -> period 3,
        atom_z <= 36 -> period 4,
        atom_z <= 54 -> period 5,
        atom_z <= 86 -> period 6,
        atom_z <= 118 -> period 7.
        This integer return value is commonly used as a categorical or ordinal feature in downstream DeepChem code (for grouping elements by row or for constructing periodicity-aware descriptors). There are no side effects.
    
    Behavior, defaults, and failure modes:
        The function contains no external side effects and does not modify input data. It deterministically returns the period integer based solely on numeric comparisons of atom_z against fixed cutoffs. The code will return period 1 for any atom_z value less than or equal to 2 (this includes zero or negative integers in the current implementation, but such values are not physically meaningful atomic numbers). For physically meaningful atomic numbers 1 through 118 the mapping above produces periods 1 through 7. If atom_z is greater than 118 the function raises a RuntimeError with the message "Unimplemented atomz: %d" % atom_z because elements beyond Z=118 are not supported by this implementation. This RuntimeError is the primary failure mode and signals that the requested atomic number lies outside the supported periodic table range used in DeepChem. Users should validate atomic numbers before calling this function in production code if inputs may exceed the supported range.
    
    References:
        Kasim, Muhammad F., and Sam M. Vinko. "Learning the exchange-correlation functional from nature with fully differentiable density functional theory." Physical Review Letters 127.12 (2021): 126403.
        Implementation inspiration: https://github.com/diffqc/dqc/blob/master/dqc/utils/periodictable.py
    """
    from deepchem.utils.periodic_table_utils import get_period
    return get_period(atom_z)


################################################################################
# Source: deepchem.utils.pytorch_utils.tallqr
# File: deepchem/utils/pytorch_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def deepchem_utils_pytorch_utils_tallqr(V: torch.Tensor, MV: torch.Tensor = None):
    """deepchem.utils.pytorch_utils.tallqr: Compute a QR-like decomposition for tall-and-skinny matrices using PyTorch, producing an orthogonal factor Q and an upper-triangular factor R. This function is implemented in DeepChem's PyTorch utilities and is used to orthonormalize column bases that arise in linear-algebra subroutines common in molecular machine learning, quantum chemistry, and other scientific applications supported by DeepChem. For the common case MV is None, tallqr performs a standard thin QR for a tall-and-skinny V; when MV is provided, the routine produces a Q that is orthogonal with respect to the metric defined by MV (an "M-orthogonal" basis), which is useful when orthonormality must be measured against a nontrivial inner product or preconditioner.
    
    This implementation computes V^T @ MV, forms a Cholesky factorization of the resulting small Gram matrix, and obtains R from that factorization. It then computes Q = V @ R^{-1}. The routine is optimized for the case where V has many more rows than columns ("tall and skinny") and operates on PyTorch tensors without modifying the inputs in-place.
    
    Args:
        V (torch.Tensor): V is the matrix (or batched matrices) to be decomposed. Practical role: each column of V is a candidate basis/vector to be orthogonalized; in DeepChem this often represents a small set of feature or basis vectors (e.g., nguess columns) evaluated on na points or degrees of freedom. Expected shape notation used in this project: (`*BV`, na, nguess), where `*BV` denotes optional batch dimensions associated with V. The function uses V to form the Gram-like matrix V^T @ MV and to compute the orthogonal factor Q by right-multiplying V with the inverse of R. V must be a torch.Tensor; incompatible shapes will raise a runtime error when matmul is attempted.
        MV (torch.Tensor): (`*BM`, na, nguess) where M is the basis to make Q M-orthogonal. MV serves as the second argument in the Gram-like product V^T @ MV; when MV is provided, the orthogonality of Q is measured with respect to MV (useful for M-weighted inner products or preconditioned bases). If MV is None (default=None), the function sets MV = V and computes the standard thin QR for V. MV must be a torch.Tensor with last two dimensions matching (na, nguess) and with batch dimensions `*BM` that are broadcastable with `*BV` so that torch.matmul(V.transpose(-2, -1), MV) is well-defined.
    
    Behavior, defaults, side effects, and failure modes:
        - Default behavior: if MV is None, MV is set to V and the routine computes a standard tall-and-skinny QR: the returned Q and R satisfy Q @ R ≈ V up to numerical precision in typical full-rank cases.
        - Numerical method: the routine forms VTV = V^T @ MV, computes an upper-triangular R from a Cholesky factorization of that small (nguess x nguess) Gram matrix, computes R^{-1}, and sets Q = V @ R^{-1}. This avoids performing a full (potentially expensive) QR on the tall dimension and is efficient when nguess (number of columns) is small relative to na (number of rows).
        - No in-place modification: inputs V and MV are not modified in-place; the function returns new tensors Q and R.
        - Batch handling: V and MV may include batch dimensions (`*BV`, `*BM`) which must be broadcastable for the matmul operations; the intermediate Gram matrix will have batch shape corresponding to the broadcasted `*BMV` and final R has shape (`*BMV`, nguess, nguess).
        - Failure modes: the Cholesky factorization requires the Gram matrix V^T @ MV to be (Hermitian) positive definite. If V^T @ MV is not positive definite (e.g., because columns are linearly dependent, MV is ill-conditioned, or numerical round-off causes near-singularity), torch.linalg.cholesky will raise a runtime/linalg error. In such cases R cannot be formed and inversion of R will fail; torch.inverse will raise an error or produce NaNs if R is singular or nearly singular. Users should ensure the column set and metric lead to a positive-definite Gram matrix (regularize if necessary) before calling tallqr.
        - Precision and dtype: the computation preserves PyTorch dtype and device semantics; numerical precision and stability depend on the tensor dtype (float32 vs float64) and the condition number of the Gram matrix.
    
    Returns:
        Q (torch.Tensor): The orthogonal (or M-orthogonal) factor. Shape: (`*BV`, na, nguess) in project notation: Q has the same row dimension and number of columns as V; in practice Q provides an orthonormal basis for the column space of V with respect to the inner product defined by MV (or the standard Euclidean inner product when MV is None). Q is computed as V @ R^{-1} and, when MV is None and V has full column rank, Q @ R ≈ V.
        R (torch.Tensor): The upper-triangular factor obtained from the Cholesky-based factorization of V^T @ MV. Shape: (`*BM`, nguess, nguess) where `*BM` denotes batch dimensions after broadcasting V and MV (denoted `*BMV` in the implementation). R is the small triangular matrix that, together with Q, reconstructs V in the standard case (MV is None) and encodes the Gram information when MV is provided.
    
    \"\"\"
    """
    from deepchem.utils.pytorch_utils import tallqr
    return tallqr(V, MV)


################################################################################
# Source: deepchem.utils.pytorch_utils.to_fortran_order
# File: deepchem/utils/pytorch_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def deepchem_utils_pytorch_utils_to_fortran_order(V: torch.Tensor):
    """Convert a PyTorch tensor so that its last two dimensions use Fortran (column-major) memory ordering.
    
    This utility is used in DeepChem's PyTorch integration to ensure matrices (the last two dimensions of a tensor) have column-major layout required by some linear-algebra code paths and external libraries that expect Fortran-order arrays (for example, BLAS/LAPACK-style routines). The function operates only on the final two dimensions of V and preserves the tensor's shape, dtype, and device while adjusting memory layout. It is useful when preparing 2-D matrices or batches of matrices for operations that rely on column-major memory layout in the life-sciences and molecular machine-learning workflows supported by DeepChem.
    
    Args:
        V (torch.Tensor): The input tensor whose last two dimensions are to be converted to Fortran (column-major) order. V is expected to represent a matrix or batch of matrices where the final two dimensions index matrix rows and columns. The function accesses V.transpose(-2, -1) internally, so V must have at least two dimensions; its dtype and device are preserved. If V is already laid out so that its last two dimensions are column-major (i.e., V.transpose(-2, -1).is_contiguous() is True), the original tensor object is returned unchanged. If V is contiguous in the default (row-major) layout, the function constructs a contiguous copy of the transposed last-two-dimensions and then transposes back to produce a tensor whose last two dimensions are column-major. In that case a new tensor object (with possibly different storage) is returned and the original V is not modified in-place.
    
    Returns:
        outV (torch.Tensor): A tensor with the same shape, dtype, and device as V but arranged so that the last two dimensions are in Fortran (column-major) order. If V already has column-major layout for the last two dimensions, outV is V itself. If V was row-major and convertible, outV is a new tensor (not the same Python object as V) that provides the requested memory layout. The returned tensor may be non-contiguous in PyTorch terms even though its last-two-dimension ordering is column-major.
    
    Raises:
        RuntimeError: If V does not have at least two dimensions or if neither V.is_contiguous() nor V.transpose(-2, -1).is_contiguous() holds, the function cannot produce a Fortran-ordered last-two-dimension layout and raises a RuntimeError with the message "Only the last two dimensions can be made Fortran order." This signals that the memory layout of V is incompatible with the conversion strategy used here and that the caller must supply a tensor with a compatible contiguous layout or explicitly reorder/copy data before calling.
    """
    from deepchem.utils.pytorch_utils import to_fortran_order
    return to_fortran_order(V)


################################################################################
# Source: deepchem.utils.rdkit_utils.get_contact_atom_indices
# File: deepchem/utils/rdkit_utils.py
# Category: valid
################################################################################

def deepchem_utils_rdkit_utils_get_contact_atom_indices(
    fragments: List,
    cutoff: float = 4.5
):
    """deepchem.utils.rdkit_utils.get_contact_atom_indices computes which atom indices in each fragment are within a spatial contact region relative to atoms in other fragments in a molecular complex.
    
    This function is used in DeepChem workflows (for example in molecular machine learning and drug discovery pipelines) to trim large molecular complexes by keeping only atoms that are close to inter-molecular contact regions. Trimming reduces memory and compute cost for downstream calculations (feature generation, scoring, or model inputs). The function iterates over all unordered pairs of fragments, computes pairwise distances between the coordinate arrays of the two fragments using compute_pairwise_distances, and marks any atom in either fragment as a contact atom if it lies within cutoff angstroms of any atom in the other fragment. The final output is, for each fragment, the sorted list of 0-based atom indices that should be retained for subsequent contact-based analyses.
    
    Args:
        fragments (List): As returned by rdkit_util.load_complex: a Python list whose elements are tuples of the form (coords, mol). coords is a (N_atoms, 3) array-like structure containing Cartesian coordinates in angstroms for the atoms of that fragment, and mol is the corresponding RDKit molecule object. The order of atoms in coords must match the atom ordering in mol; returned atom indices refer to this ordering. This argument provides the molecular complex to be analyzed and is required for identifying inter-fragment contacts.
        cutoff (float): The distance threshold in angstroms used to define a contact between two atoms on different fragments. The default is 4.5. An atom is considered in contact if its Euclidean distance to any atom on another fragment is strictly less than this cutoff. This parameter controls sensitivity: larger values include more atoms as contacts, smaller values are more restrictive.
    
    Returns:
        List: A Python list of length len(fragments). Each entry is a sorted list of integer atom indices (0-based) for the corresponding fragment indicating atoms that are within cutoff angstroms of at least one atom on another fragment. If a fragment has no atoms within the cutoff to any other fragment, its corresponding entry is an empty list. The returned lists do not modify the input fragments or RDKit molecule objects.
    
    Behavior, defaults, and failure modes:
        The function computes pairwise distances for every unordered fragment pair (O(n_fragments^2) pairs) and collects atom indices using NumPy nonzero on the boolean matrix pairwise_distances < cutoff. The function does not alter the input fragments or RDKit molecules; it only reads coords and returns index lists. The cutoff default of 4.5 angstroms is chosen as a common practical threshold for interatomic contact in structural biology and medicinal chemistry but can be adjusted for domain-specific needs.
        Common failure modes include TypeError or IndexError if fragments is not a list of two-element tuples or if coords elements are not array-like with shape (N_atoms, 3). If compute_pairwise_distances (used internally) raises an exception for malformed coordinate arrays or incompatible types, that exception will propagate. Performance and memory consumption grow with the number of fragments and atoms per fragment; for very large complexes consider pre-filtering or batching to avoid high memory usage.
    """
    from deepchem.utils.rdkit_utils import get_contact_atom_indices
    return get_contact_atom_indices(fragments, cutoff)


################################################################################
# Source: deepchem.utils.rdkit_utils.load_molecule
# File: deepchem/utils/rdkit_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def deepchem_utils_rdkit_utils_load_molecule(
    molecule_file: str,
    add_hydrogens: bool = True,
    calc_charges: bool = True,
    sanitize: bool = True,
    is_protein: bool = False
):
    """deepchem.utils.rdkit_utils.load_molecule converts a molecular file on disk into a pair of (xyz coordinates, RDKit Mol) suitable for use in DeepChem preprocessing and modeling pipelines in drug discovery, materials science, quantum chemistry, and biology.
    
    This function reads common molecular file formats and returns the molecule's 3D coordinates (as produced by get_xyz_from_mol) together with an rdkit.Chem.Mol object representing the same molecule. It is used by DeepChem code that requires both geometric coordinates and an RDKit molecule object for further featurization, charge assignment, and downstream modeling.
    
    Args:
        molecule_file (str): Path or filename of the input molecular file on disk. The function recognizes file types by filename suffix and supports files ending with ".mol2", ".sdf", ".pdbqt", and ".pdb". The value is used to select the RDKit reader routine (MolFromMol2File, SDMolSupplier, MolFromPDBBlock, or MolFromPDBFile). If the suffix is not recognized a ValueError is raised.
        add_hydrogens (bool): If True (default True), add hydrogens to the molecule using pdbfixer via apply_pdbfixer. This modifies or replaces the RDKit molecule to include explicit hydrogen atoms so that subsequent geometry- or atom-based calculations (for example, featurizers that expect explicit hydrogens) have the required atoms. If False, hydrogens will not be added and downstream code must handle implicit hydrogens or missing explicit H atoms.
        calc_charges (bool): If True (default True), compute partial atomic charges for the molecule using the repository's compute_charges routine. This function updates the RDKit molecule in place with charge information that many DeepChem models and features rely on. If False, no charge calculation is performed and the molecule remains without the computed charges.
        sanitize (bool): If True (default True), attempt to sanitize the RDKit molecule using Chem.SanitizeMol after reading and any optional pdbfixer corrections. Sanitization enforces valence, aromaticity, and other RDKit internal consistency checks. If sanitization fails an exception is caught and a warning is logged; the molecule is returned possibly in a non-sanitized state. If False, no explicit RDKit sanitization is attempted.
        is_protein (bool): If True (default False), indicate that the input file should be treated as a protein when running cleanup procedures (apply_pdbfixer). This flag alters hydrogenation and repair behavior applied by pdbfixer so that peptide chains and protein-specific conventions are respected. When False the molecule is treated as a small molecule and protein-specific fixes are not applied.
    
    Behavior and side effects:
        The function reads the file identified by molecule_file using RDKit readers appropriate to the filename suffix. For ".mol2" it calls Chem.MolFromMol2File(..., sanitize=False, removeHs=False). For ".sdf" it constructs an SDMolSupplier and currently selects the first molecule from the supplier (note: code contains a TODO to change this behavior). For ".pdbqt" it converts the pdbqt to a pdb block and calls Chem.MolFromPDBBlock(..., sanitize=False, removeHs=False). For ".pdb" it calls Chem.MolFromPDBFile(..., sanitize=False, removeHs=False). If reading yields a None molecule a ValueError is raised.
        If add_hydrogens or calc_charges is True, apply_pdbfixer is called with hydrogenate=add_hydrogens and is_protein=is_protein; this may return a modified RDKit molecule that includes added hydrogens and other fixes. If sanitize is True, the function calls Chem.SanitizeMol and will catch and log any exception raised during sanitization; sanitization errors do not raise but are reported via logging and the unsanitized molecule may be returned. If calc_charges is True, compute_charges is called and updates the RDKit molecule in place with computed partial charges.
        The xyz value returned is computed by get_xyz_from_mol(my_mol) from the final RDKit molecule object and therefore reflects any modifications from hydrogen addition, sanitization, or charge computation.
    
    Failure modes and requirements:
        RDKit must be installed and importable; absence of RDKit will cause an ImportError before this function runs. If the filename suffix is unrecognized the function raises ValueError("Unrecognized file type for ..."). If RDKit fails to produce a molecule object the function raises ValueError("Unable to read non None Molecule Object"). Sanitization failures are caught and logged as warnings rather than raised. The function's SDF handling currently returns only the first molecule in the supplier; multiple-molecule SDF files may not be fully supported by this implementation.
    
    Returns:
        Tuple (xyz, mol) if the input file contains a single molecule: a two-element tuple where the first element is the xyz coordinates as produced by get_xyz_from_mol for the final RDKit molecule (reflecting any hydrogenation or fixes) and the second element is the rdkit.Chem.Mol object corresponding to that molecule. The ordering is significant: xyz is first and rdkit_mol second and this ordering is relied upon by other DeepChem code.
        List[Tuple(xyz, mol)]: if the file contains multiple molecules the function may return a list of (xyz, rdkit.Chem.Mol) tuples corresponding to each molecule. Note: current implementation for ".sdf" uses SDMolSupplier and returns the first molecule only (see TODO in source), so multi-molecule handling may be limited in practice.
    
    Raises:
        ValueError: If the file suffix is not recognized or if RDKit returns no molecule object when attempting to read the file.
    """
    from deepchem.utils.rdkit_utils import load_molecule
    return load_molecule(
        molecule_file,
        add_hydrogens,
        calc_charges,
        sanitize,
        is_protein
    )


################################################################################
# Source: deepchem.utils.rdkit_utils.merge_molecules
# File: deepchem/utils/rdkit_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def deepchem_utils_rdkit_utils_merge_molecules(molecules: list):
    """Merge a list of RDKit molecule objects into a single RDKit molecule by concatenating
    their molecular graphs. This helper is used in DeepChem preprocessing and dataset
    construction for tasks in drug discovery, materials science, quantum chemistry,
    and biology (for example, building a single molecule representation from multiple
    fragments or components prior to featurization or model input).
    
    Args:
        molecules (list): List of RDKit molecule objects to merge. Each element is
            expected to be an RDKit Mol object (from rdkit.Chem) representing a
            disconnected fragment or component that should be placed into a single
            molecule record. This function does not validate chemical correctness
            beyond calling RDKit's CombineMols; callers should ensure inputs are
            valid RDKit molecule instances. The list may be empty, contain a single
            molecule, or contain multiple molecules.
    
    Returns:
        rdkit molecule or None: If molecules is empty, returns None to indicate no
        merged molecule could be produced. If molecules contains exactly one element,
        that element is returned unchanged (same object) to avoid unnecessary copying.
        If molecules contains two or more elements, returns a new RDKit Mol created
        by iteratively applying rdkit.Chem.rdmolops.CombineMols to concatenate atom
        and bond lists from each input molecule into a single Mol. The returned object
        is suitable for downstream DeepChem operations such as featurization and
        conformer generation.
    
    Behavior and side effects:
        - Uses rdkit.Chem.rdmolops.CombineMols to perform the merge; CombineMols
          appends atoms and bonds from the next molecule to the combined molecule,
          producing a new RDKit Mol. No additional bonds are created between fragments
          by this function (fragments remain disconnected in the merged Mol unless
          inputs already contained connecting bonds).
        - For an empty input list the function returns None; for a single-element
          list it returns that element directly; for multiple elements it returns a
          newly constructed RDKit Mol.
        - The function does not perform input sanitization, kekulization, or
          valence/chemistry checks beyond those performed internally by RDKit when
          operating on the returned Mol.
        - Original input molecule objects are not modified by this function; the
          combined result is a new RDKit Mol when two or more inputs are provided.
    
    Failure modes and errors:
        - If RDKit is not installed or importable, the function will raise ImportError
          when attempting to import rdkit.Chem.rdmolops.
        - If any element of molecules is not a valid RDKit Mol object, rdkit.Chem's
          CombineMols is likely to raise a TypeError or AttributeError; callers should
          pass only RDKit molecule objects.
        - The function does not check for or resolve atom index collisions, stereochemistry
          conflicts, or chemically implausible merges; such issues must be handled by
          the caller before or after merging as appropriate.
    """
    from deepchem.utils.rdkit_utils import merge_molecules
    return merge_molecules(molecules)


################################################################################
# Source: deepchem.utils.rdkit_utils.merge_molecules_xyz
# File: deepchem/utils/rdkit_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def deepchem_utils_rdkit_utils_merge_molecules_xyz(xyzs: list):
    """Merges coordinates of multiple molecules into a single contiguous coordinate matrix.
    
    This function is a small utility used by DeepChem's RDKit integration and featurization pipelines to combine per-molecule atomic coordinate arrays into one consolidated array suitable for downstream tasks such as constructing molecular graphs, batching molecules for model input, or performing geometry-based descriptors and quantum-chemistry workflows. It takes a Python list of per-molecule coordinate arrays (each describing the 3D positions of atoms in one molecule) and returns a single NumPy array containing all atom coordinates stacked along the first (row) dimension.
    
    Args:
        xyzs (list): List of numpy arrays, each array describing atomic coordinates for one molecule. Each element must be a NumPy array with shape (N_i, 3) where N_i is the number of atoms in the i-th molecule and the second dimension of size 3 represents the (x, y, z) coordinates. In practical DeepChem usage, these arrays are produced by RDKit-based molecule-to-geometry conversions or by other coordinate-extraction utilities; the function expects these per-molecule arrays to already be in units and ordering consistent with the caller's workflow.
    
    Returns:
        numpy.ndarray: A new NumPy array containing all coordinates stacked row-wise. The returned array has shape (N_total, 3), where N_total == sum_i N_i is the total number of atoms across all input molecules. The array's numeric dtype is determined by NumPy based on the input arrays. This returned array is a copy and does not modify the input arrays; it is intended for use in downstream DeepChem processing such as featurizers, batching, or geometric computations.
    
    Behavior and failure modes:
    - The implementation performs two np.vstack operations followed by np.array() to produce the final array; this results in a contiguous copy of the stacked data and may allocate memory proportional to the total number of atoms. Expect memory usage proportional to the size of the output array.
    - If xyzs is an empty list, np.vstack will raise a ValueError; callers should guard against empty input if an empty output is desired.
    - If any element of xyzs is not a 2-D array with second dimension equal to 3 (for example, wrong dimensionality or shape (N_i, 2) or (N_i, 4)), np.vstack or subsequent operations will typically raise a ValueError or produce an output that is semantically incorrect for 3D coordinates. The function does not perform explicit validation beyond relying on NumPy's stacking behavior.
    - If elements of xyzs are not NumPy arrays (for example, lists or other sequence types), NumPy will attempt to convert them during vstack; this may succeed but can produce arrays with object dtype or cause errors. For predictable behavior in DeepChem pipelines, provide NumPy arrays of shape (N_i, 3).
    - No in-place modification of input arrays is performed; the function has no side effects other than allocating and returning the stacked NumPy array.
    """
    from deepchem.utils.rdkit_utils import merge_molecules_xyz
    return merge_molecules_xyz(xyzs)


################################################################################
# Source: deepchem.utils.rdkit_utils.reduce_molecular_complex_to_contacts
# File: deepchem/utils/rdkit_utils.py
# Category: valid
################################################################################

def deepchem_utils_rdkit_utils_reduce_molecular_complex_to_contacts(
    fragments: List,
    cutoff: float = 4.5
):
    """Reduce a molecular complex to only the atoms that participate in inter-fragment contacts.
    
    This utility function deepchem.utils.rdkit_utils.reduce_molecular_complex_to_contacts is used in DeepChem preprocessing and featurization pipelines (for example, when preparing protein–ligand complexes for machine learning models in drug discovery and computational chemistry). Large molecular complexes can contain many atoms that are far from any contact interface; removing those atoms reduces memory usage and speeds up downstream computations (distance-based features, scoring functions, graph construction, etc.). The function identifies contact atoms by calling get_contact_atom_indices with the supplied cutoff (in angstroms) and then constructs a new, trimmed complex for each fragment by calling get_mol_subset. The input list order is preserved and a new list is returned; the input fragments are not modified in-place.
    
    Args:
        fragments (List): As returned by rdkit_util.load_complex: a Python list where each element is a tuple (coords, mol). coords must be a numeric array-like object with shape (N_atoms, 3) giving XYZ coordinates in angstroms for the fragment's atoms. mol must be an RDKit molecule object corresponding to those atoms. In the DeepChem context, fragments typically represent components of a molecular complex such as a protein chain, ligand, or cofactor; this argument provides both geometry and chemistry that the function uses to determine contacts.
        cutoff (float): The cutoff distance in angstroms used to determine which atoms are in contact between fragments. The default is 4.5. This floating-point value is passed to get_contact_atom_indices and is interpreted as a radial distance: two atoms from different fragments are considered to be in contact if their Euclidean distance is <= cutoff. The cutoff must be supplied in the same length units as coords (angstroms). Negative or nonsensical cutoff values are not meaningful and may produce no contacts or unpredictable results; callers should use a positive value appropriate for the type of contact they wish to capture.
    
    Returns:
        List: A new list with the same length and ordering as the input fragments list. Each element is a tuple (coords, MolecularShim) representing the reduced fragment. coords is a coordinate array trimmed to only the atoms identified as contact atoms for that fragment (conceptually shape (N_contact_atoms, 3) where N_contact_atoms is the number of atoms kept). MolecularShim is a shim object used by DeepChem in place of creating an RDKit sub-molecule directly (the repository uses this shim because constructing RDKit sub-molecules is nontrivial). This function does not mutate the original fragments list or the original RDKit molecule objects; it returns newly constructed fragment representations.
    
    Behavior and side effects:
        This function first calls get_contact_atom_indices(fragments, cutoff) to compute, for each fragment, the indices of atoms deemed to be in contact with atoms in other fragments. It then iterates over the input fragments in order and calls get_mol_subset(coords, mol, keep) for each fragment to build a reduced fragment containing only the selected atom indices. The resulting reduced fragments are collected into a new list that is returned. No files are read or written. The original fragments objects and their contained coordinate arrays are not modified by this function.
    
    Failure modes and edge cases:
        If the fragments argument does not conform to the expected structure (not a list of (coords, mol) tuples, coords not having numeric (N_atoms, 3) shape, or mol not being an RDKit molecule), underlying calls such as get_contact_atom_indices or get_mol_subset are likely to raise exceptions (TypeError, ValueError, or RDKit-specific errors). If get_contact_atom_indices returns an empty index list for a fragment, behavior depends on get_mol_subset: it may return a fragment representation with no coordinates or raise an error; callers who need a guaranteed minimum size should validate the output. The function assumes coordinates are in angstroms; passing coordinates in other units without adjusting cutoff will produce incorrect contact detection.
    """
    from deepchem.utils.rdkit_utils import reduce_molecular_complex_to_contacts
    return reduce_molecular_complex_to_contacts(fragments, cutoff)


################################################################################
# Source: deepchem.utils.sequence_utils.MSA_to_dataset
# File: deepchem/utils/sequence_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def deepchem_utils_sequence_utils_MSA_to_dataset(msa_path: str):
    """deepchem.utils.sequence_utils.MSA_to_dataset converts a multiple sequence alignment file into a DeepChem NumpyDataset suitable for downstream processing in DeepChem workflows for computational biology and life-science tasks. This function reads an MSA stored in FASTA format, extracts sequence identifiers and per-residue characters using Biopython's SeqIO, and packages the resulting sequences and ids into a deepchem.data.datasets.NumpyDataset instance so the alignment can be consumed by DeepChem featurizers, models, or dataset utilities.
    
    The function opens the file at msa_path for reading, parses records with SeqIO.parse(..., 'fasta'), appends each record.id to the ids list, and for each record iterates over the record to build a sequence as a list of residues (each residue is the element yielded by Biopython when iterating a SeqRecord, typically single-character strings). The function then constructs and returns NumpyDataset(X=sequences, ids=ids). NumpyDataset is imported inside the function implementation to avoid a circular import with DeepChem utilities.
    
    Behavior details, side effects, and failure modes: the function performs a file read and returns a new NumpyDataset; it does not write to disk or modify the input file. The file is opened using a with statement and will be closed on return or on exception. The function does not perform validation beyond FASTA parsing: it does not enforce that sequences are all the same length (though MSAs are generally expected to be aligned and of equal length), does not convert residues to numeric encodings, and does not remove or specially treat gap characters — any characters present in the FASTA sequences are preserved in the per-residue lists. If msa_path does not point to a readable file, an IOError or FileNotFoundError will be raised by the open call; if the file is not valid FASTA, Bio.SeqIO.parse may raise an exception such as ValueError or yield no records, resulting in an empty dataset. Downstream DeepChem components may expect numeric feature arrays rather than lists of characters; users should apply appropriate featurization after calling this function.
    
    Args:
        msa_path (str): Filesystem path to a multiple sequence alignment file in FASTA format. This is the path that will be opened for reading; it must be a string type and should point to a valid, readable FASTA file containing one or more sequence records. The function uses Biopython's SeqIO.parse to read records from this path and uses each record's .id as the entry identifier in the returned dataset.
    
    Returns:
        NumpyDataset: A deepchem.data.datasets.NumpyDataset instance containing the parsed alignment. The dataset is constructed with X=sequences and ids=ids where sequences is a Python list of sequences and each sequence is itself a list of residues as yielded by Biopython when iterating a SeqRecord (typically single-character strings), and ids is a list of the corresponding record.id values in the same order as the sequences. This NumpyDataset can be passed to DeepChem featurizers and models for further processing.
    """
    from deepchem.utils.sequence_utils import MSA_to_dataset
    return MSA_to_dataset(msa_path)


################################################################################
# Source: deepchem.utils.sequence_utils.hhblits
# File: deepchem/utils/sequence_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def deepchem_utils_sequence_utils_hhblits(
    dataset_path: str,
    database: str = None,
    data_dir: str = None,
    evalue: float = 0.001,
    num_iterations: int = 2,
    num_threads: int = 4
):
    """deepchem.utils.sequence_utils.hhblits runs an HH-suite sequence search (as implemented in the repository by calling the hhsearch/hhblits binaries) to build a multiple-sequence alignment (MSA) for a protein sequence or an existing MSA and returns the path to the produced .a3m file. This function is used in the DeepChem toolchain to generate profile/MSA inputs for downstream protein modeling and bioinformatics workflows (for example, preparing inputs for structure prediction or feature generation in molecular machine learning). The function requires HH-suite binaries (hhsearch or hhblits) to be installed and available on the system PATH and requires a Hidden Markov Model (HMM) reference database. If data_dir is not provided, the function reads the DEEPCHEM_DATA_DIR environment variable to locate the database. The function constructs and executes an external command (via system_call) that writes a results.a3m file into the same directory as dataset_path.
    
    Args:
        dataset_path (str): Path to an input file that is either a single protein sequence or an existing multiple-sequence alignment (MSA). The path may point to a FASTA file (commonly with extensions .fas or .fasta) or to an MSA/HMM file (commonly with extensions .a3m, .a2m, or .hmm). The function uses os.path.splitext to detect the file extension and saves output files into the same directory as dataset_path. The practical significance in DeepChem is that dataset_path is the sequence source from which the HH-suite search is started to produce an enriched MSA for downstream machine-learning or analysis pipelines.
        database (str): Name of the HH-suite HMM reference database to search against. This parameter is the database name as used by HH-suite (not an absolute filesystem path). The function combines data_dir and database to form the path passed to the HH-suite command. In DeepChem workflows this is the curated HMM database that provides sequence families against which the input sequence or MSA will be searched to build the resulting MSA/profile.
        data_dir (str): Path to the directory that contains one or more HH-suite HMM databases. If None, the function attempts to read the environment variable DEEPCHEM_DATA_DIR to locate the database directory. If DEEPCHEM_DATA_DIR is unset or empty, the function raises an exception instructing the user to download or place an HH-suite database in the data directory. This parameter controls where the database name given in database will be resolved on disk.
        evalue (float): E-value cutoff used for the HH-suite search (passed to the external command as the -e option). Lower values are more stringent; the default 0.001 is a common default to limit matches to statistically significant hits. In DeepChem-style MSA generation this value controls the inclusiveness of homologous sequences added to the resulting alignment.
        num_iterations (int): Number of search iterations (passed to the external command as the -n option). This controls how many rounds of profile search HH-suite will perform to expand the alignment. The default 2 is a conservative choice; increasing this increases sensitivity but also runtime and the risk of accumulating false positives.
        num_threads (int): Number of CPU threads to request from the HH-suite binary (passed as the -cpu option). The default 4 allows parallelism for speed on multi-core systems; set this to match available hardware to control resource usage in DeepChem workflows.
    
    Returns:
        str: Path to the generated MSA file (results.a3m) saved in the same directory as dataset_path. The function guarantees that, on successful execution, a file named results.a3m will be written alongside the input file and the returned string is the absolute or relative filesystem path to that file. Side effects: the function executes an external HH-suite command (constructed from the inputs) via system_call, which will spawn a separate process; this may write additional HH-suite output files into the same directory depending on the HH-suite configuration. Failure modes: if data_dir is None and DEEPCHEM_DATA_DIR is not set or is empty, the function raises an exception with instructions to download a database; if the input file has an unsupported extension the function raises an exception indicating an unsupported file type; if the HH-suite binary is not installed or not on PATH, or if the specified database path does not exist or is incompatible, the external command will fail and the error will propagate from system_call. The function does not perform retry logic or internal parsing of HH-suite output; callers should verify the returned file exists and inspect HH-suite logs for detailed error information.
    """
    from deepchem.utils.sequence_utils import hhblits
    return hhblits(
        dataset_path,
        database,
        data_dir,
        evalue,
        num_iterations,
        num_threads
    )


################################################################################
# Source: deepchem.utils.sequence_utils.hhsearch
# File: deepchem/utils/sequence_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def deepchem_utils_sequence_utils_hhsearch(
    dataset_path: str,
    database: str = None,
    data_dir: str = None,
    evalue: float = 0.001,
    num_iterations: int = 2,
    num_threads: int = 4
):
    """deepchem.utils.sequence_utils.hhsearch runs an HH-suite hhsearch multisequence alignment search for a given sequence dataset and returns the path to the produced multiple sequence alignment (MSA) file. Within the DeepChem project (which provides tools for machine learning in biology and chemistry), this function is used to search a Hidden Markov Model (HMM) reference database to obtain an MSA for protein sequence analysis, feature generation, or downstream modeling workflows. This function invokes an external hhsearch/hh-suite binary on the host system and therefore performs filesystem and external-process side effects described below.
    
    Args:
        dataset_path (str): Path to the input multiple sequence alignment dataset or query sequence file. This path is used to determine the working directory in which results are saved (the function computes save_dir = dirname(abspath(dataset_path))). The function expects common sequence/MSA filename extensions as used by hh-suite: .fas, .fasta, .a3m, .a2m, .hmm. If the file extension is not one of these supported types, the function raises an exception ("Unsupported file type"). The dataset file provides the query sequences for the hhsearch run.
        database (str): Name of the HMM database to search against. This is not a full path but the database directory or basename expected to be found inside data_dir (the function constructs the database path with os.path.join(data_dir, database)). The database must be a preformatted HH-suite HMM reference database (from the HH-suite project at https://github.com/soedinglab/hh-suite). If the named database directory is missing or not a valid HH-suite database, hhsearch will typically fail when the external command is executed.
        data_dir (str): Path to the directory that contains the HMM database named by the database argument. If set to None, the function attempts to read the environment variable DEEPCHEM_DATA_DIR and use that value as the data directory. If DEEPCHEM_DATA_DIR is not set or is an empty string, the function raises an exception instructing the user to download a database (the code raises an error with a message pointing to the HH-suite database download instructions). Practical significance: this parameter locates the reference HMM database files required by hhsearch and is typically set to DeepChem's shared data directory when integrated into DeepChem workflows.
        evalue (float): E-value cutoff passed to the hhsearch command via the -e option. Default is 0.001. The evalue controls the statistical significance threshold for reported matches; lower values are more stringent. The value is formatted into the external command exactly as provided (the function converts it to a string and appends it to the constructed hhsearch command).
        num_iterations (int): Number of iterations. Default is 2. Note: in the current implementation this parameter is accepted by the function signature for API compatibility but is not used when constructing the hhsearch command (i.e., it does not modify the command line or behavior). This parameter is therefore ignored by the code as written; callers should not rely on it to change hhsearch behavior until the implementation is updated.
        num_threads (int): Number of CPU threads passed to hhsearch via the -cpu option. Default is 4. The function converts this integer to a string and inserts it into the command line, so it controls hhsearch parallelism at the external-process level.
    
    Behavior and side effects:
        - The function constructs a shell command invoking the external hhsearch binary (command begins with 'hhsearch') and passes the dataset file (-i), the database path (-d), the output MSA path (-oa3m), CPU thread count (-cpu), and the e-value cutoff (-e). For some input extensions the function also appends '-M first' to the command; the constructed command is executed via system_call(command).
        - The output MSA file is written to a file named results.a3m inside the directory containing dataset_path (save_dir). The function returns the absolute path to that results.a3m file as a str.
        - Because the function invokes an external program, it requires hhsearch/hh-suite binaries to be installed and available in the system PATH. The original documentation references the HH-suite project and hhblits/hhsearch binaries available from https://github.com/soedinglab/hh-suite.
        - If the external hhsearch command generates additional files (for example, .hhr result summaries produced by HH-suite tools), those files may be created in the same save directory as side effects of running hhsearch; the function itself returns only the results.a3m path.
    
    Defaults:
        - If data_dir is None, the function uses the environment variable DEEPCHEM_DATA_DIR. If that variable is not present or is empty, the function raises an exception instructing the user to download and place an HH-suite database in the data directory.
        - Default parameter values are evalue=0.001, num_iterations=2, num_threads=4 (see note above that num_iterations is currently unused).
    
    Failure modes and exceptions:
        - Missing or empty DEEPCHEM_DATA_DIR (when data_dir is None) triggers an exception telling the user to download a database.
        - Unsupported input file extension triggers an exception with message 'Unsupported file type'.
        - If the named database directory (os.path.join(data_dir, database)) does not exist or is not a valid HH-suite database, the external hhsearch invocation will fail and system_call is expected to propagate an error or return a nonzero exit code depending on its implementation.
        - If the hhsearch/hh-suite binaries are not installed or not on PATH, the attempted system call will fail; the function does not itself install binaries.
        - Because the function executes a shell command, callers should be aware of potential shell injection risks if untrusted input is passed into dataset_path or database; the function uses os.path.abspath and os.path.join when building the command but the command string is assembled directly and executed by system_call.
    
    Returns:
        str: Absolute path to the produced MSA file results.a3m saved in the same directory as dataset_path. The returned string points to the file that hhsearch writes via the -oa3m option. Side effects: the function writes results.a3m to disk, may create other hh-suite output files in the same directory, and runs an external hhsearch process. If execution fails, an exception related to the missing database, unsupported file type, missing binary, or the external command failure will be raised instead of returning a path.
    """
    from deepchem.utils.sequence_utils import hhsearch
    return hhsearch(
        dataset_path,
        database,
        data_dir,
        evalue,
        num_iterations,
        num_threads
    )


################################################################################
# Source: deepchem.utils.sequence_utils.system_call
# File: deepchem/utils/sequence_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def deepchem_utils_sequence_utils_system_call(command: str):
    """Execute a shell command and return its captured standard output as raw bytes.
    
    This function is a minimal wrapper around subprocess.Popen used by deepchem.utils.sequence_utils to run system-level commands from Python code. In the DeepChem project (which provides tools for deep learning in drug discovery, materials science, quantum chemistry, and biology), this utility is intended to let sequence-processing code invoke external executables or shell utilities as part of data preparation or pipeline steps. The implementation calls subprocess.Popen([command], stdout=subprocess.PIPE, shell=True) and then reads the entire stdout stream via p.stdout.read(), returning the raw bytes produced by the child process.
    
    Behavior and side effects:
        - Spawns a new shell process and executes the provided command string. Because shell=True is used, the command is interpreted by the system shell.
        - The function captures only the child process's standard output (stdout) and returns it as bytes. Standard error (stderr) is not captured and will be directed to the parent process's stderr by default.
        - The call blocks while reading stdout and will wait until the child process closes its stdout pipe; there is no timeout or asynchronous behavior.
        - The entire stdout content is read into memory before being returned; commands that produce very large output may consume significant memory.
        - No exit status or return code is checked or returned; the function does not surface nonzero exit codes. Exceptions raised by subprocess.Popen (for example, OSError if the shell cannot be spawned) are not caught and will propagate to the caller.
        - Because the command string is executed through a shell, passing untrusted input as command can lead to shell injection vulnerabilities; callers must sanitize or avoid passing untrusted data.
    
    Args:
        command (str): The shell command to execute. This must be a Python str containing the exact command that should be interpreted by the system shell. In DeepChem sequence-processing contexts this typically represents an external tool invocation or a short shell script used to transform or analyze sequence data. The function passes this string to subprocess.Popen in a single-element list while enabling shell=True, so the string is interpreted by the shell. Provide fully-formed commands appropriate for the target platform; do not assume any additional quoting or argument splitting is performed by this wrapper.
    
    Returns:
        bytes: The raw bytes read from the child process's standard output (the result of p.stdout.read()). If the command writes nothing to stdout, an empty bytes object (b'') is returned. Callers that need text should decode the result using the appropriate encoding (for example, decode('utf-8')). Exceptions from subprocess.Popen or from IO operations may be raised instead of returning a value, and no information about the process exit code or stderr is provided by this function.
    """
    from deepchem.utils.sequence_utils import system_call
    return system_call(command)


################################################################################
# Source: deepchem.utils.voxel_utils.convert_atom_pair_to_voxel
# File: deepchem/utils/voxel_utils.py
# Category: valid
################################################################################

def deepchem_utils_voxel_utils_convert_atom_pair_to_voxel(
    coordinates_tuple: Tuple[numpy.ndarray, numpy.ndarray],
    atom_index_pair: Tuple[int, int],
    box_width: float,
    voxel_width: float
):
    """Converts a pair of atoms (one from each of two molecules) into voxel grid indices (i, j, k) suitable for voxel-based 3D representations used in molecular machine learning and structure-aware models in DeepChem (for example, inputs to 3D convolutional networks or voxel featurizers used in drug discovery and materials modeling).
    
    Args:
        coordinates_tuple (Tuple[np.ndarray, np.ndarray]): A tuple containing two coordinate arrays for the two molecules. Each array is expected to be a 2-D numpy array of shape (N, 3) and (M, 3) respectively, where rows are atomic coordinates in Cartesian Angstrom units and columns correspond to x, y, z. These arrays provide the 3D positions from which the atom indices in atom_index_pair will be converted into voxel grid indices.
        atom_index_pair (Tuple[int, int]): A tuple of two integer indices (index_in_first_array, index_in_second_array). The first integer selects an atom (row) from the first array in coordinates_tuple and the second integer selects an atom from the second array. These indices are used to locate the atoms whose voxel coordinates will be computed.
        box_width (float): The side length of the cubic spatial box in Angstroms that defines the domain over which voxelization is performed. The box is discretized into voxels of size voxel_width; box_width controls the spatial extent and thus the mapping from continuous coordinates to discrete voxel indices.
        voxel_width (float): The width of a single voxel in Angstroms. This value determines the resolution of the voxel grid (smaller values produce finer grids). It is used together with box_width to compute integer i,j,k indices for each atom position.
    
    Returns:
        np.ndarray: A numpy array of shape (2, 3) where each row corresponds to the voxel grid indices [i, j, k] for the atom specified by the corresponding entry in atom_index_pair. The first row is the voxel indices for the atom from the first coordinate array, and the second row is for the atom from the second coordinate array. The returned values represent discrete voxel coordinates within the cubic box defined by box_width and voxel_width.
    
    Behavior and failure modes:
        This function iterates over the pair (coordinates_array, atom_index) for the two inputs in coordinates_tuple and atom_index_pair, and for each calls convert_atom_to_voxel(coordinates, atom_index, box_width, voxel_width) to compute that atom's voxel indices, collecting the two results into the returned (2, 3) array. The function performs no in-place mutation of the input coordinate arrays and has no side effects beyond allocating and returning the indices array.
        Common failure modes include IndexError if an atom_index is out of range for its corresponding coordinate array, ValueError if coordinate arrays are not two-dimensional with three columns (i.e., not shape (N, 3)), ZeroDivisionError or undefined behavior if voxel_width is zero, and logical errors if box_width or voxel_width are non-positive. Callers should validate that coordinates are in Angstroms, that atom_index_pair refers to valid rows in the provided arrays, and that box_width and voxel_width are positive floats appropriate for the desired voxel resolution.
    """
    from deepchem.utils.voxel_utils import convert_atom_pair_to_voxel
    return convert_atom_pair_to_voxel(
        coordinates_tuple,
        atom_index_pair,
        box_width,
        voxel_width
    )


################################################################################
# Source: deepchem.utils.voxel_utils.convert_atom_to_voxel
# File: deepchem/utils/voxel_utils.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for convert_atom_to_voxel because the docstring has no description for the argument 'coordinates'
################################################################################

def deepchem_utils_voxel_utils_convert_atom_to_voxel(
    coordinates: numpy.ndarray,
    atom_index: int,
    box_width: float,
    voxel_width: float
):
    """Converts a single atom's 3D Cartesian coordinates into integer voxel grid indices (i, j, k) by shifting coordinates into a box origin and discretizing by the voxel size. This is used in DeepChem featurizers that build 3D voxel grids (for example, inputs to 3D convolutional neural networks in molecular machine learning and computational chemistry) to determine which voxel a particular atom occupies.
    
    The function offsets the atom coordinates by (box_width/2, box_width/2, box_width/2) so that the box center is treated as the origin at index (0,0,0) after discretization, divides the shifted coordinates by voxel_width, applies a floor operation to map continuous positions to integer voxel indices, and casts the result to integers using numpy.floor and astype(int). No in-place modification of inputs occurs.
    
    Args:
        coordinates (numpy.ndarray): Array with coordinates of all atoms in the molecule, shape (N, 3). Coordinates are expected to be 3D Cartesian positions (same length unit as box_width and voxel_width, typically Angstroms). This argument provides the source positions from which the voxel index for the selected atom is computed.
        atom_index (int): Index of an atom in the molecule. This selects which row of coordinates to convert to voxel indices. If atom_index is out of bounds for coordinates (e.g., negative or >= N), a standard numpy IndexError will be raised by the indexing operation.
        box_width (float): Size of the cubic box in Angstroms that defines the spatial region to be voxelized. The function shifts coordinates by box_width/2 along each axis so that the box center corresponds to the zero-origin used for indexing. If atoms lie outside the ±box_width/2 range along any axis, the resulting voxel index may be negative or greater than or equal to box_width/voxel_width; callers should validate or clamp indices if they must fall within a fixed grid.
        voxel_width (float): Size of a single voxel in Angstroms used to discretize space. The shifted coordinates are divided by voxel_width and then floored to produce integer indices. Passing voxel_width <= 0 will lead to a division by zero or invalid results; callers must ensure voxel_width is a positive float.
    
    Returns:
        numpy.ndarray: A 1D numpy array of length 3 with integer voxel indices [i, j, k] for the specified atom. These indices indicate the voxel coordinates within a grid defined by box_width and voxel_width and are suitable for indexing into a 3D array representing atom occupancy or feature channels. The returned indices may lie outside the valid range of a preallocated grid if the atom is outside the box; the caller is responsible for handling such cases (for example by clipping indices or ignoring out-of-box atoms).
    """
    from deepchem.utils.voxel_utils import convert_atom_to_voxel
    return convert_atom_to_voxel(coordinates, atom_index, box_width, voxel_width)


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
