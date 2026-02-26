"""
Regenerated Google-style docstrings for module 'tdc'.
README source: others/readme/tdc/README.md
Generated at: 2025-12-02T02:49:53.826437Z

Total functions: 118
"""


import numpy

################################################################################
# Source: tdc.chem_utils.evaluator.calculate_internal_pairwise_similarities
# File: tdc/chem_utils/evaluator.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_chem_utils_evaluator_calculate_internal_pairwise_similarities(smiles_list: list):
    """Calculates pairwise internal chemical similarities for a list of SMILES strings using molecular fingerprints.
    
    This function is used in TDC's cheminformatics utilities to quantify chemical similarity within a single set of molecules (for example, to assess dataset redundancy, chemical diversity, or to select nearest-neighbor analogs when preparing splits or analyses). It converts the input SMILES strings to molecule objects via get_mols, computes fingerprints via get_fingerprints, and then computes pairwise Tanimoto similarities using RDKit's DataStructs.BulkTanimotoSimilarity in a memory-efficient incremental manner. The returned matrix is symmetric and the diagonal is explicitly set to zero to avoid counting self-similarity in downstream thresholding or aggregation steps.
    
    Args:
        smiles_list (list of str): Ordered list of SMILES strings representing the chemical structures to compare. Each element should be a SMILES string (e.g., "CCO"). The order of this list determines the row/column ordering in the returned similarity matrix. The function does not modify this list. Note: conversion from SMILES to molecule objects and fingerprint computation are performed by the helper functions get_mols and get_fingerprints; errors or failures in parsing or fingerprinting (for example, invalid SMILES) will propagate from those helpers.
    
    Returns:
        np.ndarray: A two-dimensional NumPy array of shape (n, n) where n is the number of fingerprints produced by get_fingerprints for the provided SMILES list. Element [i, j] contains the Tanimoto similarity between the fingerprint of molecule i and molecule j. The matrix is symmetric (similarity[i, j] == similarity[j, i]) and the diagonal entries are set to zero to exclude self-similarity in subsequent analyses. If the input list is empty or no fingerprints are produced, an empty array with shape (0, 0) is returned.
    
    Behavior and side effects:
        - If len(smiles_list) > 10000, a warning is logged (via logger.warning) to indicate a large computation; this is because computational time and memory scale approximately O(n^2) with the number of molecules.
        - The function internally calls get_mols(smiles_list) and get_fingerprints(mols). These helper functions typically use RDKit to parse SMILES into molecule objects and to compute binary or count-based fingerprints; therefore, parsing/featurization errors or missing RDKit dependencies will raise exceptions originating from those helpers or from RDKit.
        - The function computes pairwise similarities by iterating over fingerprints and calling DataStructs.BulkTanimotoSimilarity for each fingerprint against previously processed fingerprints; this avoids explicitly computing all pairs at once but still produces a full n-by-n matrix in memory.
        - Returned similarity values correspond to the fingerprint-based Tanimoto metric (commonly ranging between 0 and 1, with higher values indicating greater fingerprint overlap). The diagonal is set to zero by design to prevent self-similarity from biasing neighbor-based analyses.
    
    Failure modes and limitations:
        - Invalid or unparsable SMILES may cause get_mols or get_fingerprints to raise exceptions or to omit molecules; the function does not attempt to repair or impute missing molecules.
        - Very large inputs may exhaust memory because the final similarity matrix is dense and of size n^2.
        - The similarity values reflect the chosen fingerprint type computed by get_fingerprints; different fingerprint schemes will produce different similarity scales and interpretations.
    """
    from tdc.chem_utils.evaluator import calculate_internal_pairwise_similarities
    return calculate_internal_pairwise_similarities(smiles_list)


################################################################################
# Source: tdc.chem_utils.evaluator.calculate_pc_descriptors
# File: tdc/chem_utils/evaluator.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_chem_utils_evaluator_calculate_pc_descriptors(smiles: list, pc_descriptors: list):
    """Calculate physical–chemical (PC) descriptor vectors for a list of molecules represented as SMILES strings.
    
    This function is used in TDC's cheminformatics utilities to produce numerical feature vectors that summarize molecular properties (for example, molecular weight, logP, polar surface area) required by downstream tasks such as ADME prediction, model evaluation, and benchmark construction. For each SMILES string in the input list, the function calls the internal helper _calculate_pc_descriptors(smiles, pc_descriptors) to compute the requested descriptor values, collects the non-None results, and returns them as a NumPy array suitable as input features for machine learning models or statistical analyses.
    
    Args:
        smiles (list): A Python list of SMILES strings. Each element is a textual representation of a single small-molecule structure using the SMILES notation. The function iterates over this list and attempts to compute the requested descriptors for each SMILES. Practical significance: this is the set of molecules for which PC descriptors are being computed for use as model features in TDC tasks (for example, single-instance prediction tasks such as ADME).
        pc_descriptors (list): A Python list of descriptor names (strings). Each string identifies a physical–chemical descriptor to compute for every molecule (for example, "MolWt", "LogP", "TPSA"). The helper _calculate_pc_descriptors is expected to recognize these names and return a numeric vector with one value per requested descriptor. Practical significance: this list defines which numerical features will be produced and hence which molecular properties will be available to downstream ML models and evaluations.
    
    Returns:
        numpy.ndarray: A NumPy array of numeric descriptor vectors. Each row corresponds to one molecule for which _calculate_pc_descriptors returned a non-None result; each column corresponds to one descriptor requested in pc_descriptors. The returned array therefore has shape (N, D) where N is the number of successfully processed SMILES (N <= len(smiles)) and D is len(pc_descriptors). If all calls to the helper return None, an empty NumPy array is returned. The numeric type and precise shape follow directly from the helper's output; downstream code can treat rows as feature vectors for model input.
    
    Behavior and side effects:
        The function performs no I/O and does not modify its inputs. It processes molecules in the order provided by the smiles list but filters out molecules for which the helper returns None; thus the output preserves relative ordering of successful computations but may be shorter than the input list. The function uses the internal helper _calculate_pc_descriptors to perform per-molecule computation; any behavior (including dtype, order of descriptors, or error conditions) arising from that helper will affect this function.
    
    Defaults and performance:
        There are no hidden defaults in this function beyond those implemented by the helper. Runtime scales roughly linearly with the number of SMILES and number of requested descriptors; large lists will consume time proportional to len(smiles) * len(pc_descriptors).
    
    Failure modes and errors:
        If smiles is not an iterable of strings, or pc_descriptors is not an iterable of strings, the function may raise TypeError or propagate exceptions from the helper. If a descriptor name in pc_descriptors is invalid or unsupported, the helper may raise a ValueError or return None for that molecule; such molecules will be skipped (their results excluded from the returned array). If all molecules are skipped, the function returns an empty NumPy array rather than None. Calling code should validate inputs and handle the possibility of an empty result.
    """
    from tdc.chem_utils.evaluator import calculate_pc_descriptors
    return calculate_pc_descriptors(smiles, pc_descriptors)


################################################################################
# Source: tdc.chem_utils.evaluator.canonicalize
# File: tdc/chem_utils/evaluator.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_chem_utils_evaluator_canonicalize(smiles: str):
    """Convert a SMILES string into a deterministic canonical SMILES representation suitable for use in TDC data pipelines, evaluators, and molecule-generation workflows.
    
    Args:
        smiles (str): Input SMILES string representing a small-molecule structure. This parameter is the molecular input provided by users or upstream TDC data functions (for example, dataset loaders, processors, or molecule-generation oracles). The function expects a textual SMILES encoding (type str). The practical significance in the TDC domain is that canonicalizing SMILES produces a consistent, reproducible string form that is useful for deduplication, deterministic dataset splits, feature extraction, model input normalization, and fair evaluation on benchmarks.
    
    Returns:
        str or None: If the input SMILES parses successfully, returns a canonical SMILES string (type str) produced by RDKit's Chem.MolToSmiles with isomericSmiles=True, preserving stereochemical information when present. This canonical form provides a consistent textual representation across equivalent molecule inputs and is intended for downstream use in TDC evaluators, oracles, and data loaders. If the input cannot be parsed into an RDKit molecule (Chem.MolFromSmiles returns None), the function returns None to indicate an invalid or unparsable SMILES; callers should handle this case (for example, by filtering out or logging invalid entries). The function does not modify its input argument and has no other side effects beyond using RDKit to parse and serialize the molecule.
    
    Behavior and failure modes:
        The implementation uses RDKit's Chem.MolFromSmiles to parse the provided SMILES and Chem.MolToSmiles(..., isomericSmiles=True) to generate the canonical output. Canonicalization is deterministic for a given RDKit version and ensures a reproducible ordering of atoms in the serialized SMILES; it does not perform additional normalization steps such as tautomer or protonation standardization. If the provided smiles string is not a valid SMILES or cannot be parsed, the function returns None rather than raising an exception; callers should therefore check for None before using the returned value.
    """
    from tdc.chem_utils.evaluator import canonicalize
    return canonicalize(smiles)


################################################################################
# Source: tdc.chem_utils.evaluator.continuous_kldiv
# File: tdc/chem_utils/evaluator.py
# Category: valid
################################################################################

def tdc_chem_utils_evaluator_continuous_kldiv(
    X_baseline: numpy.ndarray,
    X_sampled: numpy.ndarray
):
    """tdc.chem_utils.evaluator.continuous_kldiv computes the continuous Kullback–Leibler divergence between two empirical continuous distributions represented as numpy arrays. In the Therapeutics Data Commons (TDC) context, this function is used to quantify how a distribution of sampled molecular property values (for example, properties produced by a generative oracle or model) diverges from a baseline distribution (for example, property values in a reference dataset), enabling evaluation of distributional fidelity in generation and benchmarking workflows.
    
    This function fits Gaussian kernel density estimates (KDEs) to the two input arrays, evaluates the two estimated densities on a common grid spanning the combined support of the inputs, and then computes the KL divergence D_KL(P || Q) using scipy.stats.entropy (natural-log base, result in nats). It applies small additive constants to avoid zeros in densities and to stabilize KDE evaluation. Note that the implementation mutates the input numpy arrays in-place by adding a small constant; pass copies if you need to preserve the original arrays.
    
    Args:
        X_baseline (numpy array): 1-D numpy array of baseline continuous samples. In TDC usage this typically contains reference measurements or property values (for example, experimentally observed molecular properties) that define the target distribution P. The function treats these values as empirical samples used to fit a Gaussian KDE for P. The array must be numeric and finite; NaNs or infinities will cause KDE or entropy computations to fail. The function will add 1e-5 to this array in-place to avoid zeros before KDE fitting.
        X_sampled (numpy array): 1-D numpy array of sampled continuous values to compare against the baseline (defines distribution Q). In TDC workflows this commonly comes from a generative model, oracle outputs, or resampled data whose distributional divergence from the baseline is being evaluated. This array must be numeric and finite; the function will add 1e-5 to this array in-place to avoid zeros before KDE fitting.
    
    Returns:
        float: KL divergence D_KL(P || Q) computed as a non-negative floating-point value in nats. This value quantifies how much the KDE-estimated distribution of X_baseline (P) diverges from that of X_sampled (Q). Smaller values indicate closer agreement between distributions. The computation uses a common evaluation grid of 1000 points spanning the combined min and max of the inputs, adds 1e-10 to the evaluated densities for numerical stability, and then calls scipy.stats.entropy(P, Q) to compute the result.
    
    Behavior, defaults, and failure modes:
        The function fits scipy.stats.gaussian_kde to each input and evaluates densities on a linspace of 1000 points covering the pooled range of X_baseline and X_sampled. It adds 1e-5 to inputs (in-place) and 1e-10 to evaluated densities to avoid division-by-zero or log-of-zero issues. If inputs are extremely small, identical constants, too few unique samples, or degenerate (all values identical), gaussian_kde may fail (for example, due to singular covariance) and raise a linear algebra error. If inputs contain NaNs or infinities, KDE or entropy computations will raise errors. The function requires scipy.stats.gaussian_kde and scipy.stats.entropy to be available. Because inputs are modified in-place, callers who must preserve original arrays should pass copies (for example, X_baseline.copy()). The function does not perform input reshaping; it expects 1-D arrays as provided in typical TDC evaluation pipelines.
    """
    from tdc.chem_utils.evaluator import continuous_kldiv
    return continuous_kldiv(X_baseline, X_sampled)


################################################################################
# Source: tdc.chem_utils.evaluator.discrete_kldiv
# File: tdc/chem_utils/evaluator.py
# Category: valid
################################################################################

def tdc_chem_utils_evaluator_discrete_kldiv(
    X_baseline: numpy.ndarray,
    X_sampled: numpy.ndarray
):
    """tdc.chem_utils.evaluator.discrete_kldiv calculates the discrete Kullback–Leibler (KL) divergence between two empirical numeric distributions by binning their values into histograms and computing D_KL(P || Q) = sum_i P_i log(P_i / Q_i). In the Therapeutics Data Commons (TDC) context, this function is useful for quantifying how a sampled distribution (for example, molecule property values produced by a generative oracle or model) diverges from a baseline distribution (for example, a reference dataset of experimental measurements or desired property distribution) when both are represented as one-dimensional numeric arrays.
    
    Args:
        X_baseline (numpy array): 1-D numeric array representing the baseline empirical samples. This array defines the histogram bin edges (bins=10, density=True) used to approximate the baseline probability density P. In TDC workflows, X_baseline typically holds property values or feature samples from a reference dataset against which generated or candidate distributions are compared. The function expects numeric values; non-numeric entries (NaN/Inf or non-scalar types) may cause SciPy histogram routines to raise errors or produce undefined behavior.
        X_sampled (numpy array): 1-D numeric array representing the sampled empirical samples to compare against the baseline. The histogram for X_sampled is computed using the same bin edges as X_baseline so that P and Q are aligned. In TDC use cases, X_sampled commonly contains values produced by a model, oracle, or resampling procedure whose distributional divergence from X_baseline is being evaluated.
    
    Returns:
        float: The computed KL divergence D_KL(P || Q) as returned by scipy.stats.entropy(P, Q). P and Q are the density-normalized histogram counts (density=True) for X_baseline and X_sampled respectively, computed with 10 bins and identical bin edges. A small constant (1e-10) is added to every bin of P and Q to avoid zero probabilities and numerical instability. The returned value is non-negative with 0.0 indicating identical binned distributions under this discretization. Possible failure modes include: SciPy not being installed (ImportError), inputs that are not one-dimensional numeric arrays (ValueError or unexpected histogram results), or inputs containing NaN/Inf which may yield invalid histogram buckets; callers should pre-clean inputs and ensure SciPy is available. The function has no side effects and does not modify its input arrays.
    """
    from tdc.chem_utils.evaluator import discrete_kldiv
    return discrete_kldiv(X_baseline, X_sampled)


################################################################################
# Source: tdc.chem_utils.evaluator.diversity
# File: tdc/chem_utils/evaluator.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_chem_utils_evaluator_diversity(list_of_smiles: list):
    """tdc.chem_utils.evaluator.diversity evaluates the internal chemical diversity of a set of molecules used in TDC generation and dataset analysis workflows. It computes the average pairwise Tanimoto distance between Morgan fingerprints for a deduplicated list of SMILES, providing a single scalar metric that quantifies how chemically diverse the input molecule set is (useful for assessing generative model outputs, dataset curation, and benchmark analysis in therapeutics discovery).
    
    Args:
        list_of_smiles (list): A Python list of SMILES strings representing molecules. This argument is the input molecule set whose internal diversity will be measured. The function first deduplicates this list by calling unique_lst_of_smiles (so identical SMILES do not bias the diversity estimate), then converts each unique SMILES to an RDKit Mol object with Chem.MolFromSmiles, and computes Morgan fingerprints for each molecule with radius=2, nBits=2048, and useChirality=False. The caller is responsible for providing SMILES strings; invalid SMILES may produce MolFromSmiles returns of None and can lead to downstream errors.
    
    Returns:
        div (float): The internal diversity score computed as the mean of pairwise Tanimoto distances (distance = 1 - TanimotoSimilarity) across all unique molecule pairs. The distance lies in the range [0.0, 1.0] when similarity is defined on valid fingerprints: 0.0 indicates identical fingerprints (no diversity) and values closer to 1.0 indicate greater dissimilarity (higher diversity). If the input contains fewer than two unique valid molecules, no pairwise distances are available and NumPy's mean over an empty sequence will yield numpy.nan and emit a RuntimeWarning. Potential failure modes include exceptions raised by RDKit functions (for example, if MolFromSmiles returns None and GetMorganFingerprintAsBitVect is called on it) or other runtime errors when RDKit is not installed. Computational complexity is O(n^2) in the number of unique molecules because all unique pairwise comparisons are performed.
    """
    from tdc.chem_utils.evaluator import diversity
    return diversity(list_of_smiles)


################################################################################
# Source: tdc.chem_utils.evaluator.fcd_distance
# File: tdc/chem_utils/evaluator.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_chem_utils_evaluator_fcd_distance(
    generated_smiles_lst: list,
    training_smiles_lst: list
):
    """Compute the Fréchet ChemNet Distance (FCD) between a set of generated SMILES strings and a set of training SMILES strings using an available backend implementation (TensorFlow or PyTorch). In the Therapeutics Data Commons (TDC) generation context, FCD is used as a distributional similarity metric to evaluate how closely a molecular generative model's outputs match the chemical space of the training data; lower values indicate the generated distribution is closer to the training distribution. This function chooses the implementation at runtime: it attempts to use the TensorFlow-based FCD implementation (fcd) and its wrapper fcd_distance_tf, and if that import fails it attempts the PyTorch-based implementation (fcd_torch) and delegates to fcd_distance_torch. If neither implementation is available, an ImportError is raised instructing how to install a supported backend.
    
    Args:
        generated_smiles_lst (list): A list of SMILES strings produced by a molecular generative model. Each element is expected to be a SMILES representation (str) of a small molecule; these are the samples whose distributional similarity to the training set will be evaluated. In TDC generation benchmarks, this argument represents the candidate molecules produced by a model or oracle and serves as the "generated" distribution input to the FCD computation.
        training_smiles_lst (list): A list of SMILES strings drawn from the training data distribution. Each element is expected to be a SMILES representation (str) of a small molecule; these are treated as the reference distribution against which generated_smiles_lst is compared. In TDC workflows this typically corresponds to the dataset used to train or condition a generative model.
    
    Returns:
        fcd_distance (float): The computed Fréchet ChemNet Distance between the generated and training SMILES distributions. This scalar is non-negative and measures distance in a learned ChemNet feature space; lower values indicate that the generated set is more similar to the training set in that feature space. The returned value is produced by either fcd_distance_tf or fcd_distance_torch depending on which backend was selected at runtime.
    
    Raises:
        ImportError: If neither the TensorFlow-based FCD package ('FCD' / fcd) nor the PyTorch-based package ('fcd_torch') can be imported, an ImportError is raised with guidance to install one of these packages (for example, "pip install FCD" for the TensorFlow backend or "pip install fcd_torch" for the PyTorch backend).
        Exception: Errors raised by the selected backend implementation (for example, errors parsing invalid SMILES, memory errors for very large input lists, or internal numerical exceptions) are propagated to the caller. Callers should ensure inputs are valid SMILES strings and be aware that computing FCD can be computationally and memory intensive for large lists.
    """
    from tdc.chem_utils.evaluator import fcd_distance
    return fcd_distance(generated_smiles_lst, training_smiles_lst)


################################################################################
# Source: tdc.chem_utils.evaluator.fcd_distance_tf
# File: tdc/chem_utils/evaluator.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_chem_utils_evaluator_fcd_distance_tf(
    generated_smiles_lst: list,
    training_smiles_lst: list
):
    """Evaluate the Fréchet ChemNet Distance (FCD) between two sets of molecules represented as SMILES strings using a TensorFlow-based ChemNet reference model and return a bounded similarity score derived from the FCD.
    
    This function is used in molecular generation and evaluation workflows (for example, in Therapeutics Data Commons tasks for molecule generation) to quantify how closely the distribution of generated molecules matches the distribution of molecules from a training set. Practically, it canonicalizes SMILES, feeds them through a pretrained ChemNet model to obtain high-level activations, computes multivariate Gaussian statistics (mean and covariance) for each set, computes the Fréchet distance between those Gaussians, and maps the Fréchet distance to a similarity score via the exponential transform implemented in the function. The function caches the loaded ChemNet model in the module-level global variable chemnet and writes the pretrained model file to the system temporary directory the first time it is needed.
    
    Args:
        generated_smiles_lst (list): List (of SMILES string) representing molecules produced by a generative model. Each element is expected to be a SMILES string. These molecules are canonicalized and converted into ChemNet activations; the resulting distribution is compared to the reference (training) distribution. Supplying an empty list or elements that are not valid SMILES may raise errors from the fcd canonicalization or prediction routines.
        training_smiles_lst (list): List (of SMILES string) representing molecules from the training set used as the reference distribution. Each element is expected to be a SMILES string. This list is used to build the reference multivariate Gaussian (mean and covariance) of ChemNet activations. Supplying an empty list or invalid SMILES may raise errors from the fcd routines.
    
    Returns:
        float: A similarity score computed as exp(-0.2 * FCD) where FCD is the Fréchet ChemNet Distance between the ChemNet activation distributions of training_smiles_lst and generated_smiles_lst. Because the Fréchet distance is non-negative, the returned value lies in the interval (0, 1], with values closer to 1 indicating that the generated distribution is more similar to the training distribution. The returned float is intended for use as a single-number evaluation metric in molecule generation benchmarks (for example, to rank generators or report distributional fidelity on TDC leaderboards).
    
    Behavior, side effects, and failure modes:
        - On first call (per process), the function loads the pretrained ChemNet model file "ChemNet_v0.13_pretrained.h5" from the installed fcd package via pkgutil.get_data, writes it to the system temporary directory, and calls fcd.load_ref_model to create a model object. That model object is stored in the module-level global variable chemnet so subsequent calls reuse the loaded model and avoid repeated disk writes and model-loading overhead.
        - The function depends on the external fcd package for SMILES canonicalization (fcd.canonical_smiles), ChemNet predictions (fcd.get_predictions), and FCD calculation (fcd.calculate_frechet_distance), and on numpy (np) for mean and covariance computations. If these packages are not available or their APIs change, the function will raise the corresponding ImportError or AttributeError propagated from the underlying calls.
        - The temporary file write may fail (e.g., due to lack of permissions or insufficient disk space); in that case, an OSError or IOError will be raised when attempting to write the model file to the system temporary directory.
        - If either input list is empty, numpy.mean or numpy.cov may raise warnings or errors (for example, returning NaNs or raising exceptions) when computing statistics; callers should ensure non-empty, well-formed SMILES lists to obtain meaningful scores.
        - If fcd.canonical_smiles or fcd.get_predictions encounters invalid SMILES or other molecule parsing issues, those routines may raise exceptions; such exceptions indicate malformed input SMILES or unexpected behavior in the fcd library.
        - The function performs I/O and model loading on first use and CPU/GPU work when computing predictions; the cost scales with the number of SMILES and the ChemNet model complexity.
    
    Practical significance in TDC and molecule generation:
        - This metric is suitable for distributional evaluation of molecule generators in TDC generation tasks: it captures differences in high-level learned features (ChemNet activations) between generated molecules and a reference training set, and maps those differences to a single interpretable score for ranking and reporting on leaderboards.
        - Use this function when you need a TensorFlow/ChemNet-based distribution similarity metric that is consistent with prior molecule-generation literature using Fréchet ChemNet Distance.
    """
    from tdc.chem_utils.evaluator import fcd_distance_tf
    return fcd_distance_tf(generated_smiles_lst, training_smiles_lst)


################################################################################
# Source: tdc.chem_utils.evaluator.fcd_distance_torch
# File: tdc/chem_utils/evaluator.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_chem_utils_evaluator_fcd_distance_torch(
    generated_smiles_lst: list,
    training_smiles_lst: list
):
    """Compute the Fréchet ChemNet Distance (FCD) between a set of generated molecules and a set of reference (training) molecules using the fcd_torch implementation on CPU.
    
    This function is used within Therapeutics Data Commons (TDC) workflows for quantitative evaluation of molecule generation models and oracles. FCD is a distributional metric computed in a pretrained ChemNet feature space and is intended to measure how closely the distribution of generated molecules matches the distribution of training molecules; in practical TDC benchmarking, lower FCD values indicate that generated molecules are more similar to the training distribution. The implementation invoked here sets an environment variable to permit duplicate OpenMP libraries, imports the fcd_torch.FCD class, instantiates an FCD evaluator with device="cpu" and n_jobs=8, and then computes the FCD by calling the evaluator with the two SMILES lists.
    
    Args:
        generated_smiles_lst (list): List of SMILES strings produced by a generative model. Each element is a textual SMILES representation of a single molecule. This argument provides the candidate distribution whose similarity to the reference distribution is being assessed; in TDC generation tasks this would typically be the output of a molecule generator or oracle.
        training_smiles_lst (list): List of SMILES strings from the training/reference set. Each element is a textual SMILES representation of a single molecule used as the reference distribution for evaluation; in TDC benchmarking this is typically the data used to train or define the target distribution.
    
    Behavior and side effects:
        The function sets the environment variable KMP_DUPLICATE_LIB_OK to "True" to avoid runtime errors arising from duplicate OpenMP libraries. It imports the FCD class from the fcd_torch package and constructs an FCD instance with device="cpu" and n_jobs=8 (these defaults are hard-coded in the implementation). The computation is performed by fcd_torch and happens synchronously on the calling thread; execution time depends on the number and complexity of SMILES provided. No files are written by this function. The only persistent side effect is the change to the KMP_DUPLICATE_LIB_OK environment variable for the current process.
    
    Failure modes and requirements:
        This function requires the external Python package fcd_torch to be installed and importable; an ImportError will be raised if it is missing. The FCD evaluator may raise exceptions (for example, ValueError or other runtime errors) if input SMILES are syntactically invalid or if either list is empty; callers should validate or filter SMILES beforehand when appropriate. Because the FCD computation may be computationally intensive, callers should also be prepared to handle long runtimes or memory pressure for very large lists.
    
    Returns:
        float: A scalar FCD distance between the generated_smiles_lst distribution and the training_smiles_lst distribution as computed by fcd_torch. In TDC use-cases this scalar is used as an evaluation metric for generative models and leaderboards; smaller values indicate greater similarity of the generated distribution to the training distribution.
    """
    from tdc.chem_utils.evaluator import fcd_distance_torch
    return fcd_distance_torch(generated_smiles_lst, training_smiles_lst)


################################################################################
# Source: tdc.chem_utils.evaluator.get_fingerprints
# File: tdc/chem_utils/evaluator.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_chem_utils_evaluator_get_fingerprints(
    mols: list,
    radius: int = 2,
    length: int = 4096
):
    """Converts a list of small molecules into ECFP (extended-connectivity/Morgan) fixed-length bit-vector fingerprints for use as molecular descriptors in machine-learning workflows (e.g., TDC benchmarks for activity, ADME, and safety prediction).
    
    Args:
        mols (list): A list of RDKit molecule objects (rdkit.Chem.Mol). Each element is treated as one chemical instance and is converted, in the same order, to a fingerprint bit vector using RDKit's AllChem.GetMorganFingerprintAsBitVect. In TDC workflows, these fingerprints serve as input features for models that predict molecular properties (activity screening, ADME, toxicity, etc.).
        radius (int): ECFP/Morgan fingerprint radius (neighborhood radius used to generate circular atom environments). This controls the structural neighborhood each bit captures (default 2, which corresponds to commonly used "ECFP4" behavior in many molecular ML applications). Increasing radius captures larger local substructures; decreasing it captures smaller neighborhoods.
        length (int): Number of bits in the output binary fingerprint vector (fixed-length bit vector size). The function produces bit vectors of this length for each molecule (default 4096). A larger length reduces hash collisions at the cost of larger input dimensionality for downstream models.
    
    Returns:
        list: A list of RDKit bit-vector fingerprint objects produced by rdkit.Chem.AllChem.GetMorganFingerprintAsBitVect, in the same order as the input mols list. Each list element is a fixed-length binary fingerprint representing the corresponding molecule's local structural features and is intended for use as an input descriptor in TDC model training, evaluation, or similarity computations.
    
    Behavior and side effects:
        This function applies AllChem.GetMorganFingerprintAsBitVect to each molecule and does not modify the input mol objects. It is deterministic for a given set of RDKit molecules and the same radius and length parameters. Computational cost scales approximately linearly with the number of molecules.
    
    Failure modes and requirements:
        RDKit must be available in the runtime environment; otherwise NameError/ImportError will occur. If elements of mols are not valid RDKit Mol objects (for example, None or other types), RDKit will raise an exception (TypeError/ValueError) when attempting to generate the fingerprint. The function does not perform molecule sanitization or additional validation beyond what RDKit's GetMorganFingerprintAsBitVect performs.
    """
    from tdc.chem_utils.evaluator import get_fingerprints
    return get_fingerprints(mols, radius, length)


################################################################################
# Source: tdc.chem_utils.evaluator.get_mols
# File: tdc/chem_utils/evaluator.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_chem_utils_evaluator_get_mols(smiles_list: list):
    """tdc.chem_utils.evaluator.get_mols converts a sequence of SMILES strings into RDKit RDMol objects for downstream TDC tasks such as dataset processing, evaluation, or molecule-generation oracles.
    
    Args:
        smiles_list (list): A Python list of SMILES strings. Each entry is expected to be a SMILES representation of a small molecule (text format commonly used across TDC datasets and oracles). This argument is the primary input used by TDC data functions and evaluators to obtain molecular graph objects suitable for cheminformatics operations and model evaluation.
    
    Behavior:
        This function iterates over smiles_list and for each SMILES string calls rdkit.Chem.MolFromSmiles to parse it into an RDKit RDMol object. It yields each successfully parsed RDMol as it is produced (the function is a generator). Invalid SMILES that cause MolFromSmiles to return None are silently skipped (not yielded). Any exceptions raised by RDKit or string processing are caught; the exception is logged via the module logger at warning level and the function continues processing the remaining SMILES. Because this function yields results, callers that need a concrete list should materialize it with list(get_mols(smiles_list)).
    
    Side effects and requirements:
        - Requires RDKit (rdkit.Chem) to be available in the runtime environment because it calls Chem.MolFromSmiles.
        - Emits warnings to the module logger when an exception occurs while parsing a SMILES string; logging is the only side effect.
        - Does not modify the input list or other global state beyond logging.
        - Skips malformed or unparsable SMILES instead of raising, so downstream code should account for missing molecules if some inputs are invalid.
    
    Failure modes and robustness:
        - If RDKit is not installed or Chem.MolFromSmiles is unavailable, the function will raise an ImportError or AttributeError outside the try/except scope; ensure RDKit is present.
        - Individual parsing errors are caught and logged; no single malformed SMILES will stop iteration over the remaining entries.
        - The generator yields only non-None RDKit RDMol objects; users should check for expected count or empty outputs if many inputs are invalid.
    
    Returns:
        generator: A generator that yields RDKit RDMol objects corresponding to successfully parsed SMILES strings from smiles_list. The caller can iterate over the generator or convert it to a list to obtain all parsed molecules at once.
    """
    from tdc.chem_utils.evaluator import get_mols
    return get_mols(smiles_list)


################################################################################
# Source: tdc.chem_utils.evaluator.kl_divergence
# File: tdc/chem_utils/evaluator.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_chem_utils_evaluator_kl_divergence(
    generated_smiles_lst: list,
    training_smiles_lst: list
):
    """tdc.chem_utils.evaluator.kl_divergence evaluates how closely a set of generated molecules matches the distribution of a reference training set using averaged Kullback–Leibler (KL) divergences computed on a fixed set of physicochemical descriptors and an internal pairwise similarity measure. In the Therapeutics Data Commons (TDC) generation and benchmarking context, this function is used to quantify distributional similarity between molecules produced by a generative model (or oracle) and molecules in a training dataset; the resulting scalar is useful for model evaluation, benchmark reporting, and leaderboard comparisons where preservation of training distribution statistics is desired.
    
    Args:
        generated_smiles_lst (list): list (of SMILES string): SMILES strings produced by a generative method. Each entry is canonicalized via RDKit's Chem.MolFromSmiles and Chem.MolToSmiles; invalid SMILES that cannot be parsed by RDKit are filtered out before any calculations. These SMILES represent the sample distribution to be compared against the training/reference distribution.
        training_smiles_lst (list): list (of SMILES string): SMILES strings from the training dataset that serve as the reference distribution. These SMILES are canonicalized in the same way as generated_smiles_lst and invalid entries are filtered out.
    
    Detailed behavior:
        The function computes KL divergences across a fixed subset of nine physicochemical descriptors (pc_descriptor_subset): "BertzCT", "MolLogP", "MolWt", "TPSA", "NumHAcceptors", "NumHDonors", "NumRotatableBonds", "NumAliphaticRings", and "NumAromaticRings". The first four descriptors in this list are treated as continuous-valued and are compared with a continuous KL divergence routine (continuous_kldiv). The remaining five descriptors are treated as discrete/integer-valued and are compared with a discrete KL divergence routine (discrete_kldiv). Descriptor matrices for both generated and training molecules are computed by calculate_pc_descriptors(list_of_smiles, pc_descriptor_subset) (these helper functions are part of the same TDC chemistry utilities and must be available in the runtime environment).
    
        In addition to descriptor-wise KL divergences, the function computes an internal pairwise similarity measure for each set (calculate_internal_pairwise_similarities), reduces each pairwise matrix to per-molecule maxima (max over rows), and computes a continuous KL divergence between the two resulting distributions of maximum similarities. That internal similarity KL divergence is included with the descriptor-based KL divergences.
    
        Each individual KL divergence value is transformed to a bounded similarity-like score via np.exp(-kldiv) so that individual transformed values lie in [0, 1] (np.exp(0) == 1 for identical distributions). The function returns the arithmetic mean of these transformed values across all computed divergences (descriptor-wise and internal similarity), producing a single float summary score in [0, 1] where values closer to 1 indicate closer alignment between generated and training distributions according to the chosen descriptors and similarity measure.
    
    Side effects and defaults:
        The function canonicalizes all input SMILES using RDKit; invalid SMILES are silently removed (filtered out) before descriptor computation. The descriptor subset and the choice of which indices are continuous versus discrete are fixed in the implementation (as listed above) and cannot be changed through this function's arguments. No external state is modified by this function.
    
    Failure modes and edge cases:
        If RDKit is not available or Chem.MolFromSmiles / Chem.MolToSmiles raise exceptions, this function will propagate those exceptions. If all SMILES in either input list are invalid and are filtered out, downstream descriptor- or similarity-calculation functions may raise errors (for example due to empty arrays) or produce shapes that cause exceptions; callers should ensure at least one valid SMILES remains in each list. If any of the helper functions calculate_pc_descriptors, continuous_kldiv, discrete_kldiv, or calculate_internal_pairwise_similarities are unavailable or raise errors, those exceptions will propagate. The implementation contains a commented-out section that attempted cross-set similarity KL calculations but that code path is inactive; identical input sets may still be handled, but certain previously attempted computations were disabled due to numerical issues.
    
    Returns:
        float: A single aggregated score in [0, 1] representing the averaged, transformed KL divergences over the selected physicochemical descriptors and internal pairwise similarity. A value of 1.0 indicates perfect agreement (KL divergence 0 for all compared distributions after transformation), and values closer to 0 indicate larger distributional discrepancies.
    """
    from tdc.chem_utils.evaluator import kl_divergence
    return kl_divergence(generated_smiles_lst, training_smiles_lst)


################################################################################
# Source: tdc.chem_utils.evaluator.novelty
# File: tdc/chem_utils/evaluator.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_chem_utils_evaluator_novelty(generated_smiles_lst: list, training_smiles_lst: list):
    """tdc.chem_utils.evaluator.novelty evaluates the novelty of a set of generated SMILES strings relative to a reference training set. In the Therapeutics Data Commons (TDC) context, this function quantifies how many generated small-molecule candidates are new compared to molecules seen during training, which is a common metric when assessing molecule generation oracles and generative models for drug discovery.
    
    Args:
        generated_smiles_lst (list): List of SMILES strings produced by a generative model or oracle. This argument is the set under evaluation; the function first passes it through unique_lst_of_smiles to remove duplicate entries (unique_lst_of_smiles is expected to return a list of SMILES strings). The practical role of this parameter is to represent the candidate molecules whose novelty relative to the training data is being measured.
        training_smiles_lst (list): List of SMILES strings that were used to train the generative model or that represent the known reference corpus. This list is also passed through unique_lst_of_smiles to remove duplicates before comparison. In the TDC workflow, this represents the known chemical space against which generated molecules are compared to assess novelty.
    
    Returns:
        float: A novelty score in the closed interval [0.0, 1.0]. The score equals the fraction of unique generated molecules that do not appear in the (deduplicated) training set. A value of 1.0 indicates that none of the generated molecules are present in the training set (maximal novelty), and 0.0 indicates all generated molecules are present in the training set (no novelty). Internally, the function computes 1 - (count_present / total_generated) where count_present is the number of generated SMILES found in the training list after deduplication.
    
    Behavior, side effects, and failure modes:
        The function calls unique_lst_of_smiles on both inputs to remove duplicates; unique_lst_of_smiles must return a list of SMILES strings. Comparisons between generated and training molecules are performed using Python string membership (the in operator) on SMILES strings, so results are sensitive to SMILES formatting/canonicalization—differences in tautomers, stereochemical notation, or canonicalization status will affect membership checks. There are no external side effects or modifications to global state other than creating local deduplicated lists.
        If the deduplicated generated_smiles_lst is empty, the function will attempt to divide by zero and raise a ZeroDivisionError; callers should ensure the generated list is non-empty (or handle this exception) before invoking the function. The function does not perform additional validation of SMILES syntax; invalid SMILES are treated as ordinary strings for membership comparison. Performance depends on list sizes because membership checks are performed on Python lists (O(n*m) behavior in the worst case).
    """
    from tdc.chem_utils.evaluator import novelty
    return novelty(generated_smiles_lst, training_smiles_lst)


################################################################################
# Source: tdc.chem_utils.evaluator.single_molecule_validity
# File: tdc/chem_utils/evaluator.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_chem_utils_evaluator_single_molecule_validity(smiles: str):
    """Evaluate whether a single SMILES string corresponds to a chemically parsable molecule. This function is used throughout TDC to screen individual molecular inputs (for example in dataset loading, data processing, molecule generation oracles, and evaluation pipelines) and to ensure that downstream routines receive SMILES that can be converted to a molecular object for feature calculation or model scoring.
    
    Args:
        smiles (str): A SMILES (Simplified Molecular Input Line Entry System) string representing a single molecule. In the therapeutics and cheminformatics context used by TDC, this argument is the textual molecular representation produced by molecule generators, dataset files, or user input. The function strips leading and trailing whitespace from this string before parsing; therefore, common accidental whitespace does not affect validity.
    
    Returns:
        bool: True if the input SMILES can be parsed into a molecule object and that parsed molecule contains at least one atom; False otherwise. Practically, a True return indicates the SMILES is suitable for downstream TDC workflows (feature extraction, oracle scoring, training/evaluation), while False indicates the SMILES should be rejected or logged and excluded from cheminformatics processing.
    
    Behavior and failure modes:
        The function attempts to parse the provided SMILES using an underlying SMILES-to-molecule parser (as invoked by Chem.MolFromSmiles in the implementation). It returns False if the trimmed input is an empty string, if the parser returns no molecule (parser failure), or if the parsed molecule has zero atoms. The function does not perform chemical sanitization beyond parsing and atom-count checking: it does not validate formal valence consistency, stereochemical completeness, tautomer standardization, or other domain-specific chemical correctness checks. The function is effectively side-effect free (it does not modify global state or the input); however, if an argument that is not a str is passed, attribute access (for example the call to .strip()) may raise an exception (TypeError or AttributeError) depending on the object type.
    """
    from tdc.chem_utils.evaluator import single_molecule_validity
    return single_molecule_validity(smiles)


################################################################################
# Source: tdc.chem_utils.evaluator.uniqueness
# File: tdc/chem_utils/evaluator.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_chem_utils_evaluator_uniqueness(list_of_smiles: list):
    """tdc.chem_utils.evaluator.uniqueness evaluates the uniqueness of a collection of SMILES strings by canonicalizing and deduplicating them and returning the fraction of distinct molecules present. This function is typically used within the Therapeutics Data Commons (TDC) workflow to quantify molecular diversity of outputs from molecule generation oracles, distribution-learning models, and dataset curation steps; a higher value indicates greater diversity among the provided SMILES.
    
    Args:
        list_of_smiles (list): A Python list of SMILES strings (each element is expected to be a SMILES string representing a molecule). The function passes this list to unique_lst_of_smiles to canonicalize representations and remove duplicates; therefore different SMILES that represent the same chemical structure are treated as identical after canonicalization. The caller must provide an iterable list object; non-list inputs or list elements that are not valid SMILES strings or not handled by unique_lst_of_smiles may cause downstream errors. The function does not modify the input list object in place.
    
    Returns:
        float: The uniqueness score computed as len(canonical_smiles_lst) / len(list_of_smiles), where canonical_smiles_lst is the list returned by unique_lst_of_smiles(list_of_smiles). This value is the fraction of unique canonicalized molecules in the input and will be in the range [0.0, 1.0] for non-empty inputs (0.0 means no unique molecules, 1.0 means all input SMILES are unique after canonicalization). If list_of_smiles is empty, the implementation performs division by len(list_of_smiles) and will raise a ZeroDivisionError; callers should guard against empty inputs when necessary.
    
    Behavior and side effects:
        The function calls unique_lst_of_smiles to perform canonicalization and deduplication and then computes the ratio of unique entries to total entries. It has no external side effects (it does not write files or modify external state) and is deterministic given a deterministic unique_lst_of_smiles implementation. Performance and memory use scale with the length of list_of_smiles and with the cost of canonicalizing each SMILES string.
    
    Failure modes:
        Passing an empty list_of_smiles will result in a ZeroDivisionError due to division by zero. Passing a non-list object as list_of_smiles or list elements that are invalid or incompatible with unique_lst_of_smiles may raise exceptions originating from unique_lst_of_smiles (such as parsing or type errors).
    """
    from tdc.chem_utils.evaluator import uniqueness
    return uniqueness(list_of_smiles)


################################################################################
# Source: tdc.chem_utils.featurize._xyz2mol.AC2BO
# File: tdc/chem_utils/featurize/_xyz2mol.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_chem_utils_featurize__xyz2mol_AC2BO(
    AC: numpy.ndarray,
    atoms: list,
    charge: int,
    allow_charged_fragments: bool = True,
    use_graph: bool = True
):
    """AC2BO converts an atom connectivity/count matrix (AC) and per-atom atomic numbers into a chemically plausible bond-order (BO) matrix using the valence-assignment algorithm implemented in this module (referred to in the code as "implementation of algorithm shown in Figure 2"). It is used in the TDC chemistry featurization pipeline to infer bond orders and valence-electron information from pairwise connectivity counts produced when converting 3D coordinate data into a molecular graph (the surrounding module is responsible for interpreting .xyz-like inputs and producing AC). The function implements the algorithmic steps referenced in the source: building per-atom candidate valences, computing unsaturated atoms (UA) and degree of unsaturation (DU), enumerating UA pairings, proposing BO matrices, and validating them against valence and total charge constraints. The returned BO matrix and the module-level atomic_valence_electrons structure are used downstream to construct molecular graphs and features for machine-learning tasks in TDC (for example, producing graph inputs for small-molecule predictive tasks and generation oracles).
    
    Args:
        AC (numpy.ndarray): Square adjacency/count matrix for atoms in the candidate molecule. AC[i, j] contains a nonnegative integer count representing the connectivity between atom i and atom j as derived by the upstream geometry-to-connectivity routine. This matrix is summed across axis=1 in the algorithm to obtain per-atom connectivity counts (the algorithm treats these sums as the current observed valence demand). AC is expected to be consistent with the atoms list (same ordering and length). The function uses AC to generate candidate bond-order matrices and to compute degree-of-unsaturation (DU) and unsaturated-atom (UA) sets.
        atoms (list): A list of integer atomic numbers (one entry per atom, in the same order as rows/columns of AC). This list identifies the chemical element of each atom and is used to look up allowable valences from the module-level atomic_valence global. It is essential for determining which valence assignments are legal for each atom and thus for proposing and validating bond orders consistent with chemical valence rules used by TDC featurizers.
        charge (int): Net integer charge of the entire molecule (sum of formal charges). The algorithm enforces that any proposed bond-order assignment is compatible with this global molecular charge via internal validation routines (e.g., charge_is_OK and BO_is_OK). This parameter is required to detect and allow or disallow charged fragments depending on allow_charged_fragments, and to prefer BO solutions that satisfy the charge constraint.
        allow_charged_fragments (bool): If True (default), permit intermediate or final bond-order assignments that produce charged fragments when checking solutions; if False, reject assignments that fragment the molecule into charged components. This flag alters the validation performed by BO_is_OK and charge_is_OK and therefore affects which BO solution is accepted. In TDC's featurization context, toggling this changes whether the function may return BO matrices representing formally charged fragments (useful when modeling ionic species versus neutral molecules).
        use_graph (bool): If True (default), use graph-based heuristics when generating candidate UA pairings and when computing BO proposals (the function calls get_UA_pairs and get_BO with use_graph forwarded). If False, the pairing and BO proposals use alternate, non-graph heuristics. Practically, this flag controls whether structural/graph connectivity information is exploited to choose UA pairings and can affect both runtime and the chemical plausibility of proposed BO matrices.
    
    Behavior, side effects, defaults, and failure modes:
        The function enumerates all combinations of allowable integer valences per atom (derived from the module-level atomic_valence table) that are at least as large as each atom's current observed connectivity (sum of its AC row). For each valence combination it computes unsaturated atoms (UA) and degree of unsaturation (DU) using get_UA. If no UA remain for a valence choice, it directly validates the original AC as a BO candidate via BO_is_OK. Otherwise it enumerates UA pairings via get_UA_pairs (optionally using graph heuristics controlled by use_graph), proposes bond-order matrices with get_BO, and validates them with BO_is_OK and charge_is_OK. The algorithm maintains a running best_BO (initialized to AC) and may return immediately upon finding a BO that passes validation; otherwise, it returns the best candidate found according to the heuristic sum-of-entries comparison while ensuring valences are not exceeded and charge compatibility is preserved.
        Side effects: The function reads and uses module-level globals atomic_valence and atomic_valence_electrons. It may print diagnostic messages via the module's print_sys helper and may call sys.exit() and terminate the process if it detects an impossible per-atom valence situation (when no allowable valences are >= the observed connectivity for an atom). No files are written.
        Defaults: allow_charged_fragments defaults to True and use_graph defaults to True, reflecting the typical behavior in TDC featurizers to allow charged fragments and to exploit graph structure when generating BO proposals.
        Failure modes: If the per-atom observed connectivity (row-sum of AC) exceeds the maximum allowed valence for a given atomic number (lookup in atomic_valence), the function prints an explanatory message and calls sys.exit(), terminating execution. If no fully valid BO solution is found for any enumerated valences, the function returns the best_BO candidate selected by internal heuristics (the candidate with the largest summed bond-orders meeting valence bounds and charge compatibility), which may not fully satisfy all chemical constraints; callers should therefore verify BO validity if strict correctness is required. The function assumes AC and atoms are consistent in length and ordering; passing mismatched inputs can lead to incorrect behavior or errors.
    
    Returns:
        tuple: A pair (BO_matrix, atomic_valence_electrons_out) where BO_matrix is a numpy.ndarray representing the chosen bond-order matrix (same shape as AC) and atomic_valence_electrons_out is the module-level atomic_valence_electrons structure that the algorithm uses to validate electron/valence counts. BO_matrix is the first valid bond-order assignment discovered that satisfies the validation routines (BO_is_OK and charge_is_OK), or, if no fully valid assignment is found, the best heuristic candidate (best_BO) selected by the algorithm. The atomic_valence_electrons_out return value mirrors the internal global used for per-element valence-electron information and is returned so downstream code in the TDC featurizer can use the exact electron-count mapping employed during validation.
    """
    from tdc.chem_utils.featurize._xyz2mol import AC2BO
    return AC2BO(AC, atoms, charge, allow_charged_fragments, use_graph)


################################################################################
# Source: tdc.chem_utils.featurize._xyz2mol.BO_is_OK
# File: tdc/chem_utils/featurize/_xyz2mol.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_chem_utils_featurize__xyz2mol_BO_is_OK(
    BO: numpy.ndarray,
    AC: numpy.ndarray,
    charge: int,
    DU: list,
    atomic_valence_electrons: list,
    atoms: list,
    valences: list,
    allow_charged_fragments: bool = True
):
    """Sanity check for bond orders when converting Cartesian/XYZ-derived connectivity into a molecular graph. This function is used in the tdc.chem_utils.featurize._xyz2mol pipeline to validate whether a proposed bond-order matrix and related per-atom data are chemically consistent with integer total charge, per-atom valence limits, and the expected extra bond orders (DU). In the Therapeutics Data Commons (TDC) context this check prevents creating chemically impossible small-molecule graphs from XYZ-derived connectivity during featurization and ensures downstream featurizers and ML models receive chemically plausible molecular graphs.
    
    Args:
        BO (numpy.ndarray): Bond-order matrix proposed for the molecule. BO is a square numpy array where entry BO[i, j] encodes the bond order between atom i and j (for example 0 for no bond, 1 for single, 2 for double, etc.). BO is used to validate per-atom valence counts and to compare against AC and DU to ensure the total number of extra bond orders equals the provided DU specification.
        AC (numpy.ndarray): Adjacency/contact matrix or integer matrix representing expected single-bond connectivity for the same atom ordering as BO. AC is a square numpy array with the same shape as BO; it typically encodes the connectivity baseline (e.g., single-bond counts) used together with BO to compute differences that must match DU. The function computes (BO - AC).sum() and compares it to sum(DU) as part of the validation.
        charge (int): Integer total molecular charge for the proposed molecule. This scalar is used to validate whether the distribution of electrons implied by BO, AC, atomic_valence_electrons, atoms, and valences is consistent with the given net charge via the internal charge_is_OK check. In TDC’s featurization pipeline, an incorrect charge will mark the candidate molecule invalid to avoid producing charged fragments unintentionally.
        DU (list): List of integers representing per-atom or per-structure "extra bond order" counts used by the algorithm to account for double bonds, rings, or unsaturation constraints in the conversion from XYZ-derived connectivity to bond orders. DU is summed and compared to the summed difference between BO and AC (i.e., (BO - AC).sum()). DU therefore plays a practical role in ensuring the proposed bond orders supply the expected additional bonding beyond the adjacency baseline.
        atomic_valence_electrons (list): List of integers giving the number of valence electrons for each atom in the same atom ordering as BO and AC. This data is consumed by the internal charge_is_OK routine to check electron accounting against the provided net charge and the proposed bond orders; accurate valence-electron counts are required for correct charge validation in the featurization step.
        atoms (list): List describing each atom in the molecule in the same order as BO/AC. In the XYZ-to-molecule context this is typically a list of atomic identifiers (e.g., atomic numbers or element symbols) used by charge_is_OK to reason about expected valences and electron counts per element. The entries must align with atomic_valence_electrons and valences positions.
        valences (list): List of allowed valence counts for each atom (same ordering as BO). These per-atom valence limits are used first by valences_not_too_large(BO, valences) to ensure no atom exceeds its permitted valence given the proposed BO; this prevents creation of chemically impossible bonding patterns during featurization.
        allow_charged_fragments (bool): Flag (default True) that controls whether charge_is_OK should permit charged fragments as part of the validation. When True (the default), the charge validation permits fragment-level charges that still satisfy the global charge accounting; when False, the routine is stricter and may reject configurations that produce separated charged fragments. Set this to False if fragment neutrality is required for downstream processing in TDC pipelines.
    
    Returns:
        bool: True if the proposed bond-order matrix and associated per-atom information pass the chemical sanity checks, meaning (1) no atom exceeds its allowed valence (checked via valences_not_too_large), (2) the sum of bond-order differences (BO - AC) equals sum(DU), and (3) the electronic/charge accounting is consistent with the provided net charge (checked via charge_is_OK). Returns False otherwise. There are no side effects: this function does not modify its inputs. Note that malformed inputs (for example BO and AC of incompatible shapes, non-numeric entries where numeric arrays are expected, or mismatched list lengths among atomic_valence_electrons, atoms, and valences) will typically raise Python or NumPy exceptions upstream of the boolean return rather than being handled internally; callers should ensure consistent shapes and element ordering before invoking this check.
    """
    from tdc.chem_utils.featurize._xyz2mol import BO_is_OK
    return BO_is_OK(
        BO,
        AC,
        charge,
        DU,
        atomic_valence_electrons,
        atoms,
        valences,
        allow_charged_fragments
    )


################################################################################
# Source: tdc.chem_utils.featurize._xyz2mol.int_atom
# File: tdc/chem_utils/featurize/_xyz2mol.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_chem_utils_featurize__xyz2mol_int_atom(atom: str):
    """tdc.chem_utils.featurize._xyz2mol.int_atom maps a chemical element symbol string to a 1-based integer index used by the XYZ-to-molecule featurization utilities in TDC. This function is used in the pipeline that converts atomic coordinates and element labels (from XYZ files or similar sources) into integer-coded atom features required by downstream molecule construction and machine-learning featurizers.
    
    Args:
        atom (str): The atomic symbol or name for a single atom (for example 'C', 'H', 'O', or full names if present). The function lowercases this input before lookup, so the caller may provide uppercase, lowercase, or mixed-case element symbols; the value must be a Python str. The mapping is performed against the global list __ATOM_LIST__ (expected to contain lowercase element identifiers). There are no defaults; the caller must supply a valid element string.
    
    Returns:
        int: A 1-based integer index corresponding to the position of the atom string in the global __ATOM_LIST__. The function finds the zero-based index of the lowercased atom in __ATOM_LIST__ and returns index + 1. This 1-based convention is used by the XYZ-to-molecule featurizer to represent atom types in feature vectors and encoded molecule representations.
    
    Raises:
        ValueError: If the lowercased atom string is not present in __ATOM_LIST__, str.index(...) will raise ValueError indicating the atom is unknown to the featurizer mapping.
        NameError: If the global name __ATOM_LIST__ is not defined in the module namespace, attempting to access it will raise NameError.
        TypeError: If a non-str value is passed for atom, lowercasing (atom.lower()) may raise AttributeError; callers must pass a str to avoid type-related errors.
    
    Behavior and side effects:
        The function performs no mutations; it only reads the global __ATOM_LIST__ and returns an integer. It forces the input to lowercase before lookup to make matching case-insensitive relative to the expected lowercase entries in __ATOM_LIST__. It does not validate that the returned index corresponds to a chemically valid atomic number; it only encodes positions in __ATOM_LIST__. Consumers should ensure __ATOM_LIST__ is populated with the intended ordering and identifiers before calling int_atom.
    """
    from tdc.chem_utils.featurize._xyz2mol import int_atom
    return int_atom(atom)


################################################################################
# Source: tdc.chem_utils.featurize._xyz2mol.read_xyz_file
# File: tdc/chem_utils/featurize/_xyz2mol.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_chem_utils_featurize__xyz2mol_read_xyz_file(
    filename: str,
    look_for_charge: bool = True
):
    """Reads a molecular geometry in XYZ file format and returns atomic numbers, formal charge, and 3D coordinates suitable for downstream featurization and molecule-reconstruction workflows in the TDC chem_utils.featurize pipeline (adapted from the xyz2mol project).
    
    Args:
        filename (str): Path to an input XYZ-format file to read. The function expects the file to follow the conventional XYZ structure used by many molecular geometry tools and the xyz2mol converter: the first line is an integer giving the declared number of atoms, the second line is a title/comment line (this line may contain a substring "charge=<int>" that encodes the overall formal charge), and each subsequent line contains an atomic symbol followed by three Cartesian coordinates separated by whitespace (for example: "C 0.000 0.000 0.000"). This file is opened for reading using the platform default encoding; passing an invalid path raises FileNotFoundError. The contents are parsed and returned for use in TDC featurizers, molecule-graph builders, or oracle evaluations that require atomic identity and 3D positions.
        look_for_charge (bool): Flag intended to control whether the function inspects the title/comment line for a "charge=" token and, if present, parse the integer value after the equals sign into the returned charge. In this implementation the parameter is accepted and defaults to True, but the current source inspects the second (title) line for "charge=" regardless of this flag (i.e., the flag has no effect). If the title line contains "charge=<int>" the integer is parsed and returned as the charge; otherwise the returned charge defaults to 0. Note that the parsing expects an integer immediately following '=' and will raise ValueError if that substring cannot be converted to int.
    
    Returns:
        tuple:
            atoms (list[int]): A list of integers obtained by converting each parsed atomic symbol (string) into its integer atomic identifier via the module-local int_atom function. Each integer represents an element's atomic number and is intended for use in featurizers and graph constructors that require numeric atom identifiers. If int_atom is not defined or fails for a symbol, a NameError or ValueError may be raised.
            charge (int): The overall molecular formal charge parsed from the second line if the title contains "charge=<int>", or 0 otherwise. This integer is important for downstream chemistry tools and featurizers that need to account for net charge when building molecular graphs, computing partial charges, or reconciling with other data sources.
            xyz_coordinates (list[list[float]]): A list of 3-element lists, each containing floats [x, y, z] for an atom in the same order as atoms. These Cartesian coordinates are used by geometry-aware featurizers, distance-based descriptors, and any reconstruction of 3D molecular structure. Each coordinate is parsed with float(); malformed numeric fields will raise ValueError.
    
    Behavioral details, side effects, and failure modes:
        - The function opens and reads the file specified by filename; it does not write to disk or change global state. If the file does not exist, a FileNotFoundError is raised by open().
        - The first line is parsed as an integer (num_atoms) but this parsed value is not validated against the actual number of atom lines read; a mismatch will not itself raise an error but may indicate a malformed file.
        - The second line is stored as title and inspected for the substring "charge="; if present the function attempts to parse the integer after '=' as the molecular charge. If parsing fails, a ValueError will be raised.
        - Each remaining non-header line is expected to contain exactly four whitespace-separated tokens: an atomic symbol and three numeric coordinate strings. Lines that do not split into four tokens will raise ValueError or IndexError when unpacking; coordinate tokens that cannot be converted to float will raise ValueError.
        - The function converts atomic symbols to integers by calling int_atom(atom). The int_atom function must be available in the same module or imported; otherwise NameError will occur. The function does not validate chemical consistency beyond this conversion.
        - Whitespace and trailing newline characters are handled via str.split(); comments beyond the expected format will likely cause parsing errors.
        - The look_for_charge parameter exists for API compatibility but, in the current source, the code always inspects the title line for "charge=" regardless of the flag value; users requiring different behavior should preprocess the file or modify the code.
    
    Practical significance:
        This reader is used in the TDC chemistry featurization pipeline to ingest simple XYZ geometry files produced by quantum chemistry packages, molecular editors, or the xyz2mol conversion utilities. The outputs (atomic numbers, charge, and coordinates) are core inputs to downstream TDC functions that build molecular graphs, compute geometry-dependent descriptors, or feed 3D-aware oracles and models used in therapeutic ML tasks (for example, property prediction, docking-aware scoring, or generative model evaluation).
    """
    from tdc.chem_utils.featurize._xyz2mol import read_xyz_file
    return read_xyz_file(filename, look_for_charge)


################################################################################
# Source: tdc.chem_utils.featurize._xyz2mol.str_atom
# File: tdc/chem_utils/featurize/_xyz2mol.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_chem_utils_featurize__xyz2mol_str_atom(atom: int):
    """tdc.chem_utils.featurize._xyz2mol.str_atom converts a 1-based integer atom identifier into a string atom label by indexing the module-level atom list. This function is used in the XYZ-to-molecule featurization pipeline within TDC to map numeric atom identifiers (for example, those parsed from XYZ coordinate files or simple integer atom codes produced during parsing) to their canonical string representations (such as element symbols) stored in the module's global __ATOM_LIST__. The returned string is intended for downstream featurizers and molecular feature extraction that require element labels (e.g., building atom feature vectors, constructing molecular graphs, or generating SMILES-compatible element tokens) in TDC's small-molecule and molecular featurization utilities.
    
    Args:
        atom (int): A 1-based integer index that identifies an atom in the module-global __ATOM_LIST__. The integer value is used as an index into __ATOM_LIST__ after subtracting one (i.e., __ATOM_LIST__[atom - 1]). In the TDC featurization context, this represents the numeric atom identifier produced by XYZ file parsing or other routines that encode atoms as integers. The function performs no type conversion; callers must supply an int. No default is provided.
    
    Returns:
        str: The string label retrieved from __ATOM_LIST__ corresponding to the provided 1-based index (the element or atom symbol used by downstream featurizers). The return value is a direct reference to the list element and is used by molecular feature construction code in tdc.chem_utils.featurize._xyz2mol.
    
    Notes on behavior and failure modes:
        - The function reads the module-level global variable __ATOM_LIST__ and does not modify it. If __ATOM_LIST__ is not defined in the module namespace at call time, a NameError will be raised.
        - If atom is not an int, Python will typically raise a TypeError when it is used to index the list; callers should ensure the argument is an integer.
        - If atom is an int but is outside the valid 1-based range for __ATOM_LIST__ (for example, less than 1 or greater than len(__ATOM_LIST__)), an IndexError will be raised due to the out-of-range list access.
        - The function assumes the mapping semantics and contents of __ATOM_LIST__ are appropriate for the featurization pipeline; it does not validate that the returned string is a valid chemical element symbol.
    """
    from tdc.chem_utils.featurize._xyz2mol import str_atom
    return str_atom(atom)


################################################################################
# Source: tdc.chem_utils.featurize._xyz2mol.xyz2AC
# File: tdc/chem_utils/featurize/_xyz2mol.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_chem_utils_featurize__xyz2mol_xyz2AC(
    atoms: list,
    xyz: numpy.ndarray,
    charge: int,
    use_huckel: bool = False
):
    """Convert a list of atom types and their 3D coordinates into an atom connectivity (AC) matrix and an RDKit molecule object used by TDC featurization pipelines.
    
    This function is used in therapeutic-molecule featurization within TDC to transform raw molecular geometry (atom identifiers and Cartesian coordinates) into two representations required by downstream workflows: (1) an atom connectivity matrix that encodes which atoms are bonded (used by graph-based featurizers and split/evaluation procedures) and (2) an RDKit molecule object (rdkit.Chem.rdchem.Mol) that can be used for chemistry-aware operations, canonicalization, and integration with other RDKit-based tools. The function dispatches to one of two connectivity-inference implementations: a Huckel-based method when use_huckel is True, and a van-der-Waals distance heuristic when use_huckel is False (the default). It does not perform file I/O; it constructs in-memory objects for immediate use by dataset processing, model input preparation, or oracle evaluation in TDC.
    
    Args:
        atoms (list): Integer atom types for each atom in the molecule, in the same order as the coordinates provided in xyz. Each list element is an integer representing the atomic number or atom type expected by the downstream connectivity routines. This list defines which element each coordinate corresponds to and is required for correct bond inference and RDKit molecule construction.
        xyz (numpy.ndarray): Cartesian coordinates corresponding to the atoms list. This numpy array contains the 3D positions used to infer inter-atomic distances and thereby propose bonds. The order of rows (or entries) in this array must match the order of atoms in the atoms parameter. The function relies on these coordinates to compute distance-based or Huckel-based connectivity.
        charge (int): Formal total molecular charge used when constructing the RDKit molecule and when selecting bond orders in Huckel-based inference. This integer disambiguates valence and electron counts during connectivity and RDKit molecule creation and should reflect the true net charge of the molecule being processed.
        use_huckel (bool): Whether to use the Huckel-based connectivity inference method (True) or the default van der Waals distance heuristic (False). Default is False. When True, the function delegates to xyz2AC_huckel which attempts to infer bond orders and connectivity consistent with Huckel-like rules and the provided charge; when False, it delegates to xyz2AC_vdW which uses geometric (distance-based) criteria derived from atomic van der Waals radii to propose bonds. Choose True when bond order inference informed by formal electronic considerations is required; choose False for faster, purely geometric adjacency suitable for many graph-based featurizers.
    
    Returns:
        tuple:
            ac (numpy.ndarray): Atom connectivity matrix produced by the selected inference method. This matrix encodes inferred bonds between atoms (typically as adjacency entries) and is intended for use as the adjacency input to graph featurizers and ML pipelines in TDC.
            mol (rdkit.Chem.rdchem.Mol): An RDKit molecule object reconstructed from the atoms, inferred connectivity, and provided charge. This object can be used for chemistry-aware downstream processing such as canonicalization, property computation, or conversion to common molecular string formats.
    
    Behavior and failure modes:
        - The function delegates to xyz2AC_huckel if use_huckel is True and to xyz2AC_vdW otherwise; any errors raised by those helper functions propagate to the caller.
        - Inputs must be consistent: the length of atoms must match the number of coordinate entries in xyz; otherwise a ValueError or IndexError may be raised by the underlying routines.
        - RDKit must be available in the runtime environment; if RDKit is not importable, RDKit-based molecule construction will raise an ImportError propagated from the underlying implementation.
        - The function does not perform file I/O or mutate external state; it returns in-memory objects. If connectivity inference cannot produce a valid molecule, the underlying helper may raise an exception or return values consistent with its own contract.
        - Use of use_huckel may be slower and more sensitive to chemically inconsistent inputs (incorrect charge or atom typing) because it attempts to infer bond orders consistent with formal electronic structure, whereas the vdW method is purely geometric and typically faster.
    """
    from tdc.chem_utils.featurize._xyz2mol import xyz2AC
    return xyz2AC(atoms, xyz, charge, use_huckel)


################################################################################
# Source: tdc.chem_utils.featurize._xyz2mol.xyz2AC_huckel
# File: tdc/chem_utils/featurize/_xyz2mol.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_chem_utils_featurize__xyz2mol_xyz2AC_huckel(
    atomicNumList: list,
    xyz: numpy.ndarray,
    charge: int
):
    """tdc.chem_utils.featurize._xyz2mol.xyz2AC_huckel computes an atom connectivity (adjacency) matrix for a small molecule by constructing a prototype RDKit molecule from atomic numbers, placing a provided 3D conformer, and running an extended Hückel / reduced overlap population calculation (rdEHTTools) to infer bonds via an absolute pair population cutoff. This function is used in the TDC featurization pipeline to convert atomic-number + coordinate representations into a binary adjacency matrix suitable for molecular machine-learning tasks and benchmarks in TDC.
    
    Args:
        atomicNumList (list): A Python list of integer atomic numbers (Z) that defines the atom types and ordering for the molecule. The list length defines the number of atoms N. This list is passed to get_proto_mol to create an RDKit molecule containing N atoms (initially without bonds); the ordering must match the rows of the xyz array because positions are assigned by index.
        xyz (numpy.ndarray): A NumPy array of atom coordinates used as a single conformer for the molecule. Expected shape is (N, 3), where N == len(atomicNumList). Coordinates are interpreted as Cartesian coordinates and are assigned to the RDKit conformer in the same atom order as atomicNumList. A mismatch in lengths (len(atomicNumList) != xyz.shape[0]) or malformed coordinates will cause incorrect geometry or downstream failures in the Huckel calculation.
        charge (int): An integer total molecular charge passed to the internal Huckel calculation. The implementation applies this charge only to a temporary copy of the RDKit molecule (the first atom's formal charge is set to charge in that copy) prior to running rdEHTTools.RunMol; the returned RDKit mol preserves the original atom formal charges (i.e., the charge is not permanently applied to the returned mol). The choice to apply the full charge to atom index 0 is arbitrary and therefore affects the Huckel-derived connectivity.
    
    Returns:
        ac (numpy.ndarray): An N x N NumPy array of integers (dtype int) representing the inferred atom connectivity/adjacency matrix. ac[i, j] == 1 indicates a predicted bond between atoms i and j inferred by taking the absolute value of the reduced overlap population for the pair and applying a threshold of 0.15 (an arbitrary cutoff chosen in this implementation). The matrix is symmetric, has zeros on the diagonal, and is intended as a binary adjacency for downstream featurization in TDC.
        mol (rdkit.Chem.Mol): An RDKit Mol object corresponding to the prototype molecule created from atomicNumList with the provided conformer attached. This returned mol contains the conformer positions assigned from xyz but does not carry the arbitrary formal-charge modification applied to the temporary copy used by the Huckel computation.
    
    Behavior, side effects, defaults, and failure modes:
        - The function constructs a prototype RDKit molecule via get_proto_mol(atomicNumList), creates and assigns a Chem.Conformer from xyz, and uses a copy of that molecule to run rdEHTTools.RunMol to obtain the reduced overlap population matrix.
        - The Huckel run is performed on a temporary copy (mol_huckel) where the formal charge is set on atom index 0; the returned mol is the original prototype with conformer but without that forced formal-charge change.
        - A pair population absolute-value threshold of 0.15 is used to detect bonds; this cutoff is implementation-specific and may need adjustment for different chemical systems or accuracy requirements.
        - Dependencies: RDKit and rdEHTTools (and any configuration they require) must be available; if rdEHTTools.RunMol fails, raises, or returns an unexpected result (for example, missing GetReducedOverlapPopulationMatrix or a matrix with incompatible size), the function may raise an exception or produce an incorrect adjacency matrix.
        - Input validation is minimal: callers must ensure atomicNumList length matches xyz.shape[0], atomic numbers are valid integers, and xyz contains valid numeric coordinates. Invalid inputs or unrealistic geometries can lead to runtime errors or chemically implausible connectivities.
        - The method is heuristic (extended Hückel/population-based) and intended for featurization in ML workflows (as in TDC); it does not replace full quantum chemical bonding analysis and may misassign bonds in some cases.
    """
    from tdc.chem_utils.featurize._xyz2mol import xyz2AC_huckel
    return xyz2AC_huckel(atomicNumList, xyz, charge)


################################################################################
# Source: tdc.chem_utils.featurize._xyz2mol.xyz2mol
# File: tdc/chem_utils/featurize/_xyz2mol.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_chem_utils_featurize__xyz2mol_xyz2mol(
    atoms: list,
    coordinates: numpy.ndarray,
    charge: int = 0,
    allow_charged_fragments: bool = True,
    use_graph: bool = True,
    use_huckel: bool = False,
    embed_chiral: bool = True
):
    """Generate an RDKit molecule object (or multiple candidate molecules) and an associated bond-order matrix from an explicit list of atom types and their 3D Cartesian coordinates. This function is used in TDC chemical featurization workflows to convert atomic coordinates (for example, from quantum calculations or molecular mechanics) into RDKit mol objects that can be further processed for machine learning tasks such as property prediction, featurization, or molecule generation/oracle evaluation.
    
    Args:
        atoms (list): List of atom identifiers corresponding to each atom in the system. In practice these are integer atomic numbers or other atomic type encodings expected by the downstream connectivity routine (xyz2AC). The ordering must match the ordering of the coordinates array. This argument defines the element identity used to predict connectivity and chemical valence when constructing RDKit molecules.
        coordinates (numpy.ndarray): A NumPy array of 3 x N Cartesian coordinates (three rows by N columns) describing the 3D positions of the N atoms listed in atoms. The coordinate geometry is used to infer inter-atomic connectivity via distance-/geometry-based heuristics (or Huckel connectivity if use_huckel is True). Supplying coordinates with an incompatible shape or ordering will cause downstream failures in connectivity prediction.
        charge (int): Total formal charge of the entire molecular system (default: 0). This total charge guides how the connectivity and formal charges are assigned to atoms and fragments during AC2mol. The charge parameter is necessary for correct valence and formal charge assignment in the generated RDKit molecules.
        allow_charged_fragments (bool): If True (default), AC2mol may produce molecular fragments that carry formal charge(s) when necessary to satisfy valence and total charge constraints. If False, the function will prefer to produce radical species (unpaired electrons) instead of isolated charged fragments; this changes the chemical forms returned and can affect downstream featurization and validity checks.
        use_graph (bool): If True (default), use a graph-based approach (via networkx in AC2mol) when converting the atom connectivity (AC) matrix into RDKit molecules and bond orders. Using the graph mode can change how ambiguous connectivity is resolved into discrete bonding patterns; set to False to avoid the graph-based post-processing behavior.
        use_huckel (bool): If True (default False), use a Huckel-based method inside xyz2AC for predicting atom connectivity from coordinates and atomic types. The Huckel option changes the heuristics used to infer bonds from geometry and can produce different AC matrices (and thus different returned molecules); when False, a distance/valence-based heuristic is used.
        embed_chiral (bool): If True (default), run stereochemistry embedding/verification (via chiral_stereo_check) on each RDKit molecule produced. This may annotate or mutate the RDKit mol objects to explicitly set stereocenters. If False, stereochemical embedding is skipped and stereochemistry information may be incomplete.
    
    Returns:
        tuple: A pair (mols, BO) returned on success.
            mols (list): A list of RDKit Mol objects (one or more candidate molecules) constructed from the input atoms, coordinates, and total charge. Multiple molecules can be returned when connectivity is ambiguous or when AC2mol generates alternative bonding assignments or disconnected fragments. Each RDKit Mol is suitable for downstream TDC featurization functions and evaluators that expect rdkit molecule objects.
            BO (numpy.ndarray): The bond-order matrix produced alongside the molecules that encodes the assigned bond orders between atom indices. The BO matrix corresponds to the connectivity used to construct the returned RDKit Mol objects and can be used for downstream analyses, validation, or as an explicit numeric feature.
    
    Behavior and side effects:
        This function first calls xyz2AC(atoms, coordinates, charge, use_huckel=use_huckel) to compute an atom connectivity (AC) matrix and a minimal RDKit Mol shell. It then calls AC2mol(mol, AC, atoms, charge, allow_charged_fragments=allow_charged_fragments, use_graph=use_graph) to convert the AC matrix into discrete RDKit Mol object(s) and a bond-order (BO) matrix. If embed_chiral is True, chiral_stereo_check is invoked on each resulting RDKit Mol and may mutate those Mol objects to set explicit stereochemistry. The returned RDKit molecules are therefore potentially modified in-place by the chiral embedding step.
    
    Failure modes and notes for users:
        - Invalid input shapes or types (for example, coordinates not a 3 x N numpy.ndarray or atoms length not matching N) will typically cause exceptions raised by xyz2AC or subsequent routines.
        - If RDKit is not available or if the downstream helper functions (xyz2AC, AC2mol, chiral_stereo_check) raise errors, this function will propagate those exceptions.
        - The function can return multiple candidate molecules when bonding is ambiguous; callers should handle lists of molecules accordingly (e.g., by selecting the chemically valid/lowest-energy candidate for downstream ML tasks).
        - The chemical realism of the returned molecules depends on heuristics (distance/valence rules or Huckel method) and provided total charge; verify outcomes when using this function for benchmarking or data generation in TDC workflows.
    """
    from tdc.chem_utils.featurize._xyz2mol import xyz2mol
    return xyz2mol(
        atoms,
        coordinates,
        charge,
        allow_charged_fragments,
        use_graph,
        use_huckel,
        embed_chiral
    )


################################################################################
# Source: tdc.chem_utils.featurize.molconvert.atom2onehot
# File: tdc/chem_utils/featurize/molconvert.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_chem_utils_featurize_molconvert_atom2onehot(atom: str):
    """Convert a chemical atom symbol to a one-hot encoded feature vector used by the TDC molecular featurization pipeline.
    
    This function is used by tdc.chem_utils.featurize.molconvert to turn a single atom symbol (for example, 'C' for carbon or 'Cl' for chlorine) into a numeric feature suitable for machine learning models used in TDC small-molecule tasks (e.g., ADME prediction, molecular generation oracles). The one-hot vector has a single 1 at the index corresponding to the atom in the module-level atom_types list and 0s elsewhere. The produced feature is a 2-D NumPy array shaped (1, N) so it can be concatenated or batched consistently with node/atom feature matrices used across TDC data functions and model input pipelines.
    
    Args:
        atom (str): A single atom symbol string that must match exactly one entry in the module-level atom_types list used by molconvert. The string represents the chemical element label (for example 'C', 'N', 'O', 'Cl') and determines which position in the one-hot vector is set to 1. This parameter is required; no default is provided. The function expects an exact match on the atom_types entries (case- and spelling-sensitive) because the index is obtained via atom_types.index(atom).
    
    Returns:
        numpy.ndarray: A 2-D NumPy array of shape (1, len(atom_types)) and dtype float64 (NumPy default) where all entries are 0 except a single 1 at the column index corresponding to the provided atom. This array is intended as an atom-level feature vector for downstream machine-learning workflows in TDC.
    
    Behavior, side effects, and failure modes:
        The function constructs the one-hot vector by creating a NumPy zeros array and setting the element at the index returned by atom_types.index(atom) to 1. It does not modify atom_types or any other global state (no side effects). If atom is not present in atom_types, list.index() raises a ValueError; callers should ensure the atom symbol is valid for the current molconvert atom_types vocabulary or catch ValueError to handle unknown atom types. The function does not perform normalization beyond one-hot encoding and does not accept or return other data structures (e.g., Python lists) as the official output; downstream code in TDC expects a NumPy array shaped (1, N).
    """
    from tdc.chem_utils.featurize.molconvert import atom2onehot
    return atom2onehot(atom)


################################################################################
# Source: tdc.chem_utils.featurize.molconvert.mol2file2smiles
# File: tdc/chem_utils/featurize/molconvert.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_chem_utils_featurize_molconvert_mol2file2smiles(molfile: str):
    """tdc.chem_utils.featurize.molconvert.mol2file2smiles: Convert a MOL2-format file to a canonical SMILES string for use in TDC molecule featurization and dataset processing pipelines.
    
    Args:
        molfile (str): Path to a MOL2-format file on disk. This parameter is a string that identifies the file to be read by RDKit's Chem.MolFromMol2File function. In the TDC context, callers supply MOL2 files that represent small-molecule structures (for example, dataset entries or intermediate outputs from molecular modeling tools). The function reads this file from the filesystem; it does not modify the file or any global state.
    
    Returns:
        str: A canonicalized SMILES string corresponding to the molecule read from the provided MOL2 file. This string is produced by RDKit (Chem.MolToSmiles) and then passed through the module's canonicalize function to produce a normalized SMILES representation consistent with TDC featurization conventions. The returned SMILES is suitable for downstream tasks in TDC such as descriptor computation, model input, or dataset canonicalization.
    
    Behavior and side effects:
        The function uses RDKit's Chem.MolFromMol2File to parse the provided MOL2 file and then Chem.MolToSmiles to convert the RDKit Mol object into a SMILES string. The SMILES is then canonicalized by the local canonicalize function to enforce a consistent SMILES representation across TDC data functions. The function performs a synchronous read of the specified file path and has no other side effects (it does not write files or change external state).
    
    Failure modes and error handling:
        If the file path does not exist, is not readable, or is not a valid MOL2-formatted file, RDKit's parser (Chem.MolFromMol2File) may fail to create a molecule object (it may return None) or raise an error; in that case, the subsequent call to Chem.MolToSmiles will raise an exception. This function does not catch or convert RDKit parsing or SMILES-generation exceptions, so callers should handle file- and parser-related exceptions (for example, FileNotFoundError or RDKit parsing errors) as appropriate in their TDC data processing pipelines.
    
    Practical significance in TDC:
        Converting MOL2 files to canonical SMILES is a common preprocessing step in TDC for standardizing molecular representations before featurization, splitting, and model evaluation. This function provides a minimal, deterministic conversion that integrates with TDC's canonicalization conventions to help ensure consistency of molecular inputs across datasets and downstream machine-learning workflows.
    """
    from tdc.chem_utils.featurize.molconvert import mol2file2smiles
    return mol2file2smiles(molfile)


################################################################################
# Source: tdc.chem_utils.featurize.molconvert.mol_conformer2graph3d
# File: tdc/chem_utils/featurize/molconvert.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_chem_utils_featurize_molconvert_mol_conformer2graph3d(mol_conformer_lst: list):
    """Convert a list of (molecule, conformer) pairs into a list of 3D graph representations suitable for 3D molecular featurization and graph-based ML in TDC. This function is used in TDC's chem_utils featurization pipeline to turn RDKit-style molecule objects and their corresponding 3D conformers into numeric adjacency and bond-type matrices plus an index-to-atom mapping. The produced graphs are practical inputs for downstream tasks such as 3D graph neural networks, distance-based descriptors, and molecular property prediction in the TDC framework.
    
    Args:
        mol_conformer_lst (list): A list where each element is a tuple (mol, conformer). Here, mol is expected to be an RDKit-like molecule object that implements GetNumAtoms(), GetAtoms(), and GetBonds(); conformer is expected to be a conformer object that implements GetAtomPosition(i) for atom indices 0..GetNumAtoms()-1. Each tuple represents a single molecular instance and its single 3D geometry (conformer). The role of this parameter is to provide both topological (atoms and bonds) and geometric (3D coordinates) information needed to build the graph. Practical significance: callers supply experimental or computed conformers (for example, from RDKit or other conformer generators) paired with molecule objects so that TDC can featurize them into 3D graph inputs for model training and evaluation.
    
    Returns:
        list: A list (graph3d_lst) with one element per input tuple. Each element is a tuple (idx2atom, distance_adj_matrix, bondtype_adj_matrix) where:
            idx2atom (dict): A mapping from atom index (int, 0-based) to atom symbol (str) obtained from mol.GetAtoms(); this provides node identity information used for interpreting per-node features or for downstream atom-type conditioning.
            distance_adj_matrix (np.array): A 2D NumPy array of shape (n, n) where n is mol.GetNumAtoms().distance_adj_matrix[i, j] is the Euclidean distance (float) between atom i and atom j computed via the distance3d function on the conformer coordinates. This matrix is symmetric by construction (distance_adj_matrix[i, j] == distance_adj_matrix[j, i]) and is intended for use as continuous pairwise geometric features in 3D graph models.
            bondtype_adj_matrix (np.array): A 2D NumPy integer array of shape (n, n) encoding bond types between atom pairs using the mapping {"SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3, "AROMATIC": 4}. Entries are set based on mol.GetBonds() and bond.GetBondType(); in the current implementation bond entries are assigned to the (a1, a2) index for each bond (where a1 and a2 are the bond endpoint indices retrieved by GetBeginAtom().GetIdx() and GetEndAtom().GetIdx()). Note: due to how assignments are performed in the implementation, bondtype_adj_matrix may not be symmetrically assigned (i.e., a2, a1 may remain zero), so consumers should be aware if their downstream code expects a symmetric bond-type matrix.
    
    Behavior, side effects, defaults, and failure modes:
        This function performs the following steps for each (mol, conformer) pair: (1) reads atom count via mol.GetNumAtoms(), (2) builds an idx2atom dict mapping indices to atom symbols, (3) collects 3D coordinates from conformer.GetAtomPosition(i) and constructs a positions array, (4) computes pairwise Euclidean distances for all i < j and populates a symmetric distance_adj_matrix, and (5) iterates bonds from mol.GetBonds() to populate bondtype_adj_matrix using the bond-to-integer mapping shown above. There are no external side effects: inputs are not modified and no global state is mutated; the function returns newly created NumPy arrays and dicts.
    
        Complexity: computing pairwise distances scales O(n^2) in the number of atoms n due to the all-pairs distance computation.
    
        Failure modes and input validation expectations: The function assumes each mol has at least one atom; if mol.GetNumAtoms() returns 0, the code attempts to concatenate an empty list of position arrays and will raise a ValueError. Each conformer must implement GetAtomPosition(i) for all atom indices 0..n-1; if a conformer is missing or does not provide positions or if indices are out of range, AttributeError or IndexError may be raised. mol is expected to provide GetAtoms(), GetBonds(), bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx(), and bond.GetBondType(); if these methods are not present (i.e., a non-RDKit object), AttributeError will be raised. The function does not perform explicit type checking or error handling for these conditions; callers should validate inputs or catch exceptions.
    
        Special cases: an empty input list returns an empty list. The bond-type mapping is fixed in the implementation as {"SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3, "AROMATIC": 4} and there is no handling for uncommon bond types beyond these string keys.
    
        Practical guidance: Use RDKit molecule and conformer objects (or objects implementing the same methods) with valid 3D coordinates when calling this function. After obtaining the returned list, confirm whether you require a symmetric bond-type adjacency; if so, symmetrize bondtype_adj_matrix yourself by, for example, taking the elementwise maximum of the matrix and its transpose.
    """
    from tdc.chem_utils.featurize.molconvert import mol_conformer2graph3d
    return mol_conformer2graph3d(mol_conformer_lst)


################################################################################
# Source: tdc.chem_utils.featurize.molconvert.molfile2smiles
# File: tdc/chem_utils/featurize/molconvert.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_chem_utils_featurize_molconvert_molfile2smiles(molfile: str):
    """tdc.chem_utils.featurize.molconvert.molfile2smiles converts a molecular file in MDL MOL format into a canonical SMILES string suitable for downstream featurization and machine-learning pipelines in TDC.
    
    Args:
        molfile (str): Filesystem path to a MOL-format file (for example an MDL V2000 or V3000 .mol file). This string argument is passed to RDKit's Chem.MolFromMolFile to read and parse the molecule. In the TDC context, callers provide pathnames for individual molecule files extracted from datasets or generated by tools; the function reads the file content (side effect: file I/O read only) and interprets its atoms, bonds, stereochemistry, and other MOL-recorded information to produce a molecular graph object.
    
    Returns:
        str: A canonicalized SMILES string representing the same molecule encoded in the input MOL file. The function first uses RDKit to convert the MOL file into an RDKit Mol object, then renders a SMILES string via Chem.MolToSmiles, and finally applies the module's canonicalize routine to normalize the SMILES. The returned SMILES is intended as a stable, standardized one-line molecular representation for use in featurizers, dataset records, or model inputs in TDC.
    
    Behavior and practical notes:
        - The function performs read-only file I/O on the path given by molfile; it does not modify or write any files.
        - The canonicalization step ensures consistent SMILES ordering and formatting across molecules, which is important for reproducible featurization, deduplication, and model training in therapeutic ML tasks.
        - This function is typically used during dataset preprocessing and featurization within TDC to convert per-molecule MOL files into the SMILES strings required by many cheminformatics tools and ML pipelines.
    
    Failure modes and exceptions:
        - If the file path does not exist or is not accessible, the underlying RDKit call will fail and an exception will propagate to the caller.
        - If the file contents are not a valid MOL-format representation or RDKit cannot parse the molecule, Chem.MolFromMolFile may return None and subsequent calls (Chem.MolToSmiles or canonicalize) will raise an exception. Callers should validate file existence and handle exceptions when processing large batches of files.
        - The function does not perform extensive error correction (for example, sanitization of malformed valences); such preprocessing must be done before calling this utility when needed.
    """
    from tdc.chem_utils.featurize.molconvert import molfile2smiles
    return molfile2smiles(molfile)


################################################################################
# Source: tdc.chem_utils.featurize.molconvert.raw3D2pyg
# File: tdc/chem_utils/featurize/molconvert.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_chem_utils_featurize_molconvert_raw3D2pyg(raw3d_feature: tuple):
    """tdc.chem_utils.featurize.molconvert.raw3D2pyg converts a raw 3D molecular feature tuple into a torch_geometric.data.Data object suitable for graph neural network (PyG) consumption in the TDC (Therapeutics Data Commons) small-molecule featurization pipeline.
    
    Args:
        raw3d_feature (tuple): A tuple representing per-molecule raw 3D features produced or expected by upstream TDC featurizers. In the current implementation this tuple must contain exactly two elements in this order:
            - atom_string_list (list): A Python list of atom descriptor strings, length N, where N is the number of atoms in the molecule. Each element is a textual representation for an atom (for example, an element symbol possibly with additional annotation) that the helper function atomstring2atomfeature can map to a numeric feature vector. In TDC, these atom strings are used to derive fixed-size per-atom feature vectors that serve as node features for graph neural networks.
            - positions (np.array): A NumPy array of shape (N, 3) containing the 3D Cartesian coordinates (x, y, z) for each of the N atoms. These coordinates are converted to a torch.Tensor and assigned to the Data.pos attribute so PyG GNNs can use geometric information.
        Note: older or alternative representations sometimes include a third element y (a float label) as in prior versions of this helper, but the current function unpacks exactly two elements and therefore will raise a ValueError if a tuple of a different length is provided. If a label value is present upstream, it must be handled outside this function or the calling code must supply a two-element tuple.
    
    Returns:
        torch_geometric.data.Data: A PyG Data object with the following populated attributes:
            - x (torch.Tensor): A torch tensor obtained by converting the NumPy array produced by atomstring2atomfeature(atom_string_list). This tensor holds per-atom numeric features (node features) and has shape (N, F) where F is the feature dimension returned by atomstring2atomfeature. These features are used as input node attributes for graph neural networks within TDC workflows.
            - pos (torch.Tensor): A torch tensor converted from the provided NumPy positions array with shape (N, 3), representing 3D atomic coordinates and used by geometry-aware GNN models.
        The Data.y attribute is not set by this function; if a label/target is required it must be attached to the returned Data externally by the caller.
    
    Behavior, side effects, and failure modes:
        - The function calls atomstring2atomfeature(atom_string_list) to map atom strings to a NumPy array of per-atom features; this helper must be available in the runtime namespace and must return a NumPy array convertible with torch.from_numpy. If atomstring2atomfeature is missing or raises an exception, raw3D2pyg will fail.
        - Inputs are converted using torch.from_numpy for both atom features and positions; therefore positions and the output of atomstring2atomfeature must be NumPy arrays with dtypes compatible with torch.from_numpy. Passing Python lists or arrays of incompatible dtype will raise a TypeError or ValueError from torch.
        - The function does not modify the input objects in place; it constructs new torch tensors and a new PyG Data object.
        - The function expects the input tuple to have exactly two elements; providing a tuple with a different length will raise a ValueError at unpacking. Providing shapes inconsistent between atom_string_list length and positions first dimension (N) will likely cause downstream shape mismatches in model training or immediate tensor-shape errors when constructing the Data object.
        - No device (CPU/GPU) transfer is performed; tensors are created on the default CPU. If GPU tensors are required, the caller must move them to the appropriate device after receiving the Data object.
        - The function does not attach labels/targets (y) to the returned Data object even if such a value exists upstream; callers must manage labels separately to ensure consistency with TDC benchmarking and training pipelines.
    """
    from tdc.chem_utils.featurize.molconvert import raw3D2pyg
    return raw3D2pyg(raw3d_feature)


################################################################################
# Source: tdc.chem_utils.featurize.molconvert.sdffile2coulomb
# File: tdc/chem_utils/featurize/molconvert.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_chem_utils_featurize_molconvert_sdffile2coulomb(sdf: str):
    """tdc.chem_utils.featurize.molconvert.sdffile2coulomb converts an MDL SDF file into Coulomb matrix feature vectors for downstream molecular machine learning tasks in TDC (for example, property prediction or molecular representation benchmarking). The function reads an SDF file, extracts SMILES strings for each molecule using sdffile2smiles_lst, and then computes Coulomb matrix features for those molecules using smiles_lst2coulomb. The resulting NumPy array is intended as an ML-ready numeric descriptor representing interatomic Coulombic interactions and relative atomic composition.
    
    Args:
        sdf (str): Path to an MDL SDF file on the filesystem containing one or more molecule records in standard SDF format. This string must be a valid filesystem path readable by the executing process; the function delegates parsing to sdffile2smiles_lst which expects SDF-formatted content. The provided file is not modified by this function; it is only read.
    
    Returns:
        np.array: A NumPy array containing Coulomb matrix features computed for the molecules in the input SDF file. Each entry in the array corresponds to the Coulomb feature vector for one molecule, in the same order as the molecules appear in the input SDF. This array is intended to be passed directly into downstream TDC model training or evaluation pipelines that accept NumPy feature arrays.
    
    Behavior and failure modes:
        The function internally calls sdffile2smiles_lst(sdf) to obtain a list of SMILES strings, then calls smiles_lst2coulomb(smiles_lst) to compute Coulomb features. If the input SDF contains no valid molecules, the helpers may return an empty list and the function will return an empty NumPy array. If the provided path does not exist or is not readable, a FileNotFoundError or an I/O-related exception raised by the underlying file operations will propagate. Malformed SDF content or unparsable molecules can cause parsing errors or exceptions propagated from sdffile2smiles_lst or smiles_lst2coulomb. There are no side effects such as writing files; this function only performs in-memory conversions and requires the standard TDC dependencies (for example, numpy) available in the runtime environment.
    """
    from tdc.chem_utils.featurize.molconvert import sdffile2coulomb
    return sdffile2coulomb(sdf)


################################################################################
# Source: tdc.chem_utils.featurize.molconvert.sdffile2graph3d_lst
# File: tdc/chem_utils/featurize/molconvert.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_chem_utils_featurize_molconvert_sdffile2graph3d_lst(sdffile: str):
    """Convert an SDF file into a list of 3D molecular graph representations.
    
    This function is used in the TDC chem_utils featurize pipeline to produce 3D graph inputs for downstream molecular machine learning workflows (for example, graph neural networks or other graph-based featurizers used in TDC datasets and tasks). The implementation first parses the provided SDF file into a list of molecular conformers by calling sdffile2mol_conformer(sdffile) and then converts those conformers into graph representations via mol_conformer2graph3d(...). The returned graphs preserve per-atom indexing, inter-atomic geometric information, and bond-type information required for 3D-aware featurization and benchmarking within TDC.
    
    Args:
        sdffile (str): Path to an SDF file (string). The file is expected to contain one or more molecules with conformer (3D coordinate) information in standard SDF/MDL format. This parameter is the sole input and must be a file path accessible to the executing process. The function will read this file; it does not modify or write to the file.
    
    Returns:
        list: graph3d_lst, a Python list where each element is a 3D graph representation produced from a single molecule/conformer in the input SDF. Each graph element includes at least the following components:
            idx2atom (dict): A dictionary mapping atom indices (integers) to atom metadata or atom objects as produced by the mol_conformer2graph3d conversion. This mapping is used to associate matrix rows/columns with specific atoms for featurization and interpretation.
            distance_adj_matrix (np.array): A NumPy array representing pairwise geometric distances between atoms in the conformer (a symmetric matrix). The distances are expressed in the same linear units used in the input SDF coordinates (commonly angstroms).
            bondtype_adj_matrix (np.array): A NumPy array encoding bond connectivity and bond-type information between atom pairs. The precise numeric encoding of bond types is determined by mol_conformer2graph3d; users should consult that function for the exact encoding schema.
    
    Behavior, side effects, and failure modes:
        - The function only reads the file at sdffile and returns an in-memory list of graph objects; it has no other side effects (it does not write to disk).
        - Internally this function delegates parsing and conversion to sdffile2mol_conformer and mol_conformer2graph3d; any exceptions raised by those helper functions (for example, file-not-found, parse errors, or unsupported SDF contents) will propagate to the caller.
        - If the SDF contains no valid conformers, the function will return an empty list (or propagate whatever result sdffile2mol_conformer/mole_conformer2graph3d produce in that case).
        - Large SDF files with many molecules or many conformers may consume substantial memory because the entire graph list and adjacency matrices are held in memory; callers should manage memory accordingly.
        - The semantic interpretation of the returned arrays (distance units, bond-type integer encoding) follows the conventions of mol_conformer2graph3d; for precise encoding details consult that function's documentation.
    """
    from tdc.chem_utils.featurize.molconvert import sdffile2graph3d_lst
    return sdffile2graph3d_lst(sdffile)


################################################################################
# Source: tdc.chem_utils.featurize.molconvert.sdffile2mol_conformer
# File: tdc/chem_utils/featurize/molconvert.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_chem_utils_featurize_molconvert_sdffile2mol_conformer(sdffile: str):
    """Convert an SDF (Structure-Data File) into a list of molecule/conformer pairs suitable for downstream molecular featurization and machine learning workflows in TDC.
    
    This function reads the SDF given by sdffile using RDKit's PandasTools.LoadSDF, extracts the RDKit ROMol objects (stored in the DataFrame column "ROMol"), and for each molecule retrieves the first conformer (conformer id = 0). The result is a Python list of 2-tuples (mol, conformer) preserving the order of molecules as they appear in the SDF. This is typically used in TDC featurization pipelines to obtain 3D coordinate information for each molecule for tasks such as descriptor calculation, graph construction, or other 3D-aware machine learning models in drug discovery.
    
    Args:
        sdffile (str): Path to an input SDF file on disk. The function expects a file path string pointing to a valid SDF file that can be parsed by RDKit. LoadSDF is called with smilesName="SMILES" and will produce a pandas DataFrame with a "ROMol" column of RDKit molecule objects; the SDF is expected to contain molecular records (ideally with 3D coordinates/conformers) that RDKit can convert to ROMol objects.
    
    Returns:
        list[tuple]: A list of tuples (mol, conformer) where mol is the RDKit ROMol object (rdkit.Chem.rdchem.Mol) extracted from the SDF and conformer is the RDKit Conformer object (rdkit.Chem.rdchem.Conformer) obtained by calling mol.GetConformer(id=0) for that molecule. The list preserves the order of molecules in the input SDF. This return value is directly usable by downstream TDC featurization functions that require both the molecular graph (ROMol) and its 3D coordinates (Conformer).
    
    Behavior, side effects, and failure modes:
        This function only reads the input file and returns in-memory RDKit objects; it does not modify or write any files. If the SDF file is empty or contains no valid ROMol objects, the function will return an empty list. If a ROMol entry is None or a molecule does not contain any conformers, calling mol.GetConformer(id=0) will raise an exception propagated from RDKit (for example AttributeError or an RDKit-specific error); likewise, LoadSDF may raise file/parse-related exceptions (e.g., FileNotFoundError, pandas/RDKit parsing errors) if the input path is invalid or the file is malformed. Callers should validate input files or catch RDKit/pandas exceptions when using this function.
    """
    from tdc.chem_utils.featurize.molconvert import sdffile2mol_conformer
    return sdffile2mol_conformer(sdffile)


################################################################################
# Source: tdc.chem_utils.featurize.molconvert.sdffile2selfies_lst
# File: tdc/chem_utils/featurize/molconvert.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_chem_utils_featurize_molconvert_sdffile2selfies_lst(sdf: str):
    """Convert an SDF file into a list of SELFIES strings.
    
    This function is part of the TDC chem_utils.featurize.molconvert utilities and is used to produce a machine-learning-friendly, robust string representation (SELFIES) for each molecule contained in an SDF-format file. Concretely, the function first extracts SMILES strings from the provided SDF input by calling sdffile2smiles_lst(sdf), then converts each SMILES to a SELFIES string via smiles2selfies. SELFIES are a compact, unambiguous molecular string representation often used in generative models and featurization pipelines; in TDC workflows this conversion is used to prepare molecular inputs for tasks such as property prediction, generation, and benchmarking described in the TDC README.
    
    Args:
        sdf (str): A string argument forwarded to sdffile2smiles_lst. In typical usage this is a filesystem path to an SDF-format file containing one or more molecules (for example, a ".sdf" file retrieved as part of a TDC dataset). The function does not modify the file system; it only reads the input via sdffile2smiles_lst. If your environment uses a different accepted form for sdffile2smiles_lst (for example, an in-memory SDF string), pass that same string value — the parameter is passed through unchanged.
    
    Returns:
        list: A list of SELFIES strings. Each element is the SELFIES encoding produced by smiles2selfies for the corresponding SMILES string returned by sdffile2smiles_lst(sdf). The order of entries in the returned list matches the order of molecules as returned by sdffile2smiles_lst, so downstream workflows that rely on molecule ordering (for labels or indices) can maintain alignment.
    
    Behavior and side effects:
        - This function delegates SMILES extraction and SMILES->SELFIES conversion to sdffile2smiles_lst and smiles2selfies respectively; any behavior, defaults, or preprocessing performed by those functions (for example, handling of salts, multiple records, or stereochemistry) will affect the output.
        - The function itself has no side effects (it does not write to disk or alter global state); it only returns an in-memory list of strings.
        - The returned SELFIES list is intended as an input feature representation for molecular machine learning tasks in TDC (e.g., model training, generation oracles, and benchmark evaluation).
    
    Failure modes and exceptions:
        - If the provided sdf value does not point to a readable SDF file or is otherwise not acceptable to sdffile2smiles_lst, sdffile2smiles_lst may raise exceptions such as FileNotFoundError, IOError, or format-specific parsing errors.
        - If sdffile2smiles_lst returns invalid or unexpected SMILES strings, smiles2selfies may raise a ValueError or another exception indicating an invalid SMILES input.
        - No additional validation is performed by this wrapper; callers should handle or propagate exceptions raised by the underlying functions as appropriate for their TDC data-processing pipeline.
    """
    from tdc.chem_utils.featurize.molconvert import sdffile2selfies_lst
    return sdffile2selfies_lst(sdf)


################################################################################
# Source: tdc.chem_utils.featurize.molconvert.sdffile2smiles_lst
# File: tdc/chem_utils/featurize/molconvert.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_chem_utils_featurize_molconvert_sdffile2smiles_lst(sdffile: str):
    """tdc.chem_utils.featurize.molconvert.sdffile2smiles_lst converts a molecular SDF (Structure-Data File) on disk into a Python list of SMILES strings suitable for downstream featurization and machine-learning workflows in TDC. This function is used in the small-molecule data-processing pipeline to extract the canonical text representations (SMILES) produced by RDKit from an SDF file so that those strings can be passed to featurizers, graph builders, or TDC oracles and evaluators.
    
    Args:
      sdffile (str): Path to the input SDF file on the local filesystem. In the TDC context, this should be the file containing one or more molecule records in SDF format obtained from dataset preparation or external sources; the string must be a filesystem-accessible path. The function passes this path to rdkit.Chem.PandasTools.LoadSDF to load the file into a pandas.DataFrame and expects that LoadSDF will add a column named "SMILES" containing the SMILES string for each record.
    
    Returns:
      list: A list of SMILES strings extracted from the input SDF file. Each element corresponds to the entry in the DataFrame column "SMILES" created by RDKit's LoadSDF and therefore represents the molecular SMILES used by downstream TDC featurizers and benchmarks. Note that if RDKit/LoadSDF fails to generate or populate a SMILES value for a particular record, the corresponding list element will reflect whatever value LoadSDF placed in the "SMILES" column (for example, pandas.NaN or None); callers should validate or filter list elements before using them for model training or oracle evaluation.
    
    Behavior, side effects, and failure modes:
      - The function imports and uses rdkit.Chem.PandasTools.LoadSDF at runtime; RDKit must be installed in the Python environment. If RDKit is not installed, an ImportError will be raised when this function is invoked.
      - LoadSDF reads the entire SDF into memory as a pandas.DataFrame; large SDF files may consume substantial memory and impact performance. Plan for memory use when processing large datasets.
      - The function does not modify the input file on disk; it only reads from the provided path.
      - If the provided sdffile path does not exist or is not a readable SDF file, LoadSDF or the underlying file I/O will raise an exception (e.g., FileNotFoundError, OSError). If the loaded DataFrame does not contain a "SMILES" column, attempting to access df["SMILES"] will raise a KeyError.
      - The function returns the raw column values as produced by LoadSDF; it performs no additional validation, canonicalization, or sanitization of SMILES strings. Callers in the TDC pipeline should perform any required validation or normalization (for example, filtering out None/NaN entries, standardizing stereochemistry, or sanitizing invalid SMILES) before using the returned list for featurization or model input.
    """
    from tdc.chem_utils.featurize.molconvert import sdffile2smiles_lst
    return sdffile2smiles_lst(sdffile)


################################################################################
# Source: tdc.chem_utils.featurize.molconvert.selfies2smiles
# File: tdc/chem_utils/featurize/molconvert.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_chem_utils_featurize_molconvert_selfies2smiles(selfies: str):
    """Convert a molecular string in SELFIES format to a canonical SMILES string.
    
    This function decodes a SELFIES (Self-Referencing Embedded Strings) representation of a small molecule into a SMILES (Simplified Molecular Input Line Entry System) string and then canonicalizes the result. In the TDC (Therapeutics Data Commons) codebase this conversion is used when downstream data functions, featurizers, or molecule-generation oracles require a SMILES representation for model input, property evaluation, or dataset standardization. The implementation calls sf.decoder(selfies) to perform the SELFIES→SMILES decode and then calls canonicalize(...) to produce a deterministic canonical SMILES suitable for consistent featurization and comparison across datasets and experiments.
    
    Args:
      selfies (str): A SELFIES string encoding a molecular structure. SELFIES is a robust, machine-readable molecular string format used in molecule generation and representation tasks. The parameter is expected to be a Python str containing a valid SELFIES token sequence; this function does not perform prior validation beyond what the underlying sf.decoder provides.
    
    Returns:
      smiles (str): A SMILES string corresponding to the decoded molecule, passed through canonicalize to yield a canonicalized SMILES. The returned canonical SMILES is intended for use in downstream TDC workflows (featurization, evaluation, and model inputs) where a stable, unique string representation of the molecule is required.
    
    Behavior and failure modes:
    This function is pure (no external side effects) and deterministic for a given input SELFIES string because the output is canonicalized. If the input is not a valid SELFIES string, the underlying sf.decoder or canonicalize may raise an exception (for example, decoding errors or downstream canonicalization errors); callers should validate inputs or catch exceptions as appropriate. The function preserves the exact types shown in the signature: it accepts a str and returns a str.
    """
    from tdc.chem_utils.featurize.molconvert import selfies2smiles
    return selfies2smiles(selfies)


################################################################################
# Source: tdc.chem_utils.featurize.molconvert.smiles2DGL
# File: tdc/chem_utils/featurize/molconvert.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_chem_utils_featurize_molconvert_smiles2DGL(smiles: str):
    """tdc.chem_utils.featurize.molconvert.smiles2DGL: Convert a SMILES string into a dgl.DGLGraph representing the 2D molecular graph used in TDC molecular featurization and graph-based ML pipelines.
    
    Args:
        smiles (str): A SMILES string representing a small molecule. This string is first canonicalized by the module's canonicalize(smiles) helper and then parsed with RDKit's Chem.MolFromSmiles to produce an RDKit Mol object. The SMILES should be a valid, RDKit-parsable representation; canonicalization may reorder atoms, and the returned graph's atom indices correspond to RDKit atom indices after canonicalization. This function is intended for use in the TDC (Therapeutics Data Commons) featurization workflow to produce DGL graphs for graph neural network models (for tasks such as ADMET prediction, activity screening, and other molecule-based tasks described in the TDC README).
    
    Returns:
        dgl.DGLGraph: A DGL graph whose nodes correspond to atoms in the parsed molecule and whose directed edges correspond to chemical bonds. Each bond in the RDKit Mol is represented as two opposite directed edges (one for each direction). The number of nodes in the graph equals mol.GetNumAtoms() from RDKit. This function does not assign atom or bond feature vectors (node/edge attributes); it constructs only the graph topology (nodes and directed edges). The returned object is created with dgl.DGLGraph() and populated via add_nodes and add_edges.
    
    Behavior and side effects:
        The function canonicalizes the input SMILES and then uses RDKit to parse it. It constructs an initially featureless DGL graph with one node per atom and directed edges for each bond in both directions. No additional node/edge attributes (for example atom type, formal charge, bond type) are added by this function — downstream code in TDC is expected to attach such features if required for model input. The atom ordering and edge endpoints reflect RDKit's atom indices after canonicalization.
    
    Failure modes and requirements:
        This function requires RDKit and DGL to be available in the runtime environment. If Chem.MolFromSmiles returns None (for example when the SMILES is invalid or unparsable), subsequent calls such as mol.GetNumAtoms() will raise an exception (e.g., AttributeError). Invalid or empty SMILES strings, or missing RDKit/DGL installations, will therefore cause the function to fail. Users should validate or catch exceptions upstream if they expect malformed SMILES.
    """
    from tdc.chem_utils.featurize.molconvert import smiles2DGL
    return smiles2DGL(smiles)


################################################################################
# Source: tdc.chem_utils.featurize.molconvert.smiles2ECFP2
# File: tdc/chem_utils/featurize/molconvert.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_chem_utils_featurize_molconvert_smiles2ECFP2(smiles: str):
    """tdc.chem_utils.featurize.molconvert.smiles2ECFP2 converts a SMILES string for a small molecule into an ECFP2 (Morgan fingerprint with radius 1) numeric fingerprint suitable for machine learning workflows in TDC, producing a fixed-length numeric vector representation used by downstream models and dataset featurizers.
    
    Args:
      smiles (str): Input canonical or non-canonical SMILES string that identifies a single small-molecule chemical structure. In the TDC context this string is the primary molecular identifier used by dataset loaders and featurizers. The function first canonicalizes this SMILES using canonicalize(smiles), then parses it into an RDKit molecule with smiles_to_rdkit_mol(smiles). Supplying an invalid SMILES or a SMILES that cannot be parsed by RDKit will cause smiles_to_rdkit_mol to fail or return None, which will raise an exception when the fingerprint is computed.
    
    Returns:
      numpy.ndarray: One-dimensional numpy.ndarray of dtype numpy.float64 and length 2048 containing the ECFP2/Morgan (radius=1) fingerprint expressed as numeric bit values (0.0 or 1.0). Implementation details: the function internally computes an RDKit bit vector fingerprint object (rdkit.DataStructs.cDataStructs.UIntSparseIntVect) via AllChem.GetMorganFingerprintAsBitVect(molecule, 1, nBits=2048) and then converts that RDKit fingerprint into the returned numpy array using DataStructs.ConvertToNumpyArray. The returned array is the value to be used as the fixed-length numeric feature vector for machine learning models and featurizers in TDC.
    
    Behavior, defaults, and failure modes:
      The function uses a fixed fingerprint size of 2048 bits (nbits = 2048) and a Morgan radius of 1, which corresponds to the ECFP2 variant commonly used in cheminformatics and TDC molecular featurization pipelines. Side effects include calls to external helper functions canonicalize and smiles_to_rdkit_mol and to RDKit (AllChem and DataStructs). If RDKit is not installed or these helper functions raise errors, the function will raise an ImportError or the underlying exception from RDKit or the helper functions. If the input SMILES cannot be parsed into a valid RDKit molecule (e.g., invalid syntax or unsupported characters), the fingerprint computation will fail and an exception will be raised. The function always returns a numpy.ndarray; it does not modify global state or write files.
    """
    from tdc.chem_utils.featurize.molconvert import smiles2ECFP2
    return smiles2ECFP2(smiles)


################################################################################
# Source: tdc.chem_utils.featurize.molconvert.smiles2ECFP4
# File: tdc/chem_utils/featurize/molconvert.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_chem_utils_featurize_molconvert_smiles2ECFP4(smiles: str):
    """tdc.chem_utils.featurize.molconvert.smiles2ECFP4 converts a SMILES string into an ECFP4 (Morgan radius 2) fingerprint and returns the fingerprint as a numeric 1D NumPy array suitable for machine learning featurization in therapeutic chemistry workflows.
    
    This function is used in TDC molecular featurization pipelines to produce a fixed-length binary fingerprint representation of a small molecule given its canonical SMILES. Internally it canonicalizes the SMILES, converts it to an RDKit molecule, computes the Morgan fingerprint with radius 2 and 2048 bits via RDKit AllChem.GetMorganFingerprintAsBitVect, and converts the resulting RDKit bit vector into a NumPy array of floats. ECFP4 fingerprints (Morgan radius 2) are commonly used as input features for property prediction, ADMET modeling, activity screening, and other single-instance prediction tasks in TDC.
    
    Args:
        smiles (str): A molecule representation in SMILES (Simplified Molecular Input Line Entry System) format. This string will be canonicalized by canonicalize(smiles) before conversion. Provide a valid, parseable SMILES for a single molecule; passing an invalid or nonstandard SMILES may cause RDKit to return None or raise an error during molecule parsing or fingerprint generation.
    
    Returns:
        numpy.ndarray: A 1D NumPy array of length 2048 and dtype float64 containing the ECFP4 fingerprint bits converted to numeric values. Each element corresponds to a hashed fingerprint bit (0.0 or 1.0) produced by AllChem.GetMorganFingerprintAsBitVect with radius=2 and nBits=2048. Although older documentation may reference RDKit's rdkit.DataStructs.cDataStructs.UIntSparseIntVect, this implementation returns the converted NumPy array as the practical output for downstream ML tasks.
    
    Behavior and side effects:
        The function performs deterministic preprocessing and feature computation: it canonicalizes the input SMILES, parses it to an RDKit Mol object using smiles_to_rdkit_mol, computes a 2048-bit Morgan fingerprint at radius 2, and converts that bit vector into a NumPy array. No global state is modified; the function returns the computed array and has no other side effects.
    
    Failure modes and errors:
        If canonicalize(smiles) or smiles_to_rdkit_mol(smiles) cannot process the input (for example, malformed SMILES), RDKit functions may return None or raise exceptions; subsequently GetMorganFingerprintAsBitVect may raise an error. Consumers should validate or catch exceptions for invalid SMILES inputs. The function assumes RDKit and NumPy are available in the environment; missing dependencies will result in ImportError/NameError when this function is invoked.
    """
    from tdc.chem_utils.featurize.molconvert import smiles2ECFP4
    return smiles2ECFP4(smiles)


################################################################################
# Source: tdc.chem_utils.featurize.molconvert.smiles2ECFP6
# File: tdc/chem_utils/featurize/molconvert.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_chem_utils_featurize_molconvert_smiles2ECFP6(smiles: str):
    """Convert a SMILES string into an ECFP6 (Morgan fingerprint with radius=3)
    bit vector and return it as a dense NumPy array suitable for machine
    learning feature input in the TDC molecular pipelines.
    
    This function is used across TDC to featurize small-molecule SMILES for
    downstream tasks such as single-instance property prediction, ADMET
    predictors, or molecule generation oracles. It canonicalizes the input
    SMILES, converts it to an RDKit Mol object, computes a 2048-bit Morgan
    bit fingerprint with radius 3 (commonly referred to as ECFP6), and
    converts the RDKit bit vector into a one-dimensional numpy.ndarray of
    float64 values (0.0 or 1.0) so the fingerprint can be consumed by
    numerical ML models and data loaders.
    
    Args:
        smiles (str): A SMILES string that represents a small-molecule
            chemical structure. The function first canonicalizes this string
            via the module's canonicalize function to ensure consistent
            representation (important for reproducible fingerprinting and
            canonical dataset processing in TDC). The canonicalized SMILES is
            then parsed to an RDKit Mol using smiles_to_rdkit_mol before
            fingerprint computation.
    
    Behavior and defaults:
        The fingerprint size is fixed to 2048 bits (nbits = 2048) and the
        Morgan fingerprint radius is fixed to 3 (ECFP6 convention). The
        function calls RDKit AllChem.GetMorganFingerprintAsBitVect to obtain
        an RDKit explicit bit vector and then converts it to a NumPy array
        of dtype numpy.float64 using rdkit.DataStructs.ConvertToNumpyArray.
        The returned array therefore has a deterministic length of 2048 and
        contains floating point 0.0 or 1.0 values corresponding to the bit
        vector. These choices are hard-coded to produce consistent, dense
        feature vectors expected by the TDC feature pipelines and model
        evaluators.
    
    Side effects and compatibility:
        The input string is not modified in place; canonicalize returns a
        new standardized SMILES. The function depends on RDKit and numpy; if
        RDKit is not available, import or RDKit function calls will raise
        exceptions upstream. This function intentionally converts the RDKit
        fingerprint to a NumPy array to avoid returning RDKit-specific data
        structures and to interoperate with TDC's model training and
        evaluation utilities.
    
    Failure modes:
        If the input SMILES cannot be parsed by smiles_to_rdkit_mol, that
        helper may return None or RDKit may raise an error; in such cases
        AllChem.GetMorganFingerprintAsBitVect will raise an exception. Callers
        should validate or filter SMILES beforehand when processing large
        datasets. Also note that the function does not perform error
        correction for chemically invalid SMILES; it expects a syntactically
        valid SMILES string as input.
    
    Referential note:
        The implementation approach follows standard RDKit fingerprinting
        practices (see upstream benchmarking examples such as
        github.com/rdkit/benchmarking_platform/blob/master/scoring/fingerprint_lib.py)
        and is intended to produce ECFP6 features for downstream TDC tasks.
    
    Returns:
        numpy.ndarray: A one-dimensional numpy.ndarray of dtype numpy.float64
            and length 2048 containing the ECFP6/Morgan fingerprint bits as
            floating point values (0.0 or 1.0). This dense numeric vector is
            ready for use as an input feature in TDC machine learning
            pipelines and evaluators.
    """
    from tdc.chem_utils.featurize.molconvert import smiles2ECFP6
    return smiles2ECFP6(smiles)


################################################################################
# Source: tdc.chem_utils.featurize.molconvert.smiles2PyG
# File: tdc/chem_utils/featurize/molconvert.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_chem_utils_featurize_molconvert_smiles2PyG(smiles: str):
    """Convert a SMILES string into a torch_geometric.data.Data graph suitable for use with PyTorch Geometric (PyG) graph neural networks in TDC molecular machine-learning workflows.
    
    This function is used in TDC to transform a textual molecule representation (SMILES) into a graph Data object that encodes atom-level features and bond connectivity so that downstream graph-based models (for example, ADMET/property predictors and other small-molecule tasks in TDC) can consume the molecule as input. Execution steps performed by this function: the input SMILES is canonicalized, parsed by RDKit (Chem.MolFromSmiles) into an RDKit Mol object, per-atom feature vectors are computed via get_atom_features and stacked into a torch tensor, element indices for each atom are computed against ELEM_LIST (but are computed locally and not attached to the returned Data), and bond connectivity is converted into an edge_index tensor by enumerating each bond in both directions (undirected bonds are represented as two directed edges). The returned Data contains node features (x) and edge_index suitable for PyG graph layers.
    
    Args:
        smiles (str): A SMILES string representing the molecular structure to convert. The string is first passed to canonicalize(smiles) to produce a canonical SMILES, then parsed with RDKit Chem.MolFromSmiles. This parameter is the primary input for creating the molecular graph; providing an invalid or unparsable SMILES will cause RDKit to return None, which leads to downstream AttributeError when the function attempts to access Mol methods (for example, GetNumAtoms). The caller is responsible for ensuring the string is a valid SMILES in the expected chemical notation used by RDKit.
    
    Returns:
        torch_geometric.data.Data: A PyG Data object with at least the following fields populated from the source code: x is a torch tensor produced by torch.stack over get_atom_features(atom) for each atom in the molecule (shape: [n_atoms, atom_feature_dim], dtype determined by get_atom_features), and edge_index is a torch.LongTensor encoding directed bond indices arranged as edge_index = bond_features.T where bond_features was constructed by adding both (idx1, idx2) and (idx2, idx1) for every RDKit bond (resulting in shape [2, n_bond_pairs], where n_bond_pairs equals 2 * number_of_RDKIT_bonds). No explicit target labels or the computed element-index tensor (y) are attached to the returned Data in the current implementation; y is computed locally in the function but not assigned to Data. External dependencies required at runtime include RDKit (Chem), torch, and torch_geometric; errors from canonicalize, RDKit parsing, tensor stacking, or missing dependencies will propagate as exceptions.
    """
    from tdc.chem_utils.featurize.molconvert import smiles2PyG
    return smiles2PyG(smiles)


################################################################################
# Source: tdc.chem_utils.featurize.molconvert.smiles2daylight
# File: tdc/chem_utils/featurize/molconvert.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_chem_utils_featurize_molconvert_smiles2daylight(s: str):
    """tdc.chem_utils.featurize.molconvert.smiles2daylight converts a SMILES string into a 2048-dimensional Daylight-style binary fingerprint used by TDC for molecular featurization in machine learning tasks (for example, ADME and other small-molecule benchmarks). The function canonicalizes the input SMILES, constructs an RDKit Mol object, computes the RDKit FingerprintMol, and sets bits corresponding to the on-bits returned by the fingerprint. This fingerprint is commonly used as a fixed-length representation of molecular structure for downstream model input, evaluation, and dataset processing within the TDC library.
    
    Args:
        s (str): Input SMILES string representing a small molecule. The function first calls canonicalize(s) to normalize the SMILES according to the module's canonicalization rules, then passes the canonicalized string to RDKit's Chem.MolFromSmiles to create an RDKit molecule object. The SMILES string is the primary identifier of molecular structure used by TDC data loaders and featurizers.
    
    Behavior and side effects:
        The function uses NumFinger = 2048 to produce a fixed-length (2048,) fingerprint. After obtaining the RDKit fingerprint object via FingerprintMols.FingerprintMol(mol), it queries the set bits with GetOnBits() and sets those bit positions to 1 in a NumPy array of zeros. The returned array therefore encodes presence/absence of fingerprint bits as 1/0 values. If any error occurs (for example, RDKit is not available on the environment, canonicalization or MolFromSmiles fails, the fingerprint computation fails, or bit indices are invalid), the function catches the exception, emits a diagnostic message via print_sys indicating the failure and the SMILES string, and returns an all-zero fingerprint instead of raising. Because exceptions are caught internally, callers should check whether the returned fingerprint is all zeros to detect conversion failures. The function relies on RDKit APIs (Chem.MolFromSmiles and FingerprintMols.FingerprintMol) and print_sys for logging; these are external dependencies and must be available for normal operation.
    
    Returns:
        numpy.array: A 1-D NumPy array of length 2048 representing the Daylight-style fingerprint produced for the canonicalized SMILES. Values are 0 or 1 (stored as numeric array values) indicating absent/present fingerprint bits. On normal success the array encodes the fingerprint bits set by FingerprintMols.FingerprintMol.GetOnBits(). If conversion fails for any reason (invalid SMILES, missing RDKit, or other runtime error), the function returns a numpy.array of shape (2048,) filled with zeros and prints a diagnostic message; no exception is propagated.
    """
    from tdc.chem_utils.featurize.molconvert import smiles2daylight
    return smiles2daylight(s)


################################################################################
# Source: tdc.chem_utils.featurize.molconvert.smiles2graph2D
# File: tdc/chem_utils/featurize/molconvert.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_chem_utils_featurize_molconvert_smiles2graph2D(smiles: str):
    """convert SMILES string into a two-dimensional molecular graph feature suitable for graph-based molecular
    featurization in TDC's small-molecule workflows (e.g., single-instance prediction, ADME benchmarks,
    and molecule generation/oracle evaluations). This function canonicalizes the input SMILES string,
    parses it into an RDKit molecule, and returns a mapping from atom indices to element symbols and
    an adjacency matrix that encodes bond types as integer codes. The outputs are intended to be used
    for constructing graph inputs for graph neural networks, featurizers, or any downstream model that
    expects atom-wise indices and a bond-typed adjacency matrix.
    
    Args:
        smiles (str): A SMILES string representing a small molecule. This is the raw string input used
            by TDC featurizers and data loaders. The function first canonicalizes this SMILES using the
            repository's canonicalize(smiles) helper (to produce a consistent representation across
            datasets and runs) and then converts the canonical SMILES to an RDKit molecule via
            smiles2mol(smiles). The caller must provide a valid, non-empty SMILES; invalid or
            unparsable SMILES will cause the underlying canonicalize/smiles2mol or RDKit routines to
            raise an exception.
    
    Returns:
        idx2atom (dict): A dictionary mapping atom indices (integers) to atom element symbols (strings),
            e.g., {0: 'C', 1: 'N', ...}. The integer keys correspond to the RDKit atom GetIdx() values
            used internally and match the row/column indices of the returned adjacency matrix. In the
            context of TDC, this mapping is useful for deriving atom-level features (element one-hot,
            atomic numbers, or domain-specific atom annotations) that align with the adjacency matrix.
        adj_matrix (np.array): A square numpy array of shape (n_atoms, n_atoms) and dtype int that
            encodes bond connectivity and bond types between atoms. Each nonzero entry adj_matrix[i, j]
            is an integer code produced by bondtype2idx(bond_type) for the bond between atoms i and j.
            The matrix is symmetric (undirected bond representation) because bonds are set for both
            (i, j) and (j, i). In TDC workflows this adjacency encodes the 2D chemical graph (bond
            topology and bond type codes) required by graph-based ML models and featurizers.
    
    Raises:
        ValueError: If the provided SMILES cannot be canonicalized or parsed into a molecule, the
            underlying canonicalize, smiles2mol, or RDKit routines will fail (for example, returning
            None or raising a parse error). The function surfaces these errors to the caller rather than
            silently returning invalid outputs.
    
    Behavior and side effects:
        The function is pure with respect to external state: it does not mutate its input string and
        does not write files. It relies on canonicalize(smiles), smiles2mol(smiles), and bondtype2idx(bond_type)
        helpers from the same module/package and on RDKit molecule methods (GetNumAtoms, GetAtoms,
        GetBonds, GetIdx, GetSymbol). For an empty molecule (zero atoms) the function will return an
        empty dict and an adjacency array of shape (0, 0). The canonicalization step enforces a stable
        SMILES ordering across datasets, improving reproducibility when the resulting graphs are used
        in TDC benchmarks and leaderboards.
    """
    from tdc.chem_utils.featurize.molconvert import smiles2graph2D
    return smiles2graph2D(smiles)


################################################################################
# Source: tdc.chem_utils.featurize.molconvert.smiles2maccs
# File: tdc/chem_utils/featurize/molconvert.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_chem_utils_featurize_molconvert_smiles2maccs(s: str):
    """Convert a SMILES string into a MACCS structural fingerprint represented as a NumPy array.
    
    This function is used in TDC molecular featurization pipelines to produce fixed-length numerical descriptors (fingerprints) from a SMILES representation of a small molecule. It first normalizes the input SMILES by calling canonicalize(s) so the same molecule maps consistently to the same fingerprint in downstream tasks. It then uses RDKit's Chem.MolFromSmiles to parse the canonical SMILES into an RDKit Mol object, generates MACCS keys via MACCSkeys.GenMACCSKeys, and converts the resulting RDKit fingerprint into a one-dimensional numpy.ndarray of dtype numpy.float64 suitable for machine learning feature inputs (for example, as input features for TDC single-instance prediction tasks or other molecular ML models).
    
    Args:
        s (str): Input molecule represented as a SMILES string. In the TDC context this is the textual molecular representation provided by datasets or users that will be converted into a numerical fingerprint feature. The function will first canonicalize this SMILES string using canonicalize(s) to ensure consistent representation before parsing with RDKit.
    
    Returns:
        numpy.array: A one-dimensional numpy.ndarray of dtype numpy.float64 containing the MACCS fingerprint bits converted to numeric values. This array is intended to be used as a fixed-length feature vector in molecular machine learning pipelines and downstream TDC data functions. The exact length of the array corresponds to the MACCS key vector produced by RDKit.
    
    Raises:
        Exception: If the input SMILES cannot be canonicalized or parsed by RDKit (for example, if Chem.MolFromSmiles returns None) or if RDKit or its fingerprint utilities are unavailable, the function will propagate an exception raised by those libraries. Callers should validate or handle invalid SMILES strings and ensure RDKit and numpy are available in the runtime environment.
    """
    from tdc.chem_utils.featurize.molconvert import smiles2maccs
    return smiles2maccs(s)


################################################################################
# Source: tdc.chem_utils.featurize.molconvert.smiles2mol
# File: tdc/chem_utils/featurize/molconvert.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_chem_utils_featurize_molconvert_smiles2mol(smiles: str):
    """Convert a SMILES string into an rdkit.Chem.rdchem.Mol object for use in TDC's
    molecule featurization and downstream ML pipelines.
    
    This function performs three steps: (1) canonicalize the input SMILES to a
    normalized/canonical SMILES representation (to ensure consistent representation
    across datasets and splits used in TDC benchmarks), (2) parse the canonical SMILES
    with RDKit via Chem.MolFromSmiles to produce an rdkit.Chem.rdchem.Mol, and
    (3) apply Chem.Kekulize to convert aromatic representations into an explicit
    Kekulé bond assignment in-place on the returned Mol. The returned rdkit Mol is
    the standard object used by TDC featurizers, descriptor calculators, and graph
    converters (e.g., for GNN inputs, oracles, and other molecule-processing
    functions). If RDKit cannot parse the SMILES, the function returns None to
    signal an invalid or unsupported SMILES string.
    
    Args:
        smiles (str): A SMILES (Simplified Molecular Input Line Entry System)
            string representing a small-molecule chemical structure. In the TDC
            context this is typically read from dataset entries and should be a
            valid SMILES. The function canonicalizes this string internally to
            produce a normalized representation so that identical molecules from
            different sources map to the same canonical form for consistent
            featurization, dataset splits, and benchmarking.
    
    Returns:
        rdkit.Chem.rdchem.Mol or None: An rdkit.Chem.rdchem.Mol instance corresponding
        to the input SMILES after canonicalization and in-place Kekulé assignment.
        This object is intended for downstream RDKit-based processing within TDC
        (descriptor computation, graph conversion for ML models, oracle evaluation,
        etc.). If RDKit fails to parse the canonical SMILES (invalid/unsupported
        input), the function returns None to indicate parsing failure.
    
    Behavior and side effects:
        The function calls canonicalize(smiles) to obtain a canonical SMILES string;
        canonicalize normalizes representation but does not modify caller data
        structures. Chem.MolFromSmiles is used to parse the canonical SMILES into a
        Mol; if parsing fails, the function immediately returns None. If parsing
        succeeds, Chem.Kekulize is called and modifies the returned Mol in-place to
        assign explicit single/double bonds (Kekulé form). The returned Mol should
        be treated as owned by the caller for subsequent processing. The function
        assumes RDKit is available and imported as Chem.
    
    Failure modes:
        - Invalid or syntactically incorrect SMILES will result in None being
          returned (parsing failed).
        - RDKit operations (parsing or kekulization) may raise errors; such errors
          will propagate unless caught by the caller. Users should handle None
          return values and consider wrapping calls in try/except if they need to
          catch RDKit-derived exceptions.
    """
    from tdc.chem_utils.featurize.molconvert import smiles2mol
    return smiles2mol(smiles)


################################################################################
# Source: tdc.chem_utils.featurize.molconvert.smiles2morgan
# File: tdc/chem_utils/featurize/molconvert.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_chem_utils_featurize_molconvert_smiles2morgan(
    s: str,
    radius: int = 2,
    nBits: int = 1024
):
    """tdc.chem_utils.featurize.molconvert.smiles2morgan converts a SMILES string into a fixed-length Morgan (circular) fingerprint suitable for machine learning workflows in the Therapeutics Data Commons (TDC) ecosystem. In TDC this function is used to featurize small-molecule representations (SMILES) into numeric vectors for downstream tasks such as ADMET prediction, activity screening, and benchmark dataset preparation where a consistent, fixed-size molecular descriptor is required.
    
    Args:
        s (str): Input SMILES string representing a small molecule. This is the canonical textual representation of a chemical structure that the function will first canonicalize using the module's canonicalize(s) helper and then parse with RDKit (Chem.MolFromSmiles). The SMILES string is the primary molecular identifier used across TDC datasets and tutorials to represent compounds.
        radius (int): Radius parameter for the Morgan fingerprint algorithm (default: 2). This integer controls the local neighborhood size around each atom that is considered when generating circular substructure identifiers; larger values capture larger substructures. The radius is a standard hyperparameter when computing Morgan (ECFP) fingerprints for molecular machine learning features.
        nBits (int): Length of the output fingerprint bit vector in bits (default: 1024). This integer sets the fixed dimensionality of the returned feature vector used across TDC pipelines so that all molecules produce descriptors of identical length for model input.
    
    Returns:
        numpy.array: A one-dimensional numpy array containing the fingerprint features for the input SMILES. On successful execution, the returned array represents the Morgan fingerprint of length equal to nBits; each element corresponds to a bit position in the fingerprint and encodes the presence (typically 1) or absence (typically 0) of hashed circular substructures as produced by RDKit's AllChem.GetMorganFingerprintAsBitVect and converted via DataStructs.ConvertToNumpyArray. If an exception occurs at any stage (for example, canonicalization failure, invalid SMILES parsing, missing RDKit installation, or fingerprint conversion error), the function prints an error message via print_sys indicating the SMILES that failed and returns a numpy array of zeros with length nBits as a deterministic fallback. This behavior ensures downstream TDC workflows receive a fixed-size numeric vector even when individual molecules cannot be processed.
    """
    from tdc.chem_utils.featurize.molconvert import smiles2morgan
    return smiles2morgan(s, radius, nBits)


################################################################################
# Source: tdc.chem_utils.featurize.molconvert.smiles2rdkit2d
# File: tdc/chem_utils/featurize/molconvert.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_chem_utils_featurize_molconvert_smiles2rdkit2d(s: str):
    """tdc.chem_utils.featurize.molconvert.smiles2rdkit2d converts a SMILES string into a 200-dimensional normalized RDKit 2D descriptor vector suitable for use as a fixed-length molecular feature vector in TDC molecular ML pipelines (for example, small-molecule property prediction and ADMET benchmarks described in the TDC README). The function canonicalizes the input SMILES, computes descriptors via descriptastorus's RDKit2DNormalized generator, replaces any NaN descriptor values with zeros, and returns the descriptor vector as a numpy.array of length 200.
    
    Args:
        s (str): A molecule representation in SMILES format. This argument is canonicalized by the function (via canonicalize(s)) to produce a standardized SMILES before descriptor computation; any error raised by canonicalize will propagate to the caller. The SMILES string represents the chemical structure to be featurized for downstream tasks such as molecular property prediction, dataset construction, or model input in TDC workflows.
    
    Returns:
        numpy.array: A one-dimensional numpy array containing 200 normalized RDKit 2D descriptor values for the input molecule (shape (200,)). The values are produced by descriptastorus.descriptors.rdNormalizedDescriptors.RDKit2DNormalized().process and the function discards the first element of the generator output (keeping elements [1:]). Any NaN entries in the computed descriptors are set to 0 before returning. If descriptor computation fails at runtime (for example, if the generator raises an exception while processing the canonicalized SMILES), the function prints a warning indicating the SMILES that failed and returns a numpy.array of zeros with length 200 as a fallback.
    
    Behavior, side effects, defaults, and failure modes:
        - Dependency requirement: The function requires the third-party package descriptastorus and pandas-flavor to be importable. If the import of descriptastorus.descriptors.rdDescriptors or rdNormalizedDescriptors fails, the function raises an ImportError with the message "Please install pip install git+https://github.com/bp-kelley/descriptastorus and pip install pandas-flavor" reflecting the exact installation advice used in the source code.
        - Canonicalization: The input SMILES is passed through canonicalize(s) to normalize representation; this can change the input string to a standardized form used for descriptor computation and downstream reproducibility.
        - Descriptor computation: The function instantiates rdNormalizedDescriptors.RDKit2DNormalized and calls its process method on the canonicalized SMILES. The resulting descriptor vector is taken from the returned sequence starting at index 1 (i.e., generator.process(s)[1:]).
        - NaN handling: After descriptor computation, any NaN values in the feature vector are replaced with 0 to avoid NaNs propagating into downstream ML models.
        - Fallback on processing error: If descriptor computation raises an exception (but the import succeeded), the function prints a diagnostic via the existing print_sys call stating that descriptastorus was not able to process that SMILES and that it will convert to all-zero features, and then returns a length-200 numpy.array of zeros. This fallback ensures the function always returns a numpy.array when imports succeed, but the returned all-zero vector should be treated as a missing/invalid-features indicator in downstream analyses.
        - Exceptions: Import errors for missing dependencies are raised immediately (see above). Other exceptions raised by canonicalize or by the descriptor generator may be propagated or handled as described (processing exceptions result in an all-zero vector and a printed warning).
    """
    from tdc.chem_utils.featurize.molconvert import smiles2rdkit2d
    return smiles2rdkit2d(s)


################################################################################
# Source: tdc.chem_utils.featurize.molconvert.smiles2selfies
# File: tdc/chem_utils/featurize/molconvert.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_chem_utils_featurize_molconvert_smiles2selfies(smiles: str):
    """tdc.chem_utils.featurize.molconvert.smiles2selfies: Convert a SMILES string into a SELFIES string for use in TDC molecule featurization and generation workflows.
    
    This function is part of the TDC chem_utils featurization utilities and is used to translate a molecule encoded as a SMILES string into the SELFIES representation. SMILES (Simplified Molecular-Input Line-Entry System) is a common text format for describing small-molecule chemical structures used throughout TDC datasets and tasks. SELFIES (Self-Referencing Embedded Strings) is a machine-learning-oriented, tokenizable molecular string format commonly used in generative models and oracles within TDC because it provides a robust representation for molecular generation. Internally, the function first canonicalizes the input SMILES (so that different equivalent SMILES map to a consistent representation) by calling canonicalize(smiles), then encodes the canonical SMILES to SELFIES by calling sf.encoder(smiles). There are no external side effects (no file I/O or global state changes); the operation is purely functional and returns the encoded string.
    
    Args:
        smiles (str): A SMILES string representing a small-molecule chemical structure. In TDC workflows this string is typically drawn from dataset entries or generated by molecular models. The function accepts a SMILES string in any (canonical or non-canonical) valid SMILES form; the function will first canonicalize the input to produce a deterministic intermediate representation prior to encoding. Passing a non-str value is not supported and will result in a runtime error from the called routines.
    
    Returns:
        selfies (str): A SELFIES string that encodes the same molecule represented by the input SMILES. This returned string is produced by sf.encoder after canonicalization and is intended for downstream tasks in TDC such as model tokenization, molecule generation, and oracle scoring. If the input SMILES is invalid or cannot be canonicalized/encoded, the underlying canonicalize or sf.encoder calls will raise an exception (for example, ValueError or a library-specific parsing error); such exceptions propagate to the caller.
    """
    from tdc.chem_utils.featurize.molconvert import smiles2selfies
    return smiles2selfies(smiles)


################################################################################
# Source: tdc.chem_utils.featurize.molconvert.smiles_lst2coulomb
# File: tdc/chem_utils/featurize/molconvert.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_chem_utils_featurize_molconvert_smiles_lst2coulomb(smiles_lst: list):
    """tdc.chem_utils.featurize.molconvert.smiles_lst2coulomb: Convert a list of SMILES strings into Coulomb-matrix features for molecular machine learning within the TDC workflows.
    
    This function is used in TDC (Therapeutics Data Commons) featurization pipelines to turn canonical 1D chemical representations (SMILES) into fixed-size numerical descriptors (Coulomb matrices) that can be consumed by downstream ML models for tasks such as property prediction, ADMET modeling, and benchmark evaluation. The implementation (from the local molconvert utilities) constructs internal Molecule objects from each SMILES string, performs a 3D geometry optimization with the UFF force field, and produces Coulomb matrix representations using a CoulombMatrix object configured with cm_type="UM" and n_jobs=-1. The output preserves input ordering so that each row of the returned array corresponds to the SMILES at the same index in smiles_lst.
    
    Args:
        smiles_lst (list): A Python list of SMILES strings. Each element must be a valid SMILES textual representation of a single molecule because the function calls Molecule(smiles, "smiles") to parse and then performs a UFF optimization to generate 3D coordinates. In the TDC domain, these SMILES typically come from dataset columns (e.g., molecule SMILES in ADMET or activity benchmarks) and represent the chemical structures to be featurized. Invalid or unparsable SMILES will raise exceptions when Molecule(...) is invoked or during geometry optimization.
    
    Returns:
        numpy.ndarray: A 2-D NumPy array of shape (nmol, max_atom_n**2), where nmol is len(smiles_lst) and max_atom_n is the maximum atom count among the input molecules. Each row is the Coulomb matrix for the corresponding input SMILES flattened in row-major order (i.e., features[i].reshape(max_atom_n, max_atom_n) gives the square Coulomb matrix for molecule i, with the actual Coulomb matrix occupying the top-left block). Smaller molecules are padded (zeros) to reach the fixed size max_atom_n**2 so that the array is rectangular and suitable for batch ML input. The array is produced by calling .to_numpy() on the internal representation, and its dtype is numeric (floating point) as returned by the CoulombMatrix representation.
    
    Behavior, defaults, side effects, and failure modes:
        - Geometry optimization: For each SMILES the function runs a UFF optimizer (optimizer="UFF" as coded) to produce 3D coordinates; this is required to compute interatomic distances for the Coulomb matrix. The optimizer and CoulombMatrix configuration are fixed by the implementation.
        - Parallelism: CoulombMatrix is constructed with n_jobs=-1, which attempts to use all available CPU cores for representation generation; this can increase throughput but may increase CPU usage on multi-core systems.
        - Preserves input order: The i-th row of the returned array corresponds to smiles_lst[i].
        - Memory and compute: The returned matrix size scales as O(nmol * max_atom_n^2). For datasets containing very large molecules (large atom counts) this can lead to high memory consumption or slow computation.
        - Errors: Invalid SMILES, failures in molecule parsing, problems during UFF optimization, or resource exhaustion (memory/CPU) will raise exceptions propagated from the underlying Molecule/CoulombMatrix implementations. The function does not perform SMILES correction, sanitization beyond what Molecule(...) performs, or catch those exceptions.
        - No external side effects: The function does not write to disk or modify global state; its observable effect is the returned NumPy array.
    """
    from tdc.chem_utils.featurize.molconvert import smiles_lst2coulomb
    return smiles_lst2coulomb(smiles_lst)


################################################################################
# Source: tdc.chem_utils.featurize.molconvert.xyzfile2selfies
# File: tdc/chem_utils/featurize/molconvert.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_chem_utils_featurize_molconvert_xyzfile2selfies(xyzfile: str):
    """tdc.chem_utils.featurize.molconvert.xyzfile2selfies converts a molecular structure in an XYZ-format file into a SELFIES string representation. This function is intended for use in Therapeutics Data Commons (TDC) workflows that require a robust, canonical, graph-based string encoding of a small molecule for downstream machine learning tasks such as molecule generation, featurization for property prediction, or oracle evaluation.
    
    The function implements the conversion pipeline used in this module: it reads the input XYZ file, converts the 3D coordinates and element labels to a SMILES string using xyzfile2smiles, canonicalizes that SMILES using canonicalize to produce a standardized representation, and then encodes the canonical SMILES into SELFIES using smiles2selfies. The resulting SELFIES string is suitable as an input representation for TDC molecule generation oracles and other components that accept SELFIES as a molecular string format.
    
    Args:
        xyzfile (str): Path to a file in XYZ format containing a single molecule's geometry and element labels. The string should be a filesystem path (relative or absolute) pointing to a readable XYZ file. In the TDC context, this argument provides the raw 3D structural information required to infer bonds and produce a SMILES representation; the function itself does not perform file writing or persistent side effects.
    
    Returns:
        str: A SELFIES string corresponding to the canonicalized SMILES derived from the input XYZ file. This SELFIES string is a tokenized, robust molecular representation intended for use in TDC molecule generation and featurization pipelines and for downstream evaluators and oracles that expect SELFIES input.
    
    Behavior and failure modes:
        The conversion follows three steps: XYZ -> SMILES (via xyzfile2smiles), SMILES canonicalization (via canonicalize), and SMILES -> SELFIES (via smiles2selfies). No files are created or modified by this function; it only reads the input file. If the input file path does not exist or is not readable, a FileNotFoundError or an I/O related exception will be raised by the underlying file operations. If the XYZ content is malformed, cannot be parsed, or cannot be mapped to a valid SMILES, the underlying xyzfile2smiles or canonicalize functions will raise an exception (for example ValueError or a domain-specific parsing error). If the SMILES produced cannot be converted to SELFIES, smiles2selfies will raise an exception. The caller should catch these exceptions as appropriate for their TDC pipeline. The function does not change global state and returns the SELFIES string on success.
    """
    from tdc.chem_utils.featurize.molconvert import xyzfile2selfies
    return xyzfile2selfies(xyzfile)


################################################################################
# Source: tdc.chem_utils.featurize.molconvert.xyzfile2smiles
# File: tdc/chem_utils/featurize/molconvert.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_chem_utils_featurize_molconvert_xyzfile2smiles(xyzfile: str):
    """tdc.chem_utils.featurize.molconvert.xyzfile2smiles converts an XYZ-format molecular geometry file into a canonical SMILES string using the module's conversion utilities. This function is intended for use in TDC workflows that require a text-based, connectivity-oriented molecular representation (SMILES) derived from 3D coordinate files for downstream featurization, dataset construction, oracle scoring, or model input.
    
    Args:
        xyzfile (str): Path to an input XYZ-format file or any string identifier that the underlying xyzfile2mol converter accepts. The string must point to a file containing a single molecule in standard XYZ format (atom labels and Cartesian coordinates). In the therapeutics domain (see TDC README), users commonly have small-molecule geometries saved as XYZ files from quantum chemistry or molecular mechanics packages; this parameter provides that file to the conversion pipeline. The function will read the file from disk (side effect: file I/O), parse the atomic symbols and coordinates, and hand the parsed geometry to the internal converter.
    
    Returns:
        str: A canonical SMILES string representing the molecular connectivity inferred from the input XYZ geometry. The returned SMILES is produced by the sequence of internal operations: xyzfile2mol(xyzfile) to obtain a molecular object, mol2smiles(mol) to convert that object to a SMILES string, and canonicalize(smiles) to produce a standardized (canonical) SMILES suitable for consistent downstream use in TDC datasets, featurizers, and benchmarks.
    
    Behavior and practical details:
        The function implements a three-step conversion pipeline (xyzfile2mol -> mol2smiles -> canonicalize) to transform 3D coordinates into a canonical 1D SMILES representation. This canonical SMILES is useful in TDC for dataset deduplication, molecule matching across datasets, feeding SMILES-based featurizers, and providing inputs to oracles and benchmarking pipelines described in the TDC README.
    
    Side effects:
        The primary side effect is reading the specified file from disk. The function does not modify global state or write files. All file I/O and parsing are performed by the underlying xyzfile2mol implementation.
    
    Failure modes and errors:
        If the file path is invalid, the XYZ file is malformed, contains multiple molecules when only one is expected, or contains unsupported atom labels or formats, the underlying conversion utilities (xyzfile2mol, mol2smiles, canonicalize) may raise exceptions (for example, file I/O errors, parsing errors, or chemical conversion errors). These exceptions propagate to the caller; this function does not catch them. Additionally, converting 3D coordinates to a connectivity-based representation may lose 3D stereochemical information in cases where the input geometry is ambiguous, and canonicalization may change the SMILES ordering to a standardized form (i.e., the returned SMILES may not preserve input atom ordering).
    
    Notes on domain significance:
        In TDC's therapeutics workflows, converting XYZ geometry files to canonical SMILES enables integration of computed or experimental 3D structures with SMILES-based datasets, ML featurizers, and oracles. Use this function when you need a deterministic, canonical text representation of a molecule derived from a 3D coordinate file for downstream machine learning pipelines, leaderboard submissions, or dataset preparation.
    """
    from tdc.chem_utils.featurize.molconvert import xyzfile2smiles
    return xyzfile2smiles(xyzfile)


################################################################################
# Source: tdc.chem_utils.oracle.oracle.SA
# File: tdc/chem_utils/oracle/oracle.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_chem_utils_oracle_oracle_SA(s: str):
    """tdc.chem_utils.oracle.oracle.SA: Compute the Synthetic Accessibility (SA) score for a molecule given its SMILES string. This function is used as a molecule-generation oracle within TDC to evaluate the synthesizability of candidate molecules during goal-oriented or distribution-learning tasks and other molecule-scoring workflows described in the TDC README.
    
    Args:
        s (str): A SMILES string that encodes the chemical structure of a small molecule. This argument should be an RDKit-parsable SMILES string. If s is None or cannot be parsed into an RDKit Mol object (Chem.MolFromSmiles returns None), the function returns a sentinel score of 100 to indicate an invalid or unscorable input. The function calls RDKit's Chem.MolFromSmiles to convert the SMILES string to a molecule and then calls calculateScore(mol) to compute the SA score; any exceptions raised by RDKit or calculateScore (for example, if RDKit is not available or calculateScore fails) are not caught here and will propagate to the caller.
    
    Returns:
        float: The Synthetic Accessibility (SA) score for the input molecule as a floating-point numeric value. In the context of TDC molecule generation oracles, this score is used to quantify how synthetically accessible a proposed molecule is (used to rank or guide generation). A return value of 100 is used as a special sentinel to indicate that the input was None or not a valid SMILES string and therefore could not be scored. The function has no other side effects beyond calling RDKit and calculateScore.
    """
    from tdc.chem_utils.oracle.oracle import SA
    return SA(s)


################################################################################
# Source: tdc.chem_utils.oracle.oracle.askcos
# File: tdc/chem_utils/oracle/oracle.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_chem_utils_oracle_oracle_askcos(
    smiles: str,
    host_ip: str,
    output: str = "plausibility",
    save_json: bool = False,
    file_name: str = "tree_builder_result.json",
    num_trials: int = 5,
    max_depth: int = 9,
    max_branching: int = 25,
    expansion_time: int = 60,
    max_ppg: int = 100,
    template_count: int = 1000,
    max_cum_prob: float = 0.999,
    chemical_property_logic: str = "none",
    max_chemprop_c: int = 0,
    max_chemprop_n: int = 0,
    max_chemprop_o: int = 0,
    max_chemprop_h: int = 0,
    chemical_popularity_logic: str = "none",
    min_chempop_reactants: int = 5,
    min_chempop_products: int = 5,
    filter_threshold: float = 0.1,
    return_first: str = "true"
):
    """tdc.chem_utils.oracle.oracle.askcos performs a retrosynthetic analysis query against a running ASKCOS server and returns a single scalar score or metric used as a molecule-generation oracle in the TDC (Therapeutics Data Commons) framework. This function is used by TDC molecule generation workflows to query ASKCOS (ASKCOS GitHub: https://github.com/connorcoley/ASKCOS) running behind an HTTP endpoint and to extract one of several summary metrics (plausibility score, estimated number of synthetic steps, synthesizability, or price) for a target molecule represented as a SMILES string. The function sends an initial price query and, if the price response indicates zero, invokes the ASKCOS Tree Builder API with configurable search and filtering parameters, optionally saves the raw JSON response to disk, analyzes the returned tree structure via tree_analysis, and returns the requested scalar metric. The function prints a short status line for each attempt when retrying the Tree Builder request.
    
    Args:
        smiles (str): The target molecule encoded as a SMILES string. In TDC generation tasks this is the candidate molecule to be scored by ASKCOS for retrosynthetic plausibility, synthesizability, estimated price, or estimated number of synthetic steps. It is passed verbatim to ASKCOS API endpoints as the "smiles" parameter.
        host_ip (str): The base URL (including protocol and host, e.g., "http://127.0.0.1:5000") of a running ASKCOS server that accepts REST requests. Per the ASKCOS project, the server can be run via Docker; this string is concatenated with known API paths ("/api/price/" and "/api/treebuilder/") to build request URLs. The function issues HTTP GET requests to this host and uses verify=False in requests calls (disables SSL verification).
        output (str): Which scalar metric to return from the ASKCOS analysis. Must be one of "num_step", "plausibility", "synthesizability", or "price". "plausibility" returns the ASKCOS plausibility score (commonly used as an oracle target in generation); "num_step" returns the estimated retrosynthetic tree depth (number of steps); "synthesizability" returns ASKCOS synthesizability metric; "price" returns the price estimate returned by the ASKCOS price API. If a value outside the allowed set is provided, the function raises a NameError before issuing any network requests.
        save_json (bool): If True, the raw JSON response returned by the ASKCOS endpoint (either the price response or the treebuilder response) is written to disk. This is a side effect used for debugging or archival of ASKCOS outputs; defaults to False. The file is written using a standard JSON dump and will overwrite an existing file with the same name.
        file_name (str): File path/name used to save the raw JSON response when save_json is True. Default is "tree_builder_result.json". The path is relative to the current working directory unless an absolute path is provided. File I/O errors (e.g., permissions, disk full) may be raised by the underlying open()/json.dump() calls.
        num_trials (int): Number of attempts to contact the ASKCOS Tree Builder endpoint if the first treebuilder response contains an "error" key. For each attempt the function prints "Trying to send the request, for the X times now". Defaults to 5. If all attempts contain an "error" key, the last response (error-containing JSON) proceeds to downstream analysis and may cause tree_analysis or subsequent logic to raise an exception.
        max_depth (int): Passed to ASKCOS Tree Builder as "max_depth". Controls the maximum retrosynthesis tree depth (maximum number of sequential reaction steps allowed) during the search; in TDC this constrains how deep a retrosynthetic route the oracle will consider. Default 9.
        max_branching (int): Passed as "max_branching". Sets the maximum branching factor for the search tree (how many child expansions are allowed per node) and thus controls search breadth and computational cost. Default 25.
        expansion_time (int): Passed as "expansion_time" (seconds). Controls the time budget allowed for the ASKCOS Tree Builder expansion/search process. Larger values allow longer searches. Default 60.
        max_ppg (int): Passed as "max_ppg". Controls the maximum number of products or product-generating events per generation/expansion step in the ASKCOS Tree Builder (affects the number of candidate reactions considered per node). Default 100.
        template_count (int): Passed as "template_count". Instructs ASKCOS how many retrosynthetic templates to consider during expansion; increasing this can increase coverage at the cost of compute time. Default 1000.
        max_cum_prob (float): Passed as "max_cum_prob". Cumulative probability cutoff used during template ranking/pruning inside ASKCOS; the search will consider templates accumulating up to this cumulative model probability. Default 0.999.
        chemical_property_logic (str): Passed as "chemical_property_logic". Optional logic parameter name used by ASKCOS to enable chemical-property-based filtering. Accepted logic strings are controlled by the ASKCOS server; by default "none" disables chemical-property-based constraints. In TDC use this to restrict candidate expansions by properties predicted by chemprop models when the ASKCOS server supports such filtering.
        max_chemprop_c (int): Passed as "max_chemprop_c". Integer threshold forwarded to ASKCOS chemprop/property-based filtering, commonly used to restrict predicted carbon-related properties (e.g., maximum predicted value or count) when chemical_property_logic is enabled. Default 0 (disabled).
        max_chemprop_n (int): Passed as "max_chemprop_n". Integer threshold forwarded to ASKCOS chemprop/property-based filtering for nitrogen-related property constraints. Default 0 (disabled).
        max_chemprop_o (int): Passed as "max_chemprop_o". Integer threshold forwarded to ASKCOS chemprop/property-based filtering for oxygen-related property constraints. Default 0 (disabled).
        max_chemprop_h (int): Passed as "max_chemprop_h". Integer threshold forwarded to ASKCOS chemprop/property-based filtering for hydrogen-related property constraints. Default 0 (disabled).
        chemical_popularity_logic (str): Passed as "chemical_popularity_logic". Optional logic parameter name used by ASKCOS to enable popularity-based filtering of templates or reactions (e.g., require templates to exceed frequency thresholds); default "none" disables popularity-based filtering.
        min_chempop_reactants (int): Passed as "min_chempop_reactants". Minimum frequency count threshold for popular reactant templates when chemical_popularity_logic is enabled; templates with reactant counts below this threshold may be excluded by the ASKCOS server. Default 5.
        min_chempop_products (int): Passed as "min_chempop_products". Minimum frequency count threshold for popular product templates when chemical_popularity_logic is enabled; default 5.
        filter_threshold (float): Passed as "filter_threshold". A floating threshold used by ASKCOS Tree Builder to filter low-probability templates or reactions; higher values result in stricter filtering. Default 0.1.
        return_first (str): Passed as "return_first". Controls whether ASKCOS should return only the first found route ("true") or multiple routes depending on server implementation. Default "true". The exact semantics are determined by the ASKCOS Tree Builder API.
    
    Behavior, side effects, defaults, and failure modes:
        The function first issues an HTTP GET to host_ip + "/api/price/" with "smiles" to query price. If the returned JSON contains "price" equal to 0, the function assembles Tree Builder parameters (as listed above) and issues up to num_trials HTTP GET requests to host_ip + "/api/treebuilder/" with those parameters. For each Tree Builder attempt the function prints a short progress message; it stops retrying early if the returned JSON does not contain an "error" key. All requests use requests.get with verify=False (SSL verification disabled). If save_json is True the final JSON response (from the price endpoint when price != 0, or the last treebuilder response when price == 0) is written to disk at file_name using json.dump, which may overwrite an existing file. After receiving the JSON response, the function calls tree_analysis(resp.json()) to extract summary metrics. The mapping from the selected output to the returned scalar is enforced in the function and an invalid output value causes a NameError before any network traffic.
    
        Common failure modes that callers should handle or be aware of:
        - Invalid output value: raises NameError immediately.
        - Network errors (requests.exceptions.RequestException) if the ASKCOS host is unreachable, times out, or connection fails.
        - Non-JSON or unexpected JSON responses: accessing resp.json()["price"] or later expected keys may raise KeyError or ValueError if the server response is malformed.
        - If the price endpoint returns a non-zero price, the function will not call the Tree Builder endpoint and will pass the price-response JSON into tree_analysis, which may be unsuitable and can raise exceptions depending on tree_analysis expectations.
        - File I/O errors when save_json is True (e.g., permissions, disk space) will propagate from open()/json.dump.
        - SSL verification is disabled (verify=False) which can expose requests to man-in-the-middle risks; use only with trusted hosts or in controlled environments.
    
    Returns:
        float or int: A single scalar metric extracted from ASKCOS via tree_analysis or the price endpoint, chosen according to the output parameter:
            If output == "plausibility": returns the ASKCOS plausibility score (p_score) for the best retrosynthetic route; this float is intended as a measure of how chemically plausible the proposed retrosynthesis is and is commonly used by TDC as an oracle objective.
            If output == "num_step": returns the retrosynthetic tree depth (depth) as an integer, representing the estimated number of synthetic steps.
            If output == "synthesizability": returns the synthesizability metric (synthesizability) produced by tree_analysis/ASKCOS as a numeric score; used to quantify ease of synthesis.
            If output == "price": returns the numeric price estimate (price) returned by the ASKCOS price API.
        Note: the exact numeric types are determined by ASKCOS/tree_analysis outputs and may be int or float. Exceptions from network, JSON decoding, or tree_analysis are propagated to the caller.
    """
    from tdc.chem_utils.oracle.oracle import askcos
    return askcos(
        smiles,
        host_ip,
        output,
        save_json,
        file_name,
        num_trials,
        max_depth,
        max_branching,
        expansion_time,
        max_ppg,
        template_count,
        max_cum_prob,
        chemical_property_logic,
        max_chemprop_c,
        max_chemprop_n,
        max_chemprop_o,
        max_chemprop_h,
        chemical_popularity_logic,
        min_chempop_reactants,
        min_chempop_products,
        filter_threshold,
        return_first
    )


################################################################################
# Source: tdc.chem_utils.oracle.oracle.canonicalize
# File: tdc/chem_utils/oracle/oracle.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_chem_utils_oracle_oracle_canonicalize(
    smiles: str,
    include_stereocenters: bool = True
):
    """tdc.chem_utils.oracle.oracle.canonicalize: Produce a deterministic, RDKit-based canonical SMILES string from an input SMILES. This function is used throughout TDC for tasks that require a stable molecular representation (for example, molecule generation oracles, dataset canonicalization, deduplication, and downstream evaluation) and follows the canonicalization behavior implemented by RDKit; the underlying algorithmic approach is discussed in https://pubs.acs.org/doi/full/10.1021/acs.jcim.5b00543.
    
    Args:
        smiles (str): Input SMILES string to canonicalize. This is the textual molecular representation accepted by RDKit's Chem.MolFromSmiles. In the context of TDC, providing canonicalized SMILES ensures deterministic identifiers for molecules used by oracles, dataset loaders, and evaluation pipelines. The function expects a string in valid SMILES format; passing a non-string value may result in an error from RDKit or from the Python runtime.
        include_stereocenters (bool): Whether to preserve stereochemical information in the returned canonical SMILES. When True (the default), RDKit's isomeric SMILES output is enabled (Chem.MolToSmiles(..., isomericSmiles=True)), so chiral centers and double-bond stereochemistry markers are retained in the canonical string. When False, stereochemical markers are omitted, producing a canonical SMILES that ignores stereochemistry—useful when stereochemistry is irrelevant for a task such as coarse-grained dataset deduplication. The default value is True to preserve stereochemical detail commonly required in drug discovery and molecular generation tasks.
    
    Returns:
        str or None: The canonicalized SMILES string produced by RDKit (Chem.MolToSmiles) when the input SMILES can be parsed into a valid molecule. The output is RDKit's canonical (deterministic) SMILES representation; its stereochemical content depends on include_stereocenters. Returns None if RDKit fails to parse the input SMILES into a molecule (i.e., the molecule is invalid or cannot be interpreted). No other side effects occur; the function performs no I/O and does not modify global state. Note that this function depends on RDKit being available (the implementation calls RDKit's Chem.MolFromSmiles and Chem.MolToSmiles), so attempting to call it without RDKit imported/installed may raise an ImportError or NameError at import/runtime.
    """
    from tdc.chem_utils.oracle.oracle import canonicalize
    return canonicalize(smiles, include_stereocenters)


################################################################################
# Source: tdc.chem_utils.oracle.oracle.drd2
# File: tdc/chem_utils/oracle/oracle.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_chem_utils_oracle_oracle_drd2(smile: str):
    """tdc.chem_utils.oracle.oracle.drd2: Evaluate the predicted DRD2 activity score for a molecule given its SMILES string.
    
    This function is an oracle-style scorer used in TDC (Therapeutics Data Commons) molecule generation and evaluation pipelines to assign a quantitative DRD2 activity likelihood to a candidate small molecule. It parses the input SMILES string with RDKit, converts the parsed molecule to the feature/fingerprint representation expected by the pretrained DRD2 classifier, and returns the classifier's positive-class score as a Python float. The function lazily loads a module-level pretrained model (drd2_model) via load_drd2_model() on first invocation and stores it as a global variable, so subsequent calls reuse the loaded model and avoid repeated disk or network loads.
    
    Args:
        smile (str): A SMILES string representing a small-molecule chemical structure. This parameter is the canonical textual input to the function: the function calls RDKit.Chem.MolFromSmiles(smile) to parse it. The SMILES value is used to construct molecular fingerprints via fingerprints_from_mol(mol) and to compute the DRD2 score with the pretrained classifier. If the SMILES cannot be parsed into an RDKit Mol, the function returns 0.0 (see failure modes). The caller is responsible for providing a valid SMILES string when expecting a meaningful score.
    
    Returns:
        float: The DRD2 score produced by the pretrained classifier for the positive/activity class. Concretely, the function calls drd2_model.predict_proba(fp)[:, 1] on the fingerprint vector and converts the resulting single-value array to a Python float before returning it. The returned float is intended to represent the model's confidence / probability-like score that the molecule is active on the dopamine D2 receptor (DRD2) and is typically used to rank or guide molecule generation and optimization. If the SMILES string cannot be parsed by RDKit (Chem.MolFromSmiles returns None), the function returns 0.0. Note that other failures can raise exceptions: failures loading the pretrained model via load_drd2_model(), errors in fingerprints_from_mol(), or errors inside the model.predict_proba call will propagate as exceptions to the caller. The function also has a side effect of creating and caching a global variable named drd2_model in the module namespace when the model is first loaded.
    """
    from tdc.chem_utils.oracle.oracle import drd2
    return drd2(smile)


################################################################################
# Source: tdc.chem_utils.oracle.oracle.gsk3b
# File: tdc/chem_utils/oracle/oracle.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_chem_utils_oracle_oracle_gsk3b(smiles: str):
    """tdc.chem_utils.oracle.oracle.gsk3b evaluates a pre-trained GSK3B classification oracle on a single molecule represented by a SMILES string and returns a probability-like score used in molecule generation and scoring workflows.
    
    This function is used within TDC's molecule generation oracles to provide a scalar objective indicating the predicted likelihood that the input molecule interacts with or is active against the GSK3 beta (GSK3B) target. It converts the input SMILES to an RDKit molecule, computes a 2048-bit Morgan fingerprint (radius=2), converts that fingerprint to a NumPy feature vector, and then applies a cached scikit-learn style classifier (loaded via load_gsk3b_model()) to obtain the predicted probability for the positive class. The function caches the loaded model in the module global gsk3_model on first call to avoid repeated expensive model-loading operations (side effect). This cached global persists for the Python process lifetime or until reassigned.
    
    Args:
        smiles (str): A SMILES string encoding the chemical structure of a single small molecule. This argument is the canonical input to the oracle; the function will attempt to parse it with RDKit (smiles_to_rdkit_mol). The SMILES should represent a valid small-molecule structure; invalid or unparsable SMILES will cause RDKit to fail and typically raise an exception (for example, ValueError or RDKit-specific parsing errors), which the caller should catch and handle. The function does not perform SMILES sanitization beyond the RDKit conversion step.
    
    Returns:
        float: A scalar score between 0 and 1 (inclusive) representing the classifier's predicted probability that the input molecule belongs to the active/positive class for GSK3B. Practically, values near 1 indicate higher predicted activity (higher oracle score) and values near 0 indicate lower predicted activity. The returned value is produced by calling gsk3_model.predict_proba on a single-row feature matrix; if the underlying model lacks predict_proba or returns an unexpected array shape, an exception (for example, AttributeError or IndexError) may be raised.
    
    Behavior and side effects:
        - On first invocation in a Python session, the function loads the underlying classifier by calling load_gsk3b_model() and stores it in the module-level global variable gsk3_model. Subsequent calls reuse this cached model to reduce latency and I/O overhead.
        - The molecular representation uses RDKit to create an RDKit Mol object and AllChem.GetMorganFingerprintAsBitVect with radius=2 and nBits=2048 to produce the fingerprint. The fingerprint is converted to a NumPy array with DataStructs.ConvertToNumpyArray and reshaped to (1, -1) before prediction.
        - The function assumes the model's predict_proba returns an array where column index 1 corresponds to the positive class probability; it returns that [0, 1] element. If the model's class ordering differs, the returned probability may not correspond to the intended positive class.
        - No file I/O is performed by this function itself besides whatever load_gsk3b_model() performs; the primary observable side effect is the population of the global gsk3_model cache.
    
    Failure modes and errors:
        - Invalid or unparsable SMILES will typically raise RDKit parsing errors or result in None from smiles_to_rdkit_mol; callers should validate or catch exceptions.
        - If load_gsk3b_model() fails (missing files, corrupted model), the function will propagate that exception.
        - If the model does not implement predict_proba or returns fewer than two probability columns, AttributeError or IndexError may be raised.
        - If fingerprint conversion fails (unexpected RDKit behavior), corresponding exceptions from RDKit or NumPy may be raised.
    
    Practical significance in TDC:
        - This oracle function is intended for use in goal-directed molecule generation, scoring candidate molecules during optimization, and benchmarking generative models against a learned prediction for GSK3B activity. The output is suitable as a scalar reward or objective in such pipelines.
    """
    from tdc.chem_utils.oracle.oracle import gsk3b
    return gsk3b(smiles)


################################################################################
# Source: tdc.chem_utils.oracle.oracle.ibm_rxn
# File: tdc/chem_utils/oracle/oracle.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_chem_utils_oracle_oracle_ibm_rxn(
    smiles: str,
    api_key: str,
    output: str = "confidence",
    sleep_time: int = 30
):
    """tdc.chem_utils.oracle.oracle.ibm_rxn performs an automatic retrosynthesis query against the IBM RXN for Chemistry service using the rxn4chemistry client and returns either a numeric retrosynthetic confidence score or the raw prediction result. This function is used within TDC as an oracle-style utility for molecule generation and evaluation tasks (see TDC oracles in the README), where retrosynthetic plausibility or confidence can be used as a scoring function for generated SMILES.
    
    This function constructs an RXN4ChemistryWrapper with the provided api_key, creates a project named "test" on the IBM RXN service, requests an automatic retrosynthesis prediction for the supplied product SMILES, and polls the service at intervals controlled by sleep_time until the job status becomes "SUCCESS". Depending on the output parameter, it either extracts and returns the confidence from the first retrosynthetic path or returns the entire results structure produced by the rxn4chemistry client.
    
    Notes on practical significance and domain role: in molecule generation and optimization workflows (generation problem in TDC), retrosynthetic confidence indicates how readily a candidate molecule might be synthesized according to IBM RXN's retrosynthesis engine; using this function as an oracle helps rank or filter generated molecules by synthetic accessibility as judged by an automated retrosynthesis model.
    
    Behavior, side effects, defaults, and failure modes:
    - The function requires the external package rxn4chemistry and an active internet connection to contact the IBM RXN service. If rxn4chemistry is not importable, the function prints an installation hint (and may subsequently raise an exception depending on the runtime environment).
    - The function creates a project on the IBM RXN service with the literal name "test". This is a side effect on the remote account associated with the api_key and may be visible in that account's dashboard or usage logs.
    - The call is blocking: the function polls the remote job status in a loop and will sleep for sleep_time seconds between polls. The default sleep_time is 30 seconds; increase or decrease this to manage polling frequency and overall latency.
    - API authentication, network errors, or service-side failures (e.g., job rejection, timeouts, or unexpected response structures) can raise exceptions from the rxn4chemistry client or from dictionary access (KeyError). If output is not "confidence" or "result", the function raises NameError with the message "This output value is not implemented."
    - The confidence value returned when output is "confidence" is taken from results["retrosynthetic_paths"][0]["confidence"], i.e., the first predicted retrosynthetic path's confidence as provided by IBM RXN. The precise numeric scale and meaning of this confidence are defined by the IBM RXN service and should be interpreted according to that service's documentation.
    
    Args:
        smiles (str): A product molecule represented as a SMILES string. This is the input chemical structure for which the function requests an automatic retrosynthesis prediction from IBM RXN. In TDC generation workflows, this SMILES would typically be a candidate generated molecule to be evaluated for synthetic plausibility.
        api_key (str): The API key (string) used to authenticate to the IBM RXN (rxn4chemistry) service. This key identifies the remote account under which the project is created and the retrosynthesis job is run; usage may count against that account's API quota or billing.
        output (str): Controls the returned object. If "confidence" (the default), the function extracts and returns the confidence value of the first retrosynthetic path from the prediction results (results["retrosynthetic_paths"][0]["confidence"]). If "result", the function returns the full result object returned by the rxn4chemistry client (the raw prediction dictionary). Any other value causes a NameError to be raised. Do not pass additional output modes unless the calling code or rxn4chemistry client is updated to support them.
        sleep_time (int): Number of seconds to sleep between polling attempts while waiting for the retrosynthesis job to complete. The function polls repeatedly until the returned results["status"] equals "SUCCESS". The default value is 30 seconds; increase to reduce polling frequency (longer latency), or decrease to poll more frequently (may increase API usage).
    
    Returns:
        float or dict: If output == "confidence", returns the numeric confidence (float) extracted from results["retrosynthetic_paths"][0]["confidence"] representing IBM RXN's confidence for the first predicted retrosynthetic path. If output == "result", returns the full prediction result object as provided by the rxn4chemistry client (typically a dict-like JSON structure containing keys such as "status", "prediction_id", and "retrosynthetic_paths"). Exceptions (e.g., ImportError/NameError/KeyError, network or API errors) may be raised instead of returning a value when prerequisites are missing or when the remote service returns unexpected responses.
    """
    from tdc.chem_utils.oracle.oracle import ibm_rxn
    return ibm_rxn(smiles, api_key, output, sleep_time)


################################################################################
# Source: tdc.chem_utils.oracle.oracle.load_pickled_model
# File: tdc/chem_utils/oracle/oracle.py
# Category: valid
################################################################################

def tdc_chem_utils_oracle_oracle_load_pickled_model(name: str):
    """tdc.chem_utils.oracle.oracle.load_pickled_model loads a pretrained Python object serialized with the pickle module from a file on disk. In the Therapeutics Data Commons (TDC) project this is typically used to restore pretrained machine learning models (for example, scikit-learn estimators) that serve as molecule-generation oracles or other prediction components in molecule-generation and property-prediction workflows described in the TDC README and tutorials.
    
    Args:
        name (str): Path or filename of the pickled file to load. This should be a string pointing to a file accessible to the running process (absolute or relative path). The file is expected to contain a Python object serialized with pickle (commonly a pretrained sklearn model used by TDC oracles). Supplying this path tells the function where to read the binary pickle data from; there is no default — callers must provide a valid path string.
    
    Returns:
        The model: The deserialized Python object loaded from the pickle file. In the TDC context this is typically a pretrained model object (for example, a scikit-learn estimator) that can be used directly by oracle functions or downstream prediction code. The function returns the unpickled object for immediate use; it does not perform type checking or conversion beyond pickle deserialization.
    
    Behavior, side effects, and failure modes:
        The function opens the file at the given path in binary read mode and calls pickle.load to reconstruct the original Python object. If the file cannot be opened, builtin exceptions such as FileNotFoundError or PermissionError will propagate to the caller. If the pickle data is malformed or incompatible with the current Python environment, pickle.UnpicklingError (or other exceptions during unpickling) may be raised and will propagate. If pickle.load raises EOFError (for example, due to an incomplete download or an empty file), the function imports sys and calls sys.exit with a specific message indicating that TDC's hosted resources (Harvard Dataverse) may be under maintenance; this causes the Python process to terminate with the provided message. No other network activity or retries are performed by this function; it performs a single local file open and unpickle operation and returns the reconstructed object on success.
    """
    from tdc.chem_utils.oracle.oracle import load_pickled_model
    return load_pickled_model(name)


################################################################################
# Source: tdc.chem_utils.oracle.oracle.parse_molecular_formula
# File: tdc/chem_utils/oracle/oracle.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_chem_utils_oracle_oracle_parse_molecular_formula(formula: str):
    """tdc.chem_utils.oracle.oracle.parse_molecular_formula: Parse a molecular formula string to extract element symbols and their integer counts for use in TDC molecule-generation oracles and data-processing functions.
    
    This function takes a plain chemical molecular formula (for example "C8H3F3Br") and returns the elemental composition as a sequence of (element_symbol, count) pairs. It is used in TDC's chem_utils and oracle code paths to provide a simple, fast breakdown of a formula for downstream scoring, filtering, or feature extraction in molecule generation and evaluation workflows. The implementation uses a regular expression to find contiguous element tokens and their optional numeric counts, preserves the order of appearance in the input string, and treats any omitted numeric count as 1.
    
    Args:
        formula (str): A molecular formula string to parse, e.g., "C8H3F3Br". The string must contain element symbols that begin with an uppercase ASCII letter and may include subsequent lowercase ASCII letters, optionally followed immediately by an integer count. The function expects a plain formula without grouped parentheses, charge annotations, isotopic notation, decimal stoichiometries, hydrates, or other nonstandard tokens; such constructs will not be expanded or interpreted and may yield partial or unexpected results. Passing a non-string value will typically raise a TypeError when processed by the underlying regular expression function.
    
    Returns:
        list[tuple[str, int]]: A list of tuples where each tuple is (element_symbol, count). element_symbol is the element token exactly as found in the formula (a str) and count is an integer representing the number of atoms of that element in the formula; if the numeric count is omitted in the input it is returned as 1. The order of tuples in the returned list follows the order in which the element tokens appear in the input string. The function does not validate element_symbol against the periodic table and does not perform expansion of grouped subformulas, so callers requiring full chemical validation or handling of parentheses/charges should preprocess the formula or use a dedicated chemistry parser.
    
    Behavioral notes and failure modes: The function is pure (no side effects, no I/O) and has linear-time behavior relative to the length of the input string. It uses the regular expression r"([A-Z][a-z]*)(\d*)" to identify tokens. It will return an empty list for an empty string. For inputs containing nonmatching characters (for example parentheses or symbols), only the matching element/count pairs are returned; such inputs may therefore produce results that do not reflect the true chemical composition unless pre-normalized. Malformed numeric substrings are not expected because the regex only captures contiguous digits; nonetheless, if a non-string is provided an exception (e.g., TypeError) will be raised by the regex machinery.
    """
    from tdc.chem_utils.oracle.oracle import parse_molecular_formula
    return parse_molecular_formula(formula)


################################################################################
# Source: tdc.chem_utils.oracle.oracle.penalized_logp
# File: tdc/chem_utils/oracle/oracle.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_chem_utils_oracle_oracle_penalized_logp(s: str):
    """Compute the penalized logP score for a molecule given as a SMILES string.
    
    This function implements a commonly used oracle in molecule generation benchmarks (as provided in TDC's "Molecule Generation Oracles" utilities) that combines three components relevant to drug-like molecule design: lipophilicity (logP), synthetic accessibility (SA), and a cycle/ring-size penalty. It parses the input SMILES with RDKit to obtain a molecule, computes MolLogP to quantify lipophilicity, computes an SA score via the calculateScore routine (a negative SA value increases the final score), and computes a cycle penalty based on the largest ring size beyond six atoms. Each component is standardized using fixed mean and standard deviation constants embedded in the implementation, and the final score is the sum of the three normalized components. This score is commonly used as an objective or oracle for goal-directed molecular generation and optimization tasks in TDC benchmarks.
    
    Args:
        s (str): A SMILES string representing the small-molecule to evaluate. The function expects a valid SMILES string parseable by RDKit's Chem.MolFromSmiles. The SMILES string is the canonical textual representation used throughout TDC for small-molecule datasets and generation tasks. If s is None or cannot be parsed into a molecule by RDKit, the function returns a sentinel value of -100.0 (see failure modes below).
    
    Returns:
        float: The penalized logP score for the input molecule. This value is computed as:
            normalized_log_p + normalized_SA + normalized_cycle
        where:
            normalized_log_p = (MolLogP(mol) - 2.4570953396190123) / 1.434324401111988
            normalized_SA = (SA - (-3.0525811293166134)) / 0.8335207024513095
            normalized_cycle = (cycle_score - (-0.0485696876403053)) / 0.2860212110245455
        and cycle_score is -max(0, largest_ring_size - 6). The numeric constants (means and standard deviations) are hard-coded in the implementation and reflect the normalization used by this oracle. The returned float may in principle range from negative to positive infinity, but in practice is bounded by plausible molecular properties. If the input is invalid (None or unparsable), the function returns -100.0 as a failure sentinel.
    
    Behavior, side effects, defaults, and failure modes:
        This function has no side effects (it does not modify external state or files) and purely computes a floating-point score from its input string. It relies on RDKit for SMILES parsing and descriptor calculation (Chem.MolFromSmiles and Descriptors.MolLogP), on a calculateScore implementation (used to produce the SA component, typically from a sascorer utility), and on networkx for cycle detection (nx.cycle_basis applied to the molecule adjacency matrix). If RDKit fails to parse the SMILES (Chem.MolFromSmiles returns None), the function treats the input as invalid and returns -100.0. If required dependencies (RDKit, networkx, or the calculateScore function) are not available in the running environment, calling this function will result in an import or NameError at module import or runtime; these dependency errors are not handled internally by this function. The hard-coded normalization constants and cycle-penalty definition are part of the oracle's specification and should not be altered externally; changing them would change the scoring semantics used in TDC molecule generation benchmarks.
    """
    from tdc.chem_utils.oracle.oracle import penalized_logp
    return penalized_logp(s)


################################################################################
# Source: tdc.chem_utils.oracle.oracle.qed
# File: tdc/chem_utils/oracle/oracle.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_chem_utils_oracle_oracle_qed(smiles: str):
    """tdc.chem_utils.oracle.oracle.qed evaluates the Quantitative Estimate of Drug-likeness (QED) score for a molecule given its SMILES representation. This function is used in TDC as a molecule-scoring oracle for generation and optimization tasks (see TDC "Molecule Generation Oracles") and provides a single scalar reward indicating predicted drug-likeness according to RDKit's QED implementation.
    
    Args:
        smiles (str): A SMILES string encoding the small-molecule structure to be scored. In practice within TDC, this string is parsed with RDKit via Chem.MolFromSmiles to construct an RDKit Mol object before scoring. If the caller provides None or a SMILES string that RDKit cannot parse into a valid molecule, the function treats the input as invalid and returns 0.0 instead of raising an error. The parameter is expected to be a Python str containing a valid (canonical or non-canonical) SMILES; other types are not supported by the signature.
    
    Returns:
        float: qed_score, a scalar between 0.0 and 1.0 inclusive that represents the QED (quantitative estimate of drug-likeness) computed by RDKit's QED.qed routine. Higher values indicate greater estimated drug-likeness. If smiles is None or RDKit fails to parse the SMILES into a molecule, the function returns 0.0. There are no persistent side effects; the function is deterministic for a given SMILES and RDKit version. Note that this implementation depends on RDKit's Chem and QED modules being available in the runtime environment; if those modules are missing or not imported correctly elsewhere in the package, calling this function may result in an error at import or runtime outside the normal invalid-SMILES handling.
    """
    from tdc.chem_utils.oracle.oracle import qed
    return qed(smiles)


################################################################################
# Source: tdc.chem_utils.oracle.oracle.similarity
# File: tdc/chem_utils/oracle/oracle.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_chem_utils_oracle_oracle_similarity(smiles_a: str, smiles_b: str):
    """tdc.chem_utils.oracle.oracle.similarity evaluates the Tanimoto (fingerprint) similarity between two molecular representations given as SMILES strings. This function is used within TDC as a lightweight molecular similarity oracle for molecule generation and evaluation tasks (for example, scoring generated candidates against target molecules in goal-oriented generation), and it implements a specific, fixed fingerprinting protocol to ensure reproducible comparisons across calls.
    
    The function computes Morgan (circular) fingerprints with radius 2, 2048 bits, and chirality disabled, using RDKit's Chem/AllChem/DataStructs modules, and then returns the RDKit-computed Tanimoto similarity between the two bit vectors. The returned similarity is a numerical proxy for structural similarity between small molecules and is commonly used in drug discovery workflows to prioritize candidates, measure diversity, or define optimization objectives.
    
    Args:
        smiles_a (str): SMILES string for molecule A. This is the canonical input representation expected by RDKit's Chem.MolFromSmiles. In the TDC context, smiles_a typically represents either a reference molecule in an oracle scoring task or one of a pair of molecules being compared. If smiles_a is None or cannot be parsed by RDKit into a molecule object, the function will not raise an exception but will return 0.0 (see failure modes below).
        smiles_b (str): SMILES string for molecule B. This has the same meaning and role as smiles_a and is interpreted by RDKit in the same way. smiles_b typically represents a candidate molecule or the second member of a comparison pair. If smiles_b is None or unparsable by RDKit, the function returns 0.0.
    
    Returns:
        float: Tanimoto similarity score between the two molecules, in the closed interval [0.0, 1.0]. A value of 1.0 indicates identical fingerprints under the fixed fingerprinting settings (radius=2, nBits=2048, useChirality=False); a value of 0.0 indicates no shared fingerprint bits or an invalid input. In particular, the function returns 0.0 when either input is None or when RDKit fails to parse a SMILES string into a molecule object. The function does not perform normalization beyond the RDKit fingerprint and similarity computation.
    
    Behavior, side effects, defaults, and failure modes:
        This function is deterministic and has no side effects: it does not modify its inputs or global state, and repeated calls with the same inputs will produce the same output. It uses RDKit functions Chem.MolFromSmiles, AllChem.GetMorganFingerprintAsBitVect (with radius=2, nBits=2048, useChirality=False), and DataStructs.TanimotoSimilarity to compute the similarity; therefore RDKit must be installed and accessible in the runtime environment. If RDKit modules referenced by the implementation are not available, calling this function will result in a NameError/ImportError at import or call time (depending on how RDKit is imported in the module). For robustness in pipelines, this function intentionally returns 0.0 for None or unparsable SMILES inputs rather than raising an exception, which allows it to be used safely in batch scoring of generated molecules where some entries may be invalid.
    """
    from tdc.chem_utils.oracle.oracle import similarity
    return similarity(smiles_a, smiles_b)


################################################################################
# Source: tdc.chem_utils.oracle.oracle.smiles_2_fingerprint_AP
# File: tdc/chem_utils/oracle/oracle.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_chem_utils_oracle_oracle_smiles_2_fingerprint_AP(smiles: str):
    """Convert a SMILES string into an RDKit Atom Pair fingerprint (Atom Pair, AP).
    
    This function is used in TDC chem_utils and molecule-generation/oracle contexts where
    compact, graph-derived molecular representations are required for similarity search,
    scoring, or as input features to generation or evaluation oracles. The function
    first converts the input SMILES string to an RDKit Mol object by calling
    smiles_to_rdkit_mol(smiles) and then computes an Atom Pair fingerprint using
    rdkit.Chem.AllChem.GetAtomPairFingerprint with a fixed maxLength=10, returning
    RDKit's sparse integer vector representation. The procedure is deterministic
    (given the same SMILES and RDKit version) and has no persistent side effects, but
    it depends on RDKit being available in the runtime environment.
    
    Args:
        smiles (str): SMILES string encoding the chemical structure to be converted.
            This is the standard text representation of a molecule used across
            cheminformatics and in TDC datasets and oracles. The function expects a
            syntactically valid SMILES that can be parsed by smiles_to_rdkit_mol
            into an RDKit Mol object. The caller is responsible for providing a
            correct SMILES; malformed or unsupported SMILES may cause parsing errors
            or exceptions from RDKit.
    
    Returns:
        rdkit.DataStructs.cDataStructs.IntSparseIntVect: The Atom Pair fingerprint
        computed by rdkit.Chem.AllChem.GetAtomPairFingerprint with maxLength=10.
        This return value is RDKit's sparse integer vector type that records atom-pair
        features (suitable for similarity computations, scoring in molecule
        generation oracles, and other cheminformatics workflows). If the input SMILES
        cannot be converted to an RDKit Mol or if RDKit raises an error during
        fingerprint computation, the function will propagate that exception (i.e.,
        it does not catch RDKit parsing or fingerprinting errors).
    """
    from tdc.chem_utils.oracle.oracle import smiles_2_fingerprint_AP
    return smiles_2_fingerprint_AP(smiles)


################################################################################
# Source: tdc.chem_utils.oracle.oracle.smiles_2_fingerprint_ECFP4
# File: tdc/chem_utils/oracle/oracle.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_chem_utils_oracle_oracle_smiles_2_fingerprint_ECFP4(smiles: str):
    """tdc.chem_utils.oracle.oracle.smiles_2_fingerprint_ECFP4 converts a SMILES string into an ECFP4 (Morgan fingerprint with radius=2) representation used by TDC for molecule generation, similarity calculations, and as input features for predictive models and oracles.
    
    Args:
        smiles (str): A SMILES (Simplified Molecular Input Line Entry System) string that encodes a small-molecule chemical structure. In the TDC context, this string is typically provided as part of dataset processing, oracle input, or model feature preparation. The function first converts this SMILES string to an RDKit molecule object using smiles_to_rdkit_mol and then computes the Morgan fingerprint; therefore the SMILES must be a valid, RDKit-parsable representation of the molecule.
    
    Returns:
        fp (rdkit.DataStructs.cDataStructs.UIntSparseIntVect): A RDKit UIntSparseIntVect representing the ECFP4 fingerprint (Morgan fingerprint with radius=2). This sparse integer vector encodes hashed circular atom environments (substructure identifiers mapped to integer counts) and is commonly used in TDC for molecular similarity, clustering, input to machine learning models, and oracle scoring functions that require fixed-length or sparse structural descriptors.
    
    Behavior and side effects:
        The function is deterministic for a given SMILES string and RDKit version: the same SMILES will produce the same fingerprint when called repeatedly under the same environment. It has no external side effects (it does not modify global state, write files, or alter inputs).
    
    Failure modes and exceptions:
        If RDKit is not available in the execution environment, importing or using RDKit functions invoked by smiles_to_rdkit_mol or AllChem.GetMorganFingerprint will raise an ImportError or related exception. If the provided SMILES cannot be parsed, smiles_to_rdkit_mol may return None or raise an informative error; this function will propagate that error (commonly a ValueError or a RDKit parsing exception). Users should validate or catch exceptions when supplying arbitrary SMILES strings. Note that differences in RDKit versions or fingerprint hashing schemes can change the numeric identifiers in the returned UIntSparseIntVect; for reproducible results across systems, ensure RDKit version consistency.
    """
    from tdc.chem_utils.oracle.oracle import smiles_2_fingerprint_ECFP4
    return smiles_2_fingerprint_ECFP4(smiles)


################################################################################
# Source: tdc.chem_utils.oracle.oracle.smiles_2_fingerprint_ECFP6
# File: tdc/chem_utils/oracle/oracle.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_chem_utils_oracle_oracle_smiles_2_fingerprint_ECFP6(smiles: str):
    """tdc.chem_utils.oracle.oracle.smiles_2_fingerprint_ECFP6: Convert a SMILES string into an ECFP6 fingerprint (Morgan fingerprint with radius 3) used as a compact molecular descriptor in TDC's molecule generation and evaluation pipelines.
    
    Args:
        smiles (str): SMILES string for a single molecule. This is the canonical input representation for small molecules in the TDC codebase and in cheminformatics more generally. The function first converts this SMILES string to an RDKit molecule object using smiles_to_rdkit_mol and then computes the circular Morgan fingerprint with radius 3 (commonly referred to as ECFP6, since diameter = 2 * radius = 6). Provide a valid SMILES string; passing None or a non-string value will result in a TypeError or a conversion error propagated from smiles_to_rdkit_mol. Non-parseable or chemically invalid SMILES will produce the same errors or will cause RDKit to raise an exception during molecule creation or fingerprinting.
    
    Returns:
        fp (rdkit.DataStructs.cDataStructs.UIntSparseIntVect): An RDKit UIntSparseIntVect that encodes the ECFP6 fingerprint as a sparse integer vector mapping hashed substructure identifiers to unsigned integer counts. In practical terms, this object is a compact, sparse representation of circular atom environments used for similarity calculations (for example, Tanimoto similarity), clustering, model inputs, and as a scoring/feature component in TDC molecule generation oracles. The function does not modify global state; it returns the fingerprint object produced by AllChem.GetMorganFingerprint.
    
    Behavior and side effects:
        The function performs two deterministic steps: (1) conversion of the input SMILES to an RDKit Mol object via smiles_to_rdkit_mol, and (2) computation of the Morgan fingerprint with radius 3 via AllChem.GetMorganFingerprint. The radius is fixed to 3 to produce ECFP6; there are no optional parameters to change radius, bit length, or representation format. The function has no side effects beyond returning the fingerprint object.
    
    Failure modes:
        If RDKit is not available in the runtime environment, import or RDKit calls invoked by smiles_to_rdkit_mol or AllChem.GetMorganFingerprint will raise ImportError or other RDKit-related exceptions. If the SMILES string is invalid or cannot be parsed, smiles_to_rdkit_mol or AllChem.GetMorganFingerprint will raise an exception (for example, ValueError or RDKit parsing errors); these exceptions propagate to the caller. Consumers should validate SMILES beforehand or handle exceptions when calling this function.
    """
    from tdc.chem_utils.oracle.oracle import smiles_2_fingerprint_ECFP6
    return smiles_2_fingerprint_ECFP6(smiles)


################################################################################
# Source: tdc.chem_utils.oracle.oracle.smiles_2_fingerprint_FCFP4
# File: tdc/chem_utils/oracle/oracle.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_chem_utils_oracle_oracle_smiles_2_fingerprint_FCFP4(smiles: str):
    """tdc.chem_utils.oracle.oracle.smiles_2_fingerprint_FCFP4 converts a SMILES string into an FCFP4 (feature-based Morgan) fingerprint using RDKit. This function is used in TDC molecule generation and oracle utilities to produce a compact, model-ready molecular descriptor that represents local atom environments (radius 2, feature-based) for similarity comparisons, scoring, and downstream machine learning models in generation and property-prediction workflows.
    
    Args:
        smiles (str): SMILES string representation of a molecule. This argument is the canonical input format for this function and must be a valid SMILES understood by RDKit. In the TDC context, callers typically pass SMILES from dataset records or generated molecules; the function first converts this string to an RDKit Mol via the internal helper smiles_to_rdkit_mol and then computes the fingerprint.
    
    Returns:
        rdkit.DataStructs.cDataStructs.UIntSparseIntVect: A RDKit UIntSparseIntVect representing the FCFP4 (feature-class Morgan) fingerprint computed with radius=2 and useFeatures=True. Practically, this sparse integer vector encodes hashed atom-environment identifiers mapped to counts/occurrences and is suitable for similarity computations, count-based descriptor inputs to models, and oracle scoring. The return value is produced directly by AllChem.GetMorganFingerprint and can be used with RDKit similarity routines or converted to dense arrays if required.
    
    Behavior and side effects:
        This function is pure (no external side effects) and deterministic for a given SMILES string and RDKit version. It relies on RDKit's molecule parsing and fingerprint routines: smiles_to_rdkit_mol(smiles) and AllChem.GetMorganFingerprint(molecule, 2, useFeatures=True). The choice of radius=2 corresponds to the "4" in FCFP4 (i.e., diameter 4), and useFeatures=True yields a feature-based fingerprint commonly used in molecule-generation oracles and descriptor-based models in TDC.
    
    Failure modes and notes:
        If the input SMILES is invalid or cannot be parsed, smiles_to_rdkit_mol or RDKit will raise an exception (for example, a parsing or sanitization error); callers should validate or catch exceptions as appropriate. If RDKit is not installed or properly imported in the environment, importing or calling RDKit functions will raise ImportError/NameError. The function does not perform chemical standardization beyond what smiles_to_rdkit_mol/ RDKit apply; differences in sanitization, stereochemistry perception, or tautomer handling depend on RDKit and the helper routine's behavior.
    """
    from tdc.chem_utils.oracle.oracle import smiles_2_fingerprint_FCFP4
    return smiles_2_fingerprint_FCFP4(smiles)


################################################################################
# Source: tdc.chem_utils.oracle.oracle.smiles_to_rdkit_mol
# File: tdc/chem_utils/oracle/oracle.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_chem_utils_oracle_oracle_smiles_to_rdkit_mol(smiles: str):
    """tdc.chem_utils.oracle.oracle.smiles_to_rdkit_mol: Convert a SMILES string into an RDKit Mol object suitable for downstream molecular processing and molecule-generation oracles in TDC.
    
    This function parses a SMILES (Simplified Molecular-Input Line-Entry System) string into an rdkit.Chem.rdchem.Mol object and performs RDKit sanitization to detect common structural problems (for example, invalid valence). It is intended for use in TDC workflows that require a validated RDKit molecule representation prior to oracle scoring, feature extraction, graph conversion, or other cheminformatics operations described in the TDC README (for example, molecule generation oracles and data processing utilities). The function does not modify any external state; it relies on RDKit's Chem.MolFromSmiles to build the molecule and Chem.SanitizeMol to validate it. If RDKit cannot parse the SMILES or if sanitization fails (e.g., due to invalid valence), the function returns None to indicate an invalid or unusable molecule rather than raising an exception.
    
    Args:
        smiles (str): SMILES string representing a chemical structure. This is the canonical input format used across TDC for small-molecule datasets and oracles; the string must be a valid SMILES notation as accepted by RDKit's MolFromSmiles.
    
    Returns:
        rdkit.Chem.rdchem.Mol or None: The parsed and sanitized RDKit Molecule object when parsing and sanitization succeed. If parsing fails (Chem.MolFromSmiles returns None) or Chem.SanitizeMol raises a ValueError due to structural issues such as invalid valence, the function returns None to signal an invalid SMILES input. Callers should check for None and handle invalid molecules (for example, by discarding them from oracle evaluations or logging the failure).
    """
    from tdc.chem_utils.oracle.oracle import smiles_to_rdkit_mol
    return smiles_to_rdkit_mol(smiles)


################################################################################
# Source: tdc.chem_utils.oracle.oracle.smina
# File: tdc/chem_utils/oracle/oracle.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_chem_utils_oracle_oracle_smina(
    ligand: numpy.ndarray,
    protein: numpy.ndarray,
    score_only: bool = False,
    raw_input: bool = False
):
    """tdc.chem_utils.oracle.oracle.smina docks a ligand to a protein pocket using the smina docking executable and can be used as a molecule-generation oracle (for example, to score candidate molecules during goal-oriented molecule generation or distribution learning in TDC). The function wraps a packaged static smina binary (oracle/smina.static), accepts either raw ML-format coordinates or prewritten SDF/PDB filenames, invokes external programs (openbabel and smina), and returns a numeric docking score when requested or otherwise invokes smina for its normal side effects (console output and any files the binary writes).
    
    Args:
        ligand (numpy.ndarray): Primary ligand input. In typical use this is a numeric coordinate matrix of shape (N_1, 3) where N_1 is the number of ligand atoms and each row is an (x, y, z) position; this is the representation expected by TDC when users supply precomputed atom coordinates. When raw_input is True the function instead expects ligand to be a two-element tuple (mol_coord, mol_atom) as used in the source code: mol_coord is a (N_1, 3) coordinate numpy.ndarray and mol_atom is a sequence of N_1 atomic symbols (strings). When raw_input is False the code passes ligand directly to the smina command line (so in practice ligand should be a filename path (for example an SDF) prepared by the caller or by the raw_input conversion). The function will, when raw_input is used, write a temporary file temp_ligand.xyz and convert it to temp_ligand.sdf using openbabel; these temporary files will be created in the current working directory and will overwrite any existing files with the same names.
    
        protein (numpy.ndarray): Protein input. In the original design this is a numeric coordinate matrix of shape (N_2, 3) where N_2 is the number of protein atoms and each row is an (x, y, z) position, matching the ligand coordinate convention. In practice the implementation passes the provided protein argument directly to the smina command line (the -r argument), so when raw_input is False the caller typically supplies a prepared receptor filename (for example a PDB or SDF file path). The function does not itself convert protein coordinate arrays to files; supplying a numpy.ndarray without converting it to a file will lead to smina receiving an invalid receiver argument and the external call failing.
    
        score_only (bool): Whether to return only the numeric docking score. Default is False. If True the function runs the smina binary with the --score_only flag, captures standard output, attempts to parse a floating-point docking score using a fixed positional parsing strategy (it reads the process output, takes the seventh-from-last line, splits by spaces and selects the second-to-last token), and returns that value as a float. This parsing is fragile: if the smina executable version or its output format changes, the parsing can raise ValueError or return an incorrect value. If score_only is False the function invokes the smina binary with --score_only (as implemented) for side effects and does not return a score; instead, it relies on the binary to produce any files or console output. Default: False.
    
        raw_input (bool): Whether the ligand input is raw ML-format coordinates that must be converted to an SDF by this function. Default is False. When True the function expects ligand to be a tuple (mol_coord, mol_atom) as described above, writes a temporary XYZ file (temp_ligand.xyz) and then invokes openbabel (obabel) to convert that XYZ into temp_ligand.sdf. If openbabel is not available or the conversion command fails the function raises an ImportError with an instruction to install openbabel ("Please install openbabel by 'conda install -c conda-forge openbabel'!"). After conversion the local variable ligand is set to the generated "temp_ligand.sdf" filename and is passed to the smina executable. Be aware that using raw_input will create and overwrite temp_ligand.xyz and temp_ligand.sdf in the current working directory; callers should manage or clean these files if persistence is undesired.
    
    Returns:
        float or None: If score_only is True the function returns a float that it parsed from the smina process output representing the docking score. The float parsing depends on a specific smina output layout and may raise ValueError or produce incorrect values if the binary output format differs. If score_only is False the function does not return a docking value (returns None) and instead relies on side effects: it runs the smina static binary (oracle/smina.static), which may write files, print results to stdout/stderr, or otherwise perform docking-related actions. Note that the code sets execute permissions on the bundled binary (oracle/smina.static) with chmod; if the binary is missing, not executable, or produces an error the function will not raise a Python exception directly but the external call will fail (os.system or os.popen will return failure codes or empty output), which can manifest as missing output, parsing errors, or a nonzero shell exit status.
    
    Behavior, side effects, defaults, and failure modes:
        - The function uses a packaged smina binary at path "oracle/smina.static". It runs os.system(f"chmod +x ./{smina_model_path}") at start to ensure executability; if that file is missing or the process lacks permissions, subsequent calls to the binary will fail.
        - When raw_input is True the function writes temp_ligand.xyz (overwriting any existing file with that name) and then runs "obabel temp_ligand.xyz -O temp_ligand.sdf" via os.system. If openbabel (obabel) is not installed or the conversion command fails, an ImportError is raised with installation instructions. The generated filenames are fixed and may conflict with concurrent invocations or other processes.
        - When score_only is True the function runs the smina binary via os.popen and parses its textual output to extract a floating-point score using a fixed positional parse (msg.split("\n")[-7].split(" ")[-2]). This parsing strategy is brittle: changes in smina output will cause parsing errors (ValueError) or to return incorrect values. The function converts the parsed token to float and returns it.
        - When score_only is False the function calls os.system to run smina and does not return a result; the primary effect is the external smina execution. Any files created or outputs produced are determined by the smina binary and not by this wrapper.
        - The function concatenates user-supplied ligand and protein values directly into shell command strings; passing untrusted input could lead to shell injection vulnerabilities. Callers should ensure inputs are sanitized and under their control.
        - The function operates in the current working directory and will create/overwrite temp_ligand.xyz and temp_ligand.sdf when raw_input is used; concurrent calls may interfere with each other.
        - The implementation relies on external executables (openbabel/obabel and the bundled smina static binary). If these are not installed or incompatible with the environment, the function will fail at the system-call level or during parsing.
        - Defaults: score_only defaults to False (no return value, only side effects), raw_input defaults to False (function assumes ligand and protein are prepared and directly consumable by smina).
    """
    from tdc.chem_utils.oracle.oracle import smina
    return smina(ligand, protein, score_only, raw_input)


################################################################################
# Source: tdc.chem_utils.oracle.oracle.tree_analysis
# File: tdc/chem_utils/oracle/oracle.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_chem_utils_oracle_oracle_tree_analysis(current: dict):
    """tdc.chem_utils.oracle.oracle.tree_analysis analyzes the JSON-like output of a retrosynthetic "tree builder" oracle used in TDC molecule-generation/oracle workflows. It extracts summary metrics that quantify whether synthesis routes were found for a query compound, how many reaction steps are implied, an aggregated plausibility score for the route(s), a binary synthesizability indicator, and a price estimate returned by the oracle. This function is used by TDC oracles and downstream evaluation code (for example, when scoring generated molecules for synthesizability and estimated synthesis cost) to convert the nested tree representation produced by synthesis planners (ASKCOS-style trees) into simple scalar and dictionary diagnostics.
    
    Args:
        current (dict): The raw tree-builder result returned by an oracle. This dictionary is expected to follow the oracle/tree structure used in TDC generation oracles: it may contain keys "error" (indicating a celery/task error), "price" (a precomputed price estimate for the query compound), or "trees" (a list of retrosynthetic tree objects). Each tree node in "trees" is expected to have "children" (list of child nodes), "plausibility" (numeric plausibility score for transformation nodes), and "ppg" (a numeric price-like field returned by nodes). This function reads those fields to compute metrics; if the input omits these keys or uses different types/shapes, a KeyError or TypeError may be raised. The dict is treated as read-only; the function has no side effects on external state.
    
    Returns:
        tuple: A 6-tuple summarizing the analysis (num_path, status, num_step, p_score, synthesizability, price).
            num_path (int): Number of root synthesis trees returned in current["trees"]. Practical role: indicates how many alternative retrosynthetic routes the oracle produced. Special sentinel values: -1 if "error" is present in input.
            status (dict): A mapping from tree depth levels to the number of child nodes found at that level, as implemented in ASKCOS-style analyses. Keys are numeric depth values produced by iterating the tree in 0.5 increments (half-integer levels correspond to transformation/plausibility layers and integer levels correspond to precursor/price layers); values are integers giving the number of nodes at that depth. Practical significance: allows callers to inspect branching factor per generation step to estimate search breadth and complexity. When an error is detected this is returned as an empty dict.
            num_step (int): Number of synthetic steps inferred from the tree analysis (an integer count derived from the traversal). This represents the integer count of reaction-step layers (the function converts its internal half-step depth counter into an integer step count). Special sentinel: returns 11 when no valid synthesis steps were found (mirrors the original implementation behavior) or other special-case returns as described below.
            p_score (float): The aggregated plausibility score computed as the product of "plausibility" values across transformation layers for the traversed path(s), multiplied by the synthesizability flag. Practical role: a single scalar estimate of the joint plausibility of the proposed route; note that it is forced to zero when synthesizability is 0. The function multiplies plausibility values at half-integer traversal depths and accumulates price at integer depths, then multiplies the final plausibility by synthesizability before returning.
            synthesizability (int): Binary indicator (0 or 1) where 1 indicates that at least one multi-level (multi-step) synthesis path was found and 0 indicates no valid multi-step path. Practical significance: quick filter a downstream pipeline can use to decide whether an oracle-recommended synthesis is feasible (1) or not (0).
            price (float or int): An accumulated price-like estimate derived from node "ppg" values during traversal (summed at integer-numbered layers), or other oracle-provided price fields. Practical role: an estimate of synthesis cost for the query compound returned by the oracle. Special sentinel values: -1 indicates that no price estimate could be computed or was not available for the found path; when the input contains a top-level "price" key the function returns that value directly.
    
    Behavior and special cases:
        - If the input dict contains the key "error", the function treats this as a task failure (for example, a celery worker error) and returns the sentinel tuple (-1, {}, 11, -1, -1, -1).
        - If the input dict contains a top-level "price" key but no detailed tree, the function returns (0, {}, 0, 1, 1, current["price"]) where current["price"] is forwarded as the price estimate; this models oracle outputs that provide only a cost estimate without tree details.
        - If at least one tree exists and the first tree's "ppg" is non-zero, the function treats that ppg value as a direct price estimate and returns (0, {}, 0, 1, 1, current[0]["ppg"]) to reflect a price-only response from the first tree node.
        - For a normal tree traversal, the function iteratively explores levels of the tree in 0.5 increments: at half-integer increments it multiplies together node "plausibility" values into p_score; at integer increments it sums node "ppg" values into price. The status dict records the number of child nodes found at each depth value.
        - After traversal, synthesizability is set to 1 if more than one status level was observed (meaning at least one non-trivial expansion was found); otherwise it is 0. If no valid depth-based steps were found, the function sets num_step to 11 and price to -1 as special indicators (preserving legacy behavior).
        - The function is pure (no external side effects) but expects the input to conform to the expected oracle tree schema. If nodes lack expected keys ("children", "plausibility", "ppg") or have incompatible types, the function may raise KeyError or TypeError. Callers should catch these exceptions when passing unvalidated or external data.
        - Numeric return values follow the oracle's conventions and sentinel values described above; downstream code should handle the special sentinel values (-1 and 11) appropriately when interpreting results.
    """
    from tdc.chem_utils.oracle.oracle import tree_analysis
    return tree_analysis(current)


################################################################################
# Source: tdc.evaluator.centroid
# File: tdc/evaluator.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_evaluator_centroid(X: numpy.ndarray):
    """tdc.evaluator.centroid computes the centroid (mean position) of all points in a vectorset X and returns that centroid value. In the Therapeutics Data Commons (TDC) context, this function is useful for summarizing the central tendency of numeric feature vectors or embeddings (for example, molecular descriptors, learned representations, or coordinate-based features) when evaluating datasets or preprocessing data for model evaluation.
    
    Args:
        X (array): (N,D) matrix where N is the number of points and D is the number of coordinate dimensions. Each row corresponds to one point (for example, a sample embedding or feature vector in a TDC dataset). The function treats X as numeric and computes the mean across rows (axis=0), effectively computing the mean position in each coordinate direction. Supplying a non-numeric array or an array with incompatible shape will result in numpy raising an exception when attempting to compute the mean.
    
    Returns:
        C (float): centroid. The centroid is computed as the mean position across all N points in each of the D coordinate directions (mathematically C = sum(X) / len(X)); in code this is implemented as X.mean(axis=0). Practically, this provides the per-dimension mean used to represent the center of the point cloud or set of embeddings. If D == 1, the returned value will be a scalar; if D > 1, the returned value represents the per-dimension means. Note that if N == 0 (empty first dimension), numpy's mean behavior applies (typically resulting in NaNs and a runtime warning). The function has no side effects and is deterministic given the same input X.
    """
    from tdc.evaluator import centroid
    return centroid(X)


################################################################################
# Source: tdc.evaluator.kabsch
# File: tdc/evaluator.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_evaluator_kabsch(P: numpy.ndarray, Q: numpy.ndarray):
    """tdc.evaluator.kabsch computes the optimal rotation matrix that rigidly aligns two sets of paired points using the Kabsch algorithm. In the Therapeutics Data Commons (TDC) context, this function is used to align paired coordinate sets such as atomic coordinates of molecular conformations or structural fragments so that downstream comparisons (for example RMSD calculation, structural matching, docking post-processing, or evaluation of conformation generators) reflect only rotational differences and not translations. The function implements the covariance-based SVD approach described by the Kabsch algorithm and returns a D x D orthonormal rotation matrix U that, when applied to P, minimizes the root-mean-square deviation between P and Q.
    
    Args:
        P (numpy.ndarray): An (N, D) array of N points in D-dimensional space representing the first set of paired vectors (for example, 3D atomic coordinates of a molecule). P is expected to have been translated so its centroid is at the origin prior to calling this function; if P is not centroid-centered, the computed rotation will not correctly represent a pure rotation aligning the original point clouds.
        Q (numpy.ndarray): An (N, D) array of N points in D-dimensional space representing the second set of paired vectors that P is to be aligned to (for example, target atomic coordinates). Q is also expected to be centroid-centered (centroid at the origin). P and Q must have the same shape and point ordering so that P[i] corresponds to Q[i] for all i.
    
    Returns:
        numpy.ndarray: A (D, D) rotation matrix U. U is an orthonormal matrix (determinant +1 after the algorithm's correction) that should be applied to P (e.g., P_rotated = P.dot(U)) to produce the best-fit rotated coordinates in the least-squares sense relative to Q. The returned matrix represents a proper rotation (right-handed coordinate system) except in degenerate cases described below.
    
    Behavior, side effects, defaults, and failure modes:
        The function performs the following steps without mutating the input arrays: computes the covariance matrix C = P^T Q, performs singular value decomposition C = V S W^T, detects whether the naive product V W^T would produce a reflection (via the sign of det(V)*det(W)), and, if necessary, flips the sign of the last singular vector to enforce a right-handed rotation, finally returning U = V W^T. There are no side effects; inputs P and Q are not modified and no global state is changed.
    
        Preconditions: P and Q must be real-valued numpy.ndarray objects with identical shapes (N, D) and N >= 1, D >= 1. Both P and Q are assumed to have been translated so that their centroids are at the origin. If centering is not performed externally, the resulting rotation will not align the original uncentered point clouds correctly.
    
        Numerical and exceptional behavior: The function relies on numpy.linalg.svd and numpy.linalg.det. If the covariance matrix is rank-deficient or nearly singular, SVD will still typically complete but the computed rotation may be numerically unstable; downstream usage (e.g., RMSD) can exhibit larger numerical error. If numpy.linalg.svd raises a numpy.linalg.LinAlgError (rare), that exception will propagate to the caller. If inputs contain NaNs or infinite values, results are undefined and may propagate NaNs into the output. For perfectly collinear or otherwise degenerate point sets, multiple optimal rotations may exist; the algorithm returns one valid rotation but the alignment may be ambiguous.
    
        Practical significance in TDC workflows: Use this function to produce rotation matrices for rigid-body alignment steps in structural comparisons, evaluation of molecule conformation generators, or preparing aligned inputs for metrics and visualizations in therapeutic machine-learning pipelines. Ensure to centroid-center P and Q (subtract their respective centroids) before calling kabsch; TDC provides higher-level utilities for dataset handling and splitting but alignment/centering is the caller's responsibility.
    """
    from tdc.evaluator import kabsch
    return kabsch(P, Q)


################################################################################
# Source: tdc.evaluator.kabsch_rmsd
# File: tdc/evaluator.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_evaluator_kabsch_rmsd(
    P: numpy.ndarray,
    Q: numpy.ndarray,
    W: numpy.ndarray = None,
    translate: bool = False
):
    """tdc.evaluator.kabsch_rmsd computes the root-mean-squared deviation (RMSD) between two sets of points after finding the optimal rotation that aligns P onto Q using the Kabsch algorithm. This function is intended for comparing 3D (or D-dimensional) coordinate sets such as atomic coordinates of molecular conformations in TDC benchmarks and evaluators where structural similarity between predicted and reference conformations is required.
    
    Args:
        P (numpy.ndarray): (N, D) array of N points in D dimensions representing the source coordinate set to be rotated onto Q. In molecular use, rows typically correspond to atoms and columns to Cartesian coordinates. The function expects P to have the same shape as Q; shape mismatches will cause downstream linear-algebra routines to fail.
        Q (numpy.ndarray): (N, D) array of N points in D dimensions representing the target coordinate set. Q is the reference to which P is aligned. Q must have the same shape and point ordering correspondence as P (i.e., the i-th row in P corresponds to the i-th row in Q).
        W (numpy.ndarray or None): (N,) optional 1-D vector of nonnegative weights for each point. When provided (not None), the implementation delegates to kabsch_weighted_rmsd to perform a weighted Kabsch alignment and compute a weighted RMSD (weights can encode per-atom importance such as atomic mass or scoring emphasis). The default is None, which triggers the unweighted Kabsch rotation and RMSD calculation.
        translate (bool): If True, translate both P and Q by subtracting their respective centroids before alignment (centroid = mean across rows). This centers both point sets at the origin prior to rotation and is typically required when comparing absolute molecular coordinates; default is False. When False, the function assumes points are already positioned appropriately or that only pure rotation should be considered.
    
    Returns:
        rmsd (float): The scalar root-mean-squared deviation after optimal rotation (and after weighting if W is provided). This value quantifies the average Euclidean distance between corresponding points in the aligned P and Q and is commonly used in TDC to evaluate structural similarity between predicted and reference molecular conformations.
    
    Behavior and side effects:
        The function does not modify the caller's input arrays in-place; it rebinds local names to translated or rotated copies as needed. If translate is True, local copies of P and Q are replaced by centroid-subtracted arrays. If W is provided, the function directly returns the result of kabsch_weighted_rmsd(P, Q, W). Otherwise, it computes a rotation using kabsch_rotate(P, Q), applies that rotation, and returns rmsd(P_rotated, Q).
        The implementation relies on downstream linear-algebra routines (kabsch_weighted_rmsd, kabsch_rotate, rmsd) and therefore may raise exceptions produced by those routines (for example, due to invalid shapes, NaNs in input, degenerate/collinear point configurations that make the covariance matrix singular, or issues in underlying SVD computations). Callers should ensure P and Q have matching shapes (N, D), W (if provided) has length N, and inputs contain finite numeric values to avoid propagated errors.
        Defaults: W defaults to None (unweighted), translate defaults to False (no centroid translation).
    """
    from tdc.evaluator import kabsch_rmsd
    return kabsch_rmsd(P, Q, W, translate)


################################################################################
# Source: tdc.evaluator.kabsch_rotate
# File: tdc/evaluator.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_evaluator_kabsch_rotate(P: numpy.ndarray, Q: numpy.ndarray):
    """tdc.evaluator.kabsch_rotate rotates the input point set P to best align with the reference point set Q using the Kabsch algorithm. In the context of TDC (Therapeutics Data Commons), this function is typically used to align molecular point clouds or atomic coordinates (predicted vs. reference conformations) prior to computing alignment-sensitive evaluation metrics such as RMSD; it computes a rigid-body orthogonal rotation (no scaling) that minimizes mean squared deviation between corresponding points.
    
    Args:
        P (numpy.ndarray): (N, D) array of N points in D dimensions representing the source coordinates to be rotated. Each row is a point (for example, an atom coordinate in a molecular conformation). This argument is the data that will be transformed to best match Q; it must have the same shape and point correspondence ordering as Q. The function does not modify the caller's original array in place; it returns a new array containing the rotated coordinates.
        Q (numpy.ndarray): (N, D) array of N points in D dimensions representing the target/reference coordinates (for example, a reference molecular conformation). Points in Q are treated as the fixed reference to which P is aligned. Q must have the same shape as P and the same point-to-point correspondence (i.e., row i in P corresponds to row i in Q).
    
    Returns:
        numpy.ndarray: (N, D) array containing the coordinates of P after applying the optimal orthogonal rotation matrix U computed by the Kabsch algorithm. The returned array is the rotated version of P and preserves inter-point distances (rigid-body rotation, no scaling or translation applied by this function beyond what the Kabsch helper implements). This return value is intended for downstream use in evaluation pipelines (e.g., computing RMSD between the rotated P and Q).
    
    Raises:
        ValueError: If P and Q do not have the same shape (different N or D) or are not two-dimensional arrays with matching point correspondence, a ValueError may be raised by downstream operations. The function relies on the helper kabsch(P, Q) and numpy.dot; errors from those calls (for example, due to incompatible shapes) will propagate.
        RuntimeError: If the underlying kabsch implementation fails due to degenerate input (for example, insufficient distinct points to define a rotation) or numerical issues during SVD, that exception may be propagated.
    
    Behavior and side effects:
        - The function computes the rotation matrix U by calling the internal kabsch(P, Q) helper, which implements the Kabsch algorithm to minimize RMSD under a rigid rotation.
        - The function then applies the rotation via a matrix multiplication (numpy.dot), producing and returning a new numpy.ndarray with the rotated coordinates. It does not perform in-place modification of the caller's input arrays.
        - The function does not perform input validation beyond what numpy and the kabsch helper enforce: inputs containing NaN or infinite values, mismatched shapes, or incorrect correspondence ordering will produce incorrect results or raise exceptions.
        - The function preserves metric properties of the point set (distances between points) because it applies an orthogonal rotation; no scaling is applied.
    """
    from tdc.evaluator import kabsch_rotate
    return kabsch_rotate(P, Q)


################################################################################
# Source: tdc.evaluator.kabsch_weighted
# File: tdc/evaluator.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_evaluator_kabsch_weighted(
    P: numpy.ndarray,
    Q: numpy.ndarray,
    W: numpy.ndarray = None
):
    """Compute the optimal rigid-body alignment (rotation and translation) that minimizes the weighted root-mean-square deviation (RMSD) between two paired point sets using the Kabsch algorithm. This function is intended for 3-dimensional point sets and is used in TDC evaluation workflows to align predicted molecular or structural coordinates (P) to reference coordinates (Q) and to produce a single scalar RMSD metric for benchmarking model predictions (for example, aligning predicted ligand/protein atom coordinates to experimental structures before computing an RMSD-based score).
    
    Args:
        P (numpy.ndarray): (N, D) array of N source points, each of dimensionality D. In practice within TDC this represents predicted 3D coordinates (atoms or labeled points) for a single structure or pose. The function implementation assumes D == 3 and will index arrays for three coordinates; supplying arrays with a different second dimension may raise IndexError or produce incorrect results.
        Q (numpy.ndarray): (N, D) array of N target/reference points paired with P, of the same shape as P. In TDC use-cases this is the ground-truth 3D coordinates to which P is aligned. The number of points N must match between P and Q; mismatched lengths will result in shape/broadcasting errors.
        W (numpy.ndarray or None): Optional (N,) vector of non-negative per-point weights that modulate the influence of each paired point in the alignment objective. If None (the default), uniform weights are used equivalent to ones(N) / N. The algorithm internally broadcasts W to shape (N, 3) for per-coordinate weighting; the implementation then normalizes weights via their sum. Supplying W whose sum is zero will cause a division-by-zero error. The function expects W to be real-valued and of length N; shapes that do not match N will raise errors.
    
    Behavior and practical details:
        - The function computes weighted centroids of P and Q, forms the weighted covariance matrix, and computes the optimal rotation using singular value decomposition (SVD) of that covariance.
        - A right-handedness/reflection correction is applied: if the product of determinants of SVD factors indicates a reflection, the smallest singular value and corresponding column of the V matrix are negated to ensure a proper rotation (no improper reflection).
        - The returned rotation matrix U is a (D, D) numpy.ndarray and the translation vector V is a (D,) numpy.ndarray such that, for each row p in P, the aligned point p' is obtained as p' = p.dot(U) + V (i.e., P' = P @ U + V). In TDC evaluation pipelines, applying U and V to predicted coordinates P produces coordinates directly comparable to Q for RMSD calculation or downstream scoring.
        - The RMSD returned is the non-negative root mean squared deviation between Q and the transformed P (P'), computed from the weighted mean-square deviation. Any small negative mean-square deviation values arising from floating-point round-off are clamped to zero before taking the square root.
        - This implementation is specialized for D == 3 and allocates intermediate arrays of length 3; using other dimensionalities is unsupported and will likely fail or produce incorrect results.
        - The function does not modify the input arrays P, Q, or the caller-provided W; all modifications occur on local copies/arrays.
    
    Failure modes and numerical considerations:
        - ValueError/IndexError may occur when P and Q have mismatched lengths or shapes that do not conform to (N, 3).
        - DivisionByZero or a RuntimeWarning may occur if the provided weight vector W sums to zero.
        - Numerical instability in SVD is handled by standard numpy.linalg.svd behavior; if SVD fails due to degenerate inputs, numpy.linalg.LinAlgError may be raised.
        - The function clamps small negative mean-square deviations (due to floating-point error) to zero before computing RMSD, ensuring a real non-negative RMSD.
    
    Returns:
        U (numpy.ndarray): Rotation matrix of shape (D, D) (in practice (3, 3)). This matrix is the orthogonal rotation that, together with translation V, best aligns P onto Q in the weighted least-squares sense.
        V (numpy.ndarray): Translation vector of shape (D,) (in practice (3,)). This vector, when added after rotation, recenters the rotated P to optimally align with Q.
        RMSD (float): Root mean squared deviation between Q and the transformed P (P' = P @ U + V). This scalar is non-negative and represents the weighted geometric discrepancy after optimal rigid-body alignment.
    """
    from tdc.evaluator import kabsch_weighted
    return kabsch_weighted(P, Q, W)


################################################################################
# Source: tdc.evaluator.kabsch_weighted_rmsd
# File: tdc/evaluator.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_evaluator_kabsch_weighted_rmsd(
    P: numpy.ndarray,
    Q: numpy.ndarray,
    W: numpy.ndarray = None
):
    """Compute the weighted root-mean-square deviation (RMSD) between two sets of points P and Q after optimal rigid-body alignment using the weighted Kabsch algorithm. This function is used in structural comparison tasks common in therapeutics and molecular modeling within TDC (for example, comparing predicted ligand or protein atom coordinates to reference structures to evaluate pose prediction or conformational similarity). The function delegates the core computation to kabsch_weighted and returns the scalar weighted RMSD value computed from the aligned coordinates.
    
    Args:
        P (numpy.ndarray): An (N, D) array of N points in D dimensions representing the first coordinate set (e.g., predicted atom coordinates). Each row is a point; D is typically 2 or 3 for planar or 3D molecular coordinates. The values should be numeric and are interpreted in the same units as Q (for molecular data commonly angstroms).
        Q (numpy.ndarray): An (N, D) array of N points in D dimensions representing the second coordinate set (e.g., reference atom coordinates). P and Q must have identical shapes: same number of points N and same dimensionality D. The correspondence between rows of P and Q is assumed (point i in P corresponds to point i in Q).
        W (numpy.ndarray, optional): A length-N 1D array of nonnegative weights for each point correspondence. If provided, weights scale each point's contribution to the optimal alignment and to the RMSD calculation so that more important atoms or coordinates can influence the result more strongly (for example, weighting heavy atoms more than hydrogens). If None (default), all points are treated with equal weight.
    
    Returns:
        float: The weighted RMSD scalar (w_rmsd) after applying the optimal rotation and translation computed by the weighted Kabsch algorithm. This value quantifies the average weighted Euclidean deviation between corresponding points in P and Q after alignment; it is expressed in the same units as the input coordinates (e.g., angstroms for molecular coordinates).
    
    Behavior, defaults, and failure modes:
        - The function computes the optimal rigid-body rotation and translation by calling kabsch_weighted(P, Q, W) and returns the RMSD value produced by that routine.
        - Default behavior when W is None is to use equal weights for all points, producing the standard unweighted Kabsch RMSD.
        - Inputs are not modified; the function is pure (no side effects).
        - The function expects numeric numpy.ndarray inputs with matching shapes. If P and Q have different shapes (different N or D) or if W does not have length N when provided, the underlying kabsch_weighted call or numpy operations will raise an exception (for example, ValueError or IndexError).
        - If W contains invalid values (NaN, infinite) or if all weights sum to zero, the computation may raise an error or produce an undefined result; callers should validate weights (nonnegative and not all zero) before calling when meaningful weighted behavior is required.
        - This routine is intended for evaluating structural similarity tasks in TDC benchmarks and should be used when a rotation/translation-invariant, optionally weighted measure of point-set deviation is desired.
    """
    from tdc.evaluator import kabsch_weighted_rmsd
    return kabsch_weighted_rmsd(P, Q, W)


################################################################################
# Source: tdc.evaluator.range_logAUC
# File: tdc/evaluator.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_evaluator_range_logAUC(
    true_y: numpy.ndarray,
    predicted_score: numpy.ndarray,
    FPR_range: tuple = (0.001, 0.1)
):
    """Calculate the log-scaled area under the ROC curve (logAUC) restricted to a specified false positive rate (FPR) interval. This function is used in Therapeutics Data Commons (TDC) to evaluate binary classifiers for molecule prioritization and virtual screening in drug discovery, where only a small fraction of top-ranked candidates can be experimentally tested. By integrating the ROC curve on a logarithmic FPR axis over a small FPR_range (default (0.001, 0.1)), the metric emphasizes classifier performance at very low FPRs (the left side of the ROC curve), which corresponds to selecting only the highest-scoring compounds for costly follow-up experiments. A perfect classifier attains a logAUC of 1 on the default range; a random classifier achieves approximately 0.0215 on that same range (see References in the original implementation).
    
    Args:
        true_y (numpy.ndarray): Ground-truth binary labels for each sample. Values must correspond to class membership with 1 indicating active/positive instances and 0 indicating inactive/negative instances. This array is used as the true labels input to sklearn.metrics.roc_curve to compute false positive rates (FPR) and true positive rates (TPR). Mismatched length with predicted_score or non-binary values may cause downstream errors from roc_curve.
        predicted_score (numpy.ndarray): Predicted score or ranking for each sample. Values need not be probabilities and do not have to lie in [0, 1]; they are treated as continuous scores for ranking and thresholding when computing the ROC curve. The ordering induced by predicted_score determines the ROC curve shape and therefore the logAUC.
        FPR_range (tuple): Two-element tuple (lower_bound, upper_bound) specifying the FPR interval over which to compute the logAUC. Defaults to (0.001, 0.1). Both bounds are interpreted as false positive rates on the usual [0, 1] scale; lower_bound must be strictly less than upper_bound. This range determines the portion of the ROC curve emphasized: smaller lower_bound values bias the metric toward very-low-FPR behavior relevant to selecting a tiny fraction of candidates for experimental validation.
    
    Behavior and implementation details:
        The function computes the ROC curve (fpr, tpr) from true_y and predicted_score using sklearn.metrics.roc_curve with pos_label=1. It then ensures the specified FPR_range endpoints are present by interpolating TPR values at those FPRs and appending them to the ROC points. Both FPR and corresponding TPR arrays are sorted, transformed to a log10 scale along the FPR axis, and trimmed to the interval [log10(lower_bound), log10(upper_bound)]. The area under this trimmed ROC segment is computed via sklearn.metrics.auc on the log10(FPR) axis and normalized by the width of the log10 interval (upper_bound - lower_bound in log10 space) so that the metric is comparable across different FPR ranges. The logarithm bias favors better performance at smaller FPRs, matching the practical need in drug discovery to prioritize very-high-scoring candidates.
    
    Defaults and side effects:
        Default FPR_range is (0.001, 0.1), selected to reflect historical practice in molecular docking and virtual screening benchmarks where only the extreme left portion of the ROC curve is relevant. The function calls sklearn.metrics.roc_curve and sklearn.metrics.auc and uses numpy interpolation and log10 operations; these libraries must be available in the runtime. There are no external side effects (no file I/O or global state modifications).
    
    Failure modes and exceptions:
        If FPR_range is None, the function raises Exception("FPR range cannot be None") as a guard against an undefined integration interval. If FPR_range[0] >= FPR_range[1], the function raises Exception("FPR upper_bound must be greater than lower_bound"). If true_y and predicted_score have mismatched lengths, contain invalid values, or if true_y does not contain the required binary labels, sklearn.metrics.roc_curve may raise a ValueError or produce incorrect results; these upstream errors are not caught and will propagate. If FPR_range values fall outside the meaningful FPR domain (typically [0, 1]), results are undefined; the implementation uses numpy.interp to estimate TPR at the requested FPR endpoints and then computes log10(FPR), so FPR_range values must be positive (log10 undefined for non-positive values).
    
    Returns:
        numpy.ndarray: A scalar numpy array containing the normalized logAUC value computed over the specified FPR_range. This value summarizes classifier performance focused on low-FPR operation: higher values indicate better discrimination at the top of the ranked list of candidates and therefore better practical utility for selecting a small number of molecules for experimental follow-up.
    """
    from tdc.evaluator import range_logAUC
    return range_logAUC(true_y, predicted_score, FPR_range)


################################################################################
# Source: tdc.evaluator.rmsd
# File: tdc/evaluator.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_evaluator_rmsd(V: numpy.ndarray, W: numpy.ndarray):
    """tdc.evaluator.rmsd calculates the root-mean-square deviation (RMSD) between two sets of vectors. In the Therapeutics Data Commons (TDC) context, this function is used as a numeric evaluator to quantify average Euclidean deviation between predicted and reference vector representations (for example, predicted molecular coordinates, embeddings, or other D-dimensional descriptors) across N points. The implementation computes RMSD as sqrt(sum((V - W)**2) / N), where N is the number of points (rows) in V. The function converts its inputs to numpy arrays internally and returns a single scalar RMSD value that can be used to compare model predictions to ground truth in benchmarking and evaluation workflows.
    
    Args:
        V (numpy.ndarray): (N, D) array representing N points with D dimensions each. In TDC usage, V typically holds reference (ground-truth) vectors such as experimentally-determined coordinates or target embeddings. The function will internally call numpy.array(V) to ensure array semantics. N is determined as len(V). If V is empty (len(V) == 0), a division-by-zero will occur.
        W (numpy.ndarray): (N, D) array representing N points with D dimensions each corresponding to the predicted vectors to be compared against V. In TDC usage, W typically holds model predictions. The function will internally call numpy.array(W). V and W are expected to have the same shape (N, D); mismatched shapes may trigger numpy broadcasting or runtime errors and can lead to incorrect results.
    
    Returns:
        float: Root-mean-square deviation between V and W computed as sqrt((diff * diff).sum() / N) where diff = numpy.array(V) - numpy.array(W) and N = len(V). This scalar quantifies the average Euclidean discrepancy per point and is commonly used in TDC benchmarks to assess geometric or embedding prediction accuracy. No side effects occur (the function does not modify inputs). Failure modes include ZeroDivisionError when V is empty and potential ValueError or unintended broadcasting when V and W have incompatible shapes.
    """
    from tdc.evaluator import rmsd
    return rmsd(V, W)


################################################################################
# Source: tdc.model_server.tokenizers.scgpt.tokenize_batch
# File: tdc/model_server/tokenizers/scgpt.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for tokenize_batch because the docstring has no description for the argument 'append_cls'
################################################################################

def tdc_model_server_tokenizers_scgpt_tokenize_batch(
    data: numpy.ndarray,
    gene_ids: numpy.ndarray,
    return_pt: bool = True,
    append_cls: bool = True,
    include_zero_gene: bool = False,
    cls_id: str = "<cls>"
):
    """Tokenize a batch of single-cell or bulk gene-expression vectors into per-sample token sequences suitable for the SCGPT tokenizer used by the TDC model server. This function converts each row in data (one sample / cell per row) into a sequence of gene identifiers and their corresponding expression counts; by default it maps gene identifiers to integer token ids using a downloaded "scgpt_vocab" vocabulary and returns PyTorch tensors for downstream SCGPT model input preparation in therapeutics workflows (for example, representing expressed genes per cell for input to models used in target discovery, activity screening, or other TDC tasks).
    
    Args:
        data (numpy.ndarray): A batch of gene-expression vectors with shape (batch_size, n_features). Each row corresponds to a single sample (for example a single cell or bulk sample) and each column corresponds to one gene feature. n_features must equal the length of gene_ids. Values represent expression counts or intensities and are treated as floats.
        gene_ids (numpy.ndarray): An array of gene identifiers with shape (n_features,). Each element corresponds to the gene name or identifier for the matching column in data. These identifiers are used to look up integer token ids in the downloaded "scgpt_vocab" mapping when return_pt is True; when return_pt is False these identifiers are retained as-is in the returned per-sample gene arrays (possibly including the cls_id if append_cls is True).
        return_pt (bool): Whether to return PyTorch tensors for the per-sample gene token ids and expression values. If True (default), the function imports torch (on-demand) and returns genes as torch.tensor of dtype torch.int64 (token ids) and values as torch.FloatTensor. If False, genes and values are returned as numpy.ndarray objects (genes containing the original gene identifiers including the inserted cls_id if append_cls is True, and values as floats). Note: selecting True requires PyTorch to be available; an ImportError will propagate if torch is not installed.
        append_cls (bool): Whether to prepend a classification token to each tokenized sample. If True (default), the function inserts the cls_id value at the front of the per-sample gene sequence and inserts a corresponding 0 value at the front of the expression counts. This produces a sequence where the first token is a fixed class token (commonly used by transformer models such as SCGPT) with zero count.
        include_zero_gene (bool): Whether to include genes with zero expression for each sample. If False (default), the function removes zero-valued features per row and only returns genes with non-zero expression (sparse representation). If True, all n_features are kept (dense representation), and the ordering of genes follows gene_ids.
        cls_id (str): The identifier string to use for the classification token when append_cls is True. Default is "<cls>". When return_pt is True, this string is looked up in the loaded vocabulary; unknown tokens (including an unknown cls_id) map to token id 0 via vocab_map.get(x, 0).
    
    Returns:
        list: A list of length batch_size where each element is a tuple (genes, counts). For each sample:
            - If return_pt is True: genes is a torch.tensor of dtype torch.int64 containing integer token ids obtained by mapping gene identifiers (and the optional cls_id) through the downloaded "scgpt_vocab" mapping (unknown identifiers map to 0). counts is a torch.FloatTensor containing the corresponding expression values (float).
            - If return_pt is False: genes is a numpy.ndarray containing the original gene identifiers (strings or whatever type was provided in gene_ids) with cls_id inserted if append_cls is True; counts is a numpy.ndarray of floats containing the corresponding expression values.
        The tuple therefore represents (gene_tokens_or_ids, expression_counts) for each input row and is intended for direct consumption by downstream SCGPT token-processing or modeling code in TDC.
    
    Behavior, side effects, defaults, and failure modes:
        - The function calls download_wrapper("scgpt_vocab", "./data", ["scgpt_vocab"]) and then pd_load("scgpt_vocab", "./data") to obtain a vocabulary mapping (vocab_map). This may perform network or disk I/O and will create or read the "./data" directory by default. The returned vocab_map is expected to be a mapping from gene identifier (the same type as elements of gene_ids and cls_id) to integer token ids.
        - When return_pt is True, the function performs an in-function import torch; absence of PyTorch will cause an ImportError to be raised at that point.
        - The function validates that data.shape[1] == len(gene_ids). If this is not true, it raises a ValueError describing the mismatch in the number of features versus provided gene_ids.
        - Unknown gene identifiers (those not present in the loaded vocab_map) are mapped to token id 0 when return_pt is True (vocab_map.get(x, 0)). This is the designed fallback behavior and may affect model inputs if many unknowns exist.
        - append_cls inserts cls_id and a zero count at the start of each sequence; the inserted cls_id will be mapped to an integer token id via vocab_map when return_pt is True (or left as the string cls_id when return_pt is False).
        - include_zero_gene controls whether the output sequences are sparse (only non-zero entries per sample) or dense (all features). Using sparse output (include_zero_gene=False) can reduce memory and computation for sparse single-cell datasets but will change ordering and per-sample sequence lengths.
        - The function iterates over rows of data and performs per-row nonzero selection and mapping; runtime and memory usage scale with batch_size and the average number of non-zero features per row.
        - Returned per-sample gene arrays/tensors may have variable length across samples when include_zero_gene is False; downstream code must handle variable-length sequences (for example via padding or batching strategies appropriate for SCGPT).
        - This function is intended specifically to prepare gene-expression inputs for SCGPT-style tokenizers used in TDC model serving and benchmarking workflows (e.g., for tasks in target discovery, activity screening, and related therapeutic machine learning problems).
    """
    from tdc.model_server.tokenizers.scgpt import tokenize_batch
    return tokenize_batch(
        data,
        gene_ids,
        return_pt,
        append_cls,
        include_zero_gene,
        cls_id
    )


################################################################################
# Source: tdc.utils.label.binarize
# File: tdc/utils/label.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_utils_label_binarize(y: list, threshold: float, order: str = "ascending"):
    """Binarization of a label list given a pre-specified numeric threshold for use in TDC data processing and label transformation pipelines.
    
    This function converts a list of continuous or ordinal labels into binary labels (0/1) according to a threshold. It is used in TDC's data processing utilities when transforming regression-style or scored outputs into binary classes for downstream single-instance prediction tasks, leaderboard evaluation, or classifier training. The output is a new numpy array of integer 0/1 values and the input list is not modified in place.
    
    Args:
        y (list): A list of labels to be binarized. In the TDC context this typically contains numeric assay measurements, predicted scores, or other scalar labels produced by datasets or model outputs. Each element of y is compared to threshold with a numeric comparison; elements must be comparable to a float (for example, ints or floats). The function will produce a 1-D output array with the same length as len(y).
        threshold (float): The numeric threshold used to decide class membership. For order="ascending", elements strictly greater than threshold become 1 and all others become 0. For order="descending", elements strictly less than threshold become 1 and all others become 0. Note that values exactly equal to threshold are treated as "not greater" and "not less" and therefore become 0 in both ordering modes.
        order (str, optional): Determines the direction of the binarization rule and defaults to "ascending". If order is "ascending", a label is mapped to 1 when label > threshold and to 0 otherwise. If order is "descending", a label is mapped to 1 when label < threshold and to 0 otherwise. Any other string value for order is invalid and triggers an AttributeError. This parameter is useful when converting a continuous score that is either positively or negatively correlated with the desired positive class (for example, higher potency scores -> positive versus lower toxicity scores -> positive).
    
    Returns:
        np.array: A new 1-D numpy array of integers (0 or 1) containing the binarized labels in the same order as the input list y. The length of the returned array equals len(y). The input list y is not modified; the function allocates and returns a fresh numpy array.
    
    Raises:
        AttributeError: Raised when order is not exactly "ascending" or "descending". The function enforces these two supported modes and will raise AttributeError("'order' must be either ascending or descending") for any other value.
        TypeError: A comparison between elements of y and threshold may raise a TypeError if elements in y are not comparable with a float (for example, if y contains non-numeric types). This TypeError is raised by underlying comparison operations and indicates that the provided labels are not suitable for numeric thresholding.
    """
    from tdc.utils.label import binarize
    return binarize(y, threshold, order)


################################################################################
# Source: tdc.utils.label.convert_back_log
# File: tdc/utils/label.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_utils_label_convert_back_log(y: list):
    """tdc.utils.label.convert_back_log: Convert a list of labels that are in log-scale ("p" units) back to linear nanomolar ("nM") concentration labels.
    
    Args:
        y (list): A list of numeric labels expressed in log-scale ("p") used in TDC datasets and tasks (for example pIC50 or pKi style labels). This argument is the input role for the conversion helper: each element is expected to be a scalar numeric value representing a negative log10 concentration label as used in therapeutics benchmarks (single-instance ADME/activity tasks, etc.). The function will internally convert this Python list to a numpy array before conversion. The original list is not modified by this function.
    
    Returns:
        np.array: A numpy array of converted labels in nanomolar units ("nM"). The return value contains the same number of elements as the input list, with each element converted from the input "p" log-scale to a linear concentration in nM. This return value is suitable for downstream evaluation, model training, or reporting in TDC benchmarks that require concentration values rather than log-scale labels.
    
    Behavior and side effects:
    This helper calls convert_y_unit(np.array(y), "p", "nM") to perform the conversion from "p" to "nM". It always constructs and returns a new numpy array and does not mutate the original input list object. The function is intended to be used when TDC datasets supply labels in log-scale and a user or evaluation routine requires the original concentration units (nM) for interpretation, metrics, or comparisons.
    
    Defaults and failure modes:
    The function assumes that elements of y are numeric (ints or floats or numeric-like strings that convert cleanly via numpy). If y contains non-numeric entries, malformed values, or types that cannot be converted to a numpy numeric array, numpy or convert_y_unit may raise TypeError or ValueError. If y is empty, the function returns an empty numpy array. The function relies on the behavior and correctness of convert_y_unit for the precise conversion semantics; errors raised by convert_y_unit (for example, unknown unit codes) will propagate to the caller.
    """
    from tdc.utils.label import convert_back_log
    return convert_back_log(y)


################################################################################
# Source: tdc.utils.label.convert_to_log
# File: tdc/utils/label.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_utils_label_convert_to_log(y: list):
    """Convert labels from nanomolar concentration units to negative-log "p" scale used in TDC label processing.
    
    This helper is part of TDC's data processing utilities for label transformation and is used when models and evaluation metrics expect potency or concentration labels on a logarithmic "p" scale (for example, converting IC50/EC50 values reported in nanomolar to pIC50-like values). The function converts the input list to a NumPy array and delegates the unit and log conversion to convert_y_unit by calling convert_y_unit(np.array(y), "nM", "p"). The operation is pure (no external side effects) and returns a NumPy array of float log-transformed labels suitable for downstream machine learning pipelines in TDC.
    
    Args:
        y (list): A list of labels to convert. Each element is expected to represent a concentration-like measurement in nanomolar (nM) as a numeric value (for example, IC50 values reported in nM). The function accepts the Python list type as given by the signature and will convert it to a NumPy array internally. Non-numeric entries (for example, strings that cannot be cast to float) can raise a ValueError or TypeError during conversion or may propagate through convert_y_unit; empty lists are accepted and will produce an empty NumPy array.
    
    Returns:
        np.array: A NumPy array containing the labels transformed to the negative-log "p" scale (the "p" unit used by TDC). The returned array has one entry per input element (shape (n,) for n input labels) and is typically of floating-point dtype. If input labels contain zeros or negative values, the log transformation is undefined and may produce -inf, NaN, or raise an error depending on convert_y_unit's handling; callers should validate or filter such values before calling this function if those outcomes are undesirable.
    """
    from tdc.utils.label import convert_to_log
    return convert_to_log(y)


################################################################################
# Source: tdc.utils.label.convert_y_unit
# File: tdc/utils/label.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_utils_label_convert_y_unit(y: list, from_: str, to_: str):
    """label unit conversion helper function used in TDC data processing to standardize assay labels between nanomolar concentration units and negative-log molar ("p") units for downstream model training, evaluation, and leaderboard submission. This function is used in the TDC pipeline to convert lists of measured activity labels (for example, IC50/EC50 values reported in nanomolar or pIC-like units) so that datasets and evaluators operate on consistent units.
    
    Args:
        y (list): A list of labels to convert. Each element is expected to be a numeric measurement corresponding to the value in the unit indicated by from_. In the TDC domain, elements typically represent either measured concentrations in nanomolar (nM) when from_ == 'nM', or negative-log10 molar potency values (for example, pIC50-style values) when from_ == 'p'. The function treats y as an array-like sequence and applies elementwise numeric transformations; non-numeric elements will cause numeric operation errors.
        from_ (str): Source units for the input labels y. Accepted values (as used in the code) are the literal strings 'nM' and 'p'. 'nM' indicates that the input list y is already in nanomolar concentration units (10^-9 M). 'p' indicates that the input list y uses a negative-log10 molar scale (p-type, e.g., pIC50). The function first applies the conversion corresponding to from_ to produce an intermediate numeric representation (the code converts 'p' to nM using the expression (10**(-y) - 1e-10) / 1e-9).
        to_ (str): Target units to produce for the returned labels. Accepted values (as used in the code) are the literal strings 'p' and 'nM'. 'p' requests conversion to the negative-log10 molar scale (p-type); the conversion implemented in the code is y = -np.log10(y * 1e-9 + 1e-10). 'nM' requests nanomolar units and leaves numeric values in nM form.
    
    Behavior and practical details:
        The function performs a two-step conversion: it first checks from_ and, if from_ == 'p', converts the input p-values to nanomolar by applying the formula (10**(-y) - 1e-10) / 1e-9. If from_ == 'nM', the function leaves y unchanged at this stage. After handling from_, it checks to_: if to_ == 'p' it converts the current numeric values to p-units using -np.log10(y * 1e-9 + 1e-10); if to_ == 'nM' it leaves the values unchanged. The small additive constant 1e-10 used in the code acts as a numerical stabilizer to avoid exact zeros inside log10 or division operations; the divisor 1e-9 implements conversion between molar and nanomolar scales (1 M = 1e9 nM). This function is therefore intended to convert between nM and p units commonly encountered in ADME/toxicity and potency datasets in TDC.
        Important implementation note and failure mode: the source code contains a duplicated line in the from_ == 'p' branch (the same transformation is applied twice). As written, that duplicated operation can produce incorrect or numerically unstable results (for example, very large or negative values) for typical p inputs and is a bug in the implementation. Users should verify converted outputs when converting from 'p' inputs and consider correcting the duplicated operation if unexpected values appear.
        Error handling and edge cases: the function does not validate that from_ and to_ are limited to the accepted strings beyond the explicit equality checks; if either parameter is a different string, the corresponding branch is skipped and the values may remain unconverted. Inputs that are zero, negative, or non-numeric may produce misleading results or raise exceptions; the code mitigates exact log(0) by adding 1e-10, but mathematically invalid inputs can still lead to nonsensical outputs. The function relies on numpy operations (np.log10 and elementwise exponentiation) and therefore the module-level numpy import must be available; passing Python lists of numeric scalars is supported because the code applies numpy operations to the sequence.
    
    Returns:
        np.array: A numpy array containing the transformed labels in the requested target units (to_). The returned array is the numeric result of the conversions described above. There are no in-place side effects on the caller's original list object beyond returning the transformed values; however, because of the duplicated transformation present in the implementation when from_ == 'p', returned values may be incorrect for that specific input case unless the implementation bug is addressed.
    """
    from tdc.utils.label import convert_y_unit
    return convert_y_unit(y, from_, to_)


################################################################################
# Source: tdc.utils.label.label_dist
# File: tdc/utils/label.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_utils_label_label_dist(y: list, name: str = None):
    """tdc.utils.label.label_dist plots the distribution of a list of labels (y) using seaborn and matplotlib so users of the Therapeutics Data Commons (TDC) can visually inspect label properties such as skewness, central tendency, spread, and outliers for datasets used in therapeutic machine-learning tasks.
    
    Args:
        y (list): A list of label values for a dataset in TDC. In the therapeutics ML domain these labels typically represent target properties or measurements (for example, activity values, ADME endpoints, or other continuous numeric outcomes). The function computes the numeric median and mean of this list (using numpy) and visualizes the distribution; therefore y must contain numeric values appropriate for median/mean computation. Practical significance: inspecting y with this function helps detect class imbalance, heavy tails, outliers, or multimodality that can inform preprocessing, model selection, and evaluation strategies for TDC tasks.
        name (None, optional): Dataset name to show in the plot title. When provided, the plot title will read "Label Distribution of <name> Dataset"; when omitted (None), the title will be "Label Distribution". In TDC workflows this is used to annotate plots so that distribution diagnostics are traceable back to a specific benchmark or dataset.
    
    Behavior and side effects:
        This function attempts to import seaborn and matplotlib.pyplot. If those imports fail, it calls tdc.utils.misc.install to install "seaborn" and "matplotlib" and then re-imports them; this may require network access and permission to install packages and can fail if installation is not permitted. The function computes median and mean using numpy.median and numpy.mean on y. It creates a matplotlib figure with two stacked subplots sharing the x-axis and with height ratios (0.15, 1). The top subplot is a boxplot summarizing the distribution (ax_box) and the bottom subplot is a kernel density / histogram plot produced by seaborn.distplot (ax_hist). The function draws vertical dashed lines on both subplots for the median (blue) and mean (green) and adds a legend on the histogram mapping "Median" and "Mean" to their numeric values. The function sets the x-label of the boxplot to an empty string and then calls plt.show() to render the figure in the current matplotlib backend. No file output is written by this function; the visible side effect is the display of the figure in the active plotting environment (e.g., interactive notebook, GUI, or non-interactive backend).
    
    Failure modes and notes:
        If y contains non-numeric values, numpy.median/numpy.mean and seaborn plotting functions will raise TypeError or ValueError; the caller should ensure y is a list-like of numeric scalars. If y is empty, numpy may emit warnings or raise errors; behavior depends on the numpy version. If seaborn or matplotlib cannot be installed or imported (for example due to lack of permissions or no network), the function will raise the corresponding ImportError or installation error. In headless or non-interactive environments, plt.show() may not display a window; users should configure an appropriate matplotlib backend (for example, using a non-interactive backend or saving the figure after calling this function). The function does not return any value and does not modify y.
    
    Returns:
        None: This function returns None. Its practical effect in the TDC workflow is to produce an on-screen visualization (boxplot and distribution plot) that helps dataset authors and modelers assess label properties and decide on preprocessing, splitting, and evaluation choices.
    """
    from tdc.utils.label import label_dist
    return label_dist(y, name)


################################################################################
# Source: tdc.utils.label.label_transform
# File: tdc/utils/label.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_utils_label_label_transform(
    y: list,
    binary: bool,
    threshold: float,
    convert_to_log: bool,
    verbose: bool = True,
    order: str = "descending"
):
    """tdc.utils.label.label_transform transforms a list of raw labels used in TDC benchmarks into a numpy array suitable for downstream evaluation or model training by optionally performing binarization or log-scale conversion.
    
    This helper function is used in TDC data processing workflows to prepare target values from datasets (for example continuous biochemical measurements such as Kd or IC50 reported in nM) into either binary class labels for classification tasks or into a log-transformed scale commonly used in pharmacology (e.g., p-scale). It is intended to be called by dataset loaders or preprocessing pipelines when a task requires label normalization or thresholding.
    
    Args:
        y (list): A list of labels in the original dataset order. Elements are expected to be numeric when thresholding or log-conversion is requested. This list may contain continuous values (for example binding affinities in nM) or already-binary values; the function preserves the input order and returns an array of the same length.
        binary (bool): If True and the input list y contains more than two unique values, perform binarization using the supplied threshold and order. Binarization converts a continuous label into a binary class label (1 or 0) to support binary classification benchmarks in TDC (for example converting potency measurements into active/inactive labels). If binary is False, no binarization is attempted.
        threshold (float): Numeric threshold used for binarization when binary is True and y contains more than two unique values. The threshold is compared elementwise against y to assign class 1 or 0 according to the order parameter. The threshold is not applied when the function does not perform binarization.
        convert_to_log (bool): If True and the input list y contains more than two unique values and binarization does not occur (binary is False or not applicable), convert continuous values to a log-scale using the internal convert_y_unit function with unit conversion from "nM" to "p". This option is useful when preparing continuous pharmacological measurements for regression tasks where a negative-log transform (p-scale) is conventional.
        verbose (bool): Whether to print intermediate processing statements to standard error. When True, the function prints a message before binarization and before log conversion. Default is True. The messages are written to sys.stderr and flushed immediately.
        order (str): Determines the direction of binarization when binary is True. If "descending", a label is assigned 1 when the original value is strictly less than threshold (y < threshold) and 0 otherwise. If "ascending", a label is assigned 1 when the original value is strictly greater than threshold (y > threshold) and 0 otherwise. Default is "descending". This parameter controls whether smaller values are considered the positive class ("descending") or larger values are considered the positive class ("ascending").
    
    Returns:
        np.array: A numpy array of transformed labels with the same length and ordering as the input list y. If binarization is performed, the array contains integer-like values 1 or 0. If log conversion is performed, the array contains the numeric log-transformed values returned by convert_y_unit. If neither operation is applicable (for example y is already binary or no transformation flags are set), the original values are returned as-is (converted to a numpy array where appropriate).
    
    Raises:
        ValueError: If binarization is requested but order is not one of the supported strings "descending" or "ascending", a ValueError is raised with a message indicating the allowed options.
        TypeError or ValueError: If elements of y cannot be interpreted as numeric values for comparison with threshold or for log conversion, numpy operations or the underlying convert_y_unit call may raise TypeError or ValueError which will propagate to the caller. These errors indicate incompatible input data types or malformed numeric values.
    
    Notes on behavior and side effects:
        - Binarization takes precedence over log conversion: if y has more than two unique values and binary is True, the function will perform binarization and will not perform log conversion even if convert_to_log is True.
        - The function determines whether y is "non-binary" by checking if the number of unique elements in y is greater than 2; if y already has two or fewer unique values, neither binarization nor log conversion is performed and the input values are returned unchanged (but may be converted to a numpy array).
        - Informational messages are printed to sys.stderr when verbose is True; these messages are flushed immediately.
        - The function relies on convert_y_unit for unit/log conversion; any domain-specific behavior of that helper (for example unit assumptions of "nM" to "p") applies here.
    """
    from tdc.utils.label import label_transform
    return label_transform(y, binary, threshold, convert_to_log, verbose, order)


################################################################################
# Source: tdc.utils.load.atom_to_one_hot
# File: tdc/utils/load.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_utils_load_atom_to_one_hot(atom: str, allowed_atom_list: list):
    """tdc.utils.load.atom_to_one_hot converts a chemical atom label into a one-hot encoded numpy vector suitable for use in TDC molecular featurization pipelines and downstream machine learning models. This helper is used when building per-atom feature vectors (for example in graph-based molecular representations) where each allowed atom type is represented by a unique index and the function produces a vector with a single 1 at that index and 0s elsewhere.
    
    Args:
        atom (str): The atom label to convert to a one-hot vector. In the TDC context this is typically an element symbol or atom type string (e.g., 'C', 'N', 'O') drawn from molecular data. The function uses this exact string to look up the index in allowed_atom_list; it must match one of the entries exactly.
        allowed_atom_list (list(str)): The ordered list of permitted atom labels that defines the one-hot encoding scheme. Each element in this list is a string corresponding to an atom type used across the dataset or featurization. The length and ordering of this list determine the size and index assignment of the output vector; the i-th entry in this list corresponds to the i-th position in the returned one-hot vector.
    
    Behavior and practical details:
        The function determines the index of atom in allowed_atom_list using allowed_atom_list.index(atom) and allocates a new numpy array of shape (len(allowed_atom_list),) initialized to zeros. It then sets the element at the found index to 1 and returns the array. The returned array length equals the number of allowed atom types and is therefore compatible with other per-atom feature vectors used in TDC dataset processing and model inputs. No input arguments are modified; the function creates and returns a newly allocated numpy array. Time complexity is O(n) to find the index where n is len(allowed_atom_list). The numpy array uses numpy's default numeric type (typically float64) as produced by numpy.zeros if not otherwise cast.
    
    Failure modes and edge cases:
        If atom is not present in allowed_atom_list, allowed_atom_list.index(atom) raises a ValueError; callers should ensure the atom label is valid for the chosen allowed_atom_list before calling or handle this exception. If allowed_atom_list does not implement list semantics (for example, lacks an index method) or contains non-string entries that cannot be compared to atom, a TypeError or AttributeError may be raised. The function does not perform normalization of atom strings (such as trimming whitespace or case conversion), so exact string equality is required.
    
    Returns:
        new_atom (numpy.array): A one-dimensional numpy array of shape (len(allowed_atom_list),) representing the one-hot encoding of atom. The array contains a single 1 at the position corresponding to atom in allowed_atom_list and 0s elsewhere. This vector is intended for use as an atomic feature in TDC molecular data processing and machine learning pipelines.
    """
    from tdc.utils.load import atom_to_one_hot
    return atom_to_one_hot(atom, allowed_atom_list)


################################################################################
# Source: tdc.utils.load.bi_distribution_dataset_load
# File: tdc/utils/load.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_utils_load_bi_distribution_dataset_load(
    name: str,
    path: str,
    dataset_names: list,
    return_pocket: bool = False,
    threshold: int = 15,
    remove_protein_Hs: bool = True,
    remove_ligand_Hs: bool = True,
    keep_het: bool = False
):
    """tdc.utils.load.bi_distribution_dataset_load loads and returns processed protein-ligand datasets used for conditional molecular (bi-distribution) generation tasks in TDC. It is a high-level wrapper that accepts a rough dataset name, resolves it against available exact dataset names, optionally triggers downloading/unzipping of dataset archives, calls dataset-specific processing routines (for example, PDBBind, DUD-E, SCPDB, CrossDock), and returns the processed protein and ligand representations that downstream conditional generation models and evaluations consume.
    
    Args:
        name (str): The rough or user-provided dataset name to identify which bi-distribution dataset to load. This value is passed to fuzzy_search to resolve to an exact dataset name from dataset_names; it can be a common alias or partial name used by practitioners when selecting a benchmark (for example, 'pdbbind' or 'dude'). If the resolved exact name maps to entries in name2id or name2idlist, the wrapper will invoke zip_data_download_wrapper to download and prepare the canonical data before processing.
        path (str): Filesystem path where datasets are saved and retrieved. This is the local directory used by the loader and any dataset-specific processors to read input files and to write any intermediate or processed outputs. In practice, this should point to a directory with sufficient disk space because processing structural protein-ligand benchmarks (e.g., PDBBind, CrossDock) may create multiple files and temporary directories.
        dataset_names (list): The list of availabel exact dataset names that fuzzy_search will use to resolve the provided name string. This list represents the canonical dataset identifiers supported by this loader for conditional protein-ligand generation tasks and is used to validate and match user input to a known dataset.
        return_pocket (bool): If True, request extraction and return of the protein pocket (local binding site) representation together with ligand data when the underlying dataset and processor support pocket extraction. Default False. Note: not all datasets support pocket extraction; for example, if name resolves to 'dude' and return_pocket is True, the function raises ImportError because DUD-E does not support pocket extraction in the current implementation.
        threshold (int): Integer threshold parameter (default 15) accepted by the loader for compatibility with dataset-specific processors and potential filtering rules. This wrapper forwards threshold to dataset processing routines when those routines accept it; in general, threshold is used by processors to apply dataset-specific numeric cutoffs or size filters during processing. The concrete interpretation of threshold (for example, minimum residue count, distance cutoff, or other dataset-specific filter) depends on the dataset processor invoked.
        remove_protein_Hs (bool): Whether to remove hydrogen atoms from protein structures during processing (default True). Removing protein hydrogens is a common preprocessing step in structure-based datasets to standardize representations and reduce file size and computational overhead for downstream modeling. Set to False to preserve explicit hydrogens when required by specific modeling workflows.
        remove_ligand_Hs (bool): Whether to remove hydrogen atoms from ligand structures during processing (default True). Analogous to remove_protein_Hs, this flag controls whether ligand-level hydrogens are stripped by processors; preserving ligand hydrogens may be necessary for certain physicochemical calculations or force-field preparations.
        keep_het (bool): Whether to keep heteroatoms (HETATM records) from protein structure files (default False). In structural datasets, HETATM entries typically include non-standard residues, cofactors, and crystallographic additives; setting keep_het=True preserves these entries for downstream use, while False removes them to yield a canonicalized protein representation.
    
    Returns:
        tuple(pandas.Series, pandas.Series): A pair (protein, ligand) containing the processed representations returned by the dataset-specific processors. Each element is a pandas.Series or a dataset-defined object holding per-sample structural or cheminformatic representations that downstream conditional generation models expect: the first element corresponds to protein-side data (for example, full protein or extracted pocket representations when return_pocket=True and supported), and the second to ligand-side data. The exact internal layout and fields of these Series are dataset-dependent and produced by process_pdbbind, process_dude, process_scpdb, or process_crossdock.
    
    Behavior and side effects:
        This function performs name resolution via fuzzy_search, may trigger downloads/unzipping via zip_data_download_wrapper when shorthand names map to remote archives, and then runs a dataset-specific processing routine that can be time- and disk-intensive (the function prints "Processing (this may take long)..." before invoking processing). Processing routines may write processed files under path and may create temporary files. The function raises ImportError when requested features are unsupported for a given dataset (for example, requesting pocket extraction for DUD-E). Dataset processors may raise their own exceptions for missing files, malformed inputs, or missing third-party dependencies; callers should handle such exceptions and ensure that path is readable/writable and that sufficient disk space and permissions exist. Default parameter values are chosen for common workflows in structure-based conditional generation: removing hydrogens and not keeping heteroatoms reduces representation complexity for many models, while threshold provides a backward-compatible tuning knob for processors that implement dataset-specific filters.
    """
    from tdc.utils.load import bi_distribution_dataset_load
    return bi_distribution_dataset_load(
        name,
        path,
        dataset_names,
        return_pocket,
        threshold,
        remove_protein_Hs,
        remove_ligand_Hs,
        keep_het
    )


################################################################################
# Source: tdc.utils.load.bm_download_wrapper
# File: tdc/utils/load.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_utils_load_bm_download_wrapper(name: str, path: str):
    """tdc.utils.load.bm_download_wrapper: Download and prepare a benchmark group for use in the Therapeutics Data Commons (TDC) benchmarks. This function resolves a user-provided approximate benchmark group name to the exact benchmark identifier used by the TDC registry, downloads the corresponding dataset archive from the Harvard Dataverse API, extracts it into a local directory, and returns the resolved exact benchmark group query name. In the TDC domain this is used to obtain benchmark groups (collections of dataset files and metadata) that are required for training, evaluation, and leaderboard submission workflows for therapeutic machine learning tasks.
    
    Args:
        name (str): A user-specified, potentially imprecise query string for the benchmark group to download. The function uses fuzzy_search against the list of available benchmark group names (benchmark_names.keys()) to resolve this to the exact canonical benchmark group name used internally. This argument represents the logical benchmark group identifier the caller expects (for example, an approximate task or group name from the TDC website) and is returned as the resolved exact name on success.
        path (str): Filesystem path to the directory where the benchmark group archive and extracted files should be saved. If the directory specified by path does not exist, the function attempts to create it. The function will save the downloaded ZIP file at os.path.join(path, name + ".zip") (as produced by the dataverse_download call) and extract its contents into the provided path, resulting in a local directory for the benchmark group (commonly os.path.join(path, name) after extraction). This argument must be a valid string path accessible by the running process.
    
    Behavior, side effects, defaults, and failure modes:
        The function first resolves the provided name by calling fuzzy_search(name, list(benchmark_names.keys())) to obtain the exact benchmark group name present in the TDC registry. It constructs a download URL using the Harvard Dataverse API base server_path ("https://dataverse.harvard.edu/api/access/datafile/") concatenated with the integer file identifier obtained from the benchmark2id mapping for the resolved name. If the directory specified by path does not exist, the function creates it using os.mkdir(path). If a local directory corresponding to the resolved benchmark group (os.path.join(path, name)) already exists, the function prints a message via print_sys and does not re-download or re-extract the archive; it simply returns the resolved name. Otherwise, the function calls dataverse_download(dataset_path, path, name, benchmark2type) to download the archive; this call is expected to produce a ZIP file named os.path.join(path, name + ".zip"). After download, the function opens and extracts that ZIP archive using ZipFile and extracts all files into the provided path. The function prints status messages through print_sys at major steps (found local copy, downloading, extracting, done). On success, the function returns the resolved exact benchmark group name as a Python str.
    
        Side effects include creating the specified path directory if missing, writing the downloaded ZIP file to disk at path, extracting files into path (creating or overwriting files/directories under path), and printing progress messages. The function depends on global mappings and utilities in the tdc.utils.load module such as benchmark_names, benchmark2id, benchmark2type, fuzzy_search, dataverse_download, and print_sys; missing or incorrect global mappings will cause failures.
    
        Failure modes include but are not limited to: fuzzy_search raising an exception if the query cannot be resolved; a missing key in benchmark2id or benchmark2type leading to a KeyError; dataverse_download or the underlying network stack raising exceptions on network failures; insufficient filesystem permissions causing OSError when creating directories or writing files; ZipFile raising ZipFile.BadZipFile or other exceptions if the downloaded file is not a valid ZIP; and other runtime errors raised by the helper functions. Callers should handle these exceptions or ensure that the environment, network access, and global mappings are correctly configured before invoking this function.
    
    Returns:
        str: The exact benchmark group query name resolved from the provided name. This exact name corresponds to the canonical key present in the TDC benchmark registry (benchmark_names) and should be used for subsequent dataset access or bookkeeping.
    """
    from tdc.utils.load import bm_download_wrapper
    return bm_download_wrapper(name, path)


################################################################################
# Source: tdc.utils.load.bm_group_load
# File: tdc/utils/load.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_utils_load_bm_group_load(name: str, path: str):
    """bm_group_load loads a benchmark group by downloading, processing, and making its files available under a local path. This utility is a thin wrapper around the internal bm_download_wrapper function and is used by TDC workflows to retrieve curated benchmark groups (for example, ADMET_Group) so they can be used for dataset splits, model training, and leaderboard submission as described in the TDC README.
    
    Args:
        name (str): the rough benchmark group name. This is the identifier or human-provided label used to look up the benchmark group remotely (for example, the high-level group name shown on the TDC website). The function uses this name to locate the canonical benchmark group, and the returned value is the exact/canonical name resolved by the downloader.
        path (str): the benchmark group path to save/retrieve. This is a local filesystem directory where downloaded benchmark group files, processed artifacts, and any extracted archives will be written and later read from. The function may create subdirectories under this path and will persist files there for subsequent TDC API calls.
    
    Returns:
        str: exact benchmark group name. The returned string is the canonical benchmark group name resolved by the download/process pipeline (i.e., the precise name used internally by TDC to refer to the loaded group). Downstream TDC code (for example, BenchmarkGroup constructors or data retrieval functions) relies on this canonical name to reference the loaded benchmark.
    
    Behavior and side effects:
        This function delegates to bm_download_wrapper(name, path) to perform network retrieval, local processing, and any post-download normalization. As a result, it will create and modify files under the provided path, including creating directories, writing downloaded archives, and writing processed dataset files. The function returns the canonical name produced by the wrapper. It does not itself perform additional validation beyond what bm_download_wrapper implements.
    
    Failure modes and errors:
        Errors raised by the underlying bm_download_wrapper propagate to the caller. Common failure modes include network connectivity errors when fetching remote data, permission or filesystem errors when creating or writing files under path (OSError), and invalid or unrecognized benchmark group names which may result in a ValueError or a wrapper-specific exception. Callers should ensure that path is writable and that the provided name corresponds to a valid TDC benchmark group.
    """
    from tdc.utils.load import bm_group_load
    return bm_group_load(name, path)


################################################################################
# Source: tdc.utils.load.dataverse_download
# File: tdc/utils/load.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_utils_load_dataverse_download(
    url: str,
    path: str,
    name: str,
    types: dict,
    id: int = None
):
    """dataverse_download downloads a dataset file from a Dataverse-style URL to local disk with a streaming progress bar and a deterministic filename pattern used by TDC dataset loaders.
    
    Args:
        url (str): The full HTTP(S) URL to download the dataset file from. In the TDC context this is typically a Dataverse file download endpoint for a specific dataset in the Therapeutics Data Commons repository. The function issues a streaming GET request (requests.get(..., stream=True)) to this URL and writes the response body to disk incrementally. The caller must ensure the URL is reachable and returns the expected file content.
        path (str): The directory path where the downloaded file will be saved. The function uses os.path.join(path, <filename>) to construct the destination path. The function does not create intermediate directories; provide an existing writable directory or create it beforehand to avoid OSError when opening the output file.
        name (str): The logical dataset name used by TDC (for example, a dataset key used by higher-level loaders). This name is used to look up the file format in the types mapping and to construct the local filename. The saved filename follows the pattern "<name>.<ext>" when id is None, or "<name>-<id>.<ext>" when id is provided.
        types (dict): A dictionary mapping dataset names (keys corresponding to name) to file format identifiers (values). In practice within TDC this mapping maps the dataset name to the file extension (for example, {"HIA_Hou": "csv"}). The function looks up types[name] to determine the extension appended to the saved filename. If name is not present in types a KeyError will be raised.
        id (int): Optional integer identifier appended to the filename when provided. When id is None the saved filename is "<name>.<ext>". When id is an integer the saved filename is "<name>-<id>.<ext>". id preserves ordering or distinguishes multiple related files for the same logical dataset within TDC. If a non-integer is passed despite the signature, it will be converted to string for filename construction, but callers should pass int or None as intended by the TDC API.
    
    Returns:
        None: This function does not return a value. Side effects: it performs an HTTP GET request to url with streaming enabled and writes the response content to a file at the constructed save path. It displays and updates a tqdm progress bar reflecting bytes written (uses block_size = 1024 and unit="iB", unit_scale=True). On successful completion the file is closed and the progress bar is closed.
    
    Behavior, defaults, and failure modes:
        - Filename construction: if id is None the destination filename is name + "." + types[name]; otherwise it is name + "-" + str(id) + "." + types[name]. The types mapping must contain name; otherwise a KeyError occurs.
        - Streaming and progress: the function requests the resource with stream=True and iterates over response.iter_content(block_size) where block_size is fixed at 1024 bytes. It reads data in 1 KiB chunks and updates the tqdm progress bar for each chunk. If the HTTP response includes a numeric "content-length" header it is used to set the progress total; if that header is absent or non-numeric the progress bar will not display a reliable overall percentage and int(...) may raise ValueError for malformed headers.
        - HTTP and network errors: the function does not call response.raise_for_status(); non-2xx HTTP responses will still cause the body to be streamed and written as-is. Network interruptions, timeouts, or connection errors raised by requests will propagate as exceptions (e.g., requests.exceptions.RequestException). Callers may want to validate response.status_code before or after calling this helper when strict HTTP error handling is required.
        - File I/O errors: opening or writing to the destination file may raise OSError (for example, permission denied, disk full, or missing directory). The function will overwrite an existing file at the same path without prompting.
        - Partial files and interruption: if the download is interrupted (exception, user interrupt, or network failure), a partially written file may remain at save_path. The function does not provide automatic resume support; callers should manage retries and cleanup as needed.
        - Content-length parsing: converting the "content-length" header to int may raise ValueError if the header is present but not a valid integer. In that case the progress bar total will not be set correctly and the exception will propagate.
        - Intended use in TDC: this utility is a lightweight helper used by TDC data loaders to fetch raw dataset files (for tasks such as single-instance ADME datasets, multi-instance tasks, or generation dataset artifacts). It is designed to be simple and dependency-light, providing visible download progress when retrieving benchmark files for downstream model training and evaluation.
    """
    from tdc.utils.load import dataverse_download
    return dataverse_download(url, path, name, types, id)


################################################################################
# Source: tdc.utils.load.distribution_dataset_load
# File: tdc/utils/load.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_utils_load_distribution_dataset_load(
    name: str,
    path: str,
    dataset_names: list,
    column_name: str
):
    """tdc.utils.load.distribution_dataset_load loads a single column of molecule representations for a distribution learning dataset after ensuring the dataset file is available locally.
    
    This function is a thin wrapper used in the Therapeutics Data Commons (TDC) to (1) resolve and, if necessary, download a distribution learning dataset using download_wrapper, (2) load the dataset into a pandas DataFrame using pd_load, and (3) return the specified column that contains molecule representations used for distribution learning (for example, SMILES strings typically used by TDC oracles). The function assumes the downloaded file is already processed into a table-like format with a column that holds molecule representations; it does not further process or validate per-entry molecular formats.
    
    Args:
        name (str): The rough dataset identifier provided by the caller. In the TDC workflow this name is passed to download_wrapper which resolves it to an exact dataset file or name (potentially overwriting or replacing this value with the resolved filename). The value should be a recognizable dataset key within TDC's available distribution-learning datasets.
        path (str): The local filesystem path (directory) used to save or retrieve dataset files. This is the location download_wrapper may write to and pd_load will read from. The path should be writable when a download is required and readable for subsequent loads.
        dataset_names (list): A list of exact dataset names that are considered valid targets for resolution by download_wrapper. This list guides download_wrapper in matching or validating the requested dataset. Each element is an exact name string as known to the TDC distribution-learning registry.
        column_name (str): The exact column name in the loaded pandas DataFrame that contains the molecule representations to return. This column specifies where each molecule is located (for example, a SMILES string column commonly used by generation/oracle workflows in TDC). The function will return the column values exactly as stored in the DataFrame.
    
    Behavior and side effects:
        The function calls download_wrapper(name, path, dataset_names) which may perform network I/O to download dataset files into path and may update or replace the input name with a resolved filename. It emits a "Loading..." message via print_sys() to indicate progress. It then calls pd_load(name, path) to read the dataset into a pandas DataFrame. The function returns df[column_name] without modifying the entries. Persistent side effects include files written to the provided path when downloads occur and console output from print_sys. The function expects the stored file to already be processed into a tabular form; it will not attempt to preprocess raw archives into a DataFrame.
    
    Failure modes and errors:
        If download_wrapper fails (e.g., network error, missing registry entry for name), an exception from download_wrapper will propagate. If pd_load cannot find or read the resolved dataset file, file I/O errors (such as FileNotFoundError or pandas read errors) will propagate. If the loaded object is not a pandas DataFrame or does not contain column_name, a KeyError or TypeError may be raised when accessing df[column_name]. Callers should ensure dataset_names contains valid exact names and that column_name exists in the processed dataset table prior to use.
    
    Returns:
        pandas.Series: A pandas Series view of the DataFrame column specified by column_name. Each element is a molecule representation used for distribution learning (e.g., SMILES strings or other representation stored by the dataset). The Series preserves the DataFrame index and contains the raw stored values without additional parsing or validation.
    """
    from tdc.utils.load import distribution_dataset_load
    return distribution_dataset_load(name, path, dataset_names, column_name)


################################################################################
# Source: tdc.utils.load.download_wrapper
# File: tdc/utils/load.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_utils_load_download_wrapper(name: str, path: str, dataset_names: list):
    """Concise wrapper to locate and download a TDC dataset file (csv, pkl, or tsv) from the Harvard Dataverse given a fuzzy query name, saving it under a local directory. This function is used by the Therapeutics Data Commons (TDC) data loaders to resolve a user-provided dataset query into the exact dataset name present in the TDC name registries, create the target directory if missing, check for an existing local copy, and call the Dataverse download helper to retrieve one or more files for that dataset.
    
    Args:
        name (str): the rough dataset query name provided by the caller. This is the user-facing identifier that may be incomplete or fuzzy; the function calls fuzzy_search(name, dataset_names) to resolve it to the exact dataset key used by the TDC registries. The resolved exact name is returned and is the canonical dataset identifier used by downstream TDC code.
        path (str): the filesystem directory path where downloaded dataset file(s) will be saved. The function ensures this directory exists by creating it with os.mkdir(path) when absent. Files are written with names constructed from the resolved dataset name and the extension mapped via the module-level mapping name2type (for example csv, pkl, or tsv). If the caller does not have permission to create or write to this path, an OSError (or subclass) may be raised.
        dataset_names (list): the list of available dataset names used by fuzzy_search to match the provided rough name to a canonical TDC dataset name. This list is typically derived from the TDC dataset registry and guides which exact dataset the function will attempt to download.
    
    Returns:
        str: the exact dataset query name resolved from the input. This return value is the canonical dataset key as found in TDC internal mappings (name2idlist or name2id) and indicates which dataset the function either found locally or attempted to download.
    
    Behavior and side effects:
    This function first resolves the provided name via fuzzy_search using dataset_names; the resolved name replaces the original input name for subsequent steps. The function consults two module-level mappings: name2idlist (mapping a dataset name to a list of Dataverse file ids when a dataset comprises multiple files) and name2id (mapping a dataset name to a single Dataverse file id). If the resolved name is present in name2idlist, the function iterates over all associated ids and for each id constructs a Dataverse API URL using the fixed server base "https://dataverse.harvard.edu/api/access/datafile/". For datasets with multiple files, downloaded files are named using the pattern "<name>-<index>.<extension>" where <index> is 1-based and <extension> is looked up from name2type[name]. For single-file datasets (resolved name found in name2id but not in name2idlist), the saved filename is "<name>.<extension>". Before downloading each file, the function checks for an existing local copy with os.path.exists and skips downloading if the expected filename already exists, printing "Found local copy..." via print_sys. When a file is not present locally, the function prints "Downloading..." via print_sys and invokes dataverse_download to perform the HTTP transfer and write the file into path.
    
    Failure modes and exceptions:
    If fuzzy_search cannot resolve the input name to a canonical name based on dataset_names, the behavior depends on fuzzy_search implementation and may raise an error; the function assumes fuzzy_search returns a valid key. If the resolved name is not present in either the name2idlist or name2id module-level mappings, a KeyError will be raised when the code attempts to index those mappings. Filesystem errors such as inability to create the directory or write files will raise OSError (or its subclasses). Network, authentication, or HTTP errors raised by dataverse_download will propagate to the caller. The function prints status messages but does not return download progress; callers should handle exceptions from underlying helpers if they need programmatic error handling.
    
    Notes on practical significance:
    This helper centralizes the dataset resolution and safe-download pattern used throughout TDC data loaders so that higher-level APIs (for example, dataset classes exposed under tdc.single_pred, tdc.multi_pred, and tdc.generation) can request datasets by a fuzzy name and rely on consistent local naming, directory management, and Dataverse access. The returned exact dataset name can be used by callers and other TDC utilities to index metadata, determine file formats via name2type, and verify which dataset files were obtained.
    """
    from tdc.utils.load import download_wrapper
    return download_wrapper(name, path, dataset_names)


################################################################################
# Source: tdc.utils.load.general_load
# File: tdc/utils/load.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_utils_load_general_load(name: str, path: str, sep: str):
    """tdc.utils.load.general_load downloads (if necessary), processes, and loads a TDC dataset file into a pandas.DataFrame.
    
    This utility is used within the TDC (Therapeutics Data Commons) data-loading workflow to retrieve any dataset registered in TDC, making it available for downstream model training, evaluation, and benchmark submission as described in the TDC README. The function delegates fetching and pre-processing to the internal download_wrapper, determines the file extension using the module-level name2type mapping, prints a brief status message, and then reads the saved file using pandas.read_csv with the provided separator. It performs no further content validation beyond what pandas.read_csv enforces.
    
    Args:
        name (str): The dataset name used to identify and fetch a dataset from the TDC repository and internal mappings. This exact identifier is passed to download_wrapper(name, path, name) and is used as a key into the module-level name2type mapping to determine the file extension (for example, mapping name -> 'csv' or other supported file type). In practice this is the canonical dataset key that appears on the TDC website and in TDC loader APIs; supplying an incorrect or unknown name may cause a KeyError or a failed download.
        path (str): The local filesystem directory where the dataset file is saved and from which it will be loaded. If the file is not already present at this path, download_wrapper is invoked and will attempt to save the downloaded/processed file under this directory; after download_wrapper returns, general_load constructs the full file path as os.path.join(path, name + "." + name2type[name]) and reads from it. The caller is responsible for providing a valid path string; filesystem permission errors or nonexistent directories can cause OSError/FileNotFoundError depending on the environment and on download_wrapper behavior.
        sep (str): The delimiter string passed directly to pandas.read_csv to parse the dataset file. This controls how pandas splits columns when reading the file (for example ',' for comma-separated values or '\\t' for tab-separated values). sep must be a string accepted by pandas.read_csv; an incorrect separator may produce parsing errors or incorrect columnization.
    
    Returns:
        pandas.DataFrame: A pandas.DataFrame containing the dataset read from the file. The DataFrame reflects the raw parsed contents of the file as returned by pandas.read_csv with the given sep; column names, dtypes, and contents depend on the specific TDC dataset. Practical significance: the returned DataFrame is the primary in-memory representation used by higher-level TDC APIs (data loaders, splitters, evaluators) for training, validation, testing, and leaderboard evaluation.
    
    Behavior and side effects:
        - Calls download_wrapper(name, path, name) which may perform network I/O to fetch and preprocess the dataset and will typically write one or more files under the provided path. The function then prints a status message via print_sys("Loading...") and reads the dataset file using pandas.read_csv.
        - The file extension is determined from the module-level mapping name2type[name]; this mapping must contain an entry for the provided name.
        - No additional content validation or schema enforcement is performed after pandas.read_csv; downstream code is expected to validate required columns or formats.
    
    Failure modes and exceptions:
        - KeyError if name is not present in the module-level name2type mapping.
        - FileNotFoundError or OSError if the expected file does not exist after download_wrapper or if there are filesystem access problems.
        - pandas.errors.ParserError or other pandas I/O exceptions if the file cannot be parsed with the provided sep.
        - Any network-related exceptions that download_wrapper may raise during download (propagated to the caller).
        - Other unexpected exceptions from download_wrapper, os.path.join, or pandas.read_csv may propagate to the caller.
    
    Notes:
        - The function requires the caller to supply all three positional arguments (name, path, sep); there are no defaults in the signature.
        - The returned DataFrame is intended to be used directly by TDC data functions (e.g., get_split, Evaluator workflows) as described in the TDC documentation.
    """
    from tdc.utils.load import general_load
    return general_load(name, path, sep)


################################################################################
# Source: tdc.utils.load.generation_dataset_load
# File: tdc/utils/load.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_utils_load_generation_dataset_load(name: str, path: str, dataset_names: list):
    """tdc.utils.load.generation_dataset_load loads a processed dataset for a generation task in TDC, optionally downloading it first and returning the dataset columns used for generative model training and evaluation.
    
    This function is a thin wrapper that (1) selects or downloads a dataset by delegating to download_wrapper, (2) prints a progress message via print_sys, and (3) loads the processed dataset file via pd_load. It is intended for use with TDC's "generation" problem (generation of new desirable biomedical entities such as molecules or sequences) and returns the two pandas Series columns commonly used by generation benchmarks: an "input" series (conditioning inputs or prompts) and a "target" series (the desired generated entities, e.g., SMILES strings). Typical usage is to obtain training/validation/test inputs and targets for molecule generation oracles and leaderboards in TDC.
    
    Args:
        name (str): The rough dataset name or identifier passed by the caller. In the TDC workflow this name is used by download_wrapper to locate and, if necessary, download the processed dataset. Its practical significance is that it selects which generation benchmark or dataset to load; it must match one of the available dataset identifiers known to download_wrapper or be transformed by download_wrapper into an exact dataset file name on disk.
        path (str): The filesystem path (directory) where datasets are saved or retrieved. This path is used by download_wrapper and pd_load to store downloaded files and to read processed dataset files. In practice, provide a writable directory for downloads and a readable directory to load previously saved processed dataset files.
        dataset_names (list): A list of available exact dataset names that download_wrapper can choose from. This list guides download_wrapper to map the rough name to an exact dataset file name. The entries should be the canonical dataset names used by the TDC generation task registry; providing an incorrect type or an empty list may cause download_wrapper to raise an error.
    
    Returns:
        pandas.Series, pandas.Series: Two pandas.Series objects read from the processed dataset file: the first is df["input"], a series of conditioning inputs or prompts for the generation task (for example scaffolds, property vectors, or other contextual inputs used to condition generative models); the second is df["target"], a series of desired generated entities (for example SMILES strings, peptide sequences, or other target representations). These Series are intended for direct use in training, validation, testing, and evaluation of generation oracles and benchmarks in TDC.
    
    Behavior and side effects:
        This function calls download_wrapper(name, path, dataset_names) which may perform network I/O and write files to path; if the file already exists, download_wrapper may return a local name without re-downloading. The function prints a "Loading..." message via print_sys and then loads the processed dataset using pd_load(name, path). The loaded DataFrame is expected to contain "input" and "target" columns; these are extracted and returned as pandas.Series objects.
    
    Failure modes and errors:
        If dataset_names is not a list, if name cannot be resolved to an available dataset by download_wrapper, or if a download fails, download_wrapper may raise a ValueError, TypeError, or a network-related exception. If pd_load cannot find or parse the file at path or if the loaded DataFrame does not contain the required "input" and "target" columns, pd_load or this function will raise IOError, FileNotFoundError, KeyError, or pandas parsing errors. Callers should ensure path is accessible and that dataset_names contains the canonical dataset identifiers.
    """
    from tdc.utils.load import generation_dataset_load
    return generation_dataset_load(name, path, dataset_names)


################################################################################
# Source: tdc.utils.load.generation_paired_dataset_load
# File: tdc/utils/load.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_utils_load_generation_paired_dataset_load(
    name: str,
    path: str,
    dataset_names: list,
    input_name: str,
    output_name: str
):
    """tdc.utils.load.generation_paired_dataset_load loads and returns paired data series for a generation task from TDC datasets, handling download, local caching, and DataFrame column extraction for paired inputs and outputs used in generation problems (for example, scaffold-to-molecule or precursor-to-product pairs in molecular generation benchmarks).
    
    Args:
        name (str): The rough dataset name provided by the caller. In the TDC workflow this is typically a human-friendly identifier for a generation benchmark (for example, a short name that maps to one of the exact dataset names). This function passes name to the internal download/selection helper to resolve and obtain the exact dataset identifier to use for loading.
        path (str): The filesystem path where datasets are saved or should be retrieved from. This path is used by the function to locate cached dataset files and to save any downloaded files. It determines the local storage location for dataset artifacts and therefore affects disk I/O, caching behavior, and reproducibility when reusing previously downloaded datasets.
        dataset_names (list): A list of available exact dataset names that the download/resolution helper can choose from. This list is used to disambiguate and validate the requested dataset when the provided name is a rough identifier. It is expected to contain the canonical dataset names for generation-paired tasks; the function will delegate to download_wrapper(name, path, dataset_names) to select and/or fetch the appropriate exact dataset.
        input_name (str): The column name in the loaded pandas DataFrame that contains the input entity representation for the paired generation task. In generation datasets this commonly corresponds to the conditioning representation (for example, a scaffold, starting fragment, or any input representation used to generate a target molecule). This argument must match an existing column in the dataset; otherwise a KeyError will be raised when attempting to index the DataFrame.
        output_name (str): The column name in the loaded pandas DataFrame that contains the output entity representation or label for the paired generation task. In molecular generation contexts this often corresponds to the target molecule SMILES string, product representation, or desired output modality paired to the input. This argument must match an existing column in the dataset; otherwise a KeyError will be raised when attempting to index the DataFrame.
    
    Behavior and side effects:
        The function first calls the internal download and selection helper download_wrapper(name, path, dataset_names). That helper may perform network I/O to download dataset files into path, may validate and select an exact dataset name from dataset_names, and may raise exceptions on network failures or if the requested dataset cannot be found. After resolution, the function prints a short status message via print_sys("Loading...") and then loads the dataset using pd_load(name, path), which returns a pandas.DataFrame. The function then extracts the two columns specified by input_name and output_name and returns them. This function therefore has side effects: it may create or modify files under path, perform network downloads, and emit console output. It does not modify the input DataFrame in-place.
    
    Failure modes and errors:
        If download_wrapper cannot resolve or download the dataset, it may raise an exception (for example, a ValueError or an I/O/network-related exception) propagated to the caller. If pd_load fails to read the dataset file (corrupted or missing file), it will raise the underlying pandas/file I/O exception. If input_name or output_name are not present as columns in the loaded DataFrame, a KeyError will be raised when attempting to access df[input_name] or df[output_name]. The function does not perform additional type conversion: the returned objects are the original pandas.Series extracted from the DataFrame.
    
    Returns:
        tuple(pandas.Series, pandas.Series): A pair of pandas.Series objects. The first element is the series corresponding to the column named by input_name and represents the input entity representations (conditioning data) for the generation task. The second element is the series corresponding to the column named by output_name and represents the paired output entity representations or labels (targets) used for generation evaluation or training. These series retain the original ordering and index of the loaded DataFrame and are intended to be used directly by downstream data processing, model training, or evaluation code in generation benchmarks.
    """
    from tdc.utils.load import generation_paired_dataset_load
    return generation_paired_dataset_load(
        name,
        path,
        dataset_names,
        input_name,
        output_name
    )


################################################################################
# Source: tdc.utils.load.multi_dataset_load
# File: tdc/utils/load.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_utils_load_multi_dataset_load(name: str, path: str, dataset_names: list):
    """Summary:
        Load a processed multi-instance prediction dataset for TDC (Therapeutics Data Commons).
        This function is a thin wrapper that ensures a dataset file for multi-instance (>2)
        prediction tasks is available on disk (under the provided path) and then reads it into
        a pandas.DataFrame for downstream use in model training, evaluation, or further
        TDC data functions. It is intended for datasets in the TDC "multi_pred" problem tier,
        where each sample consists of multiple biomedical entities (for example, combinatorial
        therapies or multi-drug experiments) and where the on-disk file is already in a
        processed, model-ready tabular format.
    
    Args:
        name (str): A rough or user-facing dataset identifier. In the TDC hierarchy this
            corresponds to a dataset requested by the caller (for example, a shorthand or
            partial name for a multi-instance benchmark). The value is passed to the
            internal download/resolve helper to find or download the exact dataset file.
        path (str): Filesystem path (directory) to save or retrieve the dataset file. This
            path is used by the internal download and file-load utilities. Side effects:
            the function may create files or directories under this path when ensuring the
            dataset is present; callers should provide a writable path and be aware that
            existing files under this path may be read or overwritten depending on the
            internal download/resolve behavior.
        dataset_names (list): A list of available exact dataset names (strings). The
            internal resolver uses this list to match or validate the requested name and
            to determine which exact dataset to download or load. The list must contain the
            canonical names known to the TDC dataset registry for multi-instance tasks.
    
    Returns:
        pandas.DataFrame: The raw dataset table loaded from disk. The returned DataFrame is
        the un-split, processed tabular representation of the requested multi-instance
        prediction dataset (columns and semantics follow the specific dataset's schema in
        TDC). No additional splitting, label extraction, or downstream transformations are
        performed by this function; downstream code should call the appropriate TDC data
        functions (for example, data.get_split or other processors) to obtain training/
        validation/test partitions or to transform labels.
    
    Behavior, side effects, and failure modes:
        - The function delegates to an internal helper (download_wrapper) to resolve the
          provided rough name against dataset_names and to ensure the dataset file exists
          under path. As part of that step the helper may attempt a network download or
          file copy; therefore network failures, permission errors, or missing dataset
          registrations will surface as exceptions raised by the helper.
        - After ensuring the file is present, the function prints "Loading..." via the
          tdc print_sys utility and calls pd_load(name, path) to read the dataset into a
          pandas.DataFrame. pd_load is expected to parse a processed, model-ready file
          (CSV/TSV/serialized pandas format depending on TDC conventions).
        - Common failure modes include: inability to resolve name to an exact dataset (e.g.,
          name not found in dataset_names), network or download failures, insufficient
          filesystem permissions at path, missing or malformed dataset files that cause the
          pandas loader to raise parsing errors, and other I/O errors. Callers should catch
          and handle FileNotFoundError, ValueError, IOError, or the specific exceptions
          raised by the TDC download and load utilities when integrating this function.
        - There are no implicit defaults or retries performed by this wrapper beyond the
          behavior of the underlying helpers. The function assumes the on-disk dataset file
          is already processed to TDC's dataset schema; it does not perform additional
          dataset-specific processing or label selection.
    """
    from tdc.utils.load import multi_dataset_load
    return multi_dataset_load(name, path, dataset_names)


################################################################################
# Source: tdc.utils.load.oracle_download_wrapper
# File: tdc/utils/load.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_utils_load_oracle_download_wrapper(name: str, path: str, oracle_names: list):
    """Wrapper to locate and (if needed) download a TDC molecule-generation oracle model checkpoint into a local directory and return the canonical oracle name.
    
    Args:
        name (str): A rough or user-provided oracle query name. In the Therapeutics Data Commons (TDC) context this is typically a short identifier or partial name for a molecule-generation oracle (used for goal-oriented or distribution-learning generation tasks). The function uses fuzzy_search against the supplied oracle_names list to resolve this rough query into an exact, canonical oracle name that matches an available oracle. The argument is required and must be a Python str.
        path (str): Filesystem directory path where the resolved oracle checkpoint (model file) should be stored. If the directory does not exist, the function will attempt to create it (os.mkdir). This is a directory path, not a full filename; the function composes the saved filename as <resolved_name>.<oracle_extension> where the extension is looked up from the module-level mapping oracle2type.
        oracle_names (list): The list of available exact oracle names (strings) against which the given rough name will be matched. This list enumerates canonical oracle identifiers known to TDC; fuzzy_search uses it to find the best match. The argument must be a Python list (typically of str), and the function will return one of the entries from this list (subject to fuzzy_search behavior).
    
    Behavior and side effects:
        1. Name resolution: The function calls fuzzy_search(name, oracle_names) to resolve the provided rough name to an exact oracle name present in oracle_names. The returned canonical name is then used for subsequent steps. The practical significance is that users may pass approximate or partial names and the helper will resolve them to the exact oracle identifier used by TDC.
        2. Trivial oracle handling: If the resolved name is present in the module-level trivial_oracle_names set/list (a set of oracles that do not require downloading), the function returns the resolved name immediately without creating directories or performing any network activity. This avoids unnecessary work for built-in or trivial oracles.
        3. Download path and URL construction: For non-trivial oracles, the function constructs a Dataverse download URL by concatenating the Dataverse server base ("https://dataverse.harvard.edu/api/access/datafile/") with oracle2id[resolved_name]. oracle2id is a module-level mapping from canonical oracle name to the remote file id. The function uses oracle2type[resolved_name] (another module-level mapping) to determine the expected filename extension/type for the checkpoint.
        4. Local directory creation: If the provided path does not exist, the function will create it using os.mkdir(path). This may raise an OSError on permission errors or if the parent directory does not exist; callers should ensure the parent directory is writable or create it beforehand.
        5. Local file detection and idempotence: Before downloading, the function checks for an existing local copy named <resolved_name>.<oracle2type[resolved_name]> inside the provided path. If that exact file already exists, the function prints a status message ("Found local copy...") and skips downloading, making the wrapper safe to call repeatedly without re-downloading.
        6. Downloading: If no local file is found, the function prints a status message ("Downloading Oracle...") and calls dataverse_download(dataset_path, path, resolved_name, oracle2type) to fetch and save the remote checkpoint into the given directory, then prints "Done!" on completion. dataverse_download is expected to perform the network transfer and save the file to path with the appropriate name and extension.
        7. Logging/prints: The function uses print_sys(...) to emit simple status messages; these are side effects visible to the user and intended to indicate progress.
    
    Failure modes and important notes:
        - Resolution failures: If fuzzy_search cannot resolve the given name to an entry in oracle_names, the behavior is determined by fuzzy_search (for example, it may raise an exception or return an unexpected value). Callers should validate input names or handle exceptions from fuzzy_search.
        - Missing mappings: The function depends on module-level mappings oracle2id and oracle2type for the resolved canonical name. If the resolved name is not present as a key in oracle2id or oracle2type, a KeyError will be raised.
        - Network and remote errors: dataverse_download (and the underlying network operations) may fail due to network timeouts, HTTP errors, authentication/permission issues on Dataverse, or remote-file-not-found errors. Such errors will propagate from dataverse_download unless handled by the caller or the callee.
        - Filesystem errors: os.mkdir and file writes performed by dataverse_download may raise OSError on permission denied, disk-full, or invalid path errors.
        - Concurrency: If multiple processes call this wrapper concurrently with the same path and name, race conditions may arise when checking for the existing file and creating the directory; callers should handle concurrency externally if necessary.
    
    Returns:
        str: The canonical (exact) oracle query name resolved from the provided rough name. In the TDC domain this is the exact identifier for a molecule-generation oracle (one of the entries in oracle_names) and is the base filename (without extension) of the saved model checkpoint when a download occurs. If the oracle is trivial (in trivial_oracle_names) the function returns immediately with this canonical name and performs no download or filesystem changes.
    
    \"\"\"
    """
    from tdc.utils.load import oracle_download_wrapper
    return oracle_download_wrapper(name, path, oracle_names)


################################################################################
# Source: tdc.utils.load.oracle_load
# File: tdc/utils/load.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_utils_load_oracle_load(
    name: str,
    path: str = "./oracle",
    oracle_names: list = ['drd2', 'gsk3b', 'jnk3', 'fpscores', 'cyp3a4_veith', 'drd2_current', 'gsk3b_current', 'jnk3_current', 'qed', 'logp', 'sa', 'rediscovery', 'similarity', 'median', 'isomers', 'mpo', 'hop', 'celecoxib_rediscovery', 'troglitazone_rediscovery', 'thiothixene_rediscovery', 'aripiprazole_similarity', 'albuterol_similarity', 'mestranol_similarity', 'isomers_c7h8n2o2', 'isomers_c9h10n2o2pf2cl', 'isomers_c11h24', 'osimertinib_mpo', 'fexofenadine_mpo', 'ranolazine_mpo', 'perindopril_mpo', 'amlodipine_mpo', 'sitagliptin_mpo', 'zaleplon_mpo', 'sitagliptin_mpo_prev', 'zaleplon_mpo_prev', 'median1', 'median2', 'valsartan_smarts', 'deco_hop', 'scaffold_hop', 'novelty', 'diversity', 'uniqueness', 'validity', 'fcd_distance', 'kl_divergence', 'askcos', 'ibm_rxn', 'isomer_meta', 'rediscovery_meta', 'similarity_meta', 'median_meta', 'docking_score', 'molecule_one_synthesis', 'pyscreener', 'rmsd', 'kabsch_rmsd', 'smina', '1iep_docking', '2rgp_docking', '3eml_docking', '3ny8_docking', '4rlu_docking', '4unn_docking', '5mo4_docking', '7l11_docking', 'drd3_docking', '3pbl_docking', '1iep_docking_normalize', '2rgp_docking_normalize', '3eml_docking_normalize', '3ny8_docking_normalize', '4rlu_docking_normalize', '4unn_docking_normalize', '5mo4_docking_normalize', '7l11_docking_normalize', 'drd3_docking_normalize', '3pbl_docking_normalize', '1iep_docking_vina', '2rgp_docking_vina', '3eml_docking_vina', '3ny8_docking_vina', '4rlu_docking_vina', '4unn_docking_vina', '5mo4_docking_vina', '7l11_docking_vina', 'drd3_docking_vina', '3pbl_docking_vina']
):
    """tdc.utils.load.oracle_load is a convenience wrapper that resolves a user-provided, possibly imprecise oracle name to an exact oracle identifier, downloads any required oracle assets to local storage, performs any required processing, and returns the canonical oracle name. In the Therapeutics Data Commons (TDC) workflow, oracles are functions or datasets used by molecule generation tasks (goal-oriented and distribution-learning), and this function centralizes retrieval and local caching of those oracles so downstream code (for example, Oracle(...) callers or molecule generation pipelines) can instantiate and use them reproducibly.
    
    This function attempts to match the rough oracle name supplied by the caller against a known collection of exact oracle names (the oracle_names list provided by the module). If a match is found, the underlying oracle_download_wrapper is invoked to download and/or prepare the oracle assets into the filesystem path specified by path. Side effects include creating the target directory (path) if it does not exist and writing downloaded oracle files into that location. The function returns the resolved exact oracle name (a str) so callers can programmatically verify which oracle was loaded and pass that canonical name to other TDC utilities (for example, Oracle instantiation or benchmark bookkeeping).
    
    Note on defaults: path defaults to "./oracle" (a relative directory in the current working directory), and oracle_names defaults to the module-level list of available exact oracle names exposed by tdc.utils.load. The provided oracle_names list contains many commonly used oracles in TDC (e.g., property scorers, rediscovery and similarity tasks, MPO and docking-related oracles) and is used to disambiguate user inputs such as partial names or common aliases.
    
    Failure modes and exceptions: if the provided name cannot be resolved to any exact oracle name in oracle_names, or if the underlying download/processing fails (for example due to network errors, permission errors, or missing external dependencies), the underlying oracle_download_wrapper may raise an exception which is propagated to the caller. Callers should handle exceptions (e.g., ValueError, OSError, or network-related exceptions) as appropriate for their application to recover or to report the error. The function does not itself suppress or convert such exceptions.
    
    Args:
        name (str): The rough oracle name provided by the caller. This is the user-facing identifier (possibly an alias or partial name) that the function will attempt to resolve to an exact oracle identifier present in oracle_names. In TDC, this typically corresponds to a molecule-generation or scoring oracle (for instance, names related to docking, MPO, rediscovery, or property scorers) used to evaluate or guide generated molecules.
        path (str): The local filesystem path where oracle assets should be saved or retrieved. Defaults to "./oracle". This directory will be created if it does not exist, and downloaded oracle files or processed artifacts will be written here as a side effect. The path is relative to the current working directory unless an absolute path is provided.
        oracle_names (list): A list of available exact oracle names that the function can resolve the rough name to. By default this is the module-level oracle_names list defined in tdc.utils.load and includes the canonical oracle identifiers supported by TDC (used to disambiguate input name and to determine which assets to download and prepare).
    
    Returns:
        str: The resolved exact oracle name. This canonical name identifies the specific oracle that was downloaded and prepared and can be used by downstream TDC utilities (for example, Oracle(...) instantiation, benchmark tracking, or logging). If resolution or download fails, an exception from the underlying oracle_download_wrapper is propagated instead of returning a value.
    """
    from tdc.utils.load import oracle_load
    return oracle_load(name, path, oracle_names)


################################################################################
# Source: tdc.utils.load.pd_load
# File: tdc/utils/load.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_utils_load_pd_load(name: str, path: str):
    """Load a local TDC dataset file into a pandas DataFrame for downstream model training,
    evaluation, and benchmark workflows in the Therapeutics Data Commons (TDC) suite.
    This function is used by TDC data loaders to materialize a saved dataset (by
    dataset name and local path) into an in-memory object that machine-learning
    workflows expect. It determines the file format from the internal mapping
    name2type and supports multiple file formats commonly used by TDC datasets.
    
    Args:
        name (str): The dataset identifier (dataset name) as used in TDC. This string
            is used to look up the file type in the module-level mapping name2type
            and to construct the on-disk filename (name + "." + extension) or folder
            layout (for zipped datasets). In the TDC context, `name` is the canonical
            dataset key that links to a stored dataset artifact (for example, the
            identifiers returned by TDC data loader helper functions).
        path (str): The local filesystem path where the dataset files are stored.
            The function constructs full file paths by joining this path with the
            dataset name and its resolved extension (e.g., os.path.join(path, name + ".csv")).
            Provide the directory that contains the dataset files downloaded or saved
            via TDC tools.
    
    Returns:
        pandas.DataFrame or anndata.AnnData or dict: The in-memory representation of
        the loaded dataset suitable for downstream use in TDC pipelines.
        For tab/tsv ("tab"), csv ("csv"), Excel ("xlsx"), pickle ("pkl"), zip-packaged
        pickles ("zip"), and PyTorch tensor files ("pth"), the function returns a
        pandas.DataFrame that contains the tabular dataset and is ready for model
        training, splitting, and evaluation. For single-cell data stored as an AnnData
        H5AD ("h5ad"), the function returns an anndata.AnnData object (commonly used
        in single-cell workflows and compatible with TDC single-cell tasks). For JSON
        ("json") files, the function will attempt to convert the JSON mapping of
        lists into a pandas.DataFrame (padding shorter lists with None to equalize
        column lengths); if that conversion is not possible (for example, values are
        heterogeneous lengths or not list-like), the original Python dict parsed
        from JSON is returned. Practically, pandas.DataFrame return values are used
        by TDC's data splitters, evaluators, and model-training utilities; anndata.AnnData
        is returned when the dataset represents single-cell modalities that downstream
        functions expect.
    
    Raises:
        ValueError: If the dataset's detected file type (from name2type[name]) is not
            one of the supported formats handled by this function. Supported types
            (as implemented) include "tab" (TSV), "csv", "xlsx", "pkl" (pickle),
            "zip" (folder containing a pickle), "h5ad" (AnnData), "json", and "pth"
            (PyTorch tensor file). The ValueError communicates that the file type is
            unsupported for loading.
        Exception: When loading a ".pth" PyTorch file, the function expects either a
            torch.Tensor or a dict of torch.Tensors. If a non-tensor object is found
            in that structure, the function raises an Exception with the message
            "encountered non-tensor" to signal unexpected serialized content.
        ImportError: If an optional dependency required for a format is missing (for
            example, importing "anndata" for "h5ad" or "torch" for "pth"), Python's
            ImportError will propagate; callers should ensure optional dependencies
            are installed when using those formats.
        SystemExit (via sys.exit): If pandas raises an EmptyDataError or an EOFError
            while reading the file, the function intercepts those exceptions and
            calls sys.exit(...) with a message that the TDC hosting on Harvard Dataverse
            may be under maintenance. This results in process termination with the
            provided message instead of a Python exception stack.
    
    Behavior and side effects:
        - The function resolves the file format using the global mapping name2type
          and constructs file paths with os.path.join(path, filename). For "zip"
          type it expects a folder named after the dataset and a "name.pkl" inside that folder.
        - For "tab" files, pandas.read_csv(..., sep="\t") is used. For "csv",
          pandas.read_csv is used. For "xlsx", pandas.read_excel is used. For "pkl",
          pandas.read_pickle is used. For "h5ad", anndata.read_h5ad is used and an
          anndata.AnnData object is returned immediately. For "json", the function
          uses json.load and will attempt to create a DataFrame by padding lists to
          the same length; if padding is not possible it returns the parsed dict.
          For "pth", torch.load is used and tensors are converted to numpy arrays
          then to a pandas.DataFrame; if a dict of tensors is provided, the dict
          entries are concatenated into a single DataFrame.
        - After successful loading into a pandas.DataFrame, the function attempts to
          remove duplicate rows via df.drop_duplicates(); failure of drop_duplicates
          (for example, if the returned object is not a DataFrame) is ignored.
        - The function may import optional packages at runtime ("anndata", "torch",
          "json") depending on the required format. It also prints simple progress
          messages when loading an AnnData object using the internal print_sys call.
        - No network access or downloads are performed by this function; it operates
          on files already present at the provided path. It is typically called after
          TDC dataset download routines or when a user has saved dataset artifacts
          locally.
    
    Failure modes and recommendations:
        - Ensure name exists as a key in the module-level name2type mapping and that
          the corresponding file exists at the constructed path; otherwise, a
          FileNotFoundError or KeyError will occur.
        - For "h5ad" and "pth" formats, install the optional dependencies (anndata,
          torch) to avoid ImportError.
        - For "pth" files, confirm that the saved object is a torch.Tensor or a dict
          of torch.Tensors; mixed or unexpected serialized structures will raise an
          Exception as noted above.
        - If pandas raises EmptyDataError or EOFError while reading a file, the
          function exits the process with an explanatory message about Harvard
          Dataverse maintenance; callers that embed TDC in larger applications should
          be prepared for this behavior or prevalidate the file contents.
        - The function performs minimal validation of data contents beyond type-based
          conversion; downstream TDC processors and evaluators should be used to
          validate schema, expected columns, and value ranges for specific tasks.
    """
    from tdc.utils.load import pd_load
    return pd_load(name, path)


################################################################################
# Source: tdc.utils.load.process_crossdock
# File: tdc/utils/load.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_utils_load_process_crossdock(
    path: str,
    name: str = "crossdock",
    return_pocket: bool = False,
    threshold: int = 15,
    remove_protein_Hs: bool = True,
    remove_ligand_Hs: bool = True,
    keep_het: bool = False
):
    """tdc.utils.load.process_crossdock processes the CrossDock protein–ligand docking benchmark and returns paired protein and ligand feature dictionaries suitable for downstream machine-learning workflows in therapeutics (e.g., pocket-based docking, binding-site modeling, and GNNs on protein–ligand complexes).
    
    This function reads a preprocessed CrossDock dataset directory (for example, the layout produced by Luo et al., 2021, which stores pocket PDBs and ligand SDFs and an index.pkl). It uses RDKit to parse ligand molecules and Biopandas to parse PDB files, optionally strips hydrogen atoms, and aggregates per-complex atomic coordinates and atom-type annotations. The returned structures are lightweight Python dictionaries where each entry corresponds to one successfully processed complex from the CrossDock index. The function swallows individual parsing errors (counts them) and prints a short processing summary. This processor is intended to prepare CrossDock data for ML evaluation and model training as provided by the TDC benchmarks.
    
    Args:
        path (str): Filesystem path to the dataset root directory into which the CrossDock data were unpacked. The function will join this path with the dataset name and expects an index file at <path>/<name>/index.pkl and a data subdirectory named "crossdocked_pocket10". This argument determines where the function reads PDB/SDF files and the index used to enumerate protein–ligand pairs; it does not create new files except for normal Python file I/O while reading.
        name (str): Dataset folder name under path that contains CrossDock files; default is "crossdock". The function will construct the full data directory as os.path.join(path, name) and then expect index.pkl and a "crossdocked_pocket10" subfolder there. This parameter controls which dataset variant to read when multiple datasets are colocated under the same root path.
        return_pocket (bool): If True, the function will read pocket PDB files referenced in the index (recommended for pocket-centric analyses). If False, the function attempts to read the available PDB entry referenced by the index; note that for the preprocessed CrossDock release used here the full protein structure may not be stored and a pocket file is typically read regardless. This flag therefore signals the intent to use pocket coordinates only versus full protein coordinates; in practice the function reads the file referenced in the index.
        threshold (int): Radius in angstroms used to construct a spherical pocket around a ligand center when explicit pocket PDB files are not available in the raw dataset. This parameter is only applicable when pocket extraction from raw data must be synthesized (i.e., when pockets are not provided) and controls the spatial cutoff for selecting protein atoms near the ligand. Default is 15. If pockets are present (as in many CrossDock releases), this parameter is not applied.
        remove_protein_Hs (bool): Whether to remove hydrogen atoms from parsed protein coordinates (True by default). Removing protein hydrogens reduces feature size and is a common preprocessing step for ML tasks where explicit hydrogens are unnecessary or inconsistently annotated; set to False to retain H atoms when your downstream model requires them.
        remove_ligand_Hs (bool): Whether to remove hydrogen atoms from parsed ligand molecules (True by default). This flag is passed to the ligand extraction routine; keeping ligand hydrogens may be necessary for physicochemical calculations but can increase parser failures if input SDFs are inconsistent.
        keep_het (bool): Whether to keep HETATM records (e.g., cofactors, metal ions, water) from the protein PDB parsing (False by default). When True, HETATM lines are included in the protein atom list and may influence pocket composition and downstream descriptors; when False, these nonstandard residues are omitted.
    
    Behavior and side effects:
        The function expects a pickled index at <path>/<name>/index.pkl that yields tuples of (pocket_filename, ligand_filename, ..., rmsd). It will iterate over the index, skip entries where either pocket or ligand filename is None, and attempt to read PDB and SDF files under the "crossdocked_pocket10" subdirectory. RDKit logging is programmatically disabled at import to suppress parser messages. Ligands are read with RDKit.Chem.SDMolSupplier using sanitize=False to tolerate imperfect SDFs; the subsequent ligand extraction routine may still return None for malformed molecules, and such entries are skipped. Protein coordinates are parsed via biopandas PandasPdb and passed to the protein extraction routine together with the remove_protein_Hs and keep_het flags. The processor accumulates per-complex coordinate sets and atom-type annotations into Python lists. The function prints a short summary of the number of failures versus total index entries using the code's print_sys call. The function does not write output files; it returns in-memory dictionaries for immediate use. Default values for optional arguments are set to conservative preprocessing choices used in TDC benchmarks.
    
    Failure modes and robustness:
        Individual file parsing failures, malformed SDF/PDB records, missing files referenced by the index, and failures inside extraction helper functions are caught and counted; the function continues processing remaining entries. The summary printed at the end reports how many entries failed out of the total considered. Because exceptions are broadly caught, callers should inspect the printed summary and the sizes of the returned lists to detect substantial data loss. If the index.pkl file or expected subdirectories are missing, the function will raise a FileNotFoundError or the unpickling error raised by pickle.load. The function relies on RDKit and biopandas being importable and available in the runtime environment; ImportError will be raised if these dependencies are missing.
    
    Returns:
        tuple: Two dictionaries (protein, ligand). Each dictionary has the keys "coord" and "atom_type". The values are lists aligned by successfully processed complexes: "coord" contains per-complex coordinate collections extracted from PDB/SDF files (protein pockets or full proteins, and ligand atom coordinates, respectively), and "atom_type" contains corresponding per-complex atom-type annotations for those coordinates. These dictionaries are suitable for downstream ML data loaders and evaluation pipelines in TDC; callers should verify the list lengths and handle any complexes omitted due to parsing failures.
    """
    from tdc.utils.load import process_crossdock
    return process_crossdock(
        path,
        name,
        return_pocket,
        threshold,
        remove_protein_Hs,
        remove_ligand_Hs,
        keep_het
    )


################################################################################
# Source: tdc.utils.load.process_pdbbind
# File: tdc/utils/load.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_utils_load_process_pdbbind(
    path: str,
    name: str = "pdbbind",
    return_pocket: bool = False,
    threshold: int = 15,
    remove_protein_Hs: bool = True,
    remove_ligand_Hs: bool = True,
    keep_het: bool = False
):
    """tdc.utils.load.process_pdbbind processes a local PDBBind-style dataset directory and extracts per-complex protein and ligand atomic features suitable for downstream machine learning tasks in the Therapeutics Data Commons (TDC) ecosystem.
    
    This function expects a directory structure where each complex is stored in a subdirectory named by its PDBBind identifier and contains protein PDB files and ligand SDF files following the naming conventions used in PDBBind (e.g., "{id}/{id}_protein.pdb", "{id}/{id}_pocket.pdb", "{id}/{id}_ligand.sdf"). It reads protein coordinates using biopandas.PandasPdb and ligand molecules using RDKit, then delegates atom extraction and optional hydrogen removal to helper functions (extract_atom_from_protein and extract_atom_from_mol). The outputs are two dictionaries (protein and ligand) that collect per-complex coordinate arrays and atom-type lists; these are intended for use in TDC workflows that require numeric atomic coordinates and atom-type labels for model input, dataset split creation, and evaluation.
    
    Behavior and notable implementation details:
    - The function iterates over files in the given path using tqdm for progress reporting and ignores files named "readme" or "index".
    - If return_pocket is True, the function attempts to read the pocket PDB file named "{id}/{id}_pocket.pdb"; otherwise it reads "{id}/{id}_protein.pdb".
    - The threshold parameter is provided to control a pocket radius in the conceptual API (i.e., radius around ligand center to define a pocket) when pockets must be computed from full proteins, but in the current implementation threshold is not used; pocket selection is enacted only by reading precomputed pocket PDB files when return_pocket is True.
    - remove_protein_Hs and remove_ligand_Hs control whether hydrogen atoms are removed from the parsed protein or ligand prior to assembling coordinates and types; removal is implemented by the helper extraction functions.
    - keep_het controls whether HETATM records from the PDB (commonly cofactors or nonstandard residues) are included when extracting protein atoms.
    - Ligands are read from SDF files via RDKit with sanitize=False; if the helper extract_atom_from_mol deems a ligand contains unallowed atoms it may return None and that complex will be skipped.
    - The function suppresses exceptions raised while processing individual complexes (bare except), counts failures, and continues processing other files; after processing it prints the number of failed files.
    - If the provided path does not exist, the function terminates the process by calling sys.exit("Wrong path!").
    - The function disables RDKit logging (RDLogger.DisableLog("rdApp.*")) and imports biopandas at runtime; missing dependencies (RDKit, biopandas) will raise ImportError at call time.
    - The function prints a "Processing..." message at start and a summary "processing done, {failure}/{total_ct} fails" at the end via print_sys; these side-effecting prints are intended to inform users during long-running preprocessing.
    
    Args:
        path (str): Filesystem path to the directory that contains the PDBBind-style dataset. This directory is expected to contain one subdirectory per complex named by its identifier; each subdirectory should hold protein PDB files and ligand SDF files following the PDBBind naming convention. If the path does not exist, the function calls sys.exit("Wrong path!"), terminating the process.
        name (str): Logical name for the dataset (default: "pdbbind"). This parameter is informational in the current implementation and does not affect file discovery; it is retained to match the TDC API patterns where dataset name can be used by callers to label processing runs and downstream artifacts.
        return_pocket (bool): When True, the function attempts to read and return only the protein pocket PDB file named "{id}/{id}_pocket.pdb" for each complex; when False, it reads the full protein file "{id}/{id}_protein.pdb". This flag controls which file the function tries to load; if a pocket file is absent and only full protein files exist, set return_pocket to False or provide pocket files beforehand.
        threshold (int): Radius (in Angstroms) intended to define a pocket around the ligand center if pockets must be computed from full proteins (i.e., conceptual radius to extract atoms within threshold). In the current implementation this parameter is accepted but not used — pocket selection occurs only by reading precomputed pocket PDB files when return_pocket is True.
        remove_protein_Hs (bool): If True (default), hydrogen atoms are removed from the protein atom set during extraction via extract_atom_from_protein; if False, hydrogens from protein ATOM/HETATM records may be preserved depending on the helper function's behavior.
        remove_ligand_Hs (bool): If True (default), hydrogen atoms are removed from the ligand during extraction via extract_atom_from_mol; if the helper determines the ligand becomes invalid after removal or contains unallowed atoms, that complex will be skipped.
        keep_het (bool): If False (default), HETATM records in the protein PDB are excluded when extracting protein atoms; if True, HETATM records (common for cofactors or nonstandard residues) are included in the protein feature extraction.
    
    Returns:
        tuple: A pair (protein, ligand) where each is a dict collecting per-complex features for successful parses. The dictionaries follow the structure produced by the implementation:
        protein (dict): Contains keys "coord" and "atom_type". "coord" is a list (one entry per successfully processed complex) of per-complex coordinate arrays (each an array-like sequence of per-atom 3D coordinates as returned by the helper extraction). "atom_type" is a list of per-complex atom-type sequences (e.g., element symbols or internal atom-type labels) aligned with the coordinates. These features are intended for downstream ML pipelines that require atomic coordinates and atom types from protein structures.
        ligand (dict): Contains keys "coord" and "atom_type" analogous to the protein dict but for ligand molecules parsed from SDF files. Entries are only included for complexes where ligand parsing and validation (via extract_atom_from_mol) succeed.
        If many files are present, the returned lists will contain entries only for successfully processed complexes; failed complexes are skipped and counted in the printed summary. The function has side effects (printing progress and summary, disabling RDKit logging) and may call sys.exit if the provided path is invalid.
    """
    from tdc.utils.load import process_pdbbind
    return process_pdbbind(
        path,
        name,
        return_pocket,
        threshold,
        remove_protein_Hs,
        remove_ligand_Hs,
        keep_het
    )


################################################################################
# Source: tdc.utils.load.process_scpdb
# File: tdc/utils/load.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_utils_load_process_scpdb(
    path: str,
    name: str = "scPDB",
    return_pocket: bool = False,
    threshold: int = 15,
    remove_protein_Hs: bool = True,
    remove_ligand_Hs: bool = True,
    keep_het: bool = False
):
    """tdc.utils.load.process_scpdb processes the scPDB protein–ligand structural dataset into two Python dictionaries of parsed features suitable for downstream machine-learning workflows in TDC (for example, creating per-complex coordinate and atom-type inputs for binding-site or pocket modeling).
    
    This function traverses a directory of scPDB entries, reads per-entry protein MOL2 files and ligand SDF files, extracts atom coordinates and atom types, and aggregates those features across all successfully parsed entries. It is intended for preparatory data processing in therapeutic-ML tasks (e.g., single-instance prediction and pocket-level analyses) that use structural protein–ligand complexes from the scPDB resource. The implementation expects the scPDB data to be organized under path/name with one subdirectory per complex containing either site.mol2 (when return_pocket=True) or protein.mol2 (when return_pocket=False) and ligand.sdf. The function uses RDKit and biopandas.mol2 to read molecules and relies on helper functions extract_atom_from_mol and extract_atom_from_protein to produce coordinate and atom-type lists.
    
    Args:
        path (str): Filesystem path to the parent directory where the scPDB dataset folder is or will be located. The function will join this path with the parameter name to locate dataset entries; for example, if path is "/data" and name is "scPDB", the function will list files under "/data/scPDB". This parameter directs where the function looks for per-complex subfolders and is required for locating site.mol2/protein.mol2 and ligand.sdf files.
        name (str): The name of the dataset subdirectory under path that contains scPDB entries. Default "scPDB". The function will call os.path.join(path, name) to form the dataset directory. This value defines which folder is scanned for complex subdirectories to process.
        return_pocket (bool): If True, the function attempts to read a pocket MOL2 file named site.mol2 for each complex and treat those coordinates as the protein pocket; if False (default) it reads the full protein file named protein.mol2. Practically, set True when you wish to extract only the annotated binding-site region rather than the entire protein chain. The code reads site.mol2 when this flag is True and protein.mol2 when False.
        threshold (int): Intended radius (in the same coordinate units as the input files) to use to define a spherical pocket around a ligand center when a raw pocket file is not available. Default 15. Note: in the current implementation this parameter is documented but not applied; the function does not programmatically compute pockets from a radius and instead relies on the presence of site.mol2 when return_pocket is True.
        remove_protein_Hs (bool): Whether to remove hydrogen atoms from protein entries (True by default). Removing protein hydrogens reduces the number of atoms and is a common preprocessing step for structural feature extraction; when True, extract_atom_from_protein is intended to omit H atoms. Note: due to a hardcoded call in the current implementation, heteroatom handling may be affected (see keep_het).
        remove_ligand_Hs (bool): Whether to remove hydrogen atoms from ligand entries (True by default). When True, the function calls extract_atom_from_mol with remove_Hs=True to avoid including ligand H atoms in the returned coordinate and atom-type lists; this is a common normalization for ligand-based features.
        keep_het (bool): Whether to keep heteroatoms (HETATM records, e.g., cofactors, metal ions) from protein MOL2 files. Default False. Intended semantic: set True to preserve non-standard residues and cofactors in protein features. Important implementation note: in the current code path extract_atom_from_protein is called with keep_het=False regardless of this argument, so keep_het is accepted by the signature but is not honored by the implementation. Users requiring heteroatoms preserved must update the helper call or post-process the protein files.
    
    Behavior and side effects:
        The function lists all entries under os.path.join(path, name) and iterates over them with a tqdm progress indicator. For each entry it attempts to read either site.mol2 (if return_pocket is True) or protein.mol2 (if return_pocket is False) using biopandas.mol2.PandasMol2.read_mol2, and reads ligand.sdf using RDKit Chem.SDMolSupplier with sanitize=False. It then calls extract_atom_from_mol to extract ligand coordinates and atom types and extract_atom_from_protein to extract protein coordinates and atom types. If extract_atom_from_mol returns None (for example, because the ligand contains unallowed atom types), that complex is skipped. Any exception raised while processing an entry is caught, the entry is counted as a failure, and processing continues with subsequent entries. The function prints a short summary via print_sys indicating how many files failed versus the total examined. The function imports and disables RDKit logging (RDLogger.DisableLog) and therefore may suppress RDKit warnings as a side effect.
    
    Failure modes and limits:
        - Missing files: if site.mol2/protein.mol2 or ligand.sdf are absent or malformed for a given entry, that entry will trigger an exception, be counted as a failure, and be skipped; processing continues for other entries.
        - Unallowed ligand atoms: extract_atom_from_mol may return None for ligands with unsupported atom types; such complexes are skipped without raising an exception.
        - Parameter mismatches: threshold and keep_het are accepted but their intended effects (radius-based pocket extraction and retaining heteroatoms) are currently not implemented/used in the code path; users should not rely on those parameters unless the helper calls are updated.
        - The function aggregates results in memory; for very large scPDB subsets this may consume substantial RAM because it collects coordinate and atom-type lists for all successfully parsed entries.
    
    Returns:
        protein (dict): A dictionary of aggregated protein features with two keys: "coord" and "atom_type". Each value is a list whose entries correspond, in order of successful processing, to the protein (or pocket) coordinates and atom-type arrays/lists for a single complex. These per-complex entries are produced by extract_atom_from_protein and reflect the processing options (e.g., removal of protein hydrogens as requested by remove_protein_Hs). If many entries fail, returned lists will contain only the successfully parsed complexes.
        ligand (dict): A dictionary of aggregated ligand features with two keys: "coord" and "atom_type". Each value is a list whose entries correspond, in order of successful processing, to per-complex ligand coordinates and atom types as returned by extract_atom_from_mol. Complexes skipped due to unallowed atoms or exceptions do not contribute entries to these lists.
    
    Note:
        The returned dictionaries are intended as lightweight, ML-ready feature containers for downstream steps in TDC pipelines (for example, converting coordinates and atom types into graph inputs or pocket-level features). Because some documented parameters (threshold, keep_het) are not applied in the current implementation, users needing those behaviors should modify the helper calls or precompute pockets prior to calling this function. The function prints a processing summary and uses progress display tools (tqdm), so running it in non-interactive environments will still produce standard output.
    """
    from tdc.utils.load import process_scpdb
    return process_scpdb(
        path,
        name,
        return_pocket,
        threshold,
        remove_protein_Hs,
        remove_ligand_Hs,
        keep_het
    )


################################################################################
# Source: tdc.utils.load.property_dataset_load
# File: tdc/utils/load.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_utils_load_property_dataset_load(
    name: str,
    path: str,
    target: str,
    dataset_names: list
):
    """tdc.utils.load.property_dataset_load loads a single-instance prediction dataset for Therapeutics Data Commons (TDC), performing download (via download_wrapper), column alignment, duplicate-column removal, and filtering so the caller receives three pandas.Series objects suitable for machine learning workflows (entity representation X, target labels y, and unique entity IDs). This function is used across TDC single-instance prediction tasks (for example ADME or activity screening datasets) to obtain the canonical input representation (e.g., molecule SMILES or sequence strings), the label of interest for supervised learning, and an identifier for each entity for traceability and leaderboard submission.
    
    Args:
        name (str): the rough dataset name provided by the caller. This value is passed to download_wrapper which resolves and/or downloads the canonical dataset name from TDC hosting (for example Harvard Dataverse). After download_wrapper returns, name is set to the exact dataset identifier used to read files from path.
        path (str): the filesystem path or directory where datasets are saved or retrieved. The function reads files from and may create files under this directory when download_wrapper runs. Callers should ensure this path is writable (for downloads) and readable.
        target (str): for multi-label datasets, the label column name (or a rough label string) indicating which label/column to return. The function uses fuzzy_search to align this requested label to an actual column name in the loaded table. If target is None on entry, the function sets target = "Y" (the default column name used by many TDC datasets) before attempting alignment; this behavior implements a conservative default for single-label datasets.
        dataset_names (list): a list of available exact dataset names known to the loader. This list is provided to download_wrapper and used to validate/resolve the requested rough name into the exact dataset name that exists in TDC hosting. Elements are the exact dataset identifiers (strings) that download_wrapper recognizes.
    
    Returns:
        pandas.Series: a tuple of three pandas.Series objects: (X_series, y_series, ID_series). X_series contains the entity representation used as model input (for example SMILES strings for small-molecule datasets or sequence strings for biologics), y_series contains the target property values corresponding to the aligned target column (the label used for supervised learning), and ID_series contains a unique identifier for each entity (used for result tracking and leaderboard submission). The function attempts to return df["X"], df[target], df["ID"]; if those columns are not present it falls back to df["Drug"], df[target], df["Drug_ID"]. All returned series are filtered so rows with null target values are removed and the index is reset, ensuring the three series are aligned by index for downstream training or evaluation.
    
    Behavior and side effects:
        The function resolves and possibly downloads the dataset via download_wrapper(name, path, dataset_names) and prints a loading message (print_sys("Loading...")). It loads the dataset file via pd_load(name, path). It uses fuzzy_search to map the requested target to an actual dataframe column name when possible. Duplicate columns are removed via df.loc[:, ~df.columns.duplicated()] to avoid ambiguity. Rows where the target column is null are removed (df = df[df[target].notnull()].reset_index(drop=True)). If the expected columns and alignment code raise an exception while processing the loaded dataframe, the function attempts to open the raw dataset file on disk and inspects its content. If the raw content contains the string "Service Unavailable", it calls sys.exit with a user-facing message indicating that Harvard Dataverse hosting is under maintenance. For any other unexpected error while reading or processing, the function calls sys.exit with a request to report the error to contact@tdcommons.ai. The function therefore may terminate the Python process via sys.exit on certain failure modes; callers embedding this function in long-running services should be aware of this behavior.
    
    Failure modes and diagnostics:
        If download_wrapper cannot resolve or download the dataset, the function may raise an exception from download_wrapper or pd_load. If fuzzy_search fails to map the requested target to a dataframe column, the code will raise and trigger the fallback inspection of the raw file; this can result in sys.exit with either the "Service Unavailable" message or a request to contact contact@tdcommons.ai. If the dataset uses nonstandard column names, the function attempts the primary ("X", "ID") and fallback ("Drug", "Drug_ID") column patterns; if neither pattern matches the file, a KeyError will be raised and the fallback error-handling path will execute. The function prints minimal progress messages but does not return intermediate diagnostic objects; callers should catch exceptions or inspect stdout/stderr for the sys.exit messages when debugging.
    """
    from tdc.utils.load import property_dataset_load
    return property_dataset_load(name, path, target, dataset_names)


################################################################################
# Source: tdc.utils.load.receptor_download_wrapper
# File: tdc/utils/load.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_utils_load_receptor_download_wrapper(name: str, path: str):
    """receptor_download_wrapper(name, path)
    Wrapper that ensures a receptor structure (PDB and PDBQT) is available locally for TDC docking/oracle workflows by downloading files from the Harvard Dataverse when needed.
    
    This function is used in Therapeutics Data Commons (TDC) workflows that require receptor structures (for example, docking or molecule-generation oracle evaluation). Given an exact PDB identifier (pdbid) and a target filesystem path, the function checks for existing local copies of both the PDB and PDBQT files, creates the target directory if missing, and otherwise downloads the two files from a configured Harvard Dataverse endpoint. The function uses an internal mapping receptor2id to look up dataverse file identifiers and calls dataverse_download to fetch files; it prints progress messages via print_sys. On success the function returns the same pdbid string that was provided, allowing callers to confirm which receptor was prepared.
    
    Args:
        name (str): The exact PDB identifier (pdbid) of the receptor to prepare. This is used as the key into the internal receptor2id mapping to build dataverse file URLs and as the base filename for the saved receptor files (name + ".pdb" and name + ".pdbqt"). In TDC, this identifier links a benchmark or oracle to a specific receptor structure required for docking or related evaluation tasks.
        path (str): Filesystem directory path where the receptor files will be stored. If the directory does not exist, the function will attempt to create it. The function writes two files into this directory: "{path}/{name}.pdb" and "{path}/{name}.pdbqt", which are the standard PDB and PDBQT receptor formats used by downstream docking/oracle components in TDC.
    
    Returns:
        str: The exact pdbid provided in the name argument. Returning the same string signals to callers (for example, data loaders or oracle setup code) which receptor was found or downloaded and can be used to reference the saved files.
    
    Behavior and side effects:
        The function constructs two dataverse download URLs from an internal mapping receptor2id[name] and a hard-coded Dataverse API base URL. If both target files already exist in the given path, the function prints a "Found local copy..." message via print_sys and does not perform any network operations. If one or both files are missing, the function prints "Downloading receptor...", invokes dataverse_download for each required file (first for PDBQT then for PDB), and prints "Done!" when finished. The function creates the directory specified by path with os.mkdir if it does not exist.
    
    Failure modes and exceptions:
        If receptor2id does not contain the provided name, a KeyError (or an IndexError if the mapped value lacks expected elements) will occur when building dataset_paths. Filesystem operations can raise exceptions: os.mkdir may raise FileExistsError if path exists but is not a directory, and permission errors (PermissionError or OSError) can occur when creating directories or writing files. Network or download failures (raised by dataverse_download or underlying network libraries) will propagate to the caller; partial downloads may leave incomplete files in the target directory. Callers should handle these exceptions and verify file integrity if needed.
    
    Notes and practical significance:
        The function centralizes the logic for locating and retrieving receptor structures required by TDC benchmarks and oracles, reducing duplication across data loaders and ensuring consistent naming and storage of receptor files. Callers should provide an exact pdbid that matches entries in the package's receptor2id mapping and ensure the running environment has write permissions to the specified path and network access to the Harvard Dataverse API.
    """
    from tdc.utils.load import receptor_download_wrapper
    return receptor_download_wrapper(name, path)


################################################################################
# Source: tdc.utils.load.receptor_load
# File: tdc/utils/load.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_utils_load_receptor_load(name: str, path: str = "./oracle"):
    """tdc.utils.load.receptor_load downloads, processes, and loads a receptor PDB file given a rough PDB identifier. It is a thin wrapper around receptor_download_wrapper that resolves a user-provided rough pdb name to the canonical PDB identifier used by TDC, saves the processed receptor file(s) under the specified oracle directory, and returns the exact resolved PDB identifier. In the TDC (Therapeutics Data Commons) workflow, this function is used when preparing receptor structures for downstream tasks such as molecule-generation oracles, docking, or structure-based scoring that require canonical receptor PDB files stored in the project's oracle directory.
    
    Args:
        name (str): The rough PDB identifier or name provided by the caller. This is the user-facing identifier (for example, a case-insensitive or partial pdb code) that receptor_download_wrapper will resolve, normalize, and map to the exact canonical PDB identifier used by TDC. The practical role of this argument is to allow users to specify a receptor of interest with minimal formatting requirements; the function will interpret and translate this into the precise identifier that corresponds to the saved and processed receptor files.
        path (str): The filesystem path to the oracle directory where downloaded and processed receptor files are saved or retrieved. Defaults to "./oracle". This path is used for caching and organizing receptor data for TDC oracles and related molecular tasks. The function may create the directory if it does not exist and will write processed receptor files under this path as a side effect.
    
    Returns:
        str: The exact PDB identifier corresponding to the downloaded/processed receptor. This returned string is the canonical name that other TDC components and oracles should use to reference the saved receptor file(s). The return value indicates successful resolution and saving of the receptor; callers can combine it with the provided path to locate the stored file(s).
    
    Behavior, side effects, defaults, and failure modes:
        This function delegates the heavy lifting to receptor_download_wrapper(name, path). Side effects include network access to download PDB data (if not already cached), creation of directories under path (if necessary), and writing of processed receptor file(s) to disk for later reuse by TDC oracles and tasks. The default path is "./oracle", which provides a simple local cache location for TDC users; callers should supply an absolute or project-specific path if they require a different storage location.
        On success, the exact PDB identifier is returned and the corresponding receptor files are available under the specified path for downstream TDC functions (e.g., oracles, docking pipelines, or dataset preparation). On failure, exceptions raised by receptor_download_wrapper (for example, network errors, permission or filesystem errors, or invalid/unresolvable identifiers) propagate to the caller. Callers should handle these exceptions or validate inputs before calling this function when operating in automated pipelines.
    """
    from tdc.utils.load import receptor_load
    return receptor_load(name, path)


################################################################################
# Source: tdc.utils.load.three_dim_dataset_load
# File: tdc/utils/load.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_utils_load_three_dim_dataset_load(name: str, path: str, dataset_names: list):
    """three_dim_dataset_load downloads (if necessary), processes, and loads a 3D molecular dataset for TDC 3D-molecule tasks and returns the loaded data plus the resolved dataset path and exact dataset name. This function is used within the Therapeutics Data Commons (TDC) ecosystem to obtain TDC benchmark datasets that contain three-dimensional molecular information needed for structure-aware machine learning workflows (for example, tasks that require atomic coordinates, conformer information, or structure-based feature extraction). Internally, the function calls zip_data_download_wrapper to select and/or download the exact dataset from the available dataset_names into the provided path, prints a loading message, and then uses pd_load to read the stored dataset into a pandas.DataFrame.
    
    Args:
        name (str): A rough dataset identifier supplied by the caller. In TDC usage this is typically a task-level or shorthand dataset name; the function forwards this identifier to zip_data_download_wrapper, which selects and returns the exact dataset name that will be downloaded or loaded. The returned/resolved name may differ from this input string when a more specific dataset name is chosen by the wrapper.
        path (str): Filesystem directory path where dataset files will be saved and/or retrieved. The function will use this path when calling the download and load helpers and returns the joined path together with the resolved dataset name. Typical uses in TDC place datasets under a local data directory so that subsequent calls can reuse already-downloaded files.
        dataset_names (list): A list of available exact dataset names that the download wrapper can select from. Each element should represent an available 3D dataset identifier known to the TDC data backend. This list is used by zip_data_download_wrapper to validate or pick the exact dataset to download/load.
    
    Returns:
        pandas.DataFrame: The loaded pandas DataFrame that holds the 3D molecular information for the selected dataset. This DataFrame is intended for downstream structure-aware machine learning tasks in TDC and typically contains columns representing molecular identifiers and 3D-relevant data (for example, coordinates and per-atom annotations) as provided by the dataset. The function returns the in-memory DataFrame produced by pd_load; callers can directly use it for splitting, preprocessing, or feeding into model pipelines.
        str: The filesystem path (os.path.join(path, name)) where the selected dataset was saved or from which it was loaded. This is the canonical location under the provided path corresponding to the resolved exact dataset name and can be used to inspect raw files or to re-load data by other utilities.
        str: The exact dataset name resolved and used by the function (the value returned by zip_data_download_wrapper). This value identifies the precise 3D dataset selected from the provided dataset_names list and may differ from the original input 'name' if the wrapper disambiguates or maps the rough name to a concrete dataset identifier.
    
    Behavior and side effects:
        - Calls zip_data_download_wrapper(name, path, dataset_names) which is responsible for selecting and, if necessary, downloading and extracting the requested dataset into path. The resolved dataset name returned by that helper replaces the input name variable.
        - Prints a "Loading..." message via print_sys to indicate progress to the user.
        - Calls pd_load(resolved_name, path) to load the dataset into a pandas.DataFrame and returns this object.
        - Writes files under the provided path as part of the download/extract process; existing files at that path may be read or overwritten depending on the helper implementation.
    
    Failure modes and exceptions:
        - Network or remote-server failures during download, insufficient disk space, or filesystem permission errors can cause the download helper or pd_load to raise exceptions (e.g., requests/IO-related errors). Callers should handle such exceptions as appropriate for their environment.
        - If the provided name cannot be resolved to an available dataset by zip_data_download_wrapper (for example, not listed in dataset_names), the wrapper may raise an error or return an unexpected value; callers should validate dataset_names and catch errors from the wrapper.
        - If pd_load cannot parse or read the downloaded files (corrupt archive, missing files, incompatible format), it will raise an exception that propagates to the caller.
    
    Notes for TDC users:
        - This function is intended to be a convenience wrapper used by TDC data loaders and tutorials to obtain 3D molecular benchmark datasets with minimal code. After calling, users receive the DataFrame ready for typical TDC downstream operations such as get_split, Evaluator usage, or conversion to graph formats for structure-based models.
    """
    from tdc.utils.load import three_dim_dataset_load
    return three_dim_dataset_load(name, path, dataset_names)


################################################################################
# Source: tdc.utils.load.zip_data_download_wrapper
# File: tdc/utils/load.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_utils_load_zip_data_download_wrapper(name: str, path: str, dataset_names: list):
    """tdc.utils.load.zip_data_download_wrapper downloads and unpacks a TDC dataset archive from the Harvard Dataverse given a fuzzy dataset query name, saving and extracting the dataset into a local directory structure used by TDC data loaders.
    
    Args:
        name (str): The rough dataset query name provided by the caller (for example, a user-specified dataset identifier used in higher-level TDC dataset loaders). This value is first resolved by fuzzy_search against dataset_names to determine the exact dataset key used by the dataverse mappings (name2idlist / name2id). The resolved exact name is the dataset identifier that the function returns and that downstream TDC code (data loaders, splitters) expects to find on disk.
        path (str): The filesystem directory path where downloaded zip files and extracted dataset folders will be written. The function will create this directory if it does not exist and will create subdirectories of the form {path}/{name} and {path}/{name}-{i+1} when a dataset consists of multiple files. This path is a direct side-effect target; files and folders are created, zip archives are written under this path, and extracted contents are moved into the final dataset folder.
        dataset_names (list): A list of available dataset names that fuzzy_search uses to match the provided rough name to an exact dataset key. In the TDC context, this list typically comes from the collection of dataset identifiers supported by the TDC data loader registry. The function does not modify this list; it only uses it as the candidate space for fuzzy matching to determine which dataset to download.
    
    Returns:
        str: The exact dataset query name resolved by fuzzy_search and used for download. This returned string is the canonical dataset key (one of the entries in the TDC dataset registry) and is identical to the folder name created under path (after downloads and extraction). The return value enables calling code to know which exact dataset was downloaded and where to find its extracted files.
    
    Behavior and side effects:
        This function implements the following concrete behavior relevant to TDC dataset retrieval workflows. It resolves the caller-provided rough name to an exact dataset key by calling fuzzy_search(name, dataset_names). It then constructs download URLs against the Harvard Dataverse API base URL "https://dataverse.harvard.edu/api/access/datafile/" and uses global mappings in the module (name2idlist, name2id, and name2type) to determine one or more dataverse file IDs to fetch. If the resolved exact name appears as a key in name2idlist, the function treats the dataset as composed of multiple files and iterates over name2idlist[name], downloading each file into path and naming temporary zip files as {name}-{i+1}.zip. If the resolved name is not in name2idlist, the function falls back to name2id[name] and downloads a single zip named {name}.zip. Downloads are performed via the module-level helper dataverse_download; for multi-file datasets the helper is called with an id argument (i + 1) to indicate file index. After successful download, each zip is extracted using Python's ZipFile.extractall into the working path. For multi-file datasets the function then consolidates extracted contents by moving files from {path}/{name}-{i+1} into {path}/{name} using a shell mv command (os.system). Throughout the process the function emits progress messages via print_sys. If local copies already exist (checked by os.path.exists on target directories), the function skips re-downloading and extraction for those parts.
    
    Failure modes and important notes for users:
        Network failures, HTTP errors, dataverse API changes, or dataverse_download failures will raise exceptions originating from the underlying network or the dataverse_download helper; these are not swallowed by the wrapper. If the resolved name is not present in either name2idlist or name2id, a KeyError or NameError will be raised due to reliance on these module-level mappings. Insufficient disk space, file permissions, or inability to create directories under path will raise OSError (or subclasses) when os.mkdir or file writes are attempted. The function uses os.system with a POSIX mv command and shell redirection ("2>/dev/null") to consolidate files; this may not behave as intended on non-POSIX systems (for example, native Windows shells) and may hide move errors, so callers on such platforms should verify that files were moved correctly. Zip extraction relies on Python's ZipFile and will raise BadZipFile or related exceptions on corrupt archives. Because the function prints status messages via print_sys and writes files to disk, it is not pure and is intended to be used as a side-effecting utility in TDC dataset preparation pipelines (for example, by the high-level dataset constructors described in the TDC README).
    
    Usage significance in TDC:
        This wrapper is a low-level utility used by TDC data loaders to programmatically obtain benchmark datasets from a central dataverse repository. It enables reproducible retrieval of TDC datasets by creating a consistent on-disk layout ({path}/{name} with all dataset files extracted) that the rest of the TDC library (data splitters, evaluators, and tutorial code) expects. Calling code typically provides a user-friendly rough dataset name and a local storage path and relies on this function to resolve, download, and prepare the dataset for subsequent machine learning tasks in therapeutic research.
    """
    from tdc.utils.load import zip_data_download_wrapper
    return zip_data_download_wrapper(name, path, dataset_names)


################################################################################
# Source: tdc.utils.misc.fuzzy_search
# File: tdc/utils/misc.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_utils_misc_fuzzy_search(name: str, dataset_names: str):
    """Fuzzy matching between a user-provided dataset name and the canonical dataset name used by the TDC (Therapeutics Data Commons) library. This function is used inside TDC data-loading and utility workflows to resolve minor differences in user input (case differences, an optional "tdc." prefix, or small typographical variations) to the exact dataset identifier that TDC expects when retrieving a dataset, evaluating a benchmark, or recording leaderboard submissions.
    
    Args:
        name (str): Input dataset name provided by a user or calling code. In the TDC domain this is typically a short identifier for a benchmark or dataset (for example, names returned by TDC retrieval utilities or passed to data loaders). The function treats this string case-insensitively, strips a leading "tdc." prefix if present, and then attempts to match it against the canonical dataset identifier(s) for that task.
        dataset_names (str): The exact dataset name(s) used by TDC against which the input name is compared. In the TDC workflow this value represents the canonical dataset identifier(s) registered for a given task or dataset collection. The function tests membership of the normalized input name in this value and, if no exact case-insensitive match is found, computes a closest fuzzy match using the internal get_closet_match helper to propose the canonical name to return.
    
    Returns:
        s (str): The resolved canonical TDC dataset name corresponding to the supplied input. This return value is the exact identifier that downstream TDC data loaders and evaluators expect; callers should use this returned string when requesting dataset objects, splits, or when submitting results to TDC leaderboards.
    
    Raises:
        ValueError: Raised when no valid canonical dataset name can be resolved from the provided input and the provided dataset_names. The error message includes the proposed match and indicates that it does not belong to the expected task, guiding the user to check and supply a correct dataset identifier.
    
    Behavior and failure modes:
        The function first normalizes the input by converting it to lowercase and removing a leading "tdc." prefix if present. If the normalized input exactly matches one of the canonical names (membership tested against dataset_names), that canonical name is returned. If there is no exact match, the function calls get_closet_match(dataset_names, name) to compute a nearest fuzzy match and returns the first candidate. If the returned candidate is not contained in dataset_names, the function raises ValueError. There are no other side effects. Callers should provide dataset_names that reflect the canonical identifiers for the intended TDC task; otherwise fuzzy matching may return an incorrect candidate or raise ValueError.
    """
    from tdc.utils.misc import fuzzy_search
    return fuzzy_search(name, dataset_names)


################################################################################
# Source: tdc.utils.misc.get_closet_match
# File: tdc/utils/misc.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_utils_misc_get_closet_match(
    predefined_tokens: list,
    test_token: str,
    threshold: float = 0.8
):
    """Get the closest match for a user-provided token against a list of predefined tokens using Levenshtein-based similarity.
    
    This function is used in TDC utilities to robustly map free-form user inputs (for example, dataset names, task names, or configuration keys supplied to TDC data loaders or functions) to canonical tokens accepted by the library. It computes a case-insensitive Levenshtein similarity score (via fuzzywuzzy.fuzz.ratio) between the string form of each predefined token and the provided test_token, returns the predefined token with the highest similarity and the corresponding normalized score in [0.0, 1.0]. The function helps prevent user input errors by suggesting the closest valid token and by enforcing a minimum similarity threshold to accept a match; if the best match is below the threshold it prints available tokens (via print_sys) and raises a ValueError so callers can handle invalid inputs explicitly.
    
    Args:
        predefined_tokens (list): Predefined string tokens to compare against. In TDC this typically contains canonical names such as dataset identifiers, task names, or other accepted keys. Each element is converted to str() and lowercased before comparison. If this list is empty, the underlying numpy operations (np.nanmax / np.nanargmax) will raise an exception.
        test_token (str): User input that needs matching to existing tokens. The value is converted to str() and lowercased before computing similarity, enabling matching of inputs with different casing or minor typographical differences.
        threshold (float): The lowest accepted normalized match score required to consider a match valid. This is a float in the same units used by the return probability (0.0 to 1.0). The default is 0.8, meaning the best match must have similarity >= 0.8 to be accepted. If the best normalized similarity is strictly less than threshold, the function prints the available predefined_tokens (using print_sys) and raises ValueError indicating the supplied test_token does not match available values.
    
    Returns:
        tuple: A pair (token, score) where:
            token (str): The predefined token with the highest raw similarity score. This value is taken directly from predefined_tokens (not lowercased or otherwise altered beyond str() conversion for comparison) and represents the canonical token that best matches the user input.
            score (float): The highest normalized similarity as a float in [0.0, 1.0], obtained by dividing fuzzywuzzy.fuzz.ratio integer output (0-100) by 100. This score indicates the Levenshtein-based similarity between test_token and the returned token; higher is more similar.
    
    Raises:
        ValueError: Raised when the highest normalized similarity score is lower than threshold. The function prints predefined_tokens (via print_sys) to help the user choose a valid token and then raises a ValueError containing the original test_token and a message to double check the input.
        Exception: If predefined_tokens is empty or contains values that cause np.nanmax / np.nanargmax to fail, numpy will raise an error (e.g., ValueError for zero-size array). This function does not explicitly handle that case.
    """
    from tdc.utils.misc import get_closet_match
    return get_closet_match(predefined_tokens, test_token, threshold)


################################################################################
# Source: tdc.utils.misc.install
# File: tdc/utils/misc.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_utils_misc_install(package: str):
    """Install a pip package into the Python interpreter that is running the current process.
    
    This function is a thin utility used in the TDC (Therapeutics Data Commons) codebase to programmatically ensure that a required or optional Python package is available at runtime. It invokes the pip module using the same Python executable that launched the process (sys.executable) by calling subprocess.check_call([sys.executable, "-m", "pip", "install", package]). In the TDC domain, this is useful for installing optional dependencies needed by specific data loaders, data functions, or oracles so that downstream dataset retrieval and processing functions can run without manual intervention. The call is executed synchronously and blocks until pip completes; pip's normal output and error streams are forwarded to the current process.
    
    Args:
        package (str): The pip package specifier to install. This string is passed verbatim to pip as the install target and typically contains a package name (for example, "numpy"), a version specifier (for example, "numpy==1.21.0"), or any pip-recognized install specifier (for example, VCS/URL specifiers). The argument determines which distribution will be fetched from package indexes or URLs and installed into the site-packages of the Python interpreter identified by sys.executable.
    
    Returns:
        None: This function does not return a value. Its primary effect is a side effect: installing (or attempting to install) the requested package into the current Python environment. After successful completion, the installed package becomes importable by code running under the same Python executable. If installation fails, no return value is produced because an exception is raised.
    
    Raises:
        subprocess.CalledProcessError: If the pip subprocess exits with a non-zero status (installation failure, network error, package resolution error, permission denied, etc.). The exception contains the exit code from the pip process.
        OSError: If the subprocess cannot be started (for example, if sys.executable is not a valid executable on the host system).
        ValueError: If an invalid or empty string is provided that leads pip to error; pip itself may also report other errors arising from the specifier.
    
    Notes:
        - The installation affects the Python interpreter identified by sys.executable, so in virtual environments or environments managed by tools like conda, the package will be installed into that environment's site-packages.
        - Network access to PyPI or the specified package source is normally required unless a local wheel or cached package is used.
        - Installation may modify or upgrade existing packages and may require write permissions to the environment; when running in restricted or read-only environments (such as some CI runners or system Python installations), the call may fail or install into a user site (pip may prompt or fail unless flags like --user are used).
        - This function is intended as a convenience within TDC for programmatic dependency management; for reproducible environment setup, prefer declarative tools (for example, pip freeze, requirements files, or conda environment specifications) when preparing production or shared research environments.
    """
    from tdc.utils.misc import install
    return install(package)


################################################################################
# Source: tdc.utils.misc.load_dict
# File: tdc/utils/misc.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_utils_misc_load_dict(path: str):
    """tdc.utils.misc.load_dict: Load and return a Python object previously serialized with pickle from a local filesystem path.
    
    Opens the file at the given path in binary read mode and deserializes its contents using pickle.load. The file handle is managed with a context manager and is closed automatically when loading completes. In the Therapeutics Data Commons (TDC) project, this helper is used to restore persisted Python objects such as cached dataset splits, preprocessed data artifacts, or metadata produced by TDC data loaders and data functions so that downstream machine-learning experiments and evaluations can resume without recomputation.
    
    Args:
        path (str): Filesystem path to a pickle file that contains a Python object serialized with pickle (for example via pickle.dump). This parameter must be a string and must point to an existing, readable file on the local filesystem. There is no default value; callers must supply a valid path. Practical significance in TDC: point this argument to stored artifacts like dataset split dictionaries or cached data processing results to quickly reload those objects into memory for model training, evaluation, or benchmarking.
    
    Returns:
        object: The Python object deserialized from the pickle file at path. In TDC workflows this is typically a dataset split mapping, cached preprocessing output, or other picklable artifact used by data loaders and evaluation code. If loading is successful, the object is returned to the caller and can be used directly in downstream TDC APIs.
    
    Failure modes and side effects:
        - Raises FileNotFoundError if the path does not exist.
        - Raises PermissionError if the file cannot be opened due to insufficient permissions.
        - Raises pickle.UnpicklingError, EOFError, AttributeError, or other exceptions if the file is not a valid pickle or the serialized object cannot be reconstructed.
        - Security risk: unpickling executes arbitrary code during deserialization. Do not load pickle files from untrusted or unauthenticated sources.
        - Side effects are limited to reading from disk; no global state is mutated by this function beyond returning the deserialized object.
    """
    from tdc.utils.misc import load_dict
    return load_dict(path)


################################################################################
# Source: tdc.utils.misc.print_sys
# File: tdc/utils/misc.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_utils_misc_print_sys(s: str):
    """tdc.utils.misc.print_sys: Print a string to the process standard error stream and flush immediately.
    
    This utility function is provided in the Therapeutics Data Commons (TDC) codebase to emit messages to the operating system's standard error stream (sys.stderr). It is used within TDC to report warnings, errors, or system-level status messages that should be separated from normal program output or machine-readable output (for example, when command output is being piped or captured). The implementation uses Python's built-in print with flush=True and file=sys.stderr to guarantee immediate delivery of the text to stderr.
    
    Args:
        s (str): the string to print. In the TDC context, this parameter is expected to contain a human-readable message such as a warning, error description, or status update. The function writes the exact string content to sys.stderr and forces a flush so the message appears immediately in the terminal or any redirected stderr stream.
    
    Behavior and side effects:
        The function writes to the global sys.stderr file object and flushes its output buffer. This has the side effect of producing visible output in the process's error stream, which is suitable for diagnostics and logging separate from standard output. The function does not perform buffering beyond the immediate flush, nor does it perform formatting beyond printing the provided string. The function does not catch exceptions raised by the underlying I/O operations.
    
    Failure modes and errors:
        If the global sys.stderr file object is closed, invalid, or an I/O error occurs during writing or flushing, the underlying exception (for example, ValueError or OSError) will propagate to the caller. The function signature declares s as a str; providing a value that does not match this declared type violates the function contract in the codebase and may result in unexpected behavior.
    
    Returns:
        None: This function does not return a value. Its observable effect is the side effect of writing and flushing the provided string to sys.stderr.
    """
    from tdc.utils.misc import print_sys
    return print_sys(s)


################################################################################
# Source: tdc.utils.misc.to_submission_format
# File: tdc/utils/misc.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_utils_misc_to_submission_format(results: dict):
    """Convert evaluation results into a submission-ready summary for TDC leaderboards.
    
    Args:
        results (dict): A dictionary of evaluation metrics collected across multiple runs (typical use: five runs) for a TDC benchmark. The dictionary is expected to map metric identifiers (e.g., 'ROC-AUC', 'RMSE', or dataset-specific metric names produced by TDC Evaluator) to an iterable (e.g., list) of per-run records. Each per-run record must be a single-key mapping (a dict with one key) whose value is the numeric score for that run. The function constructs a pandas.DataFrame from this input and computes aggregated statistics per metric. This argument is the primary input used to convert per-run metric outputs (from model evaluation on TDC datasets and splits) into the compact format required for leaderboard submission.
    
    Returns:
        dict: A dictionary that maps the same metric identifiers present in the input to a two-element list [mean, std], where mean is the arithmetic mean and std is the standard deviation computed with numpy (np.mean and np.std, numpy.std uses ddof=0 by default). Both values are rounded to three decimal places to produce a concise, submission-ready summary. For example, given per-run numeric scores, the returned dict entry for a metric might look like 'ROC-AUC': [0.912, 0.023]. If the input is empty, an empty dict is returned. There are no other side effects; the function does not modify files or global state.
    
    Raises:
        ValueError: If any per-run record is not a mapping object with at least one value (so that list(i.values())[0] can extract a numeric score), or if extracted values are not numeric, the function will raise an exception (typically from pandas or numpy) during DataFrame construction or during numeric aggregation. Users should ensure the input follows the expected structure: a dict of iterables of single-key dicts containing numeric values.
    """
    from tdc.utils.misc import to_submission_format
    return to_submission_format(results)


################################################################################
# Source: tdc.utils.query.cid2smiles
# File: tdc/utils/query.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_utils_query_cid2smiles(cid: str):
    """tdc.utils.query.cid2smiles retrieves a canonical SMILES string for a small molecule identified by a PubChem Compound ID (CID). This function is used in TDC data-processing and dataset workflows to translate PubChem numeric identifiers into the molecular representation (SMILES) required by downstream components such as data loaders, oracles, and model inputs.
    
    Args:
        cid (str): PubChem CID provided as a string. This is the PubChem compound identifier that the function will query. In the TDC context, CIDs commonly appear in small-molecule datasets and must be supplied as strings (for example, "2244") so the internal network request and JSON parsing functions accept them.
    
    Returns:
        str: The canonical SMILES string extracted from PubChem for the given CID. The implementation requests the PubChem compound JSON, inspects the ["PC_Compounds"][0]["props"] block, and extracts the property with label "SMILES" and name "Canonical" via an internal helper (_parse_prop). On success this is the SMILES string used throughout TDC for molecular representations. On failure (for example, network errors, invalid CID, unexpected JSON structure, or parsing errors), the function prints the message "cid <cid> failed, use NULL string" to standard output and returns the literal string "NULL". The function handles exceptions internally and will therefore always return a str; calling code should check for the "NULL" sentinel to detect failures.
    """
    from tdc.utils.query import cid2smiles
    return cid2smiles(cid)


################################################################################
# Source: tdc.utils.query.uniprot2seq
# File: tdc/utils/query.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_utils_query_uniprot2seq(ProteinID: str):
    """tdc.utils.query.uniprot2seq: Retrieve the amino-acid sequence string for a protein from the UniProt web service given a UniProt identifier. This function performs a synchronous HTTP GET to the UniProt FASTA endpoint, reads the returned FASTA file, strips the FASTA header (first line) and concatenates the remaining lines into a single continuous amino-acid sequence string. In the Therapeutics Data Commons (TDC) workflow, this is useful when a dataset or task requires the primary sequence of a protein (for example, for single-instance prediction tasks, feature extraction, or linking UniProt entries to sequence-based models).
    
    Args:
        ProteinID (str): the Uniprot ID to query (coerced to str internally). This should be the identifier used by UniProt (for example, an accession like "P12345" or other UniProt entry identifier). The function does not validate that the string is an accession format beyond converting the input to str and appending it to the UniProt FASTA URL.
    
    Behavior and practical details:
        The function issues an HTTP request to "http://www.uniprot.org/uniprot/{ProteinID}.fasta" using urllib.request.urlopen. It expects a FASTA-formatted response where the first line is the header (prefixed by '>') and subsequent lines contain the sequence. It strips whitespace from each sequence line, decodes bytes as UTF-8, and concatenates them into a single continuous string (no newline characters or intervening whitespace). This returned string represents the amino-acid sequence as one-letter codes as provided by UniProt. The function performs no further validation of amino-acid characters and does not filter or normalize non-standard residues; it returns exactly the concatenation of sequence lines after stripping.
    
    Side effects and defaults:
        The function performs network I/O and is blocking until the remote server responds. There is no local caching, retry logic, rate limiting, or authentication handled by the function. The ProteinID is coerced to a string via str(ProteinID) before use.
    
    Failure modes and exceptions:
        If the UniProt service is unreachable, if the ProteinID does not correspond to a UniProt entry, or if the HTTP request fails, urllib.request.urlopen will raise an exception (for example, urllib.error.URLError or urllib.error.HTTPError). The function does not catch these exceptions; callers should handle network and HTTP errors as appropriate for their application. If the returned FASTA contains no sequence lines, the function will return an empty string. Changes to the UniProt FASTA endpoint URL, redirects to HTTPS, or changes in response format may also cause failures.
    
    Returns:
        str: The amino-acid sequence retrieved from UniProt for the given ProteinID. This is the concatenation of all non-header FASTA lines decoded as UTF-8 and stripped of surrounding whitespace, returned as a single continuous string of one-letter amino-acid codes.
    """
    from tdc.utils.query import uniprot2seq
    return uniprot2seq(ProteinID)


################################################################################
# Source: tdc.utils.retrieve.get_label_map
# File: tdc/utils/retrieve.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_utils_retrieve_get_label_map(
    name: str,
    path: str = "./data",
    target: str = None,
    file_format: str = "csv",
    output_format: str = "dict",
    task: str = "DDI",
    name_column: str = "Map"
):
    """Retrieve the biomedical meaning of encoded labels for a TDC dataset and return the mapping in a user-specified format.
    
    Args:
        name (str): The (possibly fuzzy) dataset name to load from the TDC collection. This string is first resolved with fuzzy_search against the dataset namespace defined by dataset_names[task], so the provided name may be corrected to a canonical dataset identifier used on disk and in TDC metadata. In the therapeutics domain (TDC), this name selects a benchmark dataset (for example, a drug–drug interaction dataset when task='DDI') whose label mapping is required for interpretation, evaluation, or leaderboard reporting.
        path (str): The filesystem path where dataset files are stored. Default "./data". This is passed to the internal pd_load function to locate and read the dataset file. In practice, set this to the directory used to cache or store TDC datasets so that get_label_map can read the dataset table from disk.
        target (None): The name of the column in the loaded dataset that contains encoded label values (the keys for the mapping). If None (default), the function uses the string "Y" as the target column. In TDC datasets the target column typically stores the machine-readable label encodings (for example, numeric class IDs for interaction types or activity labels) that models output and evaluators consume.
        file_format (str): The expected on-disk file format of the dataset (default "csv"). This parameter documents the expected input format and is accepted for compatibility with other TDC utilities. In the current implementation get_label_map calls pd_load(name, path) to read the file; pd_load is responsible for actual parsing based on available files and formats. Do not assume get_label_map itself performs any format-specific parsing beyond invoking pd_load.
        output_format (str): Controls the returned mapping structure. Allowed values are "dict", "df", and "array" (default "dict"). "dict" returns a Python dictionary mapping encoded labels (from the target column) to human-readable biomedical label names (from name_column); "df" returns the entire pandas.DataFrame loaded from disk; "array" returns a numpy.ndarray of the label names (the name_column values). The chosen format affects downstream usage: use "dict" for quick lookup during evaluation or interpretation, "df" to inspect additional metadata columns, and "array" when a lightweight sequence of label names is required.
        task (str): The TDC task namespace to resolve dataset names against (default "DDI"). This selects which subset of dataset names (dataset_names[task]) fuzzy_search uses to correct or match the provided name. In TDC, tasks represent high-level domains such as "single_pred", "DDI", etc.; the default "DDI" indicates this function will by default search among drug–drug interaction datasets unless another task is specified.
        name_column (str): The name of the column in the loaded dataset that stores the human-readable biomedical label names (default "Map"). These values are the practical meanings of encoded labels (for example, interaction type names, assay outcome descriptions, or adverse event labels) that researchers and leaderboards display and that make model outputs interpretable.
    
    Returns:
        dict/pd.DataFrame/np.array: When output_format is "dict", returns a dict built with keys from df[target].values and values from df[name_column].values; keys are the encoded labels used by models and evaluators and values are their biomedical meanings. When output_format is "df", returns the full pandas.DataFrame as loaded from disk (useful to inspect additional columns and metadata). When output_format is "array", returns the numpy.ndarray df[name_column].values containing only the human-readable label names in the order they appear in the file.
    
    Raises:
        ValueError: If output_format is not one of "dict", "df", or "array". The function enforces these three output formats and will raise this error for any other string.
        FileNotFoundError/IOError: If the dataset file resolved by fuzzy_search and pd_load does not exist at the provided path or cannot be opened. The exact exception depends on pd_load and underlying IO.
        KeyError: If the resolved DataFrame does not contain the specified target column (default "Y" when target is None) or the specified name_column (default "Map"), a KeyError will be raised when attempting to access those columns. This indicates a mismatch between expected TDC dataset schema and the actual file contents.
    
    Behavior and side effects:
        The function resolves the provided name against the TDC dataset namespace for the given task using fuzzy_search, which may alter the name to a canonical dataset identifier. It then loads the dataset table from disk via pd_load(name, path). No persistent changes are made to disk by get_label_map; its primary side effect is reading files. The function returns one of three in-memory representations (dict, DataFrame, or numpy array) as specified by output_format. Because file_format is accepted but not used directly within get_label_map, rely on pd_load and the dataset files present at path for correct parsing. This function is intended for use in the TDC therapeutics workflow to translate model label outputs (encoded targets) into human-interpretable biomedical meanings for evaluation, inspection, and reporting.
    """
    from tdc.utils.retrieve import get_label_map
    return get_label_map(
        name,
        path,
        target,
        file_format,
        output_format,
        task,
        name_column
    )


################################################################################
# Source: tdc.utils.retrieve.get_reaction_type
# File: tdc/utils/retrieve.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_utils_retrieve_get_reaction_type(
    name: str,
    path: str = "./data",
    output_format: str = "array"
):
    """Retrieve the reaction type labels for a RetroSyn reaction dataset.
    
    This function is used within the TDC (Therapeutics Data Commons) workflow to obtain reaction-category labels from a reaction dataset (specifically those indexed under the RetroSyn collection). It first performs a fuzzy match of the provided dataset name against known RetroSyn dataset names (via fuzzy_search) to find the canonical dataset identifier, then loads the dataset from disk using pd_load. The loaded dataset is expected to contain a "category" column that encodes the reaction type for each record; these reaction-type labels are commonly used for reaction classification, retrosynthesis evaluation, and other reaction-prediction tasks in drug discovery workflows supported by TDC.
    
    Args:
        name (str): Dataset name or approximate name to identify a RetroSyn reaction dataset. The function applies fuzzy matching against the RetroSyn dataset list (dataset_names["RetroSyn"]) to resolve the exact dataset identifier; this allows users to provide partially matching or slightly misspelled names and still retrieve the intended dataset.
        path (str, optional): Filesystem path to the folder where TDC datasets are stored. Defaults to "./data". This path is forwarded to the internal pd_load function which reads the dataset file(s) from disk. If the dataset file is missing, unreadable, or corrupted, pd_load (or underlying I/O) may raise an exception.
        output_format (str, optional): Desired format of the returned data. Must be either "df" to return the full pandas DataFrame as loaded by pd_load, or "array" to return the raw reaction-type labels as a numpy array extracted from the DataFrame's "category" column. Defaults to "array".
    
    Returns:
        pd.DataFrame/np.array: When output_format is "df", returns the full pandas DataFrame loaded for the resolved RetroSyn dataset (this DataFrame is the canonical table used for downstream tasks and includes at least a "category" column representing reaction types). When output_format is "array", returns a numpy array (np.array) of the values in the DataFrame's "category" column, i.e., the sequence of reaction-type labels suitable for use as labels in model training, evaluation, or analysis.
    
    Raises:
        ValueError: If output_format is not one of the supported values ("df" or "array"), a ValueError is raised with guidance to select from "df" or "array". Additionally, underlying utilities used by this function (for example, fuzzy_search to resolve the dataset name or pd_load to read files) may raise their own exceptions (such as lookup errors or I/O-related errors) if the name cannot be resolved or the dataset cannot be loaded from the provided path.
    """
    from tdc.utils.retrieve import get_reaction_type
    return get_reaction_type(name, path, output_format)


################################################################################
# Source: tdc.utils.retrieve.retrieve_benchmark_names
# File: tdc/utils/retrieve.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_utils_retrieve_retrieve_benchmark_names(name: str):
    """tdc.utils.retrieve.retrieve_benchmark_names returns all benchmark dataset names associated with a queried benchmark group registered in TDC. This function is used by higher-level TDC utilities and user code to enumerate available benchmarks (datasets) that belong to a logically defined benchmark group (for example, 'ADMET_Group' as used in TDC leaderboards and examples in the README). It performs a fuzzy match of the provided group name against the internal benchmark registry and then collects dataset names from every learning task contained in the matched group so downstream code can present, iterate over, or submit results for those benchmarks.
    
    Args:
        name (str): the name of the benchmark group to query. This should be a string identifying a benchmark group in TDC (for example, "ADMET_Group"); the function applies fuzzy matching against the internal registry of benchmark group names to select the closest available group. The parameter represents the logical grouping of related benchmarks (used for leaderboards, curated comparisons, and dataset discovery).
    
    Returns:
        list: a list of benchmarks. Each element is a dataset name (string) belonging to the matched benchmark group. The list is produced by iterating the group's tasks and aggregating every dataset name; ordering follows the iteration order of the internal registry and each task's dataset list. If the matched group contains no datasets, an empty list is returned.
    
    Raises:
        KeyError: if the fuzzy-matched name is not present in the internal benchmark registry or the registry is missing expected entries, causing a lookup failure.
        Exception: if fuzzy_search (used to match the input name to available groups) raises an error due to invalid input or internal failures. Callers should ensure `name` is a valid string and handle exceptions when the input cannot be resolved to a registered benchmark group.
    
    Notes:
    - Side effects: none on persistent state; the function performs read-only access to the in-memory benchmark registry and returns a flattened list of dataset names.
    - Practical significance: this function simplifies listing and programmatic access to all benchmarks within a thematic group (useful for building leaderboards, batch evaluation pipelines, and exploring TDC groups as shown in the README).
    """
    from tdc.utils.retrieve import retrieve_benchmark_names
    return retrieve_benchmark_names(name)


################################################################################
# Source: tdc.utils.retrieve.retrieve_dataset_names
# File: tdc/utils/retrieve.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_utils_retrieve_retrieve_dataset_names(name: str):
    """Return all available dataset names for a given TDC learning task.
    
    This function looks up the module-level mapping dataset_names and returns the list of dataset identifiers that belong to the specified learning task in the Therapeutics Data Commons (TDC) hierarchy. In the TDC domain, tasks correspond to high-level learning problems (for example, the single-instance prediction task 'ADME') and each task exposes multiple dataset variants (for example, 'HIA_Hou'). The returned dataset names are intended to be used when instantiating dataset loaders (for example, passing a name to a task-specific constructor such as ADME(name='HIA_Hou')) or when enumerating available benchmarks for model development, evaluation, and leaderboard submission.
    
    Args:
        name (str): The canonical name of the TDC learning task to query. This string must match one of the task keys used in the TDC codebase (for example, 'ADME', 'BenchmarkGroup' task names, or other task identifiers exposed by TDC). The function uses this exact value to index the internal dataset_names mapping and does not perform fuzzy matching; providing an incorrect or misspelled task name will result in a lookup failure.
    
    Returns:
        list: A list of available dataset names for the specified task. Each element of the returned list is a dataset identifier string that can be passed to TDC dataset constructors or used to inspect available benchmarks and splits. The order of names in the list reflects the entries in the internal mapping and should not be relied upon as sorted.
    
    Raises:
        KeyError: If the provided name is not a key in the internal dataset_names mapping (i.e., the task is unknown to the local TDC installation). This indicates the caller should verify available task names (for example, by consulting TDC documentation or other listing utilities).
        TypeError: If a non-string value is passed where a str is expected, callers should pass a str per the function signature.
    
    Behavior and side effects:
        This function performs an in-memory dictionary lookup and has no external side effects (no network or file I/O). It does not modify global state. It does not validate whether datasets referenced by name are currently downloadable or have their resources available; it only returns the registered dataset identifiers for the given task.
    """
    from tdc.utils.retrieve import retrieve_dataset_names
    return retrieve_dataset_names(name)


################################################################################
# Source: tdc.utils.retrieve.retrieve_label_name_list
# File: tdc/utils/retrieve.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_utils_retrieve_retrieve_label_name_list(name: str):
    """tdc.utils.retrieve.retrieve_label_name_list: Return the set of available label (target) names for a given TDC dataset.
    
    This function is part of the Therapeutics Data Commons (TDC) utilities for dataset discovery and metadata inspection. It accepts a rough dataset name provided by a user or higher-level code, uses a fuzzy lookup against the internal registry of dataset identifiers (via fuzzy_search and the module-level dataset_list), and returns the canonical list of target/label names that the matched dataset exposes. In the TDC domain, these label names identify the prediction targets used for model training, validation, testing, evaluation, and downstream tasks such as metric computation, data processing, and oracle scoring for drug discovery and therapeutic ML benchmarks.
    
    Args:
        name (str): rough dataset name. A user-provided, possibly partial or imprecise string that identifies a dataset in TDC (for example, a dataset short name shown on the TDC website or in the README). The function performs fuzzy matching against the module-level dataset_list to find the canonical dataset identifier; this parameter must be a Python str exactly as provided to the function signature.
    
    Returns:
        list: a list of available label names for the matched dataset. Each element is the name of a target/label (typically a Python str corresponding to a column or target identifier in the dataset) that downstream code can use to select columns from the dataset DataFrame, to construct model input/output interfaces, or to register evaluation metrics. The returned list is drawn from the module-level mapping dataset2target_lists for the canonical dataset name.
    
    Behavior and side effects:
        The function first calls fuzzy_search(name, dataset_list) to resolve the provided rough name to a canonical dataset identifier found in the TDC registry. It then performs a dictionary lookup dataset2target_lists[canonical_name] and returns that list. There are no persistent side effects or mutations performed by this function; it only reads module-level registries and returns the corresponding metadata.
    
    Defaults and usage notes:
        There are no implicit defaults for the name parameter; you must pass a str. This function is intended for exploratory use (for example, listing available prediction targets before calling data loaders or evaluators) and for programmatic workflows that need to enumerate targets for a given dataset.
    
    Failure modes and exceptions:
        If fuzzy_search cannot resolve the provided name to any entry in dataset_list, fuzzy_search will raise its own exception (e.g., indicating no match); that exception propagates to the caller. If fuzzy_search returns a canonical name that is not present in dataset2target_lists, a KeyError will be raised. Callers should catch these exceptions or validate dataset names against retrieve_dataset_names or the TDC website before calling this function.
    """
    from tdc.utils.retrieve import retrieve_label_name_list
    return retrieve_label_name_list(name)


################################################################################
# Source: tdc.utils.split.create_combination_generation_split
# File: tdc/utils/split.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def tdc_utils_split_create_combination_generation_split(
    dict1: dict,
    dict2: dict,
    seed: int,
    frac: list
):
    """Create a random train/validation/test split for paired protein–ligand coordinate and atom-type datasets used in TDC combination generation tasks.
    
    This function is intended for the "generation" problem in TDC where each example is a paired combination (e.g., a protein and a ligand) represented by coordinates and atom-type lists. It permutes example indices using NumPy's RNG, then slices the permuted index array according to the provided train/validation/test fractions to produce three partition dictionaries. The returned structure is suitable for downstream molecule- or complex-generation workflows (for example, goal-oriented or distribution-learning oracles in TDC) that require aligned protein and ligand inputs for each split.
    
    Args:
        dict1 (dict): A dictionary representing the first modality in the pair (conventionally the protein). Must contain at least the keys "coord" and "atom_type". "coord" should be an indexable sequence (e.g., list or array) of per-example coordinate arrays and "atom_type" an indexable sequence of corresponding atom-type arrays. The number of examples is inferred from len(dict1["coord"]). The function will raise KeyError if required keys are missing and may raise IndexError if lengths are inconsistent with dict2.
        dict2 (dict): A dictionary representing the second modality in the pair (conventionally the ligand). Must contain at least the keys "coord" and "atom_type" with indexable sequences parallel to dict1. The function assumes dict1["coord"] and dict2["coord"] have the same length and that corresponding examples should be split together. If lengths differ, indexing dict2 with indices derived from dict1 may raise IndexError.
        seed (int): An integer random seed parameter provided for API consistency and intended for reproducibility. Note: this function does not set or use the seed internally; it calls numpy.random.permutation which uses NumPy's global RNG state. To obtain reproducible splits, callers must set the NumPy RNG before calling (for example, numpy.random.seed(seed)). If the seed is ignored externally, splits will be non-deterministic.
        frac (list): A list of three numeric fractions [train_frac, val_frac, test_frac] that partition the dataset. Each element is used as a multiplier of the total example count to determine integer cut points via int(length * fraction). The function does not validate that the list sums to 1.0: rounding via integer truncation will allocate the remainder to the final partition (test) since the final slice takes any indices after the train and validation ranges. Passing non-numeric values, negative fractions, or a list of length other than 3 may produce TypeError/ValueError or incorrect splits.
    
    Returns:
        dict: A dictionary with exactly three keys "train", "valid", and "test". Each value is a dictionary with the following keys:
            "protein_coord": list of dict1["coord"] entries assigned to that split,
            "protein_atom_type": list of dict1["atom_type"] entries assigned to that split,
            "ligand_coord": list of dict2["coord"] entries assigned to that split,
            "ligand_atom_type": list of dict2["atom_type"] entries assigned to that split.
        The returned lists preserve example alignment between protein and ligand modalities so each position across the four lists corresponds to one paired example. There are no side effects on the input dictionaries; inputs are not mutated. Exceptions that may be raised include KeyError (missing required keys), IndexError (mismatched lengths between dict1 and dict2), and TypeError/ValueError for invalid frac contents.
    """
    from tdc.utils.split import create_combination_generation_split
    return create_combination_generation_split(dict1, dict2, seed, frac)


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
