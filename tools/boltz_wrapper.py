"""
Regenerated Google-style docstrings for module 'boltz'.
README source: others/readme/boltz/README.md
Generated at: 2025-12-02T00:57:54.748437Z

Total functions: 46
"""


import numpy
import torch

################################################################################
# Source: boltz.data.crop.boltz.pick_chain_token
# File: boltz/data/crop/boltz.py
# Category: valid
################################################################################

def boltz_data_crop_boltz_pick_chain_token(
    tokens: numpy.ndarray,
    chain_id: int,
    random: numpy.random.mtrand.RandomState
):
    """Pick a random token belonging to a specified chain from a structured token array used by the Boltz cropping pipeline.
    
    This function is used by Boltz preprocessing (cropping) code to select a single token (for example a residue- or atom-level entry) from the set of tokens that represent a biomolecular system. In the Boltz domain (biomolecular interaction and binding-affinity prediction), selecting a representative token from a specific chain is useful for creating localized crops or anchors when preparing inputs for structure and affinity prediction models. The selection is randomized but reproducible when a numpy.random.mtrand.RandomState is provided.
    
    Args:
        tokens (numpy.ndarray): Structured numpy array containing token records for a biomolecular example. Each record is expected to contain an "asym_id" field that identifies the chain membership of that token. The array represents the full set of tokenized entities (atoms/residues) derived from the input structures used by Boltz for tasks such as structure prediction and affinity estimation.
        chain_id (int): The chain identifier to filter tokens by. This integer is compared for equality against the values in tokens["asym_id"] to select tokens that belong to the requested chain. In the Boltz preprocessing flow, this selects a chain-specific context (e.g., a protein chain or ligand chain) from which a token will be sampled.
        random (numpy.random.mtrand.RandomState): A numpy RandomState object used to draw the random sample deterministically. Providing this RandomState ensures reproducible behavior across runs (important for reproducible inference and training pipelines in Boltz).
    
    Returns:
        numpy.ndarray: A numpy.ndarray representing the selected token record (one entry/row from the input structured array). The returned object corresponds to a single token chosen by pick_random_token from the tokens belonging to chain_id when any are present, otherwise from the full tokens array. The returned token is intended to be used as the query/anchor token for subsequent cropping or modeling steps in the Boltz pipeline.
    
    Behavior and side effects:
        The function first filters tokens by tokens["asym_id"] == chain_id to build a chain-specific view. If the filtered view contains one or more entries, pick_random_token(chain_tokens, random) is called to sample a token from that chain. If the filtered view is empty, pick_random_token(tokens, random) is called to sample from the entire token set as a fallback. The function does not modify the input tokens array or the provided RandomState; it only reads from them and returns a selected token.
    
    Failure modes and preconditions:
        The tokens array must be a numpy.ndarray with a field named "asym_id" accessible by tokens["asym_id"]; otherwise a KeyError (or similar indexing error) will be raised. The values in tokens["asym_id"] must be comparable to chain_id via equality. If the tokens array is empty and thus pick_random_token is called on an empty array, pick_random_token may raise an error (e.g., due to no available entries to sample). The random argument must be an instance of numpy.random.mtrand.RandomState; passing a different random generator type may lead to TypeError or non-deterministic behavior.
    """
    from boltz.data.crop.boltz import pick_chain_token
    return pick_chain_token(tokens, chain_id, random)


################################################################################
# Source: boltz.data.crop.boltz.pick_interface_token
# File: boltz/data/crop/boltz.py
# Category: valid
################################################################################

def boltz_data_crop_boltz_pick_interface_token(
    tokens: numpy.ndarray,
    interface: numpy.ndarray,
    random: numpy.random.mtrand.RandomState
):
    """Pick a single token from the specified intermolecular interface for use as a crop center in Boltz preprocessing and inference. This function is used by Boltz's data-cropping pipeline to choose a representative token located at or near the interface between two chains so that downstream model inputs (structure or affinity prediction) can be focused on the region of interaction.
    
    The function expects the tokens array to be a structured numpy.ndarray containing per-token fields used by the cropping logic (the implementation accesses tokens["asym_id"] to group tokens by chain and tokens["center_coords"] to compute inter-token distances). The interface argument is expected to provide chain identifiers for the two sides of the interface (the implementation reads interface["chain_1"] and interface["chain_2"]). Selection is randomized but deterministic if the provided RandomState is seeded.
    
    Behavior summary: If tokens exist only on one of the two interface chains, the function selects a random token from the available chain. If no tokens exist on either chain, it selects a random token from the entire tokens array. If tokens exist on both chains, it computes pairwise Euclidean distances between tokens' center_coords and retains tokens on each chain that lie within const.interface_cutoff of at least one token on the opposite chain. If no inter-chain distances fall below const.interface_cutoff, the cutoff is relaxed by +5.0 Å to avoid empty candidate sets. A random token is then chosen from the combined set of retained tokens from both chains. The provided RandomState controls the randomness to ensure reproducible sampling when seeded.
    
    Args:
        tokens (numpy.ndarray): Structured array of token records for a biomolecular assembly. Each record is expected to include at least the fields "asym_id" (chain identifier used to partition tokens by chain) and "center_coords" (Cartesian coordinates used to compute inter-token distances). This array supplies the pool of candidate tokens from which the function selects one representative token for an interface-centered crop.
        interface (numpy.ndarray): A record or structured ndarray that identifies the interface to sample from. The implementation accesses interface["chain_1"] and interface["chain_2"] to obtain the two asym_id values (chain identifiers) that define the interface. These identifiers are matched against tokens["asym_id"] to locate tokens belonging to each chain.
        random (numpy.random.mtrand.RandomState): Numpy RandomState used for reproducible randomized selection. If seeded externally, the same seed will produce the same token selection behavior across runs; if not seeded, selection is nondeterministic.
    
    Returns:
        numpy.ndarray: A single token record selected from the input tokens. The returned value is drawn from the same structured format as the input tokens (i.e., it contains the same fields such as "asym_id" and "center_coords") and represents the chosen token to be used as the center of an interface-focused crop for Boltz model inputs.
    
    Side effects and failure modes: The function does not modify the input arrays in-place. It relies on the presence of the "asym_id" and "center_coords" fields in tokens and on "chain_1"/"chain_2" in interface; missing fields will raise standard numpy/key-access errors. If the input tokens array is empty, downstream helper pick_random_token invoked by this function may raise an error; such errors are propagated rather than handled here. The distance threshold used to define interface membership is const.interface_cutoff from the module constants and may be relaxed by +5.0 Å in the rare case that no inter-chain pairs fall below the nominal cutoff.
    """
    from boltz.data.crop.boltz import pick_interface_token
    return pick_interface_token(tokens, interface, random)


################################################################################
# Source: boltz.data.crop.boltz.pick_random_token
# File: boltz/data/crop/boltz.py
# Category: valid
################################################################################

def boltz_data_crop_boltz_pick_random_token(
    tokens: numpy.ndarray,
    random: numpy.random.mtrand.RandomState
):
    """boltz.data.crop.boltz.pick_random_token picks a single token uniformly at random from a 1D collection of tokens using a supplied NumPy random state. In the Boltz data preprocessing and cropping pipeline (used for preparing biomolecular inputs to the model during training and inference), this function is used to select one token example from an array of candidate tokens in a reproducible way.
    
    Args:
        tokens (numpy.ndarray): An indexable NumPy array of tokens from which to pick. Each entry represents a single token for the Boltz data pipeline (for example, a discrete representation of a residue, atom feature vector, or other tokenized input used by cropping routines). The function computes len(tokens) and selects one element by integer index; the returned value is tokens[i] for a sampled index i. If tokens is empty (len(tokens) == 0) this will raise an error from numpy.random.mtrand.RandomState.randint.
        random (numpy.random.mtrand.RandomState): A NumPy RandomState instance used to draw the random integer index. Providing a RandomState allows reproducible sampling across runs and across data-loading workers when the same state/seed is used. If an object that is not a NumPy RandomState is passed, attribute or type errors may occur when randint is called.
    
    Returns:
        numpy.ndarray: The single selected token, returned as the same array element type as items in tokens (i.e., the dtype and sub-array shape are preserved). The function performs no copies beyond NumPy's normal indexing behavior and has no side effects on the input arrays; it only reads tokens and the provided random state to produce an index.
    """
    from boltz.data.crop.boltz import pick_random_token
    return pick_random_token(tokens, random)


################################################################################
# Source: boltz.data.feature.featurizer.dummy_msa
# File: boltz/data/feature/featurizer.py
# Category: valid
################################################################################

def boltz_data_feature_featurizer_dummy_msa(residues: numpy.ndarray):
    """Create a dummy multiple-sequence alignment (MSA) for a single chain used by the Boltz featurizer pipeline.
    
    Args:
        residues (numpy.ndarray): A 1-D numpy array of per-residue records for a single chain. Each element must be a mapping-like object (for example a dict or numpy.void) that supports indexing with the string key "res_type". The value at each element's "res_type" key is taken as the residue identifier for that position and must be convertible to the MSAResidue dtype used by boltz.data.types.MSAResidue. This function is typically used in the Boltz preprocessing/featurizer when no real MSA is available or when a placeholder single-sequence MSA is required for downstream structure or affinity prediction (e.g., in inference pipelines described in the repository README). If the input array elements do not provide "res_type", a KeyError will be raised; if the input is not an iterable numpy.ndarray, a TypeError may be raised.
    
    Returns:
        MSA: A boltz.data.types.MSA object representing a single-sequence (dummy) MSA suitable for downstream Boltz featurizers and models. The returned MSA has three fields constructed deterministically from the input:
        - residues: a numpy.ndarray of length equal to len(residues) with dtype MSAResidue, containing the extracted "res_type" values for each residue position. This encodes the sequence used by the model.
        - deletions: an empty numpy.ndarray with dtype MSADeletion, indicating no deletion events are present in this dummy alignment.
        - sequences: a numpy.ndarray with dtype MSASequence containing exactly one sequence record encoded as the tuple (0, -1, 0, len(residues), 0, 0). This single record encodes a placeholder sequence/metadata entry for the alignment; its fourth element is the sequence length (len(residues)). The function has no external side effects and simply returns this constructed MSA object.
    """
    from boltz.data.feature.featurizer import dummy_msa
    return dummy_msa(residues)


################################################################################
# Source: boltz.data.feature.featurizerv2.convert_atom_name
# File: boltz/data/feature/featurizerv2.py
# Category: valid
################################################################################

def boltz_data_feature_featurizerv2_convert_atom_name(name: str):
    """Convert an atom name string (as found in PDB or ligand files) into a fixed-length integer encoding used by the Boltz featurizerv2 pipeline.
    
    This function is used in the featurization stage of Boltz (a biomolecular interaction modeling system) to convert human-readable atom names (for example "CA", "N", "1HB", or " O  " from PDB records) into a numeric representation that can be included in model input features. The encoding is simple and deterministic: the function trims surrounding whitespace, casts the input to str, converts each character c to the integer ord(c) - 32 (so ASCII printable characters are mapped into small non-negative integers and a space character maps to 0), and pads with zero values so that typical atom names (up to 4 characters) produce a 4-element tuple suitable for fixed-size model inputs.
    
    Args:
        name (str): The atom name to convert. This should be the atom name string as present in structure files or generated by preprocessing (e.g., "CA", " N  ", "1HD"). The function will cast non-str inputs to str, strip leading and trailing whitespace, and then encode each remaining character. In the Boltz domain, atom names are expected to be short (commonly up to 4 characters); this parameter supplies the raw text label for a single atom that the featurizer converts into numeric features for downstream structural and affinity modeling.
    
    Returns:
        tuple[int, int, int, int]: A tuple intended to contain four integers representing the encoded atom name characters. For each character in the trimmed string, the corresponding tuple element is ord(character) - 32. If the trimmed name has fewer than four characters, the result is padded on the right with zeros so that typical, valid atom names yield a 4-tuple suitable for fixed-size model feature vectors. An empty or all-whitespace input yields (0, 0, 0, 0). Note: this function is implemented for the common case where atom names are at most four characters. If an input string has more than four characters the implementation does not explicitly truncate and can produce a tuple longer than four elements; such inputs are unsupported and may be incompatible with downstream code that expects exactly four elements.
    
    Behavior, side effects, and failure modes:
        This is a pure function with no side effects; it does not modify external state. It always casts the input to str and strips whitespace before encoding. The encoding ord(c) - 32 assumes characters are representable by Python's ord(); passing non-printable control characters (ord < 32) can produce negative integers, and passing unusual Unicode characters will produce integer values based on their Unicode code points minus 32. These cases are atypical for PDB/ligand atom names used by Boltz and may lead to unexpected downstream behavior. The function is intended for standard PDB-style atom names (ASCII, short); callers should ensure names are normalized to that convention before calling to guarantee a 4-element tuple compatible with the featurizer.
    """
    from boltz.data.feature.featurizerv2 import convert_atom_name
    return convert_atom_name(name)


################################################################################
# Source: boltz.data.feature.featurizerv2.dummy_msa
# File: boltz/data/feature/featurizerv2.py
# Category: valid
################################################################################

def boltz_data_feature_featurizerv2_dummy_msa(residues: numpy.ndarray):
    """Create a minimal, placeholder multiple-sequence-alignment (MSA) record for a single chain used by the Boltz featurization pipeline. This function is used when an actual MSA is unavailable or when a fast, deterministic placeholder is required (for example, when skipping MSA server queries during inference). The returned MSA enables downstream featurizers and the Boltz model to proceed without real alignment data by providing a single-sequence MSA with per-position residue types and no deletion events.
    
    Args:
        residues (numpy.ndarray): An array-like collection of residue records for the chain. Each element is expected to be a mapping-like object containing a "res_type" entry (the one-letter or encoded residue type used by the featurizer). The function extracts the "res_type" value for every element in this array to form the MSA residue sequence. If the input is not iterable or its elements do not provide "res_type", the function will raise a TypeError or KeyError respectively. An empty input yields a zero-length residue array and a corresponding sequence metadata entry with length zero; downstream code may assume non-zero length and could fail in that case.
    
    Returns:
        boltz.data.types.MSA: A minimal MSA object constructed for use by Boltz featurizers and inference. The returned MSA contains:
          - residues: a numpy.ndarray of the extracted residue types with dtype MSAResidue (one entry per input residue).
          - deletions: an empty numpy.ndarray with dtype MSADeletion indicating no deletion events in this placeholder MSA.
          - sequences: a numpy.ndarray with dtype MSASequence containing a single metadata tuple (0, -1, 0, len(residues), 0, 0). The fourth element of this tuple is the chain length (len(residues)); the tuple acts as the single-sequence metadata record consumed by downstream code.
    
    Behavior, side effects, and failure modes:
        - Deterministic: given the same input, the function always produces the same placeholder MSA.
        - No external I/O or network calls are performed; this is a purely local construction used to avoid MSA fetching.
        - The function does not modify its input array.
        - It relies on each element of residues exposing "res_type"; missing keys raise KeyError. Non-iterable or incompatible inputs raise TypeError. An empty residues array produces an MSA with zero-length residues which some downstream components may not accept.
    """
    from boltz.data.feature.featurizerv2 import dummy_msa
    return dummy_msa(residues)


################################################################################
# Source: boltz.data.feature.featurizerv2.get_range_bin
# File: boltz/data/feature/featurizerv2.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def boltz_data_feature_featurizerv2_get_range_bin(
    value: float,
    range_dict: dict,
    default: int = 0
):
    """boltz.data.feature.featurizerv2.get_range_bin maps a numeric value to a discrete integer bin index using a dictionary of numeric ranges. This function is used in the Boltz featurization pipeline to convert continuous scalar features (for example, distances, physicochemical descriptors, or affinity-related scalars such as log10(IC50) used by Boltz affinity predictions) into categorical bin indices that downstream model components expect.
    
    This function converts the input value to float, iterates over entries in range_dict (typically mapping tuple(low, high) -> int), and returns the first bin index whose interval contains the value according to the half-open convention low <= value < high. If no interval matches, the supplied default is returned. The function intentionally skips any key equal to the string "other" when testing intervals; that key may be reserved for other processing elsewhere in the featurizer.
    
    Args:
        value (float): The numeric value to bin. This is cast to float via float(value) so numeric-string inputs may be accepted if they can be converted; if conversion fails a ValueError or TypeError will be raised. In the Boltz domain, this can represent a continuous feature such as a distance, score, or affinity-related measurement that must be discretized for model input.
        range_dict (dict): A dictionary that defines bins. Expected usage (from source code) is a mapping where keys are 2-tuples (low, high) of floats defining half-open intervals [low, high) and values are integers representing bin indices. The function iterates over range_dict.items() in Python dict iteration order (insertion order on CPython 3.7+), so if intervals overlap the first matching interval encountered will determine the returned index. The function ignores any key that is the literal string "other" during matching.
        default (int): Integer returned when value does not fall into any of the provided intervals. Default is 0. In Boltz featurization pipelines this default often represents a fallback or background bin index used when a feature is out-of-range or missing.
    
    Returns:
        int: The bin index corresponding to the first interval (low, high) that satisfies low <= value < high, according to the iteration order of range_dict. If no matching interval is found, returns the provided default. No other side effects occur.
    """
    from boltz.data.feature.featurizerv2 import get_range_bin
    return get_range_bin(value, range_dict, default)


################################################################################
# Source: boltz.data.feature.featurizerv2.load_dummy_templates_features
# File: boltz/data/feature/featurizerv2.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for load_dummy_templates_features because the docstring has no description for the argument 'tdim'
################################################################################

def boltz_data_feature_featurizerv2_load_dummy_templates_features(
    tdim: int,
    num_tokens: int
):
    """Load dummy template features used by the v2 featurizer pipeline.
    
    This function allocates and returns a dictionary of zero-filled template feature tensors that match the keys and shapes expected by the Boltz v2 model featurizer and downstream networks. It is intended to be used when no real template structure data (for example PDB-derived templates) are available: calling code can pass these dummy tensors so the model receives a complete, well-typed feature set and does not need special-case logic for missing templates. The returned tensors follow the shapes and dtypes constructed in the v2 featurizer: rotation and translation frames, Cα/Cβ coordinates, boolean masks, an integer mapping from query tokens to template positions, and visibility flags. Note that the residue-type one-hot encoding uses the module-level constant const.num_tokens as the number of one-hot classes (num_classes) rather than the function argument num_tokens.
    
    Args:
        tdim (int): Number of templates dimension (tdim). In the Boltz v2 featurizer and model, this is the leading dimension indexing different template structures or pseudo-templates. This function allocates arrays with leading dimension tdim so each template receives its own set of per-token features. Supplying tdim=0 will produce zero-sized leading dimension tensors; very large tdim values increase memory use proportionally and may raise allocation errors.
        num_tokens (int): Number of tokens per template (num_tokens). In practice this corresponds to the per-template sequence length or number of residue positions for which features are allocated. This value is used to size the per-template second dimension of all returned tensors. Supplying num_tokens=0 will produce zero-length token dimensions; very large values increase memory use proportionally and may raise allocation errors.
    
    Returns:
        dict: A dictionary mapping feature names (strings) to torch.Tensor values. Each tensor is zero-filled and has the shapes and dtypes produced by the function implementation:
            "template_restype": torch.Tensor of shape (tdim, num_tokens, const.num_tokens). One-hot encoded residue-type tensor derived from an integer array of zeros using one_hot(..., num_classes=const.num_tokens). This represents per-template, per-token residue identity in a one-hot encoding with const.num_tokens classes. The number of one-hot classes is taken from the module-level constant const.num_tokens (not from the num_tokens argument).
            "template_frame_rot": torch.Tensor of dtype float32 and shape (tdim, num_tokens, 3, 3). Per-template, per-token rotation matrices for local residue frames.
            "template_frame_t": torch.Tensor of dtype float32 and shape (tdim, num_tokens, 3). Per-template, per-token translation vectors for local residue frames.
            "template_cb": torch.Tensor of dtype float32 and shape (tdim, num_tokens, 3). Per-template, per-token Cβ coordinates (zero if unavailable).
            "template_ca": torch.Tensor of dtype float32 and shape (tdim, num_tokens, 3). Per-template, per-token Cα coordinates (zero if unavailable).
            "template_mask_cb": torch.Tensor of dtype float32 and shape (tdim, num_tokens). Float mask indicating presence/validity of Cβ coordinates per template token (zeros here).
            "template_mask_frame": torch.Tensor of dtype float32 and shape (tdim, num_tokens). Float mask indicating validity of the frame (rotation/translation) per template token (zeros here).
            "template_mask": torch.Tensor of dtype float32 and shape (tdim, num_tokens). General per-template-token mask used by the model to indicate whether a template token exists (zeros here).
            "query_to_template": torch.Tensor of dtype int64 and shape (tdim, num_tokens). Integer mapping from query token indices to template token indices used for aligning query residues to template residues (zeros here by default).
            "visibility_ids": torch.Tensor of dtype float32 and shape (tdim, num_tokens). Float flags describing visibility or inclusion of template tokens for model attention/processing (zeros here).
    
    Behavior and side effects:
        This function performs only in-memory allocation of numpy arrays followed by conversion to torch Tensors and a one_hot conversion for residue types. It does not perform any I/O, network access, or modification of global model state. The output tensors are all zero-filled and intended as placeholders. The one_hot conversion uses const.num_tokens as the number of classes; if const.num_tokens is not set to a positive integer, the one_hot operation may raise an error. Large values for tdim or num_tokens may cause high memory usage and can raise numpy/torch allocation errors.
    
    Failure modes:
        Errors may be raised by numpy or torch when allocating very large arrays (MemoryError or torch allocation failures). The one_hot conversion may raise an exception if const.num_tokens is invalid (for example zero or negative) or if the one_hot helper function encounters unexpected input types. The function does not validate semantic consistency between num_tokens and const.num_tokens beyond using them for sizing and the one-hot class count respectively.
    """
    from boltz.data.feature.featurizerv2 import load_dummy_templates_features
    return load_dummy_templates_features(tdim, num_tokens)


################################################################################
# Source: boltz.data.feature.featurizerv2.sample_d
# File: boltz/data/feature/featurizerv2.py
# Category: valid
################################################################################

def boltz_data_feature_featurizerv2_sample_d(
    min_d: float,
    max_d: float,
    n_samples: int,
    random: numpy.random._generator.Generator
):
    """boltz.data.feature.featurizerv2.sample_d: Generate samples from a 1/d probability distribution on the interval [min_d, max_d]. This function is used by featurizerv2 in the Boltz codebase to produce distance-like scalar samples (for example, interatomic or inter-residue distance priors or augmentations) that follow an inverse-distance prior appropriate for scale-invariant sampling in biomolecular interaction and structure modeling tasks used by Boltz (binding affinity and structure prediction).
    
    Args:
        min_d (float): Minimum value of d. In the Boltz featurization context this represents the lower bound of the distance-like quantity being sampled (e.g., a small interatomic distance). Must be positive in practice because the 1/d density and inverse transform require min_d > 0. If min_d <= 0 the computation is invalid (can produce division-by-zero, NaNs, or runtime warnings).
        max_d (float): Maximum value of d. In the Boltz featurization context this represents the upper bound of the distance-like quantity being sampled (e.g., a maximum distance cutoff used for features). For a proper 1/d distribution the function assumes max_d > min_d; if max_d == min_d the function returns an array filled with min_d, and if max_d < min_d the results are mathematically invalid and may produce NaNs or raise errors.
        n_samples (int): Number of samples to generate. This is the number of independent draws from the 1/d distribution; the function returns an array of this length. n_samples must be a non-negative integer. Passing a negative integer will cause numpy.random.Generator.random to raise an error.
        random (numpy.random.Generator): Random number generator instance used to produce uniform draws in [0, 1). Providing a numpy.random.Generator allows reproducible sampling when the same generator state or seed is used. The generator's state is advanced (consumed) by this function, which is the only side effect on program state.
    
    Returns:
        numpy.ndarray: One-dimensional NumPy array of length n_samples containing samples drawn from the 1/d probability density f(d) = 1 / (d * ln(max_d/min_d)) for d in [min_d, max_d]. Values are floating-point numbers and are produced by applying the inverse CDF transform d = min_d * (max_d/min_d)**u where u ~ Uniform(0,1). If n_samples is zero an empty array is returned.
    
    Behavior, side effects, and failure modes:
        - Probability density and transform: The theoretical density implemented is f(d) = 1/(d * ln(max_d/min_d)) on [min_d, max_d]. The sampling uses inverse transform sampling via u = Uniform(0,1) and d = min_d * (max_d/min_d)**u.
        - Preconditions: For the density and inverse CDF to be mathematically valid, min_d > 0 and max_d > min_d. Violating these preconditions may lead to division-by-zero, infinite values, NaNs, or runtime errors from NumPy.
        - Reproducibility: Using a numpy.random.Generator lets callers control reproducibility via seeding. The function consumes generator state (advances the RNG).
        - No other side effects: The function does not modify input numeric values or global state other than consuming the RNG state.
        - Error propagation: If random is not a numpy.random.Generator (or does not implement random(n) with the same semantics), an AttributeError or TypeError may be raised by the call to random.random(n_samples).
    """
    from boltz.data.feature.featurizerv2 import sample_d
    return sample_d(min_d, max_d, n_samples, random)


################################################################################
# Source: boltz.data.feature.symmetry.convert_atom_name
# File: boltz/data/feature/symmetry.py
# Category: valid
################################################################################

def boltz_data_feature_symmetry_convert_atom_name(name: str):
    """boltz.data.feature.symmetry.convert_atom_name converts a biomolecular atom name string into a small, fixed-format numeric tuple used by Boltz feature pipelines for symmetry-aware atom encoding. This function is used when constructing input features for the Boltz models (e.g., encoding PDB/chemical atom names into a deterministic numeric representation that downstream embedding layers or symmetry features consume).
    
    Args:
        name (str): The atom name string to convert. In the Boltz codebase this is typically a PDB-style or chemical atom name (for example "CA", " N  ", "H1"). The function first strips leading and trailing ASCII whitespace from this string, then processes each remaining character in sequence. This parameter must be a Python str; passing a non-str will raise a TypeError when methods like strip() or ord() are applied.
    
    Returns:
        tuple[int, int, int, int]: A tuple of integers encoding the atom name characters for use in feature construction. Each integer is computed as ord(c) - 32 for a character c in the stripped name; this maps printable ASCII characters into small non-negative offsets (e.g., space -> 0). If the stripped name has fewer than four characters, the returned tuple is padded on the right with zeros to reach four elements (for example, "CA" -> (ord('C')-32, ord('A')-32, 0, 0)). If the stripped name is empty the function returns (0, 0, 0, 0). Note: the current implementation does not perform explicit truncation for names longer than four characters — such inputs will yield a tuple longer than four elements in practice (implementation detail); callers in the Boltz feature pipeline should use typical atom-name lengths (<=4) expected by PDB/chemical naming conventions.
    
    Behavior and side effects:
        The function is pure (no external side effects) and deterministic: the same input str always produces the same output tuple. It strips ASCII whitespace before encoding, so incidental padding in atom name fields (common in some PDB files) does not affect the canonical encoding. It does not perform validation of character set beyond using ord(), so non-ASCII or control characters will produce numeric values according to their Unicode code points minus 32 and may be unexpected for downstream consumers.
    
    Failure modes and practical considerations:
        Passing a non-str value will raise a TypeError when strip() is called. Passing names with unexpected Unicode characters or control codes can produce values outside the typical printable-ASCII range expected by downstream embedding tables; callers should normalize or validate atom name strings to standard PDB/chemical conventions where possible. The function is intended for short atom names commonly encountered in biomolecular structures and chemical representations; extremely long names are not truncated by the implementation and therefore may produce tuples longer than four elements (see implementation note above).
    """
    from boltz.data.feature.symmetry import convert_atom_name
    return convert_atom_name(name)


################################################################################
# Source: boltz.data.feature.symmetry.get_symmetries
# File: boltz/data/feature/symmetry.py
# Category: valid
################################################################################

def boltz_data_feature_symmetry_get_symmetries(path: str):
    """boltz.data.feature.symmetry.get_symmetries: Load and return ligand symmetry descriptors from a serialized ligand file used by the Boltz pipeline.
    
    This function opens a binary file at the given filesystem path and expects a pickled Python dictionary whose values are molecule objects that contain a "symmetries" property and per-atom "name" properties. It deserializes the top-level file with pickle.load, then for each molecule tries to read a hex-encoded pickled symmetry descriptor from the molecule property "symmetries", unhexlify and unpickle that descriptor, and collect a list of normalized atom names (using convert_atom_name) in the same order as the molecule's atom iteration. The result is a dictionary keyed by the original keys in the input file and valued by a two-tuple (sym, atom_names) where sym is the unpickled symmetry descriptor as stored in the file and atom_names is a list of strings naming the atoms. In the Boltz domain this mapping is used to preserve and apply ligand symmetry information during structure modeling and affinity prediction (for example, to identify equivalent atoms under ligand symmetry when comparing or aligning ligands).
    
    Args:
        path (str): Filesystem path to the serialized ligand symmetries file. The file must be a binary pickle that, when loaded, yields a mapping (dict) whose values are molecule-like objects exposing GetProp("symmetries") and GetAtoms()/GetProp("name") as used in the code. In practice for Boltz this file typically contains RDKit molecule objects with a "symmetries" property encoded as a hex string of a pickled symmetry object. The path parameter is required and must point to a trusted file location because the function deserializes data with pickle.
    
    Returns:
        dict: A dictionary of successfully loaded ligand symmetry entries. Each key is the same key present in the top-level input mapping. Each value is a tuple (sym, atom_names) where sym is the unpickled symmetry descriptor that was stored under the molecule's "symmetries" property and atom_names is a list of atom name strings produced by convert_atom_name() called in the order of mol.GetAtoms(). If a given entry in the input file cannot be processed (for example missing properties, malformed hex, or deserialization errors), that entry is silently skipped and not present in the returned dictionary. If no entries can be processed successfully the function returns an empty dict.
    
    Behavior, side effects, and failure modes:
        The function opens the file at path in binary read mode and calls pickle.load on its contents; this performs arbitrary-code deserialization and is unsafe on untrusted inputs. File-level errors such as FileNotFoundError, PermissionError, or pickle.UnpicklingError raised by the initial pickle.load are not caught inside the function and will propagate to the caller. For individual molecule entries, the function wraps per-entry processing in a broad try/except and will ignore any exceptions (for example missing GetProp, invalid hex in the "symmetries" property, or errors during pickle.loads of the per-molecule symmetry descriptor); those entries are skipped without raising. The returned atom_names list preserves the atom iteration order from mol.GetAtoms(), so callers that rely on atom ordering for downstream symmetry-aware operations should expect that ordering. This function has no other side effects (it does not modify the input file), but callers should only use it with trusted serialized files due to the use of pickle for deserialization.
    """
    from boltz.data.feature.symmetry import get_symmetries
    return get_symmetries(path)


################################################################################
# Source: boltz.data.mol.load_all_molecules
# File: boltz/data/mol.py
# Category: valid
################################################################################

def boltz_data_mol_load_all_molecules(moldir: str):
    """Load all pickled molecule objects from a directory into a name->Mol mapping.
    
    This function scans the filesystem directory specified by moldir for files with a ".pkl"
    extension, opens each file in binary mode, and deserializes its contents using Python's
    pickle.load. It is used in the Boltz codebase to populate an in-memory dictionary of
    rdkit.Chem.rdchem.Mol objects (molecular representations) that downstream Boltz
    inference and training code use for biomolecular interaction modeling and binding
    affinity prediction (e.g., ligand structure and affinity pipelines described in the
    project README). The returned mapping keys are the filename stems (filename without
    extension) and the values are the deserialized Mol objects expected by the Boltz
    pipelines.
    
    Args:
        moldir (str): Path to a directory on the local filesystem containing pickled
            RDKit molecule files with the ".pkl" extension. Each file is expected to be a
            pickle of an rdkit.Chem.rdchem.Mol instance that represents a molecule used
            by Boltz for tasks such as ligand structural modeling and affinity prediction.
            The function will only consider files matched by Path(moldir).glob("*.pkl").
            If moldir does not exist or contains no ".pkl" files, the function returns an
            empty dictionary. The function does not perform schema validation on the
            pickled contents; it relies on callers to ensure files contain the expected
            RDKit Mol objects.
    
    Returns:
        dict[str, rdkit.Chem.rdchem.Mol]: A dictionary mapping each loaded file's stem
        (filename without the ".pkl" suffix) to the deserialized rdkit.Chem.rdchem.Mol
        object. This mapping is used directly by Boltz's molecular pipelines for
        structure and affinity-related computations. If multiple files share the same
        filename stem, the later-loaded file will overwrite earlier entries (the final
        value for that key is the last one read). The function may return an empty dict
        if no matching files are found.
    
    Behavior, side effects, and failure modes:
        - Files enumerated by Path(moldir).glob("*.pkl") are loaded in the order returned
          by the filesystem listing; that order is not guaranteed to be stable across
          platforms. Duplicate stems will be overwritten as described above.
        - Each file is opened in binary mode and deserialized with pickle.load. Using
          pickle.load on untrusted data is a security risk; do not run this function on
          untrusted or unaudited files. Malicious pickles can execute arbitrary code
          during unpickling.
        - A tqdm progress bar is displayed (description "Loading molecules") while files
          are loaded; leave=False is used so the progress bar does not persist after
          completion. This produces console output as a side effect.
        - If a file is unreadable or the pickle is corrupted, pickle.UnpicklingError,
          EOFError, or OSError/FileNotFoundError may be raised during loading; these
          exceptions propagate to the caller. If a loaded object is not an RDKit Mol, the
          function will still return it (type annotation is rdkit.Chem.rdchem.Mol), but
          downstream Boltz code that expects a Mol may raise type or attribute errors.
        - Performance: loading many or large pickles is I/O-bound and may be slow; callers
          should consider batching or optimizing storage format if load latency matters.
    """
    from boltz.data.mol import load_all_molecules
    return load_all_molecules(moldir)


################################################################################
# Source: boltz.data.mol.load_canonicals
# File: boltz/data/mol.py
# Category: valid
################################################################################

def boltz_data_mol_load_canonicals(moldir: str):
    """Load canonicalized molecule objects from a directory for use by Boltz model pipelines.
    
    This function reads molecule files from the directory specified by moldir and constructs a dictionary of canonicalized RDKit molecule objects using the Boltz canonical token configuration. It is used by Boltz preprocessing and inference pipelines (for example, preparing ligand representations for binding affinity prediction and structural modeling described in the repository README). Internally this function delegates to load_molecules(moldir, const.canonical_tokens), so the exact canonicalization behavior follows the repository's canonical_tokens settings.
    
    Args:
        moldir (str): Filesystem path to a directory containing molecule input files to load. In the Boltz context this directory should contain the molecule dataset or exported molecule files that the model will consume for tasks such as affinity prediction and structure modeling. The function will read files from this path; moldir must be a valid path accessible to the process. Providing an empty or non-existent directory will cause the underlying loader or file I/O to raise an error.
    
    Returns:
        dict[str, rdkit.Chem.rdchem.Mol]: A mapping from string identifiers to RDKit Mol objects for each successfully loaded molecule. The string keys are the identifiers produced by the underlying load_molecules implementation (typically the per-file or per-record identifiers used when the molecules were exported). The RDKit Mol values are canonicalized according to the Boltz canonical_tokens configuration and are suitable for downstream tasks in the Boltz codebase (e.g., constructing molecular graphs for model inputs and computing descriptors for affinity prediction).
    
    Behavior and side effects:
        This function performs read-only I/O on the filesystem path given by moldir and constructs in-memory RDKit Mol objects; it does not modify files on disk. Canonicalization is applied via the repository's canonical_tokens setting. The function's behavior and exact key naming convention are determined by the underlying load_molecules implementation.
    
    Failure modes:
        If moldir does not exist, is not a directory, or is not readable, the underlying loader or Python file I/O will raise an exception (e.g., FileNotFoundError or an OS error). If RDKit fails to parse a molecule file, RDKit or the loader may raise parsing errors; depending on the underlying loader implementation, malformed records may either be omitted from the returned dictionary or cause an exception to be raised. Callers should handle these exceptions as appropriate for their Boltz preprocessing or inference workflow.
    """
    from boltz.data.mol import load_canonicals
    return load_canonicals(moldir)


################################################################################
# Source: boltz.data.mol.load_molecules
# File: boltz/data/mol.py
# Category: valid
################################################################################

def boltz_data_mol_load_molecules(moldir: str, molecules: list[str]):
    """Load pickled RDKit molecule objects from a directory for use by Boltz biomolecular modeling.
    
    This function reads binary pickle files named "<molecule>.pkl" from a filesystem directory and deserializes them into RDKit molecule objects. In the Boltz pipeline these deserialized objects typically represent CCD/ligand components used when constructing inputs for structure prediction and binding affinity prediction (e.g., affinity_pred_value and affinity_probability_binary described in the project README). The function performs no writing; it opens each file in binary read mode and returns all successfully loaded molecules in memory as a dictionary keyed by the molecule name.
    
    Args:
        moldir (str): The path to the molecules directory. This is joined with each molecule name to form the expected filename Path(moldir) / f"{molecule}.pkl". The directory must be readable by the process and contain the requested .pkl files. In practice this directory holds pre-serialized RDKit molecule objects created by the project's preprocessing or component library and is required by downstream Boltz inference and evaluation code.
        molecules (list[str]): The molecules to load. Each entry is the base filename (without extension) to load from moldir; for example, an entry "LIG1" causes the function to look for "LIG1.pkl" inside moldir. The order of names controls which files are accessed but does not affect the structure of the returned mapping. All names must be trusted and correspond to existing pickle files containing RDKit Mol objects for correct downstream behavior.
    
    Returns:
        dict[str, rdkit.Chem.rdchem.Mol]: A dictionary mapping each requested molecule name (the same strings provided in the molecules parameter) to the deserialized rdkit.Chem.rdchem.Mol object loaded from the corresponding "<name>.pkl" file. These RDKit Mol objects represent molecular structures used by Boltz for interaction and affinity modeling and are returned fully loaded into memory; callers should be prepared for memory proportional to the total size of the pickled objects.
    
    Behavior, side effects, and failure modes:
        - The function constructs filenames by appending ".pkl" to each molecule name and joining with moldir using pathlib.Path semantics; it opens each file in binary read mode.
        - If any expected file does not exist, the function raises ValueError with the message "CCD component {molecule} not found!" for the first missing molecule and does not return a partial result.
        - If a file exists but cannot be opened due to filesystem permissions or I/O errors, an OSError (or subclass) will propagate to the caller.
        - If pickle deserialization fails (for example due to incompatible or corrupted data), the underlying pickle.UnpicklingError or another exception raised by pickle.load will propagate.
        - The function uses pickle.load for deserialization; loading untrusted pickle files is a security risk. Only load .pkl files produced by trusted preprocessing steps in this project or other trusted sources.
        - The function does not validate that the unpickled object is an rdkit.Chem.rdchem.Mol; if a different object was serialized under the same filename, that object will be returned as-is and may cause failures later in Boltz inference pipelines.
        - All loaded molecules are kept in memory and returned; there is no streaming or lazy-loading.
    """
    from boltz.data.mol import load_molecules
    return load_molecules(moldir, molecules)


################################################################################
# Source: boltz.data.parse.mmcif.compute_interfaces
# File: boltz/data/parse/mmcif.py
# Category: valid
################################################################################

def boltz_data_parse_mmcif_compute_interfaces(
    atom_data: numpy.ndarray,
    chain_data: numpy.ndarray
):
    """boltz.data.parse.mmcif.compute_interfaces computes chain-chain interfaces from a gemmi-derived structure and returns unique pairs of chain indices that are within a geometric contact cutoff. This function is used by Boltz to identify which polymer chains in a biomolecular complex are in spatial contact (interfaces) so downstream modules can focus structural and affinity modeling on interacting chain pairs. The implementation maps atoms to chain indices using chain_data, filters to atoms marked present, builds a KDTree over atom coordinates, queries neighbors within const.atom_interface_cutoff, and collects unique, ordered chain index pairs.
    
    Args:
        atom_data (numpy.ndarray): Per-atom structured array extracted from a gemmi structure. This array must provide at least two named fields accessed by the code: "coords" and "is_present". "coords" is expected to be an array-like of Cartesian coordinates for each atom (used to build the KDTree and compute spatial neighborhoods). "is_present" is a boolean mask indicating which atoms to include in the interface computation (atoms with False are omitted). In the Boltz domain, atom_data represents the atomic coordinates and presence flags for a molecular assembly and is essential to determine which atoms participate in inter-chain contacts used for interaction prediction and affinity modeling.
        chain_data (numpy.ndarray): Per-chain structured array where each entry corresponds to a chain in the same gemmi-derived structure. Each chain entry must provide at least the integer field "atom_num", which is used to assign consecutive atoms in atom_data to chain indices (chain index i is repeated chain_data[i]["atom_num"] times). In practice within Boltz, chain_data encodes chain lengths so compute_interfaces can reconstruct which atoms belong to which chain and thereby determine chain-chain contacts critical for predicting complex interfaces and binding sites.
    
    Returns:
        numpy.ndarray: A NumPy array of unique chain-index pairs representing detected interfaces. Each element is a 2-tuple-like record (i, j) with integer chain indices, normalized so i <= j (the function produces (min, max) for each pair) and duplicates removed. The array is constructed with the code's Interface dtype (an array dtype used internally by Boltz). The returned pairs enumerate chain indices that have at least one pair of atoms within the geometric cutoff defined by const.atom_interface_cutoff; these pairs are intended for downstream structural processing and binding-affinity operations.
    
    Behavior, defaults, and failure modes:
        The function derives a per-atom chain index list by repeating chain indices according to chain_data["atom_num"], then filters atoms using atom_data["is_present"], and queries a KDTree built from atom_data["coords"] to find neighbors within const.atom_interface_cutoff. It excludes self-chain contacts and returns only unique unordered pairs. There are no external side effects (the function does not modify input arrays or global state) and it is essentially pure given valid inputs.
        Known failure modes include: KeyError or IndexError if atom_data does not contain the "coords" or "is_present" fields, or if chain_data entries lack "atom_num"; ValueError or shape-related errors if the total number of atoms implied by chain_data["atom_num"] does not match the length of atom_data, or if coords has an unexpected dimensionality; KDTree-related errors if coords is empty or has invalid numeric types. Performance scales with the number of present atoms because a spatial KDTree query is performed; const.atom_interface_cutoff controls neighborhood radius and therefore affects the density of neighbor lists and runtime. Users should validate that atom_data and chain_data originate from the same gemmi structure and maintain consistent ordering and counts before calling this function.
    """
    from boltz.data.parse.mmcif import compute_interfaces
    return compute_interfaces(atom_data, chain_data)


################################################################################
# Source: boltz.data.parse.mmcif.get_mol
# File: boltz/data/parse/mmcif.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for get_mol because the docstring has no description for the argument 'ccd'
################################################################################

def boltz_data_parse_mmcif_get_mol(ccd: str, mols: dict, moldir: str):
    """boltz.data.parse.mmcif.get_mol: Retrieve and (if needed) load an RDKit molecule for a given Chemical Component Dictionary (CCD) code and cache it in the provided molecule mapping.
    
    This function is used when parsing mmCIF files and assembling ligand definitions for Boltz models (biomolecular interaction and affinity prediction). It first attempts to return an already-cached molecule for the CCD code from the provided mapping. If the molecule is absent, it loads the molecule definition from disk using load_molecules(moldir, [ccd]), inserts the loaded molecule into the provided mapping (mutating it), and then returns the loaded RDKit object. This ensures downstream code that constructs features for ligands (used in Boltz structural and affinity predictions) has a consistent RDKit Mol instance for the CCD entry.
    
    Args:
        ccd (str): Chemical Component Dictionary identifier for the molecule to retrieve. In the Boltz/mmCIF domain this is the CCD code used in mmCIF ligand/chemical_component records (for example three-letter ligand codes). This string is used as the key to look up a cached molecule in mols and as the identifier passed to load_molecules when the molecule must be loaded from moldir.
        mols (dict): A mutable mapping acting as a cache of previously loaded molecules, keyed by CCD code, with values that are RDKit Mol objects (rdkit.Chem.rdchem.Mol). The function expects mols.get(ccd) to return an existing Mol if present. Although annotated as dict, the implementation also supports objects that provide a set(ccd, mol) method: if mols is not a plain dict, the function will call mols.set(ccd, mol) to add the loaded molecule. The mapping will be mutated when a molecule is loaded to store the new entry for future retrievals.
        moldir (str): Filesystem directory path where molecule definition files are stored and where load_molecules will look for molecule data. This string is forwarded to load_molecules to locate the on-disk representation of the CCD entry when a cache miss occurs.
    
    Returns:
        rdkit.Chem.rdchem.Mol: The RDKit Mol object corresponding to the requested CCD code. If the molecule was already present in mols, that cached object is returned unchanged. If not present, the returned Mol is loaded from moldir, inserted into mols (mutating it via dict assignment or mols.set), and then returned.
    
    Behavior, side effects, and failure modes:
        On a cache hit (mols contains ccd), the function performs no I/O and returns the cached RDKit Mol.
        On a cache miss, the function calls load_molecules(moldir, [ccd]) and indexes the result with [ccd] to obtain the Mol; the loaded Mol is then stored into mols (either mols[ccd] = mol or mols.set(ccd, mol)) and returned.
        The function mutates the provided mols mapping on cache misses to ensure subsequent calls reuse the loaded object; callers relying on mols immutability should pass a copy if mutation is undesired.
        Exceptions raised by load_molecules (for example if moldir is invalid, files are missing, or the CCD entry cannot be found) are propagated to the caller. In particular, a KeyError may occur if the loaded mapping does not contain the requested CCD key, and file access errors from the filesystem or parsing errors from the molecule loader may also be raised. The function does not catch these exceptions internally.
    """
    from boltz.data.parse.mmcif import get_mol
    return get_mol(ccd, mols, moldir)


################################################################################
# Source: boltz.data.parse.mmcif_with_constraints.compute_interfaces
# File: boltz/data/parse/mmcif_with_constraints.py
# Category: valid
################################################################################

def boltz_data_parse_mmcif_with_constraints_compute_interfaces(
    atom_data: numpy.ndarray,
    chain_data: numpy.ndarray
):
    """Compute chain-chain interfaces from a gemmi-derived structure for use in Boltz biomolecular interaction modeling.
    
    This function identifies which chains in a structure are in contact by examining atom coordinates and a presence mask produced from a gemmi parsing step. It is used in the Boltz preprocessing pipeline to determine chain-chain interfaces that downstream components use for tasks such as complex assembly, structural scoring, and binding affinity prediction described in the Boltz README. The algorithm assigns each atom to a chain using chain_data["atom_num"], filters to atoms marked present in atom_data["is_present"], builds a KDTree over present atom coordinates (atom_data["coords"]), and flags chain pairs that have any pair of atoms within const.atom_interface_cutoff. The returned interfaces are unique, ordered integer pairs and are suitable for downstream modules that expect an array of chain index pairs.
    
    Args:
        atom_data (numpy.ndarray): Structured numpy array containing per-atom information produced from gemmi. This array must include at least the fields "coords" (an array of 3D coordinates for each atom) and "is_present" (boolean mask marking atoms to consider). The function reads atom_data["coords"] and atom_data["is_present"] to compute distances and filter atoms. In the Boltz domain, atom_data represents the atomic-level geometry of biomolecules parsed from mmCIF files and is required so interfaces reflect physically present atoms only.
        chain_data (numpy.ndarray): Structured numpy array or sequence of mappings describing chain-level metadata produced from gemmi. Each entry must include an integer field "atom_num" that gives the number of atoms assigned to that chain in atom_data; chain_data is used to compute a chain index for each atom by repeating the chain index atom_num times in order. In the Boltz pipeline, chain_data associates contiguous slices of atom_data with chain indices so the function can map atoms back to chains.
    
    Returns:
        numpy.ndarray: A numpy.ndarray of integer pairs describing unique chain-chain interfaces discovered in the input. Each element is a length-2 tuple or 1D array (chain_i, chain_j) with Python int values (dtype matches the module's Interface dtype). Pairs are normalized such that chain_i < chain_j (min, max) and self-contacts (chain_i == chain_j) are excluded. The array contains no duplicate pairs; if no interfaces are found an empty array with shape (0, 2) and dtype Interface is returned. This return value is intended for downstream Boltz components that consume chain index pairs to focus computations (for example, per-interface modeling or affinity prediction).
    
    Behavior, side effects, and failure modes:
        - The function does not modify atom_data or chain_data in-place; it constructs and returns a new numpy.ndarray.
        - The function uses sklearn.neighbors.KDTree (via KDTree) with Euclidean metric and a cutoff distance const.atom_interface_cutoff to determine atom-atom proximity. The cutoff value is read from the module constant and therefore behavior depends on that constant being defined.
        - If atom_data lacks the required fields "coords" or "is_present", a KeyError or IndexError may be raised when accessing those fields.
        - If the sum of chain_data["atom_num"] does not match the length of atom_data (before masking), the mapping from atoms to chains will be inconsistent; this can lead to incorrect interfaces or raise an error depending on underlying numpy behavior.
        - The function excludes atoms where atom_data["is_present"] is False; such atoms do not contribute to detected interfaces.
        - Performance and memory use scale with the number of present atoms; very large structures may increase runtime and memory usage due to KDTree construction and radius queries.
        - The function relies on KDTree.query_radius behavior and numpy set/unique semantics; unexpected object types in inputs may produce TypeError or ValueError.
    """
    from boltz.data.parse.mmcif_with_constraints import compute_interfaces
    return compute_interfaces(atom_data, chain_data)


################################################################################
# Source: boltz.data.parse.mmcif_with_constraints.get_mol
# File: boltz/data/parse/mmcif_with_constraints.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for get_mol because the docstring has no description for the argument 'ccd'
################################################################################

def boltz_data_parse_mmcif_with_constraints_get_mol(ccd: str, mols: dict, moldir: str):
    """Get or load an RDKit Mol for a given CCD code, using an in-memory cache or loading from a local molecule directory.
    
    This function is used by the mmCIF-with-constraints parser in the Boltz codebase to resolve small-molecule components referenced by CCD (Chemical Component Dictionary) codes into RDKit Mol objects. These Mol objects are the molecule representations used downstream by Boltz for structural modeling and binding-affinity prediction (featurization, geometry handling, and chemistry-aware operations). If the requested CCD code is already present in the provided molecule cache/resource, the cached Mol is returned immediately; otherwise the function loads the molecule definition from files found under moldir, inserts the loaded Mol into the provided cache/resource, and returns it.
    
    Args:
        ccd (str): CCD code that identifies the small-molecule component to retrieve. This is the same short identifier used in mmCIF/PDB files (for example, "HEM" for heme). The function uses this code to look up a cached rdkit.Chem.rdchem.Mol or to locate and load the corresponding molecule file from moldir when not cached.
        mols (dict): A mutable mapping from CCD codes (str) to rdkit.Chem.rdchem.Mol objects that acts as a cache/resource of previously loaded molecules. The function calls mols.get(ccd) to check for an existing entry. If the entry is missing, the function will add the loaded Mol back into this resource: if mols is a plain dict it sets mols[ccd] = mol; if mols is a dict-like resource object that implements .set(ccd, mol), the function will call mols.set(ccd, mol). This in-place update is a side effect used to avoid reloading the same molecule multiple times during parsing and inference.
        moldir (str): Filesystem path to the directory containing molecule definition files (the local repository of component definitions that load_molecules expects). When the CCD code is not found in mols, the function delegates to load_molecules(moldir, [ccd]) and expects that call to return a mapping containing the requested CCD key. The directory should contain the molecule files (e.g., SDF/mmCIF or other formats that the project’s load_molecules supports) required to construct an rdkit.Chem.rdchem.Mol for the CCD code.
    
    Returns:
        rdkit.Chem.rdchem.Mol: An RDKit Mol object corresponding to the requested CCD code. This Mol is suitable for downstream structural and chemical processing in Boltz (e.g., conversion to features used by the model, coordinate handling, or affinity-related calculations). The returned object is either the cached Mol from mols or the newly loaded Mol that was inserted into mols as a side effect.
    
    Raises:
        KeyError: If the CCD code is not present in the result returned by load_molecules(moldir, [ccd]) when a load is attempted, a KeyError may be raised when indexing that result by ccd.
        FileNotFoundError or OSError: If moldir does not exist or is inaccessible, the underlying load_molecules call may raise filesystem-related errors.
        Exception: RDKit parsing or other I/O/parsing errors raised by load_molecules or RDKit may propagate; callers should handle or surface these errors as appropriate for their parsing/inference workflow.
    
    Behavior and side effects:
        - The function first queries mols.get(ccd). If a non-None value is returned, no I/O occurs and that value is returned immediately.
        - If the value is missing or None, the function calls load_molecules(moldir, [ccd]) to obtain the molecule definition, extracts the Mol for ccd, and then stores it back into the provided mols resource (either by assignment mols[ccd] = mol for plain dicts or by calling mols.set(ccd, mol) for dict-like resource objects). This mutation is intentional to cache loaded molecules across multiple parses and speed up Boltz inference.
        - The function is not synchronized for concurrent access; simultaneous calls with the same mols resource from multiple threads/processes may lead to redundant loads or race conditions unless the caller provides external synchronization.
    """
    from boltz.data.parse.mmcif_with_constraints import get_mol
    return get_mol(ccd, mols, moldir)


################################################################################
# Source: boltz.data.parse.schema.convert_atom_name
# File: boltz/data/parse/schema.py
# Category: valid
################################################################################

def boltz_data_parse_schema_convert_atom_name(name: str):
    """boltz.data.parse.schema.convert_atom_name converts an atom name string into a fixed-length integer encoding used by Boltz parsing and featurization pipelines for biomolecular structures. This canonicalization is used when parsing input YAMLs or PDB-style records to produce a compact, comparable representation of atom names for downstream model inputs, table keys, or embedding lookups in Boltz's biomolecular interaction prediction workflows.
    
    Args:
        name (str): The raw atom name as extracted from a parsed structure record or an input YAML describing a biomolecule. Leading and trailing whitespace is significant in many PDB-style formats and is removed by this function (name.strip()) before encoding. Typical sources are PDB/ligand atom name fields or user-provided atom identifiers in Boltz prediction inputs; the function expects a short textual atom identifier (commonly up to four characters).
    
    Returns:
        tuple[int, int, int, int]: A tuple of integers representing the encoded atom name. Encoding is performed by mapping each character c in the stripped name to ord(c) - 32 and then padding with zeros on the right until four values are present. This produces a canonical fixed-size integer vector intended for use by Boltz preprocessing, hashing, and embedding lookup. For example, an empty or all-whitespace input yields (0, 0, 0, 0). Note: the implementation pads shorter names with zeros but does not truncate names longer than four characters; if the input name contains more than four characters the function will return a tuple longer than four elements (reflecting the raw character encodings) which therefore deviates from the annotated 4-tuple contract. Also note that characters with Unicode code points less than 32 will produce negative integers because of the ord(c) - 32 mapping. The function is pure (no side effects) and deterministic. Failure modes to be aware of: unexpected negative codes from non-printable characters, and potential length mismatch if callers assume a strict 4-element tuple for all inputs.
    """
    from boltz.data.parse.schema import convert_atom_name
    return convert_atom_name(name)


################################################################################
# Source: boltz.data.parse.schema.get_global_alignment_score
# File: boltz/data/parse/schema.py
# Category: valid
################################################################################

def boltz_data_parse_schema_get_global_alignment_score(query: str, template: str):
    """Compute the global pairwise alignment score between a query sequence and a template sequence.
    
    This function boltz.data.parse.schema.get_global_alignment_score is used in the Boltz data parsing and template-matching pipeline to quantify overall sequence similarity between two biomolecular sequences (typically protein amino-acid sequences). It constructs a Biopython PairwiseAligner with scoring="blastp" and sets aligner.mode = "global", then returns the score of the top (first) alignment. In the Boltz context this score is useful for template selection, input validation, or any downstream logic that requires a single scalar measure of how well a query sequence matches a template sequence when aligned end-to-end.
    
    Args:
        query (str): The query sequence to align. In Boltz usage this is typically a protein sequence represented as an ASCII string of amino-acid one-letter codes. The function treats the string as a biological sequence for alignment; invalid characters, extremely long strings, or empty strings may trigger errors from the underlying PairwiseAligner.
        template (str): The template sequence to align against. As with query, this is typically a protein sequence string (amino-acid one-letter codes) used as the reference or template in Boltz parsing and template-matching steps.
    
    Behavior and side effects:
        The function creates a Bio.Align.PairwiseAligner with scoring="blastp" and sets mode to "global", then computes aligner.align(query, template)[0].score and returns that float. The returned score is produced by the aligner using the BLASTP substitution matrix and the PairwiseAligner default gap scoring behavior; the numeric value is therefore in the aligner's score units and is comparable only to scores computed with the same scoring settings. There are no persistent side effects (no global state is modified), and the function is deterministic given identical inputs and the same Biopython version and scoring defaults. Performance and memory use scale with the lengths of query and template according to the PairwiseAligner implementation (global alignment has higher cost for very long sequences).
    
    Failure modes and defaults:
        The function relies on the underlying Biopython PairwiseAligner implementation. If the aligner cannot process the provided strings (for example due to invalid characters, empty sequences, or other input issues), exceptions from PairwiseAligner will propagate to the caller. The scoring matrix is fixed to "blastp" and the alignment mode to "global" by this function; these defaults are chosen to produce end-to-end protein alignments consistent with BLASTP-style substitution scoring.
    
    Returns:
        float: The global alignment score for the best alignment between query and template as computed by Bio.Align.PairwiseAligner with scoring="blastp" and mode="global". Higher values indicate better overall (end-to-end) similarity under the chosen scoring scheme; this scalar is intended for use in Boltz parsing, template selection, or other downstream decisions that require a single measure of sequence-template similarity.
    """
    from boltz.data.parse.schema import get_global_alignment_score
    return get_global_alignment_score(query, template)


################################################################################
# Source: boltz.data.parse.schema.get_local_alignments
# File: boltz/data/parse/schema.py
# Category: valid
################################################################################

def boltz_data_parse_schema_get_local_alignments(query: str, template: str):
    """boltz.data.parse.schema.get_local_alignments: Align a query sequence to a template and return one or more local alignment mappings as Alignment objects. This function is used in the Boltz preprocessing and inference pipeline to map residues between an input sequence (query) and a template sequence for downstream structural modeling and binding-affinity prediction tasks in biomolecular interaction prediction.
    
    Args:
        query (str): The query sequence to align. In the Boltz context this is typically a biomolecular sequence (for example a protein sequence) provided in prediction inputs; it must be a Python string. The function passes this string to Bio.Align.PairwiseAligner and uses it as the first operand in the pairwise alignment.
        template (str): The template sequence to align against. In Boltz workflows this is typically a reference or template biomolecular sequence (for example a homologous protein sequence used for template-based modeling); it must be a Python string. The function passes this string to Bio.Align.PairwiseAligner and uses it as the second operand in the pairwise alignment.
    
    Behavior and implementation details:
        This function constructs a Bio.Align.PairwiseAligner with scoring="blastp", sets the aligner to local mode, and sets open_gap_score and extend_gap_score to -1000 to effectively disallow insertion/deletion gaps in reported alignments. It then iterates over aligner.align(query, template) and for each alignment result reads result.coordinates to obtain the aligned regions. For each result it creates a boltz.data.parse.schema.Alignment object with integer fields query_st, query_en, template_st, and template_en taken directly from the coordinate pairs and appends it to the result list. The integer indices are suitable for indexing or slicing the original Python strings to extract the aligned substrings. The function performs no I/O and has no side effects beyond allocating and returning the list of Alignment objects.
    
    Failure modes and edge cases:
        If no local alignment is found between the provided sequences, the function returns an empty list. The function relies on the Biopython PairwiseAligner implementation and may raise exceptions propagated from that library for invalid input types (for example non-string inputs) or invalid sequence contents. Because gap penalties are set to a very large negative value, reported alignments will effectively contain no internal gaps; this is an intentional behavior to produce contiguous aligned segments.
    
    Returns:
        list[boltz.data.parse.schema.Alignment]: A list of Alignment objects describing each local alignment between query and template. Each Alignment has integer attributes query_st, query_en, template_st, and template_en that mark the start and end positions of the aligned region in the original query and template sequences, respectively. The list preserves the iteration order returned by Bio.Align.PairwiseAligner; it may be empty if no alignments are found.
    """
    from boltz.data.parse.schema import get_local_alignments
    return get_local_alignments(query, template)


################################################################################
# Source: boltz.data.parse.schema.get_mol
# File: boltz/data/parse/schema.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for get_mol because the docstring has no description for the argument 'ccd'
################################################################################

def boltz_data_parse_schema_get_mol(ccd: str, mols: dict, moldir: str):
    """Get mol for a chemical component (CCD) code.
    
    Look up and return an RDKit molecule corresponding to a chemical component identifier (CCD code). This utility is used by the Boltz pipeline to obtain a ligand or small-molecule representation needed for downstream biomolecular interaction modeling and binding affinity prediction. It first checks an in-memory mapping of already-loaded molecules (mols) and, if absent, invokes load_molecules to load the requested CCD entry from persistent storage (moldir) and returns the loaded molecule for use in structural modeling and affinity calculations.
    
    Args:
        ccd (str): Chemical component identifier (CCD code) for the molecule to retrieve. This string must match the keys used in the mols mapping and the identifiers understood by load_molecules; matching is exact and case-sensitive as used by those data sources. In the Boltz domain, this typically refers to a PDB chemical component code for a small molecule or ligand needed for structure/affinity prediction.
        mols (dict): In-memory dictionary mapping CCD codes (str) to RDKit molecule objects (rdkit.Chem.rdchem.Mol). This function will query this mapping with mols.get(ccd) to return a previously loaded molecule if available. The function does not modify this dictionary itself; it performs a read-only lookup and returns the found value unmodified.
        moldir (str): Filesystem directory path where molecule files or resources are stored for loading missing CCD entries. If the CCD is not present in mols, the function delegates to load_molecules(moldir, [ccd]) to load the molecule from this directory and then returns the result. moldir should point to the dataset or chemical component directory used by the Boltz data parsing pipeline.
    
    Returns:
        rdkit.Chem.rdchem.Mol: An RDKit Mol object representing the requested chemical component. This object encodes the molecular graph and associated chemistry used by Boltz for structure modeling and affinity prediction.
    
    Behavior, side effects, and failure modes:
        - Primary behavior: return mols[ccd] if present; otherwise call load_molecules(moldir, [ccd]) and return the value at [ccd] from that result.
        - Side effects: the function itself does not mutate the mols dictionary. It may trigger file I/O and parsing inside load_molecules when loading from moldir.
        - Errors and exceptions: if load_molecules does not produce an entry for the requested CCD, indexing the returned mapping with [ccd] will raise a KeyError. File-system errors (e.g., missing moldir, permission errors) or parsing errors from the molecule loader (e.g., RDKit parsing failures) raised by load_molecules will propagate to the caller. Callers should handle KeyError and any IO/parse exceptions when using this function.
        - Performance note: using a populated mols mapping avoids the overhead of loading from disk; clients of the Boltz prediction pipeline should populate mols with frequently used CCDs to reduce repeated I/O and parsing.
    """
    from boltz.data.parse.schema import get_mol
    return get_mol(ccd, mols, moldir)


################################################################################
# Source: boltz.data.parse.schema.standardize
# File: boltz/data/parse/schema.py
# Category: fix_docstring
# Reason: Schema parsing failed: Cannot generate JSON schema for standardize because the docstring has no description for the argument 'smiles'
################################################################################

def boltz_data_parse_schema_standardize(smiles: str):
    """boltz.data.parse.schema.standardize standardizes a small-molecule SMILES string for use in the Boltz data and inference pipelines (for example, affinity prediction and structure modeling). It parses the input with RDKit (deferring RDKit sanitization), applies an exclusion check, chooses the largest connected fragment, runs a ChEMBL-style molecule standardization pipeline, and returns a canonicalized SMILES suitable for downstream Boltz preprocessing and prediction.
    
    Args:
        smiles (str): SMILES string representing the input small molecule (ligand). This string is parsed with RDKit using sanitize=False so initial RDKit sanitization is deferred; the function then performs an exclusion test (via exclude_flag with includeRDKitSanitization=False), selects the largest molecular fragment using rdMolStandardize.LargestFragmentChooser, and applies the ChEMBL-style standardize_mol pipeline implemented in the codebase. The parameter is required and must be provided as a Python str containing a SMILES representation of the molecule intended for Boltz workflows (e.g., hit discovery or ligand optimization stages described in the README).
    
    Returns:
        Optional[str]: The standardized SMILES string (Python str) produced by Chem.MolToSmiles after ChEMBL-style standardization. This standardized SMILES is the canonicalized representation intended for downstream Boltz components such as affinity prediction or structural modeling and should be stable across runs for a given chemically valid input. Although the function signature is Optional[str], in the current implementation a non-None str is returned on success; callers should be prepared for exceptions described below rather than relying on None to signal failure.
    
    Behavior, side effects, defaults, and failure modes:
        The function follows these steps in order: (1) parse the input SMILES with Chem.MolFromSmiles(smiles, sanitize=False), (2) call exclude_flag(mol, includeRDKitSanitization=False) to determine whether the molecule should be excluded from processing, (3) if excluded, raise ValueError("Molecule is excluded"), (4) choose the largest connected component with rdMolStandardize.LargestFragmentChooser().choose(mol), (5) apply the ChEMBL data curation / standardization pipeline via standardize_mol(mol), (6) convert the standardized RDKit Mol back to a SMILES string using Chem.MolToSmiles(mol), and (7) verify that the resulting SMILES can be parsed by RDKit (Chem.MolFromSmiles(smiles)). If the final SMILES cannot be parsed, the function raises ValueError("Molecule is broken"). The function intentionally defers RDKit sanitization at initial parse and uses includeRDKitSanitization=False when calling the exclusion test; this behavior is part of the curated standardization flow and may allow initially non-sanitized inputs to be processed by the ChEMBL-style pipeline.
    
        Side effects: the function does not perform I/O and does not mutate global state beyond typical RDKit object allocations. It may raise ValueError for excluded or broken molecules. RDKit errors or exceptions raised by standardize_mol, Chem.MolToSmiles, or other underlying RDKit calls may propagate; callers should handle ValueError and other RDKit exceptions as appropriate in data preprocessing pipelines. This implementation differs from the original mol-finder/data variant by explicitly raising exceptions on exclusion or broken molecules to avoid silently returning invalid representations; the behavior is intended to make molecule preprocessing safer and more explicit for downstream Boltz model inputs.
    """
    from boltz.data.parse.schema import standardize
    return standardize(smiles)


################################################################################
# Source: boltz.data.tokenize.boltz2.compute_frame
# File: boltz/data/tokenize/boltz2.py
# Category: valid
################################################################################

def boltz_data_tokenize_boltz2_compute_frame(
    n: numpy.ndarray,
    ca: numpy.ndarray,
    c: numpy.ndarray
):
    """Compute a right-handed orthonormal local frame for a single amino-acid residue from backbone atom coordinates.
    
    Args:
        n (numpy.ndarray): 3D Cartesian coordinates of the backbone N atom for the residue. In the Boltz-2 tokenization and preprocessing pipeline this vector defines the position of the peptide backbone nitrogen and is used to determine the residue's orientation relative to the alpha carbon (CA). The function subtracts the CA coordinate from this vector (v2 = n - ca) to form the second directional component before orthogonalization.
        ca (numpy.ndarray): 3D Cartesian coordinates of the backbone CA (alpha carbon) atom for the residue. In this function ca serves as the origin/translation of the local frame (t = ca) and as the reference point from which the two direction vectors v1 and v2 are formed. In the Boltz data/tokenize step the CA is typically the per-residue anchor used to express local coordinates for model inputs.
        c (numpy.ndarray): 3D Cartesian coordinates of the backbone C (carbonyl carbon) atom for the residue. The vector from CA to C (v1 = c - ca) is used as the primary axis (e1) of the local frame; this axis captures the direction of the peptide bond and is important for defining a consistent residue-aligned basis used by Boltz-2 for structural representation.
    
    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: A tuple (rot, t) containing:
            rot (numpy.ndarray): A 3x3 rotation matrix whose columns are the orthonormal basis vectors [e1, e2, e3] computed for the residue. e1 is the normalized CA->C vector, e2 is the CA->N vector with its component along e1 removed and then normalized, and e3 is the cross product e1 x e2, producing a right-handed orthonormal basis. This rotation matrix is used by the Boltz tokenization and model preprocessing to rotate coordinates into a residue-centric coordinate system for invariant feature construction and relative pose encoding.
            t (numpy.ndarray): A 3-element translation vector equal to the CA coordinates passed in (ca). Together with rot, t defines an affine transform that maps global Cartesian coordinates into the local residue frame centered on CA.
    
    Behavior and numerical details:
        The function computes v1 = c - ca and v2 = n - ca, then normalizes v1 to produce e1. It orthogonalizes v2 against e1 by subtracting the projection of v2 onto e1 to produce u2, normalizes u2 to produce e2, and computes e3 = cross(e1, e2). The rotation matrix rot is constructed by column-stacking [e1, e2, e3] and the translation t is set to ca.
        To avoid division-by-zero, the implementation adds a small epsilon (1e-10) to norms during normalization; this prevents exceptions for degenerate inputs but cannot recover a well-defined orientation when atoms are exactly collinear or extremely close together.
    
    Failure modes and validation:
        If the three input coordinates are collinear or if u2 is (near) zero length after projection, the computed e2 (and therefore e3) will be numerically unstable and the resulting rotation matrix may be ill-conditioned; the built-in epsilon prevents runtime division errors but does not guarantee a meaningful orientation in these degenerate cases. If inputs are not numeric finite arrays, or have incompatible shapes for vector arithmetic, NumPy will raise the usual exceptions (e.g., ValueError). The function has no side effects and does not modify its inputs.
    
    Domain significance:
        In Boltz-2's data tokenization and preprocessing, per-residue local frames computed by this function enable the model to represent atomic coordinates and inter-residue geometry in a rotation- and translation-aware way, which is critical for learning structure and binding-affinity relationships efficiently and for producing invariant/equivariant features used throughout the model.
    """
    from boltz.data.tokenize.boltz2 import compute_frame
    return compute_frame(n, ca, c)


################################################################################
# Source: boltz.data.tokenize.boltz2.get_unk_token
# File: boltz/data/tokenize/boltz2.py
# Category: valid
################################################################################

def boltz_data_tokenize_boltz2_get_unk_token(chain: numpy.ndarray):
    """Get the unk token id for a residue in a single molecular chain.
    
    This function selects the appropriate "unknown" residue token id used by the Boltz-2 tokenizer based on the chain's molecular type and returns the integer token id that the model uses to represent an unknown or placeholder residue. It is used during tokenization and preprocessing of input biomolecules for Boltz-2 inference and training (structure and affinity prediction), so that non-standard or missing residues are consistently encoded according to whether the chain is DNA, RNA, or a protein. The selection logic follows the repository constants: if chain["mol_type"] equals const.chain_type_ids["DNA"] the DNA unk token is chosen; if it equals const.chain_type_ids["RNA"] the RNA unk token is chosen; otherwise the protein unk token is chosen.
    
    Args:
        chain (numpy.ndarray): A numpy structured array or array-like object representing a single chain. The array is expected to expose a "mol_type" field or key (e.g., chain["mol_type"]) whose value is an integer id comparable to entries in const.chain_type_ids. This value determines whether the chain is treated as DNA, RNA, or protein for choosing the unk token. The function does not modify this array.
    
    Returns:
        int: The integer token id corresponding to the selected unknown residue token. Concretely, this is const.token_ids[const.unk_token["DNA" or "RNA" or "PROTEIN"]] depending on chain["mol_type"]. This integer is the canonical token id used by the Boltz-2 tokenizer and downstream model components to represent unknown residues.
    
    Behavior and failure modes:
        The function is deterministic and has no side effects beyond reading the provided array and module-level constants. It relies on the presence and correctness of const.chain_type_ids, const.unk_token, and const.token_ids mappings defined in the repository. If chain does not expose a "mol_type" entry, if chain["mol_type"] is not comparable to values in const.chain_type_ids, or if the expected keys are missing from the const mappings, the call will raise standard Python errors such as KeyError or TypeError. The function does not validate numeric ranges beyond these lookups and defaults to the protein unk token for any mol_type value that is not equal to the DNA or RNA chain type ids.
    """
    from boltz.data.tokenize.boltz2 import get_unk_token
    return get_unk_token(chain)


################################################################################
# Source: boltz.model.layers.relative.compute_relative_distribution_perfect_correlation
# File: boltz/model/layers/relative.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def boltz_model_layers_relative_compute_relative_distribution_perfect_correlation(
    binned_distribution_1: torch.Tensor,
    binned_distribution_2: torch.Tensor
):
    """boltz.model.layers.relative.compute_relative_distribution_perfect_correlation: Compute the distribution of the difference (relative bin index) between two discretized (binned) quantities under the assumption that the two quantities are perfectly correlated. In the Boltz model this routine is used by relative-feature layers to convert two aligned binned representations (for example, discretized distance or score histograms produced per residue or atom) into a single binned distribution over relative offsets; the output can be used downstream for modeling pairwise geometric relationships or relative likelihoods in affinity prediction.
    
    Args:
        binned_distribution_1 (torch.Tensor): The first binned distribution. Expected shape (..., K) where K is the number of bins along the last dimension. Each entry along the last dimension represents the mass (for example, a probability or count) in that bin for the first quantity. Leading (batch) dimensions (...) must match those of binned_distribution_2 exactly. The function treats the last-dimension length as K (derived from binned_distribution_1.shape[-1]) and pads a leading zero to compute cumulative sums; negative values, NaNs, or Infs in this tensor are not considered valid probability mass and can produce incorrect or undefined behavior.
        binned_distribution_2 (torch.Tensor): The second binned distribution. Expected shape (..., K), identical to binned_distribution_1 in all dimensions. Each entry along the last dimension represents the mass in that bin for the second quantity. This tensor is combined with binned_distribution_1 to produce the relative distribution. As with binned_distribution_1, values should be non-negative and (optionally) normalized if a true probability mass function is desired; mismatched shapes or incompatible leading dimensions will raise an error when attempting concatenation and subsequent operations.
    
    Behavior and algorithmic details:
        The function computes cumulative sums of each binned distribution after prepending a zero bin. It then iterates over bin offsets i in [0, K-1] and computes the overlap mass between shifted cumulative windows using elementwise minimum and maximum followed by a rectified difference (torch.relu(min - max)). The result of these overlap calculations is summed to produce the mass assigned to each relative-offset bin. The output vector has length 2*K - 1 and encodes offsets from -(K-1) to +(K-1) relative bin indices, with the center index corresponding to zero offset. Complexity is O(K^2) due to the nested-like slicing and summation across offsets. The function does not modify its input tensors in-place; it allocates a new tensor for the output and intermediate padded arrays.
    
    Device, dtype, and numerical notes:
        The returned tensor is allocated on the same device as binned_distribution_1 (the code uses binned_distribution_1.device when creating intermediate tensors). The exact dtype of the result is determined by PyTorch arithmetic rules from the inputs and intermediate constants (typically matching the floating dtype of the inputs). If the inputs are non-negative and normalized to sum to 1 along the last dimension (valid discrete probability mass functions), the returned tensor will be non-negative and will sum to 1 along its last dimension up to numerical rounding; if inputs are not normalized, the output will reflect the corresponding scaled mass and will not be normalized automatically.
    
    Failure modes and validation:
        The function assumes both inputs have identical shapes and the same K on the last axis; if shapes differ or are not compatible for the concatenation and slicing operations used, a runtime error (shape mismatch) will be raised. Inputs containing NaN or Inf will propagate and produce undefined or invalid output. Negative values in the inputs violate the distribution interpretation and can lead to misleading results despite the use of torch.relu in intermediate steps. Because the routine uses cumulative sums and multiple slices, extremely large K will increase memory and compute cost.
    
    Returns:
        torch.Tensor: A tensor of shape (..., 2*K - 1) containing the relative distribution over integer offsets between the two input binned quantities. The last-dimension index K-1 corresponds to zero relative offset (bins aligned); indices greater than K-1 correspond to positive offsets (binned_distribution_1 shifted to larger bin indices relative to binned_distribution_2), and indices less than K-1 correspond to negative offsets (binned_distribution_2 shifted to larger bin indices relative to binned_distribution_1). The returned tensor contains non-negative values computed from overlap of cumulative distributions and is suitable for use as a discretized relative-offset distribution within Boltz relative-feature layers.
    """
    from boltz.model.layers.relative import compute_relative_distribution_perfect_correlation
    return compute_relative_distribution_perfect_correlation(
        binned_distribution_1,
        binned_distribution_2
    )


################################################################################
# Source: boltz.model.loss.confidence.confidence_loss
# File: boltz/model/loss/confidence.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def boltz_model_loss_confidence_confidence_loss(
    model_out: dict,
    feats: dict,
    true_coords: torch.Tensor,
    true_coords_resolved_mask: torch.Tensor,
    multiplicity: int = 1,
    alpha_pae: float = 0.0
):
    """Compute confidence loss for boltz.model.loss.confidence.confidence_loss.
    
    This function aggregates multiple confidence-related loss terms used in the Boltz family of biomolecular interaction models (Boltz-1 / Boltz-2) to supervise the model's confidence heads during training. In the Boltz context these heads predict per-atom or per-residue uncertainty and resolution signals that guide structure prediction and downstream affinity estimation. The function calls the specialized sub-losses plddt_loss, pde_loss, resolved_loss and optionally pae_loss, combines them into a single scalar training loss, and returns a structured breakdown of the components for logging and weighting. It expects the model output dictionary to contain logits and sampled coordinates produced by the network; the returned losses are torch.Tensor objects that can be backpropagated.
    
    Args:
        model_out (Dict[str, torch.Tensor]): Dictionary containing the model output tensors produced by the network. Required keys (as used by this function) include "plddt_logits", "pde_logits", "resolved_logits", "sample_atom_coords" and, if alpha_pae > 0.0, "pae_logits". Each value should be a torch.Tensor on the device used for training. These logits and sampled coordinates are the direct inputs to the respective sub-loss functions and represent the model's confidence predictions and predicted atom coordinates.
        feats (Dict[str, torch.Tensor]): Dictionary containing the model input features (feats) that are passed to individual loss functions. In the Boltz pipeline these features provide contextual information about the biomolecular example (e.g., atom/residue masks, types or other processed inputs) and are required by the plddt/pde/resolved/pae loss implementations. Missing or incorrectly formatted entries in this dict can cause the sub-loss functions to raise errors.
        true_coords (torch.Tensor): The ground-truth atom coordinates after any symmetry correction has been applied. This tensor is used to compute geometric errors between model-predicted coordinates (from model_out["sample_atom_coords"]) and the true structure; it is required by plddt_loss, pde_loss and pae_loss. The tensor must be compatible with the coordinate-related computations performed by those sub-loss functions (same device and expected numerical dtype).
        true_coords_resolved_mask (torch.Tensor): The resolved mask after symmetry correction, supplied as a torch.Tensor. This mask indicates which atoms or positions are considered resolved in the ground truth and is used by plddt_loss, pde_loss, resolved_loss, and pae_loss to restrict error computations to resolved positions. The mask must align with true_coords and the model's sampled coordinates; mismatched shapes or devices will cause errors in the sub-loss functions.
        multiplicity (int): The diffusion batch size (default 1). This parameter is forwarded to each sub-loss (plddt_loss, pde_loss, resolved_loss, pae_loss) and controls handling of repeated or augmented samples per example in diffusion training. Use multiplicity > 1 when the training batch contains multiple diffusion samples per input; keep at the default 1 for standard single-sample batches. Negative or non-integer values will typically produce an error in the sub-loss implementations.
        alpha_pae (float): The scalar weight for the predicted aligned error (PAE) loss term (default 0.0). When alpha_pae > 0.0 the function computes pae_loss from model_out["pae_logits"] and includes alpha_pae * pae in the final loss sum; when alpha_pae == 0.0 the PAE term is skipped (pae is set to 0.0) to save computation. This weight should be tuned according to the training objective (e.g., to emphasize alignment-error supervision for Boltz-2 affinity/structure joint training).
    
    Returns:
        Dict[str, torch.Tensor]: A dictionary containing the combined loss and a breakdown of individual components. The dictionary has at least the following entries:
            "loss" (torch.Tensor): The scalar loss used for optimization; computed as plddt + pde + resolved + alpha_pae * pae where each term is a torch.Tensor returned by the corresponding sub-loss.
            "loss_breakdown" (dict): A mapping of component names to torch.Tensor values with keys "plddt_loss", "pde_loss", "resolved_loss", and "pae_loss". Note that when alpha_pae == 0.0 the "pae_loss" value will be 0.0 (a Python float assigned in the code, but conceptually representing a torch.Tensor-valued component in normal operation).
    
    Behavior, side effects, defaults and failure modes:
        - Behavior: The function calls plddt_loss, pde_loss, and resolved_loss with appropriate entries from model_out, feats, true_coords and true_coords_resolved_mask, forwarding multiplicity. If alpha_pae > 0.0 it also calls pae_loss and multiplies that term by alpha_pae. The final scalar loss is the sum described above and is returned alongside a per-term breakdown for logging/analysis.
        - Side effects: The function does not mutate its input dictionaries or tensors. It does, however, rely on and read entries from model_out and feats; any in-place modifications performed by the sub-loss functions (if present) are external to this function's contract.
        - Defaults: multiplicity defaults to 1 indicating a single diffusion sample per example; alpha_pae defaults to 0.0 causing the PAE loss to be omitted by default.
        - Failure modes: A KeyError will be raised if model_out lacks required keys ("plddt_logits", "pde_logits", "resolved_logits", "sample_atom_coords") or if "pae_logits" is missing while alpha_pae > 0.0. TypeErrors or runtime errors may be raised if tensors in model_out, feats, true_coords, or true_coords_resolved_mask have incompatible shapes, dtypes, or devices for the sub-loss computations. Errors raised by the underlying plddt_loss, pde_loss, resolved_loss, or pae_loss implementations will propagate to the caller.
        - Practical significance: This loss aggregation is used in Boltz training to align predicted confidence scores with actual coordinate errors and resolution labels, which improves the reliability of predicted structures and downstream affinity predictions in biomolecular interaction tasks described in the repository README.
    """
    from boltz.model.loss.confidence import confidence_loss
    return confidence_loss(
        model_out,
        feats,
        true_coords,
        true_coords_resolved_mask,
        multiplicity,
        alpha_pae
    )


################################################################################
# Source: boltz.model.loss.confidence.pae_loss
# File: boltz/model/loss/confidence.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def boltz_model_loss_confidence_pae_loss(
    pred_pae: torch.Tensor,
    pred_atom_coords: torch.Tensor,
    true_atom_coords: torch.Tensor,
    true_coords_resolved_mask: torch.Tensor,
    feats: dict,
    multiplicity: int = 1,
    max_dist: float = 32.0
):
    """Compute the pairwise aligned error (PAE) classification loss used by Boltz models to supervise the model's confidence in predicted atom coordinates for biomolecular interaction and affinity prediction.
    
    This function converts predicted and true Cartesian atom coordinates into per-frame token coordinates, computes the Euclidean errors between predicted and true token coordinates, discretizes those errors into bins, and compares the resulting target bin distribution against the model's predicted PAE logits using a masked negative log-likelihood. The loss is averaged over valid token pairs and across the (possibly grouped) batch and is used during training to teach the model to produce calibrated per-pair confidence estimates (PAE) that are important for downstream structural and binding-affinity predictions in Boltz.
    
    Args:
        pred_pae (torch.Tensor): The PAE logits output by the model. The last tensor dimension indexes discrete distance/error bins (num_bins). These logits are converted to a probability distribution with log_softmax and compared to the one-hot target bin encoded from the Euclidean error between predicted and true token coordinates. The tensor must be on the same device as the coordinate tensors and its last dimension must equal the number of bins expected by the downstream binning logic.
        pred_atom_coords (torch.Tensor): The predicted atom coordinates in Cartesian space provided by the model, used to compute predicted frames and to express token coordinates in those frames. The code expects a 3-coordinate per-atom representation (i.e., a final dimension of size 3). The batch-size dimension of this tensor must be compatible with multiplicity (see multiplicity) because the function reshapes the batch as B // multiplicity, multiplicity, ...; a mismatch (B not divisible by multiplicity) will raise a runtime error.
        true_atom_coords (torch.Tensor): The ground-truth atom coordinates after any symmetry correction has been applied. These coordinates are used to compute true frames, to express token coordinates in those true frames, and to form the continuous target PAE (Euclidean error) against which pred_pae is trained. This tensor must have the same device and compatible shape semantics as pred_atom_coords.
        true_coords_resolved_mask (torch.Tensor): A binary or boolean mask (torch.Tensor) indicating which true coordinates are resolved after symmetry correction. This mask is consulted when building the pairwise mask that determines which token pairs contribute to the loss (e.g., unresolved atoms are excluded). The mask must align with the indexing scheme used by frames computed from true_atom_coords and with batch grouping introduced by multiplicity.
        feats (dict): Dictionary containing model input tensors required by this loss. Required keys used by the implementation are "frames_idx", "frame_resolved_mask", and "token_pad_mask". "frames_idx" and "frame_resolved_mask" (both torch.Tensor) supply original frame indices and frame-level resolution information used by compute_frame_pred; "token_pad_mask" (torch.Tensor) is used to mask out padded tokens so they do not contribute to the loss. All tensors in feats must be on the same device as pred_pae and the coordinate tensors.
        multiplicity (int, optional): The diffusion batch grouping size. Default is 1. The implementation groups the batch dimension as B // multiplicity, multiplicity, ... when expressing coordinates in frames and when computing pairwise errors. multiplicity must be a positive integer and must divide the first (batch) dimension of pred_atom_coords and true_atom_coords exactly; otherwise the reshape operations will fail.
        max_dist (float, optional): The maximum distance used to map continuous Euclidean PAE values into discrete bin indices. Default is 32.0. The bin index is computed as floor(target_pae * num_bins / max_dist) and then clamped to [0, num_bins-1]. max_dist must be positive and non-zero; setting it to zero will produce a division-by-zero error. The value scales how continuous distances are discretized into the pred_pae bins (e.g., distance units should match the units of pred_atom_coords/true_atom_coords, typically Angstroms in structural modeling).
    
    Returns:
        torch.Tensor: A scalar torch.Tensor containing the averaged PAE loss value. The returned loss is computed by converting continuous per-token Euclidean errors into discrete bin indices, computing the negative log-likelihood of the model's pred_pae logits against the one-hot target bins, masking invalid token pairs (using frame validity, collinearity masks, resolution masks, and token padding mask), averaging over valid pairs per example, and finally averaging those per-example losses over the batch. There are no other side effects.
    
    Notes on behavior, numerical safeguards, and failure modes:
        - The function computes per-token frames with compute_frame_pred and expresses coordinates in those frames via express_coordinate_in_frame; if those helper functions impose additional requirements on tensor shapes or contents, violating those will raise runtime errors.
        - A small epsilon (1e-8) is added inside the square-root used to compute continuous Euclidean error to avoid NaNs for exact zeros.
        - The implementation clamps bin indices to the valid range [0, num_bins-1] to avoid indexing errors when errors exceed max_dist.
        - The pair_mask multiplies several masks (frame validity, collinearity, resolution, and token_pad_mask) to exclude invalid or padded pairs; if any required mask tensor is missing from feats or has incompatible shape/device, a runtime error will occur.
        - All input tensors must be on the same device; mismatched devices will raise errors during tensor operations.
        - The function assumes pred_pae's last dimension is the number of bins used for discretization; a mismatch between this dimension and downstream expectations (e.g., num_bins inferred from pred_pae vs. the discretization strategy) will lead to incorrect loss computation.
        - This loss is intended for training Boltz-style biomolecular structure and confidence prediction models and contributes to calibrating per-pair confidence (PAE) estimates used in downstream tasks such as structure selection and affinity prediction.
    """
    from boltz.model.loss.confidence import pae_loss
    return pae_loss(
        pred_pae,
        pred_atom_coords,
        true_atom_coords,
        true_coords_resolved_mask,
        feats,
        multiplicity,
        max_dist
    )


################################################################################
# Source: boltz.model.loss.confidence.pde_loss
# File: boltz/model/loss/confidence.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def boltz_model_loss_confidence_pde_loss(
    pred_pde: torch.Tensor,
    pred_atom_coords: torch.Tensor,
    true_atom_coords: torch.Tensor,
    true_coords_resolved_mask: torch.Tensor,
    feats: dict,
    multiplicity: int = 1,
    max_dist: float = 32.0
):
    """Compute the pairwise distance error (PDE) loss used by Boltz models for training predicted atomic coordinates against symmetry-corrected ground-truth coordinates.
    
    This function converts per-pair absolute distance errors between predicted and true token-level coordinates into discrete bins, computes a negative log-likelihood loss against provided PDE logits, and averages the result across resolved token pairs and the batch. In the Boltz training pipeline (used for biomolecular interaction and binding-affinity modeling described in the README), this loss encourages predicted atom coordinates to reproduce ground-truth inter-token distances after symmetry correction and is particularly relevant when training diffusion-style generative/denoising components (multiplicity corresponds to diffusion batch replication in the code).
    
    Args:
        pred_pde (torch.Tensor): The PDE logits produced by the model. These logits represent, for each token pair, an unnormalized score over a fixed number of discrete error bins. The last tensor dimension is interpreted as the number of bins (num_bins) used to discretize pairwise absolute distance errors. The function applies log_softmax to these logits along the last dimension to compute per-bin log-probabilities.
        pred_atom_coords (torch.Tensor): The predicted atomic coordinates emitted by the model. These coordinates are used to compute token-level coordinates via multiplication with the token_to_rep_atom mapping from feats and to form pairwise distances that are compared to true_atom_coords distances. This tensor must be compatible with token_to_rep_atom for batched matrix multiplication (torch.bmm) after any repetition induced by multiplicity.
        true_atom_coords (torch.Tensor): The ground-truth atomic coordinates after any symmetry correction has been applied. These coordinates are used to compute the target pairwise distances that define the absolute per-pair error (target_pde). The practical significance is that these coordinates represent the corrected reference geometry the model should match for accurate structure and affinity prediction.
        true_coords_resolved_mask (torch.Tensor): A mask (tensor) indicating which atomic coordinates are resolved/valid after symmetry correction. This mask is combined with token_to_rep_atom to produce a token-level resolved mask; that token-pair mask determines which token pairs contribute to the loss. If many pairs are unresolved, they are excluded from the per-sample averaging.
        feats (dict): Dictionary containing model input features. The implementation requires a key "token_to_rep_atom" mapping tokens to representative atom coordinates; that tensor is extracted, cast to float, and repeated multiplicity times. If "token_to_rep_atom" is absent a KeyError will be raised. The token_to_rep_atom mapping is used to compute token-level coordinates from atom coordinates via torch.bmm and to build the token-level mask.
        multiplicity (int, optional): The diffusion batch size multiplier. Defaults to 1. The function repeats the token_to_rep_atom tensor along the batch dimension multiplicity times using repeat_interleave(multiplicity, 0) so the token_to_rep_atom mapping aligns with pred/true coordinate batches that have been replicated for diffusion. This must be a positive integer; providing an incompatible value or one that does not match the effective batch replication of pred_atom_coords/true_atom_coords results in size-mismatch errors in batched matrix multiplications.
        max_dist (float, optional): The maximum distance (upper bound) used to scale continuous absolute pairwise distance errors into discrete bins. Defaults to 32.0. The bin index is computed as floor(target_pde * num_bins / max_dist) and then clamped to num_bins-1. max_dist should be positive; using zero or a negative value will produce invalid binning behavior (division by zero or incorrect indices).
    
    Behavior and implementation details:
        The function first extracts token_to_rep_atom from feats, casts it to float, and repeats it multiplicity times along the batch dimension. It computes a token-level resolved mask by applying token_to_rep_atom to true_coords_resolved_mask via batched matrix multiplication and constructs a pairwise token-pair mask (mask) that selects resolved token pairs.
        Token-level coordinates for both prediction and ground truth are computed by multiplying token_to_rep_atom with pred_atom_coords and true_atom_coords respectively (torch.bmm). Pairwise Euclidean distance matrices for predicted and true token coordinates are computed using torch.cdist; the absolute per-pair difference (target_pde) is used as the discretization target.
        num_bins is taken from pred_pde.shape[-1]. Continuous target_pde values are converted to discrete bin indices by bin_index = floor(target_pde * num_bins / max_dist) and clamped to [0, num_bins-1]. These indices are one-hot encoded and compared to the predicted logits via a negative log-probability: errors = -sum(one_hot * log_softmax(pred_pde, dim=-1), dim=-1). The per-sample pairwise errors are summed over resolved token pairs and normalized by the number of resolved pairs (torch.sum(mask, dim=(-2, -1))). A small constant (1e-7) is added to the denominator for numerical stability. Finally, the per-sample losses are averaged over the batch dimension to produce a scalar loss.
        No in-place modifications are performed on input tensors by this function.
    
    Side effects, defaults, and failure modes:
        - Requires feats to contain the "token_to_rep_atom" key; otherwise a KeyError is raised.
        - Assumes pred_pde has a last dimension equal to the number of discretization bins; mismatched shapes will raise runtime errors.
        - multiplicity should match how pred/true coordinate batches were replicated for diffusion training; otherwise torch.bmm or shape operations will fail.
        - If no token pairs are resolved for a sample (mask sums to zero), the denominator becomes 1e-7 and the normalized loss may be large or uninformative; this behavior is mitigated only by the small stabilizing constant.
        - max_dist must be positive; non-positive values lead to invalid bin indices or division errors.
        - The function expects absolute distance errors (target_pde >= 0), which follows from the use of torch.abs on the difference of pairwise distances.
    
    Returns:
        torch.Tensor: A scalar torch.Tensor containing the averaged PDE loss (dtype float-like, shape ()). This value is the mean over batch samples of the masked negative log-likelihood of the true binned pairwise distance errors under the predicted PDE logits. No other side effects occur; the returned tensor is ready to be combined with other training losses for optimization.
    """
    from boltz.model.loss.confidence import pde_loss
    return pde_loss(
        pred_pde,
        pred_atom_coords,
        true_atom_coords,
        true_coords_resolved_mask,
        feats,
        multiplicity,
        max_dist
    )


################################################################################
# Source: boltz.model.loss.confidence.plddt_loss
# File: boltz/model/loss/confidence.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def boltz_model_loss_confidence_plddt_loss(
    pred_lddt: torch.Tensor,
    pred_atom_coords: torch.Tensor,
    true_atom_coords: torch.Tensor,
    true_coords_resolved_mask: torch.Tensor,
    feats: dict,
    multiplicity: int = 1
):
    """boltz.model.loss.confidence.plddt_loss computes the pLDDT (per-residue predicted Local Distance Difference Test) classification loss used in Boltz training to supervise the model's confidence head for structure predictions and downstream biomolecular interaction tasks such as affinity-aware modeling. The function converts predicted atom coordinates and model features into token-level distances, computes target lDDT values using a pairwise distance comparison (via lddt_dist), converts those continuous targets into discrete bins, and returns the negative log-likelihood-style loss (cross-entropy on one-hot binned targets) averaged over resolved tokens and the batch. This loss is used during training to teach the model to predict per-token confidence scores that correlate with true structural accuracy, which is critical for Boltz applications in biomolecular interaction prediction and ligand optimization (see project README for context).
    
    Args:
        pred_lddt (torch.Tensor): The pLDDT logits produced by the model's confidence head. These logits are expected to have a final dimension equal to the number of discrete confidence bins (num_bins) and are used with softmax/log-softmax to compute a per-token classification loss against binned target lDDT values.
        pred_atom_coords (torch.Tensor): The predicted atom coordinates produced by the structural prediction head. These coordinates are used to compute predicted token-to-token pairwise distances and to compare with true coordinates when forming target lDDT values. They must be compatible with the coordinate-related feature tensors supplied in feats.
        true_atom_coords (torch.Tensor): The reference atom coordinates after any symmetry correction applied by preprocessing. These coordinates are used to compute true token-level pairwise distances for target lDDT calculation. They must align with the atom/token indexing implied by feats.
        true_coords_resolved_mask (torch.Tensor): A boolean or 0/1 mask indicating which atom coordinates in true_atom_coords are resolved (valid) after symmetry correction. This mask is used to build pairwise masks and to ignore unresolved atoms when computing per-token lDDT targets and the final loss.
        feats (Dict[str, torch.Tensor]): Dictionary containing model input features required by this loss. The function reads and uses the following keys from feats: "r_set_to_rep_atom" (mapping from R-set elements to representative atoms, repeated to match multiplicity and used to form neighborhood sets), "mol_type" (token-level molecule/chain type ids used to detect nucleotide chains via const.chain_type_ids), "atom_to_token" (mapping from atoms to tokens), and "token_to_rep_atom" (mapping from tokens to representative atoms). Missing keys or incompatible shapes will raise runtime errors. The practical significance of these features is that they convert atom-level coordinates into token-level coordinates and neighborhoods required to compute lDDT targets that reflect structural accuracy at the token (residue) level, which the pLDDT head predicts.
        multiplicity (int, optional): The diffusion batch multiplicity (diffusion batch size) used to repeat certain feature tensors along the batch dimension before loss calculation. Defaults to 1. This parameter is used with torch.repeat_interleave on feature tensors so the loss can be computed for multiple diffusion samples per original input when training diffusion-based generative models.
    
    Behavior and implementation details:
        The function extracts the necessary mapping tensors from feats and repeats them along the batch dimension according to multiplicity. It converts atom-level coordinates to token-level coordinates via token_to_rep_atom and computes pairwise token distances for both predicted and true coordinates using torch.cdist. A pairwise mask is constructed from true_coords_resolved_mask to ignore pairs involving unresolved atoms and self-pairs. The function expands the R-set neighborhood mask using R_set_to_rep_atom and applies a nucleotide-dependent distance cutoff (cutoff = 15 + 15 * is_nucleotide_R_element) where is_nucleotide_R_element is computed from mol_type via const.chain_type_ids["DNA"/"RNA"]; this increases the allowable neighborhood cutoff for nucleotide tokens and thereby affects which pairs contribute to lDDT target computation. The target lDDT and an associated mask_no_match are computed by calling lddt_dist(pred_d, true_d, pair_mask, cutoff, per_atom=True). target_lddt values are discretized into bin indices by multiplying by num_bins (the size of pred_lddt's last dimension), flooring, and clamping; a one-hot target distribution is formed and compared with pred_lddt via negative log-softmax to produce per-token errors. Per-token errors are masked by the token-level atom_mask (derived from true_coords_resolved_mask and token_to_rep_atom) and mask_no_match, averaged over resolved tokens with numerical stabilization (division by 1e-7 + sum of masks), and then averaged over the batch to produce a single scalar loss value.
    
    Defaults, side effects, and numerical details:
        multiplicity defaults to 1. The function does not modify its input tensors in-place; it performs pure tensor computations and returns a torch.Tensor loss. A tiny epsilon (1e-7) is added in the denominator when averaging per-token errors to avoid division by zero when no tokens pass masking criteria in a batch element.
    
    Failure modes and error conditions:
        A KeyError will be raised if feats lacks any required keys ("r_set_to_rep_atom", "mol_type", "atom_to_token", "token_to_rep_atom"). Mismatched batch dimensions between pred_atom_coords, true_atom_coords, true_coords_resolved_mask, and the mapping tensors in feats can cause broadcasting or runtime shape errors in torch.bmm/torch.cdist operations. Non-numeric or incompatible tensor dtypes (e.g., integer tensors where floating operations are required) can cause runtime errors; the code explicitly casts some feature tensors to float but assumes coordinate tensors support floating-point distance computations. If pred_lddt has last-dimension size zero or an unexpected size, indexing/clamping and one-hot operations will fail. The nucleotide-dependent cutoff logic relies on const.chain_type_ids containing "DNA" and "RNA"; missing or different chain id conventions will alter behavior. Users should ensure tensors are on the same device and have compatible dtypes before calling this function to avoid device/dtype errors.
    
    Returns:
        torch.Tensor: A scalar tensor containing the average pLDDT loss over the batch. This loss is computed as the mean (over batch) of per-element masked negative log-probability of the binned target lDDT under pred_lddt logits, where per-token contributions are masked by resolved-coordinate and lddt-matching masks. The return value is suitable for optimization (e.g., used directly with optimizer.step()) during training of Boltz confidence heads.
    """
    from boltz.model.loss.confidence import plddt_loss
    return plddt_loss(
        pred_lddt,
        pred_atom_coords,
        true_atom_coords,
        true_coords_resolved_mask,
        feats,
        multiplicity
    )


################################################################################
# Source: boltz.model.loss.confidence.resolved_loss
# File: boltz/model/loss/confidence.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def boltz_model_loss_confidence_resolved_loss(
    pred_resolved: torch.Tensor,
    feats: dict,
    true_coords_resolved_mask: torch.Tensor,
    multiplicity: int = 1
):
    """Compute the resolved-classification loss used by the Boltz confidence head. This function is part of the Boltz biomolecular interaction model loss suite and is used to supervise a binary "resolved vs not-resolved" prediction for tokens (e.g., ligand/protein tokens) during training or diffusion inference. It combines model logits, a symmetry-corrected ground-truth resolved mask, and token-level padding/mapping features to produce a single scalar loss value averaged over the (possibly repeated) diffusion batch.
    
    Args:
        pred_resolved (torch.Tensor): Predicted logits returned by the model for the resolved classification. The last dimension is expected to contain two logits per token corresponding to the classes "resolved" (index 0) and "not resolved" (index 1). These are raw, unnormalized scores; the function applies a log-softmax across the last dimension to compute per-class log-probabilities.
        feats (dict): Dictionary of input feature tensors produced by the model/data pipeline. This function reads at least the following keys from feats:
            "token_to_rep_atom" (torch.Tensor): A token-to-representative-atom mapping used to aggregate atom-level resolved labels up to token level. The code calls .repeat_interleave(multiplicity, 0) and .float() on this tensor, so providing a tensor of an appropriate dtype and batch-aligned leading dimension is required.
            "token_pad_mask" (torch.Tensor): A token-level padding mask (1 for real tokens, 0 for padding) used to ignore padded tokens when summing losses. This tensor is also repeated along the batch dimension with .repeat_interleave(multiplicity, 0) and converted to float.
            The practical significance in the Boltz domain is that token_to_rep_atom maps model tokens to the representative atomic coordinates used to derive resolved labels after symmetry correction, and token_pad_mask ensures that padding tokens from sequence batching do not contribute to the loss.
        true_coords_resolved_mask (torch.Tensor): Ground-truth resolved mask after symmetry correction that indicates which representative coordinates are resolved. This mask is supplied at the coordinate/atom level and is aggregated to tokens by multiplying with token_to_rep_atom via a batched matrix multiply. It must be batch-aligned with feats after the possible repeat_interleave induced by multiplicity.
        multiplicity (int): The diffusion batch repetition factor, default 1. When multiplicity > 1, token-level feature tensors in feats ("token_to_rep_atom" and "token_pad_mask") are repeated along the batch dimension using torch.repeat_interleave(multiplicity, 0) so that their leading batch dimension matches an expanded batch of pred_resolved. This parameter is used when the diffusion sampler duplicates examples (e.g., for parallel perturbations) and must match how pred_resolved has been constructed.
    
    Behavior and details:
        The function first extracts token_to_rep_atom and token_pad_mask from feats, converts them to float, and repeats their entries along the batch dimension multiplicity times to align with pred_resolved when multiplicity > 1. It computes a per-token reference mask (ref_mask) by performing a batched matrix multiply (torch.bmm) between token_to_rep_atom and true_coords_resolved_mask.unsqueeze(-1), then squeezes the last dimension. The function computes log probabilities with torch.nn.functional.log_softmax(pred_resolved, dim=-1) and forms per-token negative log-likelihood errors:
            errors = -ref_mask * log_prob[..., 0] - (1 - ref_mask) * log_prob[..., 1]
        where index 0 is treated as the "resolved" class and index 1 as the "not resolved" class. Errors are summed across tokens only where token_pad_mask is 1; the per-example loss is normalized by the sum of token_pad_mask plus a small epsilon (1e-7) to avoid division by zero when an example has no unpadded tokens. Finally, the function averages the per-example losses over the batch dimension and returns this scalar tensor.
        Defaults: multiplicity defaults to 1. The numerical stability epsilon (1e-7) is fixed in the implementation and prevents NaNs when dividing by zero token counts.
    
    Side effects:
        This function does not modify inputs in-place, but it creates repeated copies of token_to_rep_atom and token_pad_mask via repeat_interleave. It relies on feats containing the required keys; missing keys will raise a KeyError. The function uses torch.bmm and log_softmax which allocate intermediate tensors.
    
    Failure modes and exceptions:
        A KeyError will occur if feats does not contain "token_to_rep_atom" or "token_pad_mask".
        A RuntimeError or ValueError will be raised by PyTorch if tensor dimensions are incompatible for repeat_interleave, torch.bmm, or indexing (for example, if pred_resolved's last dimension is not size 2).
        If pred_resolved is not a floating-point tensor with its last dimension length 2 (two logits per token), the results are undefined or a runtime error may be raised. The function handles the case of zero unpadded tokens per example by adding 1e-7 before dividing to avoid division by zero; however, such examples will receive a normalized loss computed against that epsilon denominator.
        This function assumes that true_coords_resolved_mask has already undergone any required symmetry correction upstream (as indicated by the argument name); passing an uncorrected mask will produce an incorrect loss relative to Boltz's intended supervision.
    
    Returns:
        torch.Tensor: A scalar tensor containing the average resolved loss across the batch (mean of per-example normalized negative log-likelihoods). This tensor is differentiable and intended to be summed or weighted with other loss terms in the Boltz training/inference pipeline.
    """
    from boltz.model.loss.confidence import resolved_loss
    return resolved_loss(pred_resolved, feats, true_coords_resolved_mask, multiplicity)


################################################################################
# Source: boltz.model.loss.diffusion.smooth_lddt_loss
# File: boltz/model/loss/diffusion.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def boltz_model_loss_diffusion_smooth_lddt_loss(
    pred_coords: torch.Tensor,
    true_coords: torch.Tensor,
    is_nucleotide: torch.Tensor,
    coords_mask: torch.Tensor,
    nucleic_acid_cutoff: float = 30.0,
    other_cutoff: float = 15.0,
    multiplicity: int = 1
):
    """Compute a smooth lDDT-style loss between predicted and ground-truth atom coordinates for biomolecular structures, with separate distance cutoffs for nucleic acids versus other molecules and optional batch multiplicity handling. This function is used in Boltz diffusion loss computations to quantify local distance agreement (a smooth proxy of lDDT) between predicted and true pairwise atom distances, weighting only atom pairs within type-dependent distance cutoffs and using a smooth set of sigmoid thresholds (0.5, 1.0, 2.0, 4.0 Å) to produce a soft correctness score. The final scalar loss returned is 1.0 minus the mean per-sample lDDT score and is suitable for use as a training objective in biomolecular interaction and structure modeling (e.g., protein–ligand, protein–nucleic acid complexes) as in the Boltz model pipeline.
    
    Args:
        pred_coords (torch.Tensor): Predicted atom coordinates. Used to compute pairwise Euclidean distances (via torch.cdist) that are compared to true_coords distances to assess local structural agreement. Must have the same shape as true_coords and reside on the same device/dtype; the first dimension is the batch dimension and the second dimension indexes atoms.
        true_coords (torch.Tensor): Ground-truth atom coordinates. The function computes true pairwise distances from this tensor (B, N, 3-like shape is expected by the code) and uses those distances to decide which atom pairs are considered (based on cutoffs) and to compute absolute distance differences against pred_coords. The first dimension B (batch) and second dimension N (atoms) are used to reshape intermediate tensors; therefore B must be divisible by multiplicity (see multiplicity) so that reshaping via view(multiplicity, B // multiplicity, N, N) succeeds.
        is_nucleotide (torch.Tensor): Indicator tensor identifying nucleic-acid atoms versus others. This tensor is treated as a per-atom weight/indicator and is expanded into pairwise form to select nucleic-acid-specific distance cutoff logic. The tensor will be repeated along the batch dimension using repeat_interleave(multiplicity, 0) inside the function; therefore its leading dimension must be such that after repeat_interleave it matches pred_coords.shape[0]. Typical values are 0/1 with shape (batch', N) where batch' * multiplicity == pred_coords.shape[0].
        coords_mask (torch.Tensor): Atom presence mask (per-atom, per-sample). Used to zero out contributions from missing or padded atoms by forming a pairwise mask (coords_mask.unsqueeze(-1) * coords_mask.unsqueeze(-2)). This tensor is also repeated along the batch dimension with repeat_interleave(multiplicity, 0) so its leading dimension must be compatible with that operation and ultimately match pred_coords.shape[0]. The mask values should be numeric (0/1 or equivalent) and on the same device as the coordinate tensors.
        nucleic_acid_cutoff (float): Distance threshold (in the same units as coordinates, typically Å) used to decide which atom pairs among nucleic-acid atoms are included in the lDDT averaging. Default 30.0. Atom pairs where the ground-truth distance is strictly less than this cutoff (and satisfying other masks) are considered for nucleic-acid pairs.
        other_cutoff (float): Distance threshold used for non-nucleic-acid atom pairs. Default 15.0. Atom pairs where the ground-truth distance is strictly less than this cutoff (and satisfying other masks) are considered for non-nucleic-acid pairs.
        multiplicity (int): Integer number of repeated augmentations per base batch element. The function repeats is_nucleotide and coords_mask along the batch dimension using repeat_interleave(multiplicity, 0) and reshapes intermediate per-pair agreement scores with view(multiplicity, B // multiplicity, N, N) followed by mean(dim=0). Default 1. multiplicity must divide the batch size B (true_coords.shape[0]) exactly; otherwise view will fail and a runtime error will be raised. multiplicity controls how per-augmentation statistics are averaged back to per-original-sample statistics.
    
    Behavior and algorithmic details:
        The function computes pairwise Euclidean distances for true_coords and pred_coords using torch.cdist and forms their absolute difference dist_diff = |true_dists - pred_dists|. A smooth correctness score eps is computed per pair by averaging four sigmoid responses at distance-difference thresholds 0.5, 1.0, 2.0, and 4.0 (i.e., mean of sigmoid(0.5 - diff), sigmoid(1.0 - diff), sigmoid(2.0 - diff), sigmoid(4.0 - diff)). This averaging yields values in (0,1) that act like soft indicators of close agreement across multiple tolerances. When multiplicity > 1, eps is reshaped to (multiplicity, B // multiplicity, N, N) and averaged over the multiplicity dimension to produce per-original-sample eps.
        A pairwise mask is constructed by selecting nucleic-acid pairs using the expanded is_nucleotide indicator and applying nucleic_acid_cutoff, and selecting other pairs with other_cutoff; the pairwise mask excludes self-pairs via multiplication with (1 - identity) and applies coords_mask to remove missing atoms. The masked averaging sums eps over the kept pairs and divides by the count of kept pairs per sample; the denominator is clamped to at least 1 to avoid division by zero. The per-sample lDDT is num/den and the function returns 1.0 - mean(lddt) as a scalar loss.
        The function does not perform in-place modification of inputs; it allocates intermediate tensors. The device of the identity matrix is taken from pred_coords, so all tensors should be on the same device to avoid device-mismatch errors.
    
    Defaults and side effects:
        nucleic_acid_cutoff defaults to 30.0 and other_cutoff to 15.0, reflecting larger interaction distances typically used for nucleic-acid contexts versus other biomolecules in Boltz modeling. multiplicity defaults to 1, meaning no repetition/averaging unless explicitly used for augmentation or multiple predictions per base sample.
        Side effects: none on inputs; returns a torch.Tensor scalar loss. Intermediate tensors are allocated; ensure sufficient memory for O(N^2) pairwise distance computations.
    
    Failure modes and user guidance:
        - If pred_coords and true_coords have mismatched shapes or different devices/dtypes, torch.cdist and subsequent operations will raise errors. Ensure they have identical shapes and are colocated.
        - multiplicity must exactly divide the batch size B (true_coords.shape[0]); otherwise the view(multiplicity, B // multiplicity, N, N) will fail with a runtime error.
        - is_nucleotide and coords_mask must be repeatable via repeat_interleave(multiplicity, 0) to match pred_coords.shape[0]; otherwise broadcasting/repeat errors will occur.
        - For very large N (number of atoms) this function computes O(N^2) pairwise distances and masks and may be memory/time intensive; plan resources accordingly.
        - The denominator for per-sample averaging is clamped to at least 1 to avoid NaNs when no pairs are selected by the mask; in such cases the lDDT contribution for that sample will be num/den where den==1 (effectively zero if no pairs), which can bias training if many samples have no valid pairs.
    
    Returns:
        torch.Tensor: Scalar loss tensor equal to 1.0 minus the mean per-sample smooth lDDT score across the batch. This single-value tensor is ready to be used as a training loss. If all computations succeed, inputs are not mutated and no other side effects occur.
    """
    from boltz.model.loss.diffusion import smooth_lddt_loss
    return smooth_lddt_loss(
        pred_coords,
        true_coords,
        is_nucleotide,
        coords_mask,
        nucleic_acid_cutoff,
        other_cutoff,
        multiplicity
    )


################################################################################
# Source: boltz.model.loss.diffusion.weighted_rigid_align
# File: boltz/model/loss/diffusion.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def boltz_model_loss_diffusion_weighted_rigid_align(
    true_coords: torch.Tensor,
    pred_coords: torch.Tensor,
    weights: torch.Tensor,
    mask: torch.Tensor
):
    """Compute a weighted rigid alignment of predicted atom coordinates to ground-truth atom coordinates for use in Boltz diffusion loss and structural comparison.
    
    This function computes a rotation and translation that best aligns the ground-truth point cloud (true_coords) to the predicted point cloud (pred_coords) under per-atom weighting and an atom mask. It is used in the Boltz biomolecular modeling pipeline to remove global rigid-body differences (rotation + translation) before computing coordinate-based losses or RMSD-like metrics during training and inference for protein/ligand structural modeling and affinity prediction. The algorithm computes weighted centroids, centers coordinates, forms a weighted cross-covariance matrix, computes its SVD (performed in float32 internally), corrects the SVD to produce a proper rotation (determinant = 1), applies the rotation and translation, and returns the aligned ground-truth coordinates in the prediction frame. Side effects: the returned tensor is detached from the autograd graph (aligned_coords.detach_()) and the function may print warnings for degenerate/low-rank point clouds. The function does not modify the provided input tensors.
    
    Args:
        true_coords (torch.Tensor): The ground-truth atom coordinates to be aligned. Expected shape (batch_size, num_points, dim) where dim is the Cartesian dimension (typically 3 for atomic coordinates). These represent the reference coordinates from experimental structures or target conformations used by Boltz for computing alignment-sensitive losses.
        pred_coords (torch.Tensor): The predicted atom coordinates produced by the model that define the target frame for alignment. Expected shape (batch_size, num_points, dim) and the same batch_size, num_points, and dim as true_coords. In the Boltz workflow this is typically the model output to which true_coords should be aligned before loss computation.
        weights (torch.Tensor): Per-atom alignment weights that indicate the relative importance of each atom/point when computing centroids and the cross-covariance. Expected shape (batch_size, num_points). Weights are multiplied elementwise by mask before use. In practical use within Boltz this can upweight specific atom types, ligand atoms, or other sites of interest when computing the optimal rigid transform.
        mask (torch.Tensor): Binary or real-valued atom mask of shape (batch_size, num_points) that selects valid atoms/points for the alignment (e.g., 1 for present atoms, 0 for padding). The mask is multiplied with weights so masked-out atoms do not contribute to centroids, covariance, or rotation estimation.
    
    Behavior, defaults, and failure modes:
        - The function computes weighted centroids using (mask * weights) as the effective per-atom weight and centers both point sets by their weighted centroids.
        - It computes the weighted cross-covariance matrix and performs a singular value decomposition (SVD). For numerical stability and compatibility with some GPU drivers, the SVD is performed on a float32 copy of the covariance matrix and the resulting rotation is cast back to the original covariance dtype.
        - The rotation is corrected to ensure a proper rotation matrix with determinant +1 (addressing reflection ambiguity) by constructing a correction matrix F and recomputing the rotation from U, F, V components of the SVD.
        - The function prints warnings (using standard output) in two non-fatal cases: when the number of points is <= dim + 1 (the transform cannot be unique) and when singular values are excessively small (indicating low-rank cross-correlation and non-unique rotation). These warnings do not raise exceptions but indicate that the returned rotation may not be unique.
        - The returned aligned coordinates tensor is detached in-place (aligned_coords.detach_()), so it does not carry gradient history back into the model. This is an intentional side effect to prevent gradients flowing through the alignment operation when alignment is used purely as a post-processing step before a rigid-invariant loss.
        - If the covariance matrix contains NaNs or if the underlying torch.linalg.svd call fails on the platform or inputs, torch may raise a runtime error originating from the SVD; such exceptions are not caught within this function.
        - The function does not modify the provided input tensors (true_coords, pred_coords, weights, mask).
    
    Returns:
        torch.Tensor: Aligned coordinates with shape (batch_size, num_points, dim). The returned tensor represents true_coords rigidly transformed (rotated and translated) into the frame of pred_coords according to the provided weights and mask. The dtype of the returned tensor matches the numeric dtype of the input covariance computation (i.e., consistent with the input coordinate dtype after internal float32 SVD conversion). The returned tensor is detached from autograd (no gradient history) as a side effect.
    """
    from boltz.model.loss.diffusion import weighted_rigid_align
    return weighted_rigid_align(true_coords, pred_coords, weights, mask)


################################################################################
# Source: boltz.model.loss.diffusionv2.smooth_lddt_loss
# File: boltz/model/loss/diffusionv2.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def boltz_model_loss_diffusionv2_smooth_lddt_loss(
    pred_coords: torch.Tensor,
    true_coords: torch.Tensor,
    is_nucleotide: torch.Tensor,
    coords_mask: torch.Tensor,
    nucleic_acid_cutoff: float = 30.0,
    other_cutoff: float = 15.0,
    multiplicity: int = 1
):
    """smooth_lddt_loss computes a differentiable/local lDDT-style distance-based loss used during diffusion-based structure modeling in Boltz (diffusionv2). It implements "Algorithm 27" from the codebase and compares predicted and true 3D coordinates of biomolecular atoms/residues by computing per-sample local distance agreement (a smoothed lDDT-like score) over selected inter-particle pairs, then returning 1 minus the mean lDDT (so that perfect agreement yields loss 0). This loss is used in Boltz training and inference to penalize deviations in local pairwise distances between predicted structures and ground-truth structures, with different distance cutoffs applied for nucleic-acid pairs versus other pairs.
    
    Args:
        pred_coords (torch.Tensor): Predicted coordinates tensor. In the source code this is annotated as Float['b n 3']. The first (leading) batch dimension is expected to align with true_coords after any multiplicity expansion described below. pred_coords is used as the reference device for tensor allocations (pred_coords.device). In practice pred_coords may contain multiplicity-expanded predictions (multiple predicted samples per original example); the function indexes per-original-example metadata using integer division by multiplicity.
        true_coords (torch.Tensor): Ground-truth coordinates tensor. Annotated as Float['b n 3'] in the source. true_coords is compared to pred_coords by computing pairwise Euclidean distances (torch.cdist) within each batch element. The function iterates across the leading batch dimension of true_coords; if multiplicity > 1 the batch ordering is assumed to contain repeated/expanded examples such that integer division by multiplicity recovers the original example index for is_nucleotide and coords_mask lookup.
        is_nucleotide (torch.Tensor): Boolean tensor indicating whether each position/atom belongs to a nucleic acid. Annotated as Bool['b n'] in the source. This mask is used to choose different inter-particle distance cutoffs: nucleic-acid pairs use nucleic_acid_cutoff, non-nucleic pairs use other_cutoff. The tensor is indexed by floor(i / multiplicity) to obtain per-original-example nucleotide flags when multiplicity is used.
        coords_mask (torch.Tensor): Boolean tensor marking which coordinates are valid/present. Annotated as Bool['b n'] in the source. coords_mask is used to exclude pairs involving missing coordinates (it is broadcast to both pair axes). The tensor is indexed by floor(i / multiplicity) to obtain per-original-example masks when multiplicity is used. All coordinates involved in a considered pair must be True in coords_mask for that pair to be included.
        nucleic_acid_cutoff (float): Distance cutoff (in the same length units as pred_coords/true_coords) applied to pairs where the first element is a nucleotide (as determined by is_nucleotide). Default is 30.0. Pairs with true inter-particle distance strictly less than this cutoff are considered for the lDDT computation for nucleotide-containing pairs.
        other_cutoff (float): Distance cutoff applied to non-nucleotide pairs (i.e., when is_nucleotide indicates the first element is not a nucleotide). Default is 15.0. Pairs with true inter-particle distance strictly less than this cutoff are considered for the lDDT computation for other molecules (e.g., proteins, ligands).
        multiplicity (int): Integer factor describing how many predicted/true samples in the batch correspond to a single original example (multiplicity expansion). Default is 1. The implementation uses integer division i // multiplicity to recover per-original-example metadata (is_nucleotide and coords_mask). If multiplicity > 1, users must ensure the batch ordering and sizes of pred_coords/true_coords/is_nucleotide/coords_mask are consistent with this interpretation; otherwise indexing errors or incorrect pair selection will occur.
    
    Behavior and implementation details:
        The function iterates across the leading batch dimension of true_coords and for each batch element computes the matrix of true pairwise distances via torch.cdist. It builds a boolean mask selecting pairs whose true distances are below nucleic_acid_cutoff for nucleotide pairs or below other_cutoff for non-nucleotide pairs; self-pairs are explicitly excluded by multiplying by (1 - identity matrix). The coords_mask for the associated original example (retrieved using multiplicity) is applied to exclude pairs containing invalid coordinates.
        From the selected valid pairs the function extracts the corresponding true distances and computes predicted pairwise distances using F.pairwise_distance on the predicted coordinates for those pair indices. The absolute difference between true and predicted distances is mapped through four sigmoid windows centered at 0.5, 1.0, 2.0, and 4.0 (i.e., sigmoid(0.5 - diff), sigmoid(1.0 - diff), sigmoid(2.0 - diff), sigmoid(4.0 - diff)), averaged to produce an "epsilon" per pair in [0,1]. The per-sample lDDT-like score is the mean of epsilon across valid pairs; to avoid division-by-zero the denominator uses (valid_pairs_count + 1e-5).
        After computing per-sample lddt scores the function stacks them and returns 1.0 - mean(lddt) across the batch, so the returned scalar is a torch.Tensor representing the loss to be minimized during training.
    
    Defaults and numeric stability:
        Default cutoffs are nucleic_acid_cutoff=30.0 and other_cutoff=15.0. The function uses a small epsilon (1e-5) in the denominator when averaging per-sample epsilons to prevent division by zero if no pairs are selected. If no valid pairs exist for a sample, that sample's lddt contribution will be zero (eps sum is zero), producing a per-sample loss contribution of approximately 1.0 for that sample.
        The function relies on PyTorch operations (torch.cdist, F.pairwise_distance, F.sigmoid) and uses the device of pred_coords for intermediate tensors; all input tensors should be on the same device and have compatible dtypes for these operations.
    
    Side effects and performance:
        The function has no in-place side effects on inputs, but it allocates intermediate tensors and calls O(n^2) distance computations per batch element (torch.cdist), so memory and compute scale with the square of the number of positions per example. For large n this can be computationally and memory intensive; users may need to subsample positions or adjust batch size accordingly.
    
    Failure modes and required input consistency:
        Mismatched batch dimensions, incorrect multiplicity usage, or inconsistent shapes between pred_coords, true_coords, is_nucleotide, and coords_mask can lead to incorrect indexing, runtime errors, or wrong loss values. pred_coords, true_coords must have the same trailing dimension size 3 for coordinates; is_nucleotide and coords_mask must be boolean tensors with a leading dimension matching the number of original examples (i.e., true batch size before any multiplicity expansion). All tensors should be on the same torch device and have compatible dtypes. If coords_mask masks out all coordinates or no pairs pass the distance/cutoff criteria then the function will produce per-sample lddt of zero and a loss near 1.0 for those samples.
    
    Returns:
        torch.Tensor: Scalar tensor containing the loss value computed as 1.0 minus the mean smoothed lDDT score across the batch. The returned tensor is suitable for use in training (it is differentiable through the PyTorch operations used). If no valid pairs exist for any sample the loss will be near 1.0 due to the zero lDDT contributions and the added small denominator for numerical stability.
    """
    from boltz.model.loss.diffusionv2 import smooth_lddt_loss
    return smooth_lddt_loss(
        pred_coords,
        true_coords,
        is_nucleotide,
        coords_mask,
        nucleic_acid_cutoff,
        other_cutoff,
        multiplicity
    )


################################################################################
# Source: boltz.model.loss.diffusionv2.weighted_rigid_align
# File: boltz/model/loss/diffusionv2.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def boltz_model_loss_diffusionv2_weighted_rigid_align(
    true_coords: torch.Tensor,
    pred_coords: torch.Tensor,
    weights: torch.Tensor,
    mask: torch.Tensor
):
    """boltz.model.loss.diffusionv2.weighted_rigid_align computes a weighted rigid-body alignment (rotation and translation) that maps true_coords onto pred_coords by minimizing the weighted squared error between corresponding points. This function is used in the Boltz model loss pipeline (diffusionv2) to align ground-truth atomic coordinates to model-predicted coordinates before computing structural losses (e.g., RMSD or per-atom losses) during training or evaluation of biomolecular interaction and binding-affinity models such as Boltz-1/Boltz-2. The implementation follows the corrected interpretation of Algorithm 28 / equation (2) in the accompanying paper (the paper's Algorithm 28 pseudocode swaps predicted and ground-truth in the printed pseudocode; this function uses the ordering consistent with equation (2)). The algorithm computes weighted centroids, a weighted covariance matrix, performs an SVD (in float32 for numerical stability), enforces a proper rotation (determinant = +1) to resolve reflection ambiguity, and applies the rigid transform to true_coords. The function prints warnings when the alignment is underdetermined or when the covariance is numerically rank-deficient; it casts to float32 internally for SVD and returns a tensor with the same numeric dtype as the input covariance where practical.
    
    Args:
        true_coords (torch.Tensor): Float['b n 3'] tensor of ground-truth 3D coordinates for each point (atom) in the batch. In the Boltz domain this typically represents experimental or reference atomic positions for biomolecules (proteins, ligands or complexes). The function treats these as the source points to be rigidly transformed (rotated and translated) to best match pred_coords under the provided weights and mask.
        pred_coords (torch.Tensor): torch.Tensor shaped Float['b n 3'] containing predicted 3D coordinates from the model for each corresponding point (atom). In practice this is the model output that true_coords are aligned to so that downstream loss computations compare structures in a common frame.
        weights (torch.Tensor): torch.Tensor shaped Float['b n'] of non-negative per-point scalar weights. These weights determine the importance of each atom/point in the least-squares alignment objective (for example to emphasize backbone atoms, ligand atoms, or atoms known to be more reliable). The code multiplies these weights by mask and unsqueezes the last dimension for centroid and covariance computations; weights must broadcast with true_coords/pred_coords along batch and point dimensions.
        mask (torch.Tensor): torch.Tensor shaped Bool['b n'] indicating which points are present/valid in each batch element (useful for variable-length inputs such as proteins/ligands of differing sizes). mask is multiplied elementwise with weights to zero out contributions from absent points. The mask must have the same batch and point indexing as the coordinate tensors and will be cast to a numeric type when multiplied with weights.
    
    Returns:
        torch.Tensor: Float['b n 3'] containing true_coords after applying the computed rigid-body rotation and translation so that the transformed true_coords are aligned to pred_coords under the provided weights and mask. The returned tensor is detached from the gradient graph (the function calls detach_ on the aligned coordinates) and uses numerics consistent with the internal SVD casting strategy (SVD performed in float32 then cast back toward the original covariance dtype where applicable). The returned coordinates are intended for use in loss computations or structural comparisons; they do not modify the input tensors in-place aside from the internal dtype casts and the returned tensor being a detached view.
    
    Behavior, defaults, side effects, and failure modes:
        - Centroids are computed as the weighted mean of coordinates along the point dimension using weights * mask.
        - The weighted covariance matrix between pred_coords (as target frame) and centered true_coords is computed and decomposed by SVD. SVD is performed in float32 for stability; the implementation selects the CUDA-optimized driver ('gesvd') on GPU when available.
        - To ensure a proper rotation (no reflection) the determinant of the rotation is enforced to +1 by constructing a correction matrix F based on det(rot_matrix) and recomputing the rotation via U F V^H.
        - The function prints a warning if any batch element has too few valid points (mask sum per batch <= dim + 1), because in that case a unique rigid rotation cannot be determined.
        - The function prints a separate warning if the singular values of the covariance are excessively small (<= 1e-15) while the number of points exceeds dim+1, indicating near rank-deficiency and a potentially non-unique rotation.
        - The function casts the covariance to float32 for SVD and then casts rotation matrices back toward the original dtype; numerical differences may arise when inputs use float64 versus float32.
        - The function detaches the returned aligned coordinates from the autograd graph (aligned_coords.detach_()), so gradients will not flow through the returned tensor. This is an explicit side effect; if callers require gradients through the alignment, they must implement a differentiable alignment or avoid using this detached output.
        - Shape broadcasting for true_coords and pred_coords is determined by torch.broadcast_shapes; if shapes are incompatible an exception will be raised by that call. weights and mask must broadcast to the expected batch and point dimensions; otherwise tensor operations will raise runtime errors.
        - If inputs contain NaNs or infinities the SVD or other linear algebra steps may fail or produce invalid results; the function does not sanitize input values.
        - The function uses printing for warnings rather than raising exceptions to allow training loops to continue; callers should monitor stdout/stderr for these messages in large-scale training or evaluation.
    """
    from boltz.model.loss.diffusionv2 import weighted_rigid_align
    return weighted_rigid_align(true_coords, pred_coords, weights, mask)


################################################################################
# Source: boltz.model.loss.validation.compute_pae_mae
# File: boltz/model/loss/validation.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def boltz_model_loss_validation_compute_pae_mae(
    pred_atom_coords: torch.Tensor,
    feats: torch.Tensor,
    true_atom_coords: torch.Tensor,
    pred_pae: torch.Tensor,
    true_coords_resolved_mask: torch.Tensor,
    multiplicity: int = 1
):
    """Compute the pae mean absolute error for different interaction modalities in biomolecular complexes.
    
    This function is used in the Boltz model evaluation and training/validation pipelines to quantify how well the model's predicted pairwise aligned error (pred_pae) matches the geometric error derived from predicted and ground-truth atom coordinates. It converts atom coordinates into local frames (per the model's frame indices in feats), computes a continuous inter-token distance error in those frames, discretizes that continuous error into the same binning used by the model's pae prediction head, and then computes mean absolute error (MAE) per interaction modality (e.g., dna_protein, ligand_protein, intra_protein, etc.). The modality-specific MAEs are useful for benchmarking structural prediction quality on cross-molecular interfaces and intra-chain structure, which in turn is relevant for downstream tasks in biomolecular interaction prediction and binding affinity modeling described in the repository README (e.g., assessing complex structure quality used for affinity prediction).
    
    Args:
        pred_atom_coords (torch.Tensor): Predicted atom coordinates produced by the model. In the source code this tensor is expected to have shape (B, N, 3) where B is the total batch size (which may be multiplicity * effective_batch) and N is the number of tokenized atom positions; coordinates are in Cartesian (x, y, z). The function reshapes this tensor using multiplicity and expresses coordinates in predicted local frames to produce the target continuous pae. The tensor must be on a device compatible with tensors in feats and pred_pae; mismatched devices or incompatible shapes may raise indexing or broadcasting errors.
        feats (torch.Tensor): Dictionary-like mapping (as used by the model) of input features required to compute frames and masks. The function accesses feats["frames_idx"], feats["frame_resolved_mask"], feats["token_pad_mask"], feats["mol_type"], and feats["asym_id"] to compute per-token frames, resolved masks, padding masks, chain types, and chain identifiers. These entries control which token pairs are considered when computing modality-specific MAEs and totals. If required keys are missing or their shapes do not align with pred/true coordinates, the function will fail with indexing or shape-mismatch errors.
        true_atom_coords (torch.Tensor): Ground-truth atom coordinates corresponding to pred_atom_coords, expected with the same leading batch and token dimensions (source code treats it with shape (B, N, 3) before reshaping by multiplicity). These coordinates are expressed into true frames (possibly adjusted for nonpolymers after symmetry correction) and used to compute the continuous target PAE. A mismatch in batch size, token count, or device relative to pred_atom_coords or feats will cause runtime errors.
        pred_pae (torch.Tensor): Model-predicted PAE values for token pairs, in the same discretization/format the model uses. The function compares pred_pae against the discretized target PAE computed from coordinate differences. pred_pae must be broadcastable with the target PAE tensor shape produced in the function; otherwise a broadcast or indexing error will occur. This tensor represents the model's internal estimate of pairwise alignment error and is central to measuring structural calibration per modality.
        true_coords_resolved_mask (torch.Tensor): Binary/resolution mask for true coordinates that indicates which atom positions are resolved in the ground truth. The function uses this mask (indexed by frame atom indices) to exclude unresolved atoms from pair computations. Shape and batch alignment must match true_atom_coords and feats; improper masks can lead to incorrect MAE counts or runtime failures.
        multiplicity (int): Diffusion batch multiplicity (diffusion batch size). Default is 1. The function reshapes pred_atom_coords and true_atom_coords by splitting the leading batch into (batch // multiplicity, multiplicity, ...). multiplicity is used whenever the data pipeline replicates or groups samples for score-matching / diffusion training; an incorrect multiplicity (e.g., one that does not divide the leading batch dimension) will produce shape errors. Multiplicity affects how frames and masks are repeated/interpreted and therefore affects which pairs are measured when computing per-modality MAE and totals.
    
    Behavior and side effects:
        The function performs no in-place modification of user-provided tensors (it reshapes and indexes views internally), and returns modality summaries without modifying model state. Internally it calls compute_frame_pred(...) and express_coordinate_in_frame(...) to compute token-local frames and transform coordinates; those helper functions must be available and behave as in the repository for correct results. The function discretizes continuous inter-token distances into bins that mimic the model's PAE discretization (scaling, flooring, clipping to max bin, then offsetting to a center value). Pairwise masks are computed from frame validity, collinearity masks, resolved masks, and token padding masks stored in feats. It then separates pairwise comparisons into modality groups based on feats["mol_type"] and feats["asym_id"] (chain types and chain identifiers) and computes MAE and total valid pair counts per modality.
    
    Failure modes:
        - Shape or device mismatches among pred_atom_coords, true_atom_coords, pred_pae, feats entries, or true_coords_resolved_mask will raise indexing, broadcasting, or runtime errors.
        - multiplicity values that do not evenly divide the leading batch dimension of coordinate tensors will cause reshape errors.
        - Missing keys in feats (e.g., "frames_idx", "frame_resolved_mask", "token_pad_mask", "mol_type", "asym_id") will raise KeyError-like failures.
        - If all pairs for a modality are masked out, the function guards against division by zero by adding a small epsilon when normalizing MAE, but the returned total for that modality will be zero and the MAE will be computed as a finite value driven by the epsilon normalization (not NaN).
    
    Returns:
        tuple: A pair of dictionaries (mae_pae_dict, total_pae_dict).
            mae_pae_dict (dict): Mapping modality name (str) -> torch.Tensor (scalar). Each value is the mean absolute error between the discretized target PAE (derived from true and predicted coordinates in local frames) and pred_pae for that modality. Modalities correspond to interaction or intra-chain categories used in Boltz evaluation and training and are useful for assessing structural interface quality relevant to downstream affinity prediction. The keys in this mapping are exactly: "dna_protein", "rna_protein", "ligand_protein", "dna_ligand", "rna_ligand", "intra_ligand", "intra_dna", "intra_rna", "intra_protein", "protein_protein".
            total_pae_dict (dict): Mapping modality name (str) -> torch.Tensor (scalar). Each value is the total number of valid token pairs counted for the corresponding modality (i.e., the denominator used to compute the MAE, before the small epsilon to avoid division by zero). The keys are the same set listed for mae_pae_dict and indicate how many pairwise comparisons contributed to each modality's MAE; zeros indicate no valid pairs were found for that modality.
    
    No external side effects occur (no file I/O or model parameter mutations). The returned MAE and total counts are intended for logging, loss aggregation, and evaluation metrics in Boltz training and inference pipelines described in the repository README.
    """
    from boltz.model.loss.validation import compute_pae_mae
    return compute_pae_mae(
        pred_atom_coords,
        feats,
        true_atom_coords,
        pred_pae,
        true_coords_resolved_mask,
        multiplicity
    )


################################################################################
# Source: boltz.model.loss.validation.compute_pde_mae
# File: boltz/model/loss/validation.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def boltz_model_loss_validation_compute_pde_mae(
    pred_atom_coords: torch.Tensor,
    feats: torch.Tensor,
    true_atom_coords: torch.Tensor,
    pred_pde: torch.Tensor,
    true_coords_resolved_mask: torch.Tensor,
    multiplicity: int = 1
):
    """Compute the pde (pairwise distance error) mean absolute error (MAE) across specific biomolecular modality pairs.
    
    Args:
        pred_atom_coords (torch.Tensor): Predicted atom coordinates produced by the model. This tensor is used to compute predicted per-token coordinates via a weighted sum with feats["token_to_rep_atom"], and then to compute pairwise predicted distances. In the Boltz domain (biomolecular interaction prediction and binding-affinity modeling described in the README), these coordinates represent atomic positions for chains (proteins, DNA, RNA, ligands) and are required to evaluate geometric prediction quality.
        feats (torch.Tensor or dict-like): Input features or a mapping-like object containing per-token auxiliary tensors required by this metric. The function expects at minimum the keys "token_to_rep_atom", "mol_type", and "asym_id" in feats. "token_to_rep_atom" is used to aggregate atom-level coordinates to token-level coordinates; "mol_type" encodes chain types (compared against const.chain_type_ids for PROTEIN, DNA, RNA, NONPOLYMER); and "asym_id" is used to identify chain membership for intra-chain vs inter-chain masks. Missing keys, incorrect types, or incompatible shapes in feats will raise runtime errors (KeyError or shape-mismatch errors in torch.bmm).
        true_atom_coords (torch.Tensor): Ground-truth atom coordinates corresponding to pred_atom_coords. These coordinates are aggregated to token-level using feats["token_to_rep_atom"] and then used to compute true pairwise distances for the target PDE. They must be aligned with pred_atom_coords and token_to_rep_atom in ordering and batch semantics.
        pred_pde (torch.Tensor): Predicted pairwise distance error (PDE) values output by the model for token pairs. This tensor is compared against the computed target PDE (derived from true and predicted distances) to compute absolute errors. pred_pde is expected to be on a compatible device and shape for elementwise operations with the computed target PDE and pair masks.
        true_coords_resolved_mask (torch.Tensor): Boolean or numeric mask indicating which atoms (or tokens after aggregation) have resolved/valid coordinates in the ground truth. This mask is used (via feats["token_to_rep_atom"]) to build a token-level presence mask and a pair_mask that excludes self-pairs and unresolved tokens. Inconsistent masking shapes relative to token_to_rep_atom will raise runtime errors.
        multiplicity (int): Diffusion batch replication factor (diffusion batch size). Defaults to 1. The function repeats per-sequence per-token features in feats (specifically "token_to_rep_atom", "mol_type", and "asym_id") multiplicity times along the batch dimension using torch.repeat_interleave to align with pred_atom_coords / true_atom_coords when multiple diffusion samples are stacked. If multiplicity does not divide or align with the supplied coordinate tensors, downstream shape errors will occur.
    
    Behavior and computation details:
        This function aggregates atom-level coordinates to token-level coordinates using feats["token_to_rep_atom"] and computes pairwise Euclidean distance matrices for both predicted and true token coordinates (torch.cdist). A target PDE is computed as a discretized scalar per token-pair using the formula floor(abs(true_d - pred_d) * 64 / 32), clamped to a maximum bin index of 63, converted to float, scaled by 0.5 and offset by 0.25 (i.e., target_pde = clamp(floor(abs(delta) * 2), max=63)*0.5 + 0.25). The function builds a pair_mask that excludes self-pairs and any tokens that are unresolved according to true_coords_resolved_mask. Chain-type masks are derived by comparing feats["mol_type"] to const.chain_type_ids for PROTEIN, DNA, RNA, and NONPOLYMER (ligand). A same_chain_mask is built from feats["asym_id"] to separate intra-chain from inter-chain protein pairs.
    
        For each modality pair of interest, the function computes:
        - Sum of absolute errors: torch.sum(abs(target_pde - pred_pde) * modality_mask)
        - MAE for the modality: sum_abs_error / (torch.sum(modality_mask) + 1e-5)
        - Total count for the modality: torch.sum(modality_mask)
    
        The modalities computed (and used as keys in the returned dictionaries) are: "dna_protein", "rna_protein", "ligand_protein", "dna_ligand", "rna_ligand", "intra_ligand", "intra_dna", "intra_rna", "intra_protein", and "protein_protein". The function uses a small epsilon (1e-5) to avoid division-by-zero when no pairs exist for a modality.
    
    Side effects and requirements:
        - feats is read but not modified in-place by this function; however, the function calls torch.repeat_interleave on tensors obtained from feats and uses torch.bmm, torch.cdist and elementwise ops. Inputs must be on the same device and have compatible dtypes and shapes for these operations or a runtime error will occur.
        - The function depends on const.chain_type_ids being defined and containing integer identifiers for PROTEIN, DNA, RNA, and NONPOLYMER. If mol_type values do not match these identifiers, modality masks may be empty and totals zero.
        - The multiplicity parameter causes per-feature repetition via repeat_interleave; supplying an incorrect multiplicity relative to pred/true coordinate batching will cause shape mismatches in batched matrix multiplications.
        - Numerical stabilization: the 1e-5 term prevents NaNs when dividing by zero counts, but callers should still inspect returned totals to detect modalities with no contributing pairs.
    
    Failure modes:
        - KeyError if feats lacks required keys ("token_to_rep_atom", "mol_type", "asym_id").
        - RuntimeError from torch.bmm/torch.cdist if tensor shapes or devices are incompatible.
        - Logical errors (zero totals for modalities) if mol_type values are unexpected or true_coords_resolved_mask marks most tokens unresolved.
        - Device mismatch errors if inputs are on different devices (CPU vs GPU).
    
    Returns:
        tuple: A pair of dictionaries (mae_pde_dict, total_pde_dict).
        mae_pde_dict (dict): Mapping from modality name (str) to torch.Tensor scalar containing the computed mean absolute error for that modality. These MAE tensors represent average absolute differences between the discretized target PDE and pred_pde over all qualifying token pairs for the modality.
        total_pde_dict (dict): Mapping from modality name (str) to torch.Tensor scalar containing the total number of token pairs (sum of mask entries) that contributed to the corresponding MAE. Totals are returned as tensors and should be inspected to determine whether the MAE is supported by non-zero sample counts.
    """
    from boltz.model.loss.validation import compute_pde_mae
    return compute_pde_mae(
        pred_atom_coords,
        feats,
        true_atom_coords,
        pred_pde,
        true_coords_resolved_mask,
        multiplicity
    )


################################################################################
# Source: boltz.model.loss.validation.compute_plddt_mae
# File: boltz/model/loss/validation.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def boltz_model_loss_validation_compute_plddt_mae(
    pred_atom_coords: torch.Tensor,
    feats: torch.Tensor,
    true_atom_coords: torch.Tensor,
    pred_lddt: torch.Tensor,
    true_coords_resolved_mask: torch.Tensor,
    multiplicity: int = 1
):
    """Compute the pLDDT mean absolute error (MAE) per biomolecular modality (protein, ligand/nonpolymer, DNA, RNA).
    
    Args:
        pred_atom_coords (torch.Tensor): Predicted atom coordinates produced by the model for a batch of examples. This tensor is used to compute predicted token (residue/fragment) coordinates and pairwise distances that are compared to ground-truth distances to derive target lDDT values. In the Boltz inference/training pipeline (see README), these coordinates come from the model’s atomic coordinate head and must be aligned with the same token/atom indexing described by feats.
        feats (torch.Tensor): A mapping-like tensor container (actual type in code is a dict-like object accessed by keys) holding input feature tensors required to translate between atom and token representations and to identify molecule types. Required keys (accessed by this function) are "r_set_to_rep_atom", "mol_type", "atom_to_token", and "token_to_rep_atom". These features define which atoms represent tokens, how tokens map to representative atoms, and chain types (protein, DNA, RNA, nonpolymer). The function repeats some of these feature tensors along the batch dimension according to multiplicity to match diffusion batch expansion.
        true_atom_coords (torch.Tensor): Ground-truth atom coordinates for the same batch of examples as pred_atom_coords. These coordinates are used to compute true token coordinates and true pairwise distances for target lDDT computation. They must correspond elementwise to the atoms referenced by feats.
        pred_lddt (torch.Tensor): Predicted per-token/per-atom lDDT values emitted by the model that represent the model’s confidence in local structural accuracy. This tensor is compared against the computed target lDDT (derived from distances) to compute absolute errors per element and per modality. Values are expected to be in a numeric range appropriate for lDDT-style scores produced by the model.
        true_coords_resolved_mask (torch.Tensor): Boolean or numeric mask indicating which atoms in true_atom_coords are resolved/available in the ground truth. This mask is used to exclude missing atoms from distance comparisons and to form pairwise masks. The mask determines which residues/tokens contribute to each modality’s MAE; unresolved or missing atoms will be ignored.
        multiplicity (int): Diffusion batch size multiplier (default 1). When multiplicity > 1, some feature tensors in feats are repeated along the batch dimension using torch.repeat_interleave to match an expanded batch of predicted atom coordinates (e.g., when diffusion generates multiple perturbed copies per example). This parameter controls that repetition and must match the multiplicity used to expand pred_atom_coords and pred_lddt.
    
    Behavior and practical details:
        This function transforms atom-level coordinates into token-level coordinates using feats["token_to_rep_atom"] and uses feats["r_set_to_rep_atom"] to form representative sets for pairwise comparisons. It computes pairwise distances for predicted and true token coordinates, constructs pair masks that remove self-pairs and missing atoms, and applies a modality-dependent cutoff (15 Å for proteins and up to 30 Å for nucleotide R-set elements) before calling lddt_dist to compute a target lDDT per element (per-atom/per-token). The function then partitions elements into four biological modality masks according to feats["mol_type"] (protein, nonpolymer/ligand, DNA, RNA) and computes the mean absolute error between target lDDT and pred_lddt for each modality. To avoid divide-by-zero when a modality has no valid elements, a small epsilon (1e-5) is added in denominators. Features from feats are repeated along the batch dimension with repeat_interleave(multiplicity, 0) to align with multiplicity-expanded predictions.
    
    Side effects and assumptions:
        The function reads tensors from feats using specific keys; if those keys are missing, a KeyError will be raised. All input tensors must be on compatible devices and have compatible batch and atom/token indexing; otherwise, runtime size or device mismatch errors (e.g., in torch.bmm or torch.cdist) will occur. The function uses an epsilon to avoid division-by-zero but will return zero totals for modalities with no contributing elements. The function does not mutate its inputs other than using repeat/reinterpretation operations; it returns new aggregates.
    
    Failure modes:
        Missing keys in feats (KeyError), shape mismatches between pred_atom_coords and token/atom mapping tensors (runtime shape errors), device mismatches between tensors (runtime device errors), or invalid numeric types for masks (non-binary masks that cannot be interpreted as floats) will cause exceptions. If a modality has no valid atoms after masking, the corresponding MAE will be computed with denominator epsilon resulting in a finite number but its associated total count will be zero.
    
    Returns:
        mae_plddt_dict (dict): Dictionary mapping modality names ("protein", "ligand", "dna", "rna") to torch.Tensor values containing the mean absolute error between the computed target lDDT and pred_lddt for that modality. These per-modality MAE values quantify the model’s per-residue/per-token confidence calibration error and are useful for modality-specific evaluation in biomolecular structure and binding prediction tasks (as described in the README for Boltz models).
        total_dict (dict): Dictionary mapping the same modality names ("protein", "ligand", "dna", "rna") to torch.Tensor values indicating the total number of valid elements (atoms/tokens) included in the MAE computation for each modality. These totals can be used to weight or aggregate MAEs across a dataset or to detect modalities with no valid contributions.
    """
    from boltz.model.loss.validation import compute_plddt_mae
    return compute_plddt_mae(
        pred_atom_coords,
        feats,
        true_atom_coords,
        pred_lddt,
        true_coords_resolved_mask,
        multiplicity
    )


################################################################################
# Source: boltz.model.loss.validation.factored_lddt_loss
# File: boltz/model/loss/validation.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def boltz_model_loss_validation_factored_lddt_loss(
    true_atom_coords: torch.Tensor,
    pred_atom_coords: torch.Tensor,
    feats: dict,
    atom_mask: torch.Tensor,
    multiplicity: int = 1,
    cardinality_weighted: bool = False
):
    """Compute the local distance difference test (lddt) aggregated by molecular modality and pair-type masks.
    
    This function is used by Boltz training and validation pipelines to evaluate structural accuracy of predicted 3D atom coordinates in a modality-aware manner. It factorizes lddt into biologically relevant interaction classes (for example protein–ligand, DNA–protein, intra-protein, etc.) so that downstream loss weighting, monitoring, or metric reporting can distinguish performance on different biomolecular modalities. Internally it derives atom types and chain identifiers from the provided feature dictionary, builds pairwise masks (excluding self-pairs), computes pairwise distances with torch.cdist, applies a modality-dependent distance cutoff, and calls lddt_dist to compute per-modality lddt scores and per-modality pair counts or indicators.
    
    Args:
        true_atom_coords (torch.Tensor): Ground truth atom coordinates after any symmetry correction. These are the reference 3D coordinates used to compute the lddt metric against the model prediction. They must be compatible with pred_atom_coords for torch.cdist and with atom_mask for pair masking, and located on the same device and dtype as pred_atom_coords to avoid runtime errors.
        pred_atom_coords (torch.Tensor): Predicted atom coordinates produced by the model. These are compared to true_atom_coords via pairwise distance matrices (torch.cdist) to compute lddt per pair. Mismatched shapes, incompatible batch sizes, or device/dtype inconsistencies between pred_atom_coords and true_atom_coords will raise runtime errors.
        feats (dict): Input features dictionary produced by the data pipeline. This function expects specific keys present: "atom_to_token" (maps atoms to residue/token indices), "mol_type" (per-token molecule type), and "asym_id" (per-token asymmetric unit / chain id). These tensors are used to derive atom_type and atom_chain_id via batched matrix multiplications; missing keys or tensors with incompatible shapes will cause failures. The practical significance in the Boltz codebase is that feats encodes the sequence/chain and molecule-type metadata required to separate modalities such as protein, DNA, RNA, and ligands.
        atom_mask (torch.Tensor): Binary mask (or float mask) indicating which atom positions should be considered valid for pair computations. This mask is expanded to a pair_mask that excludes self-pairs (diagonal elements) and is applied to all modality-specific pair selections. Supplying an atom_mask with incorrect shape or dtype or on a different device than the coordinate tensors will lead to runtime errors. Zeroed masks will yield zero totals for affected modalities.
        multiplicity (int): Diffusion batch size used to repeat per-token features to match repeated coordinate batches. The function uses atom-level features derived from feats and then calls .repeat_interleave(multiplicity, 0) to align feature-derived atom_type and atom_chain_id with repeated coordinate batches. Default is 1. If multiplicity does not match how pred_atom_coords/true_atom_coords were tiled upstream, the derived masks will be misaligned and results will be incorrect or will raise runtime errors.
        cardinality_weighted (bool): If False (default), total counts returned for each modality are converted to binary indicators (1.0 if any valid pair exists, 0.0 otherwise) so that downstream aggregation treats each modality equally regardless of number of pairs. If True, totals are left as the raw number of valid pairs per modality (floating tensor) so losses or metrics can be weighted by pair counts. This flag controls whether per-modality normalization is by cardinality or by presence.
    
    Behavior and implementation notes:
        - Atom types and chain identifiers are computed from feats["atom_to_token"], feats["mol_type"], and feats["asym_id"] via batched matrix multiplication and cast to integer indices; those indices are mapped to modality masks using the repository's chain type identifiers (const.chain_type_ids) to separate protein, DNA, RNA, and nonpolymer/ligand atoms.
        - A pairwise validity mask (pair_mask) is created by outer-product of atom_mask and then zeroing the diagonal to exclude self-pairs.
        - A modality-dependent cutoff is computed using nucleotide presence so that interactions involving nucleotides receive an increased cutoff (the code computes cutoff = 15 + 15 * (...) as implemented). This cutoff is passed to lddt_dist which computes lddt scores over pairs within the cutoff.
        - The function computes pairwise Euclidean distance matrices for true and predicted coordinates using torch.cdist; for large atom counts this can be memory intensive and may raise out-of-memory errors on limited hardware.
        - The function builds masks for multiple modality pairs and intra-chain vs inter-chain relationships (for example dna_protein, rna_protein, ligand_protein, dna_ligand, rna_ligand, intra_ligand, intra_dna, intra_rna, intra_protein, protein_protein) and calls lddt_dist for each; outputs are collected into two dictionaries keyed by the modality names listed above.
        - Intermediate masks are explicitly deleted in the implementation to free memory, but overall the routine still temporarily allocates large pairwise tensors (pred_d, true_d, and modality masks).
    
    Failure modes and error conditions:
        - Missing required keys in feats ("atom_to_token", "mol_type", "asym_id") or tensors in feats with incompatible shapes will raise exceptions when used in batched matrix multiplications or repeat_interleave.
        - Incompatible shapes, mismatched batch sizes, or device/dtype differences between true_atom_coords, pred_atom_coords, atom_mask, and tensors inside feats will result in runtime errors (for example from torch.bmm, torch.cdist, or broadcasting operations).
        - Very large numbers of atoms can lead to out-of-memory errors due to pairwise distance matrix allocation.
        - If a modality has no valid pairs under the masks and cutoff, lddt_dist may return zeros for the lddt and zero totals; when cardinality_weighted is False these totals are converted to binary indicators, so modalities with zero pairs produce 0.0 in total_dict.
    
    Returns:
        tuple: A pair of dictionaries (lddt_dict, total_dict) capturing per-modality lddt scores and per-modality pair counts or indicators for downstream use in loss computation, monitoring, or metric reporting.
            lddt_dict (Dict[str, torch.Tensor]): Mapping from modality name to the computed lddt score tensor for that modality. Keys produced by the implementation are: "dna_protein", "rna_protein", "ligand_protein", "dna_ligand", "rna_ligand", "intra_ligand", "intra_dna", "intra_rna", "intra_protein", and "protein_protein". These scores quantify local structural agreement between pred_atom_coords and true_atom_coords for each interaction class and are used in Boltz validation/loss pipelines to evaluate structure prediction quality relevant to binding and complex modeling.
            total_dict (Dict[str, torch.Tensor]): Mapping from the same modality names to either the raw number of valid atom pairs considered for that modality (if cardinality_weighted is True) or a binary indicator (1.0 if any pairs were present, 0.0 if none) when cardinality_weighted is False (default). These totals are intended for normalization or weighting of per-modality lddt contributions in downstream loss/metric aggregation.
    """
    from boltz.model.loss.validation import factored_lddt_loss
    return factored_lddt_loss(
        true_atom_coords,
        pred_atom_coords,
        feats,
        atom_mask,
        multiplicity,
        cardinality_weighted
    )


################################################################################
# Source: boltz.model.loss.validation.factored_token_lddt_dist_loss
# File: boltz/model/loss/validation.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def boltz_model_loss_validation_factored_token_lddt_dist_loss(
    true_d: torch.Tensor,
    pred_d: torch.Tensor,
    feats: dict,
    cardinality_weighted: bool = False
):
    """Compute per-modality LDDT (local distance difference test) scores by comparing a predicted atom distogram to the ground-truth distogram, factoring the score into biologically relevant token-pair modalities (e.g., DNA–protein, RNA–protein, ligand–protein, intra-chain protein). This function is used in Boltz model validation to report how well the model predicts inter-atomic distance distributions for different molecule-type interactions and to produce modality-level pair counts (or presence flags) used for weighting or aggregation in downstream loss and evaluation pipelines.
    
    Args:
        true_d (torch.Tensor): Ground-truth atom distogram. This tensor encodes the true distance distribution or distance-related targets between atom tokens and is used as the reference when computing LDDT per pair. In Boltz validation, true_d represents the experimental or labeled distance information against which model predictions are compared.
        pred_d (torch.Tensor): Predicted atom distogram. This tensor is the model output for pairwise distance distributions between tokenized atoms; it is compared to true_d via the lddt_dist procedure to produce per-pair LDDT contributions for each modality. It must be compatible in shape and device with true_d and with the masks derived from feats.
        feats (dict): A dictionary of input feature tensors required to partition token pairs into biologically meaningful modalities. Required keys (as used by this function) are "mol_type" (token type ids used to identify PROTEIN, DNA, RNA, NONPOLYMER/ligand), "token_disto_mask" (binary mask for tokens to include in distogram comparisons), and "asym_id" (chain identifiers used to distinguish intra- vs inter-chain pairs). Each value is expected to be a torch.Tensor and have shapes consistent with true_d/pred_d (mismatched shapes will raise errors). The function reads these features to build modality masks (e.g., dna_protein, ligand_protein, intra_protein) and does not mutate feats.
        cardinality_weighted (bool): If False (default), the returned per-modality totals are converted to a binary presence indicator: any modality with at least one valid pair will have its total set to 1.0 (float) so that downstream aggregation treats each modality equally regardless of pair count. If True, totals are left as the raw floating-point counts of valid token pairs for each modality (useful to weight modality losses by actual pair cardinality). This flag controls only how total counts are represented in the second return value and does not change how LDDT values are computed.
    
    Returns:
        dict: A mapping from modality name (str) to a torch.Tensor containing the LDDT score for that modality. Modalities returned (keys) are: "dna_protein", "rna_protein", "ligand_protein", "dna_ligand", "rna_ligand", "intra_ligand", "intra_dna", "intra_rna", "intra_protein", and "protein_protein". Each tensor holds the aggregated LDDT value(s) computed by lddt_dist for the corresponding modality and is intended for use in validation reporting or as components of a composite loss in biomolecular interaction modeling.
        dict: A mapping from the same modality names (str) to torch.Tensor totals describing the number (or presence) of valid token pairs used to compute each modality LDDT. If cardinality_weighted is False (default), each entry is a float tensor equal to 1.0 when at least one pair contributed and 0.0 otherwise; if True, the entries are floating-point counts of contributing pairs (raw totals returned by lddt_dist). These totals are intended for weighting or normalizing modality LDDT contributions when aggregating overall metrics.
    
    Behavior and side effects:
        The function constructs binary masks for modality partitioning using feats["mol_type"], feats["token_disto_mask"], and feats["asym_id"], applies a distance cutoff that treats nucleotide–nucleotide interactions differently, and calls lddt_dist(pred_d, true_d, mask, cutoff) for each modality. It returns two dicts (LDDT values and totals) and does not modify its inputs. The default cardinality_weighted=False makes totals binary presence indicators to avoid overweighting abundant modalities during aggregation.
    
    Failure modes and errors:
        A KeyError will be raised if feats does not contain the required keys ("mol_type", "token_disto_mask", "asym_id"). ValueError or runtime errors may occur if true_d, pred_d, and the tensors in feats have incompatible shapes, devices, or dtypes (e.g., mismatched batch or token dimensions), or if token masks are ill-formed (e.g., non-binary or wrong rank). If a modality has zero valid pairs, its LDDT tensor is produced (typically zero or per lddt_dist semantics) and its total will be zero (or 0.0) unless cardinality_weighted=False, in which case the total remains 0.0 indicating absence.
    """
    from boltz.model.loss.validation import factored_token_lddt_dist_loss
    return factored_token_lddt_dist_loss(true_d, pred_d, feats, cardinality_weighted)


################################################################################
# Source: boltz.model.loss.validation.weighted_minimum_rmsd
# File: boltz/model/loss/validation.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def boltz_model_loss_validation_weighted_minimum_rmsd(
    pred_atom_coords: torch.Tensor,
    feats: torch.Tensor,
    multiplicity: int = 1,
    nucleotide_weight: float = 5.0,
    ligand_weight: float = 10.0
):
    """Compute weighted root-mean-square deviation (RMSD) between predicted atom coordinates and ground-truth coordinates after a rigid alignment, with optional per-atom weighting that emphasizes nucleotides and nonpolymeric ligands. This function is used in Boltz model validation and loss evaluation to quantify structural accuracy of biomolecular predictions (for example protein–ligand or protein–nucleic acid complexes) and to select the best RMSD across multiple diffusion samples (multiplicity). The rigid alignment of ground-truth coordinates to predictions is performed with weighted_rigid_align inside a no-grad block so that the alignment step does not create gradient paths; the RMSD values are computed from the aligned ground truth and the predictions and can be used for evaluation and monitoring during inference or training validation.
    
    Args:
        pred_atom_coords (torch.Tensor): Predicted atom coordinates for each diffusion sample. Inferred shape is (batch * multiplicity, num_atoms, 3), where the first dimension must match the repeated feats entries after applying multiplicity. The tensor must be a floating-point torch.Tensor representing 3D coordinates in the same distance units as feats["coords"] (e.g., Angstroms if that is the unit used by the input features). This is the denoised / predicted coordinate set whose RMSD against the aligned ground-truth coordinates is computed and reported.
        feats (torch.Tensor): Dictionary-like container (typically a mapping of tensors) of input features required by Boltz. The function expects the following keys to be present with compatible shapes: "coords" (ground-truth atom coordinates, inferred shape (batch, num_atoms, 3)), "atom_resolved_mask" (mask of resolved atoms, inferred shape (batch, num_atoms)), "atom_to_token" (mapping from atoms to token/channel dimensions, used to infer chain/atom types, e.g., shape (batch, num_atoms, token_count)), and "mol_type" (per-token molecule/chain type identifiers, inferred shape (batch, token_count)). These feature tensors are used to build alignment masks and per-atom alignment weights, and to identify nucleotides and nonpolymeric ligands for weight scaling. The function will fail if required keys are missing or if the shapes are incompatible with pred_atom_coords and multiplicity.
        multiplicity (int): Diffusion batch size (default 1). multiplicity indicates how many diffusion/denoising samples were generated per example in feats; the function repeats feats along the batch dimension with repeat_interleave(multiplicity, 0) to match pred_atom_coords and then computes per-sample RMSDs. The final returned best_rmsd is obtained by reshaping the per-sample RMSDs to (batch, multiplicity) and taking the minimum along the multiplicity axis, yielding one best RMSD per original input example. multiplicity must be a positive integer; mismatched multiplicity relative to pred_atom_coords batching will lead to shape errors.
        nucleotide_weight (float): Scalar weight multiplier applied to atoms identified as nucleotides (DNA or RNA) when constructing alignment weights (default 5.0). This increases the contribution of nucleotide atoms to the weighted alignment and subsequent RMSD calculation, which is useful when nucleotide parts of complexes (e.g., RNA/DNA) should be emphasized in structural accuracy metrics during validation.
        ligand_weight (float): Scalar weight multiplier applied to atoms identified as nonpolymeric ligands (default 10.0). This increases the contribution of ligand atoms to the weighted alignment and RMSD calculation, which is useful when ligand placement/pose is especially important (for example in binding affinity prediction and ligand optimization stages).
    
    Behavior and side effects:
        The function constructs per-atom alignment weights from feats["coords"] and per-atom type information derived from feats["atom_to_token"] and feats["mol_type"], scaling weights by nucleotide_weight and ligand_weight for DNA/RNA and NONPOLYMER chain types respectively (using the chain type identifiers in const.chain_type_ids). The ground-truth coordinates are repeated to match multiplicity, then rigidly aligned to pred_atom_coords using weighted_rigid_align called under torch.no_grad(), so that the alignment computation does not produce gradient history. After alignment, a weighted mean-squared-error is computed between pred_atom_coords and the aligned ground truth, reduced per sample and converted to RMSD by taking the square root of the weighted average squared distance. The function does not perform any in-place modification of the inputs, but it relies on weighted_rigid_align and torch.no_grad() semantics: the alignment result is detached from the autograd graph, therefore gradients do not flow through the alignment operation and only flow through pred_atom_coords in the subsequent MSE/RMSD computation.
        Default values: multiplicity defaults to 1 (no repetition), nucleotide_weight defaults to 5.0, ligand_weight defaults to 10.0. These defaults reflect typical emphasis used in Boltz validation for nucleotide and ligand atoms but can be adjusted to change alignment/RMSD weighting behavior.
    
    Failure modes and warnings:
        The function will raise runtime or shape errors if feats is missing required keys ("coords", "atom_resolved_mask", "atom_to_token", "mol_type") or if tensor shapes are incompatible with the expected repeat_interleave and batch reshaping behavior. If for any sample the sum of align_weights * atom_mask is zero (for example, if all atoms are masked out for that sample), the RMSD computation will perform a division by zero and produce NaN or Inf for that sample; callers should ensure that atom_resolved_mask marks at least one atom per sample or filter such cases. Inputs should be floating-point tensors for coordinate arithmetic; integer dtypes for coordinate tensors will lead to dtype errors. multiplicity must be a positive integer; non-positive or non-integer values are invalid.
    
    Returns:
        torch.Tensor: rmsd — A 1-D torch.Tensor of per-sample RMSD values computed after rigid alignment. The length is batch * multiplicity (i.e., one RMSD for each prediction sample). Each element is the weighted RMSD (square root of weighted mean squared error) in the same distance units as the provided coordinates.
        torch.Tensor: best_rmsd — A 1-D torch.Tensor of per-original-example best RMSD values (one per entry in feats before repetition). This is obtained by reshaping the per-sample rmsd into shape (batch, multiplicity) and taking the minimum along the multiplicity axis; it is intended for evaluation scenarios where multiple diffusion samples are generated per input and the best (lowest) structural error is reported.
    """
    from boltz.model.loss.validation import weighted_minimum_rmsd
    return weighted_minimum_rmsd(
        pred_atom_coords,
        feats,
        multiplicity,
        nucleotide_weight,
        ligand_weight
    )


################################################################################
# Source: boltz.model.loss.validation.weighted_minimum_rmsd_single
# File: boltz/model/loss/validation.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def boltz_model_loss_validation_weighted_minimum_rmsd_single(
    pred_atom_coords: torch.Tensor,
    atom_coords: torch.Tensor,
    atom_mask: torch.Tensor,
    atom_to_token: torch.Tensor,
    mol_type: torch.Tensor,
    nucleotide_weight: float = 5.0,
    ligand_weight: float = 10.0
):
    """Compute a weighted minimum RMSD between predicted and ground-truth atom coordinates after performing a rigid alignment of the ground-truth coordinates to the predictions.
    
    This function is used in the Boltz codebase for validation and loss computation of biomolecular structural predictions. It computes per-sample root-mean-square-deviation (RMSD) where atoms are weighted according to their macromolecular role (standard chain, nucleotide, or nonpolymer/ligand). The weighting increases the contribution of nucleotides and ligands using the nucleotide_weight and ligand_weight scalars, which makes the metric more sensitive to errors on those atom classes; this behavior is important in Boltz models that jointly model complex structures and binding affinities, where ligands and nucleic acids often require higher fidelity. The ground-truth coordinates are rigidly aligned to the predictions using weighted_rigid_align (called under torch.no_grad()), then a weighted RMSD is computed using atom_mask to ignore unresolved/missing atoms.
    
    Args:
        pred_atom_coords (torch.Tensor): Predicted atom coordinates produced by the model. These are the coordinates that the ground-truth atom positions are rigidly aligned to for RMSD computation. The tensor must be compatible with atom_coords and atom_mask for broadcasting/elementwise operations.
        atom_coords (torch.Tensor): Ground-truth (reference) atom coordinates. These are rigidly aligned to pred_atom_coords using weighted_rigid_align with the computed alignment weights and atom_mask. The function creates align_weights with shape derived from atom_coords.shape[:2], so atom_coords must allow that usage.
        atom_mask (torch.Tensor): Resolved atom mask that indicates which atoms are present/resolved and should participate in alignment and RMSD computation. It is multiplied with align_weights in the denominator and numerator; if all entries are zero for a sample, division by zero may occur and produce NaN/inf values.
        atom_to_token (torch.Tensor): Mapping from atoms to sequence tokens/residues used to aggregate per-token mol_type into per-atom types. In the implementation this tensor is multiplied with mol_type to produce an atom_type index; it therefore must be shaped and valued so that torch.bmm(atom_to_token.float(), mol_type.unsqueeze(-1).float()).squeeze(-1) yields meaningful integer chain/type IDs for each atom.
        mol_type (torch.Tensor): Per-token chain/type identifiers used to determine whether a token/atom is DNA, RNA, or NONPOLYMER (ligand). The code converts mol_type to per-atom atom_type via atom_to_token and then compares those IDs against the chain type constants (const.chain_type_ids["DNA"], const.chain_type_ids["RNA"], const.chain_type_ids["NONPOLYMER"]) to compute extra weighting for nucleotides and ligands.
        nucleotide_weight (float): Scalar weight added for atoms identified as DNA or RNA. Default is 5.0. Effective per-atom weight is 1 + nucleotide_weight for nucleotide atoms (before considering ligand weighting). Increasing this value makes the RMSD more sensitive to errors on nucleotide atoms; it may change the relative contribution of different molecule classes to the aggregated metric.
        ligand_weight (float): Scalar weight added for atoms identified as NONPOLYMER (ligands). Default is 10.0. Effective per-atom weight is 1 + ligand_weight for ligand atoms (before considering nucleotide weighting). Increasing this value makes the RMSD more sensitive to errors on ligand atoms.
    
    Behavior and side effects:
        The function computes per-atom alignment weights align_weights initialized as ones with shape derived from atom_coords.shape[:2], then multiplies them by (1 + nucleotide_weight * is_nucleotide + ligand_weight * is_ligand) where is_nucleotide and is_ligand are boolean indicators derived from atom_type comparisons against chain type IDs defined elsewhere in the codebase (const.chain_type_ids). It then calls weighted_rigid_align(atom_coords, pred_atom_coords, align_weights, mask=atom_mask) inside a torch.no_grad() block to obtain atom_coords_aligned_ground_truth (the ground-truth coordinates rigidly aligned to the predictions). After alignment, it computes the per-atom squared errors between pred_atom_coords and atom_coords_aligned_ground_truth, sums over the coordinate dimension, applies align_weights and atom_mask, and takes the square root of the weighted mean squared error per sample to produce RMSD.
        No parameters are modified in-place by this function; weighted_rigid_align is called without gradient tracking here, so this function does not contribute gradients if used inside a training step. The function relies on const.chain_type_ids to identify DNA, RNA and NONPOLYMER types; those identifiers must be present and correct in the surrounding code.
        Defaults: nucleotide_weight defaults to 5.0 and ligand_weight defaults to 10.0 as in the original implementation.
        Failure modes: mismatched tensor shapes or incompatible batching/leading dimensions among pred_atom_coords, atom_coords, atom_mask, atom_to_token, and mol_type will raise runtime errors when performing torch.bmm or elementwise operations. If for any sample the denominator torch.sum(align_weights * atom_mask, dim=-1) is zero, the division will yield NaN or infinity for that sample's RMSD. Device or dtype mismatches between inputs may raise errors; inputs should be on the same device and compatible dtypes.
    
    Returns:
        torch.Tensor: rmsd. Per-sample weighted RMSD values computed after rigid alignment. This tensor contains one RMSD value per example in the batch dimension implied by the inputs, computed as sqrt(sum(mse_loss * align_weights * atom_mask) / sum(align_weights * atom_mask)).
        torch.Tensor: atom_coords_aligned_ground_truth. The ground-truth atom coordinates after application of the weighted rigid alignment that aligns atom_coords to pred_atom_coords using align_weights and atom_mask. This tensor is produced by weighted_rigid_align under torch.no_grad() and mirrors the structure of atom_coords.
        torch.Tensor: align_weights. The per-sample, per-atom weights used for alignment and RMSD computation. These weights are derived from atom_coords.shape[:2] and equal 1 for standard atoms, increased by nucleotide_weight for atoms identified as DNA/RNA, and increased by ligand_weight for atoms identified as NONPOLYMER (ligand), as described above.
    """
    from boltz.model.loss.validation import weighted_minimum_rmsd_single
    return weighted_minimum_rmsd_single(
        pred_atom_coords,
        atom_coords,
        atom_mask,
        atom_to_token,
        mol_type,
        nucleotide_weight,
        ligand_weight
    )


################################################################################
# Source: boltz.model.modules.confidence_utils.compute_aggregated_metric
# File: boltz/model/modules/confidence_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def boltz_model_modules_confidence_utils_compute_aggregated_metric(
    logits: torch.Tensor,
    end: float = 1.0
):
    """Compute an aggregated scalar metric (plddt-like confidence) from histogram logits.
    
    This function is used in the Boltz codepath that transforms a discrete histogram prediction (logits over bins) into a single continuous confidence score per element (for example, a per-residue predicted local distance difference test score, pLDDT-like, used in Boltz structural predictions). It converts logits over num_bins into probabilities with a softmax over the last dimension, constructs bin center values evenly spaced from half a bin to the provided maximum value `end`, and returns the probability-weighted sum of those centers. The computation preserves device placement (bounds are created on logits.device) and is differentiable with respect to the input logits, so it can be used in training or inference pipelines that require gradients.
    
    Args:
        logits (torch.Tensor): A tensor of logits where the last dimension enumerates histogram bins. The function applies softmax along the last dimension to obtain probabilities and then computes a weighted sum of bin centers. The tensor may have arbitrary leading dimensions (e.g., batch and sequence/residue axes); the last dimension must be the number of bins. logits should be a floating-point tensor; integer dtypes may raise an error or be implicitly cast by PyTorch operations. Computation occurs on logits.device.
        end (float): Max value of the metric, by default 1.0. This value defines the upper bound of the continuous metric produced by the aggregation: bin centers are placed between 0.5 * bin_width and `end` with bin_width = end / num_bins. Typical usage in Boltz structural outputs is end=1.0 to produce a confidence value normalized on the 0..1 scale, but any positive float may be provided to rescale the metric.
    
    Returns:
        torch.Tensor: A tensor containing the aggregated metric (named `plddt` in the implementation). The returned tensor has the same leading dimensions as `logits` but with the final (bin) dimension removed (i.e., it is the probability-weighted sum over the last dimension of `logits`). Values lie on the same scale defined by `end` (for a proper probability distribution over bins, outputs will be within the interval (0, end)). The tensor is created on the same device as `logits` and the operation is differentiable with respect to `logits`.
    
    Notes and failure modes:
        - If logits.shape[-1] == 0 (no bins) the function will raise an error due to division by zero when computing the bin width.
        - If `end` is non-positive the bin width will be zero or negative, yielding semantically invalid bin centers; callers should provide a positive `end`.
        - The function does not modify `logits` in-place.
        - The function assumes the last dimension of `logits` corresponds to histogram bins; providing inputs with a different layout will produce incorrect results.
    """
    from boltz.model.modules.confidence_utils import compute_aggregated_metric
    return compute_aggregated_metric(logits, end)


################################################################################
# Source: boltz.model.modules.confidence_utils.compute_ptms
# File: boltz/model/modules/confidence_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def boltz_model_modules_confidence_utils_compute_ptms(
    logits: torch.Tensor,
    x_preds: torch.Tensor,
    feats: dict,
    multiplicity: int
):
    """Compute pTM and ipTM confidence scores from pairwise PAE logits and predicted coordinates.
    
    Args:
        logits (torch.Tensor): Per-residue pair error (PAE) logits produced by the model. The last dimension of this tensor enumerates the discrete PAE bins produced by the network (referred to in the code as num_bins). These logits are converted to probabilities with softmax and combined with a precomputed tm_value to produce expected TM-like values for each residue pair; logits must therefore be on a device compatible with torch operations used here.
        x_preds (torch.Tensor): Predicted coordinates / frames used to identify and mask collinear or overlapping tokens via compute_frame_pred. This tensor is passed to compute_frame_pred (called with inference=True) and is required so that the function can compute a per-token frame validity mask used in pTM/ipTM calculations. The exact coordinate layout is determined by the broader Boltz model code (used in Boltz-2 inference and affinity/structure predictions).
        feats (dict): Dictionary of input feature tensors required by this function. The implementation accesses at least the following keys: "frames_idx" (indexing used by compute_frame_pred), "token_pad_mask" (per-token padding mask), "asym_id" (asymmetric chain identifiers used to distinguish chains and compute inter-chain masks), and "mol_type" (token type identifying protein vs ligand chains). These tensors are expected to be torch.Tensor objects and to be aligned with the model outputs; missing or shape-mismatched entries will raise runtime errors (KeyError or shape errors).
        multiplicity (int): The batch multiplicity of the diffusion roll-out, i.e., how many replicated feature/examples each input example was unrolled into during diffusion. multiplicity is used with torch.repeat_interleave to expand per-example feat tensors so masks and asym_id align with the logits and x_preds used here. This integer controls how feats are tiled and therefore affects the batch dimension assumed by the function.
    
    Returns:
        ptm (torch.Tensor): Per-example predicted TM-like score (pTM) computed from the expected TM values over intra-sample residue pairs that pass the computed pair_mask_ptm. This tensor contains one scalar per expanded batch item (multiplicity applied) and represents an overall structural confidence metric for each predicted complex produced by the Boltz model. The computation uses a softmax over logits, multiplies by a precomputed tm_value function, aggregates using pair_mask_ptm, normalizes with a small epsilon to avoid division-by-zero, and returns the maximum aggregated value across residues (torch.max over residue axis).
        iptm (torch.Tensor): Per-example interface predicted TM-like score (ipTM) computed similarly to ptm but restricted to inter-chain residue pairs (pair_mask_iptm). This scalar per example measures confidence in the predicted interfaces between different asymmetric chains and is useful in downstream tasks in Boltz (for example, assessing predicted binding interfaces during affinity prediction or complex modeling).
        ligand_iptm (torch.Tensor): Per-example ipTM restricted to ligand–protein interface residue pairs. The function identifies ligand and protein tokens using feats["mol_type"] and const.chain_type_ids (as used in the source code), constructs a mask selecting ligand↔protein pairs, and computes the ipTM score for that subset. This value is particularly relevant to Boltz affinity and docking contexts where ligand–protein interfacial quality is of interest.
        protein_iptm (torch.Tensor): Per-example ipTM restricted to protein–protein inter-chain interfaces (protein tokens on both sides). Computed analogously to ligand_iptm but limited to protein↔protein residue pairs, this metric is useful for evaluating confidence in multi-protein assemblies predicted by the model.
        chain_pair_iptm (dict): Nested dictionary mapping asymmetric chain identifiers to dictionaries of pairwise ipTM tensors. The outer keys are unique asym_id integers (from feats["asym_id"]), each mapping to an inner dict whose keys are target asym_id integers and whose values are torch.Tensor objects containing the per-example ipTM computed for that specific ordered pair of chains. This structure lets downstream code inspect pairwise inter-chain confidence (e.g., which chain–chain interfaces are predicted with high confidence).
    
    Behavior and side effects:
        - The function calls compute_frame_pred(x_preds, feats["frames_idx"], feats, multiplicity, inference=True) to compute a per-token frame validity mask; compute_frame_pred is an external dependency and must be available in the runtime environment.
        - feats tensors are repeated using torch.repeat_interleave(multiplicity, 0) so the function expects feats to correspond to the base (pre-multiplicity) batch layout. The multiplicity parameter must therefore match how logits and x_preds were produced; mismatched multiplicity will produce incorrect masks or runtime shape errors.
        - A small constant (1e-5) is added to denominators when normalizing aggregated sums to avoid division-by-zero when masks are empty for a given example. If a mask is all zeros (no valid pairs), the normalized sum becomes zero; the max operation then yields zero for that example.
        - No in-place modification of input tensors is performed, but new tensors and Python dictionaries are allocated for masks and outputs.
        - The function relies on tm_function and const.chain_type_ids existing in the module namespace; if these are missing, a NameError will be raised.
    
    Failure modes:
        - KeyError if required keys ("frames_idx", "token_pad_mask", "asym_id", "mol_type") are missing from feats.
        - Shape or broadcasting errors if logits, x_preds, and feats do not have compatible batch dimensions after applying multiplicity with repeat_interleave.
        - Device mismatches (inputs on different devices) will raise torch errors when operations attempt to combine tensors on different devices.
        - If all mask values for a given example are zero, the computed score for that example will be zero (the implementation guards against NaN/divide-by-zero but cannot invent valid data).
    
    Practical significance in Boltz / biomolecular modeling:
        - pTM and ipTM are structural confidence metrics used within Boltz (Boltz-1/Boltz-2) to assess overall fold correctness (pTM) and interface correctness (ipTM). These scores are used in model inference pipelines (for example, when running boltz predict) to rank and filter predicted complexes and to inform downstream affinity estimation and candidate selection in hit-discovery and lead-optimization workflows described in the repository README. chain_pair_iptm and the ligand/protein-specific ipTM values are especially useful when discriminating ligand–protein binding interfaces from decoys or assessing individual chain–chain interaction quality.
    """
    from boltz.model.modules.confidence_utils import compute_ptms
    return compute_ptms(logits, x_preds, feats, multiplicity)


################################################################################
# Source: boltz.model.modules.confidence_utils.tm_function
# File: boltz/model/modules/confidence_utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def boltz_model_modules_confidence_utils_tm_function(d: torch.Tensor, Nres: torch.Tensor):
    """Compute the rescaling function used for pTM confidence scoring in Boltz.
    
    This function implements the rescaling step used by Boltz confidence modules to convert raw distance-like inputs into a bounded rescaling factor for the pTM (predicted TM-score) computation. In the Boltz model family for biomolecular interaction and structure prediction, pTM is used as a confidence metric for predicted protein/complex structures; this function provides the elementwise rescaling 1 / (1 + (d / d0)^2) where d0 is a length-dependent reference distance derived from the number of residues. The implementation follows the formula in the source code:
    d0 = 1.24 * (torch.clip(Nres, min=19) - 15) ** (1 / 3) - 1.8
    
    Args:
        d (torch.Tensor): The input tensor containing distance-like values or scores to be rescaled for pTM computation. In practice within Boltz this is a per-position or per-pair quantity derived from structural predictions; the function applies the rescaling elementwise and returns the same dtype/type as d.
        Nres (torch.Tensor): A tensor containing the number of residues (sequence length) used to compute the reference distance d0. In common usage this is a scalar tensor representing the protein or complex length; the implementation clips Nres at a minimum of 19 before computing d0 to ensure a sensible reference distance for short sequences.
    
    Returns:
        torch.Tensor: A tensor of the same type as the input representing the rescaled values computed as 1 / (1 + (d / d0) ** 2). This output is used downstream by Boltz pTM/confidence modules as a bounded factor (in (0, 1]) contributing to predicted TM-score estimates.
    
    Behavior, side effects, defaults, and failure modes:
        This function is pure (no side effects) and deterministic given the same inputs. It computes d0 from Nres using the expression shown above; because Nres is clipped with min=19, d0 is positive for common integer Nres inputs >= 19, avoiding a division-by-zero in typical use. If Nres contains non-finite values (NaN or inf) or d contains non-finite values, those values will propagate to the output according to PyTorch arithmetic rules. The function relies on PyTorch broadcasting rules when d and Nres have different shapes; callers should ensure shapes are compatible for the intended elementwise operation. Numerical stability follows standard floating-point semantics of PyTorch; extremely large magnitudes in d or pathological Nres values can produce inf/NaN outputs.
    """
    from boltz.model.modules.confidence_utils import tm_function
    return tm_function(d, Nres)


################################################################################
# Source: boltz.model.modules.utils.center_random_augmentation
# File: boltz/model/modules/utils.py
# Category: fix_args
# Reason: Missing type hints for some parameters
################################################################################

def boltz_model_modules_utils_center_random_augmentation(
    atom_coords: torch.Tensor,
    atom_mask: torch.Tensor,
    s_trans: float = 1.0,
    augmentation: bool = True,
    centering: bool = True,
    return_second_coords: bool = False,
    second_coords: torch.Tensor = None
):
    """center_random_augmentation centers and optionally applies randomized rigid-body augmentations
    to batched atomic coordinate tensors used by Boltz models for biomolecular interaction
    prediction. This function implements the preprocessing described as "Algorithm 19" in the
    codebase: it can subtract a per-sample mean coordinate (centering), apply a shared random
    rotation to all atoms (via randomly_rotate), and add a per-sample random translation
    sampled from a normal distribution scaled by s_trans. These transforms are commonly used
    before feeding coordinates into structure/affinity prediction networks in the Boltz family
    to remove absolute position dependence and to provide data augmentation that improves
    generalization.
    
    Args:
        atom_coords (torch.Tensor): A batched tensor of atomic coordinates to transform.
            In typical usage within Boltz this has shape (B, N, 3) where B is batch size and N
            is number of atom slots; coordinates are in the model's native length units.
            Each sample in the batch will be processed independently. This tensor is read and
            a transformed tensor is returned; callers should not rely on in-place modification.
        atom_mask (torch.Tensor): A batched mask tensor indicating which atom slots are
            present for each sample. Typical shape is (B, N). Nonzero entries are treated as
            "present" and are used to compute the per-sample mean when centering is enabled.
            If centering is requested and a sample's mask sums to zero (no present atoms),
            the mean computation will divide by zero and produce invalid values (NaN/Inf).
        s_trans (float): Standard deviation scaling factor for the random translation noise.
            A per-sample translation vector is sampled from a standard normal and multiplied
            by s_trans; the default 1.0 applies unit-scale translations. Scaling controls the
            magnitude of translation augmentation. This value is passed directly to the
            translation sampling step and must be a finite float.
        augmentation (bool): If True (default), apply random rotation and translation
            augmentations after optional centering. Rotation is applied via the helper
            randomly_rotate called with return_second_coords=True so that any provided
            second_coords receive the same rotation. Translation is a single random vector
            sampled per sample (shape broadcastable to all atoms) and added to all atoms in
            that sample. If False, no rotation or translation is applied.
        centering (bool): If True (default), compute the per-sample mean coordinate using
            atom_mask as weights and subtract that mean from atom_coords (and from
            second_coords if provided). Centering removes absolute position information and
            is commonly used before rotation so rotations occur about the sample center.
            If False, no mean subtraction occurs and subsequent rotations/translations are
            applied relative to the origin.
        return_second_coords (bool): If True, the function returns a tuple (atom_coords,
            second_coords) after applying the selected transforms. This is useful when a
            paired coordinate tensor (for example, target or alternative conformation) is
            supplied so that both tensors undergo identical rigid-body transforms. If False
            (default), only the transformed atom_coords tensor is returned.
        second_coords (torch.Tensor): Optional second batched coordinate tensor to transform
            with the same centering, rotation, and translation as atom_coords. When provided,
            second_coords must be shape-compatible with atom_coords (e.g., same (B, N, 3)
            layout) so that the same operations can be applied elementwise; mismatched shapes
            will raise runtime broadcasting/indexing errors. If None (default) and
            return_second_coords is True, the returned second_coords will be None.
    
    Returns:
        torch.Tensor or tuple[torch.Tensor, torch.Tensor]: If return_second_coords is False
        (default), returns the transformed atom_coords tensor. If return_second_coords is
        True, returns a tuple (atom_coords, second_coords) where both tensors have had the
        enabled centering/augmentation transforms applied; second_coords will be None if no
        second_coords input was provided. Side effects: the function performs deterministic
        tensor operations but relies on PyTorch random state for augmentation sampling;
        randomness can be controlled externally via torch.manual_seed. Failure modes include
        division by zero (if a sample's atom_mask sums to zero when centering=True) and
        runtime shape/broadcasting errors if second_coords has incompatible shape.
    """
    from boltz.model.modules.utils import center_random_augmentation
    return center_random_augmentation(
        atom_coords,
        atom_mask,
        s_trans,
        augmentation,
        centering,
        return_second_coords,
        second_coords
    )


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
